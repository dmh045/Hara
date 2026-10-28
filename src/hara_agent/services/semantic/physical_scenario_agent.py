"""A workflow-scheduled proposal task; no scoring or approval authority."""
from dataclasses import asdict
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from hara_agent.contracts.physical_scenario_plan import (
    MODEL_CATALOG, PARAMETER_UNITS, PHYSICAL_PLAN_VERSION, PhysicalScenarioPlan, CONTROL_IDENTITIES,
    INTERVENTION_ASSUMPTION_FIELDS,
)
from hara_agent.infrastructure.llm import LLMRequest
from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint


def physical_prompt_view(context: dict) -> tuple[dict, dict]:
    """Lossless request projection; full provenance remains the plan identity.

    Only exact repeated context and source excerpts are referenced. No source,
    qualifier, scope, or numeric input is summarized or truncated.
    """
    view = deepcopy(context)
    evidence = view.get("evidence", {})
    if "conditions" in view and evidence.get("SCENARIO.conditions") == view["conditions"]:
        evidence["SCENARIO.conditions"] = {"$context_ref": "conditions"}
    project = evidence.get("PROJECT.operating_context", {})
    if "project_context" in view and project.get("value") == view["project_context"]:
        project["value"] = {"$context_ref": "project_context"}

    counts = Counter()

    def count(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "excerpt" and isinstance(item, str) and len(item) > 128:
                    counts[item] += 1
                else:
                    count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)

    count(view)
    texts = {hashlib.sha256(text.encode()).hexdigest(): text for text, n in counts.items() if n > 1}

    def intern(value):
        if isinstance(value, dict):
            return {key: {"$source_text": hashlib.sha256(item.encode()).hexdigest()}
                    if key == "excerpt" and isinstance(item, str) and counts[item] > 1
                    else intern(item) for key, item in value.items()}
        if isinstance(value, list):
            return [intern(item) for item in value]
        return value

    view = intern(view)
    structures = Counter()

    def structured_key(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def count_structures(value):
        if isinstance(value, (dict, list)):
            key = structured_key(value)
            if len(key) > 400:
                structures[key] += 1
            for item in value.values() if isinstance(value, dict) else value:
                count_structures(item)

    count_structures(view)
    shared = {}

    def share(value):
        if not isinstance(value, (dict, list)):
            return value
        key = structured_key(value)
        identity = hashlib.sha256(key.encode()).hexdigest() if structures[key] > 1 else None
        if identity in shared:
            return {"$context_value": identity}
        projected = ({k: share(v) for k, v in value.items()} if isinstance(value, dict)
                     else [share(v) for v in value])
        if identity is not None:
            shared[identity] = projected
            return {"$context_value": identity}
        return projected

    view = share(view)
    if shared:
        view["$context_values"] = shared
    return view, texts


def expand_physical_prompt_view(view: dict, source_texts: dict) -> dict:
    """Reconstruct the original context for audits and offline equivalence tests."""
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$source_text"}:
                return source_texts[value["$source_text"]]
            if set(value) == {"$context_ref"}:
                return expand(view[value["$context_ref"]])
            if set(value) == {"$context_value"}:
                return expand(view["$context_values"][value["$context_value"]])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    return expand({key: value for key, value in view.items() if key != "$context_values"})


def normalize_item_basis_refs(candidate: dict, evidence_ids: set[str]) -> list[dict]:
    """Resolve only unique supplied Item block coordinates with omitted padding."""
    def coordinate(value):
        match = re.fullmatch(r"ITEM\.(?:T-(\d+)-R-(\d+)|P-(\d+))", value)
        return tuple(int(v) if v is not None else None for v in match.groups()) if match else None

    by_coordinate = {}
    for identity in evidence_ids:
        if key := coordinate(identity):
            by_coordinate.setdefault(key, []).append(identity)
    changes = []

    def visit(value):
        if isinstance(value, dict):
            refs = value.get("basis_refs")
            if isinstance(refs, list):
                for index, ref in enumerate(refs):
                    matches = by_coordinate.get(coordinate(ref), []) if isinstance(ref, str) else []
                    if isinstance(ref, str) and ref not in evidence_ids and len(matches) == 1:
                        refs[index] = matches[0]
                        changes.append({"original": ref, "resolved": matches[0]})
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(candidate)
    return changes


class PhysicalScenarioAgent:
    RESPONSE_VALIDATION_VERSION = "physical-response-validation-v3-complete-json"
    PROMPT_VERSION = "post-fault-physical-scenario-v31-driver-responses"
    CONTROL_PROMPT_VERSION = "post-fault-control-inputs-v14-driver-responses"
    CONTROL_ANALYSIS_POLICY = (
        "区分两件事：来源能确认的接口/正常接管动作，与本次故障下尚未验证的通道独立性。"
        "读取ITEM原文后，按本故障作用范围检查控制通道：自动指令异常不自动代表所有执行器或人工通道同时失效；"
        "也不能仅凭正常退出动作就把故障后独立性当成事实。允许在相容的代表工况中明确假设某独立通道仍可用，"
        "用remaining_controls和conditions说明该假设成立所需条件及不覆盖的共同失效；"
        "这些是待审核的分析假设，不要求原文逐字证明故障后可靠性。原文未证明不是拒绝所有条件性假设的理由。"
        "对远程干预和目标规避，同样需要在本代表的几何、可见性、突发性和行为条件下判断；"
        "有合理依据时提出明确true或false及条件，不能把未提及、静止、车外直接当作false。"
        "不要为得到完整评级机械填值，也不要额外假定独立故障。若连条件性工况也无法合理成立，具体说明矛盾或方法缺口。"
        "driver_configurations已列明将要实例化的驾驶员配置；能力是否保留与人员是否在驾驶位分开，不重写这些配置。"
        "当intervention_inputs包含emergency_braking_available且存在车内驾驶员时，必须分析制动能力："
        "用failed_controls或remaining_controls声明结论；若检查来源与故障后仍无法合理提出条件性结论，"
        "在同一假设或control_assessment的unresolved_controls.braking中说明具体原因。"
        "这两种表示互斥：如果本代表工况已条件性保留driver_braking或braking，"
        "不得同时在unresolved_controls.braking声明未知；尚未验证的独立性和未覆盖共同失效写入conditions和uncovered_conditions。"
        "若连条件性人工路径也不能成立，则保留unresolved_controls.braking并不把driver_braking列为remaining。"
        "driver_positions是可选字段，只能用in_driver_seat或outside_driver_seat；没有位置限定时省略，不写空字符串。"
        "空的控制数组不等于已完成分析；不要为通过检查强填可用或不可用。"
    )
    ANALYSIS_STRATEGY = (
        "本任务完成当前失效的有限条件性代表工况：先比较与故障有关的对象运动交互，再选择模型和取点。"
        "对象、方向、净距、初始运动及故障运动必须共同成立。"
        "依据model_input_status与model_catalog.required逐项沿用适用输入或提出假设；"
        "缺少实测距离、加速度或残余阻力时，可以自行选取有场景依据的POINT并解释选值、参考点、"
        "有效阶段和未覆盖范围。假设不等于测量、批准或车辆能力上界，不设通用数值。"
        "仅当连条件性代表也无法合理成立时保留SYMBOLIC，说明具体相容性/模型障碍及参数关系；"
        "仅说原文未提供实测值不是充分理由。无接触代表同样有效，不以得到某个评级为选值目标。"
        "默认只给代表点；只有来源明确要求范围，或能指出接触、停止、方向、使能或规避条件的"
        "具体变化时才增加RANGE。数值随参数连续变化本身不是展开理由。符号几何下不展开无新结果的范围。"
        "额外假设须属于当前失效实际引起的不同危险机制，不附加另一个独立故障；"
        "同一对象、轨迹和危险机制的不同数学近似模型不是两个独立机制，选最适用模型即可。"
        "例如纵向扭矩超量不能凭空附加左右扭矩不对称或转向故障，驻车失效不额外生成无因果依据的行驶加速。"
        "计算与评级全部由Python完成。"
    )

    def __init__(self, client, *, response_checkpoint_dir: Path | None = None):
        self.client = client
        self.responses = ResponseCheckpoint(response_checkpoint_dir) if response_checkpoint_dir else None
        config = getattr(client, "config", None)
        self.response_format = ({"type": "json_object"}
            if getattr(config, "scenario_response_format", "none") == "json_object" else None)

    def propose(self, context: dict, *, on_response=lambda record: None) -> tuple[PhysicalScenarioPlan, dict]:
        example = {
            "hypotheses": [{
                "hypothesis_id": "h1", "model": "one model_catalog key", "rationale": "mechanism and applicability",
                "basis_refs": ["a supplied evidence ID"],
                "geometry": {"road_user_type": "from method vocabulary", "collision_type": "from method vocabulary",
                    "object_position": "FRONT|REAR|LEFT|RIGHT", "ego_longitudinal_direction": "FORWARD|REVERSE",
                    "ego_lateral_direction": "LEFT|RIGHT; required for curved/lateral motion, omit for longitudinal-only models",
                    "object_longitudinal_direction": "FORWARD|REVERSE|LEFT|RIGHT; required for moving target, omit for stationary",
                    "object_extent": "POINT_ON_TRAVEL_PATH|EXTENDED_ALONG_TRAVEL_PATH|FINITE_CONTACT_ENVELOPE", "object_id": "target1"},
                "parameters": [{"field": "one parameter_units key", "unit": "exact declared unit",
                    "kind": "POINT", "value": 0, "rationale": "why this conditional representative is useful"}],
                "conditions": ["scope of this hypothesis"], "uncovered_conditions": ["limits of this analysis"],
                "driver_positions": ["in_driver_seat", "outside_driver_seat"],
                "failed_controls": ["affected control"], "remaining_controls": ["independent control, if supported"],
                "unresolved_controls": {"braking": "optional: specific unresolved cause after assessment; omit if braking capability is declared"},
                "condition_changes": {"initial_state": False, "target": False, "rationale": "same initial world and target; set true only for a justified independent child condition"},
                "grade_frame": {"coordinate_frame": "BODY_FORWARD_POSITIVE",
                    "slope_reference": "EGO_FORWARD|ROAD_REFERENCE|UNSPECIFIED",
                    "ego_forward_relative_to_reference": "SAME|OPPOSITE (required with ROAD_REFERENCE)",
                    "heading_relative_to_downhill": "UPHILL|DOWNHILL", "downhill_body_direction": "FRONT|REAR",
                    "orientation_basis": "SOURCE|ANALYSIS_ASSUMPTION", "rationale": "required for rollaway; omit grade_frame for other models"},
            }], "uncovered_conditions": ["not an exhaustive vehicle capability envelope"],
        }
        system = (
            "输入使用无损引用：$context_ref 指向 context 的同名字段，$source_text 指向 source_texts 中的完整原文；"
            "$context_value指向context.$context_values中的完整结构，可递归引用，来源、对象与阶段未省略。"
            "引用处保留来源身份、位置和适用范围；读取所引用全文，包括否定和限定，不得把引用当成缺失来源。"
            "先遵守 output_contract 的精确字段、枚举和引用ID。geometry_values列的是可输出的键，"
            "每个运动假设默认适用于全部driver_configurations；不要仅因车外不能踩踏板就另造相同运动方案，"
            "独立人工制动通道可用与驾驶员是否在驾驶位由程序分别处理。确有只适用某驾驶员配置的假设，"
            "必须用driver_positions列出适用配置，不能只写在自然语言里后再交叉展开。"
            "不要用自然语言标签或自行创造对象类别。不能把不支持的障碍物伪称车辆；"
            "若分析另一个合法对象，必须声明condition_changes.target及新object_id，说明新条件和未覆盖对象。"
            "SYMBOLIC使用symbol字符串，不能把符号放进value；POINT只有value；RANGE必须有lower/value/upper。"
            "RANGE始终满足lower<=value<=upper，包括负数，例如-8<=-7<=-6.5，不能按绝对值排列。"
            "basis_refs只能选basis_ref_ids中的完整ID，不得把它扩展成未提供的字段路径。"
            "failed_controls/remaining_controls只能选control_identities中的完整键。"
            "遵守analysis_strategy；先形成有依据的可计算代表工况，确实不能合理定点才保留符号关系。"
            "你是故障后物理场景分析员。使用给定项目事实、来源限定、FM、Atom 和已确认方法约束，"
            "提出 1–3 个有工程意义的条件性物理假设；原文未提供故障加速度并不禁止提出分析假设。"
            "若目录中没有能表示当前危险机制的模型，返回hypotheses空数组，并在uncovered_conditions具体说明模型缺口。"
            "不得为了满足模型目录把后车追尾改成前方静止目标碰撞，或把移动目标换成静止目标；"
            "合法新增条件仍必须代表当前失效的危险机制，不能以另一危险机制替代原问题。"
            "有source_condition_focus时，在本次现有假设中落实相关来源条件：新增operating_condition="
            "{source_field:odd_road_types或odd_weather_conditions,value:该来源列表中的原文值,"
            "physical_effect:该条件如何影响本假设的几何/可见性/运动/干预}。"
            "车位朝向和单侧/双侧障碍约束要体现到对象位置、路径或净距；照明/能见度须体现在"
            "感知或干预条件中，不能自动改变路面附着或E。允许范围不等于当前事实。"
            "焦点与当前机制或室内外环境不相容时，选择列表内另一相关条件并解释；没有适用条件时"
            "在uncovered_conditions明确记录。不能只复述标签就声称覆盖，也不为每个标签新增假设。"
            "优先覆盖既有条件允许的不同对象运动交互，同一接触关系仅数值变化不占新增机制槽位；"
            "先比较静止、同向接近、对向接近或横穿哪些与当前失效和ODD相关，不能因为没有实测对象速度就默认全是静止目标。"
            "无具体对象的上下文可提出有依据的移动目标代表；明确已有静止对象时保留原工况，"
            "额外对象必须声明新条件。同一方案中优先补充不同且物理相容的机制，而不是只改变距离。"
            "三个实例预算由不同交互和范围采样共同使用，默认每种交互一个代表点。"
            "不得把假设声称为原文事实、车辆真实能力或可信上界。不得输出 S/E/C/ASIL、批准状态、"
            "碰撞速度、TTC或collision_branches。程序会计算这些量；Atom只能在operating_atoms中选所给候选ID。"
            "优先保留已给定的对象类别和运行条件；新对象需说明适用性，不得扩大 ODD。"
            "对象维度NOT_APPLICABLE/未选择并不表示来源证明周围没有对象；可提出ODD内明确的条件性对象，"
            "声明condition_changes.target=true并重新绑定。只有来源明确禁止时才不能这样做。"
            "对象描述必须与road_user_type一致：VEHICLE只表示车辆，不能同时泛指墙、立柱或任意障碍物。"
            "已有同一目标的collision_type、位置、距离必须保留，不能因为某枚举更常见就覆盖已选条件。"
            "正常速度约束仅约束初始速度；故障可加速、恒速或减速。区分失效控制与剩余控制，"
            "自动制动指令异常用automatic_braking_command，不能直接等同整个braking能力失效；"
            "驾驶员制动路径用driver_braking。结合Item的接口、接管条件和本次失效作用范围，"
            "说明独立驾驶员制动路径是否仍可用；允许明确标识的条件性独立路径假设，"
            "但正常状态下可踩制动退出AVP，并不证明故障后制动通道一定独立有效。"
            "不能仅为得到C值声称remaining_controls；无法合理作条件性判断的能力用unresolved_controls具体说明。"
            "intervention_inputs是下游实际需要的干预条件；应在同一次方案中逐项检查。"
            "emergency_braking_available由你明确声明的failed_controls/remaining_controls派生，"
            "请说明当前故障后驾驶员制动能力是否保留及其条件；它与驾驶员是否在车内是两个条件。"
            "同时检查远程干预和目标参与者能否规避。已有事实足够时沿用；有合理条件性依据时，"
            "用可选intervention_assumptions明确给出remote_intervention_available和/或"
            "other_road_user_avoidance_possible，每项为{value:布尔值,rationale:理由,basis_refs:引用ID数组,"
            "conditions:适用条件数组}。未知就省略，不把未提及解释为false；有App不等于故障后能远程制动，"
            "目标静止不等于人员不能逃避。remote为true须有剩余remote_control路径及未失效的执行能力。"
            "在本次方案中一并提出安全状态候选，使用可选safe_state_proposal："
            "{description:简明目标状态,required_controls:依赖的剩余控制键数组,basis_refs:引用ID数组,"
            "conditions:适用条件数组}。描述希望达到的车辆状态，不能仅复述控制指令；"
            "依赖须是当前故障后仍可用的控制，不能凭空增加冗余、时间/距离指标或自动批准。"
            "不同驾驶员位置的可达性须分别检查。无依据时省略并在未覆盖条件说明具体障碍。"
            "安全状态候选只用于待审核方案，不证明到达时间，不输出FTTI，也不改变无干预轨迹和评级。"
            "不得依赖已失效的控制消除危险。坡道溜车初速为 0，使用来源坡度和假设残余阻力。"
            "侧向模型只支持沿纵向运动路径延伸的静止侧向目标，不能假装成纵向 TTC。"
            "纯转向改变运动方向且无切向加速时，选CONSTANT_SPEED_CURVATURE_TO_CONTACT："
            "总速度保持所选初始速度，提出有依据的turn_radius_m、侧向有效接触净距和model_valid_until_s；"
            "geometry.ego_lateral_direction明确LEFT或RIGHT的位移方向，不能仅看目标在哪侧就认定朝它转弯。"
            "半径、无侧滑和等速必须列为分析假设，不能冒充实测转向/附着能力；不覆盖失稳或同时驱动异常。"
            "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT表示惯性坐标独立横向加速/漂移，"
            "固定vx再增加vy会增大总速度，不能用它代替纯转向，也不能将运行暴露Atom的正常向心加速度直接当故障横向加速度。"
            "RECTILINEAR_MOVING_TARGET_CONTACT支持同向、对向、后方接近和横穿的匀速移动目标；"
            "使用车体坐标x前、y左的初始中心偏移，接触包络半长/半宽为双方投影半尺寸之和，"
            "须解释尺寸和有效时段假设。geometry用FINITE_CONTACT_ENVELOPE并给出"
            "object_longitudinal_direction=FORWARD|REVERSE|LEFT|RIGHT（均相对车体，非道路行驶方向）。"
            "自车沿当前行驶方向匀加减速；不支持转弯、反向启动和多阶段限速。"
            "后车接近等机制不能因本车先停车就宣布无接触。若有相容依据假设停车后保持原位置，"
            "在该假设顶层给post_stop_motion=HOLD_POSITION，并在conditions和rationale说明保持条件、坡度及控制相容性；"
            "程序会在同一model_valid_until_s内继续计算移动目标。缺少该假设时只计算至本车停车，不代表停车后无风险。"
            "路径交叉但到达时间不同可以无接触；不得为了碰撞强行调参。"
            "参数可以 POINT(value)、RANGE(lower,value,upper；value 是代表值)、SYMBOLIC(symbol)。"
            "不确定真实上界时给条件性代表值/敏感性范围或符号条件，不编造最大速度。"
            "距离、净距等缺少实测值时，也可提出有依据的条件性代表值并说明适用范围；"
            "无法合理选值则保留明确符号，程序输出参数关系，不得借用其他对象或阶段的距离。"
            "依据必须引用 evidence 中的 ID，但分析假设无需在来源中逐字出现。"
            "只返回规定 JSON；禁止附加字段。"
            "显式 SYMBOLIC 不得用父场景旧阶段数值补回。未填写参数只有同对象、同阶段、同角色和适用模型才可继承。"
            "同一初始静止条件不能同时是非零初速；同一移动目标不能改成静止。需要新初始工况/新对象时，"
            "用 condition_changes 明确声明并说明依据；新对象给不同 object_id，程序重新绑定受影响 Atom。"
            "例如来源初速非零而坡道模型要求静止，必须明确 initial_state=true 及独立静止工况依据，"
            "不能一边写初速0一边声明初始条件未变；也可以保留原条件并说明当前模型不适用。"
            "坡道模型必须提供output_shape中的grade_frame，其他模型省略它；"
            "道路参考必须选用model_input_status.grade_reference_mapping中相容的朝向组合。车头朝上坡时重力向后，"
            "车头朝下坡时向前；模型运动方向必须沿重力，目标是否在该方向由程序判断。"
            "静止起步纯重力向后时，不能声称撞上车前静止目标；可保留为不接触对照，"
            "另设车后目标必须声明新对象条件，不能只在文字中换目标。"
            "来源只有上坡标签而未说明朝向时，以明确分析假设选择少量朝向，不把标签自动当作车头方向。"
            "correction_diagnostics 如存在，保留有效分支并针对具体冲突有限修正，不能修改原始事实或声称批准。"
            "correction_diagnostics只属于输入，输出顶层只能含hypotheses和uncovered_conditions。"
        )
        output_contract = {
            "basis_ref_ids": sorted(context["evidence"]),
            "geometry_values": {field: sorted(values) for field, values in context.get("risk_vocabulary", {}).items()},
            "control_identities": sorted(CONTROL_IDENTITIES),
            "control_arrays": {
                "fields": ["failed_controls", "remaining_controls"],
                "type": "flat JSON array of control identity strings; [] when none",
                "shape_example": ["driver_braking", "braking"],
                "meaning": "Shape only, not a default capability. Do not nest arrays or output expressions such as [[...]][0]. Put reasons and conditions in the existing rationale/conditions fields.",
            },
            "driver_positions": {"optional": True, "allowed": context.get("driver_configurations", []),
                "meaning": "hypothesis applicability; omitted means all supplied configurations"},
            "unresolved_controls": {"optional_keys": sorted(CONTROL_IDENTITIES | INTERVENTION_ASSUMPTION_FIELDS),
                "values": "nonempty reasons after assessment, never a default for omitted analysis"},
            "intervention_assumptions": {
                "type": "object", "optional_keys": sorted(INTERVENTION_ASSUMPTION_FIELDS),
                "entry_shape": {"value": "boolean, never null", "rationale": "reason",
                    "basis_refs": ["a supplied evidence ID"], "conditions": ["explicit applicability"]},
                "unknown": "omit that key; use {} if all unknown, never null or an array",
                "derived_inputs": "driver_in_vehicle comes from each driver branch; emergency_braking_available comes from failed_controls/remaining_controls, not additional keys here",
            },
            "parameter_payloads": {"POINT": {"value": "finite number"},
                "RANGE": {"lower": "finite number", "value": "representative number", "upper": "finite number"},
                "SYMBOLIC": {"symbol": "nonempty string; omit value/lower/upper"}},
            "moving_target_geometry": {"object_extent": "FINITE_CONTACT_ENVELOPE",
                "object_longitudinal_direction": ["FORWARD", "REVERSE", "LEFT", "RIGHT"]},
            "post_stop_motion": {"optional": True, "allowed": ["HOLD_POSITION"],
                "placement": "hypotheses[] sibling of geometry and parameters",
                "meaning": "Explicit conditional stationary continuation after ego stop; omitted retains pre-stop scope. Explain grade/control compatibility in conditions and rationale."},
        }
        # Grading tables belong to the deterministic scorer, not this proposal
        # task. Keep the complete contract in the cache identity/audit context;
        # send all physical/source/engineering constraints and the already
        # explicit intervention_inputs, without downstream S/C classifications.
        prompt_inputs = deepcopy(context)
        policy = prompt_inputs.get("method_identity", {}).get("project_analysis_policy", {})
        for downstream in ("controllability", "severity_evaluation"):
            policy.pop(downstream, None)
        # Make the active response capability explicit alongside the motion
        # catalog. Otherwise a proposal can incorrectly reject a supported
        # response just because its deceleration is absent from the unmitigated
        # motion model. This is a projection of MethodContract, not a new rule.
        if timing := policy.get("timing_analysis"):
            output_contract["response_timing"] = deepcopy(timing)
        prompt_context, source_texts = physical_prompt_view(prompt_inputs)
        content = {"output_contract": output_contract, "context": prompt_context, "source_texts": source_texts,
                   "model_catalog": MODEL_CATALOG,
                   "parameter_units": PARAMETER_UNITS, "output_shape": example,
                   "control_identities": sorted(CONTROL_IDENTITIES),
                   "contract_version": PHYSICAL_PLAN_VERSION}
        if focus := context.get("source_condition_focus"):
            example["hypotheses"][0]["operating_condition"] = {
                **focus, "physical_effect": "Explain the actual effect on this geometry/motion/visibility; choose a compatible alternative from the allowed list if needed"}
        controls_only = context.get("task_scope") == "CONTROL_INPUTS_ONLY"
        if controls_only:
            # Same bounded agent/transport/repair ledger, with motion explicitly
            # out of scope. Avoid contradictory motion-generation instructions.
            system = (
                "你是故障后物理场景分析员。当前轨迹和碰撞条件已经由现有生产路径确定。"
                "本次只分析 intervention_inputs 中缺少的干预能力，不重新生成运动、参数、对象或碰撞分支。"
                "读取 context 的原文、限定与来源；$context_ref和$source_text是完整来源引用；"
                "$context_value指向context.$context_values中的完整结构，可递归引用，来源、对象与阶段未省略。"
                "根据功能、失效作用范围及现有条件，区分自动指令故障 automatic_braking_command、"
                "制动能力本身 braking 和独立驾驶员制动路径 driver_braking。已有明确控制条件必须保留。"
                "正常时踩制动能退出功能，不等于故障后独立制动一定有效；允许有依据的条件性能力假设，"
                "明确写出适用条件，不能声称工程批准。驾驶员位置由各分支提供，本任务不改变它。"
                "远程干预和目标规避通过 intervention_assumptions 表达，每项包含 value 布尔值、"
                "rationale、basis_refs、conditions。仅可用output_contract.intervention_assumptions.optional_keys，"
                "不是intervention_inputs中的所有字段；driver_in_vehicle由分支提供，"
                "emergency_braking_available由控制清单派生。未知项省略，全部未知用{}，不输出null或数组。"
                "未知不能当成false，静止目标不等于人员无法逃避，"
                "有App不等于故障后远程制动有效。remote为true须有remaining_controls中的remote_control"
                "和仍可用的执行能力。不能依赖失效的控制。无法形成合理假设时具体说明未覆盖条件。"
                "failed_controls/remaining_controls只能使用control_identities；basis_refs只能使用"
                "output_contract.basis_ref_ids。每项假设必须有来源依据和明确条件。"
                "只返回output_shape规定的JSON：hypotheses为空，control_assessment为控制能力分析。"
                "禁止输出S/E/C/ASIL、批准状态、速度、距离、TTC、碰撞分支、几何或新运动模型。"
                "correction_diagnostics仅用于纠正相容性问题，保留有效输入。"
            )
            content["output_shape"] = {"hypotheses": [], "uncovered_conditions": [], "control_assessment": {
                "rationale": "failure scope and capability basis", "basis_refs": ["MF.description"],
                "conditions": ["explicit conditional applicability"], "uncovered_conditions": [],
                "failed_controls": [], "remaining_controls": [], "intervention_assumptions": {},
                "unresolved_controls": {}}}
            content["model_catalog"] = {}
            content["parameter_units"] = {}
        system += self.CONTROL_ANALYSIS_POLICY
        system += (
            "安全状态和时间分析在同一次请求内完成，不另造运动世界：每个假设/control_assessment给出"
            "safe_state_proposal或safe_state_unresolved_reason。proposal包含description、required_controls、"
            "basis_refs、conditions，可选driver_positions限定人员位置。必须使用remaining_controls，"
            "只依赖人工踏板的方案不能覆盖车外驾驶员；车外无可用路径要明确说明。"
            "先检查故障之外的自动/远程制动链是否仍可作为相容的条件性处置，不要默认所有停车都只能靠人工踏板。"
            "同一运动覆盖两个驾驶员位置而处置不同，使用可选driver_control_assessments对象，键为"
            "in_driver_seat/outside_driver_seat。每项为完整控制分析：rationale、basis_refs、conditions、"
            "uncovered_conditions、failed_controls、remaining_controls，以及适用的intervention_assumptions、"
            "safe_state_proposal或safe_state_unresolved_reason、hazard_timing或timing_unresolved_reason。"
            "该对象可放在hypothesis或control_assessment内；不要复制运动假设，不新增参数取点。"
            "明确说明每个所请求驾驶员位置的处置或具体缺口；不能只写车内方案就默认完成车外分析。"
            "自动路径须同时说明remaining_controls中的automatic_braking_command与braking，远程路径须说明remote_control与braking；"
            "命令链、执行能力、故障独立性和响应条件分别交代，有App或完整执行器并不自动证明这条链可用。"
            "优先采用适用于本工况的剩余路径；无法成立时保留明确局部缺口，不能转移目标来让停车看起来安全。"
            "已提出依赖剩余制动的停车安全状态时，应在proposal内给出transition={model:CONSTANT_NET_DECELERATION,"
            "net_deceleration_mps2:正数,rationale:理由,basis_refs:引用数组}。该数值是抵消故障驱动力/坡度后"
            "的净减速度假设，不是车辆实测能力，不得沿用方法示例的默认减速度。停止是否安全仍须考虑目标轨迹。"
            "移动目标使用该响应时，transition还须包含post_stop_motion:HOLD_POSITION，说明停止后由哪个剩余控制"
            "持续保持；目标仍按原轨迹运动，不能在本车停车时结束分析。停车后仍被横穿者/后车撞到就不是避碰安全状态。"
            "停车保持不必机械依赖EPB：若本代表可由持续的人工/常规制动维持静止，应清楚声明这一路径及条件；"
            "若确实需要parking_hold，则必须单独分析其可用性，不能把未声明能力藏进safe_state_proposal。"
            "不能只声明可以停车却省略响应建模；若连条件性响应也不适用，在同一hypothesis/control_assessment的"
            "timing_unresolved_reason中写具体障碍，单纯缺少实测减速度不能作为拒绝分析假设的理由。"
            "可附hazard_timing={time_origin:FAULT_ONSET,fault_to_model_start_s:非负数,"
            "hazard_criterion:LOSS_OF_BRAKING_AVOIDABILITY或FIRST_CONTACT_BOUND,rationale:理由,basis_refs:引用数组}。"
            "前者用于有制动响应路径的失去避碰能力边界，包括静止目标和同一有限包络的直线移动目标；后者只是接触时间证据。"
            "不能把模型初态自动当故障时刻，需说明二者关系；不输出计算结果或最终FTTI。"
            "output_contract.response_timing若存在，是当前方法已接通的响应计算能力。"
            "model_catalog描述无干预故障运动，不能仅因其中没有制动阶段就拒绝已列入supported_models的响应分析；"
            "沿用同一几何、对象及模型适用假设，由程序检查响应是否真的避开接触。"
            "移动目标的other_road_user_avoidance_possible=true必须给出同一项中的avoidance={"
            "recognition_delay_s:非负秒,escape_direction:FORWARD|REVERSE|LEFT|RIGHT,escape_speed_mps:正数,"
            "available_clearance_m:正数}，并论证可见性和逃生空间；程序检验同一几何，不是只检查理由非空。"
            "初始静止的本车被横穿者接触时，要检查即使本车不动是否仍会接触；不能把原本就发生的接触归因于溜车。"
        )
        if not controls_only and context.get("operating_atom_options"):
            # Keep the output example consistent with the enforced contract.
            # Otherwise live models repeatedly follow the incomplete example
            # and consume a schema-repair request just to add these bindings.
            example["hypotheses"][0]["operating_atoms"] = {
                "EGO_ACTION": "atom_id from context.operating_atom_options.EGO_ACTION",
                "EGO_DYNAMICS": "atom_id from context.operating_atom_options.EGO_DYNAMICS",
                "TRAFFIC_PATTERN": None,
            }
            output_contract["operating_atoms"] = {
                "required_per_hypothesis": True,
                "allowed_ids": {dimension: [a["atom_id"] for a in options]
                                for dimension, options in context["operating_atom_options"].items()},
                "nullable": ["TRAFFIC_PATTERN"],
                "omission_reason": "For a justified null, supply operating_atom_exclusions={TRAFFIC_PATTERN: explanation why the supplied catalog relations do not describe this particular world}. Null without explanation remains unresolved; no rule or approval is inferred.",
                "compound_selections": list({row["atom_id"]: {
                    dimension: row["atom_id"] for dimension in row.get("fills", [])}
                    for options in context["operating_atom_options"].values() for row in options
                    if len(row.get("fills", [])) > 1}.values()),
                "compound_rule": "Each compound selection is indivisible in the base AND every driver-specific effective selection. Do not combine one of its dimensions with a different atom. Driver outside does not imply vehicle parked/holding; keep the shared moving action unless its definition actually restricts the driver.",
            }
            system += (
                "operating_atom_options是当前方法的有界候选，不是评级规则。每个假设必须输出operating_atoms="
                "{EGO_ACTION:候选ID,EGO_DYNAMICS:候选ID,TRAFFIC_PATTERN:候选ID或null}，"
                "与本假设真实运行状态、动作和交互一致。只从所给相应候选选择，不能沿用矛盾的父动作。"
                "故障后加速不改变正常使用暴露；EGO_DYNAMICS表示故障前初态。"
                "TRAFFIC_PATTERN只表达候选目录所定义的特殊交通关系，并非只要有移动目标就必须选一个。"
                "若目录条目均不描述本假设，填null并给出operating_atom_exclusions={TRAFFIC_PATTERN:具体理由}，"
                "说明本世界的关系以及为何候选关系不适用；不能用缺信息、较低评级或故障后速度替代适用性判断。"
                "若关系尚不清楚则保留null而不声明排除；已有来源明确要求的交通关系不能被排除。"
                "fills列明复合Atom占用的全部维度；选中复合Atom时，它占用的维度都填写同一个ID，不能与另一Atom重复占用。"
                "WHERE/ROAD沿用来源，不为提高评级改动。不能把停车入位写成路口横穿或对向超车。"
                "驻车保持必须选驻留动作，不能为了初速为0选红灯/人行横道交通停车。候选动作中的驾驶员位置限定也必须满足。"
                "同一运动方案适用两种驾驶员位置而动作条目不同，用driver_operating_atoms="
                "{in_driver_seat:{EGO_ACTION:对应ID},outside_driver_seat:{EGO_ACTION:对应ID}}覆盖各自动作，"
                "不因此复制物理假设；没有驾驶员限定的共同动作无需此字段。"
            )
        system += ("每个hypotheses成员都必须单独填写非空basis_refs，引用所给evidence ID；"
                   "parameters、intervention_assumptions或safe_state_proposal里的引用不能替代该字段。")
        attempts = []
        errors = []
        previous_response = None
        for repair in (False, True):
            request = LLMRequest(task="construct_post_fault_physical_scenario",
                system_prompt=system, user_prompt=json.dumps({**content, "repair_errors": errors,
                    **({"previous_invalid_response": previous_response} if repair else {})}, ensure_ascii=False, default=str, separators=(",", ":")),
                schema_name="PhysicalScenarioPlan", prompt_version=self.CONTROL_PROMPT_VERSION if controls_only else self.PROMPT_VERSION,
                # Motion plans include up to three geometries and their parameter
                # provenance; real complete plans can exceed 4096 output tokens.
                metadata={"physical_plan_key": context["plan_key"], "schema_repair": repair,
                    **({"strict_no_format_retry": True} if repair else {})},
                max_tokens=4096 if controls_only else 8192,
                response_format=self.response_format)
            response = self.responses.complete(self.client, request) if self.responses else self.client.complete_json(request)
            record = {"request": asdict(request), "response": asdict(response), "schema_repair": repair}
            attempts.append(record)
            on_response(record)  # Persist even a semantically invalid response before repair.
            try:
                candidate = deepcopy(response.data)
                envelope_normalization = ""
                if (not controls_only and isinstance(candidate, dict) and "hypotheses" not in candidate
                        and {"hypothesis_id", "model", "geometry", "parameters", "basis_refs", "uncovered_conditions"} <= candidate.keys()):
                    # A real single-hypothesis reply omitted only the outer
                    # list. Preserve every field and still validate the full
                    # hypothesis; this is not a repair of its physical claims.
                    candidate = {"hypotheses": [candidate],
                        "uncovered_conditions": deepcopy(candidate["uncovered_conditions"])}
                    envelope_normalization = "WRAPPED_SINGLE_MOTION_HYPOTHESIS"
                if controls_only and isinstance(candidate, dict):
                    # Real control-only replies sometimes omit the outer plan
                    # envelope. Preserve the complete control object and its
                    # own limits; never repair capability values or conflicts.
                    if ("control_assessment" not in candidate
                            and "remaining_controls" in candidate
                            and "uncovered_conditions" in candidate):
                        candidate = {"hypotheses": [],
                            "uncovered_conditions": deepcopy(candidate["uncovered_conditions"]),
                            "control_assessment": candidate}
                        envelope_normalization = "WRAPPED_CONTROL_ASSESSMENT"
                    elif (isinstance(candidate.get("control_assessment"), dict)
                            and "uncovered_conditions" not in candidate
                            and "uncovered_conditions" in candidate["control_assessment"]):
                        candidate["uncovered_conditions"] = deepcopy(candidate["control_assessment"]["uncovered_conditions"])
                        envelope_normalization = "PROJECTED_CONTROL_LIMITS"
                normalized_refs = normalize_item_basis_refs(candidate, set(context["evidence"]))
                deferred_states = []
                control_array_normalizations = []
                # An optional safe-state proposal cannot establish an unassessed
                # or failed control. Reject that proposal locally, retaining the raw reply;
                # do not discard otherwise valid motion or approve the control.
                if isinstance(candidate, dict) and isinstance(candidate.get("hypotheses"), list):
                    for hypothesis in [*candidate["hypotheses"], candidate.get("control_assessment")]:
                        if not isinstance(hypothesis, dict):
                            continue
                        for field in ("failed_controls", "remaining_controls"):
                            values = hypothesis.get(field)
                            if (isinstance(values, list) and len(values) == 1
                                    and isinstance(values[0], list) and values[0]
                                    and all(isinstance(v, str) and v in CONTROL_IDENTITIES for v in values[0])):
                                hypothesis[field] = list(values[0])
                                control_array_normalizations.append({"hypothesis_id": hypothesis.get("hypothesis_id", "control_assessment"),
                                    "field": field, "normalization": "UNWRAPPED_SINGLE_CONTROL_ARRAY"})
                        proposal = hypothesis.get("safe_state_proposal")
                        controls = proposal.get("required_controls") if isinstance(proposal, dict) else None
                        remaining = hypothesis.get("remaining_controls")
                        failed = hypothesis.get("failed_controls")
                        if (isinstance(controls, list) and controls
                                and all(isinstance(c, str) for c in controls)
                                and isinstance(remaining, list) and isinstance(failed, list)
                                and all(isinstance(c, str) for c in remaining + failed)
                                and set(controls) - set(remaining)
                                and set(controls) <= CONTROL_IDENTITIES
                                and isinstance(hypothesis.get("uncovered_conditions"), list)):
                            failed_dependency = bool(set(controls) & set(failed))
                            deferred_states.append({"hypothesis_id": hypothesis.get("hypothesis_id"),
                                "reason": "SAFE_STATE_REQUIRES_FAILED_CONTROL" if failed_dependency else "SAFE_STATE_REQUIRES_UNRESOLVED_CONTROL",
                                "proposal": hypothesis.pop("safe_state_proposal")})
                            hypothesis["safe_state_unresolved_reason"] = (
                                "安全状态提案依赖已失效控制，提案不采用；原始提案保留在响应审计中。" if failed_dependency else
                                "安全状态提案所需控制能力尚未成立，提案局部待定；原始提案保留在响应审计中。")
                            hypothesis["uncovered_conditions"].append(hypothesis["safe_state_unresolved_reason"])
                plan = PhysicalScenarioPlan.from_dict(candidate, evidence_ids=set(context["evidence"]))
                if context.get("operating_atom_options") and not controls_only:
                    for hypothesis in plan.hypotheses:
                        if "operating_atoms" not in hypothesis:
                            raise ValueError("Each motion hypothesis must bind operating_atoms to its actual initial operation")
                        for dimension, selected in hypothesis["operating_atoms"].items():
                            if selected is None and dimension == "TRAFFIC_PATTERN":
                                continue
                            if selected not in {row["atom_id"] for row in context["operating_atom_options"].get(dimension, [])}:
                                raise ValueError("operating_atoms selection outside supplied shortlist: " + dimension)
                        for atoms in hypothesis.get("driver_operating_atoms", {}).values():
                            if atoms["EGO_ACTION"] not in {row["atom_id"] for row in context["operating_atom_options"].get("EGO_ACTION", [])}:
                                raise ValueError("driver_operating_atoms selection outside supplied shortlist: EGO_ACTION")
                        from hara_agent.services.analysis.scenario_constraint_service import ScenarioConstraintExecutor
                        fills = {row["atom_id"]: row.get("fills", [])
                            for options in context["operating_atom_options"].values() for row in options}
                        for override in ({}, *hypothesis.get("driver_operating_atoms", {}).values()):
                            selection = {**hypothesis["operating_atoms"], **override}
                            conflicts = ScenarioConstraintExecutor.compound_atom_conflicts(
                                {d: {a} if a is not None else set() for d, a in selection.items()}, fills)
                            if conflicts:
                                required = {a: {d: a for d in fills[a] if d in selection}
                                    for a in selection.values() if a in fills and len(fills[a]) > 1}
                                raise ValueError("; ".join(conflicts) + "; selected=" + json.dumps(selection)
                                    + "; if retaining a selected compound Atom, its required assignments are " + json.dumps(required)
                                    + ". Change the inconsistent Atom bindings in your complete reply, or choose a compatible non-compound alternative from the supplied shortlist. Correcting these bindings does not require changing valid physical parameters. Driver location alone does not change a moving vehicle into parking/holding.")
                if controls_only != (plan.control_assessment is not None):
                    raise ValueError("Output must match task_scope; control-only tasks cannot replace existing motion")
                if context.get("previous_plan") is not None:
                    PhysicalScenarioPlan.validate_correction_scope(context["previous_plan"], plan.to_dict(),
                        context.get("correction_diagnostics", []), context.get("driver_configurations", []))
                return plan, {"responses": attempts, "validation_errors": errors,
                    "request_count": sum((1 + a["response"]["usage"].get("format_retry_calls", 0))
                        for a in attempts if not a["response"]["usage"].get("response_checkpoint_hit")),
                    "response_cache_hits": sum(bool(a["response"]["usage"].get("response_checkpoint_hit")) for a in attempts),
                    "schema_repairs": int(repair), "model": response.model,
                    "control_envelope_normalization": envelope_normalization,
                    "control_array_normalizations": control_array_normalizations,
                    "deferred_safe_states": deferred_states, "normalized_basis_refs": normalized_refs}
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
                previous_response = response.data
                if response.usage.get("format_retry_calls"):
                    # Syntax repair already used the single correction. Do not
                    # stack another schema repair onto the same proposal task.
                    break
        raise ValueError("Physical plan schema repair exhausted: " + "; ".join(errors))
