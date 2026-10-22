"""A workflow-scheduled proposal task; no scoring or approval authority."""
from dataclasses import asdict
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from hara_agent.contracts.physical_scenario_plan import (
    MODEL_CATALOG, PARAMETER_UNITS, PHYSICAL_PLAN_VERSION, PhysicalScenarioPlan, CONTROL_IDENTITIES,
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

    return intern(view), texts


def expand_physical_prompt_view(view: dict, source_texts: dict) -> dict:
    """Reconstruct the original context for audits and offline equivalence tests."""
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$source_text"}:
                return source_texts[value["$source_text"]]
            if set(value) == {"$context_ref"}:
                return expand(view[value["$context_ref"]])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    return expand(view)


class PhysicalScenarioAgent:
    PROMPT_VERSION = "post-fault-physical-scenario-v10-intervention-safe-state"
    ANALYSIS_STRATEGY = (
        "本任务先完成当前失效的一个条件性代表工况：对象、方向、净距、初始运动及故障运动必须共同成立。"
        "依据model_input_status与model_catalog.required逐项沿用适用输入或提出假设；"
        "缺少实测距离、加速度或残余阻力时，可以自行选取有场景依据的POINT并解释选值、参考点、"
        "有效阶段和未覆盖范围。假设不等于测量、批准或车辆能力上界，不设通用数值。"
        "仅当连条件性代表也无法合理成立时保留SYMBOLIC，说明具体相容性/模型障碍及参数关系；"
        "仅说原文未提供实测值不是充分理由。无接触代表同样有效，不以得到某个评级为选值目标。"
        "默认只给代表点；只有来源明确要求范围，或能指出接触、停止、方向、使能或规避条件的"
        "具体变化时才增加RANGE。数值随参数连续变化本身不是展开理由。符号几何下不展开无新结果的范围。"
        "额外假设须属于当前失效实际引起的不同危险机制，不附加另一个独立故障；"
        "例如纵向扭矩超量不能凭空附加左右扭矩不对称或转向故障，驻车失效不额外生成无因果依据的行驶加速。"
        "计算与评级全部由Python完成。"
    )

    def __init__(self, client, *, response_checkpoint_dir: Path | None = None):
        self.client = client
        self.responses = ResponseCheckpoint(response_checkpoint_dir) if response_checkpoint_dir else None

    def propose(self, context: dict, *, on_response=lambda record: None) -> tuple[PhysicalScenarioPlan, dict]:
        example = {
            "hypotheses": [{
                "hypothesis_id": "h1", "model": "one model_catalog key", "rationale": "mechanism and applicability",
                "basis_refs": ["a supplied evidence ID"],
                "geometry": {"road_user_type": "from method vocabulary", "collision_type": "from method vocabulary",
                    "object_position": "FRONT|REAR|LEFT|RIGHT", "ego_longitudinal_direction": "FORWARD|REVERSE",
                    "object_extent": "POINT_ON_TRAVEL_PATH|EXTENDED_ALONG_TRAVEL_PATH", "object_id": "target1"},
                "parameters": [{"field": "one parameter_units key", "unit": "exact declared unit",
                    "kind": "POINT", "value": 0, "rationale": "why this conditional representative is useful"}],
                "conditions": ["scope of this hypothesis"], "uncovered_conditions": ["limits of this analysis"],
                "failed_controls": ["affected control"], "remaining_controls": ["independent control, if supported"],
                "condition_changes": {"initial_state": False, "target": False, "rationale": "same initial world and target; set true only for a justified independent child condition"},
            }], "uncovered_conditions": ["not an exhaustive vehicle capability envelope"],
        }
        system = (
            "输入使用无损引用：$context_ref 指向 context 的同名字段，$source_text 指向 source_texts 中的完整原文。"
            "引用处保留来源身份、位置和适用范围；读取所引用全文，包括否定和限定，不得把引用当成缺失来源。"
            "先遵守 output_contract 的精确字段、枚举和引用ID。geometry_values列的是可输出的键，"
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
            "不得把假设声称为原文事实、车辆真实能力或可信上界。不得输出 S/E/C/ASIL、批准状态、"
            "碰撞速度、TTC、collision_branches 或 Atom ID。程序会计算这些量。"
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
            "不能仅为得到C值声称remaining_controls；不支持的独立性留在未覆盖条件。"
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
            "所有支持模型当前只计算静止目标，移动目标列为未覆盖。"
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
            "坡道模型必须提供 grade_frame：coordinate_frame=BODY_FORWARD_POSITIVE，"
            "slope_reference=EGO_FORWARD|ROAD_REFERENCE|UNSPECIFIED，heading_relative_to_downhill=UPHILL|DOWNHILL，"
            "downhill_body_direction=FRONT|REAR，orientation_basis=SOURCE|ANALYSIS_ASSUMPTION，rationale。"
            "道路参考另给 ego_forward_relative_to_reference=SAME|OPPOSITE。车头朝上坡时重力向后，"
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
            "parameter_payloads": {"POINT": {"value": "finite number"},
                "RANGE": {"lower": "finite number", "value": "representative number", "upper": "finite number"},
                "SYMBOLIC": {"symbol": "nonempty string; omit value/lower/upper"}},
        }
        prompt_context, source_texts = physical_prompt_view(context)
        content = {"output_contract": output_contract, "context": prompt_context, "source_texts": source_texts,
                   "model_catalog": MODEL_CATALOG,
                   "parameter_units": PARAMETER_UNITS, "output_shape": example,
                   "control_identities": sorted(CONTROL_IDENTITIES),
                   "contract_version": PHYSICAL_PLAN_VERSION}
        attempts = []
        errors = []
        previous_response = None
        for repair in (False, True):
            request = LLMRequest(task="construct_post_fault_physical_scenario",
                system_prompt=system, user_prompt=json.dumps({**content, "repair_errors": errors,
                    **({"previous_invalid_response": previous_response} if repair else {})}, ensure_ascii=False, default=str),
                schema_name="PhysicalScenarioPlan", prompt_version=self.PROMPT_VERSION,
                metadata={"physical_plan_key": context["plan_key"], "schema_repair": repair}, max_tokens=4096)
            response = self.responses.complete(self.client, request) if self.responses else self.client.complete_json(request)
            record = {"request": asdict(request), "response": asdict(response), "schema_repair": repair}
            attempts.append(record)
            on_response(record)  # Persist even a semantically invalid response before repair.
            try:
                plan = PhysicalScenarioPlan.from_dict(response.data, evidence_ids=set(context["evidence"]))
                return plan, {"responses": attempts, "validation_errors": errors,
                    "request_count": sum(not a["response"]["usage"].get("response_checkpoint_hit") for a in attempts),
                    "response_cache_hits": sum(bool(a["response"]["usage"].get("response_checkpoint_hit")) for a in attempts),
                    "schema_repairs": int(repair), "model": response.model}
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
                previous_response = response.data
        raise ValueError("Physical plan schema repair exhausted: " + "; ".join(errors))
