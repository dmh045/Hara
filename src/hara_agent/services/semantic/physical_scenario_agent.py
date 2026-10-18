"""A workflow-scheduled proposal task; no scoring or approval authority."""
from dataclasses import asdict
import json
from pathlib import Path

from hara_agent.contracts.physical_scenario_plan import (
    MODEL_CATALOG, PARAMETER_UNITS, PHYSICAL_PLAN_VERSION, PhysicalScenarioPlan, CONTROL_IDENTITIES,
)
from hara_agent.infrastructure.llm import LLMRequest
from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint


class PhysicalScenarioAgent:
    PROMPT_VERSION = "post-fault-physical-scenario-v2-scoped-inputs-grade-frame"

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
            "你是故障后物理场景分析员。使用给定项目事实、来源限定、FM、Atom 和已确认方法约束，"
            "提出 1–3 个有工程意义的条件性物理假设；原文未提供故障加速度并不禁止提出分析假设。"
            "不得把假设声称为原文事实、车辆真实能力或可信上界。不得输出 S/E/C/ASIL、批准状态、"
            "碰撞速度、TTC、collision_branches 或 Atom ID。程序会计算这些量。"
            "优先保留已给定的对象类别和运行条件；新对象需说明适用性，不得扩大 ODD。"
            "正常速度约束仅约束初始速度；故障可加速、恒速或减速。区分失效控制与剩余控制，"
            "不得依赖已失效的控制消除危险。坡道溜车初速为 0，使用来源坡度和假设残余阻力。"
            "侧向模型只支持沿纵向运动路径延伸的静止侧向目标，不能假装成纵向 TTC。"
            "所有支持模型当前只计算静止目标，移动目标列为未覆盖。"
            "参数可以 POINT(value)、RANGE(lower,value,upper；value 是代表值)、SYMBOLIC(symbol)。"
            "不确定真实上界时给条件性代表值/敏感性范围或符号条件，不编造最大速度。"
            "依据必须引用 evidence 中的 ID，但分析假设无需在来源中逐字出现。"
            "只返回规定 JSON；禁止附加字段。"
            "显式 SYMBOLIC 不得用父场景旧阶段数值补回。未填写参数只有同对象、同阶段、同角色和适用模型才可继承。"
            "同一初始静止条件不能同时是非零初速；同一移动目标不能改成静止。需要新初始工况/新对象时，"
            "用 condition_changes 明确声明并说明依据；新对象给不同 object_id，程序重新绑定受影响 Atom。"
            "坡道模型必须提供 grade_frame：coordinate_frame=BODY_FORWARD_POSITIVE，"
            "slope_reference=EGO_FORWARD|ROAD_REFERENCE|UNSPECIFIED，heading_relative_to_downhill=UPHILL|DOWNHILL，"
            "downhill_body_direction=FRONT|REAR，orientation_basis=SOURCE|ANALYSIS_ASSUMPTION，rationale。"
            "道路参考另给 ego_forward_relative_to_reference=SAME|OPPOSITE。车头朝上坡时重力向后，"
            "车头朝下坡时向前；模型运动方向必须沿重力，目标是否在该方向由程序判断。"
            "来源只有上坡标签而未说明朝向时，以明确分析假设选择少量朝向，不把标签自动当作车头方向。"
            "correction_diagnostics 如存在，保留有效分支并针对具体冲突有限修正，不能修改原始事实或声称批准。"
        )
        content = {"context": context, "model_catalog": MODEL_CATALOG,
                   "parameter_units": PARAMETER_UNITS, "output_shape": example,
                   "control_identities": sorted(CONTROL_IDENTITIES),
                   "contract_version": PHYSICAL_PLAN_VERSION}
        attempts = []
        errors = []
        for repair in (False, True):
            request = LLMRequest(task="construct_post_fault_physical_scenario",
                system_prompt=system, user_prompt=json.dumps({**content, "repair_errors": errors}, ensure_ascii=False, default=str),
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
        raise ValueError("Physical plan schema repair exhausted: " + "; ".join(errors))
