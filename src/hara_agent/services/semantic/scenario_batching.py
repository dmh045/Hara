from __future__ import annotations

import json

from hara_agent.models import MalfunctionCandidate, ScenarioCandidate
from hara_agent.config.llm_config import LLMConfig

from .scenario_contract import RISK_DIMENSION_VALUES
from .scenario_evidence import (
    CausalBreakpoint, CausalEvidenceSelection, CausalEvidenceSelector,
    DEFAULT_CAUSAL_EVIDENCE_BUDGET, EvidenceBasisType, FactRegistry,
    build_fact_registry,
)


DEFAULT_SCENARIO_BATCH_MAX_CHARS = LLMConfig.scenario_batch_max_chars
DEFAULT_SCENARIO_BATCH_MAX_ITEMS = LLMConfig.scenario_batch_max_items


DRIVER_COMPARISON_TASK = """只有输入 driver_reuse_comparison 时，额外比较其中明确列出的驾驶员配置；它表示非驾驶员结构化输入相同，不是无关性证明，也不是新增待返回身份。分别核查 enablement（功能在这些配置下使能）、trigger（失效触发）、interaction（故障车辆与当前目标交互）、hazard（危险事件形成）。当前运行状态已明确给定时，只判断该状态下的依赖，不要求证明全项目所有状态的使能；若该状态本身依赖驾驶员动作或在位条件，仍必须分别判断。trajectory_scope=UNMITIGATED时，比较的是已给定的无避险响应轨迹：不要求证明驾驶员不能踩制动或无法改变碰撞，也不把模型未加入干预等同于人员没有干预能力。驾驶员避险属于各分支后续C/FTTI评价；这里只证明既定失效、初态和未缓解轨迹形成同一危险事件。其他场景按其真实依赖判断。若有驾驶员动作、位置相关使能、不同目标或车内乘员伤害依赖，不复用。没有提到驾驶员不等于已证明无关。
请求比较时返回 dependency_metadata={complete:false,driver_noninterference:{enablement:{independent:布尔值,evidence_refs:[精确ID],reason:一句依据或具体障碍},trigger:{...},interaction:{...},hazard:{...}}。complete=false仅表示没有声明全局依赖完备，不否定四项局部证明。可使用已验证且限于当前世界的条件性引用；结论仍为条件性。任一项不能证明则该项independent=false，evidence_refs允许为空，按当前身份正常完成因果判断；不要为复用编造事实。未请求比较时不分析或输出复用证明。"""


class ScenarioBatchSizeError(ValueError):
    """A scenario request cannot fit within the configured prompt budget."""


class ScenarioAdaptiveBatchError(RuntimeError):
    """An adaptive batch cannot be split further after runtime pressure."""


class ScenarioSchemaContractError(ValueError):
    """A Scenario assessment violates the declared item-level JSON contract."""


class ScenarioCoverageContractError(ValueError):
    """A Scenario response omits or adds identities relative to its request."""


class ScenarioFactConsistencyError(ValueError):
    """Authoritative structured Scenario facts are internally inconsistent."""


def _scenario_json(
    malfunction: MalfunctionCandidate,
    scenario: ScenarioCandidate,
    project_registry: FactRegistry | None = None,
    causal_evidence_budget: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET,
) -> str:
    registry = build_fact_registry(malfunction, scenario, project_registry)
    selection = CausalEvidenceSelector(
        max_evidence=causal_evidence_budget,
    ).select(registry)
    selected_facts = {}
    for ref in selection.selected_refs:
        if ref.startswith("SCN."):
            key = ref[4:]
        elif ref.startswith("DERIVED."):
            key = ref[8:]
        else:
            continue
        if key in scenario.facts and scenario.facts[key] not in (None, ""):
            selected_facts[key] = scenario.facts[key]
    comparison = scenario.analysis_instance.get("driver_reuse_comparison")
    if comparison:
        # Physical evidence consumes the ordinary top-k selection. Supply the
        # existing operating/driver prerequisites only for the requested proof;
        # do not infer enablement from an omitted driver dependency.
        prerequisite_refs = {"SCN.driver_control_policy", "SCN.operating_mode", "SCN.vehicle_state"}
        prerequisites = [record for record in registry.records
            if (record.evidence_ref in prerequisite_refs
                or record.metadata.get("parameter") == "DRIVER_IN_VEHICLE")
            and record.approval_status.value == "FINALIZED"][:causal_evidence_budget]
        comparison = {**comparison, "task": DRIVER_COMPARISON_TASK,
            "trajectory_scope": "UNMITIGATED" if scenario.analysis_instance.get("physical_hypothesis") else "CURRENT_SCENARIO",
            "prerequisite_evidence": {
            record.evidence_ref: {"value": record.value, "kind": record.kind.value,
                "source_excerpts": list(dict.fromkeys(source.excerpt for source in record.source_refs))}
            for record in prerequisites}}
    return json.dumps(
        {
            "scenario_id": scenario.scenario_id,
            "source_scenario_id": scenario.source_scenario_id,
            "atomic_variant": scenario.atomic_variant,
            "semantic_fingerprint": scenario.semantic_fingerprint,
            # Compatibility field: this is only the selected causal subset,
            # never the complete Scenario facts map.
            "facts": selected_facts,
            "driver_configuration_for_reuse": {
                field: scenario.facts[field]
                for field in ("driver_position", "allowed_driver_position", "driver_in_vehicle")
                if field in scenario.facts
            },
            **({"driver_reuse_comparison": comparison} if comparison else {}),
            "source_operating_conditions": scenario.facts.get("source_operating_conditions", {}),
            "selected_method_conditions": {
                dimension: binding.get("atom_label") or binding.get("method_value", "")
                for dimension, binding in scenario.facts.get("method_scenario_dimensions", {}).items()
                if isinstance(binding, dict) and binding.get("resolution_status") == "RESOLVED"
            },
            "physical_interaction": {
                "target_id": scenario.facts.get("object_id", scenario.scenario_id + ":target"),
                "impact_time_id": scenario.facts.get("impact_time_id", ""),
                "conditions": {key: scenario.facts[key] for key in (
                    "object_position", "ego_longitudinal_direction", "object_longitudinal_direction",
                    "ego_speed_kph", "object_speed_kph", "relative_distance_m", "fault_motion_model",
                    "ego_impact_speed_kph", "relative_speed_kph", "ttc_s", "first_contact_time_s",
                    "lateral_clearance_m", "turn_radius_m", "ego_lateral_direction", "slope_percent", "residual_resistance_mps2", "motion_model_outcome",
                    "object_offset_x_m", "object_offset_y_m", "contact_half_length_m", "contact_half_width_m", "selected_source_condition",
                ) if key in scenario.facts},
                "branches": [{key: value for key, value in branch.items() if not key.startswith("_")}
                             for branch in scenario.facts.get("collision_branches", [])],
                "condition_basis": scenario.analysis_instance.get("physical_input_production", {}).get("rule_id", ""),
                "evidence_refs": {key: next(ref for ref in ("DERIVED." + key, "SCN." + key)
                    if registry.resolve_record(ref) is not None)
                    for key in ("ego_speed_kph", "object_speed_kph", "relative_distance_m", "fault_motion_model",
                                "ego_impact_speed_kph", "relative_speed_kph", "ttc_s", "motion_model_outcome")
                    if any(registry.resolve_record(prefix + key) is not None for prefix in ("DERIVED.", "SCN."))},
                "derived_physics_evidence": {"DERIVED." + key: {"value": registry.resolve_record("DERIVED." + key).value,
                    "kind": registry.resolve_record("DERIVED." + key).kind.value}
                    for key in ("ego_impact_speed_kph", "relative_speed_kph", "closing_speed_kph", "ttc_s", "first_contact_time_s", "motion_model_outcome")
                    if registry.resolve_record("DERIVED." + key) is not None},
            },
            "conditional_physical_analysis": {**{key: scenario.analysis_instance.get("physical_hypothesis", {}).get(key)
                for key in ("hypothesis_id", "hypothesis", "sample_parameters", "calculation_nature",
                            "stationary_ego_counterfactual", "avoidance_validation")
                if key in scenario.analysis_instance.get("physical_hypothesis", {})},
                **({"parameter_relations_ref": "DERIVED.parameter_relations"}
                   if "DERIVED.parameter_relations" in selection.selected_refs else {}),
                **({"counterfactual_requirement": "本车静止也会与同一目标接触。若仍判定因果成立，"
                    "b_to_i或i_to_h必须引用DERIVED.stationary_ego_counterfactual，"
                    "结合实际轨迹说明新增危险；仅有接触不够。异常阻滞须有预期运动依据，"
                    "不能一律把静止当作健康状态。无增量依据则返回具体因果断点。"}
                   if "DERIVED.stationary_ego_counterfactual" in selection.selected_refs else {})},
            "causal_evidence_view": selection.compact_view,
            **({"shared_causal_evidence": scenario.analysis_instance["shared_causal_evidence"]}
               if scenario.analysis_instance.get("shared_causal_evidence") else {}),
            **(
                {
                    "analysis_scenario_assumptions": scenario.analysis_instance.get(
                        "assumptions", []
                    )
                }
                if scenario.analysis_instance else {}
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _prompt_parts(
    malfunction: MalfunctionCandidate,
) -> tuple[str, str]:
    allowed_dimensions = json.dumps(list(RISK_DIMENSION_VALUES), ensure_ascii=False)
    prefix = (
        f"Malfunction={malfunction.description}\n"
        f"FunctionalEffect={malfunction.functional_effect}\n"
        f"VehicleLevelHazard={malfunction.vehicle_level_hazard}\n"
        "Scenarios="
    )
    suffix = (
        "The causal_evidence_view is a compact selected subset of the complete EvidenceRegistry. "
        "Use exact evidence_ref keys shown in that view, driver_reuse_comparison.prerequisite_evidence, or physical_interaction.evidence_refs/derived_physics_evidence; full provenance, source excerpts, "
        "and downstream risk-method metadata are intentionally omitted from this prompt. "
        "Validated analytical Scenario atom facts shown in causal_evidence_view are bounded "
        "Method-instantiated operating conditions and may be cited for the current Scenario only. "
        "Records marked calculation_nature=CONDITIONAL are already instantiated and scope-checked inputs/results: "
        "assess causality under those stated conditions, without claiming source fact or engineering approval. "
        "Their lack of engineering approval alone is not a B_TO_I failure. Still check whether the malfunction "
        "actually explains this motion and whether the same trajectory creates the stated hazard. "
        "When analysis_scenario_assumptions are present, they are SCENARIO_DEFINED analysis conditions only; "
        "they remain unvalidated assumptions, are not causal evidence, and MUST NOT be cited for any causal hop or "
        "risk-dimension change. Do not convert them into project facts or combine template "
        "options into one atomic world. "
        "For m_to_b, finalized MF.description and MF.functional_effect are mandatory "
        "DIRECT_FACT anchors for the malfunction's defined behavior/effect when present. "
        "Do not treat missing Scenario trigger or applicability context as an M_TO_B failure; "
        "that context may affect applicability or the downstream B_TO_I transition. "
        "Never cite MF.vehicle_level_hazard as the m_to_b anchor. "
        "\n这里只评估当前batch列出的Scenario。必须逐一返回，不得遗漏，不得增加未提供Scenario。"
        "返回assessments数组，每项包含scenario_id、physically_feasible、functionally_relevant、"
        "causally_relevant、breakpoint、causal_chain、risk_dimension_changes、hazardous_event、"
        "physical_interaction、confidence、status。physical_interaction必须返回结构化目标/方位/关系/阶段主张；"
        "无当前外部目标主张时返回{\"target\":\"OTHER\"}。保留场景的hazardous_event必须结合当前失效和具体场景；Potential Harm"
        "由下游确定性 MethodContract 计算，不在此阶段生成。不得复制其他子系统模板。"
        "risk_dimension_changes必须为JSON object array；每项包含dimension、evidence_refs、reason。"
        "每项evidence_refs必须非空且引用当前场景已列出的证据；没有可引用证据时不输出该维度，"
        "不能用空引用补齐驾驶员干预等维度。整组允许为[]，这不改变已支持的因果结论。"
        "dimension必须严格从以下canonical allowed values选择："
        f"{allowed_dimensions}。无变化返回[]；"
        f"causal_chain只包含m_to_b、b_to_i、i_to_h；后续潜在伤害由下游确定性计算；basis_type只能是"
        f"{json.dumps([item.value for item in EvidenceBasisType])}；breakpoint只能是"
        f"{json.dumps([item.value for item in CausalBreakpoint])}。"
        "evidence_refs只能逐字引用当前Scenario已列出的证据ID；禁止自造ref或引用JSON路径。"
        "禁止使用缩写或别名（例如S/E/C），除非该字符串本身位于上述canonical allowed values中。"
        "confidence必须为0.0到1.0范围内的JSON number；禁止使用high、medium、low等文字等级，"
        "禁止字符串数字、百分数、null或缺失。"
        "不要返回rationale；可读说明由Python根据已验证的结构化因果字段确定性生成。"
        "不要在输出中重复shared_causal_evidence或输入物理方案；复用统计由程序计算。"
    )
    return prefix, suffix


def build_scenario_user_prompt(
    malfunction: MalfunctionCandidate,
    scenarios: list[ScenarioCandidate],
    project_registry: FactRegistry | None = None,
    causal_evidence_budget: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET,
) -> str:
    prefix, suffix = _prompt_parts(malfunction)
    serialized = [json.loads(
        _scenario_json(
            malfunction, item, project_registry, causal_evidence_budget,
        )) for item in scenarios]
    shared = {}
    if len(serialized) == 2:
        # Share exact input values, including the often much larger physical
        # hypothesis. Keep differing objects, phases and driver inputs local;
        # this is transmission sharing, never a shared causal conclusion.
        for key in ("source_operating_conditions", "selected_method_conditions", "facts",
                    "physical_interaction", "conditional_physical_analysis"):
            left, right = (item.get(key, {}) for item in serialized)
            common = {field: value for field, value in left.items() if field in right
                      and json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                      == json.dumps(right[field], sort_keys=True, ensure_ascii=False, separators=(",", ":"))}
            if common:
                shared[key] = common
                for item in serialized:
                    item[key] = {field: value for field, value in item[key].items() if field not in common}
        lines = [item["causal_evidence_view"].splitlines() for item in serialized]
        common_lines = [line for line in lines[0] if line in set(lines[1])]
        shared["causal_evidence_view"] = "\n".join(common_lines)
        for item, individual in zip(serialized, lines):
            item["causal_evidence_view"] = "\n".join(line for line in individual if line not in common_lines)
    payload = json.dumps({"shared_context": shared,
                         "shared_context_semantics": (
                             "shared_context为每个成员共有的原始输入；字典字段按键与成员同名字段合并，"
                             "causal_evidence_view按行合并，还原后逐成员独立判断。"
                             "成员的对象、阶段、驾驶员身份和不同参数仍各自有效；共有输入不代表共有结论。"),
                         "scenarios": serialized}, ensure_ascii=False,
                         separators=(",", ":")) if shared else json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
    return prefix + payload + suffix


def estimate_scenario_prompt_chars(
    malfunction: MalfunctionCandidate,
    scenarios: list[ScenarioCandidate],
    *,
    system_prompt: str,
    project_registry: FactRegistry | None = None,
    causal_evidence_budget: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET,
) -> int:
    return len(system_prompt) + len(build_scenario_user_prompt(
        malfunction, scenarios, project_registry, causal_evidence_budget,
    ))


def build_scenario_batches(
    malfunction: MalfunctionCandidate,
    scenarios: list[ScenarioCandidate],
    *,
    max_chars: int,
    max_items: int | None = None,
    system_prompt: str = "",
    project_registry: FactRegistry | None = None,
    causal_evidence_budget: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET,
) -> list[list[ScenarioCandidate]]:
    """Stable target-budget batching; every input appears exactly once.

    ``max_chars`` limits batches containing more than one Scenario.  A single
    Scenario cannot be split without changing its declared analysis instance,
    so it is retained as a singleton for the provider to evaluate rather than
    being silently omitted by a local batching heuristic.
    """
    if max_chars <= 0:
        raise ValueError("scenario batch max_chars必须大于0")
    if max_items is not None and max_items <= 0:
        raise ValueError("scenario batch max_items必须大于0")
    prefix, suffix = _prompt_parts(malfunction)
    fixed_chars = len(system_prompt) + len(prefix) + len(suffix) + 2
    batches: list[list[ScenarioCandidate]] = []
    current: list[ScenarioCandidate] = []
    current_payload_chars = 0
    for scenario in scenarios:
        serialized_chars = len(_scenario_json(
            malfunction, scenario, project_registry, causal_evidence_budget,
        ))
        single_chars = fixed_chars + serialized_chars
        if single_chars > max_chars:
            if current:
                batches.append(current)
                current = []
                current_payload_chars = 0
            batches.append([scenario])
            continue
        next_payload_chars = current_payload_chars + serialized_chars + (1 if current else 0)
        exceeds_chars = estimate_scenario_prompt_chars(malfunction, [*current, scenario],
            system_prompt=system_prompt, project_registry=project_registry,
            causal_evidence_budget=causal_evidence_budget) > max_chars
        exceeds_items = max_items is not None and len(current) >= max_items
        if current and (exceeds_chars or exceeds_items):
            batches.append(current)
            current = []
            current_payload_chars = 0
        current.append(scenario)
        current_payload_chars += serialized_chars + (1 if len(current) > 1 else 0)
    if current:
        batches.append(current)
    return batches


def select_scenario_causal_evidence(
    malfunction: MalfunctionCandidate,
    scenario: ScenarioCandidate,
    project_registry: FactRegistry | None = None,
    causal_evidence_budget: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET,
) -> CausalEvidenceSelection:
    """Return the same deterministic selection used by prompt serialization."""

    registry = build_fact_registry(malfunction, scenario, project_registry)
    return CausalEvidenceSelector(max_evidence=causal_evidence_budget).select(registry)
