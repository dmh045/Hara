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
                   if "DERIVED.parameter_relations" in selection.selected_refs else {})},
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
        "Use only exact evidence_ref keys shown in that view; full provenance, source excerpts, "
        "and downstream risk-method metadata are intentionally omitted from this prompt. "
        "Validated analytical Scenario atom facts shown in causal_evidence_view are bounded "
        "Method-instantiated operating conditions and may be cited for the current Scenario only. "
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
        "evidence_refs只能逐字引用当前Scenario对象内fact_registry已给出的key；禁止自造ref。"
        "禁止使用缩写或别名（例如S/E/C），除非该字符串本身位于上述canonical allowed values中。"
        "confidence必须为0.0到1.0范围内的JSON number；禁止使用high、medium、low等文字等级，"
        "禁止字符串数字、百分数、null或缺失。"
        "不要返回rationale；可读说明由Python根据已验证的结构化因果字段确定性生成。"
        "仅在四项驾驶员无关性都有明确引用时附加dependency_metadata；否则省略该可选字段。"
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
        for key in ("source_operating_conditions", "selected_method_conditions", "facts"):
            left, right = (item.get(key, {}) for item in serialized)
            common = {field: value for field, value in left.items() if field in right and right[field] == value}
            if common:
                shared[key] = common
                for item in serialized:
                    item[key] = {field: value for field, value in item[key].items() if field not in common}
        lines = [item["causal_evidence_view"].splitlines() for item in serialized]
        common_lines = [line for line in lines[0] if line in set(lines[1])]
        shared["causal_evidence_view"] = "\n".join(common_lines)
        for item, individual in zip(serialized, lines):
            item["causal_evidence_view"] = "\n".join(line for line in individual if line not in common_lines)
    payload = json.dumps({"shared_context": shared, "scenarios": serialized}, ensure_ascii=False,
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
