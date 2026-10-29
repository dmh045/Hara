from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import re
from typing import Any, Mapping

from hara_agent.contracts import MethodContract
from hara_agent.models import ReviewStatus
from hara_agent.workflow.state import HARAState

from .engineering_text_mapper import EngineeringReportTextMapper
from .quantity_format import boundary_notes
from .report_schema import ReportSchema
from .view_model import (
    AuditReferenceView, FunctionView, GeneratedChildView, HARAReportRowView, HARAReportViewModel,
    MalfunctionView, MethodBasisView, SafetyGoalView, ScenarioDetailView, SummaryView,
)
from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService


def _value(evidence: Any) -> str:
    value = getattr(evidence, "value", evidence)
    return str(getattr(value, "value", value) or "")


def _status(evidence: Any) -> str:
    value = getattr(evidence, "status", "")
    return str(getattr(value, "value", value))


_CALCULATED = "已计算"
_TRIAL = "待审试算"
_UNCALCULATED = "未计算"
_TRIAL_MARKERS = (
    "CONDITIONAL_TRIAL", "CANDIDATE_DEFAULT", "CANDIDATE_POLICY",
    "TRIAL_PENDING", "PENDING_ANALYSIS_ASSUMPTION", "待审试算", "条件性",
    "CALCULATED_PENDING_REVIEW",
)


class HARAReportProjectionService:
    """Project committed runtime facts into reviewer-facing report values."""

    def __init__(self, schema: ReportSchema):
        self.schema = schema
        self.text = EngineeringReportTextMapper(schema.presentation.get("language", {}))

    @staticmethod
    def _run_scope(
        state: HARAState, trace_rows: list[Mapping[str, Any]],
        complete_risk_chain_count: int,
    ) -> dict[str, Any]:
        def event(name: str) -> Mapping[str, Any]:
            return next((
                item for item in reversed(state.audit_trail)
                if item.get("event") == name
            ), {})

        configured = event("bounded_sample_configured")
        replay = event("p7_fixed_population_replay")
        offline_sample = event("offline_source_context_replay")
        partial_recovery = event("real_partial_response_recovery")
        bounded = bool(configured)
        limits = configured.get("scope", {}) if bounded else {}
        inventory = state.item_definition.get("bounded_sample_inventory", {})
        functions = inventory.get("functions", {}) if isinstance(inventory, Mapping) else {}
        malfunctions = inventory.get("malfunctions", {}) if isinstance(inventory, Mapping) else {}
        preview = event("bounded_sample_scope_preview")
        prepared = event("scenario_candidates_prepared")
        feasibility = event("scenario_feasibility_summary")
        latest_assessments = {
            item.get("malfunction_id") or index: item
            for index, item in enumerate(state.audit_trail)
            if item.get("event") == "scenario_feasibility_assessed"
        }
        branches = [
            item.get("analytical_scenario_instantiation", {}).get("driver_configuration", {})
            for item in latest_assessments.values()
        ]
        before_branch = (
            sum(int(item["pre_branch_count"]) for item in branches)
            if branches and all("pre_branch_count" in item for item in branches)
            else None
        )
        after_branch = (
            sum(int(item["post_branch_count"]) for item in branches)
            if branches and all("post_branch_count" in item for item in branches)
            else None  # Causal identities cannot substitute for pre-physics method candidates.
        )
        method_counts = []
        for item in latest_assessments.values():
            generation = item.get("analytical_scenario_instantiation", {})
            selected = generation.get("bounded_generated_selection", {})
            method_counts.append(len(selected["selected_ids"]) if "selected_ids" in selected else
                                 generation.get("driver_configuration", {}).get("post_branch_count"))
        method_pairs = (sum(method_counts) if method_counts and all(n is not None for n in method_counts)
                        else preview.get("scenario_pairs_after_method_instantiation"))
        guidewords = state.guideword_assessments
        guideword_applicable = sum(
            item.get("status") == "FINALIZED" and item.get("applicable") is True
            for item in guidewords if isinstance(item, Mapping)
        )
        guideword_filtered = sum(
            item.get("status") == "FINALIZED" and item.get("applicable") is False
            for item in guidewords if isinstance(item, Mapping)
        )
        retained = sum(
            str(item.get("risk_eligibility", {}).get("status", "")) == "ELIGIBLE"
            for item in trace_rows
        )
        excluded = sum(
            str(item.get("risk_eligibility", {}).get("status", "")).startswith("INELIGIBLE")
            for item in trace_rows
        )
        scope = {
            "run_scope": "OFFLINE SCRIPTED SOURCE SAMPLE" if offline_sample else "OFFLINE FIXED-POPULATION REPLAY" if replay else "BOUNDED ENGINEERING SAMPLE" if bounded else "FULL PROJECT HARA",
            "report_class": (
                "OFFLINE INTEGRATION SAMPLE" if offline_sample else
                str(configured.get("report_class", "ENGINEERING_SAMPLE_ONLY")).replace("_", " ")
                if bounded else "REVISED ENGINEERING DRAFT" if replay else "PROJECT HARA"
            ),
            "full_project_population": (
                "NOT MEASURED: OFFLINE SAMPLE ONLY" if offline_sample else
                f"{len(state.risk_results)} retained historical records; no new full causal validation" if replay else "NOT MEASURED IN THIS RUN"
                if bounded or not trace_rows else
                f"{len(state.risk_results)} risk-eligible records in this full-scope run"
            ),
            "scope_note": (
                f"来源样本：{len(state.malfunctions)}/{offline_sample.get('available_malfunction_count', '未记录')} 个失效；模拟模型响应；真实 Provider 调用 0。评级仅用于验证程序链路，未经真实因果验证。" if offline_sample else
                f"Offline replay of {replay.get('source_run_id')}; 0 new Provider calls; historical results preserved." if replay else "This workbook is not a full-project HARA population report."
                if bounded else ""
            ),
            "hara_row_meaning": (
                "Offline scripted candidate results; not live validated hazardous events" if offline_sample else
                "Retained/risk-eligible rows in this bounded engineering sample"
                if bounded else "Retained/risk-eligible rows in this run"
            ),
            "provider_attempt_budget": limits.get("provider_attempt_limit"),
            "sample_function_limit": limits.get("function_limit"),
            "sample_function_ids": limits.get("function_ids", []),
            "sample_malfunction_limit": limits.get("malfunction_limit"),
            "sample_malfunction_ids": limits.get("malfunction_ids", []),
            "sample_parent_scenario_limit": limits.get("parent_scenario_limit"),
            "sample_parent_scenario_ids": limits.get("parent_scenario_ids", []),
            "scenario_pair_limit": limits.get("scenario_pair_limit"),
            "extracted_function_count": functions.get(
                "available_count", len(state.functions) if not bounded else None
            ),
            "selected_function_count": functions.get("selected_count", len(state.functions)),
            "omitted_function_count": functions.get("omitted_count", 0 if not bounded else None),
            "guideword_assessed_count": len(guidewords),
            "guideword_applicable_count": guideword_applicable,
            "guideword_filtered_count": guideword_filtered,
            "guideword_pending_count": len(guidewords) - guideword_applicable - guideword_filtered,
            "generated_malfunction_count_in_selected_function_scope": malfunctions.get(
                "available_count", len(state.malfunctions) if not bounded else None
            ),
            "selected_malfunction_count": malfunctions.get("selected_count", len(state.malfunctions)),
            "omitted_malfunction_count": malfunctions.get("omitted_count", 0 if not bounded else None),
            "available_parent_scenario_count_in_selected_scope": preview.get(
                "available_parent_scenario_count",
                prepared.get("candidate_count") if not bounded else None,
            ),
            "selected_parent_scenario_count": preview.get(
                "parent_scenario_count", prepared.get("candidate_count") if not bounded else None,
            ),
            "omitted_parent_scenario_count": (
                preview["available_parent_scenario_count"] - preview["parent_scenario_count"]
                if "available_parent_scenario_count" in preview and "parent_scenario_count" in preview
                else 0 if not bounded and "candidate_count" in prepared else None
            ),
            "method_instantiated_scenario_pair_count": method_pairs,
            "causal_candidate_count": len(trace_rows) if trace_rows else feasibility.get("scenario_pair_count"),
            "analytical_child_count_before_driver_branch": before_branch,
            "analytical_child_count_after_driver_branch": after_branch,
            "causal_retained_count": retained if trace_rows else None,
            "causal_excluded_count": excluded if trace_rows else None,
            "causal_pending_count": len(trace_rows) - retained - excluded if trace_rows else None,
            "risk_scored_count": sum(bool(item.get("risk_scoring_invoked")) for item in trace_rows),
            "risk_not_invoked_count": sum(not bool(item.get("risk_scoring_invoked")) for item in trace_rows),
            "complete_risk_chain_count": complete_risk_chain_count,
        }
        if partial_recovery:
            scope["run_scope"] = "PARTIAL REAL RESPONSE RECOVERY"
            scope["scope_note"] = (
                f"真实返回部分恢复：{partial_recovery['available_candidates']} 个生成身份中，"
                f"{partial_recovery['validated_response_identities']} 个已通过因果校验，"
                f"{partial_recovery['remaining_unvalidated']} 个尚未验证。"
                "预算已耗尽，补充风险事实模型阶段未执行；仅用已有来源和条件性输入计算。"
                "本报告不代表完整真实端到端验证或全项目覆盖。"
            )
        return scope

    @classmethod
    def _generated_children(
        cls,
        trace_rows: list[Mapping[str, Any]],
        generated_scenarios: list[Mapping[str, Any]],
        malfunctions: Mapping[str, Any],
        causal_by_pair: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    ) -> tuple[GeneratedChildView, ...]:
        candidates = {
            str(item.get("scenario_id", "")): item
            for item in generated_scenarios
            if isinstance(item, Mapping)
        }
        children: list[GeneratedChildView] = []
        for row in trace_rows:
            scenario_id = str(row.get("scenario_id", ""))
            candidate = candidates.get(scenario_id, {})
            instance = candidate.get("analysis_instance", {})
            instance = instance if isinstance(instance, Mapping) else {}
            branch = instance.get("driver_configuration_branch", {})
            branch = branch if isinstance(branch, Mapping) else {}
            context = candidate.get("context_resolution", {})
            context = context if isinstance(context, Mapping) else {}
            synthesis = context.get("scenario_synthesis", {})
            synthesis = synthesis if isinstance(synthesis, Mapping) else {}
            eligibility = row.get("risk_eligibility", {})
            eligibility = eligibility if isinstance(eligibility, Mapping) else {}
            eligibility_status = str(eligibility.get("status", ""))
            disposition = (
                "RETAINED_FOR_RISK" if eligibility_status == "ELIGIBLE"
                else "EXCLUDED_FROM_RISK" if eligibility_status.startswith("INELIGIBLE")
                else "PENDING_CAUSAL"
            )
            source = candidate.get("facts", {}).get("source_operating_conditions", {})
            source_text = "；".join(
                f"{label}：{source[key]}" for key, label in (
                    ("operating_scenario", "来源场所"), ("road_type", "来源道路/车位"),
                    ("source_slope_direction", "来源坡向"), ("source_slope_pct", "坡度百分比"))
                if source.get(key) not in (None, ""))
            causal = (causal_by_pair or {}).get((str(row.get("malfunction_id", "")), scenario_id), {})
            children.append(GeneratedChildView(
                scenario_id=scenario_id,
                parent_scenario_id=str(
                    candidate.get("source_scenario_id") or instance.get("parent_scenario_id", "")
                ),
                malfunction_id=str(row.get("malfunction_id", "")),
                driver_branch=str(branch.get("driver_position", "")),
                semantic_group_id=str(instance.get("semantic_group_id") or synthesis.get("semantic_group_id", "")),
                causal_disposition=disposition,
                risk_scoring_invoked=bool(row.get("risk_scoring_invoked")),
                selected_atom_ids=cls._selected_atom_ids(candidate),
                function_id=str(malfunctions.get(str(row.get("malfunction_id", "")), {}).get("function_id", "")),
                source_conditions=source_text,
                disposition_reason=str(causal.get("rationale") or "") or
                    "; ".join(str(reason) for reason in eligibility.get("reason_codes", [])),
            ))
        return tuple(children)

    @staticmethod
    def _clarifications(risk: Any) -> str:
        ids: list[str] = []
        if _status(risk.severity) != "FINALIZED":
            ids.append("EC-03")
        if _status(risk.exposure) != "FINALIZED":
            ids.append("EC-01")
        if _status(risk.controllability) != "FINALIZED":
            ids.append("EC-02")
        return "; ".join(ids)

    @staticmethod
    def _causal_status_by_pair(
        causal_trace: Mapping[str, Any] | None,
    ) -> dict[tuple[str, str], str]:
        statuses: dict[tuple[str, str], str] = {}
        for audit in (causal_trace or {}).get("audits", []):
            if not isinstance(audit, Mapping):
                continue
            for item in audit.get("item_salvage_audit", []):
                if not isinstance(item, Mapping):
                    continue
                parsed = item.get("parsed_assessment", {})
                if not isinstance(parsed, Mapping):
                    continue
                pair = (
                    str(parsed.get("malfunction_id", "")),
                    str(parsed.get("scenario_id", "")),
                )
                if not all(pair):
                    continue
                statuses[pair] = (
                    "METHOD_VALID — CAUSAL_REVALIDATED"
                    if bool(parsed.get("final_retain"))
                    else "CAUSAL_GAP — EXCLUDED_FROM_RISK"
                )
        return statuses

    @classmethod
    def _assessment_status(
        cls, *, risk: Any, scenario: Any, trace: Mapping[str, Any],
        causal_status: str, trace_provided: bool,
    ) -> str:
        if bool(trace.get("risk_scoring_invoked")):
            return "ELIGIBLE — RISK SCORING INVOKED"
        if causal_status:
            return causal_status
        instance = getattr(scenario, "analysis_instance", {}) or {}
        if str(instance.get("validation_status", "")).upper() == "VALIDATED":
            return "METHOD_VALID — CAUSAL_REVALIDATION_REQUIRED"
        finalized = all(
            _status(getattr(risk, field)) == "FINALIZED"
            for field in ("severity", "exposure", "controllability", "asil")
        )
        return (
            "ELIGIBLE — RISK SCORING INVOKED"
            if finalized and not trace_provided
            else "PENDING — RISK SCORING NOT EXECUTED"
        )

    @staticmethod
    def _driver_group_identity(scenario: Any) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        instance = getattr(scenario, "analysis_instance", {}) or {}
        branch = instance.get("driver_configuration_branch", {}) if isinstance(instance, Mapping) else {}
        position = str(
            facts.get("driver_position") or facts.get("allowed_driver_position")
            or (branch.get("driver_position", "") if isinstance(branch, Mapping) else "")
        )
        if position:
            return position
        in_vehicle = facts.get("driver_in_vehicle")
        return f"driver_in_vehicle={in_vehicle}" if isinstance(in_vehicle, bool) else ""

    @staticmethod
    def _score_state(
        evidence: Any, field_trace: Mapping[str, Any], *, scoring_invoked: bool,
    ) -> str:
        value = getattr(evidence, "value", None)
        if not scoring_invoked or value is None or value == "":
            return _UNCALCULATED
        if _status(evidence) == "FINALIZED":
            return _CALCULATED
        marker_text = " ".join((
            str(getattr(evidence, "review_reason", "")),
            str(field_trace.get("status", "")),
            str(field_trace.get("calculation_status", "")),
            str(field_trace.get("trial_status", "")),
        )).upper()
        return _TRIAL if any(marker in marker_text for marker in _TRIAL_MARKERS) else _UNCALCULATED

    @classmethod
    def _score_states(
        cls, risk: Any, trace: Mapping[str, Any], *, trace_provided: bool,
    ) -> dict[str, str]:
        invoked = bool(trace.get("risk_scoring_invoked")) if trace_provided else True
        states = {
            field: cls._score_state(
                getattr(risk, field),
                trace.get(field, {}) if isinstance(trace.get(field), Mapping) else {},
                scoring_invoked=invoked,
            )
            for field in ("severity", "exposure", "controllability")
        }
        asil = getattr(risk, "asil")
        asil_trace = trace.get("asil", {}) if isinstance(trace.get("asil"), Mapping) else {}
        states["asil"] = cls._score_state(
            asil, asil_trace, scoring_invoked=invoked,
        )
        if (
            states["asil"] == _UNCALCULATED
            and invoked and getattr(asil, "value", None) not in (None, "")
            and all(states[field] in {_CALCULATED, _TRIAL} for field in (
                "severity", "exposure", "controllability",
            ))
            and any(states[field] == _TRIAL for field in (
                "severity", "exposure", "controllability",
            ))
        ):
            states["asil"] = _TRIAL
        return states

    @staticmethod
    def _validate_scored_trace(risk: Any, trace: Mapping[str, Any]) -> None:
        """Refuse a scored row whose field results differ from committed state."""

        pair = f"{risk.malfunction_id}/{risk.scenario_id}"
        for field in ("severity", "exposure", "controllability", "asil"):
            field_trace = trace.get(field)
            if (
                not isinstance(field_trace, Mapping)
                or "status" not in field_trace
                or "result" not in field_trace
            ):
                raise ValueError(f"Risk execution trace lacks {field} status/result: {pair}")
            evidence = getattr(risk, field)
            trace_status = str(field_trace["status"] or "")
            if not trace_status:
                raise ValueError(f"Risk execution trace has empty {field} status: {pair}")
            trace_result = field_trace["result"]
            trace_value = str(getattr(trace_result, "value", trace_result) or "")
            if (
                trace_value != _value(evidence)
                or (trace_status == "FINALIZED") != (_status(evidence) == "FINALIZED")
            ):
                raise ValueError(f"Risk execution trace disagrees with {field} state: {pair}")

    def _display_score(self, evidence: Any, state: str) -> str:
        if state == _UNCALCULATED:
            return self.text.pending_value(evidence)
        value = _value(evidence)
        return value or "待评定"

    def _score_rationale(
        self, field: str, evidence: Any, trace: Mapping[str, Any], state: str,
    ) -> str:
        labels = {
            "severity": "S", "exposure": "E",
            "controllability": "C", "asil": "ASIL",
        }
        if field == "exposure" and state == _TRIAL:
            return self.text.exposure_rationale(evidence, trace, calculated=True)
        if state == _UNCALCULATED and _status(evidence) == "FINALIZED":
            return f"当前执行记录未证明 {labels[field]} 已完成评分，暂不展示旧值。"
        mapper = {
            "severity": self.text.severity_rationale,
            "exposure": self.text.exposure_rationale,
            "controllability": self.text.controllability_rationale,
            "asil": self.text.asil_rationale,
        }
        return mapper[field](evidence, trace)

    @staticmethod
    def _display_trace(field: str, trace: Mapping, branch: Mapping | None = None) -> dict:
        """Join saved calculation evidence for display, without recomputing it."""
        result = dict(trace.get(field, {}))
        context = trace.get("hazardous_event_risk_context", {})
        if branch:
            result = dict(branch.get("scores", {}).get(field, {}))
            context = branch.get("risk_context", {})
            if field == "exposure":
                source = trace.get(field, {})
                source_atoms = source.get("atom_bindings", source.get("atom_details", []))
                if {a.get("atom_id") for a in source_atoms} == {a.get("atom_id") for a in result.get("atom_details", [])}:
                    result["scenario_terms"] = source.get("scenario_terms", {})
            result["result"] = branch.get({"severity": "S", "exposure": "E", "controllability": "C", "asil": "ASIL"}[field])
            result["branch_grades"] = {k: branch.get(k) for k in ("S", "E", "C", "ASIL")}
        result["physical_inputs"] = {key: value.get("value") for key, value in context.items()
            if isinstance(value, Mapping) and value.get("status") == "AVAILABLE"}
        if branch:
            result["physical_inputs"].update({key: branch[key] for key in ("relative_speed_kph", "ttc_s")
                if branch.get(key) is not None})
        return result

    @staticmethod
    def _common_mapping(values: list[Mapping]) -> dict:
        """Intersect raw structured values; rounded labels never prove equality."""
        common = {}
        for key, value in values[0].items():
            if not all(key in other for other in values[1:]):
                continue
            items = [other[key] for other in values]
            if all(isinstance(item, Mapping) for item in items):
                nested = HARAReportProjectionService._common_mapping(items)
                if nested:
                    common[key] = nested
            elif all(item == value for item in items):
                common[key] = deepcopy(value)
        return common

    def _group_scenario(self, siblings: list[dict]) -> tuple[str, str]:
        if len(siblings) == 1:
            return self.text.scenario(siblings[0]["scenario"])
        scenarios = [entry["scenario"] for entry in siblings]
        common = deepcopy(scenarios[0])
        for field in ("facts", "context_resolution", "analysis_instance"):
            setattr(common, field, self._common_mapping([getattr(s, field) or {} for s in scenarios]))
        # A source choice and its physical effect form one condition. Keeping
        # only their shared source_field would describe no actual member.
        selected = [s.facts.get("selected_source_condition") for s in scenarios]
        if any(value != selected[0] for value in selected[1:]):
            common.facts.pop("selected_source_condition", None)
        for field in ("operating_mode", "operating_scenario", "atomic_variant"):
            if len({str(getattr(s, field)) for s in scenarios}) > 1:
                setattr(common, field, "")
        operational, detail = self.text.scenario(common)
        if "ego_impact_speed_kph" not in common.facts:
            operational = operational.replace("；碰撞速度另行确定", "")
            detail = detail.replace("；碰撞速度另行确定", "")
        return operational, detail

    @staticmethod
    def _status_note(states: Mapping[str, str]) -> str:
        labels = {"severity": "S", "exposure": "E", "controllability": "C", "asil": "ASIL"}
        groups = defaultdict(list)
        for field, status in states.items():
            groups[status].append(labels[field])
        return "；".join("/".join(fields) + " " + status for status, fields in groups.items())

    @staticmethod
    def _normalized_hazardous_event(value: Any) -> str:
        return re.sub(r"[\W_]+", "", str(value or "").casefold())

    @staticmethod
    def _semantic_group_id(scenario: Any) -> str:
        instance = getattr(scenario, "analysis_instance", {}) or {}
        if isinstance(instance, Mapping) and instance.get("semantic_group_id"):
            return str(instance["semantic_group_id"])
        context = getattr(scenario, "context_resolution", {}) or {}
        synthesis = context.get("scenario_synthesis", {}) if isinstance(context, Mapping) else {}
        return str(synthesis.get("semantic_group_id", "")) if isinstance(synthesis, Mapping) else ""

    @staticmethod
    def _parent_scenario_id(scenario: Any) -> str:
        instance = getattr(scenario, "analysis_instance", {}) or {}
        return str(
            getattr(scenario, "source_scenario_id", "")
            or (instance.get("parent_scenario_id", "") if isinstance(instance, Mapping) else "")
            or getattr(scenario, "scenario_id", "")
        )

    @classmethod
    def _base_group_key(
        cls, *, risk: Any, scenario: Any, hazardous_event_id: str,
    ) -> tuple[str, str, str]:
        malfunction_id = str(risk.malfunction_id)
        semantic_group_id = cls._semantic_group_id(scenario)
        if semantic_group_id:
            return "semantic_group", malfunction_id, semantic_group_id
        if hazardous_event_id:
            return "hazardous_event", malfunction_id, hazardous_event_id
        parent_id = cls._parent_scenario_id(scenario)
        if parent_id and parent_id != str(scenario.scenario_id):
            return "parent_scenario", malfunction_id, parent_id
        return "scenario", malfunction_id, str(scenario.scenario_id)

    @classmethod
    def _representative_rank(cls, entry: Mapping[str, Any]) -> tuple[int, str]:
        scenario = entry["scenario"]
        context = getattr(scenario, "context_resolution", {}) or {}
        synthesis = context.get("scenario_synthesis", {}) if isinstance(context, Mapping) else {}
        label = str(synthesis.get("coverage_label", "") if isinstance(synthesis, Mapping) else "")
        if not label:
            label = str(getattr(scenario, "atomic_variant", "") or "").rsplit(":", 1)[-1]
        order = {"typical": 0, "representative": 0, "boundary": 1, "extreme": 2, "demanding": 2}
        return order.get(label.casefold(), 3), str(scenario.scenario_id)

    @staticmethod
    def _highest_known_tuple(siblings: list[dict], asil_order: list[str]) -> tuple:
        candidates = []
        for entry in siblings:
            risk = entry["risk"]
            if entry["trace"].get("risk_scoring_invoked") is False:
                continue
            branches = risk.branch_evaluations
            if branches:
                candidates.extend((entry, tuple(str(b.get(k) or "待评定") for k in ("S", "E", "C", "ASIL")), b)
                                  for b in branches if b.get("ASIL") in asil_order)
            elif _value(risk.asil) in asil_order:
                candidates.append((entry, tuple(_value(getattr(risk, f)) or "待评定"
                                  for f in ("severity", "exposure", "controllability", "asil")), {}))
        if not candidates:
            return None, []
        maximum = max(asil_order.index(values[-1]) for _, values, _ in candidates)
        highest = [(entry, values, branch) for entry, values, branch in candidates if asil_order.index(values[-1]) == maximum]
        # A tie does not make already calculated dimensions unknown. Keep one
        # actual branch tuple; the other tied tuples remain in scenario detail.
        return highest[0][1], highest

    def _member_display_tuple(self, entry: dict, asil_order: list[str], *, trace_provided: bool) -> tuple:
        """Select one actual tuple, keeping all other branches in the detail view."""
        risk, trace = entry["risk"], entry["trace"]
        states = self._score_states(risk, trace, trace_provided=trace_provided)
        values = tuple(self._display_score(getattr(risk, field), states[field])
                       for field in ("severity", "exposure", "controllability", "asil"))
        headline, highest = self._highest_known_tuple([entry], asil_order)
        branch = highest[0][2] if highest else {}
        if headline:
            values = headline
        elif risk.branch_evaluations and (not trace_provided or trace.get("risk_scoring_invoked")):
            # No known ASIL: retain one incomplete branch, never splice the
            # individually known S/E/C values of different branches together.
            branch = risk.branch_evaluations[0]
            values = tuple(str(branch.get(key) or "待评定") for key in ("S", "E", "C", "ASIL"))
        if values[-1] == "QM" and any(not b.get("ASIL") for b in risk.branch_evaluations):
            values = (*values[:3], "待评定")
        return values, branch

    def _group_grade_summary(self, siblings: list[dict], *, trace_provided: bool) -> tuple:
        """Retain distributions and bases in explanations, never in grade cells."""
        rationales, distributions = {}, {}
        for field, label in (("severity", "S"), ("exposure", "E"), ("controllability", "C"), ("asil", "ASIL")):
            counts, branch_counts, bases = Counter(), Counter(), Counter()
            for entry in siblings:
                states = self._score_states(entry["risk"], entry["trace"], trace_provided=trace_provided)
                counts[self._display_score(getattr(entry["risk"], field), states[field])] += 1
                bases[self._score_rationale(field, getattr(entry["risk"], field),
                    self._display_trace(field, entry["trace"]), states[field])] += 1
                if entry["trace"].get("risk_scoring_invoked") is not False:
                    branch_counts.update(str(b.get(label) or "待评定") for b in entry["risk"].branch_evaluations)
            observed = set(counts) | set(branch_counts)
            known = sorted(observed - {"待评定"})
            distributions[field] = {"members": dict(counts), "collision_branches": dict(branch_counts)}
            distribution = "、".join(f"{value}×{count}" for value, count in counts.most_common())
            if len(observed) == 1 and known:
                reason = f"组内 {len(siblings)} 个成员的 {label} 共同值为 {known[0]}；仅为该维度共同值，未形成完整风险结论。"
            elif "待评定" not in observed:
                reason = f"组内 {label} 分布：{distribution}；该维度均已有数值，等级随成员或碰撞分支条件不同；完整风险元组见场景明细。"
            elif known:
                reason = f"组内 {label} 分布：{distribution}；已计算等级保留，另有成员或碰撞分支尚无数值，具体缺口见场景明细。"
            else:
                reason = f"组内 {label} 分布：{distribution}；该维度尚无已计算等级，具体缺口见场景明细。"
            if branch_counts and set(branch_counts) != set(counts):
                reason += "碰撞分支分布：" + "、".join(f"{value}×{count}" for value, count in branch_counts.most_common()) + "。"
            if len(bases) == 1:
                basis = next(iter(bases))
            else:
                basis = "成员依据示例：" + "；".join(
                    f"{count} 个成员：{text}" for text, count in bases.most_common(2))
                basis += "各成员及分支的完整依据见风险追溯。"
            rationales[field + "_rationale"] = basis + "\n" + reason
        return rationales, distributions

    def _missing_asil_explanation(self, risk: Any, trace: Mapping, method: Any, *, detail: bool = True) -> str:
        """Explain incomplete tuples using the active matrix, without assigning C."""
        if _value(risk.asil):
            return ""
        missing = [label for field, label in (("severity", "S"), ("exposure", "E"), ("controllability", "C"))
                   if not _value(getattr(risk, field))]
        reasons = [self._score_rationale(field, getattr(risk, field),
                   trace.get(field, {}) if detail else {**trace.get(field, {}), "known_physical_inputs": {}}, _UNCALCULATED)
                   for field in ("severity", "exposure", "controllability") if not _value(getattr(risk, field))]
        text = "ASIL未决：" + ("/".join(missing) + "尚无等级。" if missing else "分支结果尚未闭合。")
        text += "；".join(dict.fromkeys(reasons))
        if not detail:
            return text
        tuples = risk.branch_evaluations or [{"S": _value(risk.severity), "E": _value(risk.exposure),
                                               "C": _value(risk.controllability), "ASIL": ""}]
        relations = []
        for branch in tuples:
            if branch.get("ASIL") or not branch.get("S") or not branch.get("E") or branch.get("C"):
                continue
            matches = [m for m in getattr(getattr(method, "asil", None), "mappings", ())
                       if m.severity == branch["S"] and m.exposure == branch["E"]]
            if matches:
                relations.append(f"{branch['S']}/{branch['E']}：" + "、".join(
                    f"{m.controllability}→{m.result}" for m in matches))
        if relations:
            text += " 活动矩阵条件关系：" + "；".join(dict.fromkeys(relations)) + "。未选择C，不作为已评定ASIL。"
        return text

    @staticmethod
    def _safe_state_display(goal: Any, catalog: Mapping, scenario_ids=None, *, scenarios=None) -> str:
        scope = set(goal.associated_scenario_ids if scenario_ids is None else scenario_ids)
        candidates = [c for c in catalog.get("safe_state_candidates", []) if c["scenario_id"] in scope]
        gaps = [g for g in catalog.get("safe_state_gaps", []) if g["scenario_id"] in scope]
        covered = {c["scenario_id"] for c in candidates}
        missing = len(scope - covered)
        def narrative(record, field):
            scenario = (scenarios or {}).get(record["scenario_id"])
            return EngineeringReportTextMapper.hazardous_event(record[field], scenario) if scenario is not None else record[field]
        reasons = list(dict.fromkeys(narrative(g, "reason").rstrip("。；") for g in gaps))
        # SG/group cells are a review summary. Keep complete examples, rather
        # than truncating a qualifier or repeating every member's narrative.
        # All original member explanations remain in safe_state_evidence.
        gap_text = "；".join(reasons)
        if len(reasons) > 2:
            examples = sorted(reasons, key=lambda reason: (len(reason), reason))[:2]
            gap_text = "缺口示例：" + "；".join(examples) + "。完整逐成员原因见报告追溯"
        if candidates:
            variants = defaultdict(list)
            for candidate in candidates:
                path = EngineeringReportTextMapper.safe_state_path(candidate)
                key = ("path", path[0]) if path else ("text", narrative(candidate, "description"))
                variants[key].append(candidate)
            lines = []
            for members in variants.values():
                count = len({c["scenario_id"] for c in members})
                path = EngineeringReportTextMapper.safe_state_path(members[0])
                text = (path[1] + "；目标、停止位置、净减速及附加要求分别见场景明细"
                        if path and count > 1 else narrative(members[0], "description"))
                lines.append(f"候选：{text}（适用{count}个成员）")
            return "\n".join(lines) + (
                f"\n另有{missing}个成员尚无适用方案：{gap_text or '见成员控制条件'}。" if missing else "")
        prior_scope = set(catalog.get("prior_safe_state", {}).get("scenario_ids", goal.associated_scenario_ids))
        contradicted = any(g.get("reason_code") not in {"NO_ANALYSIS", "NO_PROPOSAL"} for g in gaps)
        if (not catalog.get("safe_state_candidates") and not contradicted and scope <= prior_scope
                and goal.safe_state not in {"", "待确定", "待评定", "Pending"}):
            return goal.safe_state if goal.status.value == "FINALIZED" else "候选：" + goal.safe_state
        return "安全状态待确定：" + (gap_text or "当前成员未形成可用的控制方案")

    @staticmethod
    def _selected_atom_ids(scenario: Any) -> str:
        facts = scenario.get("facts", {}) if isinstance(scenario, Mapping) else getattr(scenario, "facts", {})
        instance = scenario.get("analysis_instance", {}) if isinstance(scenario, Mapping) else getattr(scenario, "analysis_instance", {})
        if isinstance(facts, Mapping) and "scenario_atom_ids" in facts:
            selected = facts["scenario_atom_ids"]
        else:
            selected = instance.get("selected_atoms", []) if isinstance(instance, Mapping) else []
        if isinstance(selected, Mapping):
            selected = [atom for atoms in selected.values() for atom in atoms]
        if not isinstance(selected, (list, tuple)):
            return ""
        return "; ".join(dict.fromkeys(str(item) for item in selected if str(item)))

    @staticmethod
    def _source_references(scenario: Any) -> str:
        references = []
        for source in getattr(scenario, "sources", ()) or ():
            source_type = str(getattr(source, "source_type", "") or "")
            source_id = str(getattr(source, "source_id", "") or "")
            location = str(getattr(source, "location", "") or "")
            identity = ":".join(item for item in (source_type, source_id) if item)
            references.append(f"{identity}@{location}" if location else identity)
        instance = getattr(scenario, "analysis_instance", {}) or {}
        if isinstance(instance, Mapping):
            references.extend(
                str(item) for item in instance.get("method_facts_used", [])
                if str(item)
            )
        facts = getattr(scenario, "facts", {}) or {}
        speed_context = facts.get("speed_context_resolution", {})
        if isinstance(speed_context, Mapping):
            for source in speed_context.get("source_refs", []):
                if not isinstance(source, Mapping):
                    continue
                identity = ":".join(
                    str(source.get(key, ""))
                    for key in ("source_type", "source_id")
                    if str(source.get(key, ""))
                )
                location = str(source.get("location", ""))
                references.append(f"{identity}@{location}" if location else identity)
        return "; ".join(dict.fromkeys(item for item in references if item))

    @staticmethod
    def _report_scenario(scenario: Any, context: Mapping[str, Any]) -> Any:
        projected = deepcopy(scenario)
        speed = dict(context.get("contextual_speed", {}))
        query = dict(context.get("structured_semantic_query", {}))
        coverage = dict(context.get("coverage_plan", {}))
        semantic_group_id = str(context.get("semantic_group_id", ""))

        facts = dict(getattr(projected, "facts", {}) or {})
        facts.pop("ego_speed_kph", None)
        facts["ego_speed_constraint"] = {
            "min_kph": speed.get("min_kph"),
            "max_kph": speed.get("max_kph"),
        }
        facts["speed_context_resolution"] = speed
        projected.facts = facts

        provenance = dict(getattr(projected, "fact_provenance", {}) or {})
        provenance.pop("ego_speed_kph", None)
        projected.fact_provenance = provenance

        instance = dict(getattr(projected, "analysis_instance", {}) or {})
        instance.update({
            "semantic_group_id": semantic_group_id,
            "parent_scenario_id": str(getattr(scenario, "scenario_id", "")),
            "structured_semantic_query": query,
            "coverage_plan": coverage,
        })
        projected.analysis_instance = instance

        resolution = dict(getattr(projected, "context_resolution", {}) or {})
        synthesis = dict(resolution.get("scenario_synthesis", {}) or {})
        synthesis.pop("coverage_label", None)
        synthesis.update({
            "semantic_group_id": semantic_group_id,
            "coverage_intents": [
                str(item.get("coverage_label", ""))
                for item in coverage.get("variant_intents", [])
                if isinstance(item, Mapping) and str(item.get("coverage_label", ""))
            ],
            "desired_variant_count": int(coverage.get("desired_variant_count", 1)),
            "coverage_status": "PLANNED_NOT_INSTANTIATED",
        })
        resolution["scenario_synthesis"] = synthesis
        projected.context_resolution = resolution
        return projected

    def project(
        self,
        state: HARAState,
        method: MethodContract,
        *,
        risk_trace: Mapping[str, Any] | None = None,
        causal_trace: Mapping[str, Any] | None = None,
        run_summary: Mapping[str, Any] | None = None,
        style_template_hash: str = "",
        risk_trace_reference: str = "",
        generated_scenarios: list[Mapping[str, Any]] | None = None,
        scenario_projection_contexts: Mapping[
            tuple[str, str, str], Mapping[str, Any]
        ] | None = None,
    ) -> HARAReportViewModel:
        trace_rows = (risk_trace or {}).get("assessments", [])
        if not isinstance(trace_rows, list):
            raise ValueError("Risk execution trace assessments must be a list")
        trace_by_pair = {}
        for item in trace_rows:
            if not isinstance(item, Mapping):
                raise ValueError("Risk execution trace assessment must be an object")
            pair = (str(item.get("malfunction_id", "")), str(item.get("scenario_id", "")))
            if not all(pair):
                raise ValueError("Risk execution trace assessment lacks risk pair ID")
            if pair in trace_by_pair:
                raise ValueError(f"Risk execution trace has duplicate risk pair: {pair}")
            trace_by_pair[pair] = item
        if risk_trace is not None:
            missing_pairs = {
                (str(risk.malfunction_id), str(risk.scenario_id))
                for risk in state.risk_results
            } - set(trace_by_pair)
            if missing_pairs:
                raise ValueError(
                    "Risk execution trace lacks committed risk pairs: "
                    + ", ".join(f"{malfunction}/{scenario}" for malfunction, scenario in sorted(missing_pairs))
                )
        causal_status_by_pair = self._causal_status_by_pair(causal_trace)
        malfunctions = {str(item.get("malfunction_id", "")): item for item in state.malfunctions}
        scenarios = {str(item.scenario_id): item for item in state.scenarios}
        functions = {str(item.get("function_id", "")): item for item in state.functions}

        entries: list[dict[str, Any]] = []
        base_events: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        group_source_counts: Counter[str] = Counter()
        causal_by_pair = {
            (str(item.get("malfunction_id", "")), str(item.get("scenario_id", ""))): item
            for item in state.item_definition.get("scenario_assessments", [])
        }
        from hara_agent.services.analysis.asil_service import MethodContractASILService
        asil_service = (MethodContractASILService(method)
                        if getattr(method, "engineering_rules_compiled", False) else None)
        asil_order = asil_service.result_order() if asil_service else []
        for risk in state.risk_results:
            scenario = scenarios.get(risk.scenario_id)
            if scenario is None:
                raise ValueError(f"Report projection scenario foreign key missing: {risk.scenario_id}")
            trace = trace_by_pair.get((risk.malfunction_id, risk.scenario_id), {})
            if bool(trace.get("risk_scoring_invoked")):
                self._validate_scored_trace(risk, trace)
            instance = getattr(scenario, "analysis_instance", {}) or {}
            hazardous_event_id = str(
                trace.get("hazardous_event_id", "")
                or (instance.get("hazardous_event_id", "") if isinstance(instance, Mapping) else "")
            )
            if scenario_projection_contexts is not None:
                context_key = (
                    str(risk.malfunction_id),
                    str(risk.scenario_id),
                    hazardous_event_id,
                )
                if context_key not in scenario_projection_contexts:
                    raise ValueError(
                        "Fresh report projection context missing for risk pair: "
                        f"{context_key}"
                    )
                scenario = self._report_scenario(
                    scenario, scenario_projection_contexts[context_key]
                )
            base_key = self._base_group_key(
                risk=risk, scenario=scenario, hazardous_event_id=hazardous_event_id,
            )
            normalized_event = self._normalized_hazardous_event(risk.hazardous_event)
            causal_identity = hazardous_event_id or normalized_event
            event_identity = f"{causal_identity}\0{normalized_event}"
            base_events[base_key].add(event_identity)
            group_key = (
                *base_key,
                hashlib.sha256(event_identity.encode("utf-8")).hexdigest()[:12],
                self._driver_group_identity(scenario),
                tuple(
                    (
                        _value(getattr(risk, field)),
                        self._score_states(
                            risk, trace, trace_provided=risk_trace is not None,
                        )[field],
                    )
                    for field in ("severity", "exposure", "controllability", "asil")
                ),
            )
            if instance.get("p6_generation"):
                causal = causal_by_pair.get((risk.malfunction_id, risk.scenario_id), {})
                complete = all(
                    getattr(risk, field).status.value == "FINALIZED"
                    for field in ("severity", "exposure", "controllability", "asil")
                )
                if (complete and causal.get("causal_assessment", {}).get("status") == "VALIDATED"
                        and causal.get("causal_chain")):
                    proof = signature({
                        "conditions": scenario_conditions(scenario),
                        "causal_chain": causal["causal_chain"],
                        "hazardous_event": normalized_event,
                        "potential_harm": risk.potential_harm,
                        "ratings": group_key[-1],
                        "exposure_domain": risk.exposure_tf,
                        "safety_goal_id": risk.safety_goal_id,
                    })
                    base_key = ("risk_equivalent", str(risk.malfunction_id), proof)
                    causal_identity = proof
                    group_key = base_key
                else:
                    # Unknown inputs and partial scores cannot prove equality.
                    base_key = ("unproven_risk_equivalence", str(risk.malfunction_id), str(risk.scenario_id))
                    group_key = base_key
            risk_equivalence_key = base_key if base_key[0] == "risk_equivalent" else None
            display_tuple, display_branch = self._member_display_tuple(
                {"risk": risk, "trace": trace}, asil_order, trace_provided=risk_trace is not None)
            if instance.get("p6_generation"):
                facts = scenario.facts
                causal = causal_by_pair.get((risk.malfunction_id, risk.scenario_id), {})
                mechanism = (facts.get("fault_motion_model") or facts.get("collision_type") or
                             (normalized_event + str(causal.get("causal_chain", {}).get("m_to_b", {}).get("claim", ""))))
                objects = sorted(category for category in self.text._categories(scenario)
                                 if category.startswith("OBJECT_"))
                conditions = scenario_conditions(scenario)
                # Atom/parameter variants remain in detail. Intervention and
                # interaction geometry still distinguish review rows, and every
                # row has exactly one tuple and one set of review states.
                review_fields = {"driver_position", "driver_in_vehicle",
                    "ego_longitudinal_direction", "object_position",
                    "object_longitudinal_direction", "object_motion_state", "collision_type",
                    "grade_coordinate_frame", "slope_reference", "ego_heading_relative_to_downhill",
                    "downhill_body_direction", "assessment_reference", "object_extent"}
                condition_key = signature({"operating_mode": scenario.operating_mode,
                    "facts": {key: value for key, value in conditions["facts"].items() if key in review_fields}})
                group_key = ("review", str(risk.malfunction_id), str(mechanism),
                             tuple(objects), str(risk.potential_harm or ""), condition_key,
                             self._driver_group_identity(scenario), display_tuple, tuple(self._score_states(
                                 risk, trace, trace_provided=risk_trace is not None).items()))
            else:
                group_key = (*group_key, display_tuple)
            entries.append({
                "risk_equivalence_key": risk_equivalence_key,
                "risk": risk, "scenario": scenario, "trace": trace,
                "hazardous_event_id": hazardous_event_id, "base_key": base_key,
                "group_key": group_key, "causal_identity": causal_identity,
                "display_tuple": display_tuple, "display_branch": display_branch,
            })

        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            grouped[entry["group_key"]].append(entry)
        ordered_groups = sorted(
            grouped.values(),
            key=lambda items: min(str(item["scenario"].scenario_id) for item in items),
        )
        group_consistency_failures = []
        for siblings in ordered_groups:
            identities = set()
            for entry in siblings:
                child_risk = entry["risk"]
                child_malfunction = malfunctions.get(child_risk.malfunction_id, {})
                identities.add((
                    str(child_malfunction.get("function_id", "")),
                    str(child_malfunction.get("guideword", "")),
                    str(child_malfunction.get("description", "")),
                    str(child_malfunction.get("vehicle_level_hazard", "")),
                    str(entry["group_key"] if entry["group_key"][0] == "review" else entry["causal_identity"]),
                ))
            if len(identities) != 1:
                group_consistency_failures.append({
                    "group_key": list(siblings[0]["group_key"]),
                    "identity_count": len(identities),
                })
        if group_consistency_failures:
            raise ValueError(
                "Reviewer grouping invariant failed: "
                f"{len(group_consistency_failures)} inconsistent groups"
            )

        rows: list[HARAReportRowView] = []
        details: list[ScenarioDetailView] = []
        audit: list[AuditReferenceView] = []
        variant_counts: Counter[str] = Counter()
        goal_presentations = {goal.sg_id: self.text.safety_goal_presentation([
            {"scenario_id": risk.scenario_id, "malfunction_id": risk.malfunction_id,
             "malfunction": str(malfunctions.get(risk.malfunction_id, {}).get("description", "")),
             "function": str(functions.get(str(malfunctions.get(risk.malfunction_id, {}).get("function_id", "")), {}).get("name", "")),
             "functional_effect": str(malfunctions.get(risk.malfunction_id, {}).get("functional_effect", "")),
             "vehicle_hazard": str(malfunctions.get(risk.malfunction_id, {}).get("vehicle_level_hazard", "")),
             "operating_mode": scenarios[risk.scenario_id].operating_mode if risk.scenario_id in scenarios else "",
             "hazardous_event": risk.hazardous_event}
            for risk in state.risk_results if risk.safety_goal_id == goal.sg_id
        ], goal.text) for goal in state.safety_goals}
        goal_texts = {sg_id: result["text"] for sg_id, result in goal_presentations.items()}
        goal_safe_states = {goal.sg_id: self._safe_state_display(goal,
                            state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}), scenarios=scenarios)
                            for goal in state.safety_goals}
        safe_state_members = {c["scenario_id"]: c for goal in state.safety_goals
            for c in state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}).get("safe_state_candidates", [])}
        severity_descriptions = ({level.level: level.description for level in method.severity.scale.levels}
                                 if hasattr(method, "severity") else {})
        timing_inputs = {
            (risk.malfunction_id, risk.scenario_id): RiskCalculationInputService().ftti(
                risk.malfunction_id, risk.scenario_id, _value(risk.asil),
                {**scenarios[risk.scenario_id].facts,
                 "_fact_provenance": scenarios[risk.scenario_id].fact_provenance}
                if risk.scenario_id in scenarios else {},
                risk.branch_evaluations,
                method=method,
            ).to_dict()
            for risk in state.risk_results
        }
        partial_group_distributions = {}
        for offset, siblings in enumerate(ordered_groups, start=1):
            siblings.sort(key=self._representative_rank)
            headline, highest_members = self._highest_known_tuple(siblings, asil_order)
            tied_tuples = len({values for _, values, _ in highest_members}) > 1
            representative = highest_members[0][0] if highest_members else siblings[0]
            risk = representative["risk"]
            scenario = representative["scenario"]
            trace = representative["trace"]
            malfunction = malfunctions.get(risk.malfunction_id, {})
            function = functions.get(str(malfunction.get("function_id", "")), {})
            operational, detail = self._group_scenario(siblings)
            severity_trace = self._display_trace("severity", trace)
            exposure_trace = self._display_trace("exposure", trace)
            control_trace = self._display_trace("controllability", trace)
            score_states = self._score_states(
                risk, trace, trace_provided=risk_trace is not None,
            )
            values = dict(zip(("severity", "exposure", "controllability", "asil"),
                              representative["display_tuple"]))
            score_states = {field: _UNCALCULATED if values[field] == "待评定" else status
                            for field, status in score_states.items()}
            review_group_id = "RG-" + signature(representative["group_key"])[:12].upper()
            is_review_group = representative["group_key"][0] == "review"
            partial_rationales = {}
            if is_review_group and not headline:
                partial_rationales, distribution = self._group_grade_summary(
                    siblings, trace_provided=risk_trace is not None)
                partial_group_distributions[f"HARA_{offset:03d}"] = distribution
            pending_count = sum(
                any(not _value(getattr(e["risk"], f)) for f in ("severity", "exposure", "controllability", "asil"))
                or any(not all(b.get(k) for k in ("S", "E", "C", "ASIL")) for b in e["risk"].branch_evaluations)
                for e in siblings)
            causal_status = causal_status_by_pair.get((risk.malfunction_id, risk.scenario_id), "")
            assessment_status = self._assessment_status(
                risk=risk, scenario=scenario, trace=trace, causal_status=causal_status,
                trace_provided=risk_trace is not None,
            )
            group_goal_ids = sorted({entry["risk"].safety_goal_id for entry in siblings
                                     if entry["risk"].safety_goal_id})
            hara_id = f"HARA_{offset:03d}"
            row = HARAReportRowView(
                hara_id=hara_id,
                malfunction_id=risk.malfunction_id,
                scenario_id=risk.scenario_id,
                hazardous_event_id=representative["hazardous_event_id"],
                function_id=str(malfunction.get("function_id", "")),
                function_name=str(function.get("name", "")),
                function_output=str(function.get("output", "")),
                guideword=str(malfunction.get("guideword", "")),
                malfunction=str(malfunction.get("description", "")),
                hazard=str(malfunction.get("vehicle_level_hazard", "")),
                operational_scenario=operational,
                scenario_detail=detail,
                hazardous_event=self.text.hazardous_event(risk.hazardous_event, scenario),
                potential_harm=self.text.potential_harm(
                    risk.potential_harm if not headline or values["severity"] == _value(risk.severity) else "",
                    severity=(replace(risk.severity, value=values["severity"])
                              if values["severity"] in severity_descriptions else risk.severity),
                    method_description=severity_descriptions.get(values["severity"], "")),
                severity=values["severity"],
                severity_rationale=self._score_rationale(
                    "severity", risk.severity, severity_trace, score_states["severity"],
                ),
                exposure=values["exposure"],
                exposure_rationale=self._score_rationale(
                    "exposure", risk.exposure, exposure_trace, score_states["exposure"],
                ),
                controllability=values["controllability"],
                controllability_rationale=self._score_rationale(
                    "controllability", risk.controllability, control_trace,
                    score_states["controllability"],
                ),
                asil=values["asil"],
                asil_rationale=self._score_rationale(
                    "asil", risk.asil,
                    trace.get("asil", {}) if isinstance(trace.get("asil"), Mapping) else {},
                    score_states["asil"],
                ),
                ftti=self.text.ftti_value([timing_inputs[(e["risk"].malfunction_id, e["risk"].scenario_id)]
                                          for e in siblings]),
                ftti_rationale=self.text.ftti_rationale([
                    timing_inputs[(e["risk"].malfunction_id, e["risk"].scenario_id)]
                    for e in siblings]),
                sg_id="\n".join(group_goal_ids),
                safety_goal="\n".join(
                    (f"{sg_id}：" if len(group_goal_ids) > 1 else "") + goal_texts.get(sg_id, "")
                    for sg_id in group_goal_ids),
                safe_state="\n".join(
                    (f"{sg_id}：" if len(group_goal_ids) > 1 else "") + self._safe_state_display(
                        next(goal for goal in state.safety_goals if goal.sg_id == sg_id),
                        state.item_definition.get("safety_goal_catalog", {}).get(sg_id, {}),
                        [e["risk"].scenario_id for e in siblings if e["risk"].safety_goal_id == sg_id], scenarios=scenarios)
                    for sg_id in group_goal_ids) or (
                        "不适用（本行QM，无需分配SG）" if all(e["risk"].asil.value == "QM" for e in siblings)
                        else "待确定"),
                assessment_status=assessment_status,
                clarification_ids=self._clarifications(risk),
                remark=(f"审核展示分组，不代表风险等价；{len(siblings)} 个成员。" if is_review_group and len(siblings) > 1 else "")
                  + (("并列最高 ASIL 有不同元组；主表显示其中一个实际分支的完整元组，其余见明细；" if tied_tuples else
                      "最高已知 ASIL 的完整元组；") if highest_members and (len(siblings) > 1 or len(risk.branch_evaluations) > 1) else "")
                  + ("尚有未决成员，最高已知结果不是全组上界；" if pending_count and is_review_group else "")
                  + self._status_note(score_states)
                  + ("；条件差异：" + self.text.group_differences([e["scenario"] for e in siblings]) if len(siblings) > 1 else "")
                  + ("；归并依据：相同结构化场景条件、因果链、伤害机制及完整评分；原始记录见场景明细"
                     if len(siblings) > 1 and all(e["risk_equivalence_key"] is not None and e["risk_equivalence_key"] == representative["risk_equivalence_key"] for e in siblings) else "")
                  + ("；SG为历史草稿关联，本轮未重评" if group_goal_ids and any(
                      event.get("event") == "p7_fixed_population_replay" for event in state.audit_trail) else ""),
            )
            if representative["display_branch"]:
                # An aggregate max-S rationale may describe a different branch.
                # Point every headline dimension at the same selected branch.
                row = replace(row, **{field + "_rationale":
                    self._score_rationale(field, replace(getattr(risk, field), value=(
                        None if values[field] == "待评定" else values[field]),
                        status=ReviewStatus.PENDING if values[field] == "待评定" else getattr(risk, field).status),
                        self._display_trace(field, trace, representative["display_branch"]), score_states[field])
                    for field in ("severity", "exposure", "controllability", "asil")})
            elif partial_rationales:
                row = replace(row, **partial_rationales)
            if representative["display_branch"] and not headline:
                row = replace(row, remark="分支评级尚未闭合；主表列示一个实际分支的评级，全部分支见场景明细。" + row.remark)
            if headline and headline[-1] == "QM" and row.asil == "待评定":
                row = replace(row, asil_rationale=(
                    "已计算碰撞分支的已知 ASIL 为 QM；其余碰撞分支 ASIL 尚未评定，不能认定全组 QM。"
                    "完整分支评级见场景明细。"))
            if (values["severity"] == "待评定" and any(_value(e["risk"].severity) for e in siblings)
                    and (len(siblings) > 1 or len(risk.branch_evaluations) > 1)):
                row = replace(row, potential_harm="伤害类别按各成员 S 分档分别查阅方法依据；未完成成员仍待定。")
            if asil_service and row.asil in asil_order:
                known = {field: getattr(row, field) for field in ("severity", "exposure", "controllability")
                         if getattr(row, field) != "待评定"}
                if len(known) < 3:
                    # Explain a stored partial result using the active matrix.
                    # This read-only proof does not supply a missing grade.
                    result, _ = asil_service.partial_zero_decision(known)
                    if result == row.asil:
                        missing = "/".join(label for field, label in (
                            ("severity", "S"), ("exposure", "E"), ("controllability", "C")) if field not in known)
                        row = replace(row, asil_rationale=(
                            f"已有 {' / '.join(known.values())}；活动矩阵中其余维度的全部组合均对应 {result}。"
                            f"{missing} 本身仍待评定，该矩阵结论不补齐缺失维度。"))
            if len(siblings) > 1 and highest_members:
                # A deciding tuple does not erase other known member grades.
                _, distribution = self._group_grade_summary(siblings, trace_provided=risk_trace is not None)
                row = replace(row, remark=row.remark + "；成员等级分布：" + "；".join(
                    f"{label}: " + "、".join(f"{grade}×{count}" for grade, count in distribution[field]["members"].items())
                    for field, label in (("severity", "S"), ("exposure", "E"), ("controllability", "C"), ("asil", "ASIL"))))
            if row.asil == "待评定":
                explanations = Counter(self._missing_asil_explanation(e["risk"], e["trace"], method, detail=False)
                                       for e in siblings if not _value(e["risk"].asil))
                if explanations:
                    row = replace(row, asil_rationale=row.asil_rationale + "\n" + "\n".join(
                        f"{count}个成员：{reason}" for reason, count in explanations.items()))
            rows.append(row)
            group_source_counts[str(representative["base_key"][0])] += 1

            for entry in siblings:
                child_risk = entry["risk"]
                child = entry["scenario"]
                child_trace = entry["trace"]
                child_causal = causal_status_by_pair.get(
                    (child_risk.malfunction_id, child_risk.scenario_id), "",
                )
                child_status = self._assessment_status(
                    risk=child_risk, scenario=child, trace=child_trace,
                    causal_status=child_causal,
                    trace_provided=risk_trace is not None,
                )
                child_operational, child_detail = self.text.scenario(child)
                variant = self.text.variant_text(child)
                variant_counts[variant] += 1
                child_values = dict(zip(("severity", "exposure", "controllability", "asil"),
                                        entry["display_tuple"]))
                timing = timing_inputs[(child_risk.malfunction_id, child.scenario_id)]
                timing_note = ("\n条件性时间边界：" + self.text.ftti_value([timing])
                    + "（故障发生至失去所选响应的避碰能力；假设及处置轨迹见时间追溯）"
                    if timing.get("hazard_time_limit_s") is not None else "")
                if timing.get("timing_trace", {}).get("status") == "AVOIDABILITY_ALREADY_LOST":
                    timing_note = "\n" + self.text.ftti_rationale([timing])
                details.append(ScenarioDetailView(
                    hara_id=hara_id,
                    variant=variant,
                    scenario_id=child.scenario_id,
                    operational_scenario=child_operational,
                    scenario_detail=child_detail,
                    speed_constraint=self.text.speed_text(child),
                    causal_status=child_status,
                    hazardous_event=self.text.hazardous_event(child_risk.hazardous_event, child),
                    semantic_group_id=self._semantic_group_id(child),
                    object_interaction_summary=self.text.object_interaction_summary(child),
                    physical_inputs=self.text.physical_inputs(child, child_trace),
                    driver_branch=self.text.driver_branch(child, child_trace),
                    controllability_branch=self.text.controllability_branch(child_trace),
                    analysis_basis=self.text.analysis_basis(child) + "\n" + boundary_notes(child.facts, method) + timing_note
                        + ("\n" + self.text.safe_state_member(safe_state_members[child.scenario_id], child)
                           if child.scenario_id in safe_state_members else ""),
                    review_group_id=review_group_id,
                    **child_values,
                    pending_reasons=self._missing_asil_explanation(child_risk, child_trace, method) or "；".join(self._score_rationale(field, getattr(child_risk, field),
                        child_trace.get(field, {}), _UNCALCULATED)
                        for field in ("severity", "exposure", "controllability", "asil")
                        if not _value(getattr(child_risk, field))),
                    branch_results="\n".join(self.text.branch_result(branch)
                                             for branch in child_risk.branch_evaluations),
                ))
                audit.append(AuditReferenceView(
                    run_id=state.run_id,
                    method_hash=str(method.metadata.get("method_source_hash", method.metadata.get("template_hash", ""))),
                    report_schema_hash=self.schema.schema_hash,
                    style_template_hash=style_template_hash,
                    hara_id=hara_id,
                    hazardous_event_id=entry["hazardous_event_id"],
                    scenario_id=child.scenario_id,
                    risk_trace_reference=risk_trace_reference,
                    clarification_ids=self._clarifications(child_risk),
                    assessment_status=child_status,
                    semantic_group_id=self._semantic_group_id(child),
                    parent_scenario_id=self._parent_scenario_id(child),
                    variant=variant,
                    selected_atom_ids=self._selected_atom_ids(child),
                    source_references=self._source_references(child),
                ))

        summary = run_summary or {}

        score_counts = {
            field: Counter(
                self._score_states(
                    entry["risk"], entry["trace"],
                    trace_provided=risk_trace is not None,
                )[field]
                for entry in entries
            )
            for field in ("severity", "exposure", "controllability", "asil")
        }

        def count(field: str) -> int:
            return score_counts[field][_CALCULATED]

        complete_risk_chain_count = sum(
            all(
                self._score_states(
                    entry["risk"], entry["trace"],
                    trace_provided=risk_trace is not None,
                )[field] == _CALCULATED
                for field in ("severity", "exposure", "controllability", "asil")
            )
            for entry in entries
        )
        scope = self._run_scope(state, trace_rows, complete_risk_chain_count)
        scope["numeric_risk_chain_count"] = sum(
            all(value != _UNCALCULATED for value in self._score_states(
                entry["risk"], entry["trace"], trace_provided=risk_trace is not None).values())
            for entry in entries)
        scope["numeric_asil_count"] = score_counts["asil"][_CALCULATED] + score_counts["asil"][_TRIAL]
        scope["asil_without_complete_sec_count"] = scope["numeric_asil_count"] - scope["numeric_risk_chain_count"]
        coverage_notes = (run_summary or {}).get("input_coverage_notes", [])
        if coverage_notes:
            if not isinstance(coverage_notes, list) or any(not isinstance(note, str) for note in coverage_notes):
                raise ValueError("Report input coverage notes must be a list of text")
            scope["scope_note"] = "\n".join(note for note in [scope["scope_note"], *coverage_notes] if note)

        method_hash = str(method.metadata.get("method_source_hash", method.metadata.get("template_hash", "")))
        clarification_ids = "; ".join(sorted({
            item for row in rows for item in row.clarification_ids.split("; ") if item
        }))
        summary_view = SummaryView(
            run_id=state.run_id,
            method_source=str(state.method_contract.get("source_kind", "YAML_BASELINE")),
            method_hash=method_hash,
            report_schema=self.schema.report_id,
            report_schema_version=self.schema.schema_version,
            report_schema_hash=self.schema.schema_hash,
            style_template_hash=style_template_hash,
            report_status=(
                "DRAFT_READY"
                if entries and all(
                    score_counts[field][_CALCULATED] == len(entries)
                    for field in score_counts
                )
                else "CONDITIONAL TRIAL — NOT FOR RELEASE"
                if any(score_counts[field][_TRIAL] for field in score_counts)
                else "RISK SCORING PARTIAL — NOT FOR RELEASE"
                if (
                    any(bool(entry["trace"].get("risk_scoring_invoked")) for entry in entries)
                    or any(score_counts[field][_CALCULATED] for field in score_counts)
                )
                else "SCENARIO SYNTHESIS COMPLETE — CAUSAL REVALIDATION IN PROGRESS — "
                "RISK SCORING NOT YET EXECUTED"
            ),
            release_status="NOT FOR RELEASE",
            function_count=len(state.functions),
            guideword_assessment_count=len(state.guideword_assessments),
            malfunction_count=len(state.malfunctions),
            scenario_count=len(state.scenarios),
            eligible_hazardous_event_count=len(entries),
            severity_finalized=count("severity"),
            severity_pending=len(state.risk_results) - count("severity"),
            exposure_finalized=count("exposure"),
            exposure_pending=len(state.risk_results) - count("exposure"),
            controllability_finalized=count("controllability"),
            controllability_pending=len(state.risk_results) - count("controllability"),
            asil_finalized=count("asil"),
            asil_pending=len(state.risk_results) - count("asil"),
            clarification_ids=clarification_ids,
            scope=scope,
        )
        basis = MethodBasisView(
            method_source=summary_view.method_source,
            guidewords=f"{len(method.guidewords.guidewords)} 个启用的引导词",
            severity="基于相对速度；依赖交通参与者与碰撞配置",
            exposure=("基于活动项目方法；E 聚合政策与未定源值参与性见风险轨迹，不等同通用标准规定"),
            controllability="基于 MethodContract 的可控性模型；TTC 证据保持可追溯",
            asil="当前 MethodContract 矩阵",
            ftti=("支持活动方法列明轨迹的条件性制动避碰时间边界；需明确故障起点和剩余净减速响应。直线移动目标还需检查停车保持后的交互；未覆盖路径仅列接触证据，不冒充已批准FTTI。"
                  if getattr(method, "metadata", {}).get("project_analysis_policy", {}).get("timing_analysis") else
                  "本次方法未配置响应时间规则；轨迹接触时间仅作证据，不作为FTTI。"),
        )
        offline_goals = any(event.get("event") == "offline_source_context_replay" for event in state.audit_trail)
        conditional_goal_ids = {risk.safety_goal_id for risk in state.risk_results
            if risk.asil.value and risk.asil.status.value != "FINALIZED"}
        goals = tuple(
            SafetyGoalView(goal.sg_id, goal_texts[goal.sg_id],
                goal_safe_states[goal.sg_id],
                goal.max_asil or "待评定",
                "；".join(dict.fromkeys(str(malfunctions.get(risk.malfunction_id, {}).get("vehicle_level_hazard", ""))
                    for risk in state.risk_results if risk.safety_goal_id == goal.sg_id)),
                original_text=state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}).get("original_safety_goal", goal.text),
                status="离线模拟草稿" if offline_goals
                    else "条件性草稿" if goal.sg_id in conditional_goal_ids
                    else "历史草稿" if any(event.get("event") == "p7_fixed_population_replay" for event in state.audit_trail)
                    else ("待审核" if goal.status.value != "FINALIZED" else "已审核"))
            for goal in state.safety_goals
        )
        divergence_groups = sum(max(0, len(events) - 1) for events in base_events.values())
        speed_values = [item.speed_constraint for item in details if item.speed_constraint]
        projection_metrics = {
            "safety_goal_scope_note": "最高 ASIL 仅覆盖本次已计算成员；未计算成员不包含在该最大值中。"
                + " Safety State为期望达到的车辆安全状态；候选及适用条件另行审核，未生成不等于没有安全状态。"
                + (" 本页使用离线模拟响应。" if offline_goals else "")
                + (" 条件性数值仍待工程确认。" if conditional_goal_ids else ""),
            "safety_goal_stage_completed": any(event.get("event") == "safety_goals_aggregated"
                                               for event in state.audit_trail),
            "display_review_groups": len(rows),
            "grouping_explanation": (
                f"{len(details)}条风险明细组织为{len(rows)}个审核组。按同一失效、物理机制、对象及伤害分组，同时区分驾驶员位置、方向、坡向以及S/E/C/ASIL和审核状态；"
                "每行仅列一组评级，Atom和参数变体保留在明细；"
                "每条明细保留原身份、条件和分支评级，按HARA-ID查回主表。分组便于审核，不证明风险等价或ODD覆盖完整。"),
            "safety_goal_text_mapping": [goal.to_dict() for goal in goals],
            "safety_goal_display_evidence": goal_presentations,
            "safe_state_evidence": {goal.sg_id: {
                "recorded_text": goal.safe_state, "goal_status": goal.status.value,
                "candidates": state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}).get("safe_state_candidates", []),
                "gaps": state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}).get("safe_state_gaps", []),
                "prior": state.item_definition.get("safety_goal_catalog", {}).get(goal.sg_id, {}).get("prior_safe_state"),
            } for goal in state.safety_goals},
            "ftti_timing_inputs": list(timing_inputs.values()),
            "partial_group_grade_distributions": partial_group_distributions,
            "collision_branch_rows": sum(len(risk.branch_evaluations) for risk in state.risk_results),
            "child_scenario_rows": len(details),
            "main_hara_groups": len(rows),
            "grouping_reduction": len(details) - len(rows),
            "groups_not_merged_due_hazardous_event_divergence": divergence_groups,
            "group_source_counts": dict(sorted(group_source_counts.items())),
            "grouping_map": [{
                "hara_id": f"HARA_{index:03d}",
                "review_group_id": "RG-" + signature(siblings[0]["group_key"])[:12].upper(),
                "basis": "review_display" if siblings[0]["group_key"][0] == "review" else siblings[0]["base_key"][0],
                "risk_equivalence_keys": [entry["risk_equivalence_key"] for entry in siblings],
                "combinations": list(dict.fromkeys(self.text.condition_combination(entry["scenario"]) for entry in siblings)),
                "member_ratings": [{"scenario_id": entry["scenario"].scenario_id,
                     "safety_goal_id": entry["risk"].safety_goal_id,
                     "tuple": [_value(getattr(entry["risk"], f)) for f in ("severity", "exposure", "controllability", "asil")],
                     "branches": entry["risk"].branch_evaluations} for entry in siblings],
                "assessment_ids": [entry["risk"].assessment_id for entry in siblings],
                "scenario_ids": [entry["scenario"].scenario_id for entry in siblings],
            } for index, siblings in enumerate(ordered_groups, start=1)],
            "variant_distribution": dict(sorted(variant_counts.items())),
            "distinct_visible_speed_expressions": len(set(speed_values)),
            "broad_0_20_speed_rows": sum("0–20 km/h" in value for value in speed_values),
            "source_conflict_speed_rows": sum("来源存在冲突" in value for value in speed_values),
            "contextual_speed_rows": sum(
                isinstance(
                    (getattr(item["scenario"], "facts", {}) or {}).get(
                        "speed_context_resolution"
                    ), Mapping,
                )
                and str(
                    (getattr(item["scenario"], "facts", {}) or {}).get(
                        "speed_context_resolution", {}
                    ).get("classification", "")
                ) == "CONTEXTUAL_SPEED_CONSUMED"
                for item in entries
            ),
            "group_consistency_failures": len(group_consistency_failures),
            "score_status_counts": {
                field: {
                    "calculated": score_counts[field][_CALCULATED],
                    "conditional_trial": score_counts[field][_TRIAL],
                    "uncalculated": score_counts[field][_UNCALCULATED],
                }
                for field in score_counts
            },
        }
        children = self._generated_children(trace_rows, generated_scenarios or [], malfunctions, causal_by_pair)
        projection_metrics["source_condition_dispositions"] = [child.to_dict() for child in children]
        disposition_counts: dict[str, Counter] = defaultdict(Counter)
        for child in children:
            disposition_counts[child.malfunction_id][child.causal_disposition] += 1
        inventory = []
        for item in state.malfunctions:
            identity = str(item.get("malfunction_id", ""))
            counts = disposition_counts[identity]
            retained = counts["RETAINED_FOR_RISK"]
            excluded = counts["EXCLUDED_FROM_RISK"]
            pending = counts["PENDING_CAUSAL"]
            members = [entry for entry in entries if entry["risk"].malfunction_id == identity]
            distributions = []
            member_states = {entry["risk"].assessment_id: self._score_states(
                entry["risk"], entry["trace"], trace_provided=risk_trace is not None) for entry in members}
            known_asils = [entry["risk"].branch_summary.get("highest_known_asil") or
                self._display_score(entry["risk"].asil, member_states[entry["risk"].assessment_id]["asil"]) for entry in members]
            known_asils = [value for value in known_asils if value in asil_order]
            known_s = [entry["risk"].branch_summary.get("severity_maximum") or
                self._display_score(entry["risk"].severity, member_states[entry["risk"].assessment_id]["severity"]) for entry in members]
            s_order = list(method.asil.severity_levels) if hasattr(method, "asil") else []
            known_s = [value for value in known_s if value in s_order]
            conditional_count = sum(_TRIAL in states.values() for states in member_states.values())
            distributions.append(("包含条件性试算；已知最大 S=" if conditional_count else "已知最大 S=") + (max(known_s, key=s_order.index) if known_s else "Pending")
                + "；已知最高 ASIL=" + (max(known_asils, key=asil_order.index) if known_asils else "Pending")
                + ("；仅限所列假设，未证明全范围上界" if conditional_count else "；仅汇总已验证子结果，未决分支另列"))
            for dimension in ("severity", "exposure", "controllability", "asil"):
                distribution = Counter(self._display_score(getattr(entry["risk"], dimension), member_states[entry["risk"].assessment_id][dimension]) for entry in members)
                distributions.append(f"{dimension}: " + ", ".join(f"{key}×{value}" for key, value in sorted(distribution.items())))
                statuses = Counter(member_states[entry["risk"].assessment_id][dimension] for entry in members)
                distributions.append("评定性质：" + "、".join(f"{status}×{count}" for status, count in statuses.items()))
            pending_reasons = Counter(
                self._score_rationale(field, getattr(entry["risk"], field),
                    self._display_trace(field, entry["trace"]), _UNCALCULATED)
                for entry in members for field in ("severity", "exposure", "controllability", "asil")
                if member_states[entry["risk"].assessment_id][field] == _UNCALCULATED
            )
            if not members:
                distributions = ["未进入风险评分；因果判定及具体理由见生成子场景审计。"]
            coverage = [f"{len(members)} 个风险结果；完整条件见场景明细" if members
                        else "未保留风险；已判定条件和具体原因见生成子场景审计"]
            for label, field in (("对象", "road_user_type"), ("目标方位", "object_position"),
                                 ("自车方向", "ego_longitudinal_direction"), ("驾驶员", "driver_position")):
                values = sorted({str(entry["scenario"].facts[field]) for entry in members if field in entry["scenario"].facts})
                if values:
                    coverage.append(f"{label}：" + " / ".join(values))
            locations = sorted({str(entry["scenario"].facts.get("source_operating_conditions", {}).get("operating_scenario", ""))
                                for entry in members} - {""})
            if locations:
                coverage.append("位置：" + " / ".join(locations))
            inventory.append(MalfunctionView(
                identity, str(item.get("function_id", "")), str(item.get("guideword", "")),
                str(item.get("description", "")), str(item.get("vehicle_level_hazard", "")),
                retained, excluded, pending,
                "RETAINED_FOR_RISK" if retained else "PENDING_CAUSAL" if pending
                else "EXCLUDED_FROM_RISK" if excluded else "NOT_ASSESSED",
                review_group="按失效/危险声明展示；不表示风险等价，各子结果独立",
                coverage_summary="\n".join(coverage),
                risk_distribution="\n".join(distributions),
                pending_summary="\n".join(f"{reason or '待因果/方法证据'}: {count}" for reason, count in pending_reasons.items()),
            ))
        function_inventory = []
        for function in state.functions:
            children_for_function = [item for item in inventory if item.function_id == str(function.get("function_id", ""))]
            retained, excluded, pending = (sum(getattr(item, field) for item in children_for_function)
                                           for field in ("retained", "excluded", "pending"))
            function_inventory.append(FunctionView(str(function.get("function_id", "")),
                str(function.get("name", "")), str(function.get("output", "")),
                str(function.get("status", "未记录审核状态")), len(children_for_function),
                retained + excluded + pending, retained, excluded, pending))
        return HARAReportViewModel(
            tuple(rows), summary_view, basis, goals, tuple(audit), self.schema.schema_hash,
            method_hash, style_template_hash, tuple(details), projection_metrics,
            children, tuple(inventory),
            tuple((label, "、".join(value) if isinstance(value, list) else str(value))
                  for field, label in (("system_description", "系统"), ("item_boundary", "边界"),
                      ("operating_modes", "运行模式"), ("odd_locations", "允许位置"),
                      ("odd_road_types", "允许道路/车位"), ("odd_weather_conditions", "天气"),
                      ("odd_road_surfaces", "路面"))
                  if (value := state.item_definition.get("typed", {}).get(field))),
            tuple(function_inventory),
        )


__all__ = ["HARAReportProjectionService"]
