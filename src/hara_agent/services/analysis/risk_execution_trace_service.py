from __future__ import annotations

"""Read-only projections of Scenario-to-risk execution.

This service is intentionally observational: it consumes canonical state,
compiled MethodContract data and executor results, but never selects an S/E/C
value or repairs an input.
"""

from typing import Any, Iterable

from hara_agent.contracts import CalculationStatus, MethodContract
from hara_agent.models import evaluate_risk_eligibility_payload

from .scenario_physics import TTC_FORMULA_IDENTITY


def _value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _source(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return {
        "source_type": getattr(value, "source_type", "method_contract"),
        "source_id": getattr(value, "source_id", getattr(value, "workbook", "")),
        "location": getattr(value, "location", "") or (
            f"{getattr(value, 'sheet', '')}!{getattr(value, 'range', '')}"
        ).strip("!"),
    }


def _provenance(scenario: dict[str, Any], key: str) -> dict[str, Any]:
    values = scenario.get("_fact_provenance", scenario.get("fact_provenance", {}))
    return dict(values.get(key, {})) if isinstance(values, dict) and isinstance(values.get(key), dict) else {}


def _status(result: dict[str, Any]) -> str:
    return str(result.get("calculation_status", "PENDING_METHOD_SEMANTICS"))


class RiskExecutionTraceService:
    """Project lifecycle, inputs and deterministic executor decisions."""

    def __init__(self, method: MethodContract | None = None):
        self.method = method

    @property
    def method_contract_hash(self) -> str:
        if self.method is None:
            return ""
        metadata = self.method.metadata
        return str(metadata.get("method_source_hash", metadata.get("template_hash", "")))

    def _eligibility(
        self, record: dict[str, Any], *, committed: bool,
    ) -> dict[str, Any]:
        decision = evaluate_risk_eligibility_payload(record, committed=committed)
        inputs = decision.authoritative_inputs
        feasibility = {
            "status": inputs.get("assessment_status", "MISSING"),
            "disposition": "RETAIN" if decision.final_retain_value else "DO_NOT_RETAIN",
            "causal_status": inputs.get("causal_validation_status", "MISSING"),
            "assessment_available": bool(inputs.get("causal_assessment_available")),
            "persisted_review": True,
            "committed_to_state": committed,
        }
        return {
            "feasibility": feasibility,
            "risk_eligibility": {
                "status": decision.status.value,
                "reason_codes": list(decision.reason_codes),
                "authoritative_fields": inputs,
                "final_retain_value": decision.final_retain_value,
                "final_retain_role": decision.final_retain_role,
                "source_stage": (
                    "scenario_feasibility_persistence" if not committed
                    else "evaluate_risk_eligibility"
                ),
            },
        }

    def _severity_trace(self, scenario: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        structured = getattr(self.method, "structured_risk_method", None)
        semantic = (
            structured.severity.speed_semantic.value
            if structured is not None else "UNRESOLVED"
        )
        input_key = {
            "EGO_SPEED": "ego_speed_kph",
            "RELATIVE_SPEED": "relative_speed_kph",
            "IMPACT_SPEED": "impact_speed_kph",
            "DELTA_V": "delta_v_kph",
        }.get(semantic, "")
        value = scenario.get(input_key) if input_key else None
        input_provenance = _provenance(scenario, input_key) if input_key else {}
        provenance = structured.severity.semantic if structured is not None else None
        pending = result.get("reasoning", "") if _status(result) != CalculationStatus.FINALIZED.value else ""
        return {
            "executor": "SeverityMethodExecutor",
            "status": _status(result),
            "input_semantic": semantic,
            "semantic_resolution": (
                provenance.semantic_resolution.value if provenance is not None else "UNRESOLVED"
            ),
            "source_status": provenance.source_status if provenance is not None else "",
            "inputs": {
                input_key: value,
                "risk_context_field": input_key,
                "source": input_provenance.get("evidence_ref", ""),
                "status": input_provenance.get("approval", "MISSING"),
            },
            "ego_speed_constraint": {"value": scenario.get("ego_speed_constraint"), "USED_FOR_SEVERITY": False},
            "missing_inputs": [input_key] if input_key and value is None else [],
            "derived_values": {key: scenario.get(key) for key in ("delta_v_kph", "impact_speed_kph", "relative_speed_kph") if key in scenario},
            "rule_ids": [result.get("engineering_rule_id", "")] if result.get("engineering_rule_id") else [],
            "source_refs": [result.get("engineering_location", "")] if result.get("engineering_location") else [],
            "result": result.get("severity_score", ""),
            "executor_invoked": bool(result.get("executor_invoked", True)),
            "pending_reason": (
                f"MISSING_{semantic}" if input_key and value is None
                else result.get("pending_reason") or pending
            ),
        }

    @staticmethod
    def _policy_branch(levels: list[str], dependent: bool, policy: Any) -> str:
        if not levels:
            return "NO_USABLE_ATOMS"
        if all(level == policy.all_highest_operand for level in levels):
            return "ALL_E4"
        if set(policy.mixed_high_operands).issubset(set(levels)):
            return "E3_E4_MIX"
        values = [int(level[1:]) for level in levels]
        if min(values) != max(values):
            return "MIN_WHEN_UNEQUAL"
        return "SAME_COUPLED_NO_CHANGE" if dependent else "SAME_INDEPENDENT_MINUS_ONE"

    def _exposure_trace(
        self, scenario: dict[str, Any], result: dict[str, Any], binding_gaps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        structured = getattr(self.method, "structured_risk_method", None)
        method = structured.exposure if structured is not None else None
        raw_ids = scenario.get("scenario_atom_ids", ())
        atom_ids = [str(item) for item in raw_ids] if isinstance(raw_ids, (list, tuple)) else []
        selected_domain = str(result.get("actual_domain", result.get("exposure_method", "")))
        requested_domain = str(result.get("requested_domain", selected_domain))
        executor_invoked = bool(
            result.get("executor_invoked", _status(result) == CalculationStatus.FINALIZED.value)
        )
        atoms: list[dict[str, Any]] = []
        classes: list[str] = []
        dimensions: set[str] = set()
        details = result.get("atom_details", [])
        details_by_id = {
            str(item.get("atom_id", "")): item
            for item in details if isinstance(item, dict)
        }
        if method is not None:
            by_id = {item.atom_id: item for item in method.atoms}
            for atom_id in atom_ids:
                atom = by_id.get(atom_id)
                detail = details_by_id.get(atom_id, {})
                level = str(detail.get("e_rank", ""))
                atom_dimensions = list(atom.dimensions) if atom is not None else list(detail.get("dimensions", []))
                atoms.append({
                    "atom_id": atom_id, "dimension": atom_dimensions,
                    "E_class": level,
                    "used": bool(detail.get("used", False)),
                    "skipped": bool(detail.get("skipped", False)),
                    "skip_reason": str(detail.get("skip_reason", "")),
                    "requested_domain": str(detail.get("requested_domain", requested_domain)),
                    "actual_domain": str(detail.get("actual_domain", "")),
                    "available_but_unused": not executor_invoked,
                    "source": _source(atom.source_ref) if atom is not None else {},
                })
                dimensions.update(atom_dimensions)
                if bool(detail.get("used", False)) and level in {"E0", "E1", "E2", "E3", "E4"}:
                    classes.append(level)
            policy = method.aggregation_policy
            aggregation = {
                "policy_id": policy.policy_id,
                "policy_branch": str(result.get("aggregation_rule", "NOT_INVOKED")),
                "input_classes": classes,
                "coupling": str(result.get("coupling", "")),
                "coupling_consumed": bool(result.get("coupling_consumed", False)),
                "strong_couplings": [list(pair) for pair in method.strong_couplings],
            }
        else:
            aggregation = {"policy_branch": "NOT_STRUCTURED"}
        pending = result.get("reasoning", "") if _status(result) != CalculationStatus.FINALIZED.value else ""
        return {
            "executor": "ExposureMethodExecutor", "status": _status(result),
            "executor_invoked": executor_invoked,
            "coverage_status": result.get("coverage_status", ""),
            "coverage_rule_ids": list(result.get("coverage_rule_ids", [])),
            "coverage_granularity": result.get("coverage_granularity", ""),
            "coverage_gate_applied": bool(result.get("coverage_gate_applied", False)),
            "input_readiness": dict(result.get("exposure_input_readiness", {})),
            "readiness_status": str(
                result.get("exposure_input_readiness", {}).get("status", "")
            ) if isinstance(result.get("exposure_input_readiness", {}), dict) else "",
            "readiness_reason": str(
                result.get("exposure_input_readiness", {}).get("reason_code", "")
            ) if isinstance(result.get("exposure_input_readiness", {}), dict) else "",
            "missing_method_semantics": result.get("missing_method_semantics", ""),
            "component_category": scenario.get("component_category", ""),
            "dimensions": sorted(dimensions), "scenario_terms": scenario.get("method_scenario_dimensions", {}),
            "atom_bindings": atoms, "atom_ids": atom_ids,
            "requested_domain": requested_domain, "domain": selected_domain,
            "scenario_level_fallback": bool(result.get("dimension_fallback", False)),
            "intermediate_values": classes, "dependency_coupling": aggregation,
            "aggregation_policy": aggregation.get("policy_id", ""),
            "rule_ids": [result.get("engineering_rule_id", "")] if result.get("engineering_rule_id") else [],
            "result": result.get("exposure_score", ""),
            "pending_reason": result.get("pending_reason") or (
                "PENDING_EXPOSURE_ATOM" if not atom_ids else pending
            ),
            "atom_available": bool(atoms),
            "atom_used_for_scoring": bool(atoms) and executor_invoked,
            "upstream_binding_gaps": binding_gaps,
        }

    def _controllability_trace(self, scenario: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        structured = getattr(self.method, "structured_risk_method", None)
        profile = structured.controllability_profile if structured is not None else None
        relative_speed = scenario.get("relative_speed_kph")
        closing_speed = scenario.get("closing_speed_kph")
        distance = scenario.get("relative_distance_m", scenario.get("relative_distance"))
        ttc = scenario.get("ttc_s")
        provenance = scenario.get("_fact_provenance", {})
        ttc_metadata = provenance.get("ttc_s", {}) if isinstance(provenance, dict) else {}
        ttc_metadata = ttc_metadata if isinstance(ttc_metadata, dict) else {}
        rule_id = str(result.get("engineering_rule_id", ""))
        override_ids = {item.rule_id for item in (structured.controllability_overrides if structured else ())}
        decision_status = str(result.get("decision_status", ""))
        policy = str(result.get("unknown_override_policy", ""))
        action = str(result.get("unknown_policy_action", ""))
        match_states = list(result.get("rule_match_states", []))
        unresolved = decision_status == "METHOD_BRANCH_UNRESOLVED"
        if unresolved and action == "UNSPECIFIED_BLOCKED":
            action = "NO_TRANSITION_DEFINED"
        ttc_state = (
            "TTC_NOT_CLOSING" if (
                isinstance(closing_speed, (int, float)) and closing_speed <= 0
                or closing_speed is None
                and isinstance(relative_speed, (int, float)) and relative_speed <= 0
            )
            else "AVAILABLE" if ttc is not None else "TTC_MISSING"
        )
        pending = result.get("reasoning", "") if _status(result) != CalculationStatus.FINALIZED.value else ""
        return {
            "executor": "StructuredControllabilityExecutor",
            "selected_profile": profile.profile_id if profile else "",
            "profile_id": profile.profile_id if profile else "",
            "source_status": "CONFIRMED" if profile is not None else "UNRESOLVED",
            "status": _status(result),
            "decision_tree_stage": "OVERRIDE" if decision_status.startswith("OVERRIDE") or rule_id in override_ids else "TTC" if decision_status.startswith("TTC") or rule_id else "UNRESOLVED",
            "decision_status": decision_status,
            "unknown_override_policy": policy,
            "unknown_policy_action": action,
            "rule_match_states": match_states,
            "decision_inputs": list(result.get("inputs_used", [])) if unresolved else [],
            "unresolved_inputs": [
                field for field in result.get("inputs_used", [])
                if scenario.get(field) is None
            ] if unresolved else [],
            "ttc_execution_outcome": (
                "NOT_ENTERED_METHOD_BRANCH_UNRESOLVED" if unresolved
                else "ENTERED" if decision_status.startswith("TTC")
                else "NOT_ENTERED_OVERRIDE_MATCH" if rule_id in override_ids
                else "NOT_ENTERED"
            ),
            "override_inputs": {
                key: scenario.get(key)
                for key in ("driver_in_vehicle", "remote_intervention_available", "other_road_user_avoidance_possible")
            },
            "inputs": {
                key: scenario.get(key)
                for key in ("driver_in_vehicle", "remote_intervention_available", "other_road_user_avoidance_possible")
            },
            "risk_context_fields": [
                "driver_in_vehicle", "remote_intervention_available",
                "other_road_user_avoidance_possible", "relative_distance_m",
                "relative_speed_kph", "ttc_s",
            ],
            "derived_ttc": {"relative_distance_m": distance, "relative_speed_kph": relative_speed, "closing_speed_kph": closing_speed, "closing_speed_status": ttc_state, "ttc_s": ttc, "source": "DERIVED_PHYSICS" if ttc is not None else "", "formula_identity": ttc_metadata.get("formula_identity", TTC_FORMULA_IDENTITY), "risk_context_field": "ttc_s"},
            "override": {"evaluated": True, "matched": rule_id in override_ids, "rule_id": rule_id if rule_id in override_ids else ""},
            "override_resolution": "MATCHED" if rule_id in override_ids else action or "NOT_RECORDED",
            "ttc_branch_eligible": decision_status.startswith("TTC") and bool(ttc is not None),
            "ttc_band": rule_id if rule_id and rule_id not in override_ids else "",
            "rule_ids": [rule_id] if rule_id else [], "result": result.get("controllability_score", ""),
            "pending_reason": pending or ("PENDING_CONTROLLABILITY_INPUT" if ttc is None and decision_status.startswith("TTC") else ""),
        }

    def _asil_trace(self, risk: Any | None) -> dict[str, Any]:
        if risk is None:
            return {"executor": "MethodContractASILService", "status": "NOT_REACHED", "result": "", "pending_reason": "NOT_REACHED"}
        values = (risk.severity.value, risk.exposure.value, risk.controllability.value)
        valid = values[0] in {"S0", "S1", "S2", "S3"} and values[1] in {"E0", "E1", "E2", "E3", "E4"} and values[2] in {"C0", "C1", "C2", "C3"}
        return {
            "executor": "MethodContractASILService", "status": "FINALIZED" if risk.asil.status.value == "FINALIZED" else "PENDING_UPSTREAM_RISK_VALUE",
            "S": values[0], "E": values[1], "C": values[2], "matrix_key": "_".join(values) if valid else "",
            "matrix_rule": risk.asil.sources[0].location if risk.asil.sources else "", "result": risk.asil.value,
            "pending_reason": "" if risk.asil.status.value == "FINALIZED" else "missing=" + ",".join(name for name, value, allowed in (("S", values[0], {"S0", "S1", "S2", "S3"}), ("E", values[1], {"E0", "E1", "E2", "E3", "E4"}), ("C", values[2], {"C0", "C1", "C2", "C3"})) if value not in allowed),
        }

    def project(
        self, *, run_id: str, assessments: Iterable[dict[str, Any]], candidates: Iterable[Any],
        committed: bool, scored: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] | None = None,
        risks: Iterable[Any] = (), binding_gaps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        candidate_by_id: dict[str, Any] = {}
        for item in candidates:
            scenario_id = (
                str(item.get("scenario_id", ""))
                if isinstance(item, dict) else str(getattr(item, "scenario_id", ""))
            )
            if scenario_id:
                candidate_by_id[scenario_id] = item
        risk_by_pair = {(item.malfunction_id, item.scenario_id): item for item in risks}
        scored = scored or {}
        binding_gaps = binding_gaps or []
        rows = []
        for record in assessments:
            malfunction_id = str(record.get("malfunction_id", ""))
            scenario_id = str(record.get("scenario_id", ""))
            lifecycle = self._eligibility(record, committed=committed)
            pair = (malfunction_id, scenario_id)
            scored_facts = scored[pair][3] if pair in scored else {}
            context = scored_facts.get("_hazardous_event_risk_context") if isinstance(scored_facts, dict) else None
            row = {
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "hazardous_event_id": (
                    context.hazardous_event_id if context is not None
                    else str(record.get("hazardous_event_id", ""))
                ),
                **lifecycle,
                "risk_scoring_invoked": pair in scored,
            }
            if pair in scored:
                severity, exposure, controllability, facts = scored[pair]
                row.update({
                    "hazardous_event_risk_context": (
                        context.to_dict() if context is not None else {}
                    ),
                    "severity": self._severity_trace(facts, severity),
                    "exposure": self._exposure_trace(facts, exposure, binding_gaps),
                    "controllability": self._controllability_trace(facts, controllability),
                    "asil": self._asil_trace(risk_by_pair.get(pair)),
                })
            rows.append(row)
        invoked = sum(bool(item["risk_scoring_invoked"]) for item in rows)
        eligible = sum(item["risk_eligibility"]["status"] == "ELIGIBLE" for item in rows)
        stage = "COMPLETED" if invoked and invoked == eligible else "PARTIAL" if invoked else "NOT_REACHED"
        return {
            "artifact_version": "risk-execution-trace-v1", "run_id": run_id,
            "method_contract_hash": self.method_contract_hash, "risk_stage_status": stage,
            "scenario_eligibility_summary": {
                "persisted": len(rows), "committed": len(rows) if committed else 0,
                "eligible": eligible, "ineligible": sum(item["risk_eligibility"]["status"] == "INELIGIBLE_CAUSAL" for item in rows),
                "pending_feasibility": sum(item["risk_eligibility"]["status"] == "PENDING_FEASIBILITY" for item in rows),
                "not_committed": sum(item["risk_eligibility"]["status"] == "NOT_COMMITTED" for item in rows),
                "risk_scoring_invoked": invoked,
            }, "assessments": rows,
        }

    def project_review_run(self, reader: Any) -> dict[str, Any]:
        records = reader.read_all()
        return self.project(
            run_id=reader.run_id, assessments=records["scenario_feasibility"],
            candidates=records["scenario_candidate"], committed=False,
        )
