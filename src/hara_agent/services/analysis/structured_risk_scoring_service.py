from __future__ import annotations

from typing import Any
from .scenario_physics import source_is_accepted_for

from hara_agent.contracts import (
    CalculationStatus, ExposureDimensionCoverageDecision,
    ExposureDimensionCoverageStatus, MethodContract, SeveritySemanticResolution,
)

from .deterministic_risk_executor import (
    ExposureMethodExecutor, SeverityMethodExecutor,
    StructuredControllabilityExecutor,
)
from .risk_calculation_input_service import RiskCalculationInputService
from .exposure_dimension_coverage_service import ExposureDimensionCoverageService
from .exposure_input_readiness_service import (
    ExposureInputReadinessService, READY_STATUSES,
)
from .severity_delta_v_semantic_audit_service import SeverityDeltaVSemanticAuditService


class StructuredRiskScoringService:
    """Capability-driven structured S/E/C execution over MethodContract data."""

    def __init__(self, method: MethodContract):
        if method.structured_risk_method is None:
            raise ValueError("StructuredRiskScoringService requires a structured risk method")
        self.method = method
        self.structured = method.structured_risk_method
        self.inputs = RiskCalculationInputService()
        self.coverage = ExposureDimensionCoverageService(method)
        self.severity_authority = SeverityDeltaVSemanticAuditService(method)
        self.severity = SeverityMethodExecutor()
        self.exposure = ExposureMethodExecutor()
        self.exposure_readiness = ExposureInputReadinessService(method, self.exposure)
        self.controllability = StructuredControllabilityExecutor()

    def _result(
        self, *, value_key: str, value: str, status: CalculationStatus,
        reason: str, rule_id: str, source_ref, inputs_used: tuple[str, ...],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            value_key: value,
            "reasoning": reason,
            "engineering_status": (
                "FINALIZED" if status is CalculationStatus.FINALIZED else "PENDING"
            ),
            "calculation_status": status.value,
            "engineering_source_type": "method_contract",
            "engineering_source": self.structured.method_source_hash,
            "engineering_location": f"{source_ref.sheet}!{source_ref.range}",
            "engineering_excerpt": source_ref.raw_text,
            "engineering_rule_id": rule_id,
            "engineering_rule_version": (
                f"{self.method.contract_version}/{self.method.compiler_version}"
            ),
            "engineering_basis": reason,
            "inputs_used": list(inputs_used),
            "missing_fact_types": [],
            "fact_sources": [],
            "_source": "structured_method_contract_executor",
            **extra,
        }

    @staticmethod
    def _controllability_extra(judgement) -> dict[str, Any]:
        return {
            "decision_status": judgement.decision_status,
            "unknown_override_policy": judgement.unknown_override_policy.value,
            "unknown_policy_action": judgement.unknown_policy_action,
            "rule_match_states": [
                {"rule_id": rule_id, "state": state.value}
                for rule_id, state in judgement.rule_match_states
            ],
            "executor_invoked": True,
        }

    @staticmethod
    def _coverage_decision(scenario: dict[str, Any]) -> ExposureDimensionCoverageDecision:
        value = scenario.get("_exposure_dimension_coverage_decision")
        if not isinstance(value, ExposureDimensionCoverageDecision):
            raise ValueError(
                "Structured Risk scoring requires an ExposureDimensionCoverageDecision"
            )
        return value

    def _coverage_gated_exposure(
        self, scenario: dict[str, Any], decision: ExposureDimensionCoverageDecision,
    ) -> dict[str, Any] | None:
        source = self.structured.exposure.source_refs[0]
        if self.structured.exposure.aggregation_policy.policy_id == "fusa_v1":
            # The governed fusa_v1 policy compiles the original executor
            # semantics: coverage is useful review evidence, but is not an E
            # input and cannot replace the original atom-level decision.
            return None
        common = {
            "coverage_status": decision.coverage_status.value,
            "coverage_rule_ids": list(decision.coverage_rule_ids),
            "coverage_granularity": decision.granularity,
            "executor_invoked": False,
            "coverage_gate_applied": True,
        }
        if decision.coverage_status is not ExposureDimensionCoverageStatus.RESOLVED:
            return self._result(
                value_key="exposure_score", value="",
                status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                reason=(
                    "Exposure dimension coverage is not resolved by an approved "
                    "MethodContract rule; scenario atoms are not scored."
                ),
                rule_id="", source_ref=source,
                inputs_used=("component_category", "scenario_atom_ids"),
                exposure_method="", pending_reason="PENDING_METHOD_SEMANTICS",
                missing_method_semantics="EXPOSURE_DIMENSION_COVERAGE",
                exposure_result=None,
                **common,
            )
        bindings = scenario.get("method_scenario_dimensions", {})
        if not isinstance(bindings, dict):
            bindings = {}
        raw_ids = scenario.get("scenario_atom_ids", ())
        atom_ids = {
            str(item) for item in raw_ids
        } if isinstance(raw_ids, (list, tuple)) else set()
        category = str(scenario.get("component_category", "")).strip()
        domain_resolved = sum(
            category in rule.component_categories
            for rule in self.structured.exposure.domain_rules
        ) == 1
        readiness = self.coverage.readiness(
            decision, bindings=bindings, scenario_atom_ids=atom_ids,
            component_domain_resolved=domain_resolved,
        )
        if readiness["exposure_input_status"] != "READY":
            return self._result(
                value_key="exposure_score", value="",
                status=CalculationStatus.PENDING_INPUT,
                reason="Exposure required-dimension atom binding is incomplete.",
                rule_id="", source_ref=source,
                inputs_used=("component_category", "scenario_atom_ids"),
                exposure_method="", pending_reason="PENDING_INPUT",
                missing_required_dimensions=readiness["missing_required_dimensions"],
                exposure_result=None,
                **common,
            )
        return None

    def score(self, scenario: dict[str, Any], hazard_event: str) -> dict[str, dict[str, Any]]:
        if "collision_branches" in scenario:
            return self._score_collision_branches(scenario, hazard_event)
        malfunction_id = str(scenario.get("malfunction_id", "UNKNOWN"))
        scenario_id = str(scenario.get("scenario_id", "UNKNOWN"))
        severity_method = self.structured.severity
        if (
            severity_method.semantic.semantic_resolution
            is SeveritySemanticResolution.APPROVED_SOURCE_INTERNAL_CONFLICT
        ):
            s = {
                "value": "", "status": CalculationStatus.PENDING_METHOD_SEMANTICS,
                "reason": (
                    "Confirmed Severity sources contain an unresolved internal semantic conflict; "
                    "the Severity executor was not invoked."
                ),
                "rule_id": "", "source_ref": severity_method.source_ref,
                "inputs_used": (), "executor_invoked": False,
                "pending_reason": "APPROVED_SOURCE_SEMANTIC_CONFLICT",
            }
        else:
            authority = self.severity_authority.input_authority(scenario)
            if authority.status.value not in {
                "METHOD_CONFIRMED_INPUT_DIRECT", "METHOD_CONFIRMED_DERIVATION",
            }:
                s = {
                    "value": "", "status": CalculationStatus.PENDING_INPUT,
                    "reason": authority.reason,
                    "rule_id": "", "source_ref": severity_method.source_ref,
                    "inputs_used": (), "executor_invoked": False,
                    "pending_reason": authority.status.value,
                }
            else:
                severity_input = self.inputs.severity(
                    malfunction_id, scenario_id, scenario, severity_method.speed_semantic,
                )
                s = {
                    **self.severity.lookup(severity_input, severity_method),
                    "executor_invoked": True,
                }
        severity = self._result(
            value_key="severity_score", value=s["value"], status=s["status"],
            reason=s["reason"], rule_id=s["rule_id"], source_ref=s["source_ref"],
            inputs_used=s["inputs_used"],
            executor_invoked=s["executor_invoked"],
            pending_reason=s.get("pending_reason", ""),
        )
        coverage_decision = self._coverage_decision(scenario)
        gated_exposure = self._coverage_gated_exposure(scenario, coverage_decision)
        if gated_exposure is not None:
            c_input = self.inputs.controllability(malfunction_id, scenario_id, scenario)
            c = self.controllability.lookup(c_input, self.structured)
            controllability = self._result(
                value_key="controllability_score", value=c.value, status=c.status,
                reason=c.reason, rule_id=c.rule_id,
                source_ref=self.structured.controllability_profile.source_ref,
                inputs_used=c.inputs_used, selected_profile_id=c.profile_id,
                **self._controllability_extra(c),
            )
            return {
                "severity": severity,
                "exposure": gated_exposure,
                "controllability": controllability,
            }

        readiness = self.exposure_readiness.assess(scenario)
        if readiness["status"] not in READY_STATUSES:
            source = self.structured.exposure.source_refs[0]
            preview = readiness["baseline_exposure"]
            exposure = self._result(
                value_key="exposure_score", value="",
                status=CalculationStatus.PENDING_INPUT,
                reason=readiness["reason_code"], rule_id="", source_ref=source,
                inputs_used=("component_category", "scenario_atom_ids"),
                exposure_method=preview["actual_domain"],
                coverage_status=coverage_decision.coverage_status.value,
                coverage_rule_ids=list(coverage_decision.coverage_rule_ids),
                coverage_granularity=coverage_decision.granularity,
                coverage_gate_applied=False,
                executor_invoked=False,
                exposure_result="",
                requested_domain=preview["requested_domain"],
                actual_domain=preview["actual_domain"],
                dimension_fallback=bool(preview["dimension_fallback"]),
                atom_details=list(preview["atom_details"]), aggregation_rule="",
                coupling_consumed=False, coupling="",
                pending_reason=readiness["reason_code"],
                exposure_input_readiness=readiness,
            )
            c_input = self.inputs.controllability(malfunction_id, scenario_id, scenario)
            c = self.controllability.lookup(c_input, self.structured)
            controllability = self._result(
                value_key="controllability_score", value=c.value, status=c.status,
                reason=c.reason, rule_id=c.rule_id,
                source_ref=self.structured.controllability_profile.source_ref,
                inputs_used=c.inputs_used, selected_profile_id=c.profile_id,
                **self._controllability_extra(c),
            )
            return {
                "severity": severity,
                "exposure": exposure,
                "controllability": controllability,
            }

        e = self.exposure.lookup(scenario, self.structured.exposure)
        exposure = self._result(
            value_key="exposure_score", value=e["value"], status=e["status"],
            reason=e["reason"], rule_id=e["rule_id"], source_ref=e["source_ref"],
            inputs_used=e["inputs_used"], exposure_method=e["domain"],
            coverage_status=coverage_decision.coverage_status.value,
            coverage_rule_ids=list(coverage_decision.coverage_rule_ids),
            coverage_granularity=coverage_decision.granularity,
            coverage_gate_applied=False,
            executor_invoked=True,
            exposure_result=e["value"],
            requested_domain=e.get("requested_domain", ""),
            actual_domain=e.get("actual_domain", ""),
            dimension_fallback=bool(e.get("dimension_fallback", False)),
            atom_details=list(e.get("atom_details", [])),
            aggregation_rule=e.get("aggregation_rule", ""),
            coupling_consumed=bool(e.get("coupling_consumed", False)),
            coupling=e.get("coupling", ""),
            pending_reason=e.get("pending_reason", ""),
            exposure_input_readiness=readiness,
        )
        c_input = self.inputs.controllability(malfunction_id, scenario_id, scenario)
        c = self.controllability.lookup(c_input, self.structured)
        c_source = self.structured.controllability_profile.source_ref
        for item in self.structured.controllability_overrides:
            if item.rule_id == c.rule_id:
                c_source = item.source_ref
        for item in self.structured.controllability_profile.bands:
            if item.rule_id == c.rule_id:
                c_source = item.source_ref
        controllability = self._result(
            value_key="controllability_score", value=c.value, status=c.status,
            reason=c.reason, rule_id=c.rule_id, source_ref=c_source,
            inputs_used=c.inputs_used, selected_profile_id=c.profile_id,
            **self._controllability_extra(c),
        )
        controllability["avoidance_paths"] = {
            "braking": {"profile_id": c.profile_id, "status": c.status.value, "result": c.value},
            "steering": {
                "status": "UNAVAILABLE_FAILED_CONTROL" if scenario.get("steering_control_available") is False
                          else "PENDING_METHOD_SEMANTICS",
                "reason": "Failed steering cannot be an avoidance means." if scenario.get("steering_control_available") is False
                          else "Active MethodContract has no approved steering avoidance rule; no lateral TTC bands inferred.",
            },
        }
        return {"severity": severity, "exposure": exposure, "controllability": controllability}

    def _score_collision_branches(self, scenario: dict, hazard_event: str) -> dict:
        """Evaluate a finite, evidenced set through this same S/E/C chain.

        Breakpoints partition a bounded speed interval into constant table
        regions. Evaluate each boundary and each open region, including gaps;
        endpoints alone do not prove invariance for a nonmonotone table.
        """
        branches = scenario["collision_branches"]
        base = {key: value for key, value in scenario.items() if key != "collision_branches"}
        policy = self.method.metadata.get("project_analysis_policy", {}).get("severity_evaluation", {})
        metadata = scenario.get("_fact_provenance", {}).get("collision_branches", {})
        accepted = source_is_accepted_for(metadata,
            malfunction_id=str(scenario.get("malfunction_id", "")), scenario_id=str(scenario.get("scenario_id", "")))
        rows, failures = [], []
        if policy.get("status") != "CONFIRMED_FOR_CURRENT_PROJECT" or not accepted or not isinstance(branches, list) or not branches:
            failures.append("EMPTY_OR_UNAPPROVED_COLLISION_BRANCH_SET")
        else:
            for branch in branches:
                if isinstance(branch, dict) and branch.get("physically_possible") is False:
                    continue
                if (not isinstance(branch, dict) or not branch.get("branch_id")
                        or branch.get("mechanism_id") != scenario.get("harm_mechanism_id")
                        or not branch.get("mechanism_id") or not branch.get("impact_time_id")
                        or branch.get("physically_possible") is not True):
                    failures.append("BRANCH_MECHANISM_OR_PHYSICS_UNRESOLVED")
                    continue
                view = {**base, **branch}
                view["_fact_provenance"] = {**base.get("_fact_provenance", {}), **branch.get("_fact_provenance", {})}
                interval = branch.get("relative_speed_range_kph")
                points = [branch.get("relative_speed_kph")]
                if interval is not None:
                    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                            or any(self.inputs._number(value) is None for value in interval)
                            or interval[0] > interval[1]):
                        failures.append("INVALID_FINITE_SPEED_INTERVAL")
                        continue
                    low, high = map(float, interval)
                    cuts = sorted({low, high, *(float(bound) for band in self.structured.severity.bands
                        for bound in (band.lower_kph, band.upper_kph)
                        if bound is not None and low <= bound <= high)})
                    points = sorted(set(cuts + [(left + right) / 2 for left, right in zip(cuts, cuts[1:])]))
                    view["_fact_provenance"]["relative_speed_kph"] = view["_fact_provenance"].get("relative_speed_range_kph", {})
                for point in points:
                    result = self.score({**view, "relative_speed_kph": point}, hazard_event)
                    rows.append({"branch_id": branch["branch_id"], "impact_time_id": branch["impact_time_id"],
                                 "relative_speed_kph": point, "scores": result})
                    if result["severity"]["engineering_status"] != "FINALIZED":
                        failures.append("BRANCH_SEVERITY_UNRESOLVED")
        if failures or not rows:
            result = self.score(base, hazard_event)
            result["severity"].update(severity_score="", engineering_status="PENDING",
                calculation_status="PENDING_INPUT", pending_reason="COLLISION_BRANCH_SET_UNRESOLVED",
                reasoning="; ".join(sorted(set(failures))), executor_invoked=False)
        else:
            levels = list(self.method.asil.severity_levels)
            deciding = max(rows, key=lambda row: levels.index(row["scores"]["severity"]["severity_score"]))
            result = deciding["scores"]
            # E/C belong to the same selected physical branch; retain all tied
            # cases in the trace rather than borrowing a C from another object.
            result["severity"]["deciding_branch_id"] = deciding["branch_id"]
            result["severity"]["invariant"] = len({row["scores"]["severity"]["severity_score"] for row in rows}) == 1
            tied = [row for row in rows if row["scores"]["severity"]["severity_score"] == result["severity"]["severity_score"]]
            for dimension in ("exposure", "controllability"):
                score_key = dimension + "_score"
                if len({(row["scores"][dimension][score_key], row["scores"][dimension]["engineering_status"]) for row in tied}) > 1:
                    result[dimension] = {**result[dimension], score_key: "", "engineering_status": "PENDING",
                        "calculation_status": "PENDING_INPUT", "pending_reason": "DISTINCT_BRANCH_RISK_CONDITIONS",
                        "reasoning": "Equal maximum S branches have different E/C; retain separate branch evaluations."}
        result["severity"]["branch_evaluations"] = [
            {"branch_id": row["branch_id"], "impact_time_id": row["impact_time_id"],
             "relative_speed_kph": row["relative_speed_kph"],
             "S": row["scores"]["severity"]["severity_score"],
             "E": row["scores"]["exposure"]["exposure_score"],
             "C": row["scores"]["controllability"]["controllability_score"],
             "status": row["scores"]["severity"]["calculation_status"]} for row in rows]
        return result
