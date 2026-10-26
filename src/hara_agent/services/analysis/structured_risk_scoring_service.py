from __future__ import annotations

from typing import Any
from .scenario_physics import source_is_accepted_for, branch_physics_view
from .asil_service import MethodContractASILService
from copy import deepcopy
from dataclasses import asdict, replace
import math

from hara_agent.contracts import (
    CalculationStatus, ExposureDimensionCoverageDecision,
    ExposureDimensionCoverageStatus, MethodContract, SeveritySemanticResolution,
)

from .deterministic_risk_executor import (
    ExposureMethodExecutor, SeverityMethodExecutor,
    StructuredControllabilityExecutor, ControllabilityProfileExecutor,
)
from .risk_calculation_input_service import RiskCalculationInputService
from .exposure_dimension_coverage_service import ExposureDimensionCoverageService
from .exposure_input_readiness_service import (
    ExposureInputReadinessService, READY_STATUSES,
)
from .severity_delta_v_semantic_audit_service import SeverityDeltaVSemanticAuditService
from .hazardous_event_risk_context_service import HazardousEventRiskContextService


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
            "pending_reason": judgement.reason if not judgement.value else "",
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
        result = self._score(scenario, hazard_event)
        analysis = scenario.get("physical_analysis")
        controls = scenario.get("control_analysis")
        if not analysis and not controls:
            return result
        # Numeric table evaluations are retained; engineering confirmation is
        # independent and never manufactured from the proposal's validity.
        conditional_dimensions = (("severity", "controllability", "exposure") if scenario.get("physical_object_exposure_conditional")
            else ("severity", "controllability") if analysis else ("controllability",))
        for dimension in conditional_dimensions:
            result[dimension].update(engineering_status="PENDING", calculation_nature="CONDITIONAL",
                engineering_basis="条件性试算；仅适用于所列运动或干预能力假设，并非已确认车辆能力。")
        if scenario.get("physical_object_exposure_unresolved"):
            result["exposure"].update(exposure_score="", engineering_status="PENDING",
                calculation_status="PENDING_INPUT", pending_reason="ASSUMED_OBJECT_EXPOSURE_BINDING_UNRESOLVED")
        for row in result.get("branch_evaluations", []):
            row.update(status="PENDING", calculation_nature="CONDITIONAL")
            for dimension in conditional_dimensions:
                row["scores"][dimension].update(engineering_status="PENDING", calculation_nature="CONDITIONAL")
        if result.get("branch_summary"):
            result["branch_summary"].update(status="PENDING", calculation_nature="CONDITIONAL",
                vehicle_upper_bound_proven=False, range_invariance_proven=False)
        if analysis:
            result["physical_analysis"] = analysis
        if controls:
            result["control_analysis"] = controls
        return result

    def _score(self, scenario: dict[str, Any], hazard_event: str) -> dict[str, dict[str, Any]]:
        result = self._calculate(scenario, hazard_event)
        control = result["controllability"]
        hypothesis = scenario.get("control_analysis", scenario.get("physical_analysis", {})).get("hypothesis", {})
        failed_controls = hypothesis.get("failed_controls", [])
        braking_path_unresolved = scenario.get("emergency_braking_available") is False or (
            "automatic_braking_command" in failed_controls and (
                "driver_braking" not in hypothesis.get("remaining_controls", [])
                or scenario.get("emergency_braking_available") is not True))
        override_match = control.get("decision_status") == "OVERRIDE_MATCH" and control.get("engineering_rule_id") in {
            rule.rule_id for rule in self.structured.controllability_overrides}
        if (override_match and braking_path_unresolved
                and scenario.get("remote_intervention_available") is True):
            # A remote command does not restore a failed braking actuator.
            # Re-evaluate the existing override with that positive capability
            # unavailable: another independently supported evasion path may
            # still satisfy the same approved rule. No new C rule is added.
            independent = self.controllability.lookup(self.inputs.controllability(
                str(scenario.get("malfunction_id", "UNKNOWN")), str(scenario.get("scenario_id", "UNKNOWN")),
                {**scenario, "remote_intervention_available": None}), self.structured)
            override_match = independent.decision_status == "OVERRIDE_MATCH" and independent.rule_id == control["engineering_rule_id"]
        if not override_match and not braking_path_unresolved and "collision_branches" not in scenario:
            reference = self._conditional_reference_control(scenario)
            if reference is not None:
                control.update(reference)
        reference_match = control.get("decision_status") == "CONDITIONAL_REFERENCE_MATCH"
        moving = scenario.get("fault_motion_model") == "RECTILINEAR_MOVING_TARGET_CONTACT"
        crossing = moving and scenario.get("object_longitudinal_direction") in {"LEFT", "RIGHT"}
        if (scenario.get("fault_motion_model") in {"CONSTANT_LATERAL_ACCELERATION_TO_CONTACT", "CONSTANT_SPEED_CURVATURE_TO_CONTACT"} or crossing) and not (override_match or reference_match):
            result["controllability"].update(controllability_score="", engineering_status="PENDING",
                calculation_status="PENDING_METHOD_SEMANTICS", pending_reason="LATERAL_C_METHOD_NOT_AVAILABLE",
                decision_status="MODEL_NOT_APPLICABLE", engineering_rule_id="",
                reasoning="Lateral first-contact time is not a longitudinal TTC; no applicable lateral C method is compiled.")
        if (moving and not crossing and not (override_match or reference_match)
                and (scenario.get("ego_longitudinal_direction"), scenario.get("object_position"))
                not in {("FORWARD", "FRONT"), ("REVERSE", "REAR")}):
            control.update(controllability_score="", engineering_status="PENDING",
                calculation_status="PENDING_METHOD_SEMANTICS", pending_reason="REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED",
                decision_status="MODEL_NOT_APPLICABLE", engineering_rule_id="",
                reasoning="The target approaches from behind the ego travel direction; ego braking alone cannot establish an avoidance path. The computed contact time is retained, but not used by the braking TTC profile.")
        if scenario.get("physical_object_exposure_unresolved"):
            result["exposure"].update(exposure_score="", engineering_status="PENDING",
                calculation_status="PENDING_INPUT", pending_reason="ASSUMED_OBJECT_EXPOSURE_BINDING_UNRESOLVED")
        if (scenario.get("motion_model_outcome", {}).get("contact_phase") == "HELD_AFTER_STOP"
                and not (override_match or reference_match)
                and control.get("pending_reason") != "REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED"):
            control.update(controllability_score="", engineering_status="PENDING",
                calculation_status="PENDING_METHOD_SEMANTICS", pending_reason="POST_STOP_BRAKING_APPLICABILITY_UNRESOLVED",
                decision_status="MODEL_NOT_APPLICABLE", engineering_rule_id="",
                reasoning="Contact occurs after the ego is held stationary. A braking-time band alone does not establish an avoidance path for that phase.")
        if (braking_path_unresolved
                and result["controllability"].get("selected_profile_id") and not (override_match or reference_match)):
            result["controllability"].update(controllability_score="", engineering_status="PENDING",
                calculation_status="PENDING_METHOD_SEMANTICS", pending_reason="FAILED_CONTROL_PROFILE_DEPENDENCY_UNRESOLVED",
                decision_status="MODEL_DEPENDENCY_UNRESOLVED", engineering_rule_id="",
                reasoning="The failed braking path cannot establish an independent available braking avoidance path.")
        if not control.get("controllability_score") and "collision_branches" not in scenario:
            reference = self._conditional_reference_control(scenario, phase="AFTER_APPLICABLE_METHODS")
            if reference is not None:
                control.update(reference)
        if not control.get("controllability_score"):
            outcome = scenario.get("motion_model_outcome", {})
            if outcome.get("status") in {"NO_CONTACT_WITHIN_MODEL", "MODEL_GEOMETRY_NOT_APPLICABLE"}:
                control.update(calculation_status="PENDING_INPUT", decision_status="PHYSICAL_INPUT_UNRESOLVED",
                    pending_reason="PHYSICAL_CONTACT_UNRESOLVED:" + outcome["status"],
                    reasoning="The modeled trajectory has no established contact within its valid scope. This is not proof of no hazard beyond that scope, nor merely a missing C classification rule.")
            # A rejected lookup is not a finalized avoidance path. Keep its
            # attempted value for audit, separately from the usable result.
            for key in ("braking", "method_override"):
                path = control.get("avoidance_paths", {}).get(key)
                if path:
                    path.update(lookup_result=path.get("result", ""), result="",
                        status=control["calculation_status"], reason=control.get("reasoning", ""))
        return result

    def _conditional_reference_control(self, scenario: dict, *, phase="BEFORE_MODEL_GUARDS") -> dict | None:
        """Execute the selected reference's bounded scope; retain pending approval."""
        from hara_agent.contracts import SourceRef
        rules = self.method.metadata.get("project_analysis_policy", {}).get(
            "controllability", {}).get("conditional_reference_rules", [])
        metadata = scenario.get("_fact_provenance", {})
        scope = {key: scenario.get(key, "") for key in ("scenario_id", "malfunction_id")}
        for rule in rules:
            if rule.get("evaluation_phase", "BEFORE_MODEL_GUARDS") != phase:
                continue
            output = str(scenario.get("_function_output", "")).casefold()
            if rule.get("function_output_terms") and not any(
                    term.casefold() in output for term in rule["function_output_terms"]):
                continue
            if any((scenario.get(key) not in expected if isinstance(expected, list)
                    else scenario.get(key) is not expected) for key, expected in rule["when"].items()):
                continue
            fields = tuple(key for key, value in rule["when"].items() if isinstance(value, bool))
            limit = rule.get("trajectory_speed_upper_exclusive_kph")
            if limit is not None:
                fields += ("trajectory_max_speed_kph",)
            if not all(source_is_accepted_for(metadata.get(key, {}), **scope) for key in fields):
                continue
            speed = scenario.get("trajectory_max_speed_kph")
            if limit is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float))
                    or not math.isfinite(speed) or speed < 0
                    or speed >= limit):
                continue
            if rule.get("requires_contact"):
                outcome = scenario.get("motion_model_outcome", {}).get("status")
                contact_field = next((key for key in ("first_contact_time_s", "ttc_s")
                    if isinstance(scenario.get(key), (int, float)) and not isinstance(scenario[key], bool)
                    and math.isfinite(scenario[key]) and scenario[key] >= 0
                    and source_is_accepted_for(metadata.get(key, {}), **scope)), None)
                if outcome not in (None, "CONTACT") or contact_field is None:
                    continue
                fields += (contact_field,)
            # A conflict is not an absence of credited intervention. Neither a
            # reference classification nor a bound can resolve contradictory facts.
            context = scenario.get("_hazardous_event_risk_context")
            if context and any(getattr(context, key).status.value == "CONFLICT" for key in (
                    "driver_in_vehicle", "emergency_braking_available", "remote_intervention_available",
                    "other_road_user_avoidance_possible")):
                continue
            value = rule["result"]
            profile_trace = {}
            if value == "ACTIVE_PROFILE":
                if scenario.get("motion_model_outcome", {}).get("contact_phase") == "HELD_AFTER_STOP":
                    continue
                time_input = rule["profile_time_input"]
                time_value = scenario.get(time_input)
                if (not isinstance(time_value, (int, float)) or isinstance(time_value, bool)
                        or not math.isfinite(time_value) or time_value < 0
                        or not source_is_accepted_for(metadata.get(time_input, {}), **scope)
                        or scenario.get("motion_model_outcome", {}).get("status") != "CONTACT"):
                    continue
                assessment = replace(self.inputs.controllability(
                    scope["malfunction_id"], scope["scenario_id"], scenario),
                    first_contact_time_s=time_value)
                profile = self.structured.controllability_profile
                judgement = ControllabilityProfileExecutor().lookup(assessment, profile, time_input=time_input)
                if not judgement.value:
                    continue
                value = judgement.value
                fields = tuple(dict.fromkeys((*fields, *judgement.inputs_used)))
                profile_trace = {"profile_id": profile.profile_id, "band_rule_id": judgement.rule_id,
                    "time_input": time_input, "time_value_s": time_value,
                    "profile_source": asdict(profile.source_ref)}
            result = self._result(value_key="controllability_score", value=value,
                status=CalculationStatus.FINALIZED, reason=rule["rationale"],
                rule_id=rule["rule_id"], source_ref=SourceRef.from_dict(rule["source_ref"]),
                inputs_used=(*fields, "_function_output", "component_category", "failure_type", "fault_motion_model"),
                engineering_status="PENDING", calculation_nature="CONDITIONAL",
                decision_status="CONDITIONAL_REFERENCE_MATCH", pending_reason="",
                selected_profile_id=rule["rule_id"], unknown_policy_action="REFERENCE_RULE_MATCH",
                reference_rule={key: rule[key] for key in ("rule_id", "source_document", "source_location",
                    "source_sha256", "source_statement", "limitation", "when", "rationale",
                    "trajectory_speed_upper_exclusive_kph", "evaluation_phase", "requires_contact",
                    "profile_time_input") if key in rule})
            if profile_trace:
                result["reference_rule"]["profile_lookup"] = profile_trace
            if limit is not None:
                result["reference_rule"]["trajectory_max_speed_kph"] = speed
            result["reference_rule"]["matched_inputs"] = {key: scenario.get(key) for key in (*fields, *rule["when"])}
            result["avoidance_paths"] = {"reference_scenario": {
                "rule_id": rule["rule_id"], "status": "CONDITIONAL_TRIAL", "result": value}}
            return result
        return None

    def _calculate(self, scenario: dict[str, Any], hazard_event: str) -> dict[str, dict[str, Any]]:
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
            "method_override" if c.decision_status == "OVERRIDE_MATCH" else "braking": {
                "profile_id": c.profile_id, "rule_id": c.rule_id,
                "status": c.status.value, "result": c.value},
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
                view = branch_physics_view(base, branch)
                interval = branch.get("relative_speed_range_kph")
                points = [view.get("relative_speed_kph")]
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
                    point_view = {**view, **({"relative_speed_kph": point} if point is not None else {})}
                    contexts = HazardousEventRiskContextService(self.method)
                    context = contexts.build(malfunction_id=str(view.get("malfunction_id", "")),
                        scenario_id=str(view.get("scenario_id", "")),
                        hazard_node_id=str(branch["branch_id"]), scenario=point_view)
                    point_view = contexts.scoring_facts(context, point_view)
                    result = self._score(point_view, hazard_event)
                    physical_conflicts = view.get("_physical_input_conflicts", [])
                    if physical_conflicts:
                        for dimension in ("severity", "controllability"):
                            result[dimension] = {**result[dimension], dimension + "_score": "",
                                "engineering_status": "PENDING", "calculation_status": "PENDING_INPUT",
                                "pending_reason": "BRANCH_PHYSICAL_INPUT_CONFLICT",
                                "reasoning": "; ".join(item["reason"] for item in physical_conflicts)}
                    object_changed = (base.get("road_user_type") is not None
                                      and branch.get("road_user_type", base["road_user_type"]) != base["road_user_type"])
                    parent_object_binding = base.get("method_scenario_dimensions", {}).get("OBJECT", {})
                    if (object_changed and parent_object_binding.get("resolution_status") == "RESOLVED"
                            and "scenario_atom_ids" not in branch and "OBJECT" not in branch.get("method_scenario_dimensions", {})):
                        result["exposure"] = {**result["exposure"], "exposure_score": "", "engineering_status": "PENDING",
                            "calculation_status": "PENDING_INPUT", "pending_reason": "BRANCH_OBJECT_EXPOSURE_BINDING_REQUIRED",
                            "reasoning": "Changed collision object cannot inherit a different object's Exposure atom without a binding or invariance proof."}
                    if interval is not None and interval[0] != interval[1] and any("ttc" in key.lower() for key in result["controllability"].get("inputs_used", [])):
                        result["controllability"] = {**result["controllability"], "controllability_score": "",
                            "engineering_status": "PENDING", "calculation_status": "PENDING_INPUT",
                            "pending_reason": "C_INTERVAL_NOT_PROVEN", "reasoning": "S table partition does not prove C over a trajectory interval."}
                    rows.append({"branch_id": branch["branch_id"], "impact_time_id": branch["impact_time_id"],
                                 "relative_speed_kph": point_view.get("relative_speed_kph"), "scores": result,
                                 "first_contact_time_s": point_view.get("first_contact_time_s"),
                                 "risk_context": context.to_dict(),
                                 "physical_input_conflicts": physical_conflicts,
                                 "ttc_s": view.get("ttc_s"), "invalidated_parent_fields": view.get("_invalidated_parent_fields", [])})
                    if result["severity"]["engineering_status"] != "FINALIZED":
                        failures.append("BRANCH_SEVERITY_UNRESOLVED")
        if failures or not rows:
            result = self._score(base, hazard_event)
            result["severity"].update(severity_score="", engineering_status="PENDING",
                calculation_status="PENDING_INPUT", pending_reason="COLLISION_BRANCH_SET_UNRESOLVED",
                reasoning="; ".join(sorted(set(failures))), executor_invoked=False)
        else:
            levels = list(self.method.asil.severity_levels)
            deciding = max(rows, key=lambda row: levels.index(row["scores"]["severity"]["severity_score"]))
            result = deepcopy(deciding["scores"])
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
        asil_service = MethodContractASILService(self.method)
        evaluations = []
        for row in rows:
            calculated = {dimension: score[dimension + "_score"] for dimension, score in row["scores"].items()
                          if dimension in {"severity", "exposure", "controllability"}
                          and (score["engineering_status"] == "FINALIZED" or (
                              score.get("calculation_nature") == "CONDITIONAL"
                              and score.get("calculation_status") == "FINALIZED"))}
            conditional = any(score.get("calculation_nature") == "CONDITIONAL"
                              for score in row["scores"].values() if isinstance(score, dict))
            asil, sources = asil_service.partial_zero_decision(calculated)
            if len(calculated) == 3:
                asil = asil_service.determine(calculated["severity"], calculated["exposure"], calculated["controllability"])
            if row["physical_input_conflicts"]:
                asil = ""
            evaluations.append({"branch_id": row["branch_id"], "impact_time_id": row["impact_time_id"],
                "relative_speed_kph": row["relative_speed_kph"], "ttc_s": row["ttc_s"],
                "first_contact_time_s": row["first_contact_time_s"],
                "invalidated_parent_fields": row["invalidated_parent_fields"],
                "physical_input_conflicts": row["physical_input_conflicts"],
                "risk_context": row["risk_context"],
                "S": row["scores"]["severity"]["severity_score"], "E": row["scores"]["exposure"]["exposure_score"],
                "C": row["scores"]["controllability"]["controllability_score"], "ASIL": asil,
                "status": "FINALIZED" if asil and not conditional else "PENDING", "scores": row["scores"]})
        known = [row for row in evaluations if row["ASIL"]]
        order = asil_service.result_order()
        deciding_risk = max(known, key=lambda row: (order.index(row["ASIL"]), list(self.method.asil.severity_levels).index(row["S"]) if row["S"] else -1)) if known else None
        maximum_s = result["severity"]["severity_score"]
        invariant_s = result["severity"].get("invariant")
        if deciding_risk:
            result = deepcopy(deciding_risk["scores"])
            result["severity"].update(deciding_branch_id=deciding_risk["branch_id"],
                                       maximum_feasible_severity=maximum_s, invariant=invariant_s)
        result["branch_evaluations"] = evaluations
        # Keep legacy diagnostic consumers, with an identical branch list.
        result["severity"]["branch_evaluations"] = evaluations
        result["branch_summary"] = {
            "highest_known_asil": deciding_risk["ASIL"] if deciding_risk else "",
            "deciding_branch_id": deciding_risk["branch_id"] if deciding_risk else "",
            "status": "PENDING" if failures or not evaluations or any(row["status"] != "FINALIZED" for row in evaluations) else "FINALIZED",
            "unresolved_reasons": sorted(set(failures)),
            "severity_maximum": maximum_s,
            "aggregation_basis": "EACH_BRANCH_OWN_SEC_ACTIVE_MATRIX",
        }
        return result
