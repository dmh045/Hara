"""Build source-grounded Risk input context without reading Hazardous Event prose."""

from __future__ import annotations

from collections import Counter
from typing import Any

from hara_agent.contracts import (
    ControllabilityFactState, HazardousEventRiskContext, HazardousEventRiskFact,
    MethodContract, RiskContextFactAuthority, RiskContextFactStatus,
    RuleMatchState, UnknownOverridePolicy,
)
from hara_agent.models import ReviewStatus, ScenarioCandidate
from hara_agent.services.analysis.scenario_physics import (
    derive_scenario_physics, has_analysis_assumption_lineage,
    source_has_provenance, source_is_accepted_for, source_origin, source_reference,
)
from hara_agent.services.analysis.risk_vocabulary_adapter import RiskVocabularyAdapter


class HazardousEventRiskContextService:
    """One canonical source/provenance boundary for Severity and Controllability."""

    _FIELDS = (
        "road_user_type", "collision_type", "ego_speed_kph", "object_speed_kph",
        "relative_speed_kph", "impact_speed_kph", "relative_distance_m", "ttc_s",
        "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible", "direct_control_available",
        "vehicle_stability", "emergency_braking_available", "function_type",
        "has_remote_app",
    )
    _NUMERIC = {
        "ego_speed_kph", "object_speed_kph", "relative_speed_kph",
        "impact_speed_kph", "relative_distance_m", "ttc_s",
    }
    _BOOLEAN = {
        "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible", "direct_control_available",
        "emergency_braking_available", "has_remote_app",
    }
    _SCORING_FIELDS = {
        "road_user_type", "collision_type", "relative_speed_kph",
        "relative_distance_m", "ttc_s", "driver_in_vehicle",
        "remote_intervention_available", "other_road_user_avoidance_possible",
        "direct_control_available", "vehicle_stability", "function_type",
        "has_remote_app",
    }

    def __init__(self, method: MethodContract):
        if method.structured_risk_method is None:
            raise ValueError("HazardousEventRiskContextService requires structured risk method")
        self.method = method
        self.structured = method.structured_risk_method
        self.risk_vocabulary = RiskVocabularyAdapter(method)

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        return float(value)

    @staticmethod
    def _authority(metadata: dict[str, Any]) -> RiskContextFactAuthority:
        value = str(metadata.get("provenance", "")).upper()
        if has_analysis_assumption_lineage(metadata):
            return (
                RiskContextFactAuthority.DERIVED_PHYSICS
                if value == "DERIVED"
                else RiskContextFactAuthority.DIRECT_SCENARIO_FACT
            )
        if metadata.get("source_binding_kind") == "METHOD_RISK_FACT_BINDING":
            # The binding has a typed identity; a method hash alone is not
            # evidence that an arbitrary physics derivation is a project fact.
            return RiskContextFactAuthority.DIRECT_PROJECT_FACT
        if value in {"PROJECT_INPUT", "HUMAN_CONFIRMATION"}:
            return RiskContextFactAuthority.DIRECT_PROJECT_FACT
        if value in {"SCENARIO_INPUT", "DIRECT_SCENARIO_FACT"}:
            return RiskContextFactAuthority.DIRECT_SCENARIO_FACT
        if value in {"METHOD_CONTRACT", "METHOD_DEFINED", "APPROVED_RULE"}:
            return RiskContextFactAuthority.METHOD_RULE
        if value == "DERIVED":
            return RiskContextFactAuthority.DERIVED_PHYSICS
        if value == "METHOD_RULE":
            return RiskContextFactAuthority.METHOD_RULE
        return RiskContextFactAuthority.UNAVAILABLE

    def _fact(
        self, field: str, scenario: dict[str, Any], *, malfunction_id: str,
        scenario_id: str,
    ) -> HazardousEventRiskFact:
        value = scenario.get(field)
        metadata = scenario.get("_fact_provenance", scenario.get("fact_provenance", {}))
        metadata = metadata.get(field, {}) if isinstance(metadata, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        source_ref = source_reference(metadata)
        authority = self._authority(metadata)
        if field in self._NUMERIC:
            value = self._number(value)
        elif field in self._BOOLEAN:
            value = value if isinstance(value, bool) else None
        elif not isinstance(value, str) or not value.strip():
            value = None
        elif field in {"road_user_type", "collision_type"}:
            resolution = self.risk_vocabulary.resolve(field=field, raw_value=value)
            value = resolution.canonical_value if resolution.mapped else None
        if value is None:
            return HazardousEventRiskFact(
                status=RiskContextFactStatus.UNAVAILABLE,
                reason="MISSING_STRUCTURED_FACT",
            )
        if not source_is_accepted_for(
            metadata, malfunction_id=malfunction_id, scenario_id=scenario_id,
        ):
            return HazardousEventRiskFact(
                status=RiskContextFactStatus.UNAVAILABLE,
                reason=(
                    "MISSING_SOURCE_PROVENANCE"
                    if not source_has_provenance(metadata)
                    else "SOURCE_NOT_FINALIZED"
                ),
            )
        if authority is RiskContextFactAuthority.UNAVAILABLE or not source_ref:
            return HazardousEventRiskFact(
                status=RiskContextFactStatus.UNAVAILABLE,
                reason="MISSING_SOURCE_PROVENANCE",
            )
        return HazardousEventRiskFact(
            status=RiskContextFactStatus.AVAILABLE,
            value=value,
            source_type=authority,
            source_ref=source_ref,
            source_provenance=source_origin(metadata),
            derivation_rule_id=str(metadata.get("derivation_rule_id", "")),
            reason="SOURCE_GROUNDED_STRUCTURED_FACT",
        )

    def build(
        self, *, malfunction_id: str, scenario_id: str, hazard_node_id: str,
        scenario: dict[str, Any],
    ) -> HazardousEventRiskContext:
        if not hazard_node_id:
            raise ValueError("HazardousEventRiskContext requires a canonical hazard node identity")
        values = {
            field: self._fact(
                field, scenario, malfunction_id=malfunction_id,
                scenario_id=scenario_id,
            )
            for field in self._FIELDS
        }
        relative_speed = values["relative_speed_kph"]
        if (
            relative_speed.status is RiskContextFactStatus.AVAILABLE
            and (
                float(relative_speed.value) <= 0
                or scenario.get("closing_speed_kph") == 0
            )
            and values["ttc_s"].status is not RiskContextFactStatus.AVAILABLE
        ):
            values["ttc_s"] = HazardousEventRiskFact(
                status=RiskContextFactStatus.TTC_NOT_CLOSING,
                reason="TTC_NOT_CLOSING",
            )
        return HazardousEventRiskContext(
            malfunction_id=malfunction_id,
            scenario_id=scenario_id,
            hazardous_event_id=f"HE::{malfunction_id}::{scenario_id}::{hazard_node_id}",
            **values,
        )

    @classmethod
    def validate_source_conflicts(
        cls, scenario: dict[str, Any], incoming_values: dict[str, Any],
    ) -> None:
        """Reject conflicting Scenario and ProjectFact values before merge."""
        for field, incoming in incoming_values.items():
            if field not in cls._FIELDS or field not in scenario:
                continue
            current = scenario[field]
            equal = (
                str(current).strip().casefold() == str(incoming).strip().casefold()
                if isinstance(current, str) and isinstance(incoming, str)
                else current == incoming
            )
            if not equal:
                raise ValueError(
                    "FACT_SOURCE_CONFLICT: "
                    f"field={field!r} scenario_value={current!r} "
                    f"project_value={incoming!r}"
                )

    def scoring_facts(
        self, context: HazardousEventRiskContext, scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the existing scenario view with S/C fields gated by typed context."""
        result = dict(scenario)
        for field in self._SCORING_FIELDS:
            fact = getattr(context, field)
            if fact.status is RiskContextFactStatus.AVAILABLE:
                result[field] = fact.value
            else:
                result.pop(field, None)
        result["_hazardous_event_risk_context"] = context
        return result

    def severity_readiness(self, context: HazardousEventRiskContext) -> dict[str, Any]:
        method = self.structured.severity
        missing = []
        if context.relative_speed_kph.status is not RiskContextFactStatus.AVAILABLE:
            missing.append("MISSING_RELATIVE_SPEED")
        group = ""
        if context.road_user_type.status is RiskContextFactStatus.AVAILABLE:
            group = dict(method.road_user_groups).get(str(context.road_user_type.value).upper(), "")
        if not group:
            missing.append("MISSING_ROAD_USER_TYPE")
        collision_required = group == "vehicle"
        if collision_required:
            collision = ""
            if context.collision_type.status is RiskContextFactStatus.AVAILABLE:
                collision = dict(method.collision_types).get(str(context.collision_type.value).upper(), "")
            if not collision:
                missing.append("MISSING_COLLISION_TYPE")
        return {
            "status": "READY" if not missing else "PENDING_INPUT",
            "missing_reasons": missing,
            "selected_speed_semantic": method.speed_semantic.value,
            "collision_type_requirement": (
                "REQUIRED" if collision_required else "NOT_REQUIRED_BY_METHOD"
            ),
        }

    def controllability_readiness(self, context: HazardousEventRiskContext) -> dict[str, Any]:
        states = {
            field: (
                ControllabilityFactState.TRUE if getattr(context, field).value is True
                else ControllabilityFactState.FALSE if getattr(context, field).value is False
                else ControllabilityFactState.UNKNOWN
            ) if getattr(context, field).status is RiskContextFactStatus.AVAILABLE
            else ControllabilityFactState.UNKNOWN
            for field in (
                "driver_in_vehicle", "remote_intervention_available",
                "other_road_user_avoidance_possible",
            )
        }
        unknown_fields: list[str] = []
        for rule in sorted(self.structured.controllability_overrides, key=lambda item: item.priority):
            conditions = rule.all_of or rule.any_of
            values = [
                state if state is ControllabilityFactState.UNKNOWN else (
                    ControllabilityFactState.TRUE
                    if ((state is ControllabilityFactState.TRUE) is item.expected)
                    else ControllabilityFactState.FALSE
                )
                for item in conditions for state in (states[item.field],)
            ]
            if rule.all_of:
                if ControllabilityFactState.FALSE in values:
                    continue
                if ControllabilityFactState.UNKNOWN in values:
                    unknown_fields.extend(item.field for item in conditions)
                    continue
                return {"status": "READY", "branch": "OVERRIDE", "rule_id": rule.rule_id,
                        "missing_reasons": [], "rule_match_state": RuleMatchState.MATCH.value}
            if ControllabilityFactState.TRUE in values:
                return {"status": "READY", "branch": "OVERRIDE", "rule_id": rule.rule_id,
                        "missing_reasons": [], "rule_match_state": RuleMatchState.MATCH.value}
            if ControllabilityFactState.UNKNOWN in values:
                unknown_fields.extend(item.field for item in conditions)
        policy = self.structured.controllability_branch_policy.unknown_override_policy
        if unknown_fields and policy is UnknownOverridePolicy.UNSPECIFIED:
            return {"status": "PENDING_METHOD_SEMANTICS", "branch": "UNRESOLVED", "rule_id": "",
                    "missing_reasons": ["CONTROLLABILITY_UNKNOWN_BRANCH_POLICY_UNSPECIFIED"],
                    "unknown_override_policy": policy.value, "ttc_eligible": False}
        if unknown_fields and policy is UnknownOverridePolicy.BLOCK_TTC:
            return {"status": "PENDING_INPUT", "branch": "OVERRIDE", "rule_id": "",
                    "missing_reasons": [f"MISSING_{item.upper()}" for item in dict.fromkeys(unknown_fields)],
                    "unknown_override_policy": policy.value, "ttc_eligible": False}
        ttc = context.ttc_s
        if ttc.status is RiskContextFactStatus.AVAILABLE:
            return {"status": "READY", "branch": "TTC", "rule_id": "", "missing_reasons": []}
        return {
            "status": "PENDING_INPUT", "branch": "TTC", "rule_id": "",
            "missing_reasons": [
                "TTC_NOT_CLOSING" if ttc.status is RiskContextFactStatus.TTC_NOT_CLOSING
                else "MISSING_RELATIVE_DISTANCE_OR_CLOSING_SPEED"
            ],
        }

    def input_inventory(self, contexts: list[HazardousEventRiskContext]) -> list[dict[str, Any]]:
        consumers = {
            "relative_speed_kph": ["SeverityMethodExecutor", "StructuredControllabilityExecutor"],
            "road_user_type": ["SeverityMethodExecutor"],
            "collision_type": ["SeverityMethodExecutor"],
            "relative_distance_m": ["StructuredControllabilityExecutor"],
            "ttc_s": ["StructuredControllabilityExecutor"],
            "driver_in_vehicle": ["StructuredControllabilityExecutor"],
            "remote_intervention_available": ["StructuredControllabilityExecutor"],
            "other_road_user_avoidance_possible": ["StructuredControllabilityExecutor"],
        }
        inventory = []
        for field in self._FIELDS:
            facts = [getattr(item, field) for item in contexts]
            available = [item for item in facts if item.status is RiskContextFactStatus.AVAILABLE]
            reasons = Counter(item.reason for item in facts if item.status is not RiskContextFactStatus.AVAILABLE)
            inventory.append({
                "field": field,
                "current_model": "HazardousEventRiskContext",
                "producer": "HazardousEventRiskContextService",
                "source": sorted({item.source_type.value for item in available}) or ["UNAVAILABLE"],
                "consumer": consumers.get(field, []),
                "currently_populated": bool(available),
                "authority": sorted({item.source_type.value for item in available}) or ["UNAVAILABLE"],
                "missing_reason": dict(sorted(reasons.items())),
            })
        return inventory

    @staticmethod
    def _scenario_with_derived_physics(candidate: dict[str, Any]) -> dict[str, Any]:
        """Materialize the same canonical physics facts used by runtime scoring."""
        facts = candidate.get("facts", {})
        facts = dict(facts) if isinstance(facts, dict) else {}
        provenance = candidate.get("fact_provenance", {})
        provenance = dict(provenance) if isinstance(provenance, dict) else {}
        status = ReviewStatus(str(candidate.get("status", ReviewStatus.PENDING.value)))
        typed_candidate = ScenarioCandidate(
            scenario_id=str(candidate.get("scenario_id", "")),
            operating_scenario=str(candidate.get("operating_scenario", "")),
            situational_description=str(candidate.get("situational_description", "")),
            situational_detailing=str(candidate.get("situational_detailing", "")),
            facts=facts,
            operating_mode=str(candidate.get("operating_mode", "")),
            fact_provenance=provenance,
            status=status,
            semantic_fingerprint=str(candidate.get("semantic_fingerprint", "")),
            analysis_instance=dict(candidate.get("analysis_instance", {})),
        )
        scenario = {**facts, "_fact_provenance": provenance}
        for record in derive_scenario_physics(typed_candidate):
            key = record.evidence_ref.split(".", 1)[1]
            if key in scenario and scenario[key] != record.value:
                raise ValueError(
                    "Scenario contains a value that conflicts with deterministic "
                    f"physics: key={key!r}"
                )
            scenario[key] = record.value
            scenario["_fact_provenance"][key] = {
                "provenance": record.provenance.value,
                "approval": record.approval_status.value,
                "source_refs": [
                    {
                        "source_type": source.source_type,
                        "source_id": source.source_id,
                        "location": source.location,
                        "excerpt": source.excerpt,
                    }
                    for source in record.source_refs
                ],
                "evidence_ref": record.evidence_ref,
                **record.metadata,
            }
        return scenario

    def audit(self, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        candidates = {
            str(item.get("scenario_id", "")): item
            for item in records.get("scenario_candidate", []) if isinstance(item, dict)
        }
        contexts = []
        rows = []
        for assessment in records.get("scenario_feasibility", []):
            if not isinstance(assessment, dict) or assessment.get("status") != "FINALIZED":
                continue
            if not all(assessment.get(flag) is True for flag in (
                "physically_feasible", "functionally_relevant", "causally_relevant",
            )):
                continue
            scenario_id = str(assessment.get("scenario_id", ""))
            malfunction_id = str(assessment.get("malfunction_id", ""))
            candidate = candidates.get(scenario_id, {})
            scenario = self._scenario_with_derived_physics(candidate)
            causal = assessment.get("causal_assessment", {})
            chain = causal.get("causal_chain", []) if isinstance(causal, dict) else []
            hazard_node_id = str(chain[-1]) if chain else "UNPROJECTED_HAZARD_NODE"
            context = self.build(
                malfunction_id=malfunction_id, scenario_id=scenario_id,
                hazard_node_id=hazard_node_id, scenario=scenario,
            )
            contexts.append(context)
            rows.append({
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "hazardous_event_id": context.hazardous_event_id,
                "risk_context": context.to_dict(),
                "severity_readiness": self.severity_readiness(context),
                "controllability_readiness": self.controllability_readiness(context),
            })
        severity_reasons = Counter(
            reason for item in rows for reason in item["severity_readiness"]["missing_reasons"]
        )
        controllability_reasons = Counter(
            reason for item in rows for reason in item["controllability_readiness"]["missing_reasons"]
        )
        controllability = [item["controllability_readiness"] for item in rows]
        resolved_by_override = sum(
            item["status"] == "READY" and item["branch"] == "OVERRIDE"
            for item in controllability
        )
        resolved_by_ttc = sum(
            item["status"] == "READY" and item["branch"] == "TTC"
            for item in controllability
        )
        blocked_at_override = sum(
            item["status"] != "READY" and item["branch"] == "OVERRIDE"
            for item in controllability
        )
        return {
            "artifact_version": "hazardous-event-risk-context-audit-v1",
            "method_contract_hash": str(self.method.metadata.get("method_source_hash", "")),
            "runtime_yaml_read": 0,
            "hazardous_event_identity": {
                "status": "PROJECTED_FROM_CANONICAL_CAUSAL_NODE",
                "text_used_as_identity": False,
                "format": "HE::{malfunction_id}::{scenario_id}::{hazard_node_id}",
            },
            "risk_context_substrate": {
                "hazardous_event_prose_consumed": False,
                "object_atom_to_road_user_adapter": "ABSENT",
                "collision_consequence_layer": "ABSENT",
                "potential_harm_resolver": "POST_SEVERITY_HARM_LABEL_ONLY",
            },
            "source_authority_model": [item.value for item in RiskContextFactAuthority],
            "relative_speed_authority": {
                "allowed_sources": [
                    "DIRECT_SCENARIO_FACT: concrete relative_speed_kph",
                    "DERIVED_PHYSICS: approved closing-relative-speed derivation",
                ],
                "prohibited_substitutions": [
                    "EGO_SPEED_ONLY", "ODD_SPEED_CONSTRAINT", "ASSUMED_STATIC_OBJECT",
                ],
                "r3_concrete_relative_speed": sum(
                    item.relative_speed_kph.status is RiskContextFactStatus.AVAILABLE
                    for item in contexts
                ),
            },
            "collision_partner_resolution": {
                "object_atom_to_road_user_adapter": "ABSENT",
                "resolved": sum(
                    item.road_user_type.status is RiskContextFactStatus.AVAILABLE
                    for item in contexts
                ),
            },
            "collision_configuration_resolution": {
                "structured_collision_geometry_layer": "ABSENT",
                "resolved": sum(
                    item.collision_type.status is RiskContextFactStatus.AVAILABLE
                    for item in contexts
                ),
                "collision_only_method_unresolved": sum(
                    item.collision_type.status is not RiskContextFactStatus.AVAILABLE
                    for item in contexts
                ),
            },
            "controllability_input_resolution": {
                "resolved_by_override": resolved_by_override,
                "resolved_by_ttc": resolved_by_ttc,
                "blocked_at_higher_priority_override": blocked_at_override,
                "requires_ttc": sum(item["branch"] == "TTC" for item in controllability),
                "ttc_missing_reasons": dict(sorted(
                    (key, value) for key, value in controllability_reasons.items()
                    if key.startswith("TTC_") or key.startswith("MISSING_RELATIVE_")
                )),
            },
            "fact_source_conflicts": {
                "status": "FAIL_CLOSED_AT_RUNTIME",
                "r3_observed": 0,
            },
            "risk_input_inventory": self.input_inventory(contexts),
            "r3_risk_context": rows,
            "summary": {
                "causal_relevant_hazardous_events": len(rows),
                "severity_ready": sum(item["severity_readiness"]["status"] == "READY" for item in rows),
                "severity_pending_input": sum(item["severity_readiness"]["status"] != "READY" for item in rows),
                "controllability_ready": sum(item["controllability_readiness"]["status"] == "READY" for item in rows),
                "controllability_pending_input": sum(item["controllability_readiness"]["status"] != "READY" for item in rows),
                "severity_missing_reasons": dict(sorted(severity_reasons.items())),
                "controllability_missing_reasons": dict(sorted(controllability_reasons.items())),
                "controllability_resolved_by_override": resolved_by_override,
                "controllability_resolved_by_ttc": resolved_by_ttc,
            },
        }
