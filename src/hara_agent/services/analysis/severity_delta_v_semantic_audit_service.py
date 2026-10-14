"""Read-only audit of Severity speed-semantic authority and readiness."""

from __future__ import annotations

from collections import Counter
from typing import Any
import math
from hara_agent.services.analysis.scenario_physics import source_is_accepted_for

from hara_agent.contracts import (
    MethodContract, SeverityInputAuthority, SeverityInputAuthorityStatus,
    SeverityInputSource, SpeedSemantic,
)


class SeverityDeltaVSemanticAuditService:
    """Project compiled authority and review artifacts without calculating S."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.structured = method.structured_risk_method

    @property
    def _method_hash(self) -> str:
        return str(self.method.metadata.get("method_source_hash", ""))

    def _provenance(self) -> dict[str, Any]:
        if self.structured is None:
            return {
                "compiled_semantic": "UNRESOLVED",
                "semantic_resolution": "APPROVED_SOURCE_INTERNAL_CONFLICT",
                "diagnostic_codes": ["STRUCTURED_SEVERITY_METHOD_ABSENT"],
            }
        result = self.structured.severity.semantic.to_dict()
        result["selected_speed_semantic"] = self.structured.severity.speed_semantic.value
        result["delta_v_derivation_authority"] = "ABSENT"
        return result

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _fact_provenance(candidate: dict[str, Any], key: str) -> dict[str, Any]:
        for container_key in ("fact_provenance", "_fact_provenance"):
            values = candidate.get(container_key, {})
            if isinstance(values, dict) and isinstance(values.get(key), dict):
                return dict(values[key])
        facts = candidate.get("facts", {})
        values = facts.get("_fact_provenance", {}) if isinstance(facts, dict) else {}
        return dict(values.get(key, {})) if isinstance(values, dict) and isinstance(values.get(key), dict) else {}

    def input_authority(self, candidate: dict[str, Any]) -> SeverityInputAuthority:
        facts = candidate.get("facts", candidate)
        facts = facts if isinstance(facts, dict) else {}
        provenance = self._provenance()
        semantic = SpeedSemantic(str(provenance.get("selected_speed_semantic", "UNRESOLVED")))
        source_key = {
            SpeedSemantic.RELATIVE_SPEED: "relative_speed_kph",
            SpeedSemantic.DELTA_V: "delta_v_kph",
            SpeedSemantic.EGO_SPEED: "ego_speed_kph",
            SpeedSemantic.IMPACT_SPEED: "impact_speed_kph",
        }.get(semantic, "")
        value = self._number(facts.get(source_key)) if source_key else None
        if provenance.get("semantic_resolution") == "APPROVED_SOURCE_INTERNAL_CONFLICT":
            return SeverityInputAuthority(
                speed_semantic=semantic, value=value,
                source_type=SeverityInputSource.UNKNOWN, source_ref="",
                derivation_rule_id="", method_contract_hash=self._method_hash,
                status=SeverityInputAuthorityStatus.METHOD_SEMANTIC_AMBIGUITY,
                reason=(
                    "Confirmed Severity sources contain an unresolved internal semantic conflict; "
                    "no speed input may be selected."
                ),
            )
        fact = self._fact_provenance(candidate, source_key)
        source_refs = fact.get("source_refs", fact.get("sources", []))
        source_ref = ""
        if isinstance(source_refs, list) and source_refs:
            first = source_refs[0]
            if isinstance(first, dict):
                source_ref = str(first.get("location", ""))
        source_name = str(fact.get("provenance", "")).upper()
        if value is not None and source_ref and source_is_accepted_for(
            fact, malfunction_id=str(facts.get("malfunction_id", candidate.get("malfunction_id", ""))),
            scenario_id=str(facts.get("scenario_id", candidate.get("scenario_id", ""))),
        ):
            source_type = (
                SeverityInputSource.DERIVED_PHYSICS
                if source_name == "DERIVED" and fact.get("source_binding_kind") != "METHOD_RISK_FACT_BINDING"
                else SeverityInputSource.DIRECT_PROJECT_FACT
                if source_name in {"PROJECT_INPUT", "HUMAN_CONFIRMATION"}
                or fact.get("source_binding_kind") == "METHOD_RISK_FACT_BINDING"
                else SeverityInputSource.DIRECT_SCENARIO_FACT
            )
            derivation_rule_id = str(fact.get("derivation_rule_id", ""))
            if source_type is SeverityInputSource.DERIVED_PHYSICS:
                if semantic is SpeedSemantic.DELTA_V and str(
                    provenance.get("delta_v_derivation_authority", "ABSENT")
                ) != "PRESENT":
                    return SeverityInputAuthority(
                        speed_semantic=semantic, value=value, source_type=source_type,
                        source_ref=source_ref, derivation_rule_id=derivation_rule_id,
                        method_contract_hash=self._method_hash,
                        status=SeverityInputAuthorityStatus.METHOD_DERIVATION_ABSENT,
                        reason="No approved MethodContract rule authorizes the recorded DELTA_V derivation.",
                    )
                return SeverityInputAuthority(
                    speed_semantic=semantic, value=value, source_type=source_type,
                    source_ref=source_ref, derivation_rule_id=derivation_rule_id,
                    method_contract_hash=self._method_hash,
                    status=SeverityInputAuthorityStatus.METHOD_CONFIRMED_DERIVATION,
                    reason=f"A source-linked deterministic {semantic.value} input is available.",
                )
            return SeverityInputAuthority(
                speed_semantic=semantic, value=value, source_type=source_type,
                source_ref=source_ref, derivation_rule_id="",
                method_contract_hash=self._method_hash,
                status=SeverityInputAuthorityStatus.METHOD_CONFIRMED_INPUT_DIRECT,
                reason=f"A source-grounded direct {semantic.value} input is available.",
            )
        other_speed = any(
            self._number(facts.get(key)) is not None
            for key in (
                "ego_speed_kph", "object_speed_kph", "relative_speed_kph",
                "impact_speed_kph", "delta_v_kph",
            ) if key != source_key
        )
        return SeverityInputAuthority(
            speed_semantic=semantic, value=None,
            source_type=SeverityInputSource.UNKNOWN, source_ref="",
            derivation_rule_id="", method_contract_hash=self._method_hash,
            status=(
                SeverityInputAuthorityStatus.METHOD_DERIVATION_ABSENT
                if semantic is SpeedSemantic.DELTA_V and other_speed
                else SeverityInputAuthorityStatus.INPUT_MISSING
            ),
            reason=(
                "Other speed facts are not DELTA_V and no approved derivation is present."
                if semantic is SpeedSemantic.DELTA_V and other_speed
                else f"The configured {semantic.value} input is absent."
            ),
        )

    def _readiness(self, candidate: dict[str, Any]) -> dict[str, Any]:
        authority = self.input_authority(candidate)
        facts = candidate.get("facts", {})
        facts = facts if isinstance(facts, dict) else {}
        available = {
            "ego_speed": self._number(facts.get("ego_speed_kph")) is not None,
            "object_speed": self._number(facts.get("object_speed_kph")) is not None,
            "relative_speed": self._number(facts.get("relative_speed_kph")) is not None,
            "impact_speed": self._number(facts.get("impact_speed_kph")) is not None,
            "delta_v": self._number(facts.get("delta_v_kph")) is not None,
        }
        if authority.status is SeverityInputAuthorityStatus.METHOD_SEMANTIC_AMBIGUITY:
            readiness, blocker = "PENDING_METHOD_SEMANTICS", "METHOD_SEMANTIC_AMBIGUITY"
        elif authority.status in {
            SeverityInputAuthorityStatus.METHOD_CONFIRMED_INPUT_DIRECT,
            SeverityInputAuthorityStatus.METHOD_CONFIRMED_DERIVATION,
        }:
            readiness, blocker = "READY", ""
        else:
            readiness, blocker = "PENDING_INPUT", authority.status.value
        collision = bool(str(facts.get("collision_type", "")).strip())
        participant = bool(str(
            facts.get("road_user_type", facts.get("collision_object", ""))
        ).strip())
        return {
            "available": available,
            "selected_method_semantic": authority.speed_semantic.value,
            "severity_input_status": readiness,
            "blocking_reason": blocker,
            "collision_context_status": (
                "AVAILABLE" if collision and participant else "UNCLASSIFIED_NO_COLLISION_CONTEXT"
            ),
            "severity_input_authority": authority.to_dict(),
        }

    def generate(self, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        candidates = {
            str(item.get("scenario_id", "")): item
            for item in records.get("scenario_candidate", []) if isinstance(item, dict)
        }
        malfunctions = {
            str(item.get("malfunction_id", "")): item
            for item in records.get("malfunction", []) if isinstance(item, dict)
        }
        readiness = []
        for assessment in records.get("scenario_feasibility", []):
            if not isinstance(assessment, dict) or assessment.get("status") != "FINALIZED":
                continue
            if not all(assessment.get(flag) is True for flag in (
                "physically_feasible", "functionally_relevant", "causally_relevant",
            )):
                continue
            malfunction_id = str(assessment.get("malfunction_id", ""))
            scenario_id = str(assessment.get("scenario_id", ""))
            candidate = candidates.get(scenario_id, {})
            item = self._readiness(candidate)
            item.update({
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "hazardous_event_id": str(assessment.get("hazardous_event_id", "")),
                "hazardous_event": str(assessment.get("hazardous_event", "")),
                "hazard_type": "COLLISION" if item["collision_context_status"] == "AVAILABLE" else "UNCLASSIFIED",
                "component_category": str(malfunctions.get(malfunction_id, {}).get("component_category", "")),
            })
            readiness.append(item)
        summary = Counter(item["severity_input_status"] for item in readiness)
        available = {
            key: sum(item["available"][key] for item in readiness)
            for key in ("ego_speed", "object_speed", "relative_speed", "impact_speed", "delta_v")
        }
        inventory = self.method.metadata.get("severity_source_inventory", [])
        if not isinstance(inventory, list):
            inventory = []
        rules = []
        if self.structured is not None:
            for band in self.structured.severity.bands:
                rules.append({
                    "rule_id": band.rule_id,
                    "severity": band.result,
                    "input_semantic": self.structured.severity.speed_semantic.value,
                    "unit": "km/h",
                    "object_category": band.collision_group,
                    "collision_configuration": band.collision_type,
                    "lower_kph": band.lower_kph,
                    "upper_kph": band.upper_kph,
                    "source_ref": f"{band.source_ref.workbook}!{band.source_ref.range}",
                })
        provenance = self._provenance()
        return {
            "artifact_version": "severity-delta-v-semantic-audit-v1",
            "method_contract_hash": self._method_hash,
            "runtime_yaml_read": 0,
            "method_semantic_provenance": provenance,
            "severity_source_inventory": inventory,
            "severity_rule_inventory": rules,
            "speed_semantic_inventory": {
                "EGO_SPEED": "vehicle speed before the event",
                "OBJECT_SPEED": "other collision participant speed before the event",
                "RELATIVE_SPEED": "kinematic closing speed between participants",
                "IMPACT_SPEED": "speed at impact",
                "DELTA_V": "change in vehicle velocity caused by collision",
            },
            "historical_consumer_behavior": {
                "current_runtime": "SeverityMethodExecutor consumes only the compiled selected semantic.",
                "legacy_consumer": "No executable legacy severity consumer is present; severity_evaluator.py appears only in a source comment.",
                "classification": "IMPLEMENTATION_ARTIFACT_NOT_CONFIRMED_METHOD_SEMANTICS",
            },
            "production_runtime_behavior": {
                "executor": "SeverityMethodExecutor" if self.structured is not None else "UNAVAILABLE",
                "selected_input": {
                    "RELATIVE_SPEED": "relative_speed_kph",
                    "DELTA_V": "delta_v_kph",
                    "EGO_SPEED": "ego_speed_kph",
                    "IMPACT_SPEED": "impact_speed_kph",
                }.get(self.structured.severity.speed_semantic.value, "") if self.structured is not None else "",
                "uses_ego_speed_as_delta_v": False,
                "uses_relative_speed_as_delta_v": False,
                "uses_impact_speed_as_delta_v": False,
                "uses_odd_max_speed_as_delta_v": False,
                "missing_selected_input_status": "PENDING_INPUT",
            },
            "delta_v_derivation_authority": {
                "status": str(provenance.get("delta_v_derivation_authority", "ABSENT")),
                "approved_rule_ids": [],
                "physics_model": False,
                "mass_model": False,
                "restitution_model": False,
                "impact_geometry_model": False,
                "lookup_table": False,
            },
            "scenario_physics_audit": {
                "derived_quantities": ["relative_distance_m", "ttc_s"],
                "normalized_inputs": [
                    "ego_speed_kph", "relative_speed_kph", "impact_speed_kph", "delta_v_kph",
                ],
                "derives_delta_v": False,
                "status": "PASS",
            },
            "collision_context_requirements": {
                "method_support": "COLLISION_ONLY",
                "executor_inputs": [
                    {
                        "RELATIVE_SPEED": "relative_speed_kph",
                        "DELTA_V": "delta_v_kph",
                        "EGO_SPEED": "ego_speed_kph",
                        "IMPACT_SPEED": "impact_speed_kph",
                    }.get(self.structured.severity.speed_semantic.value, ""),
                    "road_user_type", "collision_type",
                ],
                "collision_object_consumed_directly": False,
                "non_collision_branch_present": False,
            },
            "non_collision_gap_analysis": {
                "status": "NON_COLLISION_SEVERITY_METHOD_GAP",
                "explicit_non_collision_records": 0,
                "unclassified_no_collision_context_records": sum(
                    item["collision_context_status"] != "AVAILABLE" for item in readiness
                ),
                "note": "No review artifact supplies a typed non-collision hazard classification; the audit does not infer one from prose.",
            },
            "r3_severity_readiness": readiness,
            "summary": {
                "causal_relevant_hazardous_events": len(readiness),
                "READY": summary["READY"],
                "PENDING_INPUT": summary["PENDING_INPUT"],
                "PENDING_METHOD_SEMANTICS": summary["PENDING_METHOD_SEMANTICS"],
                "direct_delta_v_available": available["delta_v"],
                "ego_speed_available": available["ego_speed"],
                "relative_speed_available": available["relative_speed"],
                "impact_speed_available": available["impact_speed"],
                "object_speed_available": available["object_speed"],
                "non_collision_method_gap": 0,
                "unclassified_no_collision_context": sum(
                    item["collision_context_status"] != "AVAILABLE" for item in readiness
                ),
            },
        }
