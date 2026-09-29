"""Offline, source-grounded S/C scoreability and differential-queue projection."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from hara_agent.contracts import MethodContract
from hara_agent.models import evaluate_risk_eligibility_payload

from .hazardous_event_risk_context_service import HazardousEventRiskContextService
from .risk_vocabulary_adapter import RiskVocabularyAdapter, RiskVocabularyResolution
from .scenario_physics import closing_relative_speed_kph, source_is_accepted_for


class RiskScoreabilityService:
    """Classify analytical options without inventing inputs or calling a Provider."""

    _CONTROL_FIELDS = (
        "driver_in_vehicle",
        "remote_intervention_available",
        "other_road_user_avoidance_possible",
    )
    _PROJECT_FACT_FIELDS = (
        "ego_speed_kph",
        "driver_in_vehicle",
        "remote_intervention_available",
        "other_road_user_avoidance_possible",
        "direct_control_available",
        "emergency_braking_available",
        "vehicle_stability",
    )
    _RISK_ONLY_FIELDS = {
        "road_user_type",
        "collision_type",
        "object_type",
        "object_position",
        "object_speed_kph",
        "relative_distance_m",
    }
    _DIRECTION_FIELDS = (
        "ego_longitudinal_direction",
        "object_longitudinal_direction",
    )
    _ANALYTICAL_FIELDS = frozenset(
        _RISK_ONLY_FIELDS
        | set(_CONTROL_FIELDS)
        | {"ego_speed_kph", *_DIRECTION_FIELDS}
    )
    _FINAL_ASSUMPTION_STATUSES = {"VALIDATED", "FINALIZED", "APPROVED"}
    def __init__(self, method: MethodContract):
        self.vocabulary = RiskVocabularyAdapter(method)
        self.contexts = HazardousEventRiskContextService(method)
        physics = method.metadata.get("risk_vocabulary_physics", {})
        if not isinstance(physics, dict):
            raise ValueError("MethodContract risk vocabulary physics must be an object")
        self._longitudinal_collisions = frozenset(
            str(value) for value in physics.get("longitudinal_collision_types", [])
        )
        self._ego_directions = frozenset(
            str(value) for value in physics.get("ego_longitudinal_directions", [])
        )
        self._object_directions = frozenset(
            str(value) for value in physics.get("object_longitudinal_directions", [])
        )
        self._motion_rule_id = str(physics.get("derivation_rule_id", ""))
        if not all((
            self._longitudinal_collisions, self._ego_directions,
            self._object_directions, self._motion_rule_id,
        )):
            raise ValueError("MethodContract risk vocabulary physics is incomplete")

    @staticmethod
    def _point(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _metadata(facts: dict[str, Any], field: str) -> dict[str, Any]:
        provenance = facts.get("_fact_provenance", facts.get("fact_provenance", {}))
        value = provenance.get(field, {}) if isinstance(provenance, dict) else {}
        return dict(value) if isinstance(value, dict) else {}

    def _accepted_fact(
        self, *, facts: dict[str, Any], field: str, malfunction_id: str,
        parent_scenario_id: str,
    ) -> dict[str, Any]:
        value = facts.get(field)
        metadata = self._metadata(facts, field)
        valid = source_is_accepted_for(
            metadata,
            malfunction_id=malfunction_id,
            scenario_id=parent_scenario_id,
        )
        type_valid = (
            self._point(value) if field.endswith("_kph") else
            isinstance(value, bool) if field in self._CONTROL_FIELDS
            or field in {"direct_control_available", "emergency_braking_available"}
            else isinstance(value, str) and bool(value.strip())
        )
        if valid and type_valid:
            origin = str(metadata.get("origin", metadata.get("provenance", ""))).upper()
            return {
                "status": (
                    "EXISTING_PROJECT_FACT"
                    if origin in {"PROJECT_INPUT", "HUMAN_CONFIRMATION"}
                    else "SCENARIO_DEFINED"
                ),
                "value": value,
                "provenance": metadata,
            }
        if field in facts or metadata:
            return {"status": "CONFLICT", "value": None, "provenance": metadata}
        return {"status": "MISSING", "value": None, "provenance": {}}

    @staticmethod
    def _option_values(option: dict[str, Any]) -> dict[str, Any]:
        instance = option.get("analysis_instance", {})
        return dict(instance.get("source_option_values", {})) if isinstance(instance, dict) else {}

    @staticmethod
    def _instance(option: dict[str, Any]) -> dict[str, Any]:
        value = option.get("analysis_instance", {})
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _motion_readiness(
        self, *, facts: dict[str, Any], parent_scenario_id: str,
        malfunction_id: str, object_speed_kph: Any, collision: RiskVocabularyResolution,
    ) -> dict[str, Any]:
        ego = self._accepted_fact(
            facts=facts, field="ego_speed_kph", malfunction_id=malfunction_id,
            parent_scenario_id=parent_scenario_id,
        )
        if not self._point(object_speed_kph):
            return {"status": "MISSING", "reason": "OBJECT_POINT_SPEED_MISSING"}
        # A validated, scoped engineering-analysis setting is an accepted risk
        # input.  It remains SCENARIO_DEFINED; it must never be promoted to a
        # project fact merely to make a deterministic calculation run.
        if ego["status"] not in {"EXISTING_PROJECT_FACT", "SCENARIO_DEFINED"}:
            return {
                "status": "MISSING",
                "reason": "EGO_POINT_SPEED_ENGINEERING_ASSUMPTION_REQUIRED",
                "ego": ego,
            }
        if collision.canonical_value not in self._longitudinal_collisions:
            return {
                "status": "MISSING",
                "reason": "LATERAL_OR_UNMAPPED_COLLISION_PHYSICS_UNRESOLVED",
                "ego": ego,
            }
        direction_fields = ("ego_longitudinal_direction", "object_longitudinal_direction")
        directions = [
            self._accepted_fact(
                facts=facts, field=field, malfunction_id=malfunction_id,
                parent_scenario_id=parent_scenario_id,
            )
            for field in direction_fields
        ]
        if any(item["status"] not in {"EXISTING_PROJECT_FACT", "SCENARIO_DEFINED"} for item in directions):
            return {
                "status": "MISSING",
                "reason": "LONGITUDINAL_DIRECTION_MISSING",
                "ego": ego,
            }
        ego_direction, object_direction = (
            str(item["value"]).strip().upper() for item in directions
        )
        if (
            ego_direction not in self._ego_directions
            or object_direction not in self._object_directions
        ):
            return {
                "status": "MISSING",
                "reason": "LONGITUDINAL_DIRECTION_UNSUPPORTED",
                "ego": ego,
            }
        speed = closing_relative_speed_kph(
            ego["value"], object_speed_kph,
            ego_direction=ego_direction, object_direction=object_direction,
            collision_type=collision.canonical_value,
        )
        if speed is None:
            return {
                "status": "MISSING",
                "reason": "LONGITUDINAL_RELATIVE_SPEED_UNRESOLVED",
                "ego": ego,
            }
        opposing = object_direction != "STATIONARY" and ego_direction != object_direction
        return {
            "status": "DERIVED_PHYSICS",
            "value": speed,
            "derivation_rule_id": self._motion_rule_id,
            "input_refs": [
                "SCN.ego_speed_kph", "SCN.ego_longitudinal_direction",
                "SCN.object_longitudinal_direction", "ANALYTICAL.object_speed_kph",
                "ANALYTICAL.collision_type",
            ],
            "input_values": {
                "ego_speed_kph": ego["value"],
                "ego_longitudinal_direction": ego_direction,
                "object_speed_kph": object_speed_kph,
                "object_longitudinal_direction": object_direction,
                "collision_type": collision.canonical_value,
            },
            "motion_relation": (
                "OPPOSING_LONGITUDINAL" if opposing else "SAME_OR_STATIONARY_LONGITUDINAL"
            ),
            "output_unit": "km/h",
            "scope": {
                "scenario_id": parent_scenario_id,
                "parent_scenario_id": facts.get("_parent_scenario_id", parent_scenario_id),
            },
            "ego": ego,
        }

    @staticmethod
    def _dependency_values(value: Any) -> set[str]:
        """Normalize explicit dependency declarations, never infer an empty set."""
        if not isinstance(value, list):
            return set()
        return {
            str(item).removeprefix("SCN.")
            for item in value
            if isinstance(item, str) and item.strip()
        }

    @classmethod
    def _dependency_metadata(cls, assessment: dict[str, Any]) -> dict[str, Any] | None:
        """Return positive noninterference evidence or fail closed.

        Legacy causal assessments contain edge evidence, but do not declare the
        full physical and identity dependency set.  An absent declaration is
        therefore unknown, not proof that an analytical child is harmless.
        """
        causal = assessment.get("causal_assessment", {})
        causal = causal if isinstance(causal, dict) else {}
        candidate = assessment.get("dependency_metadata", causal.get("dependency_metadata"))
        if not isinstance(candidate, dict):
            return None
        required = (
            "causal_evidence_fields",
            "physical_feasibility_fields",
            "scenario_identity_fields",
        )
        if (
            candidate.get("complete") is not True
            or not all(isinstance(candidate.get(field), list) for field in required)
            or not str(candidate.get("child_subset_refinement", "")).strip()
        ):
            return None
        return {
            "causal": cls._dependency_values(candidate["causal_evidence_fields"]),
            "physical": cls._dependency_values(candidate["physical_feasibility_fields"]),
            "identity": cls._dependency_values(candidate["scenario_identity_fields"]),
            "subset_basis": str(candidate["child_subset_refinement"]).strip(),
        }

    @staticmethod
    def _speed_envelope(parent_facts: dict[str, Any]) -> tuple[float | None, float | None]:
        envelope = parent_facts.get("ego_speed_constraint", {})
        if not isinstance(envelope, dict):
            return None, None
        lower = envelope.get("min_kph", envelope.get("speed_min_kph"))
        upper = envelope.get("max_kph", envelope.get("speed_max_kph"))
        return (
            float(lower) if isinstance(lower, (int, float)) and not isinstance(lower, bool) else None,
            float(upper) if isinstance(upper, (int, float)) and not isinstance(upper, bool) else None,
        )

    def classify_delta(
        self, *, assessment: dict[str, Any], parent_facts: dict[str, Any],
        child_values: dict[str, Any], parent_scenario_id: str,
        malfunction_id: str,
    ) -> dict[str, Any]:
        """Prove noninterference before causal reuse; otherwise remain conservative."""
        del parent_scenario_id, malfunction_id
        changed = {
            field: value for field, value in child_values.items()
            if field in self._ANALYTICAL_FIELDS and value not in (None, "")
        }
        conflicts = [
            field for field, value in changed.items()
            if field in parent_facts and parent_facts[field] not in (None, "")
            and parent_facts[field] != value
        ]
        if conflicts:
            return {
                "classification": "SOURCE_CONFLICT",
                "causal_revalidation_required": False,
                "causal_reuse_basis": "",
                "changed_fields": sorted(changed),
                "conflicting_fields": sorted(conflicts),
                "dependency_check": "PARENT_FACT_CONFLICT",
            }
        metadata = self._dependency_metadata(assessment)
        if metadata is None:
            return {
                "classification": "UNCLASSIFIED",
                "causal_revalidation_required": True,
                "causal_reuse_basis": "",
                "changed_fields": sorted(changed),
                "dependency_check": "DEPENDENCY_METADATA_INCOMPLETE",
            }
        causal_dependencies = metadata["causal"]
        physical_dependencies = metadata["physical"]
        identity_dependencies = metadata["identity"]
        dependent = sorted(
            field for field in changed
            if field in causal_dependencies or field in physical_dependencies
            or field in identity_dependencies
        )
        if dependent:
            return {
                "classification": "CAUSAL_RELEVANT_CHANGE",
                "causal_revalidation_required": True,
                "causal_reuse_basis": "",
                "changed_fields": sorted(changed),
                "dependent_fields": dependent,
                "dependency_check": "PARENT_CAUSAL_OR_PHYSICAL_DEPENDENCY",
            }
        if "ego_speed_kph" in changed:
            lower, upper = self._speed_envelope(parent_facts)
            value = changed["ego_speed_kph"]
            if (
                self._point(value)
                and (lower is None or value >= lower)
                and (upper is None or value <= upper)
            ):
                return {
                    "classification": "DETERMINISTIC_CAUSAL_REUSE",
                    "causal_revalidation_required": False,
                    "causal_reuse_basis": "VERIFIED_NONINTERFERENCE",
                    "changed_fields": sorted(changed),
                    "dependency_check": "EXPLICIT_NONINTERFERENCE_AND_POINT_WITHIN_PARENT_ENVELOPE",
                    "child_subset_refinement": metadata["subset_basis"],
                }
            return {
                "classification": "CAUSAL_RELEVANT_CHANGE",
                "causal_revalidation_required": True,
                "causal_reuse_basis": "",
                "changed_fields": sorted(changed),
                "dependency_check": "SPEED_NOT_PROVEN_SUBSET",
            }
        if changed and set(changed).issubset(self._RISK_ONLY_FIELDS):
            return {
                "classification": "DETERMINISTIC_CAUSAL_REUSE",
                "causal_revalidation_required": False,
                "causal_reuse_basis": "VERIFIED_NONINTERFERENCE",
                "changed_fields": sorted(changed),
                "dependency_check": "EXPLICIT_NONINTERFERENCE",
                "child_subset_refinement": metadata["subset_basis"],
            }
        return {
            "classification": "UNCLASSIFIED",
            "causal_revalidation_required": True,
            "causal_reuse_basis": "",
            "changed_fields": sorted(changed),
            "dependency_check": "NO_CONSERVATIVE_REUSE_PROOF",
        }

    def _control_readiness(
        self, *, facts: dict[str, Any], parent_scenario_id: str,
        malfunction_id: str, values: dict[str, Any], motion: dict[str, Any],
    ) -> dict[str, Any]:
        accepted_controls = {
            field: self._accepted_fact(
                facts=facts, field=field, malfunction_id=malfunction_id,
                parent_scenario_id=parent_scenario_id,
            )
            for field in self._CONTROL_FIELDS
        }
        scenario = dict(facts)
        scenario_provenance = dict(
            facts.get("_fact_provenance", facts.get("fact_provenance", {}))
        )
        for field, item in accepted_controls.items():
            if item["status"] in {"EXISTING_PROJECT_FACT", "SCENARIO_DEFINED"}:
                scenario[field] = item["value"]
        scenario["_fact_provenance"] = scenario_provenance
        context = self.contexts.build(
            malfunction_id=malfunction_id,
            scenario_id=parent_scenario_id,
            hazard_node_id="SCOREABILITY_PROJECTION",
            scenario=scenario,
        )
        readiness = self.contexts.controllability_readiness(context)
        override_finalized = (
            readiness["status"] == "READY" and readiness["branch"] == "OVERRIDE"
        )
        override_state = (
            "MATCH" if override_finalized else
            "UNKNOWN" if any(
                item["status"] not in {"EXISTING_PROJECT_FACT", "SCENARIO_DEFINED"}
                for item in accepted_controls.values()
            ) else "NO_MATCH" if readiness["branch"] == "TTC" else "UNKNOWN"
        )
        blockers = [
            f"{field.upper()}_ENGINEERING_ASSUMPTION_REQUIRED"
            for field, item in accepted_controls.items()
            if item["status"] == "MISSING"
        ]
        blockers.extend(
            f"{field.upper()}_SOURCE_CONFLICT"
            for field, item in accepted_controls.items()
            if item["status"] == "CONFLICT"
        )
        if not override_finalized and readiness["branch"] != "TTC":
            blockers.append("OVERRIDE_BRANCH_UNRESOLVED")
        if readiness["branch"] == "TTC" and readiness["status"] != "READY":
            blockers.extend(readiness["missing_reasons"])
        return {
            "controls": accepted_controls,
            "override_state": override_state,
            "override_finalized": override_finalized,
            "ttc_ready": readiness["status"] == "READY" and readiness["branch"] == "TTC",
            "blockers": sorted(set(blockers)),
            "method_readiness": readiness,
            "motion_status": motion["status"],
            "distance_available": self._point(values.get("obj_distance_m")),
        }

    def _record(
        self, *, option: dict[str, Any], assessment: dict[str, Any],
        scenario: dict[str, Any], malfunction: dict[str, Any],
        parent_scenario: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(option)
        values = self._option_values(option)
        malfunction_id = str(option.get("malfunction_id", ""))
        parent_id = str(option.get("parent_scenario_id", ""))
        analysis_scenario_id = str(scenario.get("scenario_id", parent_id))
        parent_facts = dict((parent_scenario or scenario).get("facts", {}))
        facts = dict(scenario.get("facts", {}))
        facts["_fact_provenance"] = dict(scenario.get("fact_provenance", {}))
        facts["_parent_scenario_id"] = parent_id
        road = self.vocabulary.resolve(
            field="road_user_type", raw_value=values.get("obj_type", ""),
        )
        collision = self.vocabulary.resolve(
            field="collision_type", raw_value=values.get("collision_type", ""),
        )
        ego = self._accepted_fact(
            facts=facts, field="ego_speed_kph", malfunction_id=malfunction_id,
            parent_scenario_id=analysis_scenario_id,
        )
        motion = self._motion_readiness(
            facts=facts, parent_scenario_id=analysis_scenario_id, malfunction_id=malfunction_id,
            object_speed_kph=values.get("obj_v_kph"), collision=collision,
        )
        s_blockers = []
        if not road.mapped:
            s_blockers.append("ROAD_USER_TYPE_UNMAPPED")
        if not collision.mapped:
            s_blockers.append("COLLISION_TYPE_UNMAPPED")
        if ego["status"] == "MISSING":
            s_blockers.append("EGO_POINT_SPEED_ENGINEERING_ASSUMPTION_REQUIRED")
        elif ego["status"] == "CONFLICT":
            s_blockers.append("EGO_POINT_SPEED_SOURCE_CONFLICT")
        if motion["status"] != "DERIVED_PHYSICS":
            s_blockers.append(str(motion["reason"]))
        controls = self._control_readiness(
            facts=facts, parent_scenario_id=analysis_scenario_id, malfunction_id=malfunction_id,
            values=values, motion=motion,
        )
        child_values = {
            "object_type": values.get("obj_type"),
            "object_position": values.get("obj_position"),
            "relative_distance_m": values.get("obj_distance_m"),
            "object_speed_kph": values.get("obj_v_kph"),
        }
        if road.mapped:
            child_values["road_user_type"] = road.canonical_value
        if collision.mapped:
            child_values["collision_type"] = collision.canonical_value
        for field in self._ANALYTICAL_FIELDS - self._RISK_ONLY_FIELDS:
            value = facts.get(field)
            if value not in (None, "") and parent_facts.get(field) != value:
                child_values[field] = value
        delta = self.classify_delta(
            assessment=assessment, parent_facts=parent_facts, child_values=child_values,
            parent_scenario_id=parent_id, malfunction_id=malfunction_id,
        )
        s_ready = not s_blockers
        c_ready = controls["override_finalized"] or controls["ttc_ready"]
        return {
            "malfunction_id": malfunction_id,
            "function_id": str(malfunction.get("function_id", "")),
            "parent_scenario_id": parent_id,
            "child_scenario_id": analysis_scenario_id if analysis_scenario_id != parent_id else "",
            "operating_mode": str(scenario.get("operating_mode", "")),
            "parent_facts": parent_facts,
            "parent_fact_provenance": dict((parent_scenario or scenario).get("fact_provenance", {})),
            "template_id": str(instance.get("source_template_id", "")),
            "option_id": str(instance.get("source_option_id", "")),
            "child_identity": {
                "source_template_id": str(instance.get("source_template_id", "")),
                "source_option_id": str(instance.get("source_option_id", "")),
                "method_contract_hash": self.vocabulary.method_contract_hash,
            },
            "mappings": {
                "road_user_type": road.to_dict(),
                "collision_type": collision.to_dict(),
            },
            "project_fact_inventory": {
                field: self._accepted_fact(
                    facts=facts, field=field, malfunction_id=malfunction_id,
                    parent_scenario_id=analysis_scenario_id,
                )
                for field in self._PROJECT_FACT_FIELDS
            },
            "motion_direction_inventory": {
                field: self._accepted_fact(
                    facts=facts, field=field, malfunction_id=malfunction_id,
                    parent_scenario_id=analysis_scenario_id,
                )
                for field in self._DIRECTION_FIELDS
            },
            "motion": motion,
            "S": {
                "ready": s_ready,
                "road_user_type_ready": road.mapped,
                "collision_type_ready": collision.mapped,
                "ego_speed": ego,
                "relative_speed": motion,
                "blockers": sorted(set(s_blockers)),
            },
            "C": {"ready": c_ready, **controls},
            "delta": delta,
            "semantic_fingerprint": [
                malfunction_id,
                str(assessment.get("hazardous_event", malfunction.get("functional_effect", ""))),
                road.canonical_value or f"UNMAPPED:{values.get('obj_type', '')}",
                collision.canonical_value or f"UNMAPPED:{values.get('collision_type', '')}",
                str(values.get("obj_position", "")),
                str(values.get("obj_v_kph", "")),
                str(values.get("obj_distance_m", "")),
                str(facts.get("ego_speed_kph", "")),
                str(facts.get("ego_longitudinal_direction", "")),
                str(facts.get("object_longitudinal_direction", "")),
                str(facts.get("driver_in_vehicle", "")),
                str(facts.get("remote_intervention_available", "")),
                str(facts.get("other_road_user_avoidance_possible", "")),
            ],
        }

    @staticmethod
    def _record_queue_status(record: dict[str, Any]) -> str:
        delta = record["delta"]
        mapping_blocked = not all(
            item["canonical_value"] for item in record["mappings"].values()
        )
        blocker_codes = record["S"]["blockers"] + record["C"]["blockers"]
        validation_codes = {
            str(item.get("code", ""))
            for item in record.get("assumption_validation_errors", [])
            if isinstance(item, dict)
        }
        source_conflict = any("SOURCE_CONFLICT" in code for code in blocker_codes) or "SOURCE_CONFLICT" in validation_codes
        engineering_blocked = any(
            "ENGINEERING_ASSUMPTION_REQUIRED" in code
            or "DIRECTION_MISSING" in code
            or "OBJECT_POINT_SPEED_MISSING" in code
            for code in blocker_codes
        )
        if delta["classification"] == "SOURCE_CONFLICT" or source_conflict:
            return "SOURCE_CONFLICT"
        if mapping_blocked:
            return "UNRESOLVED_MAPPING"
        if engineering_blocked:
            return "BLOCKED_ENGINEERING_ASSUMPTION"
        if delta["classification"] == "UNCLASSIFIED":
            return "UNCLASSIFIED"
        if not delta["causal_revalidation_required"]:
            return "DETERMINISTIC_CAUSAL_REUSE"
        if record["S"]["ready"] and record["C"]["ready"]:
            return "READY_FOR_DIFFERENTIAL_PROVIDER"
        return "DETERMINISTIC_SCORING_ONLY"

    @staticmethod
    def readiness_statuses(record: dict[str, Any]) -> tuple[str, ...]:
        """Expose orthogonal P5-D gates instead of one generic blocked bucket."""
        statuses: list[str] = []
        synthesis = str(record.get("scenario_synthesis_status", "METHOD_VALID"))
        if synthesis != "METHOD_VALID":
            statuses.append("SCENARIO_SYNTHESIS_BLOCKED")
        causal = str(record.get(
            "causal_delta_status",
            "CAUSAL_REVALIDATION_REQUIRED"
            if record.get("delta", {}).get("causal_revalidation_required") else
            "CAUSAL_REUSE_PROVEN",
        ))
        if causal not in {"CAUSAL_REUSE_PROVEN", "CAUSAL_REVALIDATED"}:
            statuses.append("CAUSAL_REVALIDATION_BLOCKED")
        s = record.get("S", {}) if isinstance(record.get("S"), dict) else {}
        c = record.get("C", {}) if isinstance(record.get("C"), dict) else {}
        blockers = [
            str(item) for item in [*s.get("blockers", []), *c.get("blockers", [])]
        ]
        if any(
            token in blocker for blocker in blockers
            for token in (
                "ENGINEERING_ASSUMPTION_REQUIRED", "DIRECTION_MISSING",
                "POINT_SPEED_MISSING", "DISTANCE_MISSING",
            )
        ):
            statuses.append("PHYSICS_ASSUMPTION_BLOCKED")
        if s.get("ready") is True:
            statuses.append("S_READY")
        exposure = record.get("exposure_readiness", {})
        if isinstance(exposure, dict) and exposure.get("status") in {
            "READY_COMPLETE", "READY_METHOD_IRRELEVANT_GAPS",
        }:
            statuses.append("E_READY")
        if c.get("override_finalized") is True:
            statuses.append("C_OVERRIDE_FINALIZED")
        if c.get("ttc_ready") is True:
            statuses.append("C_TTC_READY")
        return tuple(dict.fromkeys(statuses))

    def _scope_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the smallest context needed to prevent unsafe value sharing."""
        facts = record.get("parent_facts", {})
        facts = facts if isinstance(facts, dict) else {}
        lower, upper = self._speed_envelope(facts)
        return {
            "parent_scenario_id": record["parent_scenario_id"],
            "function_id": record["function_id"],
            "malfunction_ids": [record["malfunction_id"]],
            "operating_mode": record.get("operating_mode", ""),
            "ego_speed_envelope": {"min_kph": lower, "max_kph": upper},
            "vehicle_state": facts.get("vehicle_state", facts.get("motion_state", "")),
            "control_mode": facts.get("control_mode", ""),
            "driver_presence_semantics": facts.get("driver_presence_semantics", ""),
            "remote_control_semantics": facts.get("remote_control_semantics", ""),
        }

    def _assumption_groups(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split by every supplied compatibility dimension before sharing input."""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scopes: dict[str, dict[str, Any]] = {}
        for record in records:
            scope = self._scope_for_record(record)
            key = self._canonical_json(scope)
            grouped[key].append(record)
            scopes[key] = scope
        result = []
        for key, items in sorted(grouped.items()):
            scope = dict(scopes[key])
            scope["malfunction_ids"] = sorted({item["malfunction_id"] for item in items})
            group_id = "ASSUMPTION-" + hashlib.sha256(
                self._canonical_json(scope).encode("utf-8")
            ).hexdigest()[:16].upper()
            result.append({
                "group_id": group_id,
                "scope_type": (
                    "FUNCTION_X_PARENT_SCENARIO"
                    if scope["function_id"] else "MF_FAMILY_X_PARENT_SCENARIO"
                ),
                "scope": scope,
                "records": items,
            })
        return result

    @staticmethod
    def _field_unit(field: str) -> str:
        return "km/h" if field == "ego_speed_kph" else ""

    def _required_fields(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        """List only unresolved engineering facts; no point value is invented."""
        items = group["records"]
        required: list[tuple[str, str, dict[str, Any] | None]] = []
        if any(item["S"]["ego_speed"]["status"] == "MISSING" for item in items):
            required.append((
                "ego_speed_kph",
                "No accepted point ego speed exists; an ODD envelope is not a point speed.",
                {"min": group["scope"]["ego_speed_envelope"]["min_kph"],
                 "max": group["scope"]["ego_speed_envelope"]["max_kph"]},
            ))
        for field in self._CONTROL_FIELDS:
            if any(item["C"]["controls"][field]["status"] == "MISSING" for item in items):
                required.append((
                    field,
                    "Selected controllability branch requires an exact engineering setting.",
                    None,
                ))
        longitudinal = any(
            item["mappings"]["collision_type"]["canonical_value"] in self._longitudinal_collisions
            for item in items
        )
        if longitudinal:
            for field in self._DIRECTION_FIELDS:
                if any(item["motion_direction_inventory"][field]["status"] == "MISSING" for item in items):
                    required.append((
                        field,
                        "Longitudinal relative speed requires an explicit, source-backed direction; it is not inferred from front/rear position.",
                        None,
                    ))
        return [{
            "field": field,
            "value": None,
            "unit": self._field_unit(field),
            "scope": group["scope"],
            "source_type": "ENGINEERING_ANALYSIS_SETTING",
            "source_id": "",
            "approval": "PENDING",
            "validation_status": "REQUIRES_ENGINEERING_INPUT",
            "status": "REQUIRES_ENGINEERING_INPUT",
            "reason": reason,
            "allowed_range": allowed_range,
            "affected_option_count": len(items),
        } for field, reason, allowed_range in required]

    def refined_assumption_pack(
        self, *, checkpoint_sha256: str, supplement: dict[str, Any],
        records: list[dict[str, Any]], existing_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a human-fillable, safely scoped production assumption pack."""
        existing: dict[tuple[str, str], dict[str, Any]] = {}
        if isinstance(existing_pack, dict):
            for group in existing_pack.get("groups", []):
                if not isinstance(group, dict):
                    continue
                for item in group.get("fields", group.get("required_fields", [])):
                    if isinstance(item, dict):
                        existing[(str(group.get("group_id", "")), str(item.get("field", "")))] = item
        groups = []
        for group in self._assumption_groups(records):
            fields = self._required_fields(group)
            if not fields:
                continue
            for field in fields:
                prior = existing.get((group["group_id"], field["field"]))
                if prior is not None and self._canonical_json(prior.get("scope", {})) == self._canonical_json(group["scope"]):
                    for key in (
                        "value", "unit", "source_type", "source_id", "approval",
                        "validation_status", "status",
                    ):
                        if key in prior:
                            field[key] = prior[key]
            groups.append({
                "group_id": group["group_id"],
                "scope_type": group["scope_type"],
                "scope": group["scope"],
                "fields": fields,
                # Kept as a read-only compatibility alias for v2 consumers.
                "required_fields": fields,
            })
        return {
            "artifact_version": "risk-assumption-pack-v2",
            "source_run_id": str(supplement.get("source_run_id", "")),
            "checkpoint_sha256": checkpoint_sha256,
            "groups": groups,
            "summary": {
                "groups": len(groups),
                "ego_speed_decision_groups": sum(
                    any(item["field"] == "ego_speed_kph" for item in group["fields"])
                    for group in groups
                ),
                "control_context_decision_groups": sum(
                    any(item["field"] in self._CONTROL_FIELDS for item in group["fields"])
                    for group in groups
                ),
                "affected_options": len(records),
            },
        }

    def _validate_assumption_pack(
        self, *, pack: dict[str, Any], group_specs: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], set[str]]:
        """Validate entered values without changing historical project facts."""
        expected = {item["group_id"]: item for item in group_specs}
        entries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        errors: list[dict[str, Any]] = []
        complete: set[str] = set()
        supplied = pack.get("groups", []) if isinstance(pack, dict) else []
        supplied_by_id = {
            str(item.get("group_id", "")): item for item in supplied if isinstance(item, dict)
        }
        for group_id, expected_group in expected.items():
            supplied_group = supplied_by_id.get(group_id)
            expected_fields = self._required_fields(expected_group)
            if supplied_group is None:
                continue
            if self._canonical_json(supplied_group.get("scope", {})) != self._canonical_json(expected_group["scope"]):
                errors.append({"group_id": group_id, "code": "ASSUMPTION_SCOPE_MISMATCH"})
                continue
            values = {
                str(item.get("field", "")): item
                for item in supplied_group.get("fields", supplied_group.get("required_fields", []))
                if isinstance(item, dict)
            }
            group_valid = True
            for expected_field in expected_fields:
                field = expected_field["field"]
                entry = values.get(field)
                if entry is None or entry.get("value") is None:
                    group_valid = False
                    continue
                entry_scope = entry.get("scope", supplied_group.get("scope", {}))
                if self._canonical_json(entry_scope) != self._canonical_json(expected_group["scope"]):
                    errors.append({"group_id": group_id, "field": field, "code": "ASSUMPTION_SCOPE_MISMATCH"})
                    group_valid = False
                    continue
                if str(entry.get("source_type", "")) != "ENGINEERING_ANALYSIS_SETTING" or not str(entry.get("source_id", "")).strip():
                    errors.append({"group_id": group_id, "field": field, "code": "INVALID_ENGINEERING_SOURCE"})
                    group_valid = False
                    continue
                if str(entry.get("approval", "")).upper() not in self._FINAL_ASSUMPTION_STATUSES or str(entry.get("validation_status", entry.get("status", ""))).upper() not in self._FINAL_ASSUMPTION_STATUSES:
                    errors.append({"group_id": group_id, "field": field, "code": "ASSUMPTION_NOT_VALIDATED"})
                    group_valid = False
                    continue
                value = entry["value"]
                if field == "ego_speed_kph":
                    allowed = expected_field["allowed_range"] or {}
                    lower, upper = allowed.get("min"), allowed.get("max")
                    if not self._point(value) or (lower is not None and value < lower) or (upper is not None and value > upper):
                        errors.append({"group_id": group_id, "field": field, "code": "EGO_SPEED_OUTSIDE_ENVELOPE"})
                        group_valid = False
                        continue
                elif field in self._CONTROL_FIELDS and not isinstance(value, bool):
                    errors.append({"group_id": group_id, "field": field, "code": "CONTROL_INPUT_NOT_BOOLEAN"})
                    group_valid = False
                    continue
                elif field in self._DIRECTION_FIELDS:
                    vocabulary = self._ego_directions if field.startswith("ego_") else self._object_directions
                    if str(value).upper() not in vocabulary:
                        errors.append({"group_id": group_id, "field": field, "code": "DIRECTION_ENUM_UNSUPPORTED"})
                        group_valid = False
                        continue
                conflicts = [
                    item["parent_scenario_id"] for item in expected_group["records"]
                    if item["parent_facts"].get(field) not in (None, "")
                    and item["parent_facts"].get(field) != value
                ]
                if conflicts:
                    errors.append({
                        "group_id": group_id, "field": field, "code": "SOURCE_CONFLICT",
                        "parent_scenario_ids": sorted(set(conflicts)),
                    })
                    group_valid = False
                    continue
                entries[group_id][field] = dict(entry)
            if group_valid and len(entries[group_id]) == len(expected_fields):
                complete.add(group_id)
        return entries, errors, complete

    def _materialize_child(
        self, *, record: dict[str, Any], option: dict[str, Any],
        group: dict[str, Any], assumptions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a separate analytical child with fully scoped fact lineage."""
        material = {
            "parent_scenario_id": record["parent_scenario_id"],
            "malfunction_id": record["malfunction_id"],
            "template_id": record["template_id"],
            "option_id": record["option_id"],
            "assumptions": {field: item["value"] for field, item in sorted(assumptions.items())},
        }
        child_id = "SCN-ANALYTICAL-" + hashlib.sha256(
            self._canonical_json(material).encode("utf-8")
        ).hexdigest()[:16].upper()
        facts = dict(record["parent_facts"])
        provenance = dict(record.get("parent_fact_provenance", {}))
        scope = {
            "malfunction_id": record["malfunction_id"],
            "scenario_id": child_id,
            "parent_scenario_id": record["parent_scenario_id"],
            "function_id": record["function_id"],
        }
        template_source = {
            "source_type": "METHOD_CONTRACT",
            "source_id": str(record["child_identity"]["method_contract_hash"]),
            "location": f"{record['template_id']}:{record['option_id']}",
        }
        values = self._option_values(option)
        template_facts = {
            "object_type": values.get("obj_type"),
            "object_position": values.get("obj_position"),
            "relative_distance_m": values.get("obj_distance_m"),
            "object_speed_kph": values.get("obj_v_kph"),
        }
        road = record["mappings"]["road_user_type"].get("canonical_value", "")
        collision = record["mappings"]["collision_type"].get("canonical_value", "")
        if road:
            template_facts["road_user_type"] = road
        if collision:
            template_facts["collision_type"] = collision
        for field, value in template_facts.items():
            if value in (None, ""):
                continue
            facts[field] = value
            provenance[field] = {
                "provenance": "SCENARIO_DEFINED",
                "origin": "SCENARIO_DEFINED",
                "approval": "FINALIZED",
                "validation_status": "VALIDATED",
                "source_refs": [template_source],
                "analysis_assumption_origin": "SCENARIO_DEFINED",
                "analysis_assumption_scope": scope,
                "applicable_scope": scope,
                "parent_scenario_id": record["parent_scenario_id"],
                "template_id": record["template_id"],
                "option_id": record["option_id"],
            }
        for field, entry in assumptions.items():
            facts[field] = entry["value"]
            provenance[field] = {
                "provenance": "SCENARIO_DEFINED",
                "origin": "SCENARIO_DEFINED",
                "approval": "FINALIZED",
                "validation_status": "VALIDATED",
                "source_type": "ENGINEERING_ANALYSIS_SETTING",
                "source_id": str(entry["source_id"]),
                "source_ref": {"group_id": group["group_id"], "field": field},
                "source_refs": [{
                    "source_type": "ENGINEERING_ANALYSIS_SETTING",
                    "source_id": str(entry["source_id"]),
                    "location": f"{group['group_id']}:{field}",
                }],
                "analysis_assumption_origin": "SCENARIO_DEFINED",
                "analysis_assumption_scope": scope,
                "applicable_scope": scope,
                "parent_scenario_id": record["parent_scenario_id"],
                "template_id": record["template_id"],
                "option_id": record["option_id"],
            }
        return {
            "scenario_id": child_id,
            "source_scenario_id": record["parent_scenario_id"],
            "operating_mode": record.get("operating_mode", ""),
            "facts": facts,
            "fact_provenance": provenance,
            "analysis_instance": {
                "instance_id": child_id,
                "parent_scenario_id": record["parent_scenario_id"],
                "template_id": record["template_id"],
                "option_id": record["option_id"],
                "assumption_group_id": group["group_id"],
                "validation_status": "VALIDATED",
            },
        }

    def generate(
        self, *, checkpoint: dict[str, Any], supplement: dict[str, Any],
        assumption_pack: dict[str, Any] | None = None, checkpoint_sha256: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scenarios = {
            str(item.get("scenario_id", "")): item
            for item in checkpoint.get("scenarios", []) if isinstance(item, dict)
        }
        malfunctions = {
            str(item.get("malfunction_id", "")): item
            for item in checkpoint.get("malfunctions", []) if isinstance(item, dict)
        }
        assessments = {
            (str(item.get("malfunction_id", "")), str(item.get("scenario_id", ""))): item
            for item in checkpoint.get("item_definition", {}).get("scenario_assessments", [])
            if isinstance(item, dict) and evaluate_risk_eligibility_payload(item).eligible
        }
        base_records: list[dict[str, Any]] = []
        contexts: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        for option in supplement.get("analytical_options_pending_validation", []):
            if not isinstance(option, dict):
                continue
            key = (str(option.get("malfunction_id", "")), str(option.get("parent_scenario_id", "")))
            assessment = assessments.get(key)
            scenario = scenarios.get(key[1])
            if assessment is None or scenario is None:
                continue
            record = self._record(
                option=option, assessment=assessment, scenario=scenario,
                malfunction=malfunctions.get(key[0], {}),
            )
            base_records.append(record)
            contexts[id(record)] = (option, assessment, scenario, malfunctions.get(key[0], {}))

        assumption_groups = self._assumption_groups(base_records)
        entries, validation_errors, complete_groups = self._validate_assumption_pack(
            pack=assumption_pack or {}, group_specs=assumption_groups,
        )
        errors_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for error in validation_errors:
            errors_by_group[str(error.get("group_id", ""))].append(error)
        group_by_record = {
            id(record): group for group in assumption_groups for record in group["records"]
        }
        records: list[dict[str, Any]] = []
        materialized_children: list[dict[str, Any]] = []
        for base_record in base_records:
            group = group_by_record[id(base_record)]
            if group["group_id"] not in complete_groups:
                base_record["assumption_group_id"] = group["group_id"]
                base_record["assumption_validation_errors"] = errors_by_group[group["group_id"]]
                base_record["readiness_statuses"] = list(
                    self.readiness_statuses(base_record)
                )
                records.append(base_record)
                continue
            option, assessment, parent, malfunction = contexts[id(base_record)]
            child = self._materialize_child(
                record=base_record, option=option, group=group,
                assumptions=entries[group["group_id"]],
            )
            refreshed = self._record(
                option=option, assessment=assessment, scenario=child,
                parent_scenario=parent, malfunction=malfunction,
            )
            refreshed["assumption_group_id"] = group["group_id"]
            refreshed["materialization"] = {
                "status": "MATERIALIZED",
                "child_scenario_id": child["scenario_id"],
            }
            refreshed["readiness_statuses"] = list(
                self.readiness_statuses(refreshed)
            )
            records.append(refreshed)
            materialized_children.append(child)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[self._canonical_json(record["semantic_fingerprint"])].append(record)
        groups = []
        classifications: Counter[str] = Counter()
        delta_classes: Counter[str] = Counter()
        for fingerprint, items in sorted(grouped.items()):
            statuses = {self._record_queue_status(item) for item in items}
            status = next(iter(statuses)) if len(statuses) == 1 else "SOURCE_CONFLICT"
            classifications[status] += 1
            delta_classes.update(item["delta"]["classification"] for item in items)
            groups.append({
                "semantic_fingerprint": json.loads(fingerprint),
                "classification": status,
                "option_count": len(items),
                "options": [item["option_id"] for item in items],
                "causal": {
                    "classes": sorted({item["delta"]["classification"] for item in items}),
                    "causal_revalidation_required": any(
                        item["delta"]["causal_revalidation_required"] for item in items
                    ),
                    "causal_reuse_basis": sorted({
                        item["delta"]["causal_reuse_basis"]
                        for item in items if item["delta"]["causal_reuse_basis"]
                    }),
                },
                "blockers": sorted({
                    code for item in items for code in item["S"]["blockers"] + item["C"]["blockers"]
                }),
            })

        project_inventory: dict[str, Counter[str]] = {
            field: Counter(record["project_fact_inventory"][field]["status"] for record in records)
            for field in self._PROJECT_FACT_FIELDS
        }
        summary = {
            "historical_eligible_relations": len(assessments),
            "template_options": len(records),
            "road_user_mapped": sum(record["mappings"]["road_user_type"]["canonical_value"] != "" for record in records),
            "road_user_unresolved": sum(record["mappings"]["road_user_type"]["canonical_value"] == "" for record in records),
            "collision_mapped": sum(record["mappings"]["collision_type"]["canonical_value"] != "" for record in records),
            "collision_unresolved": sum(record["mappings"]["collision_type"]["canonical_value"] == "" for record in records),
            "ego_point_speed_existing": sum(record["S"]["ego_speed"]["status"] == "EXISTING_PROJECT_FACT" for record in records),
            "ego_point_speed_needs_engineering_decision": sum(record["S"]["ego_speed"]["status"] == "MISSING" for record in records),
            "motion_relation_deterministic": sum(record["motion"]["status"] == "DERIVED_PHYSICS" for record in records),
            "motion_relation_unresolved": sum(record["motion"]["status"] != "DERIVED_PHYSICS" for record in records),
            "s_ready": sum(record["S"]["ready"] for record in records),
            "s_mapping_blocked": sum(any("UNMAPPED" in item for item in record["S"]["blockers"]) for record in records),
            "s_engineering_blocked": sum(
                self._record_queue_status(record) == "BLOCKED_ENGINEERING_ASSUMPTION"
                for record in records
            ),
            "s_motion_blocked": sum(any(
                item in {"LONGITUDINAL_DIRECTION_MISSING", "OBJECT_POINT_SPEED_MISSING", "LATERAL_OR_UNMAPPED_COLLISION_PHYSICS_UNRESOLVED"}
                for item in record["S"]["blockers"]
            ) for record in records),
            "s_source_conflict_blocked": sum(
                self._record_queue_status(record) == "SOURCE_CONFLICT" for record in records
            ),
            "s_causal_validation_blocked": sum(record["S"]["ready"] and record["delta"]["causal_revalidation_required"] for record in records),
            "c_override_resolved": sum(record["C"]["override_state"] != "UNKNOWN" for record in records),
            "c_override_finalized": sum(record["C"]["override_finalized"] for record in records),
            "c_ttc_ready": sum(record["C"]["ttc_ready"] for record in records),
            "c_engineering_blocked": sum(any("ENGINEERING_ASSUMPTION_REQUIRED" in item for item in record["C"]["blockers"]) for record in records),
            "c_source_conflict_blocked": sum(
                self._record_queue_status(record) == "SOURCE_CONFLICT" for record in records
            ),
            "c_unknown_policy_blocked": sum(
                not record["C"]["ready"]
                and not any("ENGINEERING_ASSUMPTION_REQUIRED" in item or "SOURCE_CONFLICT" in item for item in record["C"]["blockers"])
                for record in records
            ),
            "project_fact_inventory": {field: dict(sorted(values.items())) for field, values in project_inventory.items()},
        }
        final_pack = self.refined_assumption_pack(
            checkpoint_sha256=checkpoint_sha256,
            supplement=supplement, records=base_records, existing_pack=assumption_pack,
        )
        final_pack["validation"] = {
            "status": (
                "UNFILLED" if not assumption_pack else
                "VALID" if not validation_errors and complete_groups else "INVALID_OR_INCOMPLETE"
            ),
            "valid_group_count": len(complete_groups),
            "errors": validation_errors,
        }
        payload = {
            "artifact_version": "risk-scoreability-v3",
            "method_contract_hash": self.vocabulary.method_contract_hash,
            "provider_calls": 0,
            "records": records,
            "summary": summary,
            "assumption_pack": final_pack,
            "assumption_validation": final_pack["validation"],
            "materialized_children": materialized_children,
        }
        queue = {
            "artifact_version": "differential-validation-queue-v3",
            "method_contract_hash": self.vocabulary.method_contract_hash,
            "provider_calls": 0,
            "total_options": len(records),
            "deduplicated_semantic_groups": len(groups),
            "deterministic_reuse_groups": classifications["DETERMINISTIC_CAUSAL_REUSE"],
            "provider_ready_groups": classifications["READY_FOR_DIFFERENTIAL_PROVIDER"],
            "engineering_blocked_groups": classifications["BLOCKED_ENGINEERING_ASSUMPTION"],
            "mapping_blocked_groups": classifications["UNRESOLVED_MAPPING"],
            "source_conflict_groups": classifications["SOURCE_CONFLICT"],
            "unclassified_groups": classifications["UNCLASSIFIED"],
            "classification_counts": dict(sorted(classifications.items())),
            "delta_classification_counts": dict(sorted(delta_classes.items())),
            "groups": groups,
        }
        return payload, queue

    def minimal_assumption_pack(
        self, *, checkpoint_sha256: str, supplement: dict[str, Any], records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compatibility entry point now backed by scope-safe grouping."""
        return self.refined_assumption_pack(
            checkpoint_sha256=checkpoint_sha256,
            supplement=supplement,
            records=records,
        )
