from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from dataclasses import replace, fields as dataclass_fields
from typing import Any

from hara_agent.contracts import AnalyticalPhysicalInput, MethodContract, PhysicalValueAuthority
from hara_agent.models import ScenarioCandidate
from copy import deepcopy

from .scenario_physics import (
    derive_scenario_physics,
    derive_stationary_object_speed, longitudinal_closing_speed_kph,
    select_ego_speed_from_policy,
    time_to_collision_s,
    derive_stationary_object_direction, restore_template_ego_direction,
    source_is_accepted_for,
    physical_semantics,
)


class AnalyticalPhysicsInstantiationService:
    """Produce analytical conditions before causality; inventory them after it."""

    fields = (
        "ego_speed_kph", "object_speed_kph", "relative_distance_m",
        "road_user_type", "collision_type", "ego_longitudinal_direction",
        "object_longitudinal_direction", "object_position",
    )

    def __init__(self, method: MethodContract | None = None):
        self.method = method
        self.project_policy = (
            method.metadata.get("project_analysis_policy", {}) if method is not None else {}
        )
        self.method_hash = (
            str(method.metadata.get("method_source_hash", ""))
            if method is not None else ""
        )

    def produce_inputs(self, scenario: ScenarioCandidate, malfunction: dict[str, Any], *, project_facts=None) -> ScenarioCandidate:
        """Produce finite analytical conditions in the ordinary candidate/risk path.

        Facts are constructed from compiled policy and the selected exact
        template. This does not approve a causal judgement or guess a vehicle
        fault limit. Unsupported mechanisms retain their existing dimensions.
        """
        required_mf = {"malfunction_id", "function_id", "guideword", "description", "functional_effect", "vehicle_level_hazard", "causal_chain"}
        if scenario.analysis_instance.get("physical_hypothesis"):
            # The workflow has already instantiated and causally assessed this
            # conditional world. Do not reapply a template's motion defaults.
            return scenario
        if (self.method is None or self.method.structured_risk_method is None
                or not required_mf.issubset(malfunction)):
            return scenario
        candidate = replace(scenario, facts=deepcopy(scenario.facts), fact_provenance=deepcopy(scenario.fact_provenance))
        mf_id = str(malfunction.get("malfunction_id", ""))
        facts, provenance = candidate.facts, candidate.fact_provenance
        if project_facts is not None:
            from .method_risk_fact_service import MethodRiskFactBindingService
            binding = MethodRiskFactBindingService(self.method).bind(project_facts,
                {**facts, "scenario_id": candidate.scenario_id, "malfunction_id": mf_id,
                 "operating_mode": candidate.operating_mode})
            for key in binding.audit["bound_physical_quantities"]:
                if key in facts and facts[key] != binding.values[key]:
                    provenance[key] = {**provenance.get(key, {}), "approval": "PENDING",
                                       "validation_status": "FACT_SOURCE_CONFLICT"}
                else:
                    facts[key], provenance[key] = binding.values[key], binding.provenance[key]
        before = {key: facts.get(key) for key in (*self.fields, "collision_branches", "fault_motion_model", "impact_time_id", "ego_impact_speed_kph", "relative_speed_kph")}
        templates = self.method.scenario_model.scenario_method.fm_template_catalog
        direction = restore_template_ego_direction(candidate, templates.templates if templates else ())
        if direction:
            facts["ego_longitudinal_direction"], provenance["ego_longitudinal_direction"] = direction
        for key, selected in (
            ("ego_speed_kph", select_ego_speed_from_policy(candidate, policy=self.project_policy, malfunction_id=mf_id)),
            ("object_speed_kph", derive_stationary_object_speed(candidate, method_hash=self.method_hash, malfunction_id=mf_id)),
        ):
            if selected:
                facts[key], provenance[key] = selected
        stationary = derive_stationary_object_direction(candidate, malfunction_id=mf_id)
        if stationary:
            facts["object_longitudinal_direction"], provenance["object_longitudinal_direction"] = stationary
        scope = {"malfunction_id": mf_id, "scenario_id": candidate.scenario_id}
        template_ids = {str(meta.get("source_template_id", "")) for meta in provenance.values() if isinstance(meta, dict)}
        template_ids.add(str(candidate.analysis_instance.get("fm_scenario_template_id", "")))
        from .scenario_method_service import ScenarioMethodService
        from hara_agent.models import MalfunctionCandidate
        mf_fields = {field.name for field in dataclass_fields(MalfunctionCandidate)}
        matched = ScenarioMethodService(self.method).match_fm_template(
            MalfunctionCandidate(**{key: value for key, value in malfunction.items() if key in mf_fields})) if malfunction.get("description") else None
        supported_template_ids = {matched.template.template_id} if matched and matched.injectable and matched.template else set()
        template_ids.intersection_update(supported_template_ids)
        invalid_fields = []
        for key, meta in provenance.items():
            if (isinstance(meta, dict) and meta.get("source_template_id")
                    and meta["source_template_id"] not in supported_template_ids):
                meta.update(approval="PENDING", validation_status="TEMPLATE_MECHANISM_NOT_SUPPORTED")
                invalid_fields.append(key)
        chosen = None
        required = ("ego_speed_kph", "object_speed_kph", "road_user_type", "collision_type")
        accepted = all(key in facts and source_is_accepted_for(provenance.get(key, {}), **scope) for key in required)
        for rule in self.project_policy.get("representative_motion_rules", []):
            if (accepted and rule.get("status") == "CONFIRMED_FOR_CURRENT_PROJECT"
                    and template_ids.intersection(rule.get("template_ids", []))
                    and malfunction.get("failure_type") in rule.get("failure_types", [])
                    and candidate.operating_mode in rule.get("operating_modes", [])
                    and facts.get("object_speed_kph") == rule.get("object_speed_kph")
                    and self._point(facts.get("ego_speed_kph")) and facts["ego_speed_kph"] > 0):
                closing = longitudinal_closing_speed_kph(facts["ego_speed_kph"], facts["object_speed_kph"],
                    ego_direction=facts.get("ego_longitudinal_direction"), object_direction=facts.get("object_longitudinal_direction"),
                    object_position=facts.get("object_position"), collision_type=facts.get("collision_type"))
                if closing is not None and closing > 0:
                    chosen = rule
                    break
        identity = hashlib.sha256(self._canonical_json({"scenario": candidate.scenario_id, "malfunction": mf_id,
            "object": facts.get("road_user_type"), "position": facts.get("object_position")}).encode()).hexdigest()[:16]
        impact_id = facts.get("impact_time_id") or f"IMPACT-{identity}"
        # A source explicitly describing constant post-fault acceleration can
        # select the existing neutral model. Normal actuator capability is not
        # such a statement and cannot authorize this path.
        acceleration_meta = provenance.get("longitudinal_acceleration_mps2", {})
        dynamic_inputs = ("ego_speed_kph", "object_speed_kph", "relative_distance_m",
                          "longitudinal_acceleration_mps2", "ego_longitudinal_direction", "object_position", "collision_type")
        dynamic = (not facts.get("fault_motion_model")
            and physical_semantics(acceleration_meta).get("analysis_value_semantic") == "POST_FAULT_CONSTANT_ACCELERATION"
            and all(key in facts and source_is_accepted_for(provenance.get(key, {}), **scope) for key in dynamic_inputs)
            and facts.get("object_speed_kph") == 0
            and (str(facts.get("ego_longitudinal_direction", "")).upper(), str(facts.get("object_position", "")).upper()) in {("FORWARD", "FRONT"), ("REVERSE", "REAR")})
        if dynamic:
            meta = {"provenance": "SCENARIO_DEFINED", "approval": "FINALIZED", "validation_status": "VALIDATED",
                "applicable_scope": scope, "source_refs": [ref for key in dynamic_inputs for ref in provenance[key].get("source_refs", [])],
                "producer": "AnalyticalPhysicsInstantiationService.produce_inputs", "impact_time_id": impact_id,
                "analysis_value_semantic": "POST_FAULT_MOTION", "selection_basis": "SOURCE_EXPLICIT_CONSTANT_ACCELERATION"}
            for key, value in {"fault_motion_model": "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE",
                               "travel_to_impact_m": facts["relative_distance_m"], "impact_time_id": impact_id,
                               "object_impact_speed_kph": facts["object_speed_kph"]}.items():
                facts[key], provenance[key] = value, deepcopy(meta)
        if chosen and not facts.get("fault_motion_model") and facts.get("relative_speed_kph") is None:
            refs = [ref for key in required for ref in provenance.get(key, {}).get("source_refs", [])]
            refs.append({"source_type": "method_contract", "source_id": self.method_hash,
                         "location": str(chosen["source"]), "excerpt": chosen["condition"]})
            meta = {"provenance": "SCENARIO_DEFINED", "origin": "SCENARIO_DEFINED", "approval": "FINALIZED",
                    "validation_status": "VALIDATED", "applicable_scope": scope, "source_refs": refs,
                    "method_contract_hash": self.method_hash, "producer": "AnalyticalPhysicsInstantiationService.produce_inputs",
                    "source_kind": "METHOD_REPRESENTATIVE_CONDITION", "analysis_value_semantic": "IMPACT_REPRESENTATIVE",
                    "impact_time_id": impact_id, "selection_basis": chosen["rule_id"], "condition": chosen["condition"]}
            for key, value in {"fault_motion_model": chosen["model"], "impact_time_id": impact_id,
                               "ego_impact_speed_kph": facts["ego_speed_kph"], "object_impact_speed_kph": facts["object_speed_kph"]}.items():
                facts[key] = value; provenance[key] = deepcopy(meta)
        for record in derive_scenario_physics(candidate):
            key = record.evidence_ref.removeprefix("DERIVED.")
            if key not in facts or provenance.get(key, {}).get("provenance") == "DERIVED":
                facts[key] = record.value
                provenance[key] = {**record.metadata, "provenance": record.provenance.value,
                    "approval": record.approval_status.value, "source_refs": [vars(ref) for ref in record.source_refs]}
        contact_outcome = facts.get("motion_model_outcome", {}).get("status")
        contact_supported = (contact_outcome == "CONTACT" if contact_outcome else facts.get("closing_speed_kph") != 0)
        if (accepted and contact_supported and self._point(facts.get("relative_speed_kph"))
                and source_is_accepted_for(provenance.get("relative_speed_kph", {}), **scope)
                and "collision_branches" not in facts):
            # A selected object is already one finite member; do not add objects
            # from the whole catalog just to manufacture a maximum.
            mechanism = str(facts.get("harm_mechanism_id") or f"{mf_id}:CONTACT")
            branch_id = f"BRANCH-{identity}"
            facts["harm_mechanism_id"] = mechanism
            facts["impact_time_id"] = impact_id
            facts["collision_branches"] = [{"branch_id": branch_id, "mechanism_id": mechanism,
                "impact_time_id": impact_id, "physically_possible": True,
                "road_user_type": facts["road_user_type"], "collision_type": facts["collision_type"],
                "relative_speed_kph": facts["relative_speed_kph"],
                "_fact_provenance": {key: deepcopy(provenance[key]) for key in (*required, "relative_speed_kph") if key in provenance}}]
            provenance["collision_branches"] = {"provenance": "SCENARIO_DEFINED", "origin": "SCENARIO_DEFINED",
                "approval": "FINALIZED", "validation_status": "VALIDATED", "applicable_scope": scope,
                "source_refs": [ref for key in required for ref in provenance.get(key, {}).get("source_refs", [])],
                "producer": "AnalyticalPhysicsInstantiationService.produce_inputs", "method_contract_hash": self.method_hash,
                "selection_basis": "EXISTING_TARGET_AND_SOURCE_VALID_MOTION", "causal_approval": "NOT_GRANTED_BY_PRODUCER"}
        diagnostics = {"producer": "AnalyticalPhysicsInstantiationService.produce_inputs", "before": before,
            "after": {key: facts.get(key) for key in before}, "rule_id": chosen.get("rule_id", "") if chosen else "",
            "pending_reason": "" if facts.get("collision_branches") else (
                "TEMPLATE_MECHANISM_NOT_SUPPORTED" if invalid_fields else
                "MOTION_MODEL_OR_ENVELOPE_NOT_SOURCED" if accepted else "TARGET_OR_PHYSICAL_SOURCE_UNRESOLVED"),
            "template_fields_requiring_revalidation": invalid_fields,
            "source_refs": {key: provenance.get(key, {}).get("source_refs", []) for key in before if facts.get(key) is not None}}
        if facts == scenario.facts and provenance == scenario.fact_provenance and not candidate.analysis_instance:
            return scenario
        candidate.analysis_instance = {**candidate.analysis_instance, "physical_input_production": diagnostics}
        return candidate

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _authority(metadata: dict[str, Any]) -> PhysicalValueAuthority:
        origin = str(metadata.get("origin", metadata.get("provenance", ""))).upper()
        return {
            "PROJECT_INPUT": PhysicalValueAuthority.PROJECT_FACT,
            "PROJECT_FACT": PhysicalValueAuthority.PROJECT_FACT,
            "METHOD_DEFINED": PhysicalValueAuthority.METHOD_DEFINED,
            "SCENARIO_DEFINED": PhysicalValueAuthority.SCENARIO_DEFINED,
            "DERIVED": PhysicalValueAuthority.DERIVED,
            "ENGINEERING_ANALYSIS_ASSUMPTION": PhysicalValueAuthority.ENGINEERING_ANALYSIS_ASSUMPTION,
            "HUMAN_CONFIRMATION": PhysicalValueAuthority.HUMAN_CONFIRMATION,
        }.get(origin, PhysicalValueAuthority.UNAVAILABLE)

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    def _direct_input(
        self, scenario: ScenarioCandidate, field: str, malfunction_id: str,
    ) -> AnalyticalPhysicalInput:
        value = scenario.facts.get(field)
        metadata = scenario.fact_provenance.get(field, {})
        metadata = metadata if isinstance(metadata, dict) else {}
        authority = self._authority(metadata)
        valid_value = (
            self._point(value) if field.endswith("_kph") or field == "relative_distance_m"
            else isinstance(value, str) and bool(value.strip())
        )
        if authority is not PhysicalValueAuthority.UNAVAILABLE and valid_value:
            return AnalyticalPhysicalInput(
                field=field, authority=authority, value=value,
                unit=("km/h" if field.endswith("_kph") else "m" if field == "relative_distance_m" else ""),
                reason="Existing scoped source-valid physical input retained.",
                selection_basis=str(metadata.get("selection_basis", "EXISTING_TYPED_FACT")),
                source_atom_ids=tuple(scenario.facts.get("scenario_atom_ids", [])),
                review_status=str(metadata.get("approval", "PENDING")),
            )
        if field == "ego_speed_kph":
            chosen = select_ego_speed_from_policy(
                scenario, policy=self.project_policy, malfunction_id=malfunction_id,
            )
            if chosen is not None:
                selected, selected_metadata = chosen
                lower, upper = selected_metadata["effective_range_kph"]
                return AnalyticalPhysicalInput(
                    field=field,
                    authority=PhysicalValueAuthority.SCENARIO_DEFINED,
                    value=selected, unit="km/h",
                    allowed_range={"min": lower, "max": upper, "unit": "km/h"},
                    reason="Current-operation ego speed selected by governed project rule.",
                    selection_basis=selected_metadata["selection_basis"],
                    source_atom_ids=tuple(scenario.facts.get("scenario_atom_ids", [])),
                    review_status="FINALIZED",
                    source_refs=tuple(selected_metadata["source_refs"]),
                    applicable_scope=selected_metadata["applicable_scope"],
                    project_rule_id=selected_metadata["project_rule_id"],
                    project_policy_version=selected_metadata["project_policy_version"],
                )
            envelope = scenario.facts.get("ego_speed_constraint", {})
            if isinstance(envelope, dict):
                lower = envelope.get("min_kph", envelope.get("speed_min_kph"))
                upper = envelope.get("max_kph", envelope.get("speed_max_kph"))
                if lower is not None or upper is not None:
                    return AnalyticalPhysicalInput(
                        field=field,
                        authority=PhysicalValueAuthority.ENGINEERING_ANALYSIS_ASSUMPTION,
                        value=None, unit="km/h",
                        allowed_range={"min": lower, "max": upper, "unit": "km/h"},
                        reason=(
                            "The source ODD/mode range is not a point; a scenario-specific "
                            "engineering decision is required."
                        ),
                        selection_basis="NO_AUTOMATIC_ENDPOINT_OR_MIDPOINT",
                        source_atom_ids=tuple(scenario.facts.get("scenario_atom_ids", [])),
                    )
        return AnalyticalPhysicalInput(
            field=field, authority=PhysicalValueAuthority.UNAVAILABLE,
            reason="No source-valid value is available and no default is permitted.",
            selection_basis="FAIL_CLOSED",
            source_atom_ids=tuple(scenario.facts.get("scenario_atom_ids", [])),
        )

    def instantiate(
        self, *, scenario: ScenarioCandidate, malfunction: dict[str, Any],
        causal_status: str,
    ) -> dict[str, Any]:
        if causal_status != "CAUSAL_REUSE_PROVEN" and causal_status != "CAUSAL_REVALIDATED":
            inputs = [
                AnalyticalPhysicalInput(
                    field=field, authority=PhysicalValueAuthority.UNAVAILABLE,
                    reason="Analytical Physics Instantiation is gated by causal validity.",
                    selection_basis=causal_status,
                )
                for field in self.fields
            ]
            return {
                "scenario_id": scenario.scenario_id,
                "malfunction_id": str(malfunction.get("malfunction_id", "")),
                "stage_status": "NOT_REACHED_CAUSAL_VALIDATION",
                "causal_status": causal_status,
                "inputs": [item.to_dict() for item in inputs],
                "derived": [],
                "scoreability_statuses": ["CAUSAL_REVALIDATION_BLOCKED"],
            }
        malfunction_id = str(malfunction.get("malfunction_id", ""))
        inputs = [self._direct_input(scenario, field, malfunction_id) for field in self.fields]
        by_field = {item.field: item for item in inputs}
        stationary = derive_stationary_object_speed(
            scenario, method_hash=self.method_hash, malfunction_id=malfunction_id,
        )
        if stationary is not None and by_field["object_speed_kph"].authority is PhysicalValueAuthority.UNAVAILABLE:
            _, stationary_metadata = stationary
            replacement = AnalyticalPhysicalInput(
                field="object_speed_kph", authority=PhysicalValueAuthority.DERIVED,
                value=0.0, unit="km/h",
                reason="Selected Method atom explicitly defines a stationary object.",
                selection_basis="METHOD_ATOM_EXPLICIT_STATIONARY_OBJECT",
                source_atom_ids=tuple(stationary_metadata["source_atom_ids"]),
                review_status="FINALIZED",
                source_refs=tuple(stationary_metadata["source_refs"]),
                applicable_scope=stationary_metadata["applicable_scope"],
            )
            inputs = [
                replacement if item.field == "object_speed_kph" else item
                for item in inputs
            ]
            by_field["object_speed_kph"] = replacement
        physics_facts = dict(scenario.facts)
        physics_provenance = dict(scenario.fact_provenance)
        selected = select_ego_speed_from_policy(scenario, policy=self.project_policy, malfunction_id=malfunction_id)
        if selected is not None and "ego_speed_kph" not in physics_facts:
            physics_facts["ego_speed_kph"], physics_provenance["ego_speed_kph"] = selected
        if stationary is not None and "object_speed_kph" not in physics_facts:
            physics_facts["object_speed_kph"], physics_provenance["object_speed_kph"] = stationary
        physics = replace(scenario, facts=physics_facts, fact_provenance=physics_provenance)
        derived = [{
            "field": record.evidence_ref.split(".", 1)[1], "value": record.value,
            "unit": "s" if record.evidence_ref == "DERIVED.ttc_s" else "km/h",
            "authority": "DERIVED", "review_status": record.approval_status.value,
            "derivation": record.metadata.get("formula_identity", record.metadata.get("derivation_type", "")),
            "inputs": record.metadata.get("inputs", []),
        } for record in derive_scenario_physics(physics)
            if record.metadata.get("derivation_type") != "CANONICAL_INPUT_NORMALIZATION"]
        assumption_blocked = any(
            item.authority is PhysicalValueAuthority.ENGINEERING_ANALYSIS_ASSUMPTION
            and item.value is None for item in inputs
        )
        s_ready = all(
            by_field[field].authority is not PhysicalValueAuthority.UNAVAILABLE
            and by_field[field].value not in (None, "")
            for field in ("road_user_type", "collision_type")
        ) and any(item["field"] == "relative_speed_kph" for item in derived)
        ttc_ready = any(item["field"] == "ttc_s" for item in derived)
        statuses = []
        if assumption_blocked:
            statuses.append("PHYSICS_ASSUMPTION_BLOCKED")
        if s_ready:
            statuses.append("S_READY")
        if ttc_ready:
            statuses.append("C_TTC_READY")
        return {
            "scenario_id": scenario.scenario_id,
            "malfunction_id": str(malfunction.get("malfunction_id", "")),
            "stage_status": "COMPLETE" if not assumption_blocked else "PENDING_ENGINEERING_ASSUMPTION",
            "causal_status": causal_status,
            "inputs": [item.to_dict() for item in inputs],
            "derived": derived,
            "scoreability_statuses": statuses,
        }

    def assumption_pack(self, records: list[dict[str, Any]], scenarios: dict[str, ScenarioCandidate],
                        malfunctions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scopes: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.get("stage_status") != "PENDING_ENGINEERING_ASSUMPTION":
                continue
            scenario = scenarios[record["scenario_id"]]
            malfunction = malfunctions.get(record["malfunction_id"], {})
            scope = {
                "function_semantics": malfunction.get("function_id", ""),
                "malfunction_semantics": {
                    "malfunction_id": record["malfunction_id"],
                    "description": malfunction.get("description", ""),
                },
                "parent_scenario_id": scenario.source_scenario_id,
                "selected_atom_set": sorted(scenario.facts.get("scenario_atom_ids", [])),
                "road_user_type": scenario.facts.get("road_user_type", ""),
                "collision_type": scenario.facts.get("collision_type", ""),
                "motion_relation": scenario.facts.get("motion_relation", ""),
                "operating_mode": scenario.operating_mode,
                "control_context": {
                    key: scenario.facts.get(key) for key in (
                        "driver_in_vehicle", "remote_intervention_available",
                        "other_road_user_avoidance_possible",
                    )
                },
            }
            key = self._canonical_json(scope)
            scopes[key] = scope
            grouped[key].append(record)
        groups = []
        for key, items in sorted(grouped.items()):
            pending = []
            for record in items:
                for entry in record.get("inputs", []):
                    if (
                        entry.get("authority") == "ENGINEERING_ANALYSIS_ASSUMPTION"
                        and entry.get("value") is None
                    ):
                        pending.append(entry)
            unique = {
                self._canonical_json({
                    "field": item.get("field"),
                    "allowed_range": item.get("allowed_range", {}),
                    "reason": item.get("reason", ""),
                }): item for item in pending
            }
            group_id = "PHYSICS-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16].upper()
            groups.append({
                "group_id": group_id, "scope": scopes[key],
                "affected_scenario_ids": sorted({item["scenario_id"] for item in items}),
                "fields": [
                    {
                        **item, "authority": "ENGINEERING_ANALYSIS_ASSUMPTION",
                        "review_status": "PENDING", "chosen_value": None,
                    }
                    for item in unique.values()
                ],
            })
        return {
            "artifact_version": "scenario-synthesis-engineering-assumption-pack-v1",
            "groups": groups,
            "summary": {
                "groups": len(groups),
                "fields_requiring_approval": sum(len(item["fields"]) for item in groups),
                "affected_scenarios": len({
                    scenario_id for item in groups for scenario_id in item["affected_scenario_ids"]
                }),
            },
        }

    @staticmethod
    def authority_distributions(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        fields: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for item in record.get("inputs", []):
                fields[str(item.get("field", ""))][str(item.get("authority", ""))] += 1
        return {field: dict(sorted(values.items())) for field, values in sorted(fields.items())}


__all__ = ["AnalyticalPhysicsInstantiationService"]
