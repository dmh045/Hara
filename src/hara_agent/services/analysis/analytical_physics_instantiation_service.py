from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from hara_agent.contracts import AnalyticalPhysicalInput, MethodContract, PhysicalValueAuthority
from hara_agent.models import ScenarioCandidate

from .scenario_physics import (
    TTC_FORMULA_IDENTITY, closing_relative_speed_kph,
    derive_stationary_object_speed, longitudinal_closing_speed_kph,
    select_ego_speed_from_policy,
    time_to_collision_s,
)


class AnalyticalPhysicsInstantiationService:
    """Inventory or derive physical inputs only after the causal gate passes."""

    fields = (
        "ego_speed_kph", "object_speed_kph", "relative_distance_m",
        "road_user_type", "collision_type", "ego_longitudinal_direction",
        "object_longitudinal_direction", "object_position",
    )

    def __init__(self, method: MethodContract | None = None):
        self.project_policy = (
            method.metadata.get("project_analysis_policy", {}) if method is not None else {}
        )
        self.method_hash = (
            str(method.metadata.get("method_source_hash", ""))
            if method is not None else ""
        )

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
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0

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
        derived: list[dict[str, Any]] = []
        ego = by_field["ego_speed_kph"]
        obj = by_field["object_speed_kph"]
        distance = by_field["relative_distance_m"]
        ego_direction = by_field["ego_longitudinal_direction"]
        object_direction = by_field["object_longitudinal_direction"]
        collision = by_field["collision_type"]
        relative = closing_relative_speed_kph(
            ego.value, obj.value,
            ego_direction=ego_direction.value,
            object_direction=object_direction.value,
            collision_type=collision.value,
        )
        if relative is not None:
            opposing = (
                object_direction.value != "STATIONARY"
                and ego_direction.value != object_direction.value
            )
            derived.append({
                "field": "relative_speed_kph", "value": round(relative, 6),
                "unit": "km/h", "authority": "DERIVED",
                "derivation": "OPPOSING_SUM" if opposing else "SAME_OR_STATIONARY_ABS_DIFF",
                "inputs": [
                    "ego_speed_kph", "object_speed_kph",
                    "ego_longitudinal_direction", "object_longitudinal_direction",
                ],
            })
            closing = longitudinal_closing_speed_kph(
                ego.value, obj.value,
                ego_direction=ego_direction.value,
                object_direction=object_direction.value,
                object_position=by_field["object_position"].value,
                collision_type=collision.value,
            )
            ttc = time_to_collision_s(distance.value, closing)
            if ttc is not None:
                derived.append({
                    "field": "ttc_s",
                    "value": ttc,
                    "unit": "s", "authority": "DERIVED",
                    "derivation": TTC_FORMULA_IDENTITY,
                    "closing_speed_kph": closing,
                    "inputs": ["relative_distance_m", "ego_speed_kph", "object_speed_kph",
                               "ego_longitudinal_direction", "object_longitudinal_direction",
                               "object_position"],
                })
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
