from __future__ import annotations

import re
import math
from enum import Enum
from typing import Any

from hara_agent.models import (
    EvidenceKind, EvidenceRecord, FactProvenance, ReviewStatus,
    ScenarioCandidate, SourceRef,
)


class DerivedPhysicsType(str, Enum):
    TTC = "TTC"
    RELATIVE_MOTION = "RELATIVE_MOTION"
    CANONICAL_INPUT_NORMALIZATION = "CANONICAL_INPUT_NORMALIZATION"


_ANALYSIS_ORIGIN = FactProvenance.SCENARIO_DEFINED.value
_FINAL_APPROVALS = {ReviewStatus.FINALIZED.value, "APPROVED"}
TTC_FORMULA_IDENTITY = "relative_distance_m / (relative_speed_kph / 3.6)"
TTC_CLOSING_FORMULA_IDENTITY = "relative_distance_m / (closing_speed_kph / 3.6)"


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def is_lateral_collision(value: Any) -> bool:
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in (
        "SIDE", "LATERAL", "PERPENDICULAR", "CROSSING",
    ))


def closing_relative_speed_kph(
    ego_speed_kph: Any, object_speed_kph: Any, *,
    ego_direction: Any, object_direction: Any, collision_type: Any,
) -> float | None:
    """Return the governed longitudinal relative-speed derivation."""
    ego = _nonnegative_number(ego_speed_kph)
    obj = _nonnegative_number(object_speed_kph)
    ego_dir = str(ego_direction or "").strip().upper()
    obj_dir = str(object_direction or "").strip().upper()
    if (
        ego is None or obj is None
        or ego_dir not in {"FORWARD", "REVERSE"}
        or obj_dir not in {"FORWARD", "REVERSE", "STATIONARY"}
        or is_lateral_collision(collision_type)
    ):
        return None
    opposing = obj_dir != "STATIONARY" and ego_dir != obj_dir
    return round(ego + obj if opposing else abs(ego - obj), 6)


def longitudinal_closing_speed_kph(
    ego_speed_kph: Any, object_speed_kph: Any, *,
    ego_direction: Any, object_direction: Any, object_position: Any,
    collision_type: Any,
) -> float | None:
    """Return positive approach speed only when longitudinal geometry is known."""
    ego = _nonnegative_number(ego_speed_kph)
    obj = _nonnegative_number(object_speed_kph)
    ego_dir = str(ego_direction or "").strip().upper()
    obj_dir = str(object_direction or "").strip().upper()
    position = str(object_position or "").strip().upper()
    if (
        ego is None or obj is None
        or ego_dir not in {"FORWARD", "REVERSE"}
        or obj_dir not in {"FORWARD", "REVERSE", "STATIONARY"}
        or position not in {"FRONT", "REAR"}
        or is_lateral_collision(collision_type)
    ):
        return None
    signed_ego = ego if ego_dir == "FORWARD" else -ego
    signed_obj = 0.0 if obj_dir == "STATIONARY" else (
        obj if obj_dir == "FORWARD" else -obj
    )
    approach = (signed_ego - signed_obj) * (1 if position == "FRONT" else -1)
    return round(max(0.0, approach), 6)


def select_ego_speed_from_policy(
    scenario: ScenarioCandidate, *, policy: dict[str, Any], malfunction_id: str,
) -> tuple[float, dict[str, Any]] | None:
    """Materialize the governed current-operation speed choice with its lineage."""
    rule = policy.get("ego_speed_point_selection", {}) if isinstance(policy, dict) else {}
    if (
        not isinstance(rule, dict)
        or rule.get("choice") != "UPPER_CLOSED_BOUND_OF_EFFECTIVE_OPERATION_RANGE"
        or rule.get("status") != "CONFIRMED_FOR_CURRENT_PROJECT"
        or not str(rule.get("rule_id", "")).strip()
        or scenario.facts.get("ego_speed_kph") is not None
        or scenario.facts.get("ego_speed_value_semantic") == "POST_FAULT_SPEED"
    ):
        return None
    envelope = scenario.facts.get("ego_speed_constraint", {})
    metadata = scenario.fact_provenance.get("ego_speed_constraint", {})
    if not isinstance(envelope, dict) or not isinstance(metadata, dict):
        return None
    lower = envelope.get("min_kph", envelope.get("speed_min_kph"))
    upper = envelope.get("max_kph", envelope.get("speed_max_kph"))
    if (
        isinstance(lower, bool) or not isinstance(lower, (int, float))
        or isinstance(upper, bool) or not isinstance(upper, (int, float))
        or not math.isfinite(lower) or not math.isfinite(upper)
        or lower < 0 or upper < lower
        or envelope.get("upper_inclusive", True) is not True
        or str(metadata.get("approval", "")).upper() not in _FINAL_APPROVALS
        or not source_has_provenance(metadata)
    ):
        return None
    scope = {
        "malfunction_id": malfunction_id,
        "scenario_id": scenario.scenario_id,
        "parent_scenario_id": scenario.source_scenario_id or scenario.scenario_id,
    }
    policy_source = policy.get("source_ref", {})
    refs = list(metadata.get("source_refs", metadata.get("sources", [])))
    if not isinstance(policy_source, dict) or not policy_source.get("source_id"):
        return None
    refs.append({
        **policy_source,
        "location": f"{policy_source.get('location', '')}:ego_speed_point_selection",
        "excerpt": str(rule["rule_id"]),
    })
    return float(upper), {
        "provenance": FactProvenance.SCENARIO_DEFINED.value,
        "origin": FactProvenance.SCENARIO_DEFINED.value,
        "approval": ReviewStatus.FINALIZED.value,
        "validation_status": "VALIDATED",
        "analysis_assumption_origin": FactProvenance.SCENARIO_DEFINED.value,
        "analysis_assumption_scope": scope,
        "applicable_scope": scope,
        "source_refs": refs,
        "input_fact_metadata": [_input_snapshot(
            "ego_speed_constraint", ReviewStatus.FINALIZED,
            _input_metadata(scenario, "ego_speed_constraint")[1], metadata,
        )],
        "selection_basis": "GOVERNED_CLOSED_UPPER_BOUND",
        "analysis_value_semantic": "CURRENT_OPERATION_EGO_SPEED",
        "effective_range_kph": [float(lower), float(upper)],
        "project_policy_id": str(policy.get("policy_id", "")),
        "project_policy_version": str(policy.get("version", "")),
        "project_rule_id": str(rule["rule_id"]),
        "machine_checks": [
            "FINITE_NONNEGATIVE_CLOSED_RANGE", "SOURCE_FINALIZED",
            "SCOPED_TO_MALFUNCTION_AND_SCENARIO",
        ],
    }


def derive_stationary_object_speed(
    scenario: ScenarioCandidate, *, method_hash: str,
    malfunction_id: str,
) -> tuple[float, dict[str, Any]] | None:
    """Use a selected Method atom's explicit stationary motion, if any."""
    if scenario.facts.get("object_speed_kph") is not None:
        return None
    bindings = scenario.facts.get("method_scenario_dimensions", {})
    binding = bindings.get("OBJECT", {}) if isinstance(bindings, dict) else {}
    if not isinstance(binding, dict) or binding.get("resolution_status") != "RESOLVED":
        return None
    atom_id = str(binding.get("atom_id", "")).strip()
    if not atom_id or atom_id not in scenario.facts.get("scenario_atom_ids", []):
        return None
    semantics = binding.get("method_semantics", {})
    semantics = semantics.get("object", semantics) if isinstance(semantics, dict) else {}
    if not isinstance(semantics, dict):
        return None
    motion = str(semantics.get("motion", semantics.get("motion_state", ""))).upper()
    if semantics.get("stationary") is not True and motion != "STATIONARY":
        return None
    atom_source = binding.get("atom_provenance", {})
    if not method_hash or not isinstance(atom_source, dict):
        return None
    asset = str(atom_source.get("source_asset", "")).strip()
    rule = str(atom_source.get("source_rule", "")).strip()
    if not asset or not rule:
        return None
    scope = {
        "malfunction_id": malfunction_id,
        "scenario_id": scenario.scenario_id,
        "parent_scenario_id": scenario.source_scenario_id or scenario.scenario_id,
    }
    return 0.0, {
        "provenance": FactProvenance.DERIVED.value,
        "origin": "METHOD_DEFINED",
        "approval": ReviewStatus.FINALIZED.value,
        "validation_status": "VALIDATED",
        "applicable_scope": scope,
        "source_refs": [{
            "source_type": "method_contract", "source_id": method_hash,
            "location": f"{asset}:{rule}", "excerpt": f"{atom_id}: STATIONARY",
        }],
        "selection_basis": "SELECTED_METHOD_ATOM_EXPLICIT_STATIONARY",
        "source_atom_ids": [atom_id],
        "derivation_rule_id": "SELECTED_METHOD_ATOM_STATIONARY_ZERO_SPEED",
    }


def time_to_collision_s(distance_m: Any, relative_speed_kph: Any) -> float | None:
    """Return TTC for a positive closing speed, otherwise fail closed."""
    distance = _nonnegative_number(distance_m)
    speed = _nonnegative_number(relative_speed_kph)
    if distance is None or speed is None or speed <= 0:
        return None
    return round(distance / (speed / 3.6), 6)


def _distance_m(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*m\s*", str(value or ""), re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _input_metadata(
    scenario: ScenarioCandidate, key: str,
) -> tuple[ReviewStatus, tuple[SourceRef, ...], dict[str, Any]]:
    metadata = scenario.fact_provenance.get(key, {})
    if not isinstance(metadata, dict):
        return ReviewStatus.PENDING, (), {}
    try:
        approval = ReviewStatus(metadata.get("approval", scenario.status.value))
    except ValueError:
        approval = ReviewStatus.PENDING
    sources = tuple(
        item if isinstance(item, SourceRef) else SourceRef(**item)
        for item in metadata.get("source_refs", metadata.get("sources", []))
        if isinstance(item, (SourceRef, dict))
    )
    return approval, sources, metadata


def _analysis_lineage(metadata: dict[str, Any]) -> dict[str, Any]:
    origin = str(metadata.get(
        "analysis_assumption_origin",
        metadata.get("origin", metadata.get("provenance", "")),
    )).upper()
    if origin != _ANALYSIS_ORIGIN:
        return {}
    scope = metadata.get(
        "analysis_assumption_scope", metadata.get("applicable_scope", {}),
    )
    return {
        "analysis_assumption_origin": _ANALYSIS_ORIGIN,
        "analysis_assumption_scope": dict(scope) if isinstance(scope, dict) else scope,
        "validation_status": metadata.get("validation_status", ""),
        "source_template_id": metadata.get("source_template_id", ""),
        "source_option_id": metadata.get("source_option_id", ""),
        "method_contract_hash": metadata.get("method_contract_hash", ""),
    }


def analysis_assumption_lineages(metadata: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return every analytical assumption retained by a fact or derivation."""
    inputs = metadata.get("analysis_assumption_inputs", ())
    if isinstance(inputs, list):
        lineages = [
            _analysis_lineage(item)
            for item in inputs
            if isinstance(item, dict) and _analysis_lineage(item)
        ]
        if lineages:
            return tuple(lineages)
    lineage = _analysis_lineage(metadata)
    return (lineage,) if lineage else ()


def has_analysis_assumption_lineage(metadata: dict[str, Any]) -> bool:
    return bool(analysis_assumption_lineages(metadata))


def source_reference(metadata: dict[str, Any]) -> str:
    """Return the existing RiskContext source-reference representation."""
    sources = metadata.get("source_refs", metadata.get("sources", ()))
    if not isinstance(sources, (list, tuple)) or not sources:
        return ""
    first = sources[0]
    if isinstance(first, dict):
        source_id = str(first.get("source_id", first.get("asset", ""))).strip()
        location = str(first.get("location", "")).strip()
        return f"{source_id}:{location}".strip(":")
    source_id = str(getattr(first, "source_id", "")).strip()
    location = str(getattr(first, "location", "")).strip()
    return f"{source_id}:{location}".strip(":")


def source_has_provenance(metadata: dict[str, Any]) -> bool:
    """Require every retained leaf to carry its own usable source reference."""
    inputs = metadata.get("input_fact_metadata", ())
    if isinstance(inputs, list):
        return bool(inputs) and all(
            source_has_provenance(item) for item in inputs
            if isinstance(item, dict)
        ) and all(isinstance(item, dict) for item in inputs)
    return bool(source_reference(metadata))


def analysis_assumption_is_valid_for(
    metadata: dict[str, Any], *, malfunction_id: str, scenario_id: str,
) -> bool:
    """Validate every retained analytical source against one M x Scenario."""
    lineages = analysis_assumption_lineages(metadata)
    return bool(lineages) and all(
        item["analysis_assumption_origin"] == _ANALYSIS_ORIGIN
        and str(item.get("validation_status", "")).upper() == "VALIDATED"
        and isinstance(item.get("analysis_assumption_scope"), dict)
        and str(item["analysis_assumption_scope"].get("malfunction_id", ""))
        == malfunction_id
        and str(item["analysis_assumption_scope"].get("scenario_id", ""))
        == scenario_id
        for item in lineages
    )


def source_is_accepted_for(
    metadata: dict[str, Any], *, malfunction_id: str, scenario_id: str,
) -> bool:
    """Apply the one source contract used by derived physics and RiskContext."""
    inputs = metadata.get("input_fact_metadata", ())
    if isinstance(inputs, list):
        return bool(inputs) and all(
            source_is_accepted_for(
                item, malfunction_id=malfunction_id, scenario_id=scenario_id,
            )
            for item in inputs if isinstance(item, dict)
        ) and all(isinstance(item, dict) for item in inputs)
    if not source_has_provenance(metadata):
        return False
    if (
        str(metadata.get("provenance", "")).upper() == FactProvenance.DERIVED.value
        and metadata.get("inputs")
    ):
        return False
    if has_analysis_assumption_lineage(metadata):
        return analysis_assumption_is_valid_for(
            metadata, malfunction_id=malfunction_id, scenario_id=scenario_id,
        )
    return str(metadata.get("approval", "")).upper() in _FINAL_APPROVALS


def source_origin(metadata: dict[str, Any]) -> str:
    """Expose fact origin independently from the derived/direct authority axis."""
    if has_analysis_assumption_lineage(metadata):
        return _ANALYSIS_ORIGIN
    if metadata.get("source_binding_kind") == "METHOD_RISK_FACT_BINDING":
        return "METHOD_RISK_FACT_BINDING"
    return str(metadata.get("origin", metadata.get("provenance", ""))).upper()


def _source_dict(source: SourceRef) -> dict[str, str]:
    return {
        "source_type": source.source_type,
        "source_id": source.source_id,
        "location": source.location,
        "excerpt": source.excerpt,
    }


def _input_snapshot(
    key: str, status: ReviewStatus, sources: tuple[SourceRef, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    snapshot = {
        "field": key,
        "provenance": metadata.get("provenance", ""),
        "origin": metadata.get("origin", ""),
        "approval": status.value,
        "source_refs": [_source_dict(item) for item in sources],
    }
    for field in (
        "analysis_assumption_origin", "analysis_assumption_scope",
        "validation_status", "source_template_id", "source_option_id",
        "method_contract_hash", "source_binding_kind", "applicable_scope",
        "input_fact_metadata", "analysis_assumption_inputs",
    ):
        if field in metadata:
            snapshot[field] = metadata[field]
    return snapshot


def _combined_analysis_lineage(
    lineages: list[dict[str, Any]],
) -> dict[str, Any]:
    if not lineages:
        return {}
    result = {
        "analysis_assumption_origin": _ANALYSIS_ORIGIN,
        "analysis_assumption_inputs": lineages,
    }
    scopes = {str(item.get("analysis_assumption_scope")) for item in lineages}
    statuses = {str(item.get("validation_status", "")) for item in lineages}
    if len(scopes) == 1:
        result["analysis_assumption_scope"] = lineages[0]["analysis_assumption_scope"]
    if len(statuses) == 1:
        result["validation_status"] = lineages[0]["validation_status"]
    return result


def derive_scenario_physics(
    scenario: ScenarioCandidate,
) -> tuple[EvidenceRecord, ...]:
    """Derive neutral physics from canonical scenario facts only.

    This module supplies quantities such as TTC.  It never maps a quantity to
    S/E/C or selects a project profile; those decisions remain MethodContract
    authority.
    """

    records: list[EvidenceRecord] = []
    normalized: dict[str, tuple[Any, str]] = {}
    for key in (
        "ego_speed_kph", "relative_speed_kph", "impact_speed_kph", "delta_v_kph",
    ):
        value = scenario.facts.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            continue
        normalized[key] = (float(value), key)
    distance_input_key = (
        "relative_distance_m"
        if "relative_distance_m" in scenario.facts
        else "relative_distance"
    )
    distance_m = _distance_m(scenario.facts.get(distance_input_key))
    if distance_m is not None:
        normalized["relative_distance_m"] = (distance_m, distance_input_key)

    # Preserve canonical numeric inputs as deterministic, source-linked
    # physics facts.  This is normalization only; it does not derive a new
    # collision or select an S/E/C value.
    for output_key, (value, input_key) in normalized.items():
        status, sources, input_metadata = _input_metadata(scenario, input_key)
        metadata = {
            "derivation_type": DerivedPhysicsType.CANONICAL_INPUT_NORMALIZATION.value,
            "inputs": [f"SCN.{input_key}"],
            **_combined_analysis_lineage([
                _analysis_lineage(input_metadata),
            ] if _analysis_lineage(input_metadata) else []),
        }
        if input_metadata:
            metadata["input_fact_metadata"] = [
                _input_snapshot(input_key, status, sources, input_metadata),
            ]
        if scenario.semantic_fingerprint:
            metadata["semantic_fingerprint"] = scenario.semantic_fingerprint
        records.append(EvidenceRecord(
            f"DERIVED.{output_key}", value,
            EvidenceKind.DERIVED_PHYSICS,
            FactProvenance.DERIVED,
            status,
            sources,
            metadata,
        ))
    def derived_record(
        output_key: str, value: float, input_keys: tuple[str, ...],
        derivation_type: DerivedPhysicsType, *, formula: str = "",
    ) -> EvidenceRecord:
        dependencies = [_input_metadata(scenario, key) for key in input_keys]
        refs = tuple(dict.fromkeys(
            source for _, sources, _ in dependencies for source in sources
        ))
        approval = (
            ReviewStatus.FINALIZED
            if all(status is ReviewStatus.FINALIZED for status, _, _ in dependencies)
            else ReviewStatus.PENDING
        )
        metadata = {
            "derivation_type": derivation_type.value,
            "inputs": [f"SCN.{key}" for key in input_keys],
        }
        if formula:
            metadata["formula_identity"] = formula
        if all(item for _, _, item in dependencies):
            metadata["input_fact_metadata"] = [
                _input_snapshot(key, status, sources, item)
                for key, (status, sources, item) in zip(input_keys, dependencies)
            ]
        metadata.update(_combined_analysis_lineage([
            {**_analysis_lineage(item), "input_field": key}
            for key, (_, _, item) in zip(input_keys, dependencies)
            if _analysis_lineage(item)
        ]))
        if scenario.semantic_fingerprint:
            metadata["semantic_fingerprint"] = scenario.semantic_fingerprint
        return EvidenceRecord(
            f"DERIVED.{output_key}", value,
            EvidenceKind.DERIVED_PHYSICS, FactProvenance.DERIVED,
            approval, refs, metadata,
        )

    ego = scenario.facts.get("ego_speed_kph")
    obj = scenario.facts.get("object_speed_kph")
    ego_direction = scenario.facts.get("ego_longitudinal_direction")
    object_direction = scenario.facts.get("object_longitudinal_direction")
    collision = scenario.facts.get("collision_type")
    position = scenario.facts.get("object_position")
    relative_speed = scenario.facts.get("relative_speed_kph")
    calculated_relative = closing_relative_speed_kph(
        ego, obj, ego_direction=ego_direction,
        object_direction=object_direction, collision_type=collision,
    )
    if relative_speed is None and calculated_relative is not None:
        relative_speed = calculated_relative
        records.append(derived_record(
            "relative_speed_kph", calculated_relative,
            ("ego_speed_kph", "object_speed_kph", "ego_longitudinal_direction",
             "object_longitudinal_direction", "collision_type"),
            DerivedPhysicsType.RELATIVE_MOTION,
        ))

    directional_geometry_supplied = any(
        key in scenario.facts for key in (
            "ego_longitudinal_direction", "object_longitudinal_direction",
        )
    )
    if directional_geometry_supplied:
        closing_speed = longitudinal_closing_speed_kph(
            ego, obj, ego_direction=ego_direction,
            object_direction=object_direction, object_position=position,
            collision_type=collision,
        )
        ttc_input_keys = (
            distance_input_key, "ego_speed_kph", "object_speed_kph",
            "ego_longitudinal_direction", "object_longitudinal_direction",
            "object_position", "collision_type",
        )
        formula = TTC_CLOSING_FORMULA_IDENTITY
        if closing_speed is not None:
            records.append(derived_record(
                "closing_speed_kph", closing_speed, ttc_input_keys[1:],
                DerivedPhysicsType.RELATIVE_MOTION,
            ))
    else:
        # An explicit, source-accepted relative speed retains the historical
        # closing-speed meaning only when no contradictory geometry is given.
        closing_speed = (
            relative_speed
            if "relative_speed_kph" in scenario.facts
            and not is_lateral_collision(collision)
            else None
        )
        ttc_input_keys = (distance_input_key, "relative_speed_kph")
        formula = TTC_FORMULA_IDENTITY
    ttc = time_to_collision_s(distance_m, closing_speed)
    if ttc is not None:
        records.append(derived_record(
            "ttc_s", ttc, ttc_input_keys, DerivedPhysicsType.TTC,
            formula=formula,
        ))
    return tuple(records)
