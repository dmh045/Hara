from __future__ import annotations

import re
import math
from dataclasses import replace
from enum import Enum
from typing import Any

from hara_agent.models import (
    EvidenceKind, EvidenceRecord, FactProvenance, ReviewStatus,
    ScenarioCandidate, SourceRef,
)
from hara_agent.models.project_facts import PHYSICAL_SEMANTIC_FIELDS


class DerivedPhysicsType(str, Enum):
    TTC = "TTC"
    RELATIVE_MOTION = "RELATIVE_MOTION"
    CANONICAL_INPUT_NORMALIZATION = "CANONICAL_INPUT_NORMALIZATION"
    KINEMATIC_IMPACT_SPEED = "KINEMATIC_IMPACT_SPEED"


_ANALYSIS_ORIGIN = FactProvenance.SCENARIO_DEFINED.value
CONDITIONAL_ORIGIN = "LLM_ANALYSIS_HYPOTHESIS"
_FINAL_APPROVALS = {ReviewStatus.FINALIZED.value, "APPROVED"}
TTC_CLOSING_FORMULA_IDENTITY = "relative_distance_m / (closing_speed_kph / 3.6)"


def physical_semantics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy normalization wrappers without changing a value's phase."""
    result = {key: metadata[key] for key in PHYSICAL_SEMANTIC_FIELDS if key in metadata}
    if metadata.get("derivation_type") == DerivedPhysicsType.CANONICAL_INPUT_NORMALIZATION.value:
        inputs = metadata.get("input_fact_metadata", [])
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            result = {**physical_semantics(inputs[0]), **result}
    return result


def branch_physics_view(parent: dict, branch: dict) -> dict:
    """Invalidate dependent parent values, then reuse the neutral physics service."""
    view = {**parent, **branch}
    view["_fact_provenance"] = {**parent.get("_fact_provenance", {}), **branch.get("_fact_provenance", {})}
    geometry = {"object_position", "object_id", "road_user_type", "collision_type",
                "ego_longitudinal_direction", "ego_lateral_direction", "object_longitudinal_direction"}
    speeds = {"ego_speed_kph", "object_speed_kph", "ego_impact_speed_kph", "object_impact_speed_kph"}
    model = {"fault_motion_model", "longitudinal_acceleration_mps2", "model_valid_until_s",
             "slope_percent", "residual_resistance_mps2", "lateral_speed_mps",
             "lateral_acceleration_mps2", "lateral_clearance_m", "turn_radius_m", "object_extent",
             "post_fault_speed_cap_kph", "post_stop_motion", "object_offset_x_m", "object_offset_y_m",
             "contact_half_length_m", "contact_half_width_m",
             "grade_coordinate_frame", "slope_reference", "ego_heading_relative_to_downhill",
             "downhill_body_direction", "grade_orientation_basis",
             "impact_time_id", "trajectory_id", "assessment_reference"}
    distances = {"relative_distance_m", "relative_distance", "travel_to_impact_m"}
    changed = {key for key in branch if not key.startswith("_")
               and (key not in parent or parent[key] != branch[key])}
    changed.update(key for key, metadata in branch.get("_fact_provenance", {}).items()
                   if key in view and metadata != parent.get("_fact_provenance", {}).get(key))
    # None explicitly clears a branch input; absent keys still inherit it.
    for key in changed:
        if key in branch and branch[key] is None:
            view.pop(key, None)
            view["_fact_provenance"].pop(key, None)
        elif key in branch and key not in branch.get("_fact_provenance", {}):
            view["_fact_provenance"].pop(key, None)
    # A new identifier/category alone labels the current object; it does not
    # contradict a directly sourced closing velocity. Actual geometry does.
    changed_geometry = changed & (geometry - {"object_id", "road_user_type", "collision_type"})
    changed_geometry |= changed & {"object_id", "road_user_type", "collision_type"} & parent.keys()
    if "collision_type" in changed and is_lateral_collision(view.get("collision_type")):
        changed_geometry.add("collision_type")
    phase_ids = {"impact_time_id", "trajectory_id"}
    changed_model = changed & (model - phase_ids)
    if view.get("fault_motion_model") or any(key in parent for key in ("ego_impact_speed_kph", "object_impact_speed_kph")):
        changed_model |= changed & phase_ids
    invalidate = set()
    if changed_geometry or changed & speeds or changed_model:
        invalidate.update(("relative_speed_kph", "closing_speed_kph", "ttc_s", "first_contact_time_s", "trajectory_max_speed_kph"))
    if changed & (distances | {"closing_speed_kph", "relative_speed_kph"}):
        invalidate.add("ttc_s")
    if changed_model or (changed & distances and view.get("fault_motion_model")):
        invalidate.update(("ego_impact_speed_kph", "relative_speed_kph", "ttc_s"))
    if view.get("fault_motion_model") and (changed_model or changed_geometry or changed & speeds):
        invalidate.update(("ego_impact_speed_kph", "object_impact_speed_kph"))
    # Preserve recorded dependencies as well as the neutral model's inputs.
    # Source/phase changes invalidate transitive dependants even at equal values.
    while True:
        dependants = {key for key, metadata in parent.get("_fact_provenance", {}).items()
                      if isinstance(metadata, dict) and any(
                          ref.removeprefix("SCN.").removeprefix("DERIVED.") in changed | invalidate
                          and ref.removeprefix("SCN.").removeprefix("DERIVED.") != key
                          for ref in metadata.get("inputs", []) if isinstance(ref, str))}
        if dependants <= invalidate:
            break
        invalidate.update(dependants)
    for key in invalidate - branch.keys():
        view.pop(key, None); view["_fact_provenance"].pop(key, None)
    candidate = ScenarioCandidate(str(view.get("scenario_id", "branch")), "branch", "branch", "branch",
        facts={key: value for key, value in view.items() if not key.startswith("_")},
        fact_provenance=view["_fact_provenance"])
    records = derive_scenario_physics(candidate)
    conflicts = []
    outcome = next((r.value for r in records if r.evidence_ref == "DERIVED.motion_model_outcome"), {})
    if outcome and outcome.get("status") != "CONTACT":
        for key in ("ego_impact_speed_kph", "object_impact_speed_kph", "relative_speed_kph", "ttc_s", "first_contact_time_s", "trajectory_max_speed_kph"):
            if key in view:
                view.pop(key)
                view["_fact_provenance"].pop(key, None)
                invalidate.add(key)
        conflicts.append({"reason": outcome["status"], "field": "fault_motion_model"})
    for record in records:
        key = record.evidence_ref.removeprefix("DERIVED.")
        if outcome and outcome.get("status") != "CONTACT" and key in {
            "ego_impact_speed_kph", "object_impact_speed_kph", "relative_speed_kph", "ttc_s", "first_contact_time_s", "trajectory_max_speed_kph",
        }:
            continue
        # A directly supplied branch measurement must not be silently overwritten
        # by a different computation for that same branch and time.
        if (key in branch and branch[key] is not None
                and record.metadata.get("derivation_type") != DerivedPhysicsType.CANONICAL_INPUT_NORMALIZATION.value
                and branch[key] != record.value):
            conflicts.append({"reason": "DIRECT_INPUT_CONFLICTS_WITH_DERIVATION", "field": key,
                              "supplied": branch[key], "derived": record.value})
            view["_fact_provenance"][key] = {**view["_fact_provenance"].get(key, {}),
                "approval": "PENDING", "conflicting_derived_value": record.value}
            continue
        if key in view and key not in invalidate:
            continue
        view[key] = record.value
        view["_fact_provenance"][key] = {**record.metadata,
            "provenance": record.provenance.value, "approval": record.approval_status.value,
            "source_refs": [_source_dict(ref) for ref in record.source_refs]}
    view["_invalidated_parent_fields"] = sorted(invalidate - branch.keys())
    view["_physical_input_conflicts"] = conflicts
    return view


def constant_acceleration_contact(initial_kph: Any, acceleration: Any,
                                  distance_m: Any, valid_until_s: Any = None,
                                  speed_cap_kph: Any = None) -> dict:
    """First contact along travel, ending at the first stop (no implicit reversal)."""
    initial = _nonnegative_number(initial_kph)
    distance = _nonnegative_number(distance_m)
    if (initial is None or distance is None or isinstance(acceleration, bool)
            or not isinstance(acceleration, (int, float)) or not math.isfinite(acceleration)):
        return {"status": "MISSING_MODEL_INPUT"}
    speed = initial / 3.6
    squared = speed * speed + 2 * acceleration * distance
    if squared < 0:
        return {"status": "STOPPED_BEFORE_CONTACT"}
    end_speed = math.sqrt(squared)
    if distance == 0:
        time = 0.0
    elif speed + end_speed <= 0:
        return {"status": "NO_POSITIVE_CONTACT_ROOT"}
    else:
        time = 2 * distance / (speed + end_speed)
    capped = False
    if speed_cap_kph is not None:
        cap = _nonnegative_number(speed_cap_kph)
        if cap is None or cap < initial:
            return {"status": "SPEED_CAP_INITIAL_STATE_INCOMPATIBLE"}
        cap /= 3.6
        if acceleration > 0 and end_speed > cap:
            if cap == 0:
                return {"status": "NO_POSITIVE_CONTACT_ROOT"}
            to_cap_s = (cap - speed) / acceleration
            to_cap_m = (speed + cap) * to_cap_s / 2
            time = to_cap_s + (distance - to_cap_m) / cap
            end_speed, capped = cap, True
    if valid_until_s is not None:
        horizon = _nonnegative_number(valid_until_s)
        if horizon is None or time > horizon:
            return {"status": "CONTACT_OUTSIDE_MODEL_INTERVAL"}
    return {"status": "CONTACT", "time_s": time, "impact_speed_kph": end_speed * 3.6,
            **({"speed_cap_active": True} if capped else {})}


def restore_template_ego_direction(scenario: ScenarioCandidate, templates) -> tuple[str, dict] | None:
    """Recover a compiled explicit maneuver from this candidate's exact option.

    This also repairs historical children without guessing direction from the
    object's position. Existing direction is retained for conflict review.
    """
    if scenario.facts.get("ego_longitudinal_direction"):
        return None
    for metadata in scenario.fact_provenance.values():
        if not isinstance(metadata, dict) or metadata.get("selection_basis") != "EXACT_ACTIVE_OPTION":
            continue
        option_id = metadata.get("source_option_id", "")
        for template in templates:
            for index, option in enumerate(template.required_scenarios, 1):
                if option_id == f"{template.template_id}:OPTION:{index}" and option.ego_longitudinal_direction:
                    return option.ego_longitudinal_direction, {**metadata,
                        "source_refs": [{"source_type": "method_contract", "source_id": option.source_ref.source_hash,
                                         "location": f"{option.source_ref.sheet}!{option.source_ref.range}", "excerpt": option.label}],
                        "selection_basis": "COMPILED_EXPLICIT_TEMPLATE_MANEUVER"}
    return None


def _nonnegative_number(value: Any) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        return None
    return float(value)


def is_lateral_collision(value: Any) -> bool:
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in (
        "SIDE", "LATERAL", "PERPENDICULAR", "CROSSING",
    ))


def constant_speed_curved_contact(facts: dict) -> dict:
    """Constant-curvature point path to an effective extended side boundary.

    The speed is the norm, not a fixed inertial x component. This is a
    no-slip conditional kinematic case, not a tire/grip capability model.
    Never infer radius from a pre-fault lateral-acceleration exposure Atom.
    """
    fields = ("ego_speed_kph", "turn_radius_m", "lateral_clearance_m", "model_valid_until_s")
    values = [_nonnegative_number(facts.get(k)) for k in fields]
    if any(v is None for v in values):
        return {"status": "MODEL_INPUTS_UNRESOLVED"}
    speed, radius, clearance, horizon = values
    direction, position = facts.get("ego_lateral_direction"), facts.get("object_position")
    if (min(values) <= 0 or direction not in {"LEFT", "RIGHT"} or position not in {"LEFT", "RIGHT"}
            or facts.get("ego_longitudinal_direction") not in {"FORWARD", "REVERSE"}
            or facts.get("object_extent") != "EXTENDED_ALONG_TRAVEL_PATH"
            or facts.get("object_speed_kph") != 0 or not is_lateral_collision(facts.get("collision_type"))):
        return {"status": "MODEL_GEOMETRY_NOT_APPLICABLE"}
    v = speed / 3.6
    end = min(horizon, math.pi * radius / (2 * v))
    no_contact = {"status": "NO_CONTACT_WITHIN_MODEL", "evaluated_until_s": end,
                  "limiting_condition": "MODEL_HORIZON_OR_FIRST_QUARTER_TURN"}
    if direction != position:
        return {**no_contact, "reason": "TARGET_OPPOSITE_CURVATURE"}
    if clearance > radius:
        return no_contact
    # 2*asin(sqrt(w/(2R))) avoids loss of precision for w << R.
    angle = 2 * math.asin(math.sqrt(clearance / (2 * radius)))
    time = radius * angle / v
    if time > horizon:
        return no_contact
    return {"status": "CONTACT", "time_s": time, "impact_speed_kph": speed,
            "trajectory_max_speed_kph": speed, "heading_change_rad": angle,
            "contact_x_m": (1 if facts["ego_longitudinal_direction"] == "FORWARD" else -1) * radius * math.sin(angle),
            "contact_y_m": (1 if direction == "LEFT" else -1) * clearance,
            "normal_acceleration_mps2": v * v / radius}


def rectilinear_moving_contact(facts: dict) -> dict:
    """First simultaneous envelope overlap; all inputs are model assumptions.

    Solve x(t)=dx+(u_x-v_x)t-a_x*t²/2 and y(t)=dy+u_y*t.
    Partition at boundary roots, so non-overlapping crossing times and tangency
    are distinguished without a time step or a sampling tolerance.
    """
    from hara_agent.contracts.physical_scenario_plan import MODEL_CATALOG
    fields = MODEL_CATALOG["RECTILINEAR_MOVING_TARGET_CONTACT"]["required"]
    if any(isinstance(facts.get(k), bool) or not isinstance(facts.get(k), (int, float))
           or not math.isfinite(facts[k]) for k in fields):
        return {"status": "MODEL_INPUTS_UNRESOLVED"}
    v, u = facts["ego_speed_kph"] / 3.6, facts["object_speed_kph"] / 3.6
    a, horizon = facts["longitudinal_acceleration_mps2"], facts["model_valid_until_s"]
    hx, hy = facts["contact_half_length_m"], facts["contact_half_width_m"]
    dx, dy = facts["object_offset_x_m"], facts["object_offset_y_m"]
    vectors = {"FORWARD": (1, 0), "REVERSE": (-1, 0), "LEFT": (0, 1), "RIGHT": (0, -1)}
    direction = facts.get("ego_longitudinal_direction")
    target_direction = facts.get("object_longitudinal_direction")
    if (v < 0 or u <= 0 or hx <= 0 or hy <= 0 or horizon <= 0
            or direction not in {"FORWARD", "REVERSE"} or target_direction not in vectors
            or facts.get("object_extent") != "FINITE_CONTACT_ENVELOPE"):
        return {"status": "MODEL_GEOMETRY_NOT_APPLICABLE"}
    # Target travel direction is not an impact-side classification. A crossing
    # pedestrian can meet the ego front; the finite envelopes decide contact.
    position = facts.get("object_position")
    if not {"FRONT": dx > 0, "REAR": dx < 0, "LEFT": dy > 0, "RIGHT": dy < 0}.get(position, False):
        return {"status": "MODEL_GEOMETRY_NOT_APPLICABLE", "reason": "TARGET_OFFSET_POSITION_CONFLICT"}
    if abs(dx) <= hx and abs(dy) <= hy:
        return {"status": "INITIAL_CONTACT_REQUIRES_SEPARATE_PHASE"}
    # Negative acceleration is deceleration in the travel coordinate, never
    # an implicit reverse launch. Continuing after stop requires a separate,
    # explicit holding assumption; old plans retain their pre-stop scope.
    stop = v / -a if a < 0 and v > 0 else horizon
    end = min(horizon, stop)
    if v == 0 and a < 0:
        a = 0.0
    sign = vectors[direction][0]
    ux, uy = (component * u for component in vectors[target_direction])
    q, b = -sign * a / 2, ux - sign * v
    phases = [(0.0, end, dx, b, q)]
    held = stop < horizon and facts.get("post_stop_motion") == "HOLD_POSITION"
    if held:
        stopping_position = sign * (v*stop + a*stop*stop/2)
        phases.append((stop, horizon, dx-stopping_position, ux, 0.0))
    for phase_index, (start, end, offset, b, q) in enumerate(phases):
        cuts = {start, end}
        for bound in (-hx, hx):
            c = offset - bound
            if q == 0:
                roots = [-c / b] if b else []
            else:
                disc = b*b - 4*q*c
                if disc < 0:
                    roots = []
                else:
                    # Stable quadratic roots, including a repeated root.
                    t = -.5 * (b + math.copysign(math.sqrt(disc), b))
                    roots = [t / q, c / t] if t else [-b / (2*q)]
            cuts.update(t for t in roots if start <= t <= end)
        if uy:
            cuts.update(t for bound in (-hy, hy) if start <= (t := (bound-dy)/uy) <= end)
        def inside(t):
            return abs(offset+b*t+q*t*t) <= hx and abs(dy+uy*t) <= hy
        ordered = sorted(cuts)
        for index, t in enumerate(ordered):
            # The right-hand interval is inside even if roundoff at its root is
            # a few ulps outside; do not widen the geometric contact envelope.
            if inside(t) or index+1 < len(ordered) and inside((t+ordered[index+1])/2):
                impact = 0.0 if phase_index else max(0.0, v+a*t)
                return {"status": "CONTACT", "time_s": t, "ego_impact_speed_kph": impact*3.6,
                    "object_impact_speed_kph": u*3.6,
                    "relative_speed_kph": math.hypot(sign*impact-ux, uy)*3.6,
                    "trajectory_max_speed_kph": max(v, impact)*3.6,
                    "contact_offset_x_m": offset+b*t+q*t*t, "contact_offset_y_m": dy+uy*t,
                    **({"contact_phase": "HELD_AFTER_STOP", "ego_stop_time_s": stop} if phase_index else {}),
                    "longitudinal_contact": uy == 0 and not is_lateral_collision(facts.get("collision_type"))}
    return {"status": "NO_CONTACT_WITHIN_MODEL", "evaluated_until_s": end,
            "limiting_condition": "EGO_STOP" if stop < horizon and not held else "MODEL_HORIZON"}


def moving_target_avoidance(facts: dict, proposal: dict) -> dict:
    """Check a proposed two-phase escape against the same contact envelopes.

    This establishes only the stated conditional trajectory, never human reaction
    capability. No C threshold, default reaction time or speed is introduced.
    """
    if facts.get("fault_motion_model") != "RECTILINEAR_MOVING_TARGET_CONTACT":
        return {"status": "UNSUPPORTED_AVOIDANCE_TRAJECTORY"}
    escape = proposal.get("avoidance", {})
    required = ("recognition_delay_s", "escape_speed_mps", "available_clearance_m")
    if any(not isinstance(escape.get(k), (int, float)) or isinstance(escape[k], bool)
           or not math.isfinite(escape[k]) for k in required):
        return {"status": "AVOIDANCE_GEOMETRY_UNRESOLVED"}
    original = rectilinear_moving_contact(facts)
    if original.get("status") != "CONTACT":
        return {"status": "NO_CONTACT_TO_AVOID", "original": original}
    delay = escape["recognition_delay_s"]
    if delay >= original["time_s"]:
        return {"status": "CONTACT_BEFORE_ESCAPE", "contact_time_s": original["time_s"]}
    remaining = facts["model_valid_until_s"] - delay
    speed = escape["escape_speed_mps"]
    if speed <= 0 or remaining * speed > escape["available_clearance_m"]:
        return {"status": "ESCAPE_CORRIDOR_TOO_SHORT"}
    vectors = {"FORWARD": (1, 0), "REVERSE": (-1, 0), "LEFT": (0, 1), "RIGHT": (0, -1)}
    if escape.get("escape_direction") not in vectors:
        return {"status": "AVOIDANCE_GEOMETRY_UNRESOLVED"}
    v, a = facts["ego_speed_kph"] / 3.6, facts["longitudinal_acceleration_mps2"]
    if v + a * delay < 0:
        return {"status": "AVOIDANCE_STOP_PHASE_UNRESOLVED"}
    sign = vectors[facts["ego_longitudinal_direction"]][0]
    ux, uy = [x * facts["object_speed_kph"] / 3.6 for x in vectors[facts["object_longitudinal_direction"]]]
    dx = facts["object_offset_x_m"] + ux*delay - sign*(v*delay + a*delay*delay/2)
    dy = facts["object_offset_y_m"] + uy*delay
    shifted = {**facts, "ego_speed_kph": (v+a*delay)*3.6,
        "object_offset_x_m": dx, "object_offset_y_m": dy,
        "object_speed_kph": speed*3.6, "object_longitudinal_direction": escape["escape_direction"],
        "model_valid_until_s": remaining,
        "object_position": ("FRONT" if dx > 0 else "REAR") if abs(dx) > facts["contact_half_length_m"]
            else ("LEFT" if dy > 0 else "RIGHT")}
    outcome = rectilinear_moving_contact(shifted)
    valid = outcome.get("status") == "NO_CONTACT_WITHIN_MODEL" and outcome.get("evaluated_until_s") == remaining
    return {"status": "CONDITIONAL_ESCAPE_VALIDATED" if valid else "ESCAPE_DOES_NOT_ESTABLISH_AVOIDANCE",
            "original": original, "escape": outcome, "recognition_delay_s": delay,
            "evaluated_horizon_s": facts["model_valid_until_s"]}


def braking_avoidability_boundary(speed_mps: float, acceleration_mps2: float,
                                  distance_m: float, net_deceleration_mps2: float) -> dict:
    """Latest brake onset before a stationary obstruction, including stopping travel.

    Net deceleration already includes the fault/grade forces. This is a physical
    boundary; its use as a conditional hazard criterion belongs to MethodContract.
    """
    v, a, d, b = speed_mps, acceleration_mps2, distance_m, net_deceleration_mps2
    if not all(math.isfinite(x) for x in (v, a, d, b)) or v < 0 or d <= 0 or b <= 0:
        return {"status": "RESPONSE_INPUTS_UNRESOLVED"}
    if v*v/(2*b) >= d:
        return {"status": "AVOIDABILITY_ALREADY_LOST", "time_s": 0.0,
                "stopping_time_s": v/b, "stopping_distance_m": v*v/(2*b)}
    if a < 0 and v*v/(-2*a) < d or v == 0 and a <= 0:
        return {"status": "NO_CONTACT_UNDER_THIS_MOTION"}
    q = a/2 + a*a/(2*b)
    linear = v*(1+a/b)
    constant = v*v/(2*b)-d
    if q == 0:
        delay = -constant/linear if linear > 0 else None
    else:
        discriminant = linear*linear-4*q*constant
        stable = -.5 * (linear + math.copysign(math.sqrt(max(0, discriminant)), linear))
        roots = [] if discriminant < 0 else ([stable/q, constant/stable] if stable else [0.0])
        roots = [t for t in roots if t >= 0 and v+a*t >= 0]
        delay = min(roots) if roots else None
    if delay is None:
        return {"status": "RESPONSE_BOUNDARY_UNRESOLVED"}
    onset_speed = v+a*delay
    return {"status": "CONDITIONAL_BOUNDARY", "time_s": delay,
            "stopping_time_s": onset_speed/b, "stopping_distance_m": onset_speed*onset_speed/(2*b),
            "brake_onset_speed_mps": onset_speed,
            "distance_residual_m": v*delay+a*delay*delay/2+onset_speed*onset_speed/(2*b)-d}


def moving_braking_avoidability_boundary(facts: dict, net_deceleration_mps2: float,
                                        *, release_deceleration: bool = False) -> dict:
    """First loss of the immediate-braking-connected safe window.

    Target motion and the contact envelope are unchanged. For each response
    delay, minimize longitudinal separation analytically over the lateral
    overlap interval, through fault motion, braking and a held stop. Ego
    progress is monotone in response delay with the sign of a+b, including a
    weaker braking response that releases excessive fault deceleration. Follow
    the initial safe side of the envelope, then bracket its FIRST loss;
    it does not assume arbitrary crossing-contact outcomes are monotone.
    """
    original = rectilinear_moving_contact(facts)
    if original.get("status") != "CONTACT":
        return {"status": "NO_UNMITIGATED_CONTACT_BOUNDARY", "original": original}
    b = _nonnegative_number(net_deceleration_mps2)
    v, a = facts["ego_speed_kph"]/3.6, facts["longitudinal_acceleration_mps2"]
    if release_deceleration and (a >= 0 or v <= 0 or b != 0):
        return {"status": "FAULT_RELEASE_NOT_APPLICABLE"}
    if b is None or not b and not release_deceleration:
        return {"status": "RESPONSE_INPUTS_UNRESOLVED"}
    horizon = facts["model_valid_until_s"]
    sign = 1 if facts["ego_longitudinal_direction"] == "FORWARD" else -1
    vectors = {"FORWARD": (1, 0), "REVERSE": (-1, 0), "LEFT": (0, 1), "RIGHT": (0, -1)}
    ux, uy = [k*facts["object_speed_kph"]/3.6 for k in vectors[facts["object_longitudinal_direction"]]]
    dx, dy = sign*facts["object_offset_x_m"], facts["object_offset_y_m"]
    hx, hy = facts["contact_half_length_m"], facts["contact_half_width_m"]
    start_y, end_y = 0.0, horizon
    if uy:
        enter, leave = sorted(((-hy-dy)/uy, (hy-dy)/uy))
        start_y, end_y = max(0.0, enter), min(horizon, leave)

    def separation(delay):
        onset_speed = max(0.0, v+a*delay)
        stop = delay+onset_speed/b if b else horizon
        onset_position = v*delay+a*delay*delay/2
        def position(t):
            if t <= delay:
                return v*t+a*t*t/2
            elapsed = min(t-delay, onset_speed/b) if b else t-delay
            return onset_position+onset_speed*elapsed-b*elapsed*elapsed/2
        # Global linear/quadratic coefficients locate extrema; positions below
        # are evaluated in local phase time for numerical stability.
        phases = [(0.0, delay, v, a/2),
                  (delay, stop, v+(a+b)*delay, -b/2),
                  (stop, horizon, 0.0, 0.0)]
        extrema = []
        for begin, end, c1, c2 in phases:
            lo, hi = max(begin, start_y), min(end, end_y)
            if lo > hi:
                continue
            points = [lo, hi]
            linear, quadratic = sign*ux-c1, -c2
            if quadratic and lo < (turn := -linear/(2*quadratic)) < hi:
                points.append(turn)
            # Evaluate in each phase's local time to avoid cancelling large
            # global polynomial coefficients near a small clearance boundary.
            extrema.extend((math.fsum((dx, sign*ux*t, -position(t))), t) for t in points)
        minimum, at = min(extrema)
        return {"clearance_m": minimum-hx, "minimum_at_s": at,
                "minimum_separation_m": minimum,
                "maximum_separation_m": max(x for x, _ in extrema),
                "brake_onset_speed_mps": onset_speed, "stopping_time_s": onset_speed/b if b else None,
                "stopping_distance_m": onset_speed*onset_speed/(2*b) if b else None,
                "stop_time_s": stop if b else None}

    immediate = separation(0.0)
    common = {"response_model_family": "MOVING_TARGET_FAULT_RELEASE" if release_deceleration else "MOVING_TARGET_BRAKING",
              "original_contact_time_s": original["time_s"], "evaluated_until_s": horizon,
              "lateral_overlap_interval_s": [start_y, end_y],
              "response_motion": "CONTINUE_AT_ONSET_SPEED" if release_deceleration else "BRAKE_AND_HOLD",
              **({} if release_deceleration else {"post_stop_motion": "HOLD_POSITION"}),
              "immediate_response": immediate}
    # Longitudinal separation is continuous over one lateral-overlap interval.
    # No contact means the target stays wholly ahead OR wholly behind. Using
    # only the minimum would falsely reject every initially safe rear relation.
    safe_side = (1 if immediate["minimum_separation_m"] > hx else
                 -1 if immediate["maximum_separation_m"] < -hx else 0)
    if not safe_side:
        return {**common, "status": "NO_IMMEDIATE_SAFE_BRAKING_WINDOW"}
    def margin(result):
        return (result["minimum_separation_m"]-hx if safe_side == 1 else
                -result["maximum_separation_m"]-hx)
    immediate["clearance_m"] = margin(immediate)
    common["response_clearance_side"] = "TARGET_AHEAD" if safe_side == 1 else "TARGET_BEHIND"
    # d(progress)/d(delay) has the sign of a+b before and after the stop.
    # The selected side must lose margin as delay grows towards original contact.
    if safe_side*(a+b) <= 0:
        return {**common, "status": "RESPONSE_BOUNDARY_UNRESOLVED"}
    # If the fault trajectory already decelerates to a held stop, delaying
    # braking beyond that stop cannot restart it. The original later contact
    # may be the target arriving at the held vehicle; use the natural stop as
    # the latest distinct response phase, never extend v*t+a*t²/2 backwards.
    latest = min(original["time_s"], v/-a) if a < 0 and v > 0 else original["time_s"]
    lo, hi = 0.0, latest
    for _ in range(64):
        middle = (lo+hi)/2
        if middle in (lo, hi):
            break
        if margin(separation(middle)) > 0:
            lo = middle
        else:
            hi = middle
    boundary = separation(lo)
    boundary["clearance_m"] = margin(boundary)
    return {**common, **boundary, "status": "CONDITIONAL_BOUNDARY", "time_s": lo,
            "boundary_bracket_s": [lo, hi], "response_delay_interval_proven": True,
            "proof_scope": "Fixed hypothesis and finite interaction horizon; target stays on the initial safe side during lateral overlap for every earlier response onset."}


def closing_relative_speed_kph(
    ego_speed_kph: Any, object_speed_kph: Any, *,
    ego_direction: Any, object_direction: Any, collision_type: Any,
) -> float | None:
    """Return longitudinal relative-speed magnitude, independently of TTC approach."""
    ego = _nonnegative_number(ego_speed_kph)
    obj = _nonnegative_number(object_speed_kph)
    ego_dir = str(ego_direction or "").strip().upper()
    obj_dir = str(object_direction or "").strip().upper()
    if ego is None or obj is None:
        return None
    # A zero-speed target has zero velocity in either longitudinal direction.
    # Its relative-speed magnitude is known even when travel direction is not;
    # approach and TTC still require explicit directional geometry.
    if obj == 0:
        return round(ego, 6)
    if is_lateral_collision(collision_type):
        return None
    if (
        ego_dir not in {"FORWARD", "REVERSE"}
        or obj_dir not in {"FORWARD", "REVERSE", "STATIONARY"}
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
    bindings = scenario.facts.get("method_scenario_dimensions", {})
    stationary_bindings = [binding for binding in bindings.values()
                          if isinstance(binding, dict) and binding.get("resolution_status") == "RESOLVED"
                          and binding.get("method_semantics", {}).get("ego_dynamics", {}).get("motion_state") == "STATIONARY"]
    stationary = bool(stationary_bindings) and rule.get("stationary_initial_speed_kph") == 0
    if stationary:
        refs.extend({"source_type": "method_contract", "source_id": str(binding.get("atom_id", "")),
                     "location": str(binding.get("atom_provenance", {}).get("source_rule", "")),
                     "excerpt": str(binding.get("atom_label", ""))} for binding in stationary_bindings)
    return (float(rule["stationary_initial_speed_kph"]) if stationary else float(upper)), {
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
        "selection_basis": "EXPLICIT_STATIONARY_INITIAL_STATE" if stationary else "GOVERNED_CLOSED_UPPER_BOUND",
        "analysis_value_semantic": rule.get("value_semantic", "CURRENT_OPERATION_EGO_SPEED"),
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


def derive_stationary_object_direction(
    scenario: ScenarioCandidate, *, malfunction_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Derive stationary velocity from an accepted exact zero-speed fact."""
    if "object_longitudinal_direction" in scenario.facts:
        return None
    speed = scenario.facts.get("object_speed_kph")
    source = scenario.fact_provenance.get("object_speed_kph", {})
    if (
        isinstance(speed, bool) or not isinstance(speed, (int, float))
        or float(speed) != 0.0 or not isinstance(source, dict)
        or not source_is_accepted_for(
            source, malfunction_id=malfunction_id,
            scenario_id=scenario.scenario_id,
        )
    ):
        return None
    scope = {
        "malfunction_id": malfunction_id,
        "scenario_id": scenario.scenario_id,
        "parent_scenario_id": scenario.source_scenario_id or scenario.scenario_id,
    }
    input_status, input_sources, _ = _input_metadata(
        scenario, "object_speed_kph",
    )
    return "STATIONARY", {
        "provenance": FactProvenance.DERIVED.value,
        "origin": FactProvenance.DERIVED.value,
        "approval": input_status.value,
        "validation_status": "VALIDATED",
        "applicable_scope": scope,
        "source_refs": list(source.get("source_refs", source.get("sources", []))),
        "input_fact_metadata": [_input_snapshot(
            "object_speed_kph", input_status, input_sources, source,
        )],
        "inputs": ["SCN.object_speed_kph"],
        "derivation_rule_id": "ZERO_SPEED_IMPLIES_STATIONARY_VELOCITY",
    }


def time_to_collision_s(distance_m: Any, relative_speed_kph: Any) -> float | None:
    """Return TTC for a positive closing speed, otherwise fail closed."""
    distance = _nonnegative_number(distance_m)
    speed = _nonnegative_number(relative_speed_kph)
    if distance is None or speed is None or speed <= 0:
        return None
    return round(distance / (speed / 3.6), 6)


def causal_physics_conflicts(scenario: ScenarioCandidate, payload: dict[str, Any]) -> list[str]:
    """Find explicit coordinate contradictions, without inferring a new hazard."""
    facts = scenario.facts
    claims = [str(payload.get("hazardous_event", ""))]
    claims.extend(str(hop.get("claim", "")) for hop in
                  payload.get("causal_chain", {}).values() if isinstance(hop, dict))
    text = " ".join(claims).casefold()
    terms = {"FRONT": ("前方", "前车", "in front"),
             "REAR": ("后方", "后车", "behind"),
             "LEFT": ("左侧", "left side"), "RIGHT": ("右侧", "right side")}
    opposite = {"FRONT": "REAR", "REAR": "FRONT", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    position = str(facts.get("object_position", "")).upper()
    conflicts = []
    if position in terms and not any(word in text for word in terms[position]) and any(
        word in text for word in terms[opposite[position]]
    ):
        conflicts.append("OBJECT_POSITION_CONTRADICTS_CAUSAL_CLAIM")
    closing = longitudinal_closing_speed_kph(
        facts.get("ego_speed_kph"), facts.get("object_speed_kph"),
        ego_direction=facts.get("ego_longitudinal_direction"),
        object_direction=facts.get("object_longitudinal_direction"),
        object_position=position, collision_type=facts.get("collision_type"),
    )
    interaction = payload.get("physical_interaction", {})
    interaction = interaction if isinstance(interaction, dict) else {}
    same_target = (interaction.get("target") == "SCENARIO_OBJECT"
                   and str(interaction.get("object_position", "")).upper() == position
                   and (not interaction.get("object_id") or interaction["object_id"] == facts.get(
                       "object_id", scenario.scenario_id + ":target")))
    same_phase = interaction.get("assessment_reference") in {"CURRENT_STATE", "MODEL_TRAJECTORY"}
    outcome = facts.get("motion_model_outcome", {})
    if (same_target and same_phase and interaction.get("relation") == "CONTACT"
            and isinstance(outcome, dict) and outcome.get("status") in {
                "STOPPED_BEFORE_CONTACT", "NO_POSITIVE_CONTACT_ROOT", "CONTACT_OUTSIDE_MODEL_INTERVAL", "NO_CONTACT_WITHIN_MODEL"}):
        conflicts.append("NO_CONTACT_TRAJECTORY_CONTRADICTS_CONTACT_CLAIM")
    if closing == 0:
        modeled = False
        if facts.get("fault_motion_model") == "RECTILINEAR_MOVING_TARGET_CONTACT":
            modeled = rectilinear_moving_contact(facts)["status"] == "CONTACT"
        if facts.get("fault_motion_model") in {"CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE"}:
            outcome = constant_acceleration_contact(facts.get("ego_speed_kph"),
                facts.get("longitudinal_acceleration_mps2"), facts.get("travel_to_impact_m"),
                facts.get("model_valid_until_s"))
            modeled = (outcome["status"] == "CONTACT" and facts.get("object_speed_kph") == 0
                and (str(facts.get("ego_longitudinal_direction", "")).upper(), position) in {("FORWARD", "FRONT"), ("REVERSE", "REAR")}
                and not is_lateral_collision(facts.get("collision_type")))
        asserts_contact = (same_target and same_phase
                           and interaction.get("relation") in {"APPROACH", "CONTACT"})
        if not interaction and position in terms:
            # Narrow legacy check for explicit statements about this target in
            # one clause. A generic hazard or an occupant harm is not contact.
            # New assessments provide the structured relation instead.
            target = "(?:" + "|".join(re.escape(word) for word in terms[position]) + ")"
            predicate = r"(?:距离[^，。；;]*缩小|越来越接近|碰撞|相撞)"
            asserts_contact = any(re.search(target + r"[^，。；;!?]*" + predicate, claim)
                                  and not re.search(r"(?:不|未|无)[^，。；;!?]*" + predicate, claim)
                                  for claim in claims)
        if not modeled and asserts_contact:
            conflicts.append("NOT_CLOSING_CONTRADICTS_CAUSAL_CLAIM")
    return conflicts


def _distance_m(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _nonnegative_number(value)
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
    if origin not in {_ANALYSIS_ORIGIN, CONDITIONAL_ORIGIN}:
        return {}
    scope = metadata.get(
        "analysis_assumption_scope", metadata.get("applicable_scope", {}),
    )
    return {
        "analysis_assumption_origin": origin,
        "analysis_assumption_scope": dict(scope) if isinstance(scope, dict) else scope,
        "validation_status": metadata.get("validation_status", ""),
        "source_template_id": metadata.get("source_template_id", ""),
        "source_option_id": metadata.get("source_option_id", ""),
        "method_contract_hash": metadata.get("method_contract_hash", ""),
        "calculation_scope": metadata.get("calculation_scope", ""),
        "hypothesis_id": metadata.get("hypothesis_id", ""),
        "validated_by": metadata.get("validated_by", ""),
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


def has_conditional_lineage(metadata: dict[str, Any]) -> bool:
    return any(item.get("analysis_assumption_origin") == CONDITIONAL_ORIGIN
               for item in analysis_assumption_lineages(metadata)) or any(
        has_conditional_lineage(item) for item in metadata.get("input_fact_metadata", [])
        if isinstance(item, dict))


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
        item["analysis_assumption_origin"] in {_ANALYSIS_ORIGIN, CONDITIONAL_ORIGIN}
        and (item["analysis_assumption_origin"] != CONDITIONAL_ORIGIN or (
            item.get("calculation_scope") == "CONDITIONAL"
            and item.get("validated_by") == "PhysicalHypothesisInstantiationService"
            and bool(item.get("hypothesis_id"))))
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
    scope = metadata.get("applicable_scope", {})
    if isinstance(scope, dict) and any(
        scope.get(key) and str(scope[key]) != expected
        for key, expected in (("malfunction_id", malfunction_id), ("scenario_id", scenario_id))
    ):
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
    if has_conditional_lineage(metadata):
        return CONDITIONAL_ORIGIN
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
        "calculation_scope", "hypothesis_id", "validated_by",
        *PHYSICAL_SEMANTIC_FIELDS,
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
        "ego_speed_kph", "relative_speed_kph", "closing_speed_kph",
        "impact_speed_kph", "delta_v_kph",
    ):
        value = scenario.facts.get(key)
        if _nonnegative_number(value) is None:
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
            **physical_semantics(input_metadata),
        }
        if input_metadata:
            metadata["input_fact_metadata"] = (
                input_metadata.get("input_fact_metadata", [])
                if input_metadata.get("derivation_type") == DerivedPhysicsType.CANONICAL_INPUT_NORMALIZATION.value
                else [_input_snapshot(input_key, status, sources, input_metadata)]
            )
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

    def with_derived(record):
        return replace(scenario, facts={**scenario.facts, record.evidence_ref.removeprefix("DERIVED."): record.value},
            fact_provenance={**scenario.fact_provenance, record.evidence_ref.removeprefix("DERIVED."): {
                **record.metadata, "provenance": "DERIVED", "approval": record.approval_status.value,
                "source_refs": [_source_dict(ref) for ref in record.source_refs]}})

    model = scenario.facts.get("fault_motion_model")
    if model == "RECTILINEAR_MOVING_TARGET_CONTACT":
        from hara_agent.contracts.physical_scenario_plan import MODEL_CATALOG
        keys = tuple(MODEL_CATALOG[model]["required"]) + (
            "fault_motion_model", "object_position", "object_extent", "ego_longitudinal_direction",
            "object_longitudinal_direction", "impact_time_id")
        if "post_stop_motion" in scenario.facts:
            keys += ("post_stop_motion",)
        outcome = rectilinear_moving_contact(scenario.facts)
        formula = "first simultaneous |dx+(ux-vx)t-ax*t^2/2|<=hx and |dy+uy*t|<=hy before ego stop/horizon"
        if scenario.facts.get("post_stop_motion") == "HOLD_POSITION":
            formula += "; after stop ts, xe=sign*(v0*ts+a*ts^2/2), ve=0; solve |dx+ux*t-xe|<=hx and |dy+uy*t|<=hy until horizon"
        records.append(derived_record("motion_model_outcome", outcome, keys,
            DerivedPhysicsType.KINEMATIC_IMPACT_SPEED, formula=formula))
        if outcome["status"] == "CONTACT":
            results = {k: outcome[k] for k in ("ego_impact_speed_kph", "object_impact_speed_kph",
                       "relative_speed_kph", "trajectory_max_speed_kph")}
            results["first_contact_time_s"] = outcome["time_s"]
            if outcome["longitudinal_contact"]:
                results["ttc_s"] = outcome["time_s"]
            for key, value in results.items():
                record = derived_record(key, value, keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED, formula=formula)
                record.metadata.update(impact_time_id=scenario.facts.get("impact_time_id"),
                    analysis_value_semantic="FIRST_CONTACT_TIME" if key in {"ttc_s", "first_contact_time_s"} else "IMPACT_SPEED",
                    assessment_reference="MODEL_INITIAL_STATE")
                records.append(record)
        return tuple({record.evidence_ref: record for record in records}.values())
    if model == "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE":
        grade = _nonnegative_number(scenario.facts.get("slope_percent"))
        resistance = _nonnegative_number(scenario.facts.get("residual_resistance_mps2"))
        downhill = scenario.facts.get("downhill_body_direction")
        heading = scenario.facts.get("ego_heading_relative_to_downhill")
        direction = "REVERSE" if downhill == "REAR" else "FORWARD"
        grade_inputs = ("grade_coordinate_frame", "slope_reference", "ego_heading_relative_to_downhill",
                        "downhill_body_direction", "ego_longitudinal_direction")
        if (scenario.facts.get("grade_coordinate_frame") != "BODY_FORWARD_POSITIVE"
                or (heading, downhill) not in {("UPHILL", "REAR"), ("DOWNHILL", "FRONT")}
                or scenario.facts.get("ego_longitudinal_direction") != direction):
            records.append(derived_record("motion_model_outcome", {"status": "GRADE_DIRECTION_UNRESOLVED_OR_CONFLICTING"},
                grade_inputs, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED))
            return tuple(records)
        if grade is not None:
            records.append(derived_record("gravity_body_acceleration_mps2",
                (-1 if downhill == "REAR" else 1) * 9.80665 * math.sin(math.atan(grade / 100)),
                ("slope_percent", *grade_inputs), DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                formula="body forward positive; gravity sign from downhill; resistance opposes motion"))
        position = str(scenario.facts.get("object_position", "")).upper()
        # From rest, gravity and nonnegative resistance cannot launch toward
        # an opposite-side stationary target. This is independent of the
        # unknown resistance magnitude, so check before numerical completeness.
        if (scenario.facts.get("ego_speed_kph") == 0 and scenario.facts.get("object_speed_kph") == 0
                and position in {"FRONT", "REAR", "LEFT", "RIGHT"} and position != downhill):
            records.append(derived_record("motion_model_outcome", {
                "status": "MODEL_GEOMETRY_NOT_APPLICABLE", "reason": "TARGET_OUTSIDE_GRAVITY_PATH",
                "contact_possible_in_model": False, "target_position": position, "downhill_body_direction": downhill},
                ("fault_motion_model", "ego_speed_kph", "object_speed_kph", "object_position", *grade_inputs),
                DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                formula="from rest; pure gravity with opposing resistance; stationary target outside downhill path"))
            return tuple(records)
        if grade is not None and resistance is not None and scenario.facts.get("ego_speed_kph") == 0:
            # Grade percent is rise/run, not degrees. Static residual holding
            # can prevent motion, but cannot accelerate the car uphill here.
            acceleration = max(0.0, 9.80665 * math.sin(math.atan(grade / 100)) - resistance)
            record = derived_record("longitudinal_acceleration_mps2", acceleration,
                ("slope_percent", "residual_resistance_mps2", "fault_motion_model", "ego_speed_kph", *grade_inputs),
                DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                formula="max(0, g*sin(atan(grade_percent/100))-residual_resistance); g=9.80665 m/s2")
            records.append(record)
            scenario = with_derived(record)
        else:
            records.append(derived_record("motion_model_outcome", {"status": "ROLLAWAY_INITIAL_STATE_OR_GRADE_UNRESOLVED"},
                ("fault_motion_model", "ego_speed_kph", "slope_percent", "residual_resistance_mps2"),
                DerivedPhysicsType.KINEMATIC_IMPACT_SPEED))
            return tuple(records)
    if model == "CONSTANT_SPEED_CURVATURE_TO_CONTACT":
        outcome = constant_speed_curved_contact(scenario.facts)
        keys = ("ego_speed_kph", "turn_radius_m", "lateral_clearance_m", "model_valid_until_s",
                "ego_lateral_direction", "ego_longitudinal_direction", "object_position", "object_extent",
                "object_speed_kph", "collision_type", "fault_motion_model", "impact_time_id")
        formula = "x=sign(v)*R*sin(v*t/R); y=side*R*(1-cos(v*t/R)); |v| constant; first quarter-turn only"
        records.append(derived_record("motion_model_outcome", outcome, keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
            formula=formula))
        if outcome["status"] == "CONTACT":
            for key, value in (("first_contact_time_s", outcome["time_s"]),
                               ("ego_impact_speed_kph", outcome["impact_speed_kph"]),
                               ("relative_speed_kph", outcome["impact_speed_kph"]),
                               ("trajectory_max_speed_kph", outcome["trajectory_max_speed_kph"])):
                record = derived_record(key, value, keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED, formula=formula)
                record.metadata.update(impact_time_id=scenario.facts.get("impact_time_id"),
                    analysis_value_semantic="FIRST_CONTACT_TIME" if key == "first_contact_time_s" else "IMPACT_SPEED")
                records.append(record)
        return tuple({record.evidence_ref: record for record in records}.values())
    if model == "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT":
        lateral = _nonnegative_number(scenario.facts.get("lateral_speed_mps"))
        initial = _nonnegative_number(scenario.facts.get("ego_speed_kph"))
        outcome = constant_acceleration_contact(None if lateral is None else lateral * 3.6,
            scenario.facts.get("lateral_acceleration_mps2"), scenario.facts.get("lateral_clearance_m"),
            scenario.facts.get("model_valid_until_s"))
        if (scenario.facts.get("object_position") not in {"LEFT", "RIGHT"}
                or scenario.facts.get("object_extent") != "EXTENDED_ALONG_TRAVEL_PATH"
                or scenario.facts.get("object_speed_kph") != 0 or initial is None or initial <= 0):
            outcome = {"status": "MODEL_GEOMETRY_NOT_APPLICABLE"}
        keys = ("lateral_speed_mps", "lateral_acceleration_mps2", "lateral_clearance_m", "ego_speed_kph",
                "object_position", "object_extent", "object_speed_kph", "fault_motion_model", "impact_time_id")
        if "model_valid_until_s" in scenario.facts:
            keys += ("model_valid_until_s",)
        records.append(derived_record("motion_model_outcome", outcome, keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED))
        if outcome["status"] == "CONTACT":
            # In this model v_x is constant and v_y is affine in time, so the
            # speed norm reaches its maximum at an interval endpoint. Keep
            # this bound unrounded for method applicability, not display.
            records.append(derived_record("trajectory_max_speed_kph",
                max(math.hypot(initial, lateral * 3.6),
                    math.hypot(initial, outcome["impact_speed_kph"])),
                keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                formula="max(hypot(v_x,v_y(0)),hypot(v_x,v_y(first_contact))); constant lateral acceleration"))
            for key, value in (("first_contact_time_s", outcome["time_s"]),
                               ("ego_impact_speed_kph", math.hypot(initial, outcome["impact_speed_kph"])),
                               ("relative_speed_kph", math.hypot(initial, outcome["impact_speed_kph"]))):
                record = derived_record(key, round(value, 6), keys, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                    formula="y=v_y0*t+0.5*a_y*t^2; |v_rel|=hypot(v_x,v_y(t)); stationary extended side target")
                record.metadata.update(impact_time_id=scenario.facts.get("impact_time_id"),
                    analysis_value_semantic="FIRST_CONTACT_TIME" if key == "first_contact_time_s" else "IMPACT_SPEED")
                records.append(record)
        # First lateral contact is not an input to the longitudinal C profile.
        return tuple({record.evidence_ref: record for record in records}.values())
    model_selected = model in {"CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE"}
    model_ttc = None
    model_contact_valid = not model_selected
    if model_selected and scenario.facts.get("impact_time_id"):
        outcome = constant_acceleration_contact(
            scenario.facts.get("ego_speed_kph"),
            scenario.facts.get("longitudinal_acceleration_mps2"),
            scenario.facts.get("travel_to_impact_m"),
            scenario.facts.get("model_valid_until_s"),
            scenario.facts.get("post_fault_speed_cap_kph"),
        )
        direction, position = (str(scenario.facts[key]).upper() if key in scenario.facts else None
                               for key in ("ego_longitudinal_direction", "object_position"))
        if (direction is not None and position is not None
                and (direction, position) not in {("FORWARD", "FRONT"), ("REVERSE", "REAR")}):
            outcome = {"status": "MODEL_GEOMETRY_NOT_APPLICABLE"}
        if is_lateral_collision(scenario.facts.get("collision_type")):
            outcome = {"status": "MODEL_GEOMETRY_NOT_APPLICABLE"}
        model_inputs = ("ego_speed_kph", "longitudinal_acceleration_mps2", "travel_to_impact_m",
                        "fault_motion_model", "impact_time_id")
        if "model_valid_until_s" in scenario.facts:
            model_inputs += ("model_valid_until_s",)
        if "post_fault_speed_cap_kph" in scenario.facts:
            model_inputs += ("post_fault_speed_cap_kph",)
        records.append(derived_record("motion_model_outcome", outcome, model_inputs,
                                      DerivedPhysicsType.KINEMATIC_IMPACT_SPEED))
        if outcome["status"] == "CONTACT":
            model_contact_valid = True
            impact = derived_record("ego_impact_speed_kph", round(outcome["impact_speed_kph"], 6),
                model_inputs, DerivedPhysicsType.KINEMATIC_IMPACT_SPEED,
                formula="v(t)=v0+a*t; d=v0*t+0.5*a*t^2; optional remaining cap: accelerate to cap then cruise; first contact before stop")
            impact.metadata.update(analysis_value_semantic="IMPACT_SPEED",
                                   impact_time_id=scenario.facts["impact_time_id"])
            records.append(impact)
            scenario = replace(scenario, facts={**scenario.facts, "ego_impact_speed_kph": impact.value},
                fact_provenance={**scenario.fact_provenance, "ego_impact_speed_kph": {
                    **impact.metadata, "provenance": "DERIVED", "approval": impact.approval_status.value,
                    "source_refs": [_source_dict(ref) for ref in impact.source_refs]}})
            direction = str(scenario.facts.get("ego_longitudinal_direction", "")).upper()
            position = str(scenario.facts.get("object_position", "")).upper()
            stationary_target = scenario.facts.get("object_speed_kph") == 0
            along_path = (direction, position) in {("FORWARD", "FRONT"), ("REVERSE", "REAR")}
            same_distance = scenario.facts.get("travel_to_impact_m") == distance_m
            if stationary_target and along_path and same_distance and not is_lateral_collision(scenario.facts.get("collision_type")):
                model_ttc = derived_record("ttc_s", round(outcome["time_s"], 6),
                    (*model_inputs, "object_speed_kph", "ego_longitudinal_direction", "object_position", distance_input_key),
                    DerivedPhysicsType.TTC, formula="first nonnegative root of d=v0*t+0.5*a*t^2 before stop; optional cap uses acceleration then cruise phases")
                model_ttc.metadata.update(impact_time_id=scenario.facts["impact_time_id"],
                                          assessment_reference="MODEL_INITIAL_STATE")


    ego = scenario.facts.get("ego_speed_kph")
    obj = scenario.facts.get("object_speed_kph")
    ego_direction = scenario.facts.get("ego_longitudinal_direction")
    object_direction = scenario.facts.get("object_longitudinal_direction")
    collision = scenario.facts.get("collision_type")
    position = scenario.facts.get("object_position")
    relative_speed = scenario.facts.get("relative_speed_kph")
    speed_semantic = physical_semantics(scenario.fact_provenance.get("ego_speed_kph", {})).get("analysis_value_semantic", "")
    impact_ego = scenario.facts.get("ego_impact_speed_kph")
    impact_object = scenario.facts.get("object_impact_speed_kph")
    impact_time = scenario.facts.get("impact_time_id")
    impact_phases = [physical_semantics(scenario.fact_provenance.get(key, {})).get("impact_time_id")
                     for key in ("ego_impact_speed_kph", "object_impact_speed_kph")]
    impact_phase_conflict = any(phase is not None and phase != impact_time for phase in impact_phases)
    same_impact_time = bool(impact_time) and impact_ego is not None and impact_object is not None and not impact_phase_conflict
    relative_ego = impact_ego if same_impact_time else ego if speed_semantic != "PRE_FAULT_OPERATING_SPEED" else None
    relative_object = impact_object if same_impact_time else obj
    calculated_relative = closing_relative_speed_kph(
        relative_ego, relative_object, ego_direction=ego_direction,
        object_direction=object_direction, collision_type=collision,
    )
    if impact_phase_conflict or not model_contact_valid:
        calculated_relative = None
    if relative_speed is None and calculated_relative is not None:
        relative_speed = calculated_relative
        relative_inputs = (
            ("ego_speed_kph", "object_speed_kph")
            if relative_object == 0 else
            ("ego_speed_kph", "object_speed_kph", "ego_longitudinal_direction",
             "object_longitudinal_direction", "collision_type")
        )
        if same_impact_time:
            relative_inputs = tuple({"ego_speed_kph": "ego_impact_speed_kph",
                                    "object_speed_kph": "object_impact_speed_kph"}.get(key, key)
                                    for key in relative_inputs) + ("impact_time_id",)
        records.append(derived_record(
            "relative_speed_kph", calculated_relative,
            relative_inputs,
            DerivedPhysicsType.RELATIVE_MOTION,
        ))

    directional_geometry_supplied = any(
        key in scenario.facts for key in (
            "ego_longitudinal_direction", "object_longitudinal_direction",
        )
    )
    direct_closing = _nonnegative_number(scenario.facts.get("closing_speed_kph"))
    calculated_closing = longitudinal_closing_speed_kph(
            ego, obj, ego_direction=ego_direction,
            object_direction=object_direction, object_position=position,
            collision_type=collision,
    )
    if directional_geometry_supplied and calculated_closing is not None:
        closing_speed = calculated_closing
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
        # Relative-speed magnitude alone does not establish approach geometry.
        # A separately supplied closing speed can be consumed with its own
        # source chain; downstream validation rejects unapproved inputs.
        closing_speed = direct_closing
        ttc_input_keys = (distance_input_key, "closing_speed_kph")
        formula = TTC_CLOSING_FORMULA_IDENTITY
    if model_selected:
        if model_ttc is not None:
            records.append(model_ttc)
        return tuple({record.evidence_ref: record for record in records}.values())
    ttc = time_to_collision_s(distance_m, closing_speed)
    if ttc is not None:
        records.append(derived_record(
            "ttc_s", ttc, ttc_input_keys, DerivedPhysicsType.TTC,
            formula=formula,
        ))
    return tuple({record.evidence_ref: record for record in records}.values())
