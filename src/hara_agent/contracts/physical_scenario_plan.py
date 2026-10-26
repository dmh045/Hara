"""Bounded analytical hypotheses, deliberately separate from extracted facts.

The model may propose conditions, never ratings, engineering approval or
measured vehicle limits. Units and supported models are a physics API, not
HARA business rules.
"""
from dataclasses import dataclass
import math
from typing import Any


PHYSICAL_PLAN_VERSION = "physical-plan-v3-relative-contact"
PARAMETER_UNITS = {
    "ego_speed_kph": "km/h", "relative_distance_m": "m",
    "longitudinal_acceleration_mps2": "m/s2", "slope_percent": "%",
    "residual_resistance_mps2": "m/s2", "model_valid_until_s": "s",
    "lateral_speed_mps": "m/s", "lateral_acceleration_mps2": "m/s2",
    "lateral_clearance_m": "m",
    "post_fault_speed_cap_kph": "km/h",
    "object_speed_kph": "km/h", "object_offset_x_m": "m", "object_offset_y_m": "m",
    "contact_half_length_m": "m", "contact_half_width_m": "m",
}
MODEL_CATALOG = {
    "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE": {
        "target_motion": "STATIONARY",
        "required": ["ego_speed_kph", "relative_distance_m", "longitudinal_acceleration_mps2"],
        "scope": "Straight travel toward a stationary target; first contact before stop; acceleration may be negative or zero. relative_distance_m is the remaining ego travel to first contact (surface clearance), NOT vehicle center-to-center distance. Optional post_fault_speed_cap_kph represents an explicitly remaining independent limiter, not the normal operating speed.",
    },
    "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE": {
        "target_motion": "STATIONARY",
        "required": ["ego_speed_kph", "relative_distance_m", "slope_percent", "residual_resistance_mps2"],
        "scope": "Initially stationary ego, gravity down a constant grade toward a stationary target. relative_distance_m is remaining travel to first contact (surface clearance), NOT vehicle center-to-center distance. Grade is rise/run percent; residual resistance includes remaining holding capability.",
    },
    "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT": {
        "target_motion": "STATIONARY",
        "required": ["ego_speed_kph", "lateral_clearance_m", "lateral_speed_mps", "lateral_acceleration_mps2"],
        "scope": "Steering-induced deviation while ego is moving longitudinally (speed > 0); stationary side object extended along the longitudinal swept path; lateral first contact, constant longitudinal speed. No longitudinal TTC or steering avoidance rule is inferred.",
    },
    "RECTILINEAR_MOVING_TARGET_CONTACT": {
        "target_motion": "CONSTANT_VELOCITY",
        "required": ["ego_speed_kph", "longitudinal_acceleration_mps2", "object_speed_kph",
                     "object_offset_x_m", "object_offset_y_m", "contact_half_length_m",
                     "contact_half_width_m", "model_valid_until_s"],
        "scope": "Body frame x forward, y left; initial target-center offsets signed. Ego straight constant acceleration along its travel direction, target constant velocity FORWARD/REVERSE/LEFT/RIGHT. Axis-aligned contact envelope half-length/width is the SUM of ego and target projected half-extents, justified as assumptions, not measured dimensions. First simultaneous x/y overlap before ego stops and within explicit horizon. Supports following, oncoming, rear approach and crossing; no turns, reversal or independent speed cap. Crossing contact time is not longitudinal TTC.",
    },
}
# Legacy "braking" means the braking capability itself. Command-path failure
# and an independently available driver path need separate identities.
CONTROL_IDENTITIES = {"drive", "steering", "braking", "automatic_braking_command", "driver_braking",
                      "parking_hold", "speed_limiter", "remote_control"}


def braking_capability(hypothesis: dict) -> bool | None:
    """Translate declared controls, without inferring fault independence."""
    failed = set(hypothesis.get("failed_controls", []))
    remaining = set(hypothesis.get("remaining_controls", []))
    if failed & {"braking", "driver_braking"}:
        return False
    if "automatic_braking_command" in failed and "driver_braking" not in remaining:
        return None
    return True if remaining & {"braking", "driver_braking"} else None

_GRADE_FRAME_LOCATION = (
    "grade_frame must be an object at hypotheses[].grade_frame, alongside geometry, "
    "parameters and condition_changes; not a parameter or condition_changes member. "
    "Use coordinate_frame, slope_reference, heading_relative_to_downhill, "
    "downhill_body_direction, orientation_basis, rationale, and optionally "
    "ego_forward_relative_to_reference. Preserve the proposed orientation; do not invent a new one."
)


def _object(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        detail = (f"; missing={sorted(required - value.keys())}, "
                  f"unexpected={sorted(value.keys() - required - optional)}" if isinstance(value, dict) else "; expected object")
        raise ValueError(f"Physical plan keys must be {sorted(required)} plus {sorted(optional)}" + detail)
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Physical plan requires nonempty text")
    return value


def _texts(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("Expected a bounded text array")
    return [_text(item) for item in value]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite physical number")
    return float(value)


INTERVENTION_ASSUMPTION_FIELDS = frozenset({
    "remote_intervention_available", "other_road_user_avoidance_possible",
})


def _validate_controls(h: dict, evidence_ids: set[str]) -> None:
    for key in ("conditions", "uncovered_conditions", "failed_controls", "remaining_controls"):
        _texts(h[key])
    if not set(h["failed_controls"] + h["remaining_controls"]) <= CONTROL_IDENTITIES:
        raise ValueError("Control identities must be " + ", ".join(sorted(CONTROL_IDENTITIES)))
    if set(h["failed_controls"]) & set(h["remaining_controls"]):
        raise ValueError("A failed control cannot be a remaining control")
    if "braking" in h["failed_controls"] and "driver_braking" in h["remaining_controls"]:
        raise ValueError("A failed braking capability cannot provide remaining driver braking")
    interventions = h.get("intervention_assumptions", {})
    _object(interventions, set(), INTERVENTION_ASSUMPTION_FIELDS)
    proposals = list(interventions.values())
    if "safe_state_proposal" in h:
        proposal = _object(h["safe_state_proposal"],
            {"description", "required_controls", "basis_refs", "conditions"})
        _text(proposal["description"])
        controls = _texts(proposal["required_controls"])
        if not controls or not set(controls) <= set(h["remaining_controls"]):
            raise ValueError("Safe state proposal must depend on explicitly remaining controls")
        proposals.append(proposal)
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Control proposal must be an object")
        if "value" in proposal:
            _object(proposal, {"value", "rationale", "basis_refs", "conditions"})
            if type(proposal["value"]) is not bool:
                raise ValueError("Intervention assumption requires an explicit boolean")
            _text(proposal["rationale"])
        elif proposal is not h.get("safe_state_proposal"):
            raise ValueError("Intervention assumption requires value and rationale")
        refs = _texts(proposal["basis_refs"])
        if not refs or not set(refs) <= evidence_ids or not _texts(proposal["conditions"]):
            raise ValueError("Control proposals require supplied evidence IDs and explicit conditions")
    remote = interventions.get("remote_intervention_available", {})
    if remote.get("value") is True and "remote_control" not in h["remaining_controls"]:
        raise ValueError("Remote intervention requires an explicitly remaining remote control path")


@dataclass(frozen=True)
class PhysicalScenarioPlan:
    hypotheses: tuple[dict, ...]
    uncovered_conditions: tuple[str, ...]
    version: str = PHYSICAL_PLAN_VERSION
    control_assessment: dict | None = None

    @classmethod
    def from_dict(cls, raw: dict, *, evidence_ids: set[str]) -> "PhysicalScenarioPlan":
        # Normalize a notation alias on the owned plan, retaining the Provider
        # response verbatim. This is not a conversion between physical units.
        from copy import deepcopy
        raw = deepcopy(raw)
        _object(raw, {"hypotheses", "uncovered_conditions"}, {"control_assessment"})
        control = raw.get("control_assessment")
        if control is not None:
            _object(control, {"rationale", "basis_refs", "conditions", "uncovered_conditions",
                              "failed_controls", "remaining_controls"}, {"intervention_assumptions"})
            _text(control["rationale"])
            if not _texts(control["basis_refs"]) or not set(control["basis_refs"]) <= evidence_ids:
                raise ValueError("Control assessment requires supplied evidence IDs")
            if not _texts(control["conditions"]):
                raise ValueError("Control assessment requires explicit applicability conditions")
            _validate_controls(control, evidence_ids)
            if raw["hypotheses"]:
                raise ValueError("Control-only assessment must preserve existing motion")
        hypotheses = raw["hypotheses"]
        if not isinstance(hypotheses, list) or not 0 <= len(hypotheses) <= 3:
            raise ValueError("A physical plan contains 0..3 hypotheses")
        uncovered = _texts(raw["uncovered_conditions"])
        if not hypotheses and not uncovered and control is None:
            raise ValueError("An empty physical plan requires a concrete model applicability gap")
        identities = set()
        for h in hypotheses:
            _object(h, {"hypothesis_id", "model", "rationale", "basis_refs", "geometry", "parameters",
                        "conditions", "uncovered_conditions", "failed_controls", "remaining_controls"},
                    {"condition_changes", "grade_frame", "intervention_assumptions", "safe_state_proposal", "operating_condition"})
            if "operating_condition" in h:
                condition = _object(h["operating_condition"], {"source_field", "value", "physical_effect"})
                if condition["source_field"] not in {"odd_road_types", "odd_weather_conditions"}:
                    raise ValueError("Operating condition must select an Item allowed scope")
                _text(condition["value"])
                _text(condition["physical_effect"])
            if "condition_changes" in h:
                if isinstance(h["condition_changes"], dict) and "grade_frame" in h["condition_changes"]:
                    raise ValueError(_GRADE_FRAME_LOCATION)
                changes = _object(h["condition_changes"], {"initial_state", "target", "rationale"})
                if any(type(changes[k]) is not bool for k in ("initial_state", "target")):
                    raise ValueError("Condition changes require explicit boolean flags")
                _text(changes["rationale"])
            if "grade_frame" in h:
                frame = _object(h["grade_frame"], {"coordinate_frame", "slope_reference",
                    "heading_relative_to_downhill", "downhill_body_direction", "orientation_basis", "rationale"},
                    {"ego_forward_relative_to_reference"})
                allowed = {"coordinate_frame": {"BODY_FORWARD_POSITIVE"},
                    "slope_reference": {"EGO_FORWARD", "ROAD_REFERENCE", "UNSPECIFIED"},
                    "heading_relative_to_downhill": {"UPHILL", "DOWNHILL"},
                    "downhill_body_direction": {"FRONT", "REAR"},
                    "orientation_basis": {"SOURCE", "ANALYSIS_ASSUMPTION"}}
                if any(frame[k] not in options for k, options in allowed.items()):
                    raise ValueError("Invalid grade coordinate semantics")
                if frame.get("ego_forward_relative_to_reference", "SAME") not in {"SAME", "OPPOSITE"}:
                    raise ValueError("Unknown road-to-body orientation")
                _text(frame["rationale"])
            identity = _text(h["hypothesis_id"])
            if identity in identities:
                raise ValueError("Duplicate physical hypothesis")
            identities.add(identity)
            if h["model"] not in MODEL_CATALOG:
                raise ValueError("Unknown physical model")
            _text(h["rationale"])
            refs = _texts(h["basis_refs"])
            if not refs or not set(refs) <= evidence_ids:
                raise ValueError(f"hypotheses[{identity}].basis_refs must use supplied IDs {sorted(evidence_ids)}; received {refs}")
            _validate_controls(h, evidence_ids)
            g = _object(h["geometry"], {"road_user_type", "collision_type", "object_position",
                "ego_longitudinal_direction", "object_extent", "object_id"}, {"object_longitudinal_direction"})
            for value in g.values():
                _text(value)
            if g["object_position"] not in {"FRONT", "REAR", "LEFT", "RIGHT"}:
                raise ValueError("Unknown target position")
            if g["ego_longitudinal_direction"] not in {"FORWARD", "REVERSE"}:
                raise ValueError("Post-fault travel direction must be explicit")
            if g["object_extent"] not in {"POINT_ON_TRAVEL_PATH", "EXTENDED_ALONG_TRAVEL_PATH", "FINITE_CONTACT_ENVELOPE"}:
                raise ValueError("Unknown target extent")
            if h["model"] == "RECTILINEAR_MOVING_TARGET_CONTACT":
                if (g.get("object_longitudinal_direction") not in {"FORWARD", "REVERSE", "LEFT", "RIGHT"}
                        or g["object_extent"] != "FINITE_CONTACT_ENVELOPE"):
                    raise ValueError("Moving target requires explicit body-frame direction and finite contact envelope")
            elif g.get("object_longitudinal_direction", "STATIONARY") != "STATIONARY":
                raise ValueError("Stationary-target model cannot declare a moving target")
            params = h["parameters"]
            if not isinstance(params, list) or len(params) > len(PARAMETER_UNITS):
                raise ValueError("Invalid parameter list")
            fields = set()
            for p in params:
                _object(p, {"field", "unit", "kind", "rationale"}, {"value", "lower", "upper", "symbol"})
                field = p["field"]
                if field == "grade_frame":
                    raise ValueError(_GRADE_FRAME_LOCATION)
                if PARAMETER_UNITS.get(field) == "m/s2" and p["unit"] == "m/s²":
                    p["unit"] = "m/s2"
                if field not in PARAMETER_UNITS or field in fields or p["unit"] != PARAMETER_UNITS[field]:
                    raise ValueError(f"Parameter {field!r} must be unique and use its declared unit; "
                                     f"allowed parameter units: {PARAMETER_UNITS}; received unit={p['unit']!r}")
                fields.add(field)
                _text(p["rationale"])
                payload = {"POINT": {"value"}, "RANGE": {"lower", "value", "upper"},
                           "SYMBOLIC": {"symbol"}}.get(p["kind"])
                if payload is None:
                    raise ValueError(f"parameters[{field}].kind must be POINT, RANGE or SYMBOLIC")
                missing = payload - p.keys()
                forbidden = ({"value", "lower", "upper", "symbol"} - payload) & p.keys()
                if missing or forbidden:
                    raise ValueError(f"parameters[{field}] kind={p['kind']} requires {sorted(payload)}; "
                                     f"missing={sorted(missing)}, forbidden={sorted(forbidden)}")
                if p["kind"] == "POINT":
                    values = [_number(p["value"])]
                    if {"lower", "upper", "symbol"} & p.keys():
                        raise ValueError("Point cannot include a range/symbol")
                elif p["kind"] == "RANGE":
                    values = [_number(p[key]) for key in ("lower", "value", "upper")]
                    if not values[0] <= values[1] <= values[2] or "symbol" in p:
                        raise ValueError("Range requires lower <= representative <= upper")
                elif p["kind"] == "SYMBOLIC":
                    _text(p["symbol"])
                    if {"lower", "upper", "value"} & p.keys():
                        raise ValueError("Symbolic parameter cannot claim a numeric bound")
                    values = []
                else:
                    raise ValueError("Unknown parameter kind")
                if field not in {"longitudinal_acceleration_mps2", "lateral_acceleration_mps2", "object_offset_x_m", "object_offset_y_m"} and any(v < 0 for v in values):
                    raise ValueError("Negative magnitude")
        return cls(tuple(hypotheses), tuple(uncovered), control_assessment=control)

    def to_dict(self) -> dict:
        return {"hypotheses": list(self.hypotheses), "uncovered_conditions": list(self.uncovered_conditions),
                **({"control_assessment": self.control_assessment} if self.control_assessment is not None else {})}
