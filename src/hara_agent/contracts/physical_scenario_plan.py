"""Bounded analytical hypotheses, deliberately separate from extracted facts.

The model may propose conditions, never ratings, engineering approval or
measured vehicle limits. Units and supported models are a physics API, not
HARA business rules.
"""
from dataclasses import dataclass
import math
from typing import Any


PHYSICAL_PLAN_VERSION = "physical-plan-v2-scoped-inputs-grade-frame"
PARAMETER_UNITS = {
    "ego_speed_kph": "km/h", "relative_distance_m": "m",
    "longitudinal_acceleration_mps2": "m/s2", "slope_percent": "%",
    "residual_resistance_mps2": "m/s2", "model_valid_until_s": "s",
    "lateral_speed_mps": "m/s", "lateral_acceleration_mps2": "m/s2",
    "lateral_clearance_m": "m",
    "post_fault_speed_cap_kph": "km/h",
}
MODEL_CATALOG = {
    "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE": {
        "required": ["ego_speed_kph", "relative_distance_m", "longitudinal_acceleration_mps2"],
        "scope": "Straight travel toward a stationary target; first contact before stop; acceleration may be negative or zero. Optional post_fault_speed_cap_kph represents an explicitly remaining independent limiter, not the normal operating speed.",
    },
    "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE": {
        "required": ["ego_speed_kph", "relative_distance_m", "slope_percent", "residual_resistance_mps2"],
        "scope": "Initially stationary ego, gravity down a constant grade toward a stationary target. Grade is rise/run percent; residual resistance includes remaining holding capability.",
    },
    "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT": {
        "required": ["ego_speed_kph", "lateral_clearance_m", "lateral_speed_mps", "lateral_acceleration_mps2"],
        "scope": "Steering-induced deviation while ego is moving longitudinally (speed > 0); stationary side object extended along the longitudinal swept path; lateral first contact, constant longitudinal speed. No longitudinal TTC or steering avoidance rule is inferred.",
    },
}
CONTROL_IDENTITIES = {"drive", "steering", "braking", "parking_hold", "speed_limiter", "remote_control"}


def _object(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"Physical plan keys must be {sorted(required)} plus {sorted(optional)}")
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


@dataclass(frozen=True)
class PhysicalScenarioPlan:
    hypotheses: tuple[dict, ...]
    uncovered_conditions: tuple[str, ...]
    version: str = PHYSICAL_PLAN_VERSION

    @classmethod
    def from_dict(cls, raw: dict, *, evidence_ids: set[str]) -> "PhysicalScenarioPlan":
        _object(raw, {"hypotheses", "uncovered_conditions"})
        hypotheses = raw["hypotheses"]
        if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= 3:
            raise ValueError("A physical plan contains 1..3 hypotheses")
        identities = set()
        for h in hypotheses:
            _object(h, {"hypothesis_id", "model", "rationale", "basis_refs", "geometry", "parameters",
                        "conditions", "uncovered_conditions", "failed_controls", "remaining_controls"},
                    {"condition_changes", "grade_frame"})
            if "condition_changes" in h:
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
                raise ValueError("Hypothesis basis must refer to supplied context")
            for key in ("conditions", "uncovered_conditions", "failed_controls", "remaining_controls"):
                _texts(h[key])
            if not set(h["failed_controls"] + h["remaining_controls"]) <= CONTROL_IDENTITIES:
                raise ValueError("Control identities must be " + ", ".join(sorted(CONTROL_IDENTITIES)))
            if set(h["failed_controls"]) & set(h["remaining_controls"]):
                raise ValueError("A failed control cannot be a remaining control")
            g = _object(h["geometry"], {"road_user_type", "collision_type", "object_position",
                "ego_longitudinal_direction", "object_extent", "object_id"})
            for value in g.values():
                _text(value)
            if g["object_position"] not in {"FRONT", "REAR", "LEFT", "RIGHT"}:
                raise ValueError("Unknown target position")
            if g["ego_longitudinal_direction"] not in {"FORWARD", "REVERSE"}:
                raise ValueError("Post-fault travel direction must be explicit")
            if g["object_extent"] not in {"POINT_ON_TRAVEL_PATH", "EXTENDED_ALONG_TRAVEL_PATH"}:
                raise ValueError("Unknown target extent")
            params = h["parameters"]
            if not isinstance(params, list) or len(params) > len(PARAMETER_UNITS):
                raise ValueError("Invalid parameter list")
            fields = set()
            for p in params:
                _object(p, {"field", "unit", "kind", "rationale"}, {"value", "lower", "upper", "symbol"})
                field = p["field"]
                if field not in PARAMETER_UNITS or field in fields or p["unit"] != PARAMETER_UNITS[field]:
                    raise ValueError("Unknown, duplicate or incorrectly dimensioned parameter")
                fields.add(field)
                _text(p["rationale"])
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
                if field not in {"longitudinal_acceleration_mps2", "lateral_acceleration_mps2"} and any(v < 0 for v in values):
                    raise ValueError("Negative magnitude")
        return cls(tuple(hypotheses), tuple(_texts(raw["uncovered_conditions"])))

    def to_dict(self) -> dict:
        return {"hypotheses": list(self.hypotheses), "uncovered_conditions": list(self.uncovered_conditions)}
