"""Display physical quantities without changing calculations or method thresholds."""
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from typing import Any


def quantity(value: Any, unit: str = "", *, boundary: float | None = None) -> str:
    if not isinstance(value, (int, float, Decimal)) or isinstance(value, bool):
        return "待确定"
    if not isfinite(float(value)):
        return "无有限值"
    places = 1 if unit in {"km/h", "%", "°"} else 2
    step = Decimal(1).scaleb(-places)
    raw = Decimal(str(value))
    rounded = raw.quantize(step, rounding=ROUND_HALF_UP)
    def plain(number):
        return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else str(number)
    if raw and not rounded:
        return f"< {plain(step)}" if raw > 0 else f"−{plain(step)} < 值 < 0"
    text = plain(rounded.copy_abs() if not rounded else rounded)
    if boundary is not None and raw != Decimal(str(boundary)) and rounded == Decimal(str(boundary)):
        text = ("> " if raw > rounded else "< ") + text
    return text


def quantity_range(lower: float, upper: float, unit: str, *,
                   lower_inclusive: bool = True, upper_inclusive: bool = True) -> str:
    """Endpoints remain ordered; rounded endpoints are explicitly approximations."""
    left, right = quantity(lower, unit), quantity(upper, unit)
    brackets = ("[" if lower_inclusive else "(", "]" if upper_inclusive else ")")
    collapsed = lower != upper and left == right
    suffix = "（端点舍入后相同；原始端点见追溯）" if collapsed else "（显示约值；端点包含性按原始值）"
    return f"{brackets[0]}{left}, {right}{brackets[1]} {unit}{suffix}"


FIELD_UNITS = {
    "ego_speed_kph": "km/h", "object_speed_kph": "km/h", "ego_impact_speed_kph": "km/h",
    "relative_speed_kph": "km/h", "closing_speed_kph": "km/h", "post_fault_speed_cap_kph": "km/h",
    "relative_distance_m": "m", "lateral_clearance_m": "m", "ttc_s": "s",
    "first_contact_time_s": "s", "model_valid_until_s": "s", "slope_percent": "%",
    "longitudinal_acceleration_mps2": "m/s²", "lateral_acceleration_mps2": "m/s²",
    "residual_resistance_mps2": "m/s²", "lateral_speed_mps": "m/s",
}


def boundary_notes(facts: dict, method) -> str:
    """Explain a rounded value at an active method boundary; never re-score it."""
    structured = getattr(method, "structured_risk_method", None)
    if structured is None:
        return ""
    severity = structured.severity
    group = dict(severity.road_user_groups).get(facts.get("road_user_type"))
    collision = dict(severity.collision_types).get(facts.get("collision_type"))
    speed_bounds = {bound for band in severity.bands
                    if band.collision_group == group and (not band.collision_type or band.collision_type == collision)
                    for bound in (band.lower_kph, band.upper_kph) if bound is not None}
    time_bounds = {bound for band in structured.controllability_profile.bands
                   for bound in (band.lower_ttc_s, band.upper_ttc_s) if bound is not None}
    notes = []
    for field, label, unit, bounds in (("relative_speed_kph", "碰撞相对速度", "km/h", speed_bounds),
                                      ("ttc_s", "TTC", "s", time_bounds)):
        raw = facts.get(field)
        if not isinstance(raw, (float, int)) or isinstance(raw, bool):
            continue
        display = quantity(raw, unit)
        for bound in sorted(bounds):
            if display == quantity(bound, unit) and float(raw) != float(bound):
                relation = ">" if raw > bound else "<"
                # Threshold text is the compiled source value, not an observation.
                notes.append(f"{label} {relation} {bound:g} {unit}；分档使用未舍入值，显示约为 {display} {unit}")
    return "；".join(notes)
