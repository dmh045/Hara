"""Instantiate LLM proposals into scoped conditions for the existing executors.

No network, source extraction, rating rules, or engineering approvals live here.
"""
from copy import deepcopy
from dataclasses import replace, asdict
import math

from hara_agent.contracts.physical_scenario_plan import MODEL_CATALOG, PHYSICAL_PLAN_VERSION, PARAMETER_UNITS, braking_capability, controls_for_driver
from .scenario_equivalence import signature
from .scenario_physics import CONDITIONAL_ORIGIN, derive_scenario_physics, source_is_accepted_for, physical_semantics, grade_rollaway_acceleration
from .risk_vocabulary_adapter import RiskVocabularyAdapter


def summarize_physical_hypotheses(state, method):
    """Summarize calculated points, never splice scores or assert a real bound."""
    from collections import defaultdict
    from .asil_service import MethodContractASILService
    groups = defaultdict(list)
    by_id = {c.scenario_id: c for c in state.scenarios}
    for risk in state.risk_results:
        candidate = by_id[risk.scenario_id]
        analysis = candidate.facts.get("physical_analysis")
        if analysis:
            groups[(risk.malfunction_id, analysis["parent_scenario_id"], analysis["plan_key"])].append((risk, analysis))
    s_order, a_order = list(method.asil.severity_levels), MethodContractASILService(method).result_order()
    summaries = []
    for key, members in groups.items():
        known_s = [r for r, _ in members if r.severity.value in s_order]
        known_a = [r for r, _ in members if r.asil.value in a_order]
        maximum_s = max(known_s, key=lambda r: s_order.index(r.severity.value)) if known_s else None
        maximum_a = max(known_a, key=lambda r: a_order.index(r.asil.value)) if known_a else None
        summaries.append({"malfunction_id": key[0], "parent_scenario_id": key[1], "plan_key": key[2],
            "calculation_nature": "CONDITIONAL", "evaluated_points": len(members),
            "maximum_sampled_severity": maximum_s.severity.value if maximum_s else "",
            "maximum_severity_scenario": maximum_s.scenario_id if maximum_s else "",
            "highest_sampled_asil": maximum_a.asil.value if maximum_a else "",
            "deciding_asil_scenario": maximum_a.scenario_id if maximum_a else "",
            "deciding_SEC": {k: getattr(maximum_a, v).value for k, v in (("S", "severity"), ("E", "exposure"), ("C", "controllability"))} if maximum_a else {},
            "sampled_severity_invariant": bool(known_s) and len(known_s) == len(members) and len({r.severity.value for r in known_s}) == 1,
            "range_invariance_proven": False, "vehicle_upper_bound_proven": False,
            "unresolved_points": [r.scenario_id for r, _ in members if not r.asil.value],
            "deferred_sensitivity_points": [p for _, a in members for p in a["sensitivity"]["deferred_points"]]})
    return summaries


class PhysicalInputConflict(ValueError):
    def __init__(self, reason, field, old, proposed, *, relation="SAME_INITIAL_WORLD", resolution=None):
        super().__init__(reason + ":" + field)
        self.details = {"reason_code": reason, "field": field, "old_value": old,
            "proposed_value": proposed, "relation": relation,
            "resolution": resolution or "Preserve the applicable input, or explicitly propose a supported new initial condition/target; do not silently overwrite."}


class PhysicalHypothesisInstantiationService:
    # A plan's three mechanism slots are shared with parameter sampling. Full
    # ranges and unscheduled boundaries stay in sensitivity, not paid tasks.
    MAX_INSTANCES_PER_PLAN = 3

    def __init__(self, method):
        self.method = method
        self.vocabulary = RiskVocabularyAdapter(method)
        self.physics_solve_count = 0

    def _same_input(self, field, old, proposed):
        if field in {"road_user_type", "collision_type"}:
            a = self.vocabulary.resolve(field=field, raw_value=old)
            b = self.vocabulary.resolve(field=field, raw_value=proposed)
            return a.mapped and b.mapped and a.canonical_value == b.canonical_value
        if field in {"object_position", "ego_longitudinal_direction", "object_longitudinal_direction"}:
            return isinstance(old, str) and isinstance(proposed, str) and old.strip().upper() == proposed.strip().upper()
        return old == proposed

    def task_scope(self, candidate, malfunction_id):
        """Retain solved motion; assess its controls and response once."""
        scope = dict(malfunction_id=malfunction_id, scenario_id=candidate.scenario_id)
        if not (candidate.facts.get("collision_branches") and
                source_is_accepted_for(candidate.fact_provenance.get("collision_branches", {}), **scope)):
            return "MOTION_AND_CONTROLS"
        if candidate.facts.get("control_analysis") and source_is_accepted_for(
                candidate.fact_provenance.get("control_analysis", {}), **scope):
            return ""  # An assessed limitation is not another request on resume.
        if candidate.facts.get("physical_analysis") and source_is_accepted_for(
                candidate.fact_provenance.get("physical_analysis", {}), **scope):
            # The motion task already requested control capabilities. Do not
            # fan a saved plan's unresolved assumptions into another paid pass.
            return ""
        if self.method.metadata.get("project_analysis_policy", {}).get("timing_analysis"):
            # C readiness does not establish a reachable safe state or a timed
            # response. Existing source/method motion needs the same assessment
            # already requested by the autonomous motion path.
            return "CONTROL_INPUTS_ONLY"
        from .hazardous_event_risk_context_service import HazardousEventRiskContextService
        from hara_agent.contracts import RiskContextFactStatus
        contexts = HazardousEventRiskContextService(self.method)
        context = contexts.build(**scope, hazard_node_id="control-readiness",
            scenario={**candidate.facts, "_fact_provenance": candidate.fact_provenance})
        if contexts.controllability_readiness(context)["status"] == "READY":
            return ""
        structured = self.method.structured_risk_method
        fields = {c.field for r in structured.controllability_overrides for c in (*r.all_of, *r.any_of)}
        fields.update(c.field for c in structured.controllability_profile.applicability)
        fields.discard("driver_in_vehicle")  # An operating condition, not a proposed capability.
        return "CONTROL_INPUTS_ONLY" if any(
            getattr(context, field).status is not RiskContextFactStatus.AVAILABLE for field in fields) else ""

    @staticmethod
    def _opposite_target(values, model):
        if model in {"CONSTANT_LATERAL_ACCELERATION_TO_CONTACT", "CONSTANT_SPEED_CURVATURE_TO_CONTACT", "RECTILINEAR_MOVING_TARGET_CONTACT"}:
            return False
        direction, position = values.get("ego_longitudinal_direction"), values.get("object_position")
        return bool(direction and position and (direction, position) not in {("FORWARD", "FRONT"), ("REVERSE", "REAR")})

    def boundaries(self, values: dict, model: str) -> list[dict]:
        """Invert active table boundaries; these are conditions, not ratings/capabilities."""
        if self._opposite_target(values, model):
            return []
        if model == "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE":
            grade = values.get("slope_percent")
            if grade is None:
                return []
            gravity = grade_rollaway_acceleration(grade, 0)
            rows = [{"parameter": "residual_resistance_mps2", "boundary": gravity,
                     "condition": "rollaway from rest requires 0 <= residual resistance < g*sin(atan(grade/100))"}]
            longitudinal = dict(values)
            resistance = values.get("residual_resistance_mps2")
            if resistance is not None:
                longitudinal["longitudinal_acceleration_mps2"] = max(0, gravity - resistance)
            for row in self.boundaries(longitudinal, "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE"):
                if row["parameter"] == "longitudinal_acceleration_mps2":
                    if 0 <= row["boundary"] <= gravity:
                        rows.append({**row, "parameter": "residual_resistance_mps2", "boundary": gravity - row["boundary"],
                                     "condition": "a_net=g*sin(atan(grade/100))-residual_resistance; " + row["condition"]})
                else:
                    rows.append(row)
            return rows
        lateral = model == "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT"
        if not lateral and model != "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE":
            return []
        v0 = values.get("ego_speed_kph")
        if v0 is None:
            return []
        distance_field = "lateral_clearance_m" if lateral else "relative_distance_m"
        acceleration_field = "lateral_acceleration_mps2" if lateral else "longitudinal_acceleration_mps2"
        distance, acceleration = values.get(distance_field), values.get(acceleration_field)
        vy = values.get("lateral_speed_mps")
        if lateral and vy is None:
            return []
        initial_squared = (v0 / 3.6) ** 2 + (vy ** 2 if lateral else 0)
        # For a known acceleration, invert for the missing distance. Otherwise
        # retain the existing acceleration sensitivity at a known distance.
        distance_unknown = distance is None and acceleration is not None
        if not distance_unknown and (distance is None or distance <= 0):
            return []
        rows = []
        severity = self.method.structured_risk_method.severity
        group = dict(severity.road_user_groups).get(values.get("road_user_type"))
        collision = dict(severity.collision_types).get(values.get("collision_type"))
        for band in severity.bands:
            if not group or not collision or band.collision_group != group or band.collision_type and band.collision_type != collision:
                continue
            for bound in (band.lower_kph, band.upper_kph):
                if bound is None:
                    continue
                if values.get("post_fault_speed_cap_kph") is not None and float(bound) > values["post_fault_speed_cap_kph"]:
                    continue
                delta = (float(bound) / 3.6) ** 2 - initial_squared
                if distance_unknown:
                    if acceleration == 0:
                        continue  # Speed does not vary with distance.
                    boundary = delta / (2 * acceleration)
                    if boundary < 0:
                        continue
                    parameter = distance_field
                else:
                    boundary, parameter = delta / (2 * distance), acceleration_field
                rows.append({"parameter": parameter, "boundary": boundary,
                    "method_rule_id": band.rule_id, "speed_boundary_kph": bound,
                    "condition": ("stationary extended side target; |v_rel|^2=v_x^2+v_y0^2+2*a_y*w; " if lateral else
                                  "stationary target on travel path; v^2=v0^2+2*a*d; ") +
                                 "first contact before stop/horizon; any remaining cap applies; this is a method boundary, not a rating"})
        if not lateral and values.get("post_fault_speed_cap_kph") is None:
            for band in self.method.structured_risk_method.controllability_profile.bands:
                for t in (band.lower_ttc_s, band.upper_ttc_s):
                    if t is None or t <= 0:
                        continue
                    a = acceleration if distance_unknown else 2 * (distance - v0 / 3.6 * t) / (t * t)
                    boundary = v0 / 3.6 * t + a * t * t / 2 if distance_unknown else a
                    if v0 / 3.6 + a * t >= 0 and (not distance_unknown or boundary > 0):
                        rows.append({"parameter": distance_field if distance_unknown else acceleration_field,
                            "boundary": boundary, "method_rule_id": band.rule_id, "time_boundary_s": t,
                            "condition": "stationary target on travel path; d=v0*t+a*t^2/2; first contact before stop/horizon; C profile applicability still required"})
        return rows

    def parameter_relations(self, values, h):
        """Small model-specific symbolic output, consumed by causal input and report."""
        symbols = {p["field"]: p["symbol"] for p in h["parameters"] if p["kind"] == "SYMBOLIC"}
        if not symbols:
            return []
        model = h["model"]
        scoped = {**values, **h["geometry"]}
        if self._opposite_target(scoped, model):
            texts = ["当前模型的运动方向与目标位置不构成接近；未知参数不能改变该几何条件，不给出接触速度或TTC。"]
        elif model == "RECTILINEAR_MOVING_TARGET_CONTACT":
            texts = ["车体坐标x前向、y左向；x(t)=dx+(u_x−v_x)t−a_x·t²/2，y(t)=dy+u_y·t。",
                     "接触要求同一时刻|x(t)|≤接触包络半长且|y(t)|≤接触包络半宽；取自车停止前和模型有效期内的首次重合。",
                     "路径相交不保证发生碰撞；交叉接触时间不作为纵向TTC。"]
            if h.get("grade_frame"):
                texts.insert(0, "静止起步纯重力：a=max(0,g·sin(arctan(坡度/100))−阻力)；方向由车头朝向映射，阻力足够时不启动。")
        elif model == "CONSTANT_SPEED_CURVATURE_TO_CONTACT":
            texts = ["等速圆弧代表工况：θ=v·t/R，y=R·(1−cosθ)，总速度保持v；不是纵向分量不变再叠加横向速度。",
                "目标位于转弯一侧且0<w≤R时，首次接触时间t=2R·asin(sqrt(w/(2R)))/v；须在声明有效时段及首个四分之一圆内。",
                "半径、无侧滑及等速均为条件性假设，不是车辆能力上界；侧向首次接触时间不用于纵向TTC分档。"]
        elif model == "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT":
            texts = ["w=v_y0·t+a_y·t²/2；取停止前且模型有效期内的首次非负根。",
                     "v_rel=3.6·sqrt(v_x²+v_y0²+2a_y·w)，v_x=初始车速/3.6；目标须静止并沿纵向路径延伸。",
                     "侧向首次接触时间不作为纵向TTC；C仍须适用的侧向方法。"]
        else:
            texts = ["v0=初始车速/3.6；d为沿运动方向至同一静止目标的行程，a为该阶段沿行程方向的加速度。"]
            if model == "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE":
                texts.append("g_d=g·sin(arctan(坡度/100))；a=max(0,g_d−r)。初始静止、r≥g_d时不启动；0≤r<g_d时沿下坡方向运动。")
            texts += ["v_impact(d)=3.6·sqrt(v0²+2ad)；T(d)=2d/(v0+sqrt(v0²+2ad))。",
                      "上述关系仅在d>0、根号及分母有效、首次接触发生在停止前及模型有效期内成立；未证明真实能力上界。"]
            if values.get("post_fault_speed_cap_kph") is not None:
                texts += ["存在残余限速时，上式仅用于达到限速前；若a>0且v_cap≥v0，t_cap=(v_cap−v0)/a，d_cap=(v_cap²−v0²)/(2a)。",
                          "d>d_cap时v_impact=3.6·v_cap，T=t_cap+(d−d_cap)/v_cap；v_cap使用m/s且须大于0。"]
        return [{"model": model, "relations": texts, "known_inputs": dict(values), "symbolic_parameters": symbols,
                 "parameter_ranges": {p["field"]: {k: p[k] for k in ("lower", "value", "upper", "unit")}
                                      for p in h["parameters"] if p["kind"] == "RANGE"},
                 "geometry": dict(h["geometry"]), "calculation_nature": "CONDITIONAL", "numeric_rating_produced": False}]

    def _samples(self, h, existing, limit):
        values = dict(existing)
        values.update({p["field"]: p["value"] for p in h["parameters"] if "value" in p})
        for p in h["parameters"]:
            if p["kind"] == "SYMBOLIC":
                values.pop(p["field"], None)
        boundaries = self.boundaries({**values, **h["geometry"]}, h["model"])
        samples = [values]
        symbolic_geometry = any(p["kind"] == "SYMBOLIC" and p["field"] in {
            "relative_distance_m", "lateral_clearance_m"} for p in h["parameters"])
        deferred_symbolic = []
        for p in h["parameters"]:
            if p["kind"] != "RANGE":
                continue
            # A representative is a sample, not a new regime boundary. Splitting
            # its containing interval again created two arbitrary extra worlds.
            cuts = {p["lower"], p["upper"]}
            cuts.update(row["boundary"] for row in boundaries
                        if "boundary" in row and row["parameter"] == p["field"] and p["lower"] <= row["boundary"] <= p["upper"])
            # Unknown contact geometry cannot turn same-regime parameter
            # samples into numeric contact results. Keep the entire relation,
            # but preserve stop/reversal/onset boundaries as distinct worlds.
            critical = [row["boundary"] for row in boundaries if "boundary" in row
                        and row["parameter"] == p["field"] and p["lower"] <= row["boundary"] <= p["upper"]]
            if p["lower"] <= 0 <= p["upper"]:
                critical.append(0)
                cuts.add(0)
            cuts = sorted(cuts)
            # One factor at a time; retain table boundaries and neighboring
            # open regions without a multidimensional Cartesian expansion.
            interiors = [(a + b) / 2 for a, b in zip(cuts, cuts[1:])
                         if not a < p["value"] < b]
            for value in [p["lower"], p["upper"], *cuts, *interiors]:
                sample = {**values, p["field"]: value}
                if sample not in samples:
                    if symbolic_geometry and not critical:
                        if sample not in deferred_symbolic:
                            deferred_symbolic.append(sample)
                    else:
                        samples.append(sample)
        return samples[:limit], {
            "sampling": "REPRESENTATIVE_AND_ONE_FACTOR_BOUNDARIES", "planned_point_count": len(samples),
            "deferred_points": samples[limit:] + deferred_symbolic, "method_boundaries": boundaries,
            "symbolic_geometry_deferred_count": len(deferred_symbolic),
            "deferral_basis": "FULL_PARAMETER_RELATION_RETAINED_NO_NEW_NUMERIC_CONTACT_RESULT" if deferred_symbolic else "",
            "parameter_relations": self.parameter_relations(values, h),
            "range_invariance_proven": False, "vehicle_upper_bound_proven": False,
            "instance_limit": limit, "deferred_points_evaluated": False,
        }

    @staticmethod
    def _inheritance_reason(parent, h, field, malfunction_id):
        meta = parent.fact_provenance.get(field, {})
        semantics = physical_semantics(meta)
        phase = str(semantics.get("analysis_value_semantic", meta.get("time_role", ""))).upper()
        if not source_is_accepted_for(meta, malfunction_id=malfunction_id, scenario_id=parent.scenario_id):
            return False, "PARENT_INPUT_NOT_ACCEPTED"
        if meta.get("unit") and meta["unit"] != PARAMETER_UNITS[field]:
            return False, "PARENT_INPUT_UNIT_MISMATCH"
        if meta.get("coordinate_frame") not in {None, "", "BODY_FORWARD_POSITIVE", "TRAVEL_DIRECTION"}:
            return False, "PARENT_COORDINATE_NOT_MAPPED"
        if meta.get("fault_motion_model", h["model"]) != h["model"]:
            return False, "PARENT_MODEL_NOT_APPLICABLE"
        if field == "ego_speed_kph":
            if meta.get("object_id") not in {None, "EGO", "ego", parent.facts.get("ego_id")}:
                return False, "PARENT_INPUT_OTHER_OBJECT"
            return (False, "CONTACT_SPEED_IS_NOT_INITIAL_SPEED") if "IMPACT" in phase or "POST_FAULT" in phase else (True, "PRE_FAULT_SPEED_AS_INITIAL_CONDITION")
        if field == "slope_percent":
            return True, "ROAD_GEOMETRY_CONTEXT"
        if field in {"relative_distance_m", "lateral_clearance_m", "object_speed_kph",
                     "object_offset_x_m", "object_offset_y_m", "contact_half_length_m", "contact_half_width_m"}:
            target = meta.get("object_id", parent.facts.get("object_id"))
            if h.get("condition_changes", {}).get("target") or target and target != h["geometry"]["object_id"]:
                return False, "PARENT_INPUT_OTHER_OBJECT"
            if "IMPACT" in phase or "POST_FAULT" in phase:
                return False, "PARENT_DISTANCE_NOT_AT_INITIAL_STATE"
            return True, "SAME_TARGET_INITIAL_GEOMETRY"
        if "PRE_FAULT" in phase or not ("POST_FAULT" in phase or phase == "FAULT_MOTION_PARAMETER"):
            return False, "PARENT_PARAMETER_PHASE_NOT_APPLICABLE"
        if parent.facts.get("fault_motion_model") not in {None, h["model"]}:
            return False, "PARENT_MODEL_NOT_APPLICABLE"
        if meta.get("object_id") not in {None, "EGO", "ego", parent.facts.get("ego_id")}:
            return False, "PARENT_INPUT_OTHER_OBJECT"
        if field == "longitudinal_acceleration_mps2":
            if meta.get("coordinate_frame") == "BODY_FORWARD_POSITIVE" and h["geometry"]["ego_longitudinal_direction"] != "FORWARD":
                return False, "BODY_AXIS_PARAMETER_REQUIRES_TRAVEL_MAPPING"
            if parent.facts.get("ego_longitudinal_direction") not in {None, h["geometry"]["ego_longitudinal_direction"]}:
                return False, "PARENT_TRAVEL_DIRECTION_CHANGED"
        return True, "SAME_FAULT_PHASE_ROLE_AND_MODEL"

    @staticmethod
    def _required_inputs(h):
        required = set(MODEL_CATALOG[h["model"]]["required"])
        if h["model"] == "RECTILINEAR_MOVING_TARGET_CONTACT" and h.get("grade_frame"):
            required.remove("longitudinal_acceleration_mps2")
            required.update({"slope_percent", "residual_resistance_mps2"})
        return required

    def _model_inputs(self, parent, h, malfunction_id):
        explicit = {p["field"]: p for p in h["parameters"]}
        allowed = self._required_inputs(h) | {"model_valid_until_s"}
        if h["model"] in {"CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE"}:
            allowed.add("post_fault_speed_cap_kph")
        if h["model"] == "RECTILINEAR_MOVING_TARGET_CONTACT" and h.get("grade_frame"):
            if "longitudinal_acceleration_mps2" in explicit:
                raise PhysicalInputConflict("MOVING_GRADE_ACCELERATION_IS_DERIVED", "longitudinal_acceleration_mps2",
                    explicit["longitudinal_acceleration_mps2"], None,
                    resolution="For explicit gravity motion, retain slope_percent/residual_resistance_mps2 (including symbols or ranges) and omit the independent acceleration. The existing gravity model derives it; moving target geometry stays unchanged.")
        values, audit = {}, {}
        for field in PARAMETER_UNITS:
            if field in explicit:
                parameter = explicit[field]
                if field not in allowed:
                    raise PhysicalInputConflict("PARAMETER_NOT_USED_BY_MODEL", field, None, parameter)
                audit[field] = {"action": parameter["kind"], "parameter": parameter}
                if parameter["kind"] != "SYMBOLIC":
                    values[field] = parameter["value"]
            elif field in parent.facts:
                accepted, reason = self._inheritance_reason(parent, h, field, malfunction_id)
                value = parent.facts[field]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    accepted, reason = False, "PARENT_INPUT_NOT_FINITE_NUMBER"
                if field not in allowed:
                    accepted, reason = False, "PARAMETER_NOT_USED_BY_MODEL"
                audit[field] = {"action": "INHERITED" if accepted else "EXCLUDED_FROM_ACTIVE_MODEL", "reason": reason,
                    "parent_value": parent.facts[field], "parent_metadata": deepcopy(parent.fact_provenance.get(field, {}))}
                if accepted:
                    values[field] = parent.facts[field]
        return values, audit

    def input_status(self, candidate, malfunction_id):
        """Request guidance from the same scoped view used by instantiation.

        Missing measurements do not prohibit analytical hypotheses. No value is
        filled here, and the catalog remains the sole required-field list.
        """
        models = {}
        for model, spec in MODEL_CATALOG.items():
            h = {"model": model, "parameters": [], "geometry": {
                "object_id": candidate.facts.get("object_id"),
                "ego_longitudinal_direction": candidate.facts.get("ego_longitudinal_direction")}}
            values, audit = self._model_inputs(candidate, h, malfunction_id)
            models[model] = {
                "available": {f: {"value": values[f], "unit": PARAMETER_UNITS[f],
                    "basis": audit[f]["reason"]} for f in spec["required"] if f in values},
                "needs_hypothesis": [f for f in spec["required"] if f not in values],
                "excluded_parent_inputs": {f: a["reason"] for f, a in audit.items()
                    if f in spec["required"] and a["action"] == "EXCLUDED_FROM_ACTIVE_MODEL"},
                "scope": spec["scope"],
            }
        dimensions = candidate.facts.get("method_scenario_dimensions", {})
        source = candidate.facts.get("source_operating_conditions", {})
        slope_tag = candidate.facts.get("source_slope_direction", source.get("source_slope_direction"))
        return {"models": models,
            "grade_reference_mapping": {
                "selected_source_slope_direction": slope_tag,
                "road_reference_heading_by_alignment": {alignment: self._road_heading(slope_tag, alignment)
                    for alignment in ("SAME", "OPPOSITE")},
                "meaning": "If choosing ROAD_REFERENCE, select a coherent alignment/heading pair from this mapping. The selected parent slope tag is not the project's entire allowed up/down set. These are coordinate relations, not evidence of the actual vehicle heading; declare an orientation assumption when unknown."},
            "derivable_conditions": {k: candidate.facts[k] for k in (
                "source_slope_pct", "source_slope_direction", "slope_reference",
                "ego_speed_constraint") if k in candidate.facts},
            "selected_constraints": {
                "initial_speed_range_kph": dimensions.get("EGO_DYNAMICS", {}).get("speed_intersection_kph"),
                "road_slope": dimensions.get("ROAD", {}).get("method_semantics", {}).get("slope")},
            "policy": "Use applicable inputs; jointly justify missing geometry and motion, or explicitly explain a symbolic relationship. PENDING assumptions can be computed conditionally. Constraints are not measurements."}

    def instantiate(self, candidate, malfunction, plan, *, plan_key: str, response_model: str,
                    project_context=None, function=None):
        results, diagnostics = [], []
        driver_position = candidate.facts.get("driver_position", "")
        if not driver_position:
            location = candidate.facts.get("driver_in_vehicle")
            driver_position = ("in_driver_seat" if location is True else
                               "outside_driver_seat" if location is False else "")
        if plan.control_assessment is not None:
            try:
                return [self._instantiate_controls(candidate, malfunction, controls_for_driver(plan.control_assessment,
                    driver_position),
                    plan_key, response_model)], []
            except PhysicalInputConflict as error:
                return [], [{"reason": str(error), **error.details}]
        for h in plan.hypotheses:
            if "driver_positions" in h and candidate.facts.get("driver_position") not in h["driver_positions"]:
                continue  # Explicit applicability, not a risk-equivalence judgement.
            h = controls_for_driver(h, driver_position)
            try:
                inputs, inheritance = self._model_inputs(candidate, h, malfunction.malfunction_id)
            except PhysicalInputConflict as error:
                diagnostics.append({"hypothesis_id": h["hypothesis_id"], "reason": str(error), **error.details})
                continue
            declared = {p["field"] for p in h["parameters"]}
            required = self._required_inputs(h)
            missing = required - inputs.keys() - declared
            if missing:
                diagnostics.append({"hypothesis_id": h["hypothesis_id"],
                    "reason": "REQUIRED_MODEL_INPUT_OMITTED", "fields": sorted(missing),
                    "resolution": "Jointly justify geometry and motion, or explicitly declare SYMBOLIC with a reason and scope. Never fill a default."})
                continue
            samples, sensitivity = self._samples(h, inputs, self.MAX_INSTANCES_PER_PLAN // len(plan.hypotheses))
            sensitivity["model_input_view"] = inheritance
            sensitivity["input_completeness"] = {
                "classification": "EXPLAINED_SYMBOLIC_RELATION" if any(p["kind"] == "SYMBOLIC"
                    for p in h["parameters"] if p["field"] in required)
                    else "COMPUTABLE_REPRESENTATIVE",
                "required": sorted(required),
                "symbolic_reasons": {p["field"]: p["rationale"] for p in h["parameters"] if p["kind"] == "SYMBOLIC"}}
            for index, values in enumerate(samples):
                try:
                    results.append(self._instantiate(candidate, malfunction, h, values, plan_key, response_model,
                        {**sensitivity, "point_index": index, "uncovered_conditions":
                            list(plan.uncovered_conditions) + h["uncovered_conditions"]}, project_context, function))
                except ValueError as error:
                    diagnostics.append({"hypothesis_id": h["hypothesis_id"], "point_index": index,
                                        "reason": str(error), "parameters": values,
                                        **(error.details if isinstance(error, PhysicalInputConflict) else {})})
        return results, diagnostics

    @staticmethod
    def _rescope(value, old_id, new_id):
        if isinstance(value, dict):
            return {k: PhysicalHypothesisInstantiationService._rescope(v, old_id, new_id) for k, v in value.items()}
        if isinstance(value, list):
            return [PhysicalHypothesisInstantiationService._rescope(v, old_id, new_id) for v in value]
        return new_id if value == old_id else value

    def _instantiate_controls(self, parent, malfunction, assessment, plan_key, model_name):
        """Condition the existing trajectory; never regenerate collision inputs."""
        sid = f"{parent.scenario_id}-CTRL-{signature({'plan': plan_key, 'assessment': assessment})[:16]}"
        facts = self._rescope(deepcopy(parent.facts), parent.scenario_id, sid)
        provenance = self._rescope(deepcopy(parent.fact_provenance), parent.scenario_id, sid)
        scope = {"malfunction_id": malfunction.malfunction_id, "scenario_id": sid}
        meta = {"provenance": "LLM_INFERENCE", "origin": CONDITIONAL_ORIGIN, "approval": "PENDING",
            "analysis_assumption_origin": CONDITIONAL_ORIGIN, "calculation_scope": "CONDITIONAL",
            "analysis_assumption_scope": scope, "applicable_scope": scope,
            "validation_status": "VALIDATED", "validated_by": type(self).__name__,
            "hypothesis_id": f"{plan_key}:control_assessment",
            "method_contract_hash": self.method.metadata.get("method_source_hash", ""),
            "source_refs": [{"source_type": "analysis_hypothesis", "source_id": plan_key,
                "location": "control_assessment", "excerpt": assessment["rationale"]}],
            "basis_refs": assessment["basis_refs"], "conditions": assessment["conditions"]}
        supplied = {k: p["value"] for k, p in assessment.get("intervention_assumptions", {}).items()}
        capability = braking_capability(assessment)
        old_analysis = parent.facts.get("physical_analysis", {})
        if source_is_accepted_for(parent.fact_provenance.get("physical_analysis", {}),
                malfunction_id=malfunction.malfunction_id, scenario_id=parent.scenario_id):
            old_hypothesis = old_analysis.get("hypothesis", {})
            old_capability = braking_capability(old_hypothesis)
            if capability is not None and old_capability is not None and capability != old_capability:
                raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", "emergency_braking_available",
                    old_capability, capability)
            if (set(assessment["remaining_controls"]) & set(old_hypothesis.get("failed_controls", [])) or
                    set(assessment["failed_controls"]) & set(old_hypothesis.get("remaining_controls", []))):
                raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", "controls", old_hypothesis, assessment)
        if capability is not None:
            supplied["emergency_braking_available"] = capability
        if "steering" in assessment["failed_controls"]:
            supplied["steering_control_available"] = False
        for field, value in supplied.items():
            for branch in facts.get("collision_branches", []):
                if (branch.get(field) is not None and branch[field] != value
                        and source_is_accepted_for(branch.get("_fact_provenance", {}).get(field, {}), **scope)):
                    raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", field, branch[field], value,
                        relation=f"COLLISION_BRANCH:{branch.get('branch_id', '')}")
            if facts.get(field) is not None and source_is_accepted_for(provenance.get(field, {}), **scope):
                if facts[field] != value:
                    raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", field, facts[field], value)
                continue
            facts[field], provenance[field] = value, deepcopy(meta)
            if proposal := assessment.get("intervention_assumptions", {}).get(field):
                provenance[field].update(basis_refs=proposal["basis_refs"], conditions=proposal["conditions"])
                provenance[field]["source_refs"][0]["excerpt"] = proposal["rationale"]
        facts["control_analysis"] = {"plan_key": plan_key, "hypothesis": deepcopy(assessment),
            "parent_scenario_id": parent.scenario_id, "response_model": model_name,
            "calculation_nature": "CONDITIONAL", "engineering_approval": "NOT_GRANTED"}
        provenance["control_analysis"] = deepcopy(meta)
        # New capability conditions require a fresh causal task. The original
        # parent, contact identity, trajectory and all physical values survive.
        return replace(parent, scenario_id=sid, source_scenario_id=parent.scenario_id,
            facts=facts, fact_provenance=provenance, status=type(parent.status).PENDING,
            semantic_fingerprint=signature({"parent": parent.semantic_fingerprint, "controls": assessment}),
            analysis_instance={**deepcopy(parent.analysis_instance), "control_analysis": facts["control_analysis"]})

    def operating_atom_options(self, parent, malfunction, project_context, function):
        """Reuse the production shortlist; no extra semantic request or selector."""
        from .scenario_synthesis_service import ConstrainedScenarioSynthesisService
        bindings = deepcopy(parent.facts.get("method_scenario_dimensions", {}))
        if not bindings:
            return {}
        for dimension in ("EGO_ACTION", "EGO_DYNAMICS", "TRAFFIC_PATTERN"):
            bindings[dimension] = {"binding_authority": "ANALYTICAL_SELECTION", "refinable": True,
                                   "resolution_status": "PENDING"}
        query = replace(parent, facts={**parent.facts, "method_scenario_dimensions": bindings})
        synthesis = ConstrainedScenarioSynthesisService(self.method)
        inputs = synthesis.build_input(
            malfunction=asdict(malfunction), parent=query, assessment={},
            project_context=project_context or {}, function=function)
        unbound_object = (not parent.facts.get("road_user_type") and not parent.facts.get("object_type")
                          and not bindings.get("OBJECT", {}).get("atom_id"))
        pools = inputs.dimension_candidate_sets
        if (unbound_object and not inputs.structured_semantic_query.get("object_categories")
                and not inputs.structured_semantic_query.get("unmapped_object_sources")):
            from .scenario_selection_quality import ScenarioCandidateRanker
            # This query asks which analytical objects the active method can
            # represent; it does not assert their presence in the source world.
            categories = set().union(*(ScenarioCandidateRanker.atom_categories(a)
                for a in synthesis.catalog if "OBJECT" in a.get("filled_dimensions", [])))
            choices = sorted("OBJECT_" + name for name, _ in self.method.structured_risk_method.severity.road_user_groups
                             if "OBJECT_" + name in categories)
            object_query = {**inputs.structured_semantic_query, "object_categories": choices}
            object_pools = synthesis.candidate_sets(malfunction=asdict(malfunction), parent=query,
                assessment={}, project_context=project_context or {}, query=object_query,
                fm_template=inputs.fm_scenario_template)
            pools = tuple(p for p in pools if p.dimension != "OBJECT") + tuple(
                p for p in object_pools if p.dimension == "OBJECT")
        options = {p.dimension: [{"atom_id": c.atom_id, "label": c.label,
                              "fills": list(c.dimensions), "semantics": c.method_semantics,
                              "speed_range_kph": c.speed_range_kph}
                             for c in p.candidates]
                for p in pools
                if p.dimension in {"EGO_ACTION", "EGO_DYNAMICS", "TRAFFIC_PATTERN"}
                   or p.dimension == "OBJECT" and unbound_object}
        # A shared motion request must not lose the opposite explicit driver
        # qualifier merely because the shortlist kept one family representative.
        from .scenario_selection_quality import explicit_driver_presence
        labels = [x["label"] for x in options.get("EGO_ACTION", [])]
        if any(explicit_driver_presence(label) is not None for label in labels):
            for atom in synthesis.catalog:
                label = atom.get("label", "")
                if ("EGO_ACTION" in atom.get("filled_dimensions", []) and explicit_driver_presence(label) is not None
                        and "passenger" not in label.casefold()
                        and atom["atom_id"] not in {x["atom_id"] for x in options["EGO_ACTION"]}):
                    options["EGO_ACTION"].append({"atom_id": atom["atom_id"], "label": label,
                        "fills": atom["filled_dimensions"], "semantics": atom.get("physical_semantics", {}),
                        "speed_range_kph": synthesis._speed_range(atom)})
        return options

    def _object_binding(self, parent, malfunction, geometry, project_context, function, *, initial=None,
                        dimensions=("OBJECT",), explicit=None, current_facts=None, exclusions=None):
        """Refine changed initial conditions through the existing validator.

        Post-fault speed/direction never enter initial-operation Atom selection.
        Other dimensions must retain their exact selected identities.
        """
        from hara_agent.contracts import ScenarioSynthesisAssessment, CoverageLabel, SynthesisValidationStatus
        from hara_agent.contracts.scenario_synthesis import CandidateOrigin, ScenarioDimensionApplicability
        from .scenario_synthesis_service import ConstrainedScenarioSynthesisService
        bindings = deepcopy(parent.facts.get("method_scenario_dimensions", {}))
        if not bindings:
            return {}, "NO_STRUCTURED_OBJECT_BINDING"
        synthesis = ConstrainedScenarioSynthesisService(self.method)
        for dimension in exclusions or {}:
            authority = synthesis.binding_policy.decide(parent, dimension, None)
            if authority.parent_atom_id and not authority.refinable:
                return {}, "AUTHORITATIVE_CONDITION_CANNOT_BE_EXCLUDED:" + dimension
        for dimension in dimensions:
            bindings[dimension] = {"binding_authority": "ANALYTICAL_SELECTION", "refinable": True,
                "resolution_status": "PENDING", "project_value": geometry["road_user_type"] if dimension == "OBJECT" else str(initial)}
        query_parent = replace(parent, facts={**parent.facts, "method_scenario_dimensions": bindings,
            **{k: current_facts[k] for k in ("object_speed_kph", "object_position", "object_longitudinal_direction")
               if current_facts and k in current_facts},
            "road_user_type": geometry["road_user_type"], "object_type": geometry["road_user_type"].lower(),
            **({"ego_speed_kph": initial, "ego_speed_constraint": {"min_kph": initial, "max_kph": initial}}
               if "EGO_DYNAMICS" in dimensions else {})})
        inputs = synthesis.build_input(malfunction=asdict(malfunction), parent=query_parent,
            assessment={}, project_context=project_context or {}, function=function)
        # A local rebind must not re-rank unrelated, already materialized
        # conditions. Preserve their selected identity/applicability, while the
        # existing validator still checks compound, speed and method constraints.
        pools = []
        for pool in inputs.dimension_candidate_sets:
            binding = bindings.get(pool.dimension, {})
            # A scoped hypothesis may explicitly exclude catalog alternatives.
            # Null alone remains unknown. Required/source-bound conditions still
            # go through the ordinary validator and cannot be erased this way.
            if (pool.dimension in (exclusions or {}) and explicit.get(pool.dimension) is None
                    and pool.applicability.status is ScenarioDimensionApplicability.OPTIONAL):
                pool = replace(pool, candidates=(), locked_atom_ids=(), applicability=replace(pool.applicability,
                    status=ScenarioDimensionApplicability.NOT_APPLICABLE,
                    reason=exclusions[pool.dimension],
                    source_refs=("HYPOTHESIS.operating_atom_exclusions." + pool.dimension,)))
            if pool.dimension not in dimensions:
                atom = synthesis.by_id.get(binding.get("atom_id"))
                if atom and binding.get("resolution_status") == "RESOLVED":
                    retained = synthesis._candidate(atom, origin=CandidateOrigin.BINDING_REFINEMENT,
                        refs=("PARENT." + pool.dimension,), reason="Unchanged selected parent condition", validated=True)
                    pool = replace(pool, candidates=(retained,), locked_atom_ids=(retained.atom_id,))
                if binding.get("applicability_status") == "NOT_APPLICABLE":
                    pool = replace(pool, candidates=(), locked_atom_ids=(), applicability=replace(pool.applicability,
                        status=ScenarioDimensionApplicability.NOT_APPLICABLE,
                        reason=binding.get("applicability_reason", "Unchanged parent applicability")))
            pools.append(pool)
        inputs = replace(inputs, dimension_candidate_sets=tuple(pools))
        selections = {pool.dimension: () for pool in inputs.dimension_candidate_sets}
        selections.update({dimension: (binding["atom_id"],) for dimension, binding in bindings.items()
                           if binding.get("resolution_status") == "RESOLVED" and binding.get("atom_id")})
        choices = []
        for dimension in dimensions:
            candidate_set = next((p for p in inputs.dimension_candidate_sets if p.dimension == dimension), None)
            pool = candidate_set.candidates if candidate_set else ()
            if candidate_set and candidate_set.applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE:
                if explicit and explicit.get(dimension) is not None:
                    return {}, "ATOM_SELECTED_FOR_NONAPPLICABLE_DIMENSION:" + dimension
                choices.append((dimension, (None,)))
                continue
            if explicit and dimension in explicit:
                selected_id = explicit[dimension]
                if selected_id is None and dimension == "TRAFFIC_PATTERN":
                    # Empty selection is still checked by the shared applicability validator.
                    choices.append((dimension, (None,)))
                    continue
                if dimension == "EGO_ACTION":
                    from .scenario_selection_quality import explicit_driver_presence
                    atom = synthesis.by_id.get(selected_id, {})
                    presence = explicit_driver_presence(atom.get("label", ""))
                    if presence is not None:
                        if parent.facts.get("driver_in_vehicle") is not presence:
                            return {}, "ATOM_DRIVER_CONDITION_CONFLICT:" + selected_id
                        # This is the explicitly qualified counterpart offered
                        # by operating_atom_options. The normal materializer still
                        # checks domains, compound occupancy and all other inputs.
                        if not any(c.atom_id == selected_id for c in pool):
                            selected_candidate = synthesis._candidate(atom, origin=CandidateOrigin.BINDING_REFINEMENT,
                                refs=("PARENT.driver_in_vehicle",), reason="Explicit driver-qualified operation", validated=True)
                            pool = (*pool, selected_candidate)
                            inputs = replace(inputs, dimension_candidate_sets=tuple(
                                replace(p, candidates=tuple(pool)) if p.dimension == dimension else p
                                for p in inputs.dimension_candidate_sets))
                pool = [c for c in pool if c.atom_id == selected_id]
            if dimension == "EGO_DYNAMICS" and initial is not None:
                pool = [c for c in pool if (explicit and dimension in explicit and c.speed_range_kph is None) or (c.speed_range_kph is not None and
                    (c.speed_range_kph[0] is None or c.speed_range_kph[0] <= initial) and
                    (c.speed_range_kph[1] is None or initial <= c.speed_range_kph[1]))]
            if not pool:
                return {}, "NO_APPLICABLE_CONDITION_ATOM:" + dimension
            choices.append((dimension, pool))
        reasons = []
        # Existing bounded shortlists only. Stop at the first fully validated
        # combination; this does not create an additional scenario population.
        from itertools import product
        for combination in product(*(pool for _, pool in choices)):
            selected = {dimension: ((candidate.atom_id,) if candidate else ())
                        for (dimension, _), candidate in zip(choices, combination)}
            selection = ScenarioSynthesisAssessment(inputs.semantic_group_id, CoverageLabel.TYPICAL,
                {**selections, **selected}, "Explicit independent child condition; unrelated Atom identities retained",
                ("SCN.road_user_type",), SynthesisValidationStatus.VALIDATED)
            try:
                child, _ = synthesis.materialize(synthesis_input=inputs, assessment=selection, parent=query_parent,
                    provider_evidence={"provider_calls": 0, "selection_authority": "EXISTING_OBJECT_SHORTLIST"})
                # The shared materializer supplies source-linked bindings; it
                # cannot change unrelated original conditions through this API.
                new_bindings = child.facts["method_scenario_dimensions"]
                for pool in inputs.dimension_candidate_sets:
                    if pool.applicability.source_refs == ("HYPOTHESIS.operating_atom_exclusions." + pool.dimension,):
                        new_bindings[pool.dimension].update(
                            dimension_source="PHYSICAL_HYPOTHESIS",
                            applicability_source_refs=list(pool.applicability.source_refs),
                            applicability_nature="CONDITIONAL", engineering_approval="NOT_GRANTED")
                if any(new_bindings.get(dimension, {}).get("atom_id") != binding.get("atom_id")
                       for dimension, binding in bindings.items() if dimension not in dimensions):
                    continue
                return {"method_scenario_dimensions": new_bindings,
                        "scenario_atom_ids": child.facts["scenario_atom_ids"],
                        **({"ego_speed_constraint": child.facts["ego_speed_constraint"]} if "EGO_DYNAMICS" in dimensions else {})}, "EXISTING_SYNTHESIS_VALIDATED_CONDITION_REFINEMENT"
            except ValueError as error:
                reasons.append(str(error))
        return {}, "; ".join(dict.fromkeys(reasons)) or "NO_APPLICABLE_OBJECT_ATOM"

    @staticmethod
    def _road_heading(tag, alignment):
        heading = {"上坡": "UPHILL", "下坡": "DOWNHILL", "UPHILL": "UPHILL", "DOWNHILL": "DOWNHILL"}.get(tag)
        if heading and alignment == "OPPOSITE":
            return "DOWNHILL" if heading == "UPHILL" else "UPHILL"
        return heading

    @staticmethod
    def _grade_mapping(parent, h):
        frame = h.get("grade_frame")
        if not frame:
            raise PhysicalInputConflict("GRADE_ORIENTATION_REQUIRED", "grade_frame", None, None,
                relation="Source grade magnitude/tag alone does not establish vehicle heading; propose an explicit bounded orientation assumption")
        heading = frame["heading_relative_to_downhill"]
        downhill = "REAR" if heading == "UPHILL" else "FRONT"
        source = parent.facts.get("source_operating_conditions", {})
        tag = parent.facts.get("source_slope_direction", source.get("source_slope_direction"))
        reference = parent.facts.get("slope_reference", "UNSPECIFIED")
        if reference != "UNSPECIFIED" and frame["slope_reference"] != reference:
            raise PhysicalInputConflict("SLOPE_REFERENCE_CONFLICT", "slope_reference", reference, frame["slope_reference"])
        source_heading = parent.facts.get("ego_heading_relative_to_downhill")
        tag_heading = PhysicalHypothesisInstantiationService._road_heading(tag, "SAME")
        if tag_heading and frame["slope_reference"] == "EGO_FORWARD":
            source_heading = tag_heading
        elif tag_heading and frame["slope_reference"] == "ROAD_REFERENCE":
            alignment = frame.get("ego_forward_relative_to_reference")
            if not alignment:
                raise PhysicalInputConflict("ROAD_TO_BODY_MAPPING_REQUIRED", "ego_forward_relative_to_reference", None, None)
            source_heading = PhysicalHypothesisInstantiationService._road_heading(tag, alignment)
        if source_heading and source_heading != heading:
            raise PhysicalInputConflict("GRADE_HEADING_CONFLICT", "heading_relative_to_downhill", source_heading, heading,
                relation="GRADE_REFERENCE_AND_BODY_ALIGNMENT",
                resolution="Make slope_reference, ego_forward_relative_to_reference and heading_relative_to_downhill mutually consistent with the source slope tag. For ROAD_REFERENCE, SAME preserves the source uphill/downhill direction and OPPOSITE reverses it. Changing only condition_changes.initial_state or its rationale cannot resolve a contradictory coordinate mapping. Preserve any explicitly sourced vehicle heading.")
        if frame["orientation_basis"] == "SOURCE" and not (parent.facts.get("ego_heading_relative_to_downhill") or reference != "UNSPECIFIED" and tag_heading):
            raise PhysicalInputConflict("GRADE_ORIENTATION_NOT_SOURCE_GROUNDED", "orientation_basis", "UNSPECIFIED", "SOURCE")
        direction = "REVERSE" if downhill == "REAR" else "FORWARD"
        if frame["downhill_body_direction"] != downhill or h["geometry"]["ego_longitudinal_direction"] != direction:
            raise PhysicalInputConflict("GRAVITY_DIRECTION_CONFLICT", "ego_longitudinal_direction", direction,
                h["geometry"]["ego_longitudinal_direction"], relation="Pure gravity follows downhill; residual resistance opposes motion")
        return {"grade_coordinate_frame": "BODY_FORWARD_POSITIVE", "slope_reference": frame["slope_reference"],
            "ego_heading_relative_to_downhill": heading, "downhill_body_direction": downhill,
            "grade_orientation_basis": frame["orientation_basis"]}

    def _instantiate(self, parent, malfunction, h, values, plan_key, model_name, sensitivity, project_context, function):
        g = deepcopy(h["geometry"])
        operating_condition = h.get("operating_condition")
        if operating_condition and operating_condition["value"] not in (project_context or {}).get(operating_condition["source_field"], []):
            raise PhysicalInputConflict("OPERATING_CONDITION_OUTSIDE_ITEM_SCOPE", operating_condition["source_field"],
                (project_context or {}).get(operating_condition["source_field"], []), operating_condition["value"])
        changes = h.get("condition_changes", {})
        new_initial, new_target = changes.get("initial_state", False), changes.get("target", False)
        if new_target and parent.facts.get("object_id") == g["object_id"]:
            raise PhysicalInputConflict("NEW_TARGET_REQUIRES_NEW_ID", "object_id", parent.facts.get("object_id"), g["object_id"])
        for field in ("road_user_type", "collision_type"):
            resolution = self.vocabulary.resolve(field=field, raw_value=g[field])
            if not resolution.mapped:
                raise ValueError(f"UNSUPPORTED_METHOD_VOCABULARY:{field}")
            g[field] = resolution.canonical_value
        lateral = h["model"] in {"CONSTANT_LATERAL_ACCELERATION_TO_CONTACT", "CONSTANT_SPEED_CURVATURE_TO_CONTACT"}
        if lateral:
            if g["object_position"] not in {"LEFT", "RIGHT"} or g["object_extent"] != "EXTENDED_ALONG_TRAVEL_PATH":
                raise ValueError("LATERAL_MODEL_REQUIRES_EXTENDED_SIDE_TARGET")
        elif h["model"] == "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE" and (g["ego_longitudinal_direction"], g["object_position"]) not in {("FORWARD", "FRONT"), ("REVERSE", "REAR")}:
            raise ValueError("TARGET_NOT_ON_POST_FAULT_TRAVEL_PATH")
        speed_range = parent.facts.get("ego_speed_constraint", {})
        speed_binding = parent.facts.get("method_scenario_dimensions", {}).get("EGO_DYNAMICS", {})
        if new_initial and speed_binding.get("project_speed_envelope_kph"):
            low, high = speed_binding["project_speed_envelope_kph"]
            speed_range = {"min_kph": low if low is not None else 0, "max_kph": high if high is not None else math.inf}
        initial = values.get("ego_speed_kph")
        if not new_initial and initial is not None:
            atom_range = speed_binding.get("speed_intersection_kph") or speed_binding.get("child_speed_range_kph")
            if atom_range and not ((atom_range[0] is None or atom_range[0] <= initial) and
                                   (atom_range[1] is None or initial <= atom_range[1])):
                raise PhysicalInputConflict("INITIAL_STATE_ATOM_CONFLICT", "ego_speed_kph", atom_range, initial)
        if lateral and initial is not None and initial <= 0:
            raise ValueError("STEERING_LATERAL_MODEL_REQUIRES_LONGITUDINAL_MOTION")
        previous_direction = str(parent.facts.get("ego_longitudinal_direction", "")).upper()
        if initial and not new_initial and previous_direction in {"FORWARD", "REVERSE"} and previous_direction != g["ego_longitudinal_direction"]:
            raise ValueError("DIRECTION_REVERSAL_REQUIRES_A_STOP_AND_SEPARATE_PHASE")
        if initial is not None and speed_range and not (
            speed_range.get("min_kph", speed_range.get("speed_min_kph", 0)) <= initial <=
            speed_range.get("max_kph", speed_range.get("speed_max_kph", math.inf))):
            raise ValueError("INITIAL_SPEED_OUTSIDE_OPERATING_CONSTRAINT")
        slope = values.get("slope_percent")
        from .scenario_synthesis_service import ConstrainedScenarioSynthesisService
        project_slope_max = ConstrainedScenarioSynthesisService._project_slope_max(project_context or {})
        if slope is not None and project_slope_max is not None and slope > project_slope_max:
            raise ValueError("SLOPE_OUTSIDE_PROJECT_ODD")
        boundary = parent.facts.get("source_slope_pct")
        if slope is not None and boundary is not None and parent.facts.get("source_grade_semantic") == "ALLOWED_ODD_BOUNDARY_REPRESENTATIVE" and slope != boundary:
            raise PhysicalInputConflict("SLOPE_CONFLICTS_WITH_SELECTED_BOUNDARY", "slope_percent", boundary, slope)
        road_slope = parent.facts.get("method_scenario_dimensions", {}).get("ROAD", {}).get("method_semantics", {}).get("slope", {})
        road_range = road_slope.get("range_pct", [road_slope.get("pct_min"), road_slope.get("pct_max")])
        if slope is not None and road_slope:
            lower, upper = road_range
            if (lower is not None and (slope < lower or slope == lower and not road_slope.get("min_inclusive", True))
                    or upper is not None and (slope > upper or slope == upper and not road_slope.get("max_inclusive", True))):
                raise ValueError("SLOPE_CONFLICTS_WITH_SELECTED_ROAD_ATOM")
        if slope is not None:
            from .scenario_constraint_service import ScenarioConstraintExecutor
            selected = [{"atom_id": binding.get("atom_id", dimension),
                         "physical_semantics": binding.get("method_semantics", {})}
                        for dimension, binding in parent.facts.get("method_scenario_dimensions", {}).items()]
            selected.append({"atom_id": "CURRENT_SLOPE_PARAMETER",
                             "physical_semantics": {"slope": {"range_pct": [slope, slope]}}})
            conflicts = ScenarioConstraintExecutor.physical_semantic_conflicts(selected)
            if conflicts:
                raise PhysicalInputConflict("SLOPE_CONFLICTS_WITH_SELECTED_CONDITIONS", "slope_percent", conflicts, slope)
        gravity_motion = h["model"] == "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE" or bool(h.get("grade_frame"))
        if gravity_motion and initial != 0:
            raise ValueError("ROLLAWAY_REQUIRES_STATIONARY_INITIAL_STATE")
        grade_mapping = self._grade_mapping(parent, h) if gravity_motion else {}
        if "post_fault_speed_cap_kph" in values and "speed_limiter" not in h["remaining_controls"]:
            raise ValueError("SPEED_CAP_REQUIRES_EXPLICIT_REMAINING_LIMITER")
        target = {} if h["model"] == "RECTILINEAR_MOVING_TARGET_CONTACT" else {
            "object_speed_kph": 0, "object_longitudinal_direction": "STATIONARY"}
        supplied = {**g, **values, **target,
                    "fault_motion_model": h["model"], **grade_mapping}
        if "post_stop_motion" in h:
            supplied["post_stop_motion"] = h["post_stop_motion"]
        supplied.update({field: proposal["value"]
                         for field, proposal in h.get("intervention_assumptions", {}).items()})
        # Source facts stay intact. A different analytical initial point may
        # replace a method representative within its range, never a measurement.
        for field, value in supplied.items():
            old = parent.fact_provenance.get(field, {})
            if field == "ego_speed_kph" and new_initial or new_target and field in {"object_id", "object_position", "road_user_type", "collision_type", "object_speed_kph", "object_longitudinal_direction", "relative_distance_m", "lateral_clearance_m", "other_road_user_avoidance_possible"}:
                continue
            if field in PARAMETER_UNITS and field != "ego_speed_kph" and not self._inheritance_reason(parent, h, field, malfunction.malfunction_id)[0]:
                continue
            if (field in parent.facts and not self._same_input(field, parent.facts[field], value)
                    and source_is_accepted_for(old, malfunction_id=malfunction.malfunction_id, scenario_id=parent.scenario_id)):
                if field == "object_id" and not new_target:
                    raise PhysicalInputConflict("TARGET_ID_CHANGED_WITHOUT_NEW_CONDITION", field, parent.facts[field], value)
                raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", field, parent.facts[field], value)
        identity = signature({"parent": parent.scenario_id, "plan": plan_key, "hypothesis": h, "point": values})[:16]
        sid = f"{parent.scenario_id}-PH-{identity}"

        def rescope(value):
            return self._rescope(value, parent.scenario_id, sid)

        facts, provenance = rescope(deepcopy(parent.facts)), rescope(deepcopy(parent.fact_provenance))
        if new_target:
            # An independent target has its own ability to evade. Preserve the
            # parent evidence, but do not use it as this target's C input.
            facts.pop("other_road_user_avoidance_possible", None)
            provenance.pop("other_road_user_avoidance_possible", None)
        stale = {"collision_branches", "relative_speed_kph", "closing_speed_kph", "ttc_s", "impact_speed_kph",
                 "ego_impact_speed_kph", "object_impact_speed_kph", "motion_model_outcome", "delta_v_kph",
                 "impact_time_id", "trajectory_id", "first_contact_time_s", "fault_motion_model", "trajectory_max_speed_kph",
                 "travel_to_impact_m", "gravity_body_acceleration_mps2", "post_stop_motion",
                 "grade_coordinate_frame", "slope_reference", "ego_heading_relative_to_downhill",
                 "downhill_body_direction", "grade_orientation_basis"}
        for field in stale:
            facts.pop(field, None); provenance.pop(field, None)
        # Keep the parent immutable, but expose only validated inputs to the
        # active model. Explicit symbols and other-stage values stay in audit.
        for field in PARAMETER_UNITS:
            facts.pop(field, None); provenance.pop(field, None)
            if field in values and self._inheritance_reason(parent, h, field, malfunction.malfunction_id)[0] and parent.facts.get(field) == values[field]:
                facts[field] = values[field]
                provenance[field] = rescope(deepcopy(parent.fact_provenance[field]))
        impact_id = f"CONTACT-{identity}"
        supplied.update(impact_time_id=impact_id, trajectory_id=f"TRAJECTORY-{identity}")
        if target:
            supplied["object_impact_speed_kph"] = 0
        if "relative_distance_m" in values:
            supplied["travel_to_impact_m"] = values["relative_distance_m"]
        hypothesis_id = f"{plan_key}:{h['hypothesis_id']}:{sensitivity['point_index']}"
        meta = {"provenance": "LLM_INFERENCE", "origin": CONDITIONAL_ORIGIN, "approval": "PENDING",
            "analysis_assumption_origin": CONDITIONAL_ORIGIN, "calculation_scope": "CONDITIONAL",
            "analysis_assumption_scope": {"malfunction_id": malfunction.malfunction_id, "scenario_id": sid},
            "applicable_scope": {"malfunction_id": malfunction.malfunction_id, "scenario_id": sid},
            "validation_status": "VALIDATED", "validated_by": type(self).__name__, "hypothesis_id": hypothesis_id,
            "method_contract_hash": self.method.metadata.get("method_source_hash", ""),
            "source_refs": [{"source_type": "analysis_hypothesis", "source_id": plan_key,
                             "location": hypothesis_id, "excerpt": h["rationale"]}],
            "analysis_value_semantic": "POST_FAULT_CONDITIONAL_MODEL", "impact_time_id": impact_id}
        for field, value in supplied.items():
            # Preserve actual project input and governed initial selections when
            # the proposal uses the same value; don't rewrite their origin.
            if field in facts and self._same_input(field, facts[field], value) and source_is_accepted_for(provenance.get(field, {}),
                    malfunction_id=malfunction.malfunction_id, scenario_id=sid):
                facts[field] = value  # Canonical spelling; preserve the original authority and parent record.
                continue
            facts[field], provenance[field] = value, deepcopy(meta)
            if field in h.get("intervention_assumptions", {}):
                assumption = h["intervention_assumptions"][field]
                provenance[field]["basis_refs"] = list(assumption["basis_refs"])
                provenance[field]["conditions"] = list(assumption["conditions"])
                provenance[field]["source_refs"][0]["excerpt"] = assumption["rationale"]
        if "ego_speed_kph" in values:
            provenance["ego_speed_kph"] = {**provenance["ego_speed_kph"], "analysis_value_semantic": "PRE_FAULT_OPERATING_SPEED"}
        if "steering" in [control.lower() for control in h["failed_controls"]]:
            facts["steering_control_available"], provenance["steering_control_available"] = False, deepcopy(meta)
        if braking_capability(h) is False:
            facts["emergency_braking_available"], provenance["emergency_braking_available"] = False, deepcopy(meta)
        elif braking_capability(h) is True:
            if (facts.get("emergency_braking_available") is False and source_is_accepted_for(
                    provenance.get("emergency_braking_available", {}),
                    malfunction_id=malfunction.malfunction_id, scenario_id=sid)):
                raise PhysicalInputConflict("CONFLICT_WITH_SELECTED_CONDITION", "emergency_braking_available", False, True)
            facts["emergency_braking_available"], provenance["emergency_braking_available"] = True, deepcopy(meta)
        info = {"version": PHYSICAL_PLAN_VERSION, "plan_key": plan_key, "hypothesis_id": hypothesis_id,
            "hypothesis": h, "sample_parameters": values, "sensitivity": sensitivity,
            "response_model": model_name, "calculation_nature": "CONDITIONAL", "engineering_approval": "NOT_GRANTED",
            "parent_scenario_id": parent.scenario_id,
            "input_origins": {field: provenance.get(field, {}).get("origin", provenance.get(field, {}).get("provenance")) for field in supplied},
            "consumer": "ordinary_analyze:causality->StructuredRiskScoringService->report"}
        info["model_input_view"] = sensitivity["model_input_view"]
        info["condition_changes"] = changes
        info["grade_mapping"] = {"source_tag": parent.facts.get("source_slope_direction", parent.facts.get("source_operating_conditions", {}).get("source_slope_direction")),
            "source_reference": parent.facts.get("slope_reference", "UNSPECIFIED"), **grade_mapping} if grade_mapping else {}
        # A newly assumed object does not inherit another object's Exposure.
        dimensions = []
        if new_target or not self._same_input("road_user_type", parent.facts.get("road_user_type"), g["road_user_type"]):
            dimensions.append("OBJECT")
        if new_initial and initial is not None and parent.facts.get("ego_speed_kph") != initial:
            dimensions.append("EGO_DYNAMICS")
        location = parent.facts.get("driver_in_vehicle")
        driver_position = "in_driver_seat" if location is True else "outside_driver_seat" if location is False else ""
        explicit_atoms = {**h.get("operating_atoms", {}), **h.get("driver_operating_atoms", {}).get(driver_position, {})}
        dimensions = list(dict.fromkeys([*dimensions, *explicit_atoms]))
        if dimensions:
            refinement, reason = self._object_binding(parent, malfunction, g, project_context, function,
                initial=initial, dimensions=dimensions, explicit=explicit_atoms, current_facts=facts,
                exclusions=h.get("operating_atom_exclusions"))
            if explicit_atoms and not refinement:
                raise PhysicalInputConflict("OPERATING_ATOM_BINDING_CONFLICT", "operating_atoms",
                    {d: b.get("atom_id") for d, b in parent.facts.get("method_scenario_dimensions", {}).items()},
                    explicit_atoms, relation=reason)
            if "EGO_DYNAMICS" in dimensions and not refinement:
                raise PhysicalInputConflict("NEW_INITIAL_CONDITION_BINDING_UNRESOLVED", "EGO_DYNAMICS", speed_binding, initial, relation=reason)
            facts.update(refinement)
            # Keep visible condition facts aligned with the rebound Atom IDs.
            if refinement:
                projected_facts, projected_provenance = {}, {}
                ConstrainedScenarioSynthesisService.project_validated_atom_facts(
                    facts=projected_facts, provenance=projected_provenance,
                    bindings=refinement["method_scenario_dimensions"],
                    method_hash=self.method.metadata.get("method_source_hash", ""),
                    scope={"malfunction_id": malfunction.malfunction_id, "scenario_id": sid})
                for dimension in dimensions:
                    field = {"OBJECT": "scenario_object_atom", "EGO_DYNAMICS": "ego_dynamics",
                             "EGO_ACTION": "ego_action", "TRAFFIC_PATTERN": "traffic_pattern"}[dimension]
                    facts.pop(field, None)
                    provenance.pop(field, None)
                    if field in projected_facts:
                        facts[field] = projected_facts[field]
                        provenance[field] = {**deepcopy(meta), "source_refs": meta["source_refs"] + projected_provenance[field]["source_refs"]}
            for field in refinement:
                provenance[field] = deepcopy(meta)
            facts["physical_object_exposure_unresolved"] = not bool(refinement)
            facts["physical_object_exposure_conditional"] = bool(refinement)
            info["object_binding"] = {"status": "RESOLVED" if refinement else "PENDING", "reason": reason}
        facts["physical_analysis"] = info
        provenance["physical_analysis"] = deepcopy(meta)
        if operating_condition:
            facts["selected_source_condition"] = deepcopy(operating_condition)
            provenance["selected_source_condition"] = {**deepcopy(meta), "selection_basis": "ITEM_ALLOWED_SCOPE_ANALYTICAL_SELECTION",
                "allowed_scope_sources": deepcopy((project_context or {}).get("sources", []))}
            facts["source_operating_conditions"] = {**facts.get("source_operating_conditions", {}),
                "selected_condition": deepcopy(operating_condition)}
            provenance["source_operating_conditions"] = {**deepcopy(meta),
                "parent_source_metadata": deepcopy(provenance.get("source_operating_conditions", {})),
                "selection_basis": "ITEM_ALLOWED_SCOPE_ANALYTICAL_SELECTION"}
            info["selected_source_condition"] = deepcopy(operating_condition)
        instance = deepcopy(parent.analysis_instance)
        descriptions = [parent.situational_description, parent.situational_detailing]
        if dimensions and refinement:
            instance["dimension_bindings"] = deepcopy(facts["method_scenario_dimensions"])
            instance["selected_atoms"] = list(facts["scenario_atom_ids"])
            for dimension in dimensions:
                old_id = parent.facts.get("method_scenario_dimensions", {}).get(dimension, {}).get("atom_id", "PENDING")
                new_id = facts["method_scenario_dimensions"].get(dimension, {}).get("atom_id", "PENDING")
                descriptions = [text.replace(f"{dimension}={old_id}", f"{dimension}={new_id}") for text in descriptions]
        child = replace(parent, scenario_id=sid, facts=facts, fact_provenance=provenance,
            source_scenario_id=parent.scenario_id, semantic_fingerprint=signature({"parent": parent.semantic_fingerprint, "h": h, "values": values}),
            situational_description=descriptions[0], situational_detailing=descriptions[1],
            analysis_instance={**instance, "physical_hypothesis": info})
        child.status = type(parent.status).PENDING
        self.physics_solve_count += 1
        for record in derive_scenario_physics(child):
            field = record.evidence_ref.removeprefix("DERIVED.")
            facts[field] = record.value
            provenance[field] = {**record.metadata, "provenance": record.provenance.value,
                "approval": record.approval_status.value, "source_refs": [vars(s) for s in record.source_refs]}
        proposal = h.get("intervention_assumptions", {}).get("other_road_user_avoidance_possible", {})
        if proposal.get("value") is True:
            from .scenario_physics import moving_target_avoidance
            check = moving_target_avoidance(facts, proposal)
            info["avoidance_validation"] = check
            if check["status"] != "CONDITIONAL_ESCAPE_VALIDATED":
                facts.pop("other_road_user_avoidance_possible", None)
                provenance.pop("other_road_user_avoidance_possible", None)
        if h["model"] == "RECTILINEAR_MOVING_TARGET_CONTACT":
            from .scenario_physics import rectilinear_moving_contact
            if facts.get("ego_speed_kph") == 0:
                info["stationary_ego_counterfactual"] = rectilinear_moving_contact({**facts,
                    "longitudinal_acceleration_mps2": 0})
        if facts.get("motion_model_outcome", {}).get("status") == "CONTACT" and facts.get("relative_speed_kph") is not None:
            mechanism = f"{malfunction.malfunction_id}:CONTACT"
            facts["harm_mechanism_id"] = mechanism
            facts["collision_branches"] = [{"branch_id": f"BRANCH-{identity}", "mechanism_id": mechanism,
                "impact_time_id": impact_id, "physically_possible": True}]
            provenance["collision_branches"] = deepcopy(meta)
        info["physics"] = {k: facts[k] for k in ("motion_model_outcome", "ego_impact_speed_kph", "relative_speed_kph", "ttc_s", "first_contact_time_s") if k in facts}
        child.analysis_instance["physical_input_production"] = {
            "producer": type(self).__name__, "trigger": "EXISTING_PRODUCER_INSUFFICIENT",
            "pending_reason": "" if facts.get("collision_branches") else facts.get("motion_model_outcome", {}).get("status", "SYMBOLIC_CONDITIONS"),
            "physical_analysis": info}
        return child
