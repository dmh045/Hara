from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from hara_agent.contracts import (
    CalculationStatus, ControllabilityAssessmentInput, ControllabilityBand,
    ControllabilityFactState, ControllabilityJudgement, ControllabilityProfile,
    ExposureAssessment,
    ExposureAtom, ExposureMethod, ExposureMethodDomain,
    ExposureCombinationRule, ExposureCombinationStep,
    ExposureDimensionAssessment, SeverityAssessmentInput, SeverityMethod,
    RuleMatchState, StructuredRiskMethod, UnknownOverridePolicy,
)
from hara_agent.models import ReviewStatus


class FTTIMethodExecutor:
    """Execute the compiled conditional timing policy; no defaults or approvals."""

    @staticmethod
    def evaluate(inputs, scenario, method):
        import math
        from .scenario_physics import braking_avoidability_boundary, moving_braking_avoidability_boundary, source_is_accepted_for
        policy = method.metadata.get("project_analysis_policy", {}).get("timing_analysis", {})
        if not policy:
            return inputs
        analysis = scenario.get("control_analysis") or scenario.get("physical_analysis", {})
        hypothesis = analysis.get("hypothesis", {})
        timing = hypothesis.get("hazard_timing", {})
        proposal = hypothesis.get("safe_state_proposal", {})
        response = proposal.get("transition", {})
        def pending(reason):
            return replace(inputs, reason=reason, status=CalculationStatus.PENDING_INPUT,
                           method_rule_ids=(policy["rule_id"],))
        if not analysis:
            return pending("当前场景尚未形成故障起点、危险判据及安全处置假设；已有接触时间单列。")
        analysis_field = "control_analysis" if scenario.get("control_analysis") else "physical_analysis"
        if not source_is_accepted_for(scenario.get("_fact_provenance", {}).get(analysis_field, {}),
                malfunction_id=inputs.malfunction_id, scenario_id=inputs.scenario_id):
            return pending("时间假设未绑定当前失效与场景，不能继承其他成员的处置时间。")
        moving = scenario.get("fault_motion_model") == "RECTILINEAR_MOVING_TARGET_CONTACT"
        if (scenario.get("fault_motion_model") not in policy["supported_models"]
                or not moving and scenario.get("object_speed_kph") != 0):
            return pending("当前目标/运动不适用已接入的制动避碰响应模型；接触时间单列。")
        controls = set(proposal.get("required_controls", []))
        remaining = set(hypothesis.get("remaining_controls", []))
        driver = "in_driver_seat" if scenario.get("driver_in_vehicle") is True else "outside_driver_seat"
        if (not controls or not controls <= remaining or controls & set(hypothesis.get("failed_controls", []))
                or "driver_braking" in controls and driver != "in_driver_seat"
                or proposal.get("driver_positions") and driver not in proposal["driver_positions"]):
            return pending("本成员没有适用的剩余制动安全状态路径；不能借用其他驾驶员分支。")
        if response.get("model") != policy["response_model"] or not controls & {"braking", "driver_braking"}:
            return pending(hypothesis.get("timing_unresolved_reason") or "安全状态尚未给出兼容的净减速响应模型。")
        if len(inputs.branch_timings) > 1:
            return pending("多个接触分支需分别绑定处置轨迹，不能由父场景时间覆盖。")
        if inputs.branch_timings:
            branch = inputs.branch_timings[0]
            declared = scenario.get("collision_branches", [])
            if (not any(b.get("branch_id") == branch.get("branch_id") for b in declared)
                    or branch.get("impact_time_id") and scenario.get("impact_time_id")
                    and branch["impact_time_id"] != scenario["impact_time_id"]):
                return pending("时间分析的接触分支与当前对象/接触阶段不一致。")
        v = scenario.get("ego_speed_kph")
        a = scenario.get("longitudinal_acceleration_mps2")
        d = scenario.get("travel_to_impact_m")
        outcome = scenario.get("motion_model_outcome", {})
        if scenario.get("fault_motion_model") == "CONSTANT_SPEED_CURVATURE_TO_CONTACT":
            a = 0.0
            d = v / 3.6 * outcome["time_s"] if isinstance(v, (int, float)) and outcome.get("time_s") is not None else None
        b = response.get("net_deceleration_mps2")
        offset = timing.get("fault_to_model_start_s")
        numeric_inputs = (v, a, b) if moving else (v, a, d, b)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in numeric_inputs):
            if moving:
                return pending("移动目标响应仍有符号参数；需在同一包络内关联故障行驶、制动和停车保持三段轨迹。")
            return pending("当前条件仍有符号参数；时间边界关系为 v0·t+a·t²/2+(v0+a·t)²/(2·b净)=可用行程。")
        if moving and response.get("post_stop_motion") != policy.get("moving_target_response", {}).get("post_stop_motion", "HOLD_POSITION"):
            return pending("移动目标需明确制动后保持静止及其控制依据，不能在停车瞬间截断交互。")
        result = (moving_braking_avoidability_boundary(scenario, b) if moving else
                  braking_avoidability_boundary(v/3.6, a, d, b))
        cap = scenario.get("post_fault_speed_cap_kph")
        if (cap is not None and result.get("brake_onset_speed_mps", 0) > cap/3.6):
            return pending("当前轨迹含故障后速度限幅；现有制动边界不能跨限幅阶段外推，接触时间仍保留。")
        if result.get("time_s") is None:
            if result["status"] == "NO_IMMEDIATE_SAFE_BRAKING_WINDOW":
                return replace(pending("同一移动目标轨迹下，即刻制动并停车保持仍不能建立避碰窗口；不能把停车直接当安全。其他处置策略需另行建模。"), timing_trace=result)
            return pending("未形成该危险判据的数值边界：" + result["status"])
        horizon = scenario.get("model_valid_until_s")
        if horizon is not None and result["time_s"] > horizon:
            return pending("停车避碰边界超出当前运动模型有效时域，不能外推为FTTI。")
        # A response can already be physically impossible even when its fault
        # clock has not been specified. Preserve that evidence for Safe State;
        # do not issue an FTTI number until the temporal meaning is established.
        if (timing.get("hazard_criterion") != policy["hazard_criterion"]
                or isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset)):
            return replace(pending("响应轨迹已检查；尚未绑定故障起点及失去避碰能力判据，不能作为FTTI。"), timing_trace=result)
        return replace(inputs, hazard_time_limit_s=offset+result["time_s"],
            safe_state_transition_time_s=result["stopping_time_s"],
            method_rule_ids=(policy["rule_id"],), status=CalculationStatus.CONDITIONAL,
            calculation_nature="CONDITIONAL", hazard_criterion=policy["hazard_criterion"], time_origin="FAULT_ONSET",
            reason=("初始状态已失去停车避碰能力：即刻制动也没有正的处置余量。" if result["status"] == "AVOIDABILITY_ALREADY_LOST" else
                "条件性危险时限：在所建模交互时段内失去制动避碰能力。检测、决策和制动建立延迟必须小于可用制动起始窗口；制动完成时间不再加到该时限。"),
            timing_trace={**result, "fault_to_model_start_s": offset, "net_deceleration_mps2": b,
                "available_path_distance_m": d, "required_controls": sorted(controls),
                "safe_state": proposal["description"], "conditions": proposal["conditions"],
                "source_ref": policy["source_ref"], "hypothesis": hypothesis,
                "engineering_approval": "NOT_GRANTED", "continuous_range_proven": False})


class ExposureCombinationExecutor:
    """Execute an explicit, ordered, template-sourced E combination plan."""

    @staticmethod
    def _level(item: ExposureDimensionAssessment) -> str:
        if item.selected_domain is ExposureMethodDomain.TIME:
            return item.duration_level
        if item.selected_domain is ExposureMethodDomain.FREQUENCY:
            return item.frequency_level
        return ""

    @staticmethod
    def combine_policy(
        levels: Sequence[str], *, dependent: bool, policy,
    ) -> str:
        """Apply the compiled MethodContract aggregation policy.

        The policy is supplied by the method source; this helper deliberately
        contains no FUSA levels or matrix values of its own.
        """
        values = [int(value[1:]) for value in levels]
        if all(value == int(policy.all_highest_operand[1:]) for value in values):
            return policy.all_highest_result
        if policy.matches_mixed_high(levels):
            return policy.mixed_high_result
        if min(values) != max(values):
            if policy.mixed_strategy != "MINIMUM":
                raise ValueError(f"Unsupported Exposure mixed strategy: {policy.mixed_strategy}")
            return f"E{min(values)}"
        decrement = policy.dependent_decrement if dependent else policy.independent_decrement
        return f"E{max(policy.minimum_level, values[0] - decrement)}"

    def combine(
        self,
        scenario_id: str,
        dimensions: Sequence[ExposureDimensionAssessment],
        steps: Sequence[ExposureCombinationStep],
        rules: Sequence[ExposureCombinationRule],
    ) -> ExposureAssessment:
        levels = {item.dimension: self._level(item) for item in dimensions}
        if len(levels) != len(dimensions):
            return ExposureAssessment(
                scenario_id=scenario_id, dimensions=tuple(dimensions),
                status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                reason="Exposure dimension identities must be unique.",
            )
        if not levels or any(not value for value in levels.values()):
            return ExposureAssessment(
                scenario_id=scenario_id, dimensions=tuple(dimensions),
                status=CalculationStatus.PENDING_INPUT,
                reason="One or more dimension-level E values or Z/F selections are missing.",
            )
        if len(levels) > 1 and not steps:
            return ExposureAssessment(
                scenario_id=scenario_id, dimensions=tuple(dimensions),
                status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                reason="Multiple Exposure dimensions require an explicit ordered plan.",
            )
        applied: list[str] = []
        for step in steps:
            left = levels.get(step.left_dimension, "")
            right = levels.get(step.right_dimension, "")
            if not left or not right:
                return ExposureAssessment(
                    scenario_id=scenario_id, dimensions=tuple(dimensions),
                    combination_rule_ids=tuple(applied),
                    status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                    reason="The ordered Exposure combination plan references an unavailable operand.",
                )
            matches = [
                rule for rule in rules
                if rule.review_status is ReviewStatus.FINALIZED
                and rule.relation is step.relation
                and {rule.left_level, rule.right_level} == {left, right}
                and (not rule.left_dimension or rule.left_dimension == step.left_dimension)
                and (not rule.right_dimension or rule.right_dimension == step.right_dimension)
            ]
            if len(matches) != 1:
                return ExposureAssessment(
                    scenario_id=scenario_id, dimensions=tuple(dimensions),
                    combination_rule_ids=tuple(applied),
                    status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                    reason=(
                        "The active MethodContract does not provide exactly one approved "
                        "Exposure rule for an ordered combination step."
                    ),
                )
            rule = matches[0]
            levels[step.output_dimension] = rule.result_level
            applied.append(rule.rule_id)
        final_dimension = steps[-1].output_dimension if steps else next(iter(levels))
        value = levels.get(final_dimension, "")
        return ExposureAssessment(
            scenario_id=scenario_id, dimensions=tuple(dimensions),
            combination_rule_ids=tuple(applied), value=value,
            status=CalculationStatus.FINALIZED,
            reason="Executed the approved ordered Exposure combination plan.",
        )


class ControllabilityProfileExecutor:
    """Perform deterministic TTC lookup against one explicitly selected profile."""

    @staticmethod
    def unresolved_applicability(values: dict, profile: ControllabilityProfile) -> tuple[str, ...]:
        return tuple(condition.field for condition in profile.applicability
                     if values.get(condition.field) is not condition.expected)

    @staticmethod
    def _matches(band: ControllabilityBand, ttc_s: float) -> bool:
        if band.lower_ttc_s is not None:
            if ttc_s < band.lower_ttc_s:
                return False
            if ttc_s == band.lower_ttc_s and not band.lower_inclusive:
                return False
        if band.upper_ttc_s is not None:
            if ttc_s > band.upper_ttc_s:
                return False
            if ttc_s == band.upper_ttc_s and not band.upper_inclusive:
                return False
        return True

    def lookup(
        self,
        assessment: ControllabilityAssessmentInput,
        profile: ControllabilityProfile | None,
        *, time_input: str = "ttc_s",
    ) -> ControllabilityJudgement:
        if time_input not in {"ttc_s", "first_contact_time_s"}:
            raise ValueError("Unsupported controllability time input")
        time_value = getattr(assessment, time_input)
        if profile is None or profile.review_status is not ReviewStatus.FINALIZED:
            return ControllabilityJudgement(
                value="", profile_id=profile.profile_id if profile else "",
                rule_id="", inputs_used=assessment.inputs_used,
                status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                reason="No approved Controllability profile was explicitly selected.",
            )
        unresolved = self.unresolved_applicability(assessment.to_dict(), profile)
        if unresolved:
            return ControllabilityJudgement(
                value="", profile_id=profile.profile_id, rule_id="",
                inputs_used=tuple(condition.field for condition in profile.applicability),
                status=CalculationStatus.PENDING_INPUT,
                decision_status="PROFILE_APPLICABILITY_UNRESOLVED",
                reason="TTC_PROFILE_APPLICABILITY_UNRESOLVED: " + ", ".join(unresolved),
            )
        if time_value is None:
            return ControllabilityJudgement(
                value="", profile_id=profile.profile_id, rule_id="",
                inputs_used=assessment.inputs_used,
                status=CalculationStatus.PENDING_INPUT,
                reason=f"The selected profile requires a source-grounded {time_input} input.",
            )
        matches = [
            band for band in profile.bands
            if band.review_status is ReviewStatus.FINALIZED
            and self._matches(band, time_value)
        ]
        if len(matches) != 1:
            return ControllabilityJudgement(
                value="", profile_id=profile.profile_id, rule_id="",
                inputs_used=(time_input,),
                status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                reason="The selected profile has a TTC gap or overlap at the supplied input.",
            )
        band = matches[0]
        return ControllabilityJudgement(
            value=band.result, profile_id=profile.profile_id,
            rule_id=band.rule_id, inputs_used=(time_input, *(c.field for c in profile.applicability)),
            status=CalculationStatus.FINALIZED,
            reason=f"Deterministic {time_input} lookup matched exactly one approved profile band.",
        )


class SeverityMethodExecutor:
    """Evaluate a structured severity table without textual rule inference."""

    @staticmethod
    def _matches_band(band, value: float) -> bool:
        if band.lower_kph is not None and (
            value < band.lower_kph
            or (value == band.lower_kph and not band.lower_inclusive)
        ):
            return False
        if band.upper_kph is not None and (
            value > band.upper_kph
            or (value == band.upper_kph and not band.upper_inclusive)
        ):
            return False
        return True

    def lookup(self, assessment: SeverityAssessmentInput, method: SeverityMethod) -> dict:
        if (
            method.semantic.semantic_resolution.value
            == "APPROVED_SOURCE_INTERNAL_CONFLICT"
        ):
            return {"value": "", "status": CalculationStatus.PENDING_METHOD_SEMANTICS,
                    "reason": (
                        "Confirmed Severity sources contain an unresolved internal semantic conflict."
                    ),
                    "rule_id": "", "source_ref": method.source_ref, "inputs_used": ()}
        consequence = assessment.consequence
        value_by_semantic = {
            "EGO_SPEED": consequence.ego_speed_kph,
            "RELATIVE_SPEED": consequence.relative_speed_kph,
            "IMPACT_SPEED": consequence.impact_speed_kph,
            "DELTA_V": consequence.delta_v_kph,
        }
        semantic = method.speed_semantic.value
        input_key = {
            "EGO_SPEED": "ego_speed_kph",
            "RELATIVE_SPEED": "relative_speed_kph",
            "IMPACT_SPEED": "impact_speed_kph",
            "DELTA_V": "delta_v_kph",
        }.get(semantic, "")
        value = value_by_semantic.get(semantic)
        if value is None:
            return {"value": "", "status": CalculationStatus.PENDING_INPUT,
                    "reason": f"Severity requires the configured {semantic} input.",
                    "rule_id": "", "source_ref": method.source_ref, "inputs_used": ()}
        group = dict(method.road_user_groups).get(
            consequence.road_user_type.strip().upper(), ""
        )
        if not group:
            return {"value": "", "status": CalculationStatus.PENDING_INPUT,
                    "reason": "Severity requires a canonical road-user type.",
                    "rule_id": "", "source_ref": method.source_ref,
                    "inputs_used": (input_key,)}
        collision_type = dict(method.collision_types).get(
            consequence.collision_type.strip().upper(), ""
        )
        # VRU bands in the active Method are collision-configuration independent.
        # Keep vehicle bands tied to their configured collision type.
        if group != "vehicle" and any(
            band.collision_group == group and band.collision_type == "any"
            for band in method.bands
        ):
            collision_type = "any"
        if not collision_type:
            return {"value": "", "status": CalculationStatus.PENDING_INPUT,
                    "reason": "Vehicle severity lookup requires a canonical collision type.",
                    "rule_id": "", "source_ref": method.source_ref,
                    "inputs_used": (input_key, "road_user_type")}
        matches = [
            band for band in method.bands
            if band.collision_group == group
            and band.collision_type == collision_type
            and self._matches_band(band, value)
        ]
        if len(matches) != 1:
            return {"value": "", "status": CalculationStatus.PENDING_METHOD_SEMANTICS,
                    "reason": "Severity table has a range gap or overlap at the supplied input.",
                    "rule_id": "", "source_ref": method.source_ref,
                    "inputs_used": (input_key, "road_user_type", "collision_type")}
        band = matches[0]
        return {"value": band.result, "status": CalculationStatus.FINALIZED,
                "reason": "Deterministic structured severity lookup matched one approved band.",
                "rule_id": band.rule_id, "source_ref": band.source_ref,
                "inputs_used": (input_key, "road_user_type", "collision_type")}


class ExposureMethodExecutor:
    _VALID_LEVELS = frozenset({"E0", "E1", "E2", "E3", "E4"})

    @staticmethod
    def _level(atom: ExposureAtom, domain: ExposureMethodDomain) -> str:
        return atom.duration_level if domain is ExposureMethodDomain.TIME else atom.frequency_level

    @staticmethod
    def _other_domain(domain: ExposureMethodDomain) -> ExposureMethodDomain:
        return (
            ExposureMethodDomain.FREQUENCY
            if domain is ExposureMethodDomain.TIME else ExposureMethodDomain.TIME
        )

    @classmethod
    def _atom_values(
        cls,
        atom_ids: tuple[str, ...],
        by_id: dict[str, ExposureAtom],
        domain: ExposureMethodDomain,
    ) -> tuple[list[dict], list[str]]:
        """Mirror FUSA v1 collection: one domain for the entire Scenario."""
        details: list[dict] = []
        values: list[str] = []
        for atom_id in atom_ids:
            atom = by_id.get(atom_id)
            if atom is None:
                details.append({
                    "atom_id": atom_id,
                    "requested_domain": domain.value,
                    "actual_domain": "",
                    "e_rank": "",
                    "used": False,
                    "skipped": True,
                    "skip_reason": "ATOM_NOT_IN_CATALOG",
                })
                continue
            level = cls._level(atom, domain)
            usable = level in cls._VALID_LEVELS
            raw = atom.raw_duration_value if domain is ExposureMethodDomain.TIME else atom.raw_frequency_value
            details.append({
                "atom_id": atom_id,
                "requested_domain": domain.value,
                "actual_domain": domain.value if usable else "",
                "e_rank": level if usable else "",
                "used": usable,
                "skipped": not usable,
                "skip_reason": "" if usable else "NO_VALUE_IN_SELECTED_DOMAIN",
                "dimensions": list(atom.dimensions),
                "raw_value": raw,
                "value_state": "KNOWN" if usable else "UNKNOWN_TBD" if raw.casefold() == "tbd"
                               else "SOURCE_DASH" if raw.casefold() in {"-", "dash", "—"}
                               else "SOURCE_NULL" if raw.casefold() in {"none", "null"} else "MISSING",
                "source_ref": f"{atom.source_ref.sheet}!{atom.source_ref.range}",
                "participation_rule": "FUSA_V1_VALID_LEVELS_ONLY",
            })
            if usable:
                values.append(level)
        return details, values

    @staticmethod
    def _base_result(
        *, rule, domain: ExposureMethodDomain, requested: ExposureMethodDomain,
        atom_details: list[dict], **values,
    ) -> dict:
        return {
            "rule_id": rule.rule_id,
            "source_ref": rule.source_ref,
            "inputs_used": ("component_category", "scenario_atom_ids"),
            "domain": domain.value,
            "requested_domain": requested.value,
            "actual_domain": domain.value,
            "dimension_fallback": domain is not requested,
            "atom_details": atom_details,
            "aggregation_rule": "",
            "coupling_consumed": False,
            "coupling": "",
            **values,
        }

    def preview(self, scenario: dict, method: ExposureMethod) -> dict:
        """Evaluate with the production FUSA implementation for input analysis.

        Callers that need to establish whether a prospective, source-defined
        atom could change a result must use this entry point rather than copy
        any aggregation branch.  ``lookup`` remains the scoring entry point.
        """
        return self._evaluate(scenario, method)

    def lookup(self, scenario: dict, method: ExposureMethod) -> dict:
        return self._evaluate(scenario, method)

    def _evaluate(self, scenario: dict, method: ExposureMethod) -> dict:
        category = str(scenario.get("component_category", "")).strip()
        matches = [rule for rule in method.domain_rules if category in rule.component_categories]
        if len(matches) != 1:
            return {"value": "", "status": CalculationStatus.PENDING_METHOD_SEMANTICS,
                    "reason": "Component category does not resolve to exactly one approved Z/F rule.",
                    "rule_id": "", "source_ref": method.source_refs[0],
                    "inputs_used": ("component_category",), "domain": "",
                    "requested_domain": "", "actual_domain": "",
                    "dimension_fallback": False, "atom_details": [],
                    "aggregation_rule": "", "coupling_consumed": False,
                    "coupling": "", "pending_reason": "MISSING_E_DIMENSION"}
        domain = matches[0].domain
        raw_ids = scenario.get("scenario_atom_ids", ())
        atom_ids = tuple(dict.fromkeys(str(value) for value in raw_ids)) if isinstance(raw_ids, (list, tuple)) else ()
        by_id = {atom.atom_id: atom for atom in method.atoms}
        if not atom_ids:
            return self._base_result(
                rule=matches[0], domain=domain, requested=domain, atom_details=[],
                value="", status=CalculationStatus.PENDING_INPUT,
                reason="MISSING_SCENARIO_ATOMS", pending_reason="MISSING_SCENARIO_ATOMS",
            )

        requested = domain
        atom_details, usable = self._atom_values(atom_ids, by_id, requested)
        selected = requested
        if not usable:
            selected = self._other_domain(requested)
            atom_details, usable = self._atom_values(atom_ids, by_id, selected)
        if not usable:
            reason = (
                "MISSING_CATALOG_ATOMS" if not any(atom_id in by_id for atom_id in atom_ids)
                else "MISSING_EXPOSURE_VALUE_BOTH_DOMAINS"
            )
            return self._base_result(
                rule=matches[0], domain=requested, requested=requested,
                atom_details=atom_details, value="", status=CalculationStatus.PENDING_INPUT,
                reason=reason, pending_reason=reason,
            )

        common = {
            "rule": matches[0], "domain": selected, "requested": requested,
            "atom_details": atom_details,
        }
        policy = method.aggregation_policy
        if all(level == policy.all_highest_operand for level in usable):
            return self._base_result(
                **common, value=policy.all_highest_result,
                status=CalculationStatus.FINALIZED,
                reason="FUSA_V1_ALL_E4", aggregation_rule="all_e4",
            )
        if len(usable) == 1:
            # A single usable atom has no inter-atom dependency to resolve.
            return self._base_result(
                **common, value=usable[0], status=CalculationStatus.FINALIZED,
                reason="FUSA_V1_SINGLE_ATOM", aggregation_rule="single_atom",
            )
        if policy.matches_mixed_high(usable):
            return self._base_result(
                **common, value=policy.mixed_high_result,
                status=CalculationStatus.FINALIZED,
                reason="FUSA_V1_E3_E4_MIX", aggregation_rule="e3_e4_mix",
            )
        ranks = [int(level[1:]) for level in usable]
        if min(ranks) != max(ranks):
            return self._base_result(
                **common, value=f"E{min(ranks)}",
                status=CalculationStatus.FINALIZED,
                reason="FUSA_V1_MIN_WHEN_UNEQUAL",
                aggregation_rule="min_when_unequal",
            )

        coupling = str(scenario.get("atoms_coupling", "")).strip().lower()
        if coupling not in {"coupled", "independent"}:
            return self._base_result(
                **common, value="", status=CalculationStatus.PENDING_INPUT,
                reason="MISSING_ATOMS_COUPLING",
                pending_reason="MISSING_ATOMS_COUPLING",
            )
        decrement = (
            policy.dependent_decrement if coupling == "coupled"
            else policy.independent_decrement
        )
        return self._base_result(
            **common,
            value=f"E{max(policy.minimum_level, ranks[0] - decrement)}",
            status=CalculationStatus.FINALIZED,
            reason=("FUSA_V1_SAME_COUPLED_NO_CHANGE" if coupling == "coupled"
                    else "FUSA_V1_SAME_INDEPENDENT_MINUS_ONE"),
            aggregation_rule=("same_coupled_no_change" if coupling == "coupled"
                              else "same_independent_minus_one"),
            coupling_consumed=True, coupling=coupling,
        )


class StructuredControllabilityExecutor:
    def __init__(self):
        self.profile_executor = ControllabilityProfileExecutor()

    @staticmethod
    def _fact_state(value: object) -> ControllabilityFactState:
        if isinstance(value, ControllabilityFactState):
            return value
        if isinstance(value, bool):
            return ControllabilityFactState.TRUE if value else ControllabilityFactState.FALSE
        return ControllabilityFactState.UNKNOWN

    @classmethod
    def _condition_state(cls, actual: object, expected: bool) -> ControllabilityFactState:
        state = cls._fact_state(actual)
        if state in {ControllabilityFactState.UNKNOWN, ControllabilityFactState.CONFLICT}:
            return state
        value = state is ControllabilityFactState.TRUE
        return ControllabilityFactState.TRUE if value is expected else ControllabilityFactState.FALSE

    @classmethod
    def override_state(cls, rule, scenario: dict) -> RuleMatchState:
        conditions = rule.all_of or rule.any_of
        values = [cls._condition_state(scenario.get(item.field), item.expected) for item in conditions]
        if ControllabilityFactState.CONFLICT in values:
            return RuleMatchState.CONFLICT
        if rule.all_of:
            if ControllabilityFactState.FALSE in values:
                return RuleMatchState.NO_MATCH
            if ControllabilityFactState.UNKNOWN in values:
                return RuleMatchState.UNKNOWN
            return RuleMatchState.MATCH
        if ControllabilityFactState.TRUE in values:
            return RuleMatchState.MATCH
        if ControllabilityFactState.UNKNOWN in values:
            return RuleMatchState.UNKNOWN
        return RuleMatchState.NO_MATCH

    @staticmethod
    def _fields(rule) -> tuple[str, ...]:
        return tuple(item.field for item in (*rule.all_of, *rule.any_of))

    @staticmethod
    def _with_branch_metadata(
        judgement: ControllabilityJudgement, *, policy: UnknownOverridePolicy,
        action: str, decision_status: str,
        match_states: tuple[tuple[str, RuleMatchState], ...],
    ) -> ControllabilityJudgement:
        return replace(
            judgement, decision_status=judgement.decision_status or decision_status,
            unknown_override_policy=policy, unknown_policy_action=action,
            rule_match_states=match_states,
        )

    def lookup(self, assessment: ControllabilityAssessmentInput, method: StructuredRiskMethod) -> ControllabilityJudgement:
        scenario = assessment.to_dict()
        policy = method.controllability_branch_policy.unknown_override_policy
        states: list[tuple[str, RuleMatchState]] = []
        unknown_fields: list[str] = []
        for rule in sorted(method.controllability_overrides, key=lambda item: item.priority):
            state = self.override_state(rule, scenario)
            states.append((rule.rule_id, state))
            fields = self._fields(rule)
            if state is RuleMatchState.MATCH:
                return ControllabilityJudgement(
                    value=rule.result, profile_id=method.controllability_profile.profile_id,
                    rule_id=rule.rule_id, inputs_used=fields,
                    status=CalculationStatus.FINALIZED,
                    reason="A higher-priority approved controllability override matched.",
                    decision_status="OVERRIDE_MATCH", unknown_override_policy=policy,
                    unknown_policy_action="NOT_APPLICABLE",
                    rule_match_states=tuple(states),
                )
            if state is RuleMatchState.CONFLICT:
                return ControllabilityJudgement(
                    value="", profile_id=method.controllability_profile.profile_id,
                    rule_id=rule.rule_id,
                    inputs_used=fields,
                    status=CalculationStatus.PENDING_INPUT,
                    reason="Controllability override facts contain a source conflict.",
                    decision_status="FACT_SOURCE_CONFLICT", unknown_override_policy=policy,
                    unknown_policy_action="CONFLICT_BLOCKED",
                    rule_match_states=tuple(states),
                )
            if state is RuleMatchState.UNKNOWN:
                unknown_fields.extend(fields)
        match_states = tuple(states)
        if unknown_fields:
            fields = tuple(dict.fromkeys(unknown_fields))
            if policy is UnknownOverridePolicy.UNSPECIFIED:
                return ControllabilityJudgement(
                    value="", profile_id=method.controllability_profile.profile_id,
                    rule_id="", inputs_used=fields,
                    status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                    reason="CONTROLLABILITY_UNKNOWN_BRANCH_POLICY_UNSPECIFIED",
                    decision_status="METHOD_BRANCH_UNRESOLVED", unknown_override_policy=policy,
                    unknown_policy_action="UNSPECIFIED_BLOCKED", rule_match_states=match_states,
                )
            if policy is UnknownOverridePolicy.BLOCK_TTC:
                return ControllabilityJudgement(
                    value="", profile_id=method.controllability_profile.profile_id,
                    rule_id="", inputs_used=fields,
                    status=CalculationStatus.PENDING_INPUT,
                    reason="Controllability override inputs are required by the selected BLOCK_TTC branch policy.",
                    decision_status="OVERRIDE_INPUT_PENDING", unknown_override_policy=policy,
                    unknown_policy_action="BLOCK_TTC", rule_match_states=match_states,
                )
            if policy is not UnknownOverridePolicy.SKIP_TO_TTC:
                raise ValueError(f"Unsupported unknown override policy: {policy}")
            return self._with_branch_metadata(
                self.profile_executor.lookup(assessment, method.controllability_profile),
                policy=policy, action="SKIP_TO_TTC",
                decision_status="TTC_AFTER_UNKNOWN_OVERRIDE", match_states=match_states,
            )
        return self._with_branch_metadata(
            self.profile_executor.lookup(assessment, method.controllability_profile),
            policy=policy, action="ALL_OVERRIDES_NO_MATCH",
            decision_status="TTC_AFTER_NO_MATCH", match_states=match_states,
        )
