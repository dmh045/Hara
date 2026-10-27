from __future__ import annotations

from hara_agent.models import ReviewStatus, SafetyGoal
from hara_agent.services.analysis import SafetyGoalService
from hara_agent.workflow.state import HARAState, WorkflowStage


def aggregate_safety_goals(
    state: HARAState,
    catalog: SafetyGoalService,
) -> HARAState:
    """Keep non-QM and unresolved safety intents visible without approving them."""
    malfunctions = {
        str(item.get("malfunction_id", "")): item for item in state.malfunctions
    }
    functions = {
        str(item.get("function_id", "")): item for item in state.functions
    }
    scenario_ids_by_goal: dict[str, list[str]] = {}
    risk_status_by_goal: dict[str, list[ReviewStatus]] = {}
    safe_state_gaps: dict[str, list[dict]] = {}
    scenarios = {item.scenario_id: item for item in state.scenarios}
    previous_goals = {goal.sg_id: goal for goal in state.safety_goals}

    for risk in state.risk_results:
        if risk.asil.value == "QM":
            continue
        malfunction = malfunctions.get(risk.malfunction_id)
        if malfunction is None:
            raise ValueError(f"Safety Goal聚合缺少Malfunction外键: {risk.malfunction_id!r}")
        function = functions.get(str(malfunction.get("function_id", "")))
        if function is None:
            raise ValueError(f"Safety Goal聚合缺少Function外键: {malfunction.get('function_id')!r}")
        function_name = str(function.get("name", ""))
        scenario = scenarios.get(risk.scenario_id)
        analysis = (scenario.facts.get("control_analysis") or scenario.facts.get("physical_analysis", {})) if scenario else {}
        proposal = analysis.get("hypothesis", {}).get("safe_state_proposal")
        gap = analysis.get("hypothesis", {}).get("safe_state_unresolved_reason") or (
            "本次方案未给出可达的安全状态及其控制路径" if analysis else "当前成员尚无安全状态分析方案")
        gap_code = "NO_PROPOSAL" if analysis else "NO_ANALYSIS"
        if proposal:
            # A driver-only proposal does not apply to an outside-driver child.
            # The proposal remains in the saved plan for audit, not a new fact.
            if ("driver_braking" in proposal["required_controls"]
                    and scenario.facts.get("driver_in_vehicle") is not True):
                proposal = None
                gap = "已有方案依赖驾驶员操作制动踏板，但本成员驾驶员在车外"
                gap_code = "DRIVER_CONTROL_UNAVAILABLE"
            elif proposal.get("driver_positions") and (
                    "in_driver_seat" if scenario.facts.get("driver_in_vehicle") else "outside_driver_seat") not in proposal["driver_positions"]:
                proposal = None
                gap = "已有安全状态方案不适用于本成员驾驶员位置"
                gap_code = "DRIVER_POSITION_OUT_OF_SCOPE"
            else:
                proposal = {**proposal, "plan_key": analysis["plan_key"],
                            "hypothesis_id": analysis.get("hypothesis_id", "control_assessment")}
        if proposal and proposal.get("transition") and getattr(catalog, "method", None):
            from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
            timing = RiskCalculationInputService().ftti(risk.malfunction_id, risk.scenario_id, risk.asil.value,
                {**scenario.facts, "_fact_provenance": scenario.fact_provenance}, risk.branch_evaluations,
                method=catalog.method)
            if timing.timing_trace.get("status") == "AVOIDABILITY_ALREADY_LOST":
                proposal = None
                gap = "按当前净减速假设，即刻制动也不能在障碍物前停止；原提案保留在物理方案中"
                gap_code = "RESPONSE_INSUFFICIENT"
        registration = catalog.register_intent(
            function_name=function_name,
            malfunction=str(malfunction.get("description", "")),
            guideword=str(malfunction.get("guideword", "")),
            scenario_id=risk.scenario_id,
            scenario_description=(
                scenarios[risk.scenario_id].situational_description
                if risk.scenario_id in scenarios else ""
            ),
            hazard_event=risk.hazardous_event,
            asil=str(risk.asil.value or ""),
            safe_state_proposal=proposal,
        )
        sg_id = registration["sg_id"]
        if not proposal:
            safe_state_gaps.setdefault(sg_id, []).append({"scenario_id": risk.scenario_id, "reason": gap, "reason_code": gap_code})
        risk.safety_goal_id = sg_id
        scenario_ids_by_goal.setdefault(sg_id, []).append(risk.scenario_id)
        risk_status_by_goal.setdefault(sg_id, []).extend([
            risk.severity.status,
            risk.exposure.status,
            risk.controllability.status,
            risk.asil.status,
        ])

    goals = []
    prior_catalog = state.item_definition.get("safety_goal_catalog", {})
    current_catalog = catalog.to_dict()
    for sg_id, value in current_catalog.items():
        value["safe_state_gaps"] = safe_state_gaps.get(sg_id, [])
        # Adding unresolved members must not replace the recorded original
        # description with whichever new member happens to be encountered first.
        original = prior_catalog.get(sg_id, {}).get("original_safety_goal")
        if original:
            value["original_safety_goal"] = original
        previous = previous_goals.get(sg_id)
        if previous and previous.safe_state not in {"", "待确定", "待评定", "Pending"}:
            # Re-aggregation must not erase an existing engineering proposal.
            # Preserve its recorded scope instead of applying it to new members.
            value["prior_safe_state"] = {"description": previous.safe_state,
                "scenario_ids": previous.associated_scenario_ids, "status": previous.status.value}
            if not value.get("safe_state_candidates"):
                value["safety_state"] = previous.safe_state
        statuses = risk_status_by_goal[sg_id]
        finalized = catalog.is_approved and not value.get("safe_state_candidates") and all(
            status in {ReviewStatus.FINALIZED, ReviewStatus.NOT_APPLICABLE}
            for status in statuses
        )
        status = ReviewStatus.FINALIZED if finalized else ReviewStatus.PENDING
        goals.append(SafetyGoal(
            sg_id=sg_id,
            text=str(value["safety_goal"]),
            safe_state=str(value["safety_state"]),
            max_asil=str(value["max_asil"]),
            associated_scenario_ids=list(dict.fromkeys(scenario_ids_by_goal[sg_id])),
            status=status,
        ))
        if status is ReviewStatus.PENDING:
            state.pending_reviews.append({
                "safety_goal_id": sg_id,
                "field": "safety_goal",
                "reason": (
                    "Safety Goal/Safe State semantic derivation or associated risk "
                    "evidence is not fully approved."
                ),
            })

    state.item_definition["safety_goal_catalog"] = current_catalog
    state.safety_goals = goals
    state.stage = WorkflowStage.QUALITY_GATE
    state.record(
        "safety_goals_aggregated",
        non_qm_risk_count=sum(
            risk.asil.value in {"A", "B", "C", "D"}
            for risk in state.risk_results
        ),
        unresolved_risk_count=sum(not risk.asil.value for risk in state.risk_results),
        safety_goal_count=len(goals),
    )
    return state
