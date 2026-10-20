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
    scenarios = {item.scenario_id: item for item in state.scenarios}

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
        )
        sg_id = registration["sg_id"]
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
        # Adding unresolved members must not replace the recorded original
        # description with whichever new member happens to be encountered first.
        original = prior_catalog.get(sg_id, {}).get("original_safety_goal")
        if original:
            value["original_safety_goal"] = original
        statuses = risk_status_by_goal[sg_id]
        finalized = catalog.is_approved and all(
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
