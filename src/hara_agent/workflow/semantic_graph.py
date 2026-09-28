from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hara_agent.contracts import Guideword, MethodContract, RequiredFactSpec
from hara_agent.infrastructure.llm import LLMClient
from hara_agent.models import (
    FunctionDefinition,
    GuidewordAssessment,
    GuidewordDisposition,
    MalfunctionCandidate,
    ReviewStatus,
    ScenarioCandidate,
    SourceRef,
)
from hara_agent.services.semantic import (
    ItemArtifactExtractionAgent,
    ItemEvidenceRouter,
    ItemSupplementAgent,
    TargetedProjectFactExtractionAgent,
    GuidewordApplicabilityAgent,
    MalfunctionHazardAgent,
    ScenarioFeasibilityAgent,
    ScenarioRiskFactAgent,
)
from hara_agent.services.analysis import (
    ASILLookupService,
    FailureModeSelectorResolver,
    MethodRiskFactBindingService,
    SafetyGoalService,
    ScenarioScoringService,
)
from hara_agent.services.reporting import HARAExcelRenderer
from hara_agent.services.validation import DownstreamPreflightService
from hara_agent.services.extraction import ValidatedArtifactCache

from .checkpoints import CheckpointRepository
from .graph import WorkflowGraph
from .nodes import (
    assess_guidewords,
    assess_scenarios,
    aggregate_safety_goals,
    derive_malfunctions,
    extract_item_artifacts,
    read_item_document,
    render_excel_report,
    pass_quality_gate,
    score_structured_scenarios,
)
from .state import HARAState, WorkflowStage
from .review_artifacts import ReviewArtifactWriter


@dataclass(frozen=True)
class SemanticWorkflowInputs:
    item_path: str | Path
    guidewords: list[str | Guideword]
    scenario_candidates: list[ScenarioCandidate] = field(default_factory=list)
    scenario_candidate_factory: Callable[[HARAState], tuple[list[ScenarioCandidate], dict]] | None = None
    project_context_preflight: Callable[[HARAState], dict] | None = None
    max_workers: int = 4
    progress: Callable[[str, int, int], None] | None = None
    stage_progress: Callable[[str, str, float], None] | None = None
    artifact_cache: ValidatedArtifactCache | None = None
    required_fact_specs: tuple[RequiredFactSpec, ...] = ()
    requested_operating_modes: tuple[str, ...] = ()
    method_contract: MethodContract | None = None
    review_artifact_writer: ReviewArtifactWriter | None = None
    sample_function_limit: int | None = None
    sample_malfunction_limit: int | None = None
    sample_function_ids: tuple[str, ...] = ()
    sample_malfunction_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticWorkflowAgents:
    client: LLMClient
    guidewords: GuidewordApplicabilityAgent
    malfunctions: MalfunctionHazardAgent
    scenarios: ScenarioFeasibilityAgent
    risk_facts: ScenarioRiskFactAgent | None = None


@dataclass(frozen=True)
class RiskWorkflowServices:
    scoring: ScenarioScoringService
    asil_table: ASILLookupService
    safety_goals: SafetyGoalService
    risk_fact_binding: MethodRiskFactBindingService | None = None


@dataclass(frozen=True)
class ReportingWorkflowConfig:
    template_path: str | Path
    output_path: str | Path
    renderer: HARAExcelRenderer


def _sources(values: list[dict]) -> list[SourceRef]:
    return [SourceRef(**value) for value in values]


def _function(value: dict) -> FunctionDefinition:
    payload = dict(value)
    payload["sources"] = _sources(payload.get("sources", []))
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return FunctionDefinition(**payload)


def _assessment(value: dict) -> GuidewordAssessment:
    payload = dict(value)
    payload["sources"] = _sources(payload.get("sources", []))
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    raw_disposition = payload.get("disposition")
    payload["disposition"] = (
        GuidewordDisposition(raw_disposition) if raw_disposition else None
    )
    return GuidewordAssessment(**payload)


def _guideword_assessment_values(state: HARAState) -> list[dict]:
    if state.guideword_assessments:
        return state.guideword_assessments
    # Backward compatibility for checkpoints written before the dedicated
    # matrix field existed.  New runs never overload malfunctions this way.
    return [
        value["guideword_assessment"]
        for value in state.malfunctions
        if isinstance(value, dict)
        and isinstance(value.get("guideword_assessment"), dict)
    ]


def _malfunction(value: dict) -> MalfunctionCandidate:
    payload = dict(value)
    payload["sources"] = _sources(payload.get("sources", []))
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return MalfunctionCandidate(**payload)


def _scenario_candidates(state: HARAState, inputs: SemanticWorkflowInputs) -> list[ScenarioCandidate]:
    if inputs.scenario_candidate_factory is None:
        candidates, audit = inputs.scenario_candidates, {}
    else:
        candidates, audit = inputs.scenario_candidate_factory(state)
    if inputs.review_artifact_writer is not None:
        inputs.review_artifact_writer.write_scenario_binding_gaps(
            audit.get("scenario_binding_coverage", {}),
            audit.get("scenario_binding_gaps", []),
        )
        for candidate in candidates:
            inputs.review_artifact_writer.record_scenario_candidate(
                candidate,
                generated_for_malfunction_ids=[
                    str(item.get("malfunction_id", ""))
                    for item in state.malfunctions
                    if item.get("malfunction_id")
                ],
            )
        # This factory is evaluated inside the long-running Scenario node.
        # Refresh now instead of waiting for every feasibility worker.
        inputs.review_artifact_writer.write_summary(state)
    if audit:
        state.record("scenario_candidates_prepared", **audit)
    return candidates


def _select_bounded_records(
    state: HARAState, *, field: str, identity: str, limit: int | None,
    selected_ids: tuple[str, ...] = (),
) -> HARAState:
    if limit is None:
        return state
    values = getattr(state, field)
    if selected_ids:
        indexed = {str(item.get(identity, "")): item for item in values}
        missing = [item_id for item_id in selected_ids if item_id not in indexed]
        if missing:
            raise ValueError(f"Bounded sample {field} IDs are unavailable: {missing}")
        selected = [indexed[item_id] for item_id in selected_ids]
    else:
        selected = values[:limit]
    setattr(state, field, selected)
    inventory = state.item_definition.setdefault("bounded_sample_inventory", {})
    inventory[field] = {
        "available_count": len(values),
        "selected_count": len(selected),
        "selected_ids": [str(item.get(identity, "")) for item in selected],
        "omitted_count": len(values) - len(selected),
    }
    state.record("bounded_sample_records_selected", field=field, **inventory[field])
    return state


def _assess_guidewords_after_project_context_preflight(
    state: HARAState,
    inputs: SemanticWorkflowInputs,
    agents: SemanticWorkflowAgents,
    checkpoint: Callable[[HARAState], object] | None = None,
) -> HARAState:
    if inputs.project_context_preflight is not None:
        audit = inputs.project_context_preflight(state)
        state.record("project_context_preflight_passed", **audit)
    return assess_guidewords(
        state,
        agents.guidewords,
        [_function(value) for value in state.functions],
        inputs.guidewords,
        max_workers=inputs.max_workers,
        progress=inputs.progress,
        review_artifact_writer=inputs.review_artifact_writer,
        checkpoint=checkpoint,
    )


def build_semantic_frontend_graph(
    inputs: SemanticWorkflowInputs,
    agents: SemanticWorkflowAgents,
    checkpoint_repository: CheckpointRepository | None = None,
) -> WorkflowGraph:
    """Compose the reusable evidence-extraction and scenario-selection slice."""
    graph = WorkflowGraph(
        checkpoint_repository,
        progress=inputs.stage_progress,
        review_artifact_writer=inputs.review_artifact_writer,
    )
    graph.add_node(
        WorkflowStage.INITIALIZE,
        lambda state: read_item_document(
            state,
            str(inputs.item_path),
            project_analysis_policy=(
                inputs.method_contract.metadata.get("project_analysis_policy", {})
                if inputs.method_contract is not None else {}
            ),
        ),
    )
    graph.add_node(
        WorkflowStage.EXTRACT,
        lambda state: _select_bounded_records(extract_item_artifacts(
            state,
            ItemArtifactExtractionAgent(
                agents.client,
            ),
            ItemSupplementAgent(agents.client),
            ItemEvidenceRouter(),
            targeted_agent=TargetedProjectFactExtractionAgent(agents.client),
            max_workers=inputs.max_workers,
            progress=inputs.progress,
            cache=inputs.artifact_cache,
            required_fact_specs=inputs.required_fact_specs,
            requested_operating_modes=inputs.requested_operating_modes,
            review_artifact_writer=inputs.review_artifact_writer,
        ), field="functions", identity="function_id",
            limit=inputs.sample_function_limit,
            selected_ids=inputs.sample_function_ids),
    )
    graph.add_node(
        WorkflowStage.FUNCTIONS,
        lambda state: _assess_guidewords_after_project_context_preflight(
            state, inputs, agents,
            checkpoint=(
                checkpoint_repository.save
                if checkpoint_repository is not None else None
            ),
        ),
    )
    graph.add_node(
        WorkflowStage.HAZOP,
        lambda state: _select_bounded_records(derive_malfunctions(
            state,
            agents.malfunctions,
            [_function(value) for value in state.functions],
            [_assessment(value) for value in _guideword_assessment_values(state)],
            max_workers=inputs.max_workers,
            progress=inputs.progress,
            review_artifact_writer=inputs.review_artifact_writer,
            selector_resolver=(
                FailureModeSelectorResolver(inputs.method_contract)
                if inputs.method_contract is not None else None
            ),
        ), field="malfunctions", identity="malfunction_id",
            limit=inputs.sample_malfunction_limit,
            selected_ids=inputs.sample_malfunction_ids),
    )
    graph.add_node(
        WorkflowStage.MALFUNCTIONS,
        lambda state: assess_scenarios(
            state,
            agents.scenarios,
            [_malfunction(value) for value in state.malfunctions],
            _scenario_candidates(state, inputs),
            max_workers=inputs.max_workers,
            progress=inputs.progress,
            risk_fact_agent=agents.risk_facts,
            required_fact_specs=inputs.required_fact_specs,
            method=inputs.method_contract,
            checkpoint=(
                checkpoint_repository.save
                if checkpoint_repository is not None else None
            ),
            review_artifact_writer=inputs.review_artifact_writer,
        ),
    )
    return graph


def build_hara_agent_graph(
    inputs: SemanticWorkflowInputs,
    agents: SemanticWorkflowAgents,
    risk_services: RiskWorkflowServices,
    checkpoint_repository: CheckpointRepository | None = None,
    reporting: ReportingWorkflowConfig | None = None,
) -> WorkflowGraph:
    """Compose the production path through scoring, quality gate, and reporting."""
    graph = build_semantic_frontend_graph(inputs, agents, checkpoint_repository)
    scenario_node = graph.nodes[WorkflowStage.MALFUNCTIONS]
    downstream_preflight = DownstreamPreflightService()
    graph.nodes[WorkflowStage.MALFUNCTIONS] = lambda state: (
        downstream_preflight.validate(state), scenario_node(state)
    )[1]
    graph.add_node(
        WorkflowStage.SCORING,
        lambda state: score_structured_scenarios(
            state,
            risk_services.scoring,
            risk_services.asil_table,
            risk_services.risk_fact_binding,
            inputs.review_artifact_writer,
        ),
    )
    graph.add_node(
        WorkflowStage.SAFETY_GOALS,
        lambda state: aggregate_safety_goals(state, risk_services.safety_goals),
    )
    graph.add_node(WorkflowStage.QUALITY_GATE, pass_quality_gate)
    if reporting is not None:
        graph.add_node(
            WorkflowStage.RENDER,
            lambda state: render_excel_report(
                state,
                reporting.renderer,
                reporting.template_path,
                reporting.output_path,
            ),
        )
    return graph
