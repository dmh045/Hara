from __future__ import annotations

import sys
import os
import hashlib
from pathlib import Path

from hara_agent.config import LLMConfig, RunConfig
from hara_agent.contracts import CompileStatus
from hara_agent.infrastructure.llm import LLMClient, create_llm_client
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
from hara_agent.services.analysis import (
    MethodScenarioCandidateService,
    ScenarioMethodService,
    MethodSafetyGoalService,
    ProjectFactResolver,
    SpeedResolutionResult,
    UnresolvedProjectContextError,
)
from hara_agent.models import (
    FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    SourceRef,
)
from hara_agent.services.extraction import (
    DocumentReader, ValidatedArtifactCache, select_source_blocks,
)
from hara_agent.services.reporting import HARAExcelRenderer, load_report_schema
from hara_agent.services.semantic import (
    GuidewordApplicabilityAgent,
    MalfunctionHazardAgent,
    ScenarioFeasibilityAgent,
    ScenarioRiskFactAgent,
)
from hara_agent.workflow import (
    CheckpointRepository,
    HARAState,
    ReportingWorkflowConfig,
    RiskWorkflowServices,
    SemanticWorkflowAgents,
    SemanticWorkflowInputs,
    WorkflowRunResult,
    WorkflowStage,
    ReviewArtifactWriter,
    build_hara_agent_graph,
)
from hara_agent.method_sources import MethodSourceKind, MethodSourceResolver
from hara_agent.services.analysis.risk_services import RiskScoringServices


class HARAApplication:
    """Dependency assembly for the template-driven HARA runtime."""

    def __init__(self, config: RunConfig, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    @classmethod
    def from_env(cls, config: RunConfig, llm_config: LLMConfig | None = None) -> "HARAApplication":
        config.validate()
        budget = (
            ProviderAttemptBudget(
                config.run_dir / f"{config.run_id}.provider-attempts.jsonl",
                run_id=config.run_id,
                limit=config.provider_attempt_limit,
            )
            if config.provider_attempt_limit is not None else None
        )
        return cls(
            config,
            create_llm_client(llm_config or LLMConfig.from_env(), attempt_budget=budget),
        )

    def run(self) -> WorkflowRunResult:
        self.config.validate()
        resolution = MethodSourceResolver().resolve(
            template_path=self.config.template_path,
            baseline_manifest_path=self.config.method_baseline_path,
            report_template_path=(
                self.config.template_path
                if self.config.template_path is not None
                else self.config.report_template_path
            ),
        )
        method = resolution.method
        method_ref = {
            "template_hash": method.metadata["template_hash"],
            "contract_version": method.contract_version,
            "compiler_version": method.compiler_version,
            "compile_status": method.compile_status.value,
            "engineering_rules_compiled": method.engineering_rules_compiled,
            "source_kind": resolution.source_kind.value,
            "method_source_hash": method.metadata.get("method_source_hash", method.metadata["template_hash"]),
            "report_template_hash": resolution.report_template_hash,
        }
        provider_config = getattr(self.llm_client, "config", None)
        bounded_source_context = {
            "item_sha256": hashlib.sha256(self.config.item_path.read_bytes()).hexdigest(),
            "method_source_hash": method_ref["method_source_hash"],
            "report_template_hash": method_ref["report_template_hash"],
            "operating_mode": self.config.operating_mode or "",
            "ego_speed_kph": self.config.ego_speed_kph,
            "ego_speed_source": self.config.ego_speed_source,
            "allow_aggregate_speed_fallback": self.config.allow_aggregate_speed_fallback,
            "provider": getattr(provider_config, "provider", ""),
            "model": getattr(provider_config, "model", ""),
            "extraction_thinking": getattr(provider_config, "extraction_thinking", ""),
            "guideword_thinking": getattr(provider_config, "guideword_thinking", ""),
            "malfunction_thinking": getattr(provider_config, "malfunction_thinking", ""),
            "scenario_thinking": getattr(provider_config, "scenario_thinking", ""),
        } if self.config.bounded_sample else {}
        # Preserve the compiled template semantics.  The semantic agent still
        # accepts plain strings for tests/embedders, but production must not
        # discard the normative description and source binding here.
        guidewords = list(method.guidewords.guidewords)
        candidate_service = MethodScenarioCandidateService(method)

        def report_batch(name: str, completed: int, total: int) -> None:
            print(
                f"[HARA] {name}: {completed}/{total} completed "
                f"(max_workers={self.config.max_workers})",
                file=sys.stderr,
                flush=True,
            )

        def report_stage(event: str, stage: str, elapsed: float) -> None:
            suffix = f" elapsed={elapsed:.1f}s" if event != "started" else ""
            print(f"[HARA] stage={stage} {event}{suffix}", file=sys.stderr, flush=True)

        def prepare_candidates(state: HARAState):
            return self.prepare_scenario_candidates(state, candidate_service)
        checkpoints = CheckpointRepository(self.config.run_dir)
        review_root = os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review")
        if self.config.bounded_sample:
            if getattr(self.llm_client, "attempt_budget", None) is None:
                raise ValueError("Bounded production sample requires an instrumented Provider client")
            if not self.config.resume and any(path.exists() for path in (
                checkpoints.path_for(self.config.run_id),
                self.config.run_dir / f"{self.config.run_id}.provider-attempts.jsonl",
                self.config.output_path,
                Path(review_root) / self.config.run_id,
            )):
                raise ValueError("Bounded sample run ID or output already exists; choose new paths")
        review_artifact_writer = ReviewArtifactWriter(
            self.config.run_id,
            review_root,
        )
        renderer = HARAExcelRenderer(
            method.report_contract,
            # This is the independently resolved report-layout hash. In
            # Template mode it happens to equal the method-source hash; in
            # YAML mode it intentionally does not.
            template_hash=resolution.report_template_hash,
            report_schema=load_report_schema(),
            method_contract=method,
        )
        if self.config.resume:
            state = checkpoints.load(self.config.run_id)
            if state.stage is not WorkflowStage.INITIALIZE:
                saved_selection = state.item_definition.get("source_selection", {})
                saved_fingerprint = (
                    saved_selection.get("fingerprint", "")
                    if isinstance(saved_selection, dict) else ""
                )
                current_selection = select_source_blocks(
                    DocumentReader().read(self.config.item_path).blocks,
                    method.metadata.get("project_analysis_policy", {}),
                )
                if (
                    not saved_fingerprint
                    or saved_fingerprint != current_selection.fingerprint
                    or str(Path(state.item_definition.get("source_path", "")).resolve())
                    != str(self.config.item_path.resolve())
                ):
                    raise ValueError(
                        "Checkpoint Item Definition or source-selection policy changed; "
                        "start a new run instead of reusing derived facts"
                    )
            configured_event = next((
                event for event in state.audit_trail
                if event.get("event") == "bounded_sample_configured"
            ), None)
            configured = configured_event.get("scope") if configured_event else None
            if (configured is not None) != self.config.bounded_sample:
                raise ValueError("Bounded sample mode differs from committed checkpoint")
            if configured is not None and configured != self.config.sample_scope():
                raise ValueError("Bounded sample scope differs from committed checkpoint")
            if configured is not None and configured_event.get("source_context") != bounded_source_context:
                raise ValueError("Bounded sample inputs, method, or Provider configuration changed")
            checkpoint_hash = state.method_contract.get("template_hash")
            if not checkpoint_hash and state.stage.value != "initialize":
                raise ValueError(
                    "Checkpoint predates MethodContract template binding; "
                    "restart the run instead of reusing unbound derived outputs"
                )
            if checkpoint_hash and checkpoint_hash != method_ref["template_hash"]:
                raise ValueError(
                    "Checkpoint template hash does not match the active MethodContract: "
                    f"checkpoint={checkpoint_hash}, active={method_ref['template_hash']}"
                )
            state.method_contract = method_ref
        else:
            state = HARAState(
                run_id=self.config.run_id,
                method_contract=method_ref,
            )
            if self.config.bounded_sample:
                state.record(
                    "bounded_sample_configured",
                    scope=self.config.sample_scope(),
                    source_context=bounded_source_context,
                    report_class="ENGINEERING_SAMPLE_ONLY",
                )
            state.record(
                "method_contract_compiled",
                template_path=(str(self.config.template_path) if self.config.template_path else ""),
                method_baseline_path=(
                    str(self.config.method_baseline_path)
                    if resolution.source_kind is MethodSourceKind.YAML_BASELINE else ""
                ),
                report_template_path=str(resolution.report_template_path),
                **method_ref,
                guideword_count=len(guidewords),
                scenario_dimension_count=len(method.scenario_model.dimensions),
                scoring_standard_counts={
                    "severity_rules": len(method.severity.rules),
                    "exposure_duration_rules": len(method.exposure.duration_rules),
                    "exposure_frequency_rules": len(method.exposure.frequency_rules),
                    "controllability_rules": len(method.controllability.criteria),
                },
                required_fact_types=[
                    item.fact_type.value for item in method.required_fact_specs
                ],
                warning_codes=sorted({item.code.value for item in method.warnings}),
            )
        risk_services = RiskScoringServices.from_method(method)
        graph = build_hara_agent_graph(
            SemanticWorkflowInputs(
                item_path=self.config.item_path,
                guidewords=guidewords,
                scenario_candidate_factory=prepare_candidates,
                project_context_preflight=(
                    lambda state: self.resolve_project_speed_context(state).to_dict()
                ),
                max_workers=self.config.max_workers,
                progress=report_batch,
                stage_progress=report_stage,
                artifact_cache=ValidatedArtifactCache(
                    os.getenv("HARA_ARTIFACT_CACHE_DIR", "runtime/agent/artifact-cache"),
                    os.getenv("HARA_ARTIFACT_CACHE_MODE", "readwrite"),
                ),
                required_fact_specs=method.required_fact_specs,
                method_contract=method,
                requested_operating_modes=(
                    (str(self.config.operating_mode),)
                    if self.config.operating_mode else ()
                ),
                review_artifact_writer=review_artifact_writer,
                sample_function_limit=self.config.sample_function_limit,
                sample_malfunction_limit=self.config.sample_malfunction_limit,
                sample_function_ids=self.config.sample_function_ids,
                sample_malfunction_ids=self.config.sample_malfunction_ids,
            ),
            SemanticWorkflowAgents(
                client=self.llm_client,
                guidewords=GuidewordApplicabilityAgent(self.llm_client),
                malfunctions=MalfunctionHazardAgent(
                    self.llm_client,
                    component_categories=(
                        tuple(sorted({
                            category
                            for rule in method.structured_risk_method.exposure.domain_rules
                            for category in rule.component_categories
                        }))
                        if method.structured_risk_method is not None else ()
                    ),
                    failure_types=(
                        method.scenario_model.scenario_method
                        .failure_mode_selector_taxonomy.failure_types
                        if method.scenario_model.scenario_method
                        .failure_mode_selector_taxonomy is not None else ()
                    ),
                ),
                scenarios=ScenarioFeasibilityAgent(self.llm_client),
                risk_facts=ScenarioRiskFactAgent(self.llm_client),
            ),
            RiskWorkflowServices(
                scoring=risk_services.scoring,
                asil_table=risk_services.asil,
                safety_goals=MethodSafetyGoalService(method),
                risk_fact_binding=risk_services.binding,
            ),
            checkpoint_repository=checkpoints,
            reporting=ReportingWorkflowConfig(
                template_path=resolution.report_template_path,
                output_path=self.config.output_path,
                renderer=renderer,
            ),
        )
        preview = None
        if self.config.bounded_sample and self.config.resume:
            preview = next((
                event for event in reversed(state.audit_trail)
                if event.get("event") == "bounded_sample_scope_preview"
            ), None)
            if preview is not None and not preview.get("within_pair_limit", False):
                return WorkflowRunResult(state, True, "bounded_sample_scope_exceeds_pair_limit")
        stop_before = (
            {WorkflowStage.MALFUNCTIONS}
            if self.config.bounded_sample and preview is None else None
        )
        result = graph.run(state, stop_before=stop_before)
        if (
            self.config.bounded_sample
            and result.reason == "planned_stop:malfunctions"
        ):
            preview = self._preview_bounded_scope(
                result.state, candidate_service, method,
            )
            result.state.record("bounded_sample_scope_preview", **preview)
            checkpoints.save(result.state)
            return WorkflowRunResult(
                result.state, True, "bounded_sample_scope_preview",
            )
        if (
            self.config.allow_draft
            and result.interrupted
            and result.reason == "pending_engineering_review"
        ):
            output = renderer.render(
                result.state,
                resolution.report_template_path,
                self.config.output_path,
                draft=True,
            )
            result.state.record(
                "draft_excel_report_rendered",
                output_path=str(output),
                pending_review_count=len(result.state.pending_reviews),
            )
            checkpoints.save(result.state)
            return WorkflowRunResult(
                result.state,
                True,
                "draft_report_generated_pending_review",
            )
        return result

    def resolve_project_speed_context(self, state: HARAState) -> SpeedResolutionResult:
        operating_mode = str(self.config.operating_mode or "").strip()
        if not operating_mode:
            raise UnresolvedProjectContextError(
                operating_mode,
                "production Scenario Candidate preparation requires explicit structured operating_mode",
            )
        if self.config.ego_speed_kph is not None:
            sources = []
            if self.config.ego_speed_source:
                sources.append(SourceRef(
                    "project_input",
                    self.config.ego_speed_source,
                    "CLI --ego-speed-kph",
                    f"ego_speed_kph={self.config.ego_speed_kph:g}",
                ))
            return ProjectFactResolver.explicit_project_speed(
                operating_mode,
                self.config.ego_speed_kph,
                sources=sources,
            )
        typed = state.item_definition.get("typed", {})
        if not isinstance(typed, dict) or not typed:
            raise UnresolvedProjectContextError(
                operating_mode, "typed ProjectFacts are unavailable"
            )
        facts = ItemDefinitionFacts.from_dict(typed)
        return ProjectFactResolver().resolve_speed_context(
            facts,
            operating_mode,
            allow_aggregate_fallback=self.config.allow_aggregate_speed_fallback,
        )

    def prepare_scenario_candidates(
        self,
        state: HARAState,
        candidate_service: MethodScenarioCandidateService,
    ):
        typed = state.item_definition.get("typed", {})
        if not isinstance(typed, dict) or not typed:
            raise UnresolvedProjectContextError(
                str(self.config.operating_mode or ""),
                "typed ProjectFacts are unavailable",
            )
        facts = ItemDefinitionFacts.from_dict(typed)
        resolution = self.resolve_project_speed_context(state)
        functions = []
        for value in state.functions:
            if not isinstance(value, dict):
                continue
            payload = dict(value)
            payload["sources"] = [
                item if isinstance(item, SourceRef) else SourceRef(**item)
                for item in payload.get("sources", [])
            ]
            payload["status"] = ReviewStatus(
                payload.get("status", ReviewStatus.PENDING.value)
            )
            functions.append(FunctionDefinition(**payload))
        candidates, audit = candidate_service.generate(
            project_facts=facts,
            operating_mode=resolution.operating_mode,
            speed_resolution=resolution,
            functions=functions,
        )
        if self.config.sample_parent_scenario_limit is not None:
            available = len(candidates)
            if self.config.sample_parent_scenario_ids:
                indexed = {item.scenario_id: item for item in candidates}
                missing = [
                    item_id for item_id in self.config.sample_parent_scenario_ids
                    if item_id not in indexed
                ]
                if missing:
                    raise ValueError(f"Bounded sample parent Scenario IDs are unavailable: {missing}")
                candidates = [indexed[item_id] for item_id in self.config.sample_parent_scenario_ids]
            else:
                candidates = candidates[:self.config.sample_parent_scenario_limit]
            audit = dict(audit)
            audit["bounded_sample_parent_scenarios"] = {
                "available_count": available,
                "selected_count": len(candidates),
                "selected_ids": [item.scenario_id for item in candidates],
                "omitted_count": available - len(candidates),
            }
        return candidates, audit

    def _preview_bounded_scope(
        self, state: HARAState, candidate_service: MethodScenarioCandidateService,
        method,
    ) -> dict:
        from hara_agent.services.analysis.malfunction_situation_selection import selection_function

        candidates, audit = self.prepare_scenario_candidates(state, candidate_service)
        template_service = ScenarioMethodService(method)
        typed = state.item_definition.get("typed", {})
        project_facts = ItemDefinitionFacts.from_dict(typed)
        by_malfunction = {}
        for value in state.malfunctions:
            payload = dict(value)
            payload["sources"] = [
                item if isinstance(item, SourceRef) else SourceRef(**item)
                for item in payload.get("sources", [])
            ]
            payload["status"] = ReviewStatus(
                payload.get("status", ReviewStatus.PENDING.value)
            )
            malfunction = MalfunctionCandidate(**payload)
            instances, _ = template_service.instantiate_analytical_candidates(
                malfunction, candidates, project_facts=project_facts,
                function=selection_function(next(
                    (item for item in state.functions if item.get("function_id") == malfunction.function_id), None,
                )),
            )
            by_malfunction[malfunction.malfunction_id] = len(instances)
        pair_count = sum(by_malfunction.values())
        return {
            "function_count": len(state.functions),
            "malfunction_count": len(state.malfunctions),
            "parent_scenario_count": len(candidates),
            "selected_parent_scenario_ids": [item.scenario_id for item in candidates],
            "selected_parent_scenarios": [{
                "scenario_id": item.scenario_id,
                "operating_scenario": item.operating_scenario,
                "scenario_atom_ids": list(item.facts.get("scenario_atom_ids", [])),
                "object_type": str(item.facts.get("object_type", "")),
                "ego_action": str(item.facts.get("EGO_ACTION", "")),
            } for item in candidates],
            "scenario_pairs_after_method_instantiation": pair_count,
            "scenario_pairs_by_malfunction": by_malfunction,
            "scenario_pair_limit": self.config.sample_scenario_pair_limit,
            "within_pair_limit": (
                bool(state.malfunctions) and bool(candidates)
                and 0 < pair_count <= self.config.sample_scenario_pair_limit
            ),
            "available_parent_scenario_count": dict(
                audit.get("bounded_sample_parent_scenarios", {})
            ).get("available_count", len(candidates)),
        }
