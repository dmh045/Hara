from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from hara_agent.contracts import ScenarioCausalAssessment
from hara_agent.models import (
    EvidenceValue, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    RiskAssessment, ScenarioCandidate, SourceRef, evaluate_risk_eligibility_payload,
)
from hara_agent.services.analysis import (
    ASILLookupService, MethodRiskFactBindingService,
    ExposureDimensionCoverageService, PotentialHarmResolver,
    HazardousEventRiskContextService,
    RiskCalculationInputService, RiskExecutionTraceService,
    ScenarioScoringService,
)
from hara_agent.services.semantic.scenario_evidence import FactRegistry, build_fact_registry
from hara_agent.services.semantic.project_evidence_registry import (
    MethodEvidenceProvider, build_project_evidence_registry,
)
from hara_agent.services.analysis.scenario_physics import (
    closing_relative_speed_kph, derive_scenario_physics,
    derive_stationary_object_speed,
    select_ego_speed_from_policy,
)
from hara_agent.workflow.state import HARAState, WorkflowStage
from hara_agent.workflow.review_artifacts import ReviewArtifactWriter


def _unique_index(values, key, label: str):
    result = {}
    for value in values:
        identity = key(value)
        if not identity:
            raise ValueError(f"{label} ID不能为空")
        if identity in result:
            raise ValueError(f"{label} ID不唯一，禁止静默覆盖: {identity!r}")
        result[identity] = value
    return result


def _review_status(value: str) -> ReviewStatus:
    return ReviewStatus.FINALIZED if str(value).upper() in {"FINALIZED", "APPROVED"} else ReviewStatus.PENDING


def _scoring_evidence(result: dict[str, Any], value_key: str) -> EvidenceValue[str]:
    status = _review_status(result.get("engineering_status", "PENDING"))
    sources = []
    engineering_source = str(result.get("engineering_source", "")).strip()
    if engineering_source:
        sources.append(SourceRef(
            str(result.get("engineering_source_type", "scoring_service")),
            engineering_source,
            str(result.get("engineering_location", result.get("engineering_rule_id", ""))),
            str(result.get("engineering_excerpt", result.get("engineering_basis", ""))),
        ))
    for source in result.get("fact_sources", []):
        if isinstance(source, dict):
            sources.append(SourceRef(
                str(source.get("source_type", "project_fact")),
                str(source.get("source_id", "")),
                str(source.get("location", "")),
                str(source.get("excerpt", "")),
            ))
    if result.get("template_standard_source"):
        contract_hash = str(result.get("template_standard_contract_hash", ""))
        sources.append(SourceRef(
            "method_contract" if contract_hash else "input_template",
            contract_hash or str(result["template_standard_source"]),
            f"{result.get('template_standard_sheet', '')}!{result.get('template_standard_location', '')}",
            "；".join(value for value in (
                str(result.get("template_standard_description", "")),
                str(result.get("template_standard_criterion", "")),
            ) if value),
        ))
    return EvidenceValue(
        value=str(result[value_key]),
        status=status,
        sources=sources,
        rule_version=str(result.get("engineering_rule_version", "")),
        review_reason=(
            "" if status is ReviewStatus.FINALIZED
            else str(result.get("engineering_basis", "Scoring evidence is pending."))
        ),
    )


def _validated_causal_assessment(value: dict[str, Any]) -> ScenarioCausalAssessment:
    payload = value.get("causal_assessment")
    if not isinstance(payload, dict):
        raise ValueError(
            "Scenario scoring requires a typed causal_assessment; legacy flags are not authoritative"
        )
    try:
        assessment = ScenarioCausalAssessment.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Scenario causal_assessment failed deterministic validation") from error
    dimensions = [item.dimension for item in assessment.risk_dimension_changes]
    consistent = all((
        assessment.scenario_id == str(value.get("scenario_id", "")),
        assessment.is_validated is (value.get("causally_relevant") is True),
        assessment.breakpoint is not None,
        assessment.breakpoint.value == str(value.get("breakpoint", "")),
        dimensions == list(value.get("risk_dimensions_changed", [])),
        assessment.hazardous_event == str(value.get("hazardous_event", "")),
        # Potential Harm is deliberately absent/empty at the v4 Scenario
        # semantic boundary and is attached after deterministic Severity.
    ))
    if not consistent:
        raise ValueError("Scenario causal_assessment conflicts with compatibility fields")
    return assessment


def score_structured_scenarios(
    state: HARAState,
    scoring: ScenarioScoringService,
    asil_table: ASILLookupService,
    risk_fact_binding: MethodRiskFactBindingService | None = None,
    review_artifact_writer: ReviewArtifactWriter | None = None,
) -> HARAState:
    """Score each retained malfunction-scenario association without prose heuristics."""
    scenario_by_id = _unique_index(state.scenarios, lambda item: item.scenario_id, "Scenario")
    malfunction_by_id = _unique_index(
        state.malfunctions, lambda item: str(item.get("malfunction_id", "")), "Malfunction",
    )
    function_by_id = _unique_index(
        state.functions, lambda item: str(item.get("function_id", "")), "Function",
    )
    coverage_service = (
        ExposureDimensionCoverageService(scoring.method)
        if getattr(scoring, "method", None) is not None
        and getattr(scoring.method, "structured_risk_method", None) is not None
        else None
    )
    risk_context_service = (
        HazardousEventRiskContextService(scoring.method)
        if getattr(scoring, "method", None) is not None
        and getattr(scoring.method, "structured_risk_method", None) is not None
        else None
    )
    assessments = state.item_definition.get("scenario_assessments", [])
    retained = []
    for item in assessments:
        if not isinstance(item, dict):
            continue
        decision = evaluate_risk_eligibility_payload(item)
        if decision.eligible:
            retained.append((item, _validated_causal_assessment(item)))
        elif (
            "PENDING_LEGACY_ELIGIBILITY" in decision.reason_codes
            and all(item.get(field) is True for field in (
                "physically_feasible", "functionally_relevant", "causally_relevant",
            ))
        ):
            # A legacy payload cannot promote itself through positive flags.  Keep
            # the historical explicit failure rather than silently reaching the
            # risk stage with no typed causal contract.
            _validated_causal_assessment(item)
    risks: list[RiskAssessment] = []
    assessment_counts = Counter(
        str(item.get("scenario_id", "")) for item, _ in retained
    )
    pending: list[dict[str, Any]] = []
    binding_audits: list[dict[str, Any]] = []
    calculation_audits: list[dict[str, Any]] = []
    scored_by_pair: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    calculation_inputs = RiskCalculationInputService()
    harm_resolver = PotentialHarmResolver()
    project_facts = (
        ItemDefinitionFacts.from_dict(state.item_definition["typed"])
        if risk_fact_binding is not None else None
    )

    for index, (assessment, causal_assessment) in enumerate(retained, start=1):
        scenario_id = str(assessment.get("scenario_id", ""))
        malfunction_id = str(assessment.get("malfunction_id", ""))
        if scenario_id not in scenario_by_id or malfunction_id not in malfunction_by_id:
            raise ValueError(
                f"场景评分外键缺失: scenario_id={scenario_id!r}, malfunction_id={malfunction_id!r}"
            )
        candidate = scenario_by_id[scenario_id]
        malfunction = malfunction_by_id[malfunction_id]
        method = getattr(scoring, "method", None)
        policy = (
            method.metadata.get("project_analysis_policy", {})
            if method is not None else {}
        )
        selected_speed = select_ego_speed_from_policy(
            candidate, policy=policy, malfunction_id=malfunction_id,
        )
        if selected_speed is not None:
            value, metadata = selected_speed
            # The scoring view is scoped to this malfunction. A scenario used
            # by only one assessment can also retain the fact in its checkpoint.
            scoped_facts = {**candidate.facts, "ego_speed_kph": value}
            scoped_provenance = {**candidate.fact_provenance, "ego_speed_kph": metadata}
            if assessment_counts[scenario_id] == 1:
                candidate.facts = scoped_facts
                candidate.fact_provenance = scoped_provenance
            else:
                candidate = replace(
                    candidate, facts=scoped_facts,
                    fact_provenance=scoped_provenance,
                )
        stationary_speed = derive_stationary_object_speed(
            candidate,
            method_hash=str(method.metadata.get("method_source_hash", ""))
            if method is not None else "",
            malfunction_id=malfunction_id,
        )
        if stationary_speed is not None:
            value, metadata = stationary_speed
            scoped_facts = {**candidate.facts, "object_speed_kph": value}
            scoped_provenance = {**candidate.fact_provenance, "object_speed_kph": metadata}
            if assessment_counts[scenario_id] == 1:
                candidate.facts = scoped_facts
                candidate.fact_provenance = scoped_provenance
            else:
                candidate = replace(
                    candidate, facts=scoped_facts,
                    fact_provenance=scoped_provenance,
                )
        scenario = {
            "scenario_id": scenario_id,
            "malfunction_id": malfunction_id,
            **candidate.facts,
        }
        for key in ("component_category", "failure_type"):
            if malfunction.get(key) not in (None, ""):
                scenario[key] = malfunction[key]
        scenario["_fact_provenance"] = dict(candidate.fact_provenance)
        physics_conflicts: list[dict[str, Any]] = []
        explicit_relative = scenario.get("relative_speed_kph")
        calculated_relative = closing_relative_speed_kph(
            scenario.get("ego_speed_kph"), scenario.get("object_speed_kph"),
            ego_direction=scenario.get("ego_longitudinal_direction"),
            object_direction=scenario.get("object_longitudinal_direction"),
            collision_type=scenario.get("collision_type"),
        )
        if (
            isinstance(explicit_relative, (int, float))
            and not isinstance(explicit_relative, bool)
            and calculated_relative is not None
            and abs(float(explicit_relative) - calculated_relative) > 1e-6
        ):
            physics_conflicts.append({
                "field": "relative_speed_kph",
                "explicit_value": explicit_relative,
                "derived_value": calculated_relative,
                "reason": "FACT_SOURCE_CONFLICT",
            })
        blocked_physics_fields = (
            {"relative_speed_kph", "ttc_s"} if physics_conflicts else set()
        )
        for field in blocked_physics_fields:
            scenario.pop(field, None)
            scenario["_fact_provenance"].pop(field, None)
        derived_physics = derive_scenario_physics(candidate)
        for record in derived_physics:
            key = record.evidence_ref.split(".", 1)[1]
            if key in blocked_physics_fields:
                continue
            if key in scenario and scenario[key] != record.value:
                physics_conflicts.append({
                    "field": key, "explicit_value": scenario[key],
                    "derived_value": record.value,
                    "reason": "FACT_SOURCE_CONFLICT",
                })
                scenario.pop(key, None)
                scenario["_fact_provenance"].pop(key, None)
                continue
            scenario[key] = record.value
            scenario["_fact_provenance"][key] = {
                "provenance": record.provenance.value,
                "approval": record.approval_status.value,
                "source_refs": [
                    {
                        "source_type": source.source_type,
                        "source_id": source.source_id,
                        "location": source.location,
                        "excerpt": source.excerpt,
                    }
                    for source in record.source_refs
                ],
                "evidence_ref": record.evidence_ref,
                **record.metadata,
            }
        scenario.setdefault("situational_description", candidate.situational_description)
        scenario.setdefault("situational_detailing", candidate.situational_detailing)
        if risk_fact_binding is not None and project_facts is not None:
            binding = risk_fact_binding.bind(project_facts, {
                **scenario,
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "atomic_variant": candidate.atomic_variant,
            })
            binding_source_conflicts: list[dict[str, Any]] = []
            if risk_context_service is not None:
                for field, incoming in binding.values.items():
                    try:
                        risk_context_service.validate_source_conflicts(
                            scenario, {field: incoming},
                        )
                    except ValueError as error:
                        if not str(error).startswith("FACT_SOURCE_CONFLICT:"):
                            raise
                        binding_source_conflicts.append({
                            "field": field,
                            "scenario_value": scenario[field],
                            "project_value": incoming,
                            "reason": "FACT_SOURCE_CONFLICT",
                        })
            conflicting_fields = {item["field"] for item in binding_source_conflicts}
            blocked_fields = set(conflicting_fields)
            # A derived Risk input must not survive removal of one of its
            # conflicting source facts. Independent Exposure atoms remain.
            changed = True
            while changed:
                changed = False
                for field, metadata in scenario["_fact_provenance"].items():
                    if field in blocked_fields or not isinstance(metadata, dict):
                        continue
                    inputs = metadata.get("inputs", ())
                    if isinstance(inputs, (list, tuple)) and any(
                        isinstance(item, str) and item.removeprefix("SCN.") in blocked_fields
                        for item in inputs
                    ):
                        blocked_fields.add(field)
                        changed = True
            for field in blocked_fields:
                scenario.pop(field, None)
                scenario["_fact_provenance"].pop(field, None)
            scenario.update({
                field: value for field, value in binding.values.items()
                if field not in blocked_fields
            })
            scenario["_fact_provenance"].update({
                field: value for field, value in binding.provenance.items()
                if field not in blocked_fields
            })
            binding_audits.append({
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                **binding.audit,
                "source_conflicts": binding_source_conflicts,
                "blocked_dependent_fields": sorted(blocked_fields - conflicting_fields),
            })
        hazard_event = causal_assessment.hazardous_event
        if not hazard_event:
            raise ValueError(
                f"保留场景缺少Hazardous Event: {malfunction_id}/{scenario_id}"
            )
        risk_context = None
        if risk_context_service is not None:
            risk_context = risk_context_service.build(
                malfunction_id=malfunction_id,
                scenario_id=scenario_id,
                hazard_node_id=causal_assessment.causal_chain[-1],
                scenario=scenario,
            )
            scenario = risk_context_service.scoring_facts(risk_context, scenario)

        if coverage_service is not None:
            function = coverage_service.function_from_projection(
                function_by_id.get(str(malfunction.get("function_id", "")), {})
            )
            scenario["_exposure_dimension_coverage_decision"] = coverage_service.decide(
                assessment_key=f"{malfunction_id}::{scenario_id}",
                function=function,
                operating_mode=candidate.operating_mode,
            )
        scored = scoring.score(scenario, hazard_event)
        scored_by_pair[(malfunction_id, scenario_id)] = (
            scored["severity"], scored["exposure"], scored["controllability"],
            dict(scenario),
        )
        severity = _scoring_evidence(scored["severity"], "severity_score")
        exposure = _scoring_evidence(scored["exposure"], "exposure_score")
        controllability = _scoring_evidence(
            scored["controllability"], "controllability_score"
        )
        method = getattr(scoring, "method", None)
        method_sources = []
        for source in malfunction.get("sources", []):
            method_sources.append(source if isinstance(source, SourceRef) else SourceRef(**source))
        malfunction_candidate = MalfunctionCandidate(
            malfunction_id=malfunction_id,
            function_id=str(malfunction.get("function_id", "UNKNOWN")),
            guideword=str(malfunction.get("guideword", "UNKNOWN")),
            description=str(malfunction.get("description", "malfunction")),
            functional_effect=str(malfunction.get("functional_effect", "behavior changes")),
            vehicle_level_hazard=str(malfunction.get("vehicle_level_hazard", "hazardous state")),
            causal_chain=[str(item) for item in malfunction.get("causal_chain", ("M", "B"))],
            sources=method_sources,
            status=ReviewStatus(malfunction.get("status", ReviewStatus.PENDING.value)),
        )
        evidence_registry = build_fact_registry(
            malfunction_candidate, candidate,
            build_project_evidence_registry(project_facts, method)
            if project_facts is not None else None,
        )
        if method is not None:
            # The production registry and the risk-stage registry use the
            # same provider contract. The explicit extension is harmless for
            # callers that supplied only a scenario-local registry.
            method_records = tuple(MethodEvidenceProvider(method).evidence_records())
            existing = {item.evidence_ref for item in evidence_registry.records}
            evidence_registry.extend(
                item for item in method_records if item.evidence_ref not in existing
            )
        harm = harm_resolver.resolve(
            method=method,
            severity_result=scored["severity"],
            scenario=scenario,
            registry=evidence_registry,
        ) if method is not None else None
        potential_harm = harm.potential_harm if harm is not None else ""
        if harm is not None and harm.status.value == "FINALIZED":
            causal_assessment = causal_assessment.with_harm(
                potential_harm=harm.potential_harm,
                evidence_refs=harm.evidence_refs,
                source_refs=harm.source_refs,
                review_status=ReviewStatus.FINALIZED,
            )
        exposure_method = str(scored["exposure"].get("exposure_method", "")).upper()
        valid_scores = (
            severity.value in {"S0", "S1", "S2", "S3"}
            and exposure.value in {"E0", "E1", "E2", "E3", "E4"}
            and controllability.value in {"C0", "C1", "C2", "C3"}
        )
        asil_value = (
            asil_table.determine(
                severity.value, exposure.value, controllability.value
            )
            if valid_scores else ""
        )
        asil_status = (
            ReviewStatus.FINALIZED
            if asil_value and asil_value not in {"NA", "N/A"}
            and all(
                item.status is ReviewStatus.FINALIZED
                for item in (severity, exposure, controllability)
            )
            else ReviewStatus.PENDING
        )
        asil = EvidenceValue(
            value=asil_value,
            status=asil_status,
            sources=(
                [asil_table.evidence_source(
                    severity.value, exposure.value, controllability.value
                )]
                if valid_scores else []
            ),
            review_reason=(
                "" if asil_status is ReviewStatus.FINALIZED
                else "ASIL lookup requires complete, approved S/E/C values."
            ),
        )
        risk = RiskAssessment(
            assessment_id=f"RA-{index:04d}-{malfunction_id}-{scenario_id}",
            scenario_id=scenario_id,
            severity=severity,
            exposure=exposure,
            controllability=controllability,
            asil=asil,
            malfunction_id=malfunction_id,
            hazardous_event=hazard_event,
            potential_harm=potential_harm,
            exposure_tf=exposure_method,
        )
        risks.append(risk)
        for stored in assessments:
            if (
                str(stored.get("malfunction_id", "")) == malfunction_id
                and str(stored.get("scenario_id", "")) == scenario_id
            ):
                stored["causal_assessment"] = causal_assessment.to_dict()
                stored["potential_harm"] = potential_harm
                break
        calculation_audits.append({
            "malfunction_id": malfunction_id,
            "scenario_id": scenario_id,
            "physics_conflicts": physics_conflicts,
            "severity_input": calculation_inputs.severity(
                malfunction_id, scenario_id, scenario,
                getattr(getattr(method, "structured_risk_method", None), "severity", None).speed_semantic
                if getattr(getattr(method, "structured_risk_method", None), "severity", None) is not None
                else None,
            ).to_dict(),
            "exposure_input": calculation_inputs.exposure(
                scenario_id, scenario,
            ).to_dict(),
            "controllability_input": calculation_inputs.controllability(
                malfunction_id, scenario_id, scenario,
            ).to_dict(),
            "hazardous_event_risk_context": (
                risk_context.to_dict() if risk_context is not None else {}
            ),
            "severity_readiness": (
                risk_context_service.severity_readiness(risk_context)
                if risk_context is not None and risk_context_service is not None else {}
            ),
            "controllability_readiness": (
                risk_context_service.controllability_readiness(risk_context)
                if risk_context is not None and risk_context_service is not None else {}
            ),
            "ftti_input": calculation_inputs.ftti(
                malfunction_id, scenario_id, asil_value,
            ).to_dict(),
            "potential_harm": harm.to_dict() if harm is not None else {
                "status": "PENDING_METHOD_SEMANTICS",
                "reason": "No active MethodContract was available for deterministic harm resolution.",
            },
        })
        for field_name in ("severity", "exposure", "controllability", "asil"):
            evidence = getattr(risk, field_name)
            if evidence.status is ReviewStatus.PENDING:
                pending.append({
                    "assessment_id": risk.assessment_id,
                    "field": field_name,
                    "reason": evidence.review_reason,
                })

    state.risk_results = risks
    state.pending_reviews.extend(pending)
    state.stage = WorkflowStage.SAFETY_GOALS
    state.record(
        "structured_risk_scoring_completed",
        assessment_count=len(risks),
        pending_field_count=len(pending),
        asil_source=asil_table.source,
        risk_fact_binding_audits=binding_audits,
        risk_calculation_inputs=calculation_audits,
    )
    if review_artifact_writer is not None:
        binding_gaps = []
        for event in reversed(state.audit_trail):
            if event.get("event") == "scenario_candidates_prepared":
                binding_gaps = list(event.get("scenario_binding_gaps", []))
                break
        review_artifact_writer.write_risk_execution_trace(
            RiskExecutionTraceService(getattr(scoring, "method", None)).project(
                run_id=state.run_id,
                assessments=state.item_definition.get("scenario_assessments", []),
                candidates=state.scenarios,
                committed=True,
                scored=scored_by_pair,
                risks=risks,
                binding_gaps=binding_gaps,
            )
        )
        review_artifact_writer.write_summary(state)
    return state
