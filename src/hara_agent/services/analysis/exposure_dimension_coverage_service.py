"""Method-authority audit and fail-closed Exposure dimension coverage decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from hara_agent.contracts import (
    ExposureDimensionCoverage,
    ExposureDimensionCoverageDecision,
    ExposureDimensionCoverageStatus,
    ExposureDimensionRequirementStatus,
    MethodContract,
)
from hara_agent.models import FunctionDefinition, ReviewStatus, evaluate_risk_eligibility_payload

from .scenario_coverage_service import ScenarioCoverageRuleService


class ExposureDimensionCoverageService:
    """Use only compiled approved coverage rules; never infer applicability."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.dimensions = tuple(
            item.canonical_name for item in method.scenario_model.dimensions
        )
        self.rules = ScenarioCoverageRuleService(method)

    @property
    def approved_rule_count(self) -> int:
        return len(self.rules.rules)

    @staticmethod
    def function_from_projection(value: dict[str, Any]) -> FunctionDefinition | None:
        function_id = str(value.get("function_id", "")).strip()
        name = str(value.get("name", "")).strip()
        output = str(value.get("output", "")).strip()
        if not function_id or not name or not output:
            return None
        return FunctionDefinition(
            function_id=function_id,
            name=name,
            output=output,
            description=str(value.get("description", "")),
            preconditions=[str(item) for item in value.get("preconditions", [])],
            triggers=[str(item) for item in value.get("triggers", [])],
            odd_constraints=[str(item) for item in value.get("odd_constraints", [])],
            status=ReviewStatus.FINALIZED,
        )

    def decide(
        self,
        *,
        assessment_key: str,
        function: FunctionDefinition | None,
        operating_mode: str,
    ) -> ExposureDimensionCoverageDecision:
        """Return a requirement decision without observing Scenario binding facts."""
        matches = (
            self.rules.matching_rules(function, operating_mode)
            if function is not None else ()
        )
        if len(matches) != 1:
            reason = (
                "METHOD_COVERAGE_RULE_ABSENT"
                if not self.rules.has_approved_rules
                else (
                    "NO_MATCHING_APPROVED_COVERAGE_RULE"
                    if not matches else "AMBIGUOUS_APPROVED_COVERAGE_RULE"
                )
            )
            return ExposureDimensionCoverageDecision(
                assessment_key=assessment_key,
                method_contract_hash=str(self.method.metadata.get("template_hash", "")),
                coverage_status=ExposureDimensionCoverageStatus.PENDING_METHOD_SEMANTICS,
                dimensions=tuple(
                    ExposureDimensionCoverage(
                        dimension=dimension,
                        status=ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS,
                        rationale_code=reason,
                    )
                    for dimension in self.dimensions
                ),
            )

        rule = matches[0]
        provenance = dict(rule.get("provenance", {}))
        source_ref = "#".join(filter(None, (
            str(provenance.get("source_asset", "")),
            str(provenance.get("source_rule", "")),
        )))
        rule_id = str(rule["coverage_rule_id"])
        required = set(rule.get("required_dimensions", ()))
        optional = set(rule.get("optional_dimensions", ()))
        not_applicable = set(rule.get("not_applicable_dimensions", ()))
        dimensions: list[ExposureDimensionCoverage] = []
        for dimension in self.dimensions:
            if dimension in required:
                status = ExposureDimensionRequirementStatus.REQUIRED
                rationale = "APPROVED_COVERAGE_RULE_REQUIRED"
            elif dimension in optional:
                status = ExposureDimensionRequirementStatus.OPTIONAL
                rationale = "APPROVED_COVERAGE_RULE_OPTIONAL"
            elif dimension in not_applicable:
                status = ExposureDimensionRequirementStatus.NOT_APPLICABLE
                rationale = "APPROVED_COVERAGE_RULE_NOT_APPLICABLE"
            else:
                status = ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS
                rationale = "DIMENSION_NOT_DECLARED_BY_APPROVED_COVERAGE_RULE"
            dimensions.append(ExposureDimensionCoverage(
                dimension=dimension,
                status=status,
                rule_id="" if status is ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS else rule_id,
                source_ref="" if status is ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS else source_ref,
                rationale_code=rationale,
            ))
        coverage_status = (
            ExposureDimensionCoverageStatus.PENDING_METHOD_SEMANTICS
            if any(
                item.status is ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS
                for item in dimensions
            ) else ExposureDimensionCoverageStatus.RESOLVED
        )
        return ExposureDimensionCoverageDecision(
            assessment_key=assessment_key,
            method_contract_hash=str(self.method.metadata.get("template_hash", "")),
            coverage_status=coverage_status,
            dimensions=tuple(dimensions),
            coverage_rule_ids=(rule_id,),
            granularity="FUNCTION_CONTEXT",
        )

    @staticmethod
    def readiness(
        decision: ExposureDimensionCoverageDecision,
        *,
        bindings: dict[str, dict[str, Any]],
        scenario_atom_ids: set[str],
        component_domain_resolved: bool,
    ) -> dict[str, Any]:
        """Evaluate availability after coverage is known; never infer coverage."""
        if decision.coverage_status is not ExposureDimensionCoverageStatus.RESOLVED:
            return {
                "exposure_input_status": "NOT_EVALUABLE",
                "blocking_reasons": ["PENDING_METHOD_COVERAGE_SEMANTICS"],
                "missing_required_dimensions": [],
            }
        missing = []
        for item in decision.dimensions:
            if item.status is not ExposureDimensionRequirementStatus.REQUIRED:
                continue
            binding = bindings.get(item.dimension, {})
            atom_id = str(
                binding.get("canonical_atom_id") or binding.get("atom_id") or ""
            ).strip()
            if (
                binding.get("resolution_status") != "RESOLVED"
                or not atom_id or atom_id not in scenario_atom_ids
            ):
                missing.append(item.dimension)
        if missing or not component_domain_resolved:
            return {
                "exposure_input_status": "BLOCKED",
                "blocking_reasons": (
                    (["REQUIRED_DIMENSION_MISSING_OR_UNRESOLVED"] if missing else [])
                    + (["PENDING_COMPONENT_DOMAIN"] if not component_domain_resolved else [])
                ),
                "missing_required_dimensions": missing,
            }
        return {
            "exposure_input_status": "READY",
            "blocking_reasons": [],
            "missing_required_dimensions": [],
        }


class ExposureDimensionCoverageAuditService:
    """Read-only audit.  It creates no atom bindings and computes no E value."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.coverage = ExposureDimensionCoverageService(method)
        self.domain_by_component = {
            category: rule.domain.value
            for rule in (method.structured_risk_method.exposure.domain_rules
                         if method.structured_risk_method else ())
            for category in rule.component_categories
        }

    def _authority_sources(self) -> list[dict[str, Any]]:
        sources = self.method.metadata.get("scenario_coverage_knowledge_sources", [])
        known = {
            str(item.get("source_asset", "")): dict(item)
            for item in sources if isinstance(item, dict)
        }

        def row(source: str, role: str, contains: bool, authority: bool, behavior: str) -> dict[str, Any]:
            item = known.get(source, {})
            return {
                "source": source,
                "role": role,
                "contains_coverage_semantics": contains,
                "normative_authority": authority,
                "consumer_behavior": behavior,
                "compiled_knowledge_classification": str(item.get("classification", "")),
            }

        return [
            row("raw/dimension_structure.yaml", "METHOD_DIMENSION_UNIVERSE", False, True,
                "Defines ontology, generation ordering and couplings; no required/not-applicable set."),
            row("raw/vda702_atoms.yaml", "ATOM_CATALOG", False, True,
                "Provides atom identities, dimensions and E_Z/E_F values only."),
            row("raw/atom_spec.yaml", "ATOM_PHYSICAL_SPEC", False, True,
                "Provides physical/range specifications; no coverage applicability."),
            row("raw/e_dimension_rules.yaml", "COMPONENT_TO_Z_F", False, True,
                "Selects E_Z or E_F by component category; does not select dimensions."),
            row("normalized/exposure_policy.yaml", "EXPOSURE_AGGREGATION", False, True,
                "Defines aggregation only; no dimension completeness policy."),
            row("raw/fm_scenario_templates.yaml", "SCENARIO_TEMPLATE_CONSTRAINT", False, True,
                "Constrains Scenario template matching; not an Exposure coverage rule."),
            row("raw/domain_rules/avp_low_speed.yaml", "DOMAIN_RULE", False, False,
                "Its numeric exposure section is excluded from active baseline authority."),
            row("normalized/scenario_coverage_rules.yaml", "COVERAGE_RULE_CONTAINER", False, False,
                "Compiled approved rule inventory is empty; no active coverage semantics."),
        ]

    def generate(self, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        functions = {
            str(item.get("function_id", "")): ExposureDimensionCoverageService.function_from_projection(item)
            for item in records.get("function", []) if isinstance(item, dict)
        }
        malfunctions = {
            str(item.get("malfunction_id", "")): item
            for item in records.get("malfunction", []) if isinstance(item, dict)
        }
        assessments = []
        for item in records.get("scenario_feasibility", []):
            if not isinstance(item, dict):
                continue
            if item.get("status") != "FINALIZED" and not evaluate_risk_eligibility_payload(item).eligible:
                continue
            if not all(item.get(flag) is True for flag in (
                "physically_feasible", "functionally_relevant", "causally_relevant",
            )):
                continue
            malfunction_id = str(item.get("malfunction_id", ""))
            scenario_id = str(item.get("scenario_id", ""))
            malfunction = malfunctions.get(malfunction_id, {})
            function_id = str(item.get("function_id") or malfunction.get("function_id", ""))
            decision = self.coverage.decide(
                assessment_key=f"{malfunction_id}::{scenario_id}",
                function=functions.get(function_id),
                operating_mode=str(item.get("operating_mode", "Active")),
            )
            assessments.append({
                "assessment_key": decision.assessment_key,
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "hazardous_event_id": str(item.get("hazardous_event_id", "")),
                "hazardous_event_present": bool(str(item.get("hazardous_event", "")).strip()),
                "component_category": str(malfunction.get("component_category", "")),
                "exposure_domain": self.domain_by_component.get(
                    str(malfunction.get("component_category", "")), "UNRESOLVED",
                ),
                "coverage_decision": decision.to_dict(),
                "required_dimensions": [
                    entry.dimension for entry in decision.dimensions
                    if entry.status is ExposureDimensionRequirementStatus.REQUIRED
                ],
                "not_applicable_dimensions": [
                    entry.dimension for entry in decision.dimensions
                    if entry.status is ExposureDimensionRequirementStatus.NOT_APPLICABLE
                ],
                "pending_dimensions": [
                    entry.dimension for entry in decision.dimensions
                    if entry.status is ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS
                ],
                "coverage_rule_ids": list(decision.coverage_rule_ids),
            })
        by_scenario: dict[str, set[str]] = defaultdict(set)
        for item in assessments:
            by_scenario[item["scenario_id"]].add(item["component_category"])
        summary = Counter(
            item["coverage_decision"]["coverage_status"] for item in assessments
        )
        governance = dict(self.method.metadata.get("scenario_coverage_governance", {}))
        return {
            "artifact_version": "exposure-dimension-coverage-audit-v1",
            "method_contract_hash": str(self.method.metadata.get("template_hash", "")),
            "runtime_yaml_read": 0,
            "method_dimension_universe": list(self.coverage.dimensions),
            "scenario_generation_dimensions": list(self.coverage.dimensions),
            "authority_sources": self._authority_sources(),
            "coverage_rule_inventory": {
                "approved_rule_count": self.coverage.approved_rule_count,
                "governance": governance,
                "rules": list(self.method.metadata.get("scenario_coverage_rules", [])),
                "absence_reason": "METHOD_COVERAGE_RULE_ABSENT" if not self.coverage.approved_rule_count else "",
            },
            "coverage_granularity": {
                "status": "UNRESOLVED",
                "cache_key_policy": "NO_COVERAGE_CACHE_UNTIL_METHOD_RULE_APPROVED",
                "assessment_record_key": "MALFUNCTION_ID::SCENARIO_ID",
                "shared_scenario_ids": sum(len(values) > 1 for values in by_scenario.values()),
                "shared_scenarios_with_multiple_component_categories": sum(
                    len(values) > 1 for values in by_scenario.values()
                ),
            },
            "z_f_relevance_analysis": {
                "domain_rule_count": len(
                    self.method.structured_risk_method.exposure.domain_rules
                ) if self.method.structured_risk_method else 0,
                "mapped_component_category_count": len(self.domain_by_component),
                "domain_rule_ids": [
                    item.rule_id for item in (
                        self.method.structured_risk_method.exposure.domain_rules
                        if self.method.structured_risk_method else ()
                    )
                ],
                "affects_required_dimension_set": False,
                "conclusion": "Z/F selects atom E_Z or E_F only; no confirmed source selects required dimensions.",
            },
            "historical_consumer_behavior": {
                "implementation": "ExposureMethodExecutor.lookup",
                "classification": "HISTORICAL_PARTIAL_INPUT_SCORING",
                "behavior": "Aggregates any supplied catalog atoms with usable selected-domain E values; it has no per-dimension completeness check.",
                "normative_authority": False,
                "promotion_status": "NOT_PROMOTED_TO_METHOD_COVERAGE_RULE",
            },
            "assessment_coverage_records": assessments,
            "summary": {
                "RESOLVED": summary["RESOLVED"],
                "PENDING_METHOD_SEMANTICS": summary["PENDING_METHOD_SEMANTICS"],
            },
            "outside_method_dimension_universe": ["slot_geometry"],
            "fallback_dimension_audit": {
                "source_dimensions": ["road_state", "surrounding"],
                "status": "TARGET_DIMENSION_PENDING",
            },
        }
