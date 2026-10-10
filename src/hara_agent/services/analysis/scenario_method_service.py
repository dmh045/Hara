"""Typed consumers for confirmed Scenario method knowledge.

The service instantiates source-traceable analytical Scenario candidates. It
does not emit causal evidence or change S/E/C, ASIL, or FTTI evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

from hara_agent.contracts import FMScenarioTemplate, MethodContract
from hara_agent.models import (
    FactProvenance, FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, ScenarioCandidate,
    SourceRef,
)
from hara_agent.services.semantic.scenario_contract import SCENARIO_CONTRACT_VERSION

from .failure_mode_selector_resolver import (
    FMTemplateSelectorAdapterResolver, FailureModeSelectorResolution,
    FailureModeSelectorResolver, TemplateSelectorResolution,
)
from .driver_configuration_service import DriverConfigurationBrancher
from .risk_vocabulary_adapter import RiskVocabularyAdapter
from .malfunction_situation_selection import MalfunctionSituationSelectionService


@dataclass(frozen=True)
class FMTemplateMatchResult:
    status: str
    template: FMScenarioTemplate | None
    matching_template_ids: tuple[str, ...]
    original_precedence_template_id: str
    reason: str
    matched_by: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    template_selector_resolution: tuple[TemplateSelectorResolution, ...] = ()
    selector_resolution: FailureModeSelectorResolution | None = None
    qualification_tier: str = ""
    decision_source_ref: dict[str, Any] | None = None

    @property
    def injectable(self) -> bool:
        return self.status == "STRONG_MATCH" and self.template is not None


class ScenarioMethodService:
    """Resolve explicitly compiled Scenario method context without YAML access."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.selector_resolver = FailureModeSelectorResolver(method)
        self.template_selector_resolver = FMTemplateSelectorAdapterResolver(method)
        self.risk_vocabulary = RiskVocabularyAdapter(method)
        policy = method.metadata.get("project_analysis_policy", {})
        policy = policy if isinstance(policy, dict) else {}
        decisions = policy.get("decisions", {})
        decisions = decisions if isinstance(decisions, dict) else {}
        a5 = decisions.get("A5", {})
        self.driver_brancher = DriverConfigurationBrancher(
            policy_id=(
                str(policy.get("policy_id", ""))
                if isinstance(a5, dict)
                and a5.get("status") == "CONFIRMED_FOR_CURRENT_PROJECT"
                else ""
            ),
        )

    def _with_driver_configurations(
        self, malfunction: MalfunctionCandidate,
        candidates: list[ScenarioCandidate], audit: dict[str, Any],
        project_facts: ItemDefinitionFacts | None,
    ) -> tuple[list[ScenarioCandidate], dict[str, Any]]:
        selection = audit.get("situation_selection", {})
        selection["post_template_options_count"] = len(candidates)
        selection["post_driver_branch_count"] = len(candidates)
        if project_facts is None or not self.driver_brancher.policy_id:
            return candidates, audit
        risk_facts = [
            {**asdict(fact), "approval": fact.approval.value}
            for fact in project_facts.risk_facts
        ]
        if not any(
            item.get("parameter") == "DRIVER_IN_VEHICLE"
            and item.get("approval") == "FINALIZED"
            and self.driver_brancher.allowed_position(item)[0]
            for item in risk_facts
        ):
            return candidates, audit
        expanded = []
        by_parent: dict[str, list[str]] = {}
        for candidate in candidates:
            instance = {
                **candidate.analysis_instance,
                "malfunction_id": malfunction.malfunction_id,
            }
            candidate = replace(candidate, analysis_instance=instance)
            branches = self.driver_brancher.expand(candidate, risk_facts)
            expanded.extend(branches)
            by_parent[candidate.scenario_id] = [item.scenario_id for item in branches]
        audit["driver_configuration"] = {
            "policy_id": self.driver_brancher.policy_id,
            "pre_branch_count": len(candidates),
            "post_branch_count": len(expanded),
            "branch_count": sum(
                bool(item.analysis_instance.get("driver_configuration_branch"))
                for item in expanded
            ),
            "scenario_ids_by_parent": by_parent,
        }
        audit["instance_count"] = len(expanded)
        selection["post_driver_branch_count"] = len(expanded)
        for option in audit.get("options", []):
            if isinstance(option, dict) and option.get("scenario_id") in by_parent:
                option["driver_branch_scenario_ids"] = by_parent[option["scenario_id"]]
        return expanded, audit

    def _selector_template_ids(
        self, selector_type: str, canonical_selector: str,
    ) -> tuple[str, ...]:
        """Return all source templates supported by one canonical selector.

        This is qualification-only metadata.  Candidate discovery remains the
        original keyword OR component OR failure selector relation.
        """
        if not canonical_selector:
            return ()
        catalog = self.method.scenario_model.scenario_method.fm_template_catalog
        if catalog is None:
            return ()
        identifiers: list[str] = []
        for template in catalog.templates:
            raw_values = (
                template.match.component_categories
                if selector_type == "COMPONENT_CATEGORY"
                else template.match.failure_types
            )
            if any(
                self.template_selector_resolver.resolve(
                    selector_type, raw,
                ).canonical_template_selector == canonical_selector
                for raw in raw_values
            ):
                identifiers.append(template.template_id)
        return tuple(identifiers)

    def _qualify_matches(
        self,
        matches: list[tuple[
            FMScenarioTemplate, tuple[str, ...], tuple[str, ...],
            tuple[TemplateSelectorResolution, ...],
        ]],
        *, component: str, failure_type: str,
    ) -> tuple[str, int | None, str]:
        """Select only source-supported unique template evidence.

        A structured component/failure disagreement is not resolved by tier;
        it remains fail-closed.  Keyword evidence is discovery-only and never
        lowers a unique structured qualification.
        """
        component_hits = [
            index for index, item in enumerate(matches)
            if "COMPONENT_CATEGORY" in item[1]
        ]
        failure_hits = [
            index for index, item in enumerate(matches)
            if "FAILURE_TYPE" in item[1]
        ]
        both_hits = [
            index for index in component_hits if index in set(failure_hits)
        ]
        if len(both_hits) == 1:
            return "STRONG_MATCH", both_hits[0], "TIER_1_COMPONENT_AND_FAILURE"
        if len(both_hits) > 1:
            return "AMBIGUOUS", None, "TIER_1_COMPONENT_AND_FAILURE"
        if component_hits and failure_hits and set(component_hits) != set(failure_hits):
            return "AMBIGUOUS", None, "STRUCTURED_SELECTOR_CONFLICT"
        component_templates = self._selector_template_ids(
            "COMPONENT_CATEGORY", component,
        )
        if len(component_templates) == 1 and len(component_hits) == 1:
            return "STRONG_MATCH", component_hits[0], "TIER_2_UNIQUE_COMPONENT"
        failure_templates = self._selector_template_ids("FAILURE_TYPE", failure_type)
        if len(failure_templates) == 1 and len(failure_hits) == 1:
            return "STRONG_MATCH", failure_hits[0], "TIER_3_UNIQUE_FAILURE_TYPE"
        keyword_only = [
            index for index, item in enumerate(matches)
            if item[1] == ("KEYWORD",)
        ]
        if keyword_only and not component_hits and not failure_hits and len(keyword_only) == 1:
            return "WEAK_MATCH", keyword_only[0], "TIER_4_KEYWORD_ONLY"
        return "AMBIGUOUS", None, "UNRESOLVED_SOURCE_SUPPORT"

    def match_fm_template(self, malfunction: MalfunctionCandidate) -> FMTemplateMatchResult:
        selector_resolution = self.selector_resolver.resolve(malfunction)
        catalog = self.method.scenario_model.scenario_method.fm_template_catalog
        if catalog is None:
            return FMTemplateMatchResult(
                "NO_MATCH", None, (), "", "NO_TEMPLATE_CATALOG",
                selector_resolution=selector_resolution,
            )
        policy = self.method.metadata.get("project_analysis_policy", {})
        policy = policy if isinstance(policy, dict) else {}
        governed = policy.get("engineering_decisions", {})
        governed = governed if isinstance(governed, dict) else {}
        exact_fields = {
            "malfunction_id": malfunction.malfunction_id,
            "function_id": malfunction.function_id,
            "guideword_id": malfunction.guideword_id,
            "description": malfunction.description,
            "functional_effect": malfunction.functional_effect,
            "vehicle_level_hazard": malfunction.vehicle_level_hazard,
            "component_category": selector_resolution.canonical_component_category,
            "failure_type": selector_resolution.canonical_failure_type,
        }
        scoped = [
            (decision_id, decision) for decision_id, decision in governed.items()
            if isinstance(decision, dict)
            and decision.get("status") == "CONFIRMED_FOR_CURRENT_PROJECT"
            and decision.get("approved_template_id")
            and decision.get("malfunction_id") == malfunction.malfunction_id
        ]
        if scoped:
            approved = [
                (decision_id, decision) for decision_id, decision in scoped
                if decision.get("effective_project_scope") == policy.get("project_scope")
                and all(decision.get(key) == value for key, value in exact_fields.items())
            ]
            if len(approved) != 1 or len(scoped) != 1:
                return FMTemplateMatchResult(
                    "AMBIGUOUS", None, (), "",
                    "GOVERNED_DECISION_SCOPE_MISMATCH",
                    selector_resolution=selector_resolution,
                    qualification_tier="GOVERNED_PROJECT_DECISION_SCOPE_MISMATCH",
                )
            decision_id, decision = approved[0]
            template_id = str(decision["approved_template_id"])
            matches = [
                item for item in catalog.templates
                if item.template_id == template_id
                and item.source_role == "SCENARIO_TEMPLATE_CONSTRAINT"
            ]
            option_ids = [
                f"{template_id}:OPTION:{index}"
                for index in range(1, len(matches[0].required_scenarios) + 1)
            ] if len(matches) == 1 else []
            if (
                len(matches) != 1
                or decision.get("approved_option_ids") != option_ids
                or not option_ids
            ):
                return FMTemplateMatchResult(
                    "AMBIGUOUS", None, (), "",
                    "GOVERNED_TEMPLATE_OR_OPTION_MISMATCH",
                    selector_resolution=selector_resolution,
                    qualification_tier="GOVERNED_PROJECT_DECISION_INVALID",
                )
            source_ref = policy.get("source_ref", {})
            source_ref = source_ref if isinstance(source_ref, dict) else {}
            return FMTemplateMatchResult(
                "STRONG_MATCH", matches[0], (template_id,), template_id,
                "GOVERNED_PROJECT_DECISION",
                matched_by=("ENGINEERING_DECISION",),
                matched_terms=(decision_id,),
                selector_resolution=selector_resolution,
                qualification_tier="GOVERNED_EXACT_MALFUNCTION_CLASS",
                decision_source_ref={
                    **source_ref,
                    "location": (
                        f"{source_ref.get('location', '')}:"
                        f"engineering_decisions.{decision_id}"
                    ),
                },
            )
        fm_text = " ".join((
            malfunction.description, malfunction.functional_effect,
            malfunction.vehicle_level_hazard, malfunction.guideword,
        )).casefold()
        component = selector_resolution.canonical_component_category
        failure_type = selector_resolution.canonical_failure_type
        matches: list[tuple[
            FMScenarioTemplate,
            tuple[str, ...],
            tuple[str, ...],
            tuple[TemplateSelectorResolution, ...],
        ]] = []
        for template in catalog.templates:
            match = template.match
            keyword_terms = tuple(
                keyword for keyword in match.keywords if keyword.casefold() in fm_text
            )
            component_resolutions = tuple(
                self.template_selector_resolver.resolve("COMPONENT_CATEGORY", value)
                for value in match.component_categories
            )
            failure_type_resolutions = tuple(
                self.template_selector_resolver.resolve("FAILURE_TYPE", value)
                for value in match.failure_types
            )
            matched_component_selectors = tuple(
                item for item in component_resolutions
                if component and item.canonical_template_selector == component
            )
            matched_failure_selectors = tuple(
                item for item in failure_type_resolutions
                if failure_type and item.canonical_template_selector == failure_type
            )
            component_terms = tuple(
                item.raw_template_selector for item in matched_component_selectors
            )
            failure_type_terms = tuple(
                item.raw_template_selector for item in matched_failure_selectors
            )
            matched_by = tuple(name for name, terms in (
                ("KEYWORD", keyword_terms),
                ("COMPONENT_CATEGORY", component_terms),
                ("FAILURE_TYPE", failure_type_terms),
            ) if terms)
            if matched_by:
                matches.append((
                    template, matched_by,
                    keyword_terms + component_terms + failure_type_terms,
                    matched_component_selectors + matched_failure_selectors,
                ))
        if not matches:
            return FMTemplateMatchResult(
                "NO_MATCH", None, (), "", "NO_SOURCE_DEFINED_MATCH",
                selector_resolution=selector_resolution,
            )
        original_first = min(matches, key=lambda item: item[0].original_precedence)[0]
        identifiers = tuple(item[0].template_id for item in matches)
        qualification, selected_index, qualification_tier = self._qualify_matches(
            matches, component=component, failure_type=failure_type,
        )
        if selected_index is None:
            return FMTemplateMatchResult(
                "AMBIGUOUS", None, identifiers, original_first.template_id,
                (
                    "MULTIPLE_SOURCE_DEFINED_TEMPLATES_MATCH; original first-match "
                    "precedence recorded but not consumed"
                ),
                selector_resolution=selector_resolution,
                qualification_tier=qualification_tier,
            )
        template, matched_by, matched_terms, template_selector_resolution = matches[selected_index]
        return FMTemplateMatchResult(
            qualification, template, identifiers, original_first.template_id,
            (
                "STRUCTURED_SELECTOR_MATCH"
                if qualification == "STRONG_MATCH" else "WEAK_KEYWORD_ONLY"
            ),
            matched_by=matched_by, matched_terms=matched_terms,
            template_selector_resolution=template_selector_resolution,
            selector_resolution=selector_resolution,
            qualification_tier=qualification_tier,
        )

    @staticmethod
    def _same_value(current: Any, expected: Any) -> bool:
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return (
                isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and float(current) == float(expected)
            )
        return str(current).strip().casefold() == str(expected).strip().casefold()

    def _template_source(self, option: Any) -> SourceRef:
        source = option.source_ref
        return SourceRef(
            "method_contract",
            str(self.method.metadata.get("method_source_hash", source.source_hash)),
            f"{source.sheet}!{source.range}",
            source.raw_text,
        )

    @staticmethod
    def _speed_constraint_text(scenario: ScenarioCandidate) -> str:
        speed = scenario.facts.get("ego_speed_kph")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool):
            return f"{float(speed):g} km/h"
        constraint = scenario.facts.get("ego_speed_constraint")
        if isinstance(constraint, dict):
            lower, upper = constraint.get("min_kph"), constraint.get("max_kph")
            if lower is not None or upper is not None:
                return f"{lower if lower is not None else '-∞'}..{upper if upper is not None else '+∞'} km/h"
        return "未提供"

    def instantiate_analytical_candidates(
        self, malfunction: MalfunctionCandidate, candidates: list[ScenarioCandidate],
        project_facts: ItemDefinitionFacts | None = None,
        *, function: FunctionDefinition | None = None,
    ) -> tuple[list[ScenarioCandidate], dict[str, Any]]:
        """Create isolated M×template-option scenarios from strong matches only."""
        candidates, selection = MalfunctionSituationSelectionService(self.method).select(
            malfunction, candidates, function=function, project_facts=project_facts,
        )
        result = self.match_fm_template(malfunction)
        audit: dict[str, Any] = {
            "malfunction_id": malfunction.malfunction_id,
            "qualification": result.status,
            "template_id": result.template.template_id if result.template else "",
            "matched_by": list(result.matched_by),
            "matched_terms": list(result.matched_terms),
            "selection_basis": result.reason,
            "base_candidate_count": selection["base_candidate_count"],
            "situation_selection": selection,
            "instance_count": 0,
            "options": [],
        }
        if not result.injectable:
            audit["selection_mode"] = "BASE_CANDIDATES_ONLY"
            return self._with_driver_configurations(
                malfunction, list(candidates), audit, project_facts,
            )

        assert result.template is not None
        instances: list[ScenarioCandidate] = []
        method_hash = str(self.method.metadata.get("method_source_hash", ""))
        for parent in candidates:
            for option_index, option in enumerate(result.template.required_scenarios, start=1):
                option_id = f"{result.template.template_id}:OPTION:{option_index}"
                source = self._template_source(option)
                source_ref = {
                    "asset": option.source_ref.workbook,
                    "location": f"{option.source_ref.sheet}!{option.source_ref.range}",
                    "source_hash": option.source_ref.source_hash,
                }
                values: dict[str, Any] = {
                    "object_type": option.obj_type,
                    "object_position": option.obj_position,
                    "relative_distance_m": option.obj_distance_m,
                    "object_speed_kph": option.obj_v_kph,
                }
                diagnostics: list[dict[str, Any]] = []
                if not option.obj_type.strip() or not option.obj_position.strip():
                    diagnostics.append({"code": "INVALID_TEMPLATE_OPTION", "field": "object"})
                for field in ("relative_distance_m", "object_speed_kph"):
                    value = values[field]
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                        diagnostics.append({"code": "INVALID_TEMPLATE_OPTION", "field": field})
                vocabulary_mappings = []
                for field, source_value in (
                    ("road_user_type", option.obj_type),
                    ("collision_type", option.collision_type),
                ):
                    resolution = self.risk_vocabulary.resolve(
                        field=field, raw_value=source_value,
                    )
                    if resolution.mapped:
                        values[field] = resolution.canonical_value
                        vocabulary_mappings.append({
                            "field": field,
                            **resolution.to_dict(),
                        })
                    else:
                        diagnostics.append({
                            "code": "UNMAPPED_METHOD_VALUE",
                            "field": field,
                            "source_value": source_value,
                        })
                conflicts = [
                    {
                        "field": field,
                        "scenario_value": parent.facts[field],
                        "template_value": value,
                    }
                    for field, value in values.items()
                    if field in parent.facts
                    and parent.facts[field] not in (None, "")
                    and not self._same_value(parent.facts[field], value)
                ]
                option_audit = {
                    "parent_scenario_id": parent.scenario_id,
                    "source_option_id": option_id,
                    "source_ref": source_ref,
                    "diagnostics": diagnostics,
                    "conflicts": conflicts,
                }
                if any(item["code"] == "INVALID_TEMPLATE_OPTION" for item in diagnostics):
                    option_audit["validation_status"] = "INVALID"
                    audit["options"].append(option_audit)
                    continue
                if conflicts:
                    option_audit["validation_status"] = "CONFLICT"
                    audit["options"].append(option_audit)
                    continue

                material = {
                    "contract": SCENARIO_CONTRACT_VERSION,
                    "method_contract_hash": method_hash,
                    "malfunction_id": malfunction.malfunction_id,
                    "parent_scenario_id": parent.scenario_id,
                    "parent_fingerprint": parent.semantic_fingerprint,
                    "template_id": result.template.template_id,
                    "source_option_id": option_id,
                    "source_hash": option.source_ref.source_hash,
                    "facts": values,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                instance_id = f"SCN-ANALYTICAL-{fingerprint[:16].upper()}"
                scope = {
                    "malfunction_id": malfunction.malfunction_id,
                    "scenario_id": instance_id,
                    "parent_scenario_id": parent.scenario_id,
                }
                assumptions = []
                provenance = dict(parent.fact_provenance)
                for field, value in values.items():
                    assumption = {
                        "field": field,
                        "value": value,
                        "unit": "m" if field == "relative_distance_m" else "km/h" if field == "object_speed_kph" else "",
                        "value_kind": "point" if isinstance(value, (int, float)) else "category",
                        "origin": FactProvenance.SCENARIO_DEFINED.value,
                        "source_template_id": result.template.template_id,
                        "source_option_id": option_id,
                        "source_ref": source_ref,
                        "source_hash": option.source_ref.source_hash,
                        "method_contract_hash": method_hash,
                        "selection_basis": result.qualification_tier,
                        "validation_status": "VALIDATED",
                        "applicable_scope": scope,
                    }
                    mapping = next(
                        (
                            item for item in vocabulary_mappings
                            if item["field"] == field
                        ),
                        None,
                    )
                    if mapping is not None:
                        assumption["risk_vocabulary_mapping"] = mapping
                    assumptions.append(assumption)
                    provenance[field] = {
                        "provenance": FactProvenance.SCENARIO_DEFINED.value,
                        "approval": ReviewStatus.PENDING.value,
                        "source_refs": [{
                            "source_type": source.source_type,
                            "source_id": source.source_id,
                            "location": source.location,
                            "excerpt": source.excerpt,
                        }],
                        **{key: item for key, item in assumption.items() if key not in {"field", "value"}},
                    }
                setting = (
                    f"对象={option.obj_type}，位置={option.obj_position}，"
                    f"间距={float(option.obj_distance_m):g} m，对象速度={float(option.obj_v_kph):g} km/h"
                )
                detail = (
                    f"项目速度约束：{self._speed_constraint_text(parent)}；"
                    f"本分析场景设定：{setting}"
                )
                instance = {
                    "instance_id": instance_id,
                    "parent_scenario_id": parent.scenario_id,
                    "malfunction_id": malfunction.malfunction_id,
                    "origin": FactProvenance.SCENARIO_DEFINED.value,
                    "source_template_id": result.template.template_id,
                    "source_option_id": option_id,
                    "source_ref": source_ref,
                    "source_hash": option.source_ref.source_hash,
                    "source_option_values": {
                        "obj_type": option.obj_type,
                        "obj_position": option.obj_position,
                        "obj_distance_m": option.obj_distance_m,
                        "obj_v_kph": option.obj_v_kph,
                        "collision_type": option.collision_type,
                    },
                    "method_contract_hash": method_hash,
                    "selection_basis": result.qualification_tier,
                    "validation_status": "VALIDATED",
                    "applicable_scope": scope,
                    "assumptions": assumptions,
                    "risk_vocabulary_mappings": vocabulary_mappings,
                    "diagnostics": diagnostics,
                }
                instances.append(replace(
                    parent,
                    scenario_id=instance_id,
                    situational_description=detail,
                    situational_detailing=detail,
                    facts={**parent.facts, **values},
                    context_resolution={
                        **parent.context_resolution,
                        "analytical_scenario_instantiation": {
                            "template_id": result.template.template_id,
                            "source_option_id": option_id,
                            "validation_status": "VALIDATED",
                        },
                    },
                    fact_provenance=provenance,
                    status=ReviewStatus.PENDING,
                    sources=list(dict.fromkeys([*parent.sources, source])),
                    review_reason="本分析场景设定为受控模板选项，尚未构成项目事实或工程发布确认。",
                    source_scenario_id=parent.scenario_id,
                    atomic_variant="analytical_template_option",
                    semantic_fingerprint=fingerprint,
                    scenario_contract_version=SCENARIO_CONTRACT_VERSION,
                    analysis_instance=instance,
                ))
                option_audit["validation_status"] = "VALIDATED"
                option_audit["scenario_id"] = instance_id
                audit["options"].append(option_audit)
        audit["selection_mode"] = "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
        audit["instance_count"] = len(instances)
        audit["unmapped_value_count"] = sum(
            sum(item.get("code") == "UNMAPPED_METHOD_VALUE" for item in option["diagnostics"])
            for option in audit["options"]
        )
        audit["conflict_count"] = sum(bool(option["conflicts"]) for option in audit["options"])
        return self._with_driver_configurations(
            malfunction, instances, audit, project_facts,
        )

    def fallback_terms(self) -> tuple[dict[str, object], ...]:
        """Expose unmapped fallback terms for audit; never guess a target dimension."""
        domain = self.method.scenario_model.scenario_method.domain_knowledge
        if domain is None:
            return ()
        return tuple({
            "source_dimension": item.source_dimension,
            "terms": list(item.terms),
            "target_dimension": item.target_dimension,
            "target_status": item.target_status,
            "source_role": item.source_role,
            "source_ref": {
                "asset": item.source_ref.workbook,
                "hash": item.source_ref.source_hash,
                "location": item.source_ref.range,
            },
        } for item in domain.fallback_dimensions)

    def fallback_context_for_missing_project_fact(
        self, *, source_dimension: str, explicit_project_value: str,
    ) -> dict[str, object]:
        """Return declared fallback only for an absent fact and never map it by guesswork."""
        if explicit_project_value.strip():
            return {
                "status": "EXPLICIT_PROJECT_FACT_PRESENT",
                "source_dimension": source_dimension,
                "terms": [],
            }
        for item in self.fallback_terms():
            if item["source_dimension"] == source_dimension:
                return {"status": "TARGET_DIMENSION_PENDING", **item}
        return {"status": "NO_CONFIRMED_FALLBACK", "source_dimension": source_dimension, "terms": []}
