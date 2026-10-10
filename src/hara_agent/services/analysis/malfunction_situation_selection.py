"""Conservative preselection from compiled rules and explicit scoped facts."""

from dataclasses import asdict

from hara_agent.contracts import MethodContract
from hara_agent.models import (
    FactProvenance, FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate,
    ReviewStatus, ScenarioCandidate, SourceRef,
)

from .failure_mode_selector_resolver import FailureModeSelectorResolver


def selection_function(value) -> FunctionDefinition | None:
    """Decode the existing Function schema without inferring output semantics."""
    if isinstance(value, FunctionDefinition):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return FunctionDefinition(**{
            **value, "status": ReviewStatus(value.get("status", "PENDING")),
            "sources": [x if isinstance(x, SourceRef) else SourceRef(**x) for x in value.get("sources", [])],
        })
    except (TypeError, ValueError):
        return None


class MalfunctionSituationSelectionService:
    def __init__(self, method: MethodContract):
        self.method = method
        self.rules = method.situation_selection
        self.selector = FailureModeSelectorResolver(method)

    def _facts(self, malfunction, function, project_facts):
        """Accept exact approved bindings; never classify descriptions/guidewords."""
        scope = self.method.metadata.get("project_analysis_policy", {}).get("project_scope", "")
        if not scope or function is None or function.function_id != malfunction.function_id or not function.sources:
            return {}, "FUNCTION_OR_PROJECT_SCOPE_NOT_PROVEN"
        accepted = {}
        for fact in project_facts.risk_facts if project_facts else ():
            if (
                fact.approval is not ReviewStatus.FINALIZED
                or fact.provenance not in {FactProvenance.PROJECT_INPUT, FactProvenance.HUMAN_CONFIRMATION}
                or not fact.source_refs
                or set(fact.context) - {"project_scope", "function_id", "malfunction_id", "function_output"}
                or any(fact.context.get(k) != v for k, v in {
                    "project_scope": scope, "function_id": function.function_id,
                    "malfunction_id": malfunction.malfunction_id,
                }.items())
            ):
                continue
            if fact.parameter == "FUNCTIONAL_OUTPUT" and fact.context.get("function_output") != function.output:
                continue
            accepted.setdefault(fact.parameter, []).append(fact)
        return accepted, ""

    @staticmethod
    def _value(facts, key):
        values = {fact.value for fact in facts.get(key, ())}
        return next(iter(values)) if len(values) == 1 else None

    def _trigger_audit(self, malfunction, facts):
        structured = self.method.structured_risk_method
        category = self.selector.resolve(malfunction).canonical_component_category
        routes = [r for r in structured.exposure.domain_rules if category in r.component_categories] if structured else []
        trigger = self._value(facts, "VEHICLE_EFFECT_TRIGGER")
        return {
            "trigger_semantics_status": (
                "APPROVED_TRIGGER_FACT_PRESENT_METHOD_BINDING_REQUIRED" if trigger
                else "TRIGGER_SEMANTICS_NOT_PROVEN"
            ),
            "approved_trigger_fact_value": trigger,
            "runtime_route_status": "COMPONENT_HEURISTIC_USED" if len(routes) == 1 else "COMPONENT_ROUTE_UNRESOLVED",
            "component_category": category,
            "requested_domain": routes[0].domain.value if len(routes) == 1 else "",
            "component_rule_ids": [r.rule_id for r in routes],
            "reference_principles": [asdict(p) for p in self.rules.principles if p.trigger_semantics] if self.rules else [],
            "numeric_behavior_changed": False,
            "reason": "Reference trigger definitions are audit knowledge; no approved current-project trigger-to-Method binding is installed.",
        }

    def annotate_synthesis(self, malfunction, parent, function, project_context):
        """Read-only applicability annotation; never prune synthesis shortlists."""
        try:
            typed = MalfunctionCandidate(**{
                **malfunction, "status": ReviewStatus(malfunction.get("status", "PENDING")),
                "sources": [x if isinstance(x, SourceRef) else SourceRef(**x) for x in malfunction.get("sources", [])],
            })
            project = ItemDefinitionFacts.from_dict(project_context) if all(
                project_context.get(key) for key in ("system_description", "item_boundary", "sources")
            ) else None
            _, audit = self.select(typed, [parent], function=selection_function(function), project_facts=project)
            return {"mode": "READ_ONLY_SYNTHESIS_ANNOTATION", "applied_to_candidate_sets": False, **audit}
        except (TypeError, ValueError, KeyError):
            return {
                "mode": "READ_ONLY_SYNTHESIS_ANNOTATION", "applied_to_candidate_sets": False,
                "status": "UNRESOLVED", "reason": "TYPED_SELECTION_CONTEXT_NOT_AVAILABLE",
                "provider_calls_added_by_selection": 0,
            }

    def select(
        self, malfunction: MalfunctionCandidate, candidates: list[ScenarioCandidate],
        *, function: FunctionDefinition | None = None,
        project_facts: ItemDefinitionFacts | None = None,
    ) -> tuple[list[ScenarioCandidate], dict]:
        facts, context_error = self._facts(malfunction, function, project_facts)
        output = self._value(facts, "FUNCTIONAL_OUTPUT")
        scope = self._value(facts, "FUNCTION_OPERATING_SCOPE")
        failure = self.selector.resolve(malfunction).canonical_failure_type
        project_scope = self.method.metadata.get("project_analysis_policy", {}).get("project_scope", "")
        context_error = context_error or (
            "FUNCTION_OUTPUT_OR_OPERATING_SCOPE_NOT_PROVEN"
            if output is None or scope not in {"WITHIN_FUNCTION", "OUTSIDE_FUNCTION"} else ""
        )
        families = {x.family_id: x for x in self.rules.families} if self.rules else {}
        relevant = [r for r in self.rules.rules if (
            r.functional_output == output and failure in r.canonical_failure_types
            and r.function_scope == scope
        )] if self.rules and not context_error else []
        records, retained = [], []
        for candidate in candidates:
            include, exclude, diagnostics, basis = [], [], [], []
            raw_atoms = candidate.facts.get("scenario_atom_ids", ())
            atoms = set(raw_atoms) if isinstance(raw_atoms, (list, tuple)) and all(
                isinstance(x, str) for x in raw_atoms
            ) else set()
            for rule in relevant:
                if rule.action == "AUDIT":
                    diagnostics.append(
                        "SOURCE_BROAD_SCOPE_NOT_APPLICABLE" if any(
                            families[f].mapping_status == "SOURCE_SCOPE_REVIEW"
                            for f in rule.supported_families
                        ) else "SOURCE_EVIDENCE_NOT_AVAILABLE_FOR_EXECUTION"
                    )
                elif rule.action == "INCLUDE":
                    supported = set()
                    for family_id in rule.supported_families:
                        family = families[family_id]
                        if family.mapping_status == "MAPPED":
                            supported.update((family.source_id, family.candidate_canonical_id))
                        elif family.mapping_status == "SOURCE_FAMILY_ONLY":
                            supported.add(family.source_id)
                        else:
                            diagnostics.append(family.mapping_status)
                    if atoms & supported:
                        include.append(rule)
                elif (
                    rule.action == "EXCLUDE" and rule.runtime_exclusion_authority
                    and rule.authority == "APPROVED_PROJECT_CONSTRAINT"
                    and rule.approval == "APPROVED" and rule.project_scope == project_scope
                ):
                    expected = self._value(facts, rule.constraint_parameter)
                    actual = candidate.facts.get(rule.scenario_field)
                    metadata = candidate.fact_provenance.get(rule.scenario_field, {})
                    metadata = metadata if isinstance(metadata, dict) else {}
                    applicable = metadata.get("applicable_scope", {})
                    applicable = applicable if isinstance(applicable, dict) else {}
                    if (
                        expected is not None and actual not in (None, "")
                        and metadata.get("approval") == "FINALIZED"
                        and metadata.get("provenance") in {"PROJECT_INPUT", "HUMAN_CONFIRMATION"}
                        and metadata.get("source_refs")
                        and applicable.get("malfunction_id") == malfunction.malfunction_id
                        and applicable.get("scenario_id") == candidate.scenario_id
                        and applicable.get("project_scope") == project_scope
                        and applicable.get("function_id") == malfunction.function_id
                    ):
                        if actual != expected:
                            exclude.append(rule)
                            basis.append({
                                "rule_id": rule.rule_id, "scenario_field": rule.scenario_field,
                                "actual": actual, "required": expected,
                                "project_constraint_facts": [asdict(f) for f in facts[rule.constraint_parameter]],
                                "candidate_fact_provenance": metadata,
                            })
                    else:
                        diagnostics.append("EXCLUSION_EVIDENCE_INCOMPLETE_OR_CONFLICTING")
            conflict = (include and exclude) or (
                exclude and "EXCLUSION_EVIDENCE_INCOMPLETE_OR_CONFLICTING" in diagnostics
            ) or len(exclude) > 1
            status = "UNRESOLVED" if conflict else "EXCLUDE" if exclude else "INCLUDE" if include else "UNRESOLVED"
            reason = (
                "COMPETING_RULES_UNRESOLVED" if conflict else
                "APPROVED_PROJECT_CONSTRAINT_CONTRADICTION" if exclude else
                "SOURCE_GUIDANCE_SUPPORTED" if include else
                "NO_SELECTION_CATALOG" if self.rules is None else
                context_error or "NO_APPLICABLE_POSITIVE_RULE; CONTINUE_CAUSAL"
            )
            matched = include + exclude
            records.append({
                "parent_scenario_id": candidate.scenario_id, "status": status, "reason": reason,
                "rule_ids": [r.rule_id for r in matched],
                "reference_rule_ids": [r.rule_id for r in relevant],
                "references": [asdict(ref) for r in relevant for ref in r.source_refs],
                "matched_fields": {"functional_output": output, "canonical_failure_type": failure, "function_scope": scope},
                "actual_scope": {"project_scope": project_scope, "function_id": malfunction.function_id, "malfunction_id": malfunction.malfunction_id},
                "project_evidence": {key: [asdict(f) for f in values] for key, values in facts.items()},
                "source_version": [asdict(families[f]) for r in relevant for f in r.supported_families],
                "diagnostic": sorted(set(diagnostics)), "exclusion_evidence": basis,
            })
            if status != "EXCLUDE":
                retained.append(candidate)
        return retained, {
            "malfunction_id": malfunction.malfunction_id,
            "method_contract_hash": self.method.metadata.get("method_source_hash", ""),
            "base_candidate_count": len(candidates),
            "include_count": sum(r["status"] == "INCLUDE" for r in records),
            "exclude_count": sum(r["status"] == "EXCLUDE" for r in records),
            "unresolved_passthrough_count": sum(r["status"] == "UNRESOLVED" for r in records),
            "scenarios_after_selection": len(retained),
            "exclude_rule_breakdown": [r for r in records if r["status"] == "EXCLUDE"],
            "provider_calls_added_by_selection": 0,
            "records": records, "exposure_trigger_audit": self._trigger_audit(malfunction, facts),
        }
