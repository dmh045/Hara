from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, replace
from collections.abc import Callable
from typing import Any

from hara_agent.contracts import MethodContract, ScenarioDimension
from hara_agent.models import (
    FactProvenance,
    FunctionDefinition,
    MalfunctionCandidate,
    ItemDefinitionFacts,
    ReviewStatus,
    ScenarioCandidate,
    SourceRef,
)
from hara_agent.services.analysis.project_fact_resolver import SpeedResolutionResult
from hara_agent.services.analysis.scenario_constraint_service import (
    ScenarioConstraintExecutor, ScenarioConstraintStatus,
)
from hara_agent.services.analysis.method_atom_resolver import MethodAtomResolver
from hara_agent.services.analysis.scenario_coverage_service import ScenarioCoverageRuleService
from .method_risk_fact_service import MethodRiskFactBindingService
from hara_agent.services.semantic.scenario_contract import SCENARIO_CONTRACT_VERSION


class MethodScenarioCandidateService:
    """Bind grounded ProjectFacts to MethodContract scenario dimensions."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.risk_facts = MethodRiskFactBindingService(method)
        self.constraints = ScenarioConstraintExecutor()
        self.atom_resolver = MethodAtomResolver(method)
        self.coverage_rules = ScenarioCoverageRuleService(method)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))

    def _text_binding(
        self, dimension: ScenarioDimension, project_value: str
    ) -> dict[str, Any]:
        resolution = self.atom_resolver.resolve(
            project_value, dimension.canonical_name,
        )
        if not self.atom_resolver.catalog and resolution.status == "PENDING":
            matches = [
                item for item in dimension.values
                if self._normalize(project_value) == self._normalize(item)
            ]
            if len(matches) == 1:
                return {
                    "project_value": project_value,
                    "method_value": matches[0],
                    "binding_status": "EXACT",
                    "resolution_status": "RESOLVED",
                    "unresolved_reason": "",
                    "candidate_atom_ids": [],
                    "atom_id": "",
                    "canonical_atom_id": "",
                    "filled_dimensions": [dimension.canonical_name],
                    "atom_provenance": {},
                    "dimension_source": f"{dimension.source_ref.sheet}!{dimension.source_ref.range}",
                }
        legacy_status = {
            "RESOLVED": "EXACT",
            "AMBIGUOUS": "AMBIGUOUS",
        }.get(resolution.status, "UNRESOLVED")
        atom = self.atom_resolver.by_id.get(resolution.atom_id, {})
        return {
            "project_value": project_value,
            "method_value": (
                f"{resolution.atom_id} | {atom.get('label', '')}"
                if resolution.status == "RESOLVED" else ""
            ),
            "binding_status": legacy_status,
            "resolution_status": resolution.status,
            "unresolved_reason": "" if resolution.status == "RESOLVED" else resolution.reason,
            "candidate_atom_ids": list(resolution.candidate_atom_ids),
            "atom_id": resolution.atom_id,
            "canonical_atom_id": resolution.canonical_atom_id,
            "filled_dimensions": list(resolution.filled_dimensions),
            "atom_provenance": resolution.provenance or {},
            "dimension_source": (
                f"{dimension.source_ref.sheet}!{dimension.source_ref.range}"
            ),
        }

    def _speed_binding(
        self, dimension: ScenarioDimension, speed_kph: float
    ) -> dict[str, str]:
        matches = [
            value for value in dimension.values
            if self._speed_value_matches(value, speed_kph)
        ]
        legacy_status = (
            "EXACT" if len(matches) == 1
            else "AMBIGUOUS" if len(matches) > 1
            else "UNRESOLVED"
        )
        return {
            "project_value": f"{speed_kph:g} km/h",
            "method_value": matches[0] if len(matches) == 1 else "",
            "binding_status": legacy_status,
            "resolution_status": (
                "RESOLVED" if legacy_status == "EXACT" else "PENDING"
            ),
            "unresolved_reason": (
                "" if legacy_status == "EXACT"
                else "AMBIGUOUS_BINDING" if legacy_status == "AMBIGUOUS"
                else "NO_COMPATIBLE_METHOD_ATOM"
            ),
            "dimension_source": (
                f"{dimension.source_ref.sheet}!{dimension.source_ref.range}"
            ),
        }

    @staticmethod
    def _pending_binding(
        dimension: ScenarioDimension, *, reason: str,
        project_value: str = "", candidate_values: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "project_value": project_value,
            "method_value": "",
            "binding_status": "MISSING" if not project_value else "UNRESOLVED",
            "resolution_status": "PENDING",
            "unresolved_reason": reason,
            "candidate_values": list(candidate_values or []),
            "dimension_source": f"{dimension.source_ref.sheet}!{dimension.source_ref.range}",
        }

    def _speed_constraint_binding(
        self, dimension: ScenarioDimension, speed: SpeedResolutionResult,
    ) -> dict[str, Any]:
        if speed.resolution_source == "ExplicitProjectInput":
            return self._speed_binding(dimension, speed.resolved_value)
        envelope = dict(speed.matched_envelope)
        minimum = envelope.get("speed_min_kph")
        maximum = envelope.get("speed_max_kph")
        text = f"{minimum if minimum is not None else '-∞'}..{maximum if maximum is not None else '+∞'} km/h"
        resolution = self.atom_resolver.resolve_speed_range(
            minimum_kph=minimum,
            maximum_kph=maximum,
            dimension=dimension.canonical_name,
        )
        atom = self.atom_resolver.by_id.get(resolution.atom_id, {})
        result = {
            "project_value": text,
            "method_value": (
                f"{resolution.atom_id} | {atom.get('label', '')}"
                if resolution.status == "RESOLVED" else ""
            ),
            "binding_status": (
                "EXACT" if resolution.status == "RESOLVED"
                else "AMBIGUOUS" if resolution.status == "AMBIGUOUS"
                else "UNRESOLVED"
            ),
            "resolution_status": resolution.status,
            "unresolved_reason": (
                "" if resolution.status == "RESOLVED" else resolution.reason
            ),
            "candidate_atom_ids": list(resolution.candidate_atom_ids),
            "atom_id": resolution.atom_id,
            "canonical_atom_id": resolution.canonical_atom_id,
            "filled_dimensions": list(resolution.filled_dimensions),
            "atom_provenance": resolution.provenance or {},
            "dimension_source": (
                f"{dimension.source_ref.sheet}!{dimension.source_ref.range}"
            ),
        }
        result["speed_constraint"] = {
            "min_kph": minimum,
            "max_kph": maximum,
            "unit": envelope.get("unit", "km/h"),
            "condition": envelope.get("condition", ""),
        }
        if resolution.status == "RESOLVED":
            result["speed_constraint"]["resolved_by"] = resolution.reason
        return result

    @staticmethod
    def _speed_value_matches(raw: str, speed_kph: float) -> bool:
        text = raw.casefold().replace("≤", "<=").replace("≥", ">=")
        compact = re.sub(r"\s+", "", text)
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", compact)]
        if not numbers:
            return False
        if len(numbers) == 1 and ("standstill" in text or "静止" in text):
            return speed_kph == numbers[0]
        if len(numbers) < 2:
            return False
        lower, upper = numbers[0], numbers[1]
        lower_ok = speed_kph > lower if f"{lower:g}<v" in compact else speed_kph >= lower
        upper_ok = speed_kph <= upper if f"v<={upper:g}" in compact else speed_kph < upper
        return lower_ok and upper_ok

    def _project_values(
        self,
        dimension: ScenarioDimension,
        facts: ItemDefinitionFacts,
        operating_mode: str,
        speed: SpeedResolutionResult,
    ) -> list[dict[str, str]]:
        name = dimension.canonical_name
        if dimension.semantics == "VDA702_ATOM_ID_LABEL":
            if name == "WHERE":
                from .scenario_selection_quality import explicit_environment
                qualifiers = [v for v in facts.odd_road_types if explicit_environment(v)]
                values = self._unique([*facts.odd_locations, *qualifiers] or facts.odd_road_types)
                # Refine a generic parking scope using its explicit indoor /
                # outdoor alternatives, while retaining unrelated locations.
                parent_labels = {record["parent_label"] for record in facts.odd_scope_evidence
                                 if record.get("field") == "odd_locations"
                                 and set(record.get("values", [])) <= set(values)}
                values = [v for v in values if v not in parent_labels]
            elif name == "ROAD":
                values = self._unique(facts.odd_road_surfaces)
            elif name == "EGO_ACTION":
                values = self._unique([operating_mode])
            elif name == "EGO_DYNAMICS":
                return [self._speed_constraint_binding(dimension, speed)]
            else:
                values = []
        elif name == "OPERATING_SCENARIO":
            values = self._unique(facts.odd_locations or facts.odd_road_types)
        elif name == "VEHICLE_STATE":
            values = self._unique([operating_mode])
        elif name == "WEATHER":
            values = self._unique(facts.odd_weather_conditions)
        elif name == "ROAD_SURFACE":
            values = self._unique(facts.odd_road_surfaces)
        elif name == "VEHICLE_SPEED":
            return [self._speed_constraint_binding(dimension, speed)]
        else:
            values = []
        if not values:
            return [self._pending_binding(dimension, reason="NO_ITEM_FACT")]
        return [self._text_binding(dimension, value) for value in values]

    @staticmethod
    def _binding_sets(
        dimensions: list[ScenarioDimension],
        options: list[list[dict[str, Any]]],
    ) -> list[dict[str, dict[str, Any]]]:
        """Select one anchored world without inventing unsupported cross-products."""
        anchor_index = next(
            (
                index for index, values in enumerate(options)
                if len(values) > 1 and any(value.get("project_value") for value in values)
            ),
            None,
        )
        if anchor_index is None:
            return [{
                dimension.canonical_name: dict(values[0])
                for dimension, values in zip(dimensions, options)
            }]
        results = []
        for anchor in options[anchor_index]:
            bindings: dict[str, dict[str, Any]] = {}
            for index, (dimension, values) in enumerate(zip(dimensions, options)):
                if index == anchor_index or len(values) == 1:
                    bindings[dimension.canonical_name] = dict(
                        anchor if index == anchor_index else values[0]
                    )
                    continue
                bindings[dimension.canonical_name] = (
                    MethodScenarioCandidateService._pending_binding(
                        dimension,
                        reason="AMBIGUOUS_BINDING",
                        candidate_values=[
                            str(value.get("project_value", ""))
                            for value in values if value.get("project_value")
                        ],
                    )
                )
            results.append(bindings)
        return results

    def _function_phase_bindings(
        self,
        functions: list[FunctionDefinition],
        speed: SpeedResolutionResult,
        *,
        dimension_bindings: dict[str, dict[str, Any]] | None = None,
        operating_mode: str = "",
    ) -> dict[str, Any]:
        """Preserve Function context as binding evidence, not a Method dimension."""
        bindings = {}
        for function in functions:
            entry = {
                "function_id": function.function_id,
                "nominal_operating_state": "NOMINAL_FUNCTION_EXECUTION",
                "function_phase_label": function.name,
                "function_phase_status": "PENDING_NO_STRUCTURED_PHASE_FACT",
                "preconditions": list(function.preconditions),
                "trigger_conditions": list(function.triggers),
                "odd_predicates": list(function.odd_constraints),
                "output": function.output,
                "speed_constraint": dict(speed.matched_envelope),
                "speed_constraint_status": (
                    "EXPLICIT_VALUE" if speed.resolution_source == "ExplicitProjectInput"
                    else "RANGE_CONSTRAINT"
                ),
                "source_refs": [
                    MethodScenarioCandidateService._source_dict(source)
                    for source in function.sources
                ],
            }
            if self.coverage_rules.has_approved_rules and dimension_bindings is not None:
                entry["scenario_coverage"] = self.coverage_rules.evaluate(
                    function,
                    dimension_bindings,
                    operating_mode,
                )
            bindings[function.function_id] = entry
        return {
            "status": "BOUND_FUNCTION_CONTEXTS" if bindings else "NO_FUNCTION_CONTEXT",
            "reason": (
                "Function preconditions, triggers, ODD predicates and output are "
                "preserved as per-function context; they are not Method dimensions."
                if bindings else "No Function definitions were supplied to candidate generation."
            ),
            "bindings": bindings,
        }

    def _apply_compound_fills(self, bindings: dict[str, dict[str, Any]]) -> None:
        """Propagate an already-resolved compound atom to its governed fills."""
        for binding in list(bindings.values()):
            if binding.get("resolution_status") != "RESOLVED":
                continue
            atom_id = str(binding.get("atom_id", "")).strip()
            filled = [str(item) for item in binding.get("filled_dimensions", [])]
            if not atom_id or len(filled) < 2:
                continue
            for dimension in filled:
                target = bindings.get(dimension)
                if not isinstance(target, dict):
                    continue
                if target.get("resolution_status") == "RESOLVED":
                    continue
                speed_constraint = target.get("speed_constraint")
                if isinstance(speed_constraint, dict):
                    range_resolution = self.atom_resolver.intersect_speed_range(
                        minimum_kph=speed_constraint.get("min_kph"),
                        maximum_kph=speed_constraint.get("max_kph"),
                        atom_id=atom_id,
                    )
                    if range_resolution.status != "RESOLVED":
                        target.update({
                            "candidate_atom_ids": list(range_resolution.candidate_atom_ids),
                            "unresolved_reason": range_resolution.reason,
                            "atom_provenance": range_resolution.provenance or {},
                        })
                        continue
                    minimum, maximum = range_resolution.compatible_range_kph or (None, None)
                    target["speed_constraint"] = {
                        **speed_constraint,
                        "min_kph": minimum,
                        "max_kph": maximum,
                        "resolved_by": range_resolution.reason,
                    }
                target.update({
                    "project_value": binding.get("project_value", ""),
                    "method_value": binding.get("method_value", ""),
                    "binding_status": "EXACT",
                    "resolution_status": "RESOLVED",
                    "unresolved_reason": "",
                    "candidate_atom_ids": [atom_id],
                    "atom_id": atom_id,
                    "canonical_atom_id": binding.get("canonical_atom_id", atom_id),
                    "filled_dimensions": filled,
                    "atom_provenance": binding.get("atom_provenance", {}),
                    "compound_fill": True,
                    "filled_by_atom_id": atom_id,
                })

    def _binding_coverage(
        self,
        candidates: list[ScenarioCandidate],
        functions: list[FunctionDefinition],
        dimension_options: list[list[dict[str, Any]]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        coverage: dict[str, Any] = {
            dimension.canonical_name: {
                "total": 0,
                "resolved": 0,
                "ambiguous": 0,
                "pending": 0,
                "candidate_instance_count": 0,
                "context_values": [],
                "unresolved_source_terms": [],
                "candidate_method_atoms": [],
                "reasons": {},
                "candidate_reasons": {},
            }
            for dimension in self.method.scenario_model.dimensions
        }
        gaps: list[dict[str, Any]] = []

        def record_context_value(entry: dict[str, Any], binding: dict[str, Any]) -> None:
            project_value = str(binding.get("project_value", "")).strip()
            if not project_value:
                return
            entry["total"] += 1
            status = str(binding.get("resolution_status", "PENDING"))
            entry["context_values"].append({
                "source_value": project_value,
                "status": status,
                "reason": str(binding.get("unresolved_reason", "")),
                "candidate_method_atoms": list(
                    binding.get("candidate_atom_ids", [])
                ),
            })
            if status == "RESOLVED":
                entry["resolved"] += 1
                return
            entry["ambiguous" if status == "AMBIGUOUS" else "pending"] += 1
            reason = str(binding.get("unresolved_reason", "PENDING_BINDING"))
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
            entry["unresolved_source_terms"].append(project_value)
            entry["candidate_method_atoms"].extend(
                str(item) for item in binding.get("candidate_atom_ids", [])
            )

        for dimension, values in zip(
            self.method.scenario_model.dimensions, dimension_options,
        ):
            for value in values:
                if isinstance(value, dict):
                    record_context_value(coverage[dimension.canonical_name], value)

        for candidate in candidates:
            bindings = candidate.context_resolution.get("dimension_bindings", {})
            if not isinstance(bindings, dict):
                continue
            for dimension, binding in bindings.items():
                if not isinstance(binding, dict):
                    continue
                entry = coverage.setdefault(dimension, {
                    "total": 0, "resolved": 0, "ambiguous": 0, "pending": 0,
                    "candidate_instance_count": 0,
                    "context_values": [],
                    "unresolved_source_terms": [], "candidate_method_atoms": [],
                    "reasons": {}, "candidate_reasons": {},
                })
                entry["candidate_instance_count"] += 1
                status = str(binding.get("resolution_status", "PENDING"))
                if status == "RESOLVED":
                    continue
                reason = str(binding.get("unresolved_reason", "PENDING_BINDING"))
                entry["candidate_reasons"][reason] = (
                    entry["candidate_reasons"].get(reason, 0) + 1
                )
                source_value = str(binding.get("project_value", "")).strip()
                entry["candidate_method_atoms"].extend(
                    str(item) for item in binding.get("candidate_atom_ids", [])
                )
                gaps.append({
                    "scenario_id": candidate.scenario_id,
                    "dimension": dimension,
                    "source_value": source_value,
                    "candidate_values": list(binding.get("candidate_values", [])),
                    "status": status,
                    "reason": reason,
                    "candidate_atoms": list(binding.get("candidate_atom_ids", [])),
                    "evidence": binding.get("atom_provenance", {}),
                })
        for entry in coverage.values():
            entry["unresolved_source_terms"] = sorted(set(entry["unresolved_source_terms"]))
            entry["candidate_method_atoms"] = sorted(set(entry["candidate_method_atoms"]))
        dimensions = tuple(
            item.canonical_name for item in self.method.scenario_model.dimensions
        )
        function_coverage = {}
        for function in functions:
            terms = [
                function.name, function.output, *function.preconditions,
                *function.triggers, *function.odd_constraints,
            ]
            resolved = sorted({
                resolution.atom_id
                for term in terms for dimension in dimensions
                for resolution in [self.atom_resolver.resolve(term, dimension)]
                if resolution.status == "RESOLVED" and resolution.atom_id
            })
            function_coverage[function.function_id] = {
                "function_name": function.name,
                "status": (
                    "RESOLVED_EXPLICIT_METHOD_ATOM" if resolved
                    else "PENDING_NO_EXPLICIT_METHOD_ATOM"
                ),
                "source_terms": terms,
                "resolved_atom_ids": resolved,
            }
        return {"dimensions": coverage, "functions": function_coverage}, gaps

    def generate_for_malfunction(
        self, malfunction: MalfunctionCandidate, parents: list[ScenarioCandidate],
        *, project_facts: ItemDefinitionFacts | None,
        function: FunctionDefinition | None = None,
        record_generated: Callable[[ScenarioCandidate], None] | None = None,
    ) -> tuple[list[ScenarioCandidate], dict[str, Any]]:
        """Ordinary analyze: physical options → governed atoms → driver subsets."""
        from .scenario_method_service import ScenarioMethodService
        from .scenario_synthesis_service import ConstrainedScenarioSynthesisService
        from .scenario_equivalence import coalesce_candidates, scenario_conditions, signature
        from .scenario_selection_quality import explicit_environment

        templates = ScenarioMethodService(self.method)
        if not self.atom_resolver.catalog or project_facts is None:
            return templates.instantiate_analytical_candidates(
                malfunction, parents, project_facts=project_facts, function=function,
            )
        options, audit = templates.instantiate_analytical_candidates(
            malfunction, parents, project_facts=project_facts, function=function,
            expand_driver_configurations=False,
        )
        synthesis = ConstrainedScenarioSynthesisService(self.method)
        project = asdict(project_facts)
        generated, generation_audit, coverage_pools = [], [], {}
        # Select across source contexts, not first-parent-wins. Keep distinct
        # physical options separate; the pool is already bounded by synthesis.
        # Keep a project-wide slope boundary in the broadest supplied location;
        # input ordering must not accidentally narrow it to indoor/outdoor.
        boundary_parent_id = min(parents, key=lambda p: (
            p.operating_scenario not in project_facts.odd_locations,
            bool(explicit_environment(p.operating_scenario)), p.scenario_id)).scenario_id if parents else ""
        for option in sorted(options, key=lambda c: c.scenario_id):
            physical = scenario_conditions(option)
            physical["facts"].pop("operating_scenario", None)
            physical["fact_authority"].pop("operating_scenario", None)
            physical["facts"].get("method_scenario_dimensions", {}).pop("WHERE", None)
            coverage_key = signature({"conditions": physical,
                                      "option": option.analysis_instance.get("source_option_id")})
            children, result = synthesis.generate_candidates(
                malfunction=asdict(malfunction), parent=option,
                project_context=project, function=asdict(function) if function else None,
                include_source_boundaries=(
                    option.scenario_id == boundary_parent_id or option.source_scenario_id == boundary_parent_id
                ),
            )
            result["coverage_family_id"] = coverage_key
            # Keep an unresolved case visible for causal/review diagnostics;
            # never replace an empty pool with an arbitrary default atom.
            if children:
                context_audits = result.get("source_contexts", [result])
                for child in children:
                    context = next(a for a in context_audits if a["source_operating_conditions"]
                                   == child.facts.get("source_operating_conditions"))
                    coverage_pools.setdefault(coverage_key, []).append((child, context))
            else:
                generated.append(option)
            generation_audit.append(result)
        for pool in coverage_pools.values():
            generated.extend(self._select_context_representatives(pool))
        for result in generation_audit:
            if "source_contexts" in result:
                result["scenario_ids"] = [sid for a in result["source_contexts"] for sid in a["scenario_ids"]]
        representatives, mapping = coalesce_candidates(generated)
        audit["atom_generation"] = {
            "version": "ordinary-analyze-atoms-v4-joint-source-coverage", "provider_calls": 0,
            "raw_candidate_count": len(generated),
            "unique_condition_count": len(representatives),
            "coalesced_candidate_count": len(generated) - len(representatives),
            "equivalence_mapping": mapping, "parents": generation_audit,
            "source_boundary_policy": "ONE_SOURCE_ANCHOR_PER_PHYSICAL_OPTION",
            "source_boundary_parent_id": boundary_parent_id,
            "coverage_policy": "JOINT_PRIMARY_SOURCE_AND_ROAD_COVERAGE",
            "coverage_limit": "Unselected context/variant combinations are not evaluated or proven equivalent.",
        }
        children, audit = templates._with_driver_configurations(
            malfunction, representatives, audit, project_facts,
        )
        from .analytical_physics_instantiation_service import AnalyticalPhysicsInstantiationService
        physics = AnalyticalPhysicsInstantiationService(self.method)
        children = [physics.produce_inputs(child, asdict(malfunction), project_facts=project_facts) for child in children]
        audit["physical_input_production"] = {
            "candidate_count": len(children),
            "with_collision_branches": sum(bool(child.facts.get("collision_branches")) for child in children),
            "with_motion_model": sum(bool(child.facts.get("fault_motion_model")) for child in children),
        }
        if record_generated is not None:
            final_ids = {child.scenario_id for child in children}
            for candidate in generated:
                # The caller persists final instances. Do not consume their
                # append-only identity before the equivalence proof is attached.
                if candidate.scenario_id not in final_ids:
                    record_generated(replace(candidate, analysis_instance={
                        **candidate.analysis_instance, "generation_role": "BEFORE_EQUIVALENCE",
                    }))
        return children, audit

    @staticmethod
    def _select_context_representatives(pool):
        """Cover existing alternatives without claiming untested combinations equivalent.

        Primary combinations, source conditions and WHERE/ROAD choices survive. The
        latter cannot be dropped just because another context owns the same
        action. No new combinations, input values or risk grades are created.
        """
        from .scenario_equivalence import signature, scenario_conditions
        from .scenario_selection_quality import explicit_environment
        rows = []
        for child, audit in pool:
            bindings = child.facts.get("method_scenario_dimensions", {})
            primary = audit["coverage_plan"]["primary_variation_dimensions"]
            primary_key = signature({d: bindings.get(d, {}).get("atom_id") for d in primary})
            source = child.facts.get("source_operating_conditions", {})
            source_key = signature(source)
            features = {("primary", primary_key), ("source", source_key),
                        ("where", bindings.get("WHERE", {}).get("atom_id", "")),
                        ("road", bindings.get("ROAD", {}).get("atom_id", ""))}
            parent_environment = explicit_environment(source.get("operating_scenario", ""))
            selected_environment = explicit_environment(bindings.get("WHERE", {}).get("atom_label", ""))
            # Prefer a source-qualified representative. An unqualified parent
            # should not acquire a narrower enclosure simply by lexical rank.
            specificity = (bool(parent_environment), not bool(selected_environment - parent_environment))
            rows.append((child, audit, features, source_key, specificity))
        # Generated IDs also contain method/trace identity. A change to FTTI
        # provenance must not reshuffle otherwise identical ROAD/WHERE choices.
        # Within this physical-option pool, tie-break on actual source conditions
        # and atom selections; IDs only distinguish identical remaining aliases.
        remaining = sorted(rows, key=lambda r: (signature({
            "source": r[0].facts.get("source_operating_conditions", {}),
            "atoms": {d: b.get("atom_id") for d, b in r[0].facts.get("method_scenario_dimensions", {}).items()},
        }), r[0].scenario_id))
        selected, covered, source_counts, environment_counts = [], set(), {}, {}
        while remaining:
            index = max(range(len(remaining)), key=lambda i: (
                len(remaining[i][2] - covered),
                -source_counts.get(remaining[i][3], 0),
                -sum(environment_counts.get(feature, 0) for feature in remaining[i][2]
                     if feature[0] in {"where", "road"}),
                remaining[i][4], -i))
            row = remaining[index]
            if not row[2] - covered:
                break
            remaining.pop(index)
            selected.append(row)
            covered.update(row[2])
            source_counts[row[3]] = source_counts.get(row[3], 0) + 1
            for feature in row[2]:
                if feature[0] in {"where", "road"}:
                    environment_counts[feature] = environment_counts.get(feature, 0) + 1
        exact_keys = {signature(scenario_conditions(r[0])) for r in selected if r[0].facts.get("scenario_atom_ids")}
        for row in remaining:
            child, audit, features, _, _ = row
            if child.facts.get("scenario_atom_ids") and signature(scenario_conditions(child)) in exact_keys:
                # Existing coalescing owns exact alias mapping. Coverage pruning
                # must not turn a proven duplicate into an unevaluated identity.
                selected.append(row)
                continue
            audit["deferred_context_variants"].append({
                "scenario_id": child.scenario_id,
                "selected_atoms": {d: b.get("atom_id") for d, b in child.facts.get("method_scenario_dimensions", {}).items()},
                "coverage_representatives": {kind: next(r[0].scenario_id for r in selected if (kind, value) in r[2])
                                             for kind, value in features},
                "status": "COMBINATION_NOT_EVALUATED", "risk_equivalence_proven": False})
        for _, audit in pool:
            audit["scenario_ids"] = [r[0].scenario_id for r in selected if r[1] is audit]
        return [row[0] for row in selected]

    @staticmethod
    def _source_dict(source: SourceRef) -> dict[str, str]:
        return {
            "source_type": source.source_type,
            "source_id": source.source_id,
            "location": source.location,
            "excerpt": source.excerpt,
        }

    @staticmethod
    def _fact_metadata(
        provenance: FactProvenance,
        approval: ReviewStatus,
        sources: list[SourceRef],
    ) -> dict[str, Any]:
        return {
            "provenance": provenance.value,
            "approval": approval.value,
            "source_refs": [
                MethodScenarioCandidateService._source_dict(item) for item in sources
            ],
        }

    def generate(
        self,
        *,
        project_facts: ItemDefinitionFacts,
        operating_mode: str,
        speed_resolution: SpeedResolutionResult,
        functions: list[FunctionDefinition] | None = None,
    ) -> tuple[list[ScenarioCandidate], dict[str, Any]]:
        dimensions = list(self.method.scenario_model.dimensions)
        options = [
            self._project_values(
                dimension, project_facts, operating_mode, speed_resolution
            )
            for dimension in dimensions
        ]
        binding_sets = self._binding_sets(dimensions, options)
        function_phase_binding = self._function_phase_bindings(
            list(functions or []), speed_resolution,
        )
        candidates: list[ScenarioCandidate] = []
        risk_fact_audits: list[dict[str, Any]] = []
        unresolved_count = 0
        dropped_constraint_count = 0
        rare_constraint_count = 0
        conflicting_constraint_count = 0
        project_sources = list(project_facts.sources)
        speed_sources = list(speed_resolution.source_refs)
        for bindings in binding_sets:
            self._apply_compound_fills(bindings)
            candidate_function_phase_binding = function_phase_binding
            if self.coverage_rules.has_approved_rules:
                candidate_function_phase_binding = self._function_phase_bindings(
                    list(functions or []),
                    speed_resolution,
                    dimension_bindings=bindings,
                    operating_mode=operating_mode,
                )
            unresolved = [
                name for name, value in bindings.items()
                if value.get("resolution_status") != "RESOLVED"
            ]
            if unresolved:
                constraint_evaluation = self.constraints.pending(
                    unresolved, bindings,
                )
            else:
                constraint_evaluation = self.constraints.evaluate(
                    bindings, self.method.scenario_model.constraint_rules,
                )
            if constraint_evaluation.status is ScenarioConstraintStatus.DROP:
                dropped_constraint_count += 1
                continue
            if constraint_evaluation.status is ScenarioConstraintStatus.KEEP_RARE:
                rare_constraint_count += 1
            if constraint_evaluation.status is ScenarioConstraintStatus.CONFLICT:
                conflicting_constraint_count += 1
            unresolved_count += bool(unresolved)
            facts: dict[str, Any] = {
                "method_scenario_dimensions": bindings,
                "operating_mode": operating_mode,
                "source_road_type_scope": self._unique(project_facts.odd_road_types),
                "source_weather_scope": self._unique(project_facts.odd_weather_conditions),
                "odd_scope_evidence": deepcopy(project_facts.odd_scope_evidence),
            }
            if speed_resolution.resolution_source == "ExplicitProjectInput":
                facts["ego_speed_kph"] = speed_resolution.resolved_value
            else:
                facts["ego_speed_constraint"] = dict(
                    speed_resolution.matched_envelope
                )
            if self.method.scenario_model.source_type == "YAML_BASELINE_SCENARIO_ONTOLOGY":
                facts["scenario_atom_ids"] = [
                    str(value.get("method_value", "")).split("|", 1)[0].strip()
                    for value in bindings.values()
                    if value.get("resolution_status") == "RESOLVED"
                    and "|" in str(value.get("method_value", ""))
                ]
            exposure_context = []
            structured = self.method.structured_risk_method
            if structured is not None and "scenario_atom_ids" in facts:
                selected_atoms = set(facts["scenario_atom_ids"])
                coupling = [
                    list(pair) for pair in structured.exposure.strong_couplings
                    if any(atom_id in selected_atoms for atom_id in pair)
                ]
                for atom in structured.exposure.atoms:
                    if atom.atom_id not in selected_atoms:
                        continue
                    exposure_context.append({
                        "atom_id": atom.atom_id,
                        "dimension": list(atom.dimensions),
                        "e_z": atom.duration_level,
                        "e_f": atom.frequency_level,
                        "source_rule_id": atom.source_ref.range,
                        "coupling": coupling,
                        "method_source_hash": str(
                            structured.method_source_hash
                            or self.method.metadata.get("template_hash", "")
                        ),
                    })
            if structured is None:
                # Template mode has no VDA atom catalog, but compiled
                # situation mappings can still carry equivalent Z/F context.
                for dimension_name, binding in bindings.items():
                    value = str(
                        binding.get("method_value") or binding.get("project_value") or ""
                    ).strip()
                    if not value:
                        continue
                    matches = [
                        entry for entry in self.method.exposure.situation_mappings
                        if entry.entry_id in value or entry.description in value
                    ]
                    if len(matches) == 1:
                        entry = matches[0]
                        exposure_context.append({
                            "atom_id": entry.entry_id,
                            "dimension": [dimension_name],
                            "e_z": entry.duration_rating,
                            "e_f": entry.frequency_rating,
                            "source_rule_id": entry.entry_id,
                            "coupling": [],
                            "method_source_hash": str(self.method.metadata.get("template_hash", "")),
                        })
            canonical_keys = {
                "OPERATING_SCENARIO": "operating_scenario",
                "VEHICLE_STATE": "vehicle_state",
                "WEATHER": "weather_conditions",
                "ROAD_SURFACE": "road_surface_conditions",
                "WHERE": "operating_scenario",
                "EGO_ACTION": "vehicle_state",
                "ROAD": "road_surface_conditions",
            }
            for dimension_name, key in canonical_keys.items():
                binding = bindings.get(dimension_name, {})
                facts[key] = binding.get("method_value") or binding.get("project_value", "")
            fact_provenance: dict[str, Any] = {}
            fact_provenance["source_road_type_scope"] = self._fact_metadata(
                FactProvenance.PROJECT_INPUT, project_facts.status, project_sources,
            )
            for dimension_name, key in canonical_keys.items():
                dimension = next((
                    item for item in dimensions
                    if item.canonical_name == dimension_name
                ), None)
                if dimension is None:
                    continue
                method_source = SourceRef(
                    "method_contract",
                    str(self.method.metadata["template_hash"]),
                    f"{dimension.source_ref.sheet}!{dimension.source_ref.range}",
                    dimension.source_ref.raw_text,
                )
                binding = bindings.get(dimension_name, {})
                project_value = str(binding.get("project_value", "")).strip()
                exact_binding = binding.get("resolution_status") == "RESOLVED"
                binding_approval = (
                    ReviewStatus.FINALIZED
                    if (
                        project_value
                        and bool(project_sources)
                    )
                    else ReviewStatus.PENDING
                )
                fact_provenance[key] = self._fact_metadata(
                    (
                        FactProvenance.DERIVED
                        if exact_binding else FactProvenance.PROJECT_INPUT
                    ),
                    binding_approval,
                    (
                        [*project_sources, method_source]
                        if exact_binding else project_sources
                    ),
                )
            speed_fact_key = (
                "ego_speed_kph"
                if speed_resolution.resolution_source == "ExplicitProjectInput"
                else "ego_speed_constraint"
            )
            fact_provenance[speed_fact_key] = self._fact_metadata(
                speed_resolution.provenance,
                speed_resolution.approval,
                speed_sources,
            )
            method_sources = [
                SourceRef(
                    "method_contract",
                    str(self.method.metadata["template_hash"]),
                    f"{item.source_ref.sheet}!{item.source_ref.range}",
                    item.source_ref.raw_text,
                )
                for item in dimensions
            ]
            if "scenario_atom_ids" in facts:
                fact_provenance["scenario_atom_ids"] = self._fact_metadata(
                    FactProvenance.DERIVED,
                    ReviewStatus.FINALIZED if facts["scenario_atom_ids"] else ReviewStatus.PENDING,
                    method_sources,
                )
            material = {
                "contract": SCENARIO_CONTRACT_VERSION,
                "template_hash": self.method.metadata["template_hash"],
                "bindings": bindings,
            }
            fingerprint = hashlib.sha256(
                json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            scenario_id = f"SCN-METHOD-{fingerprint[:16].upper()}"
            risk_binding = self.risk_facts.bind(project_facts, {
                **facts,
                "scenario_id": scenario_id,
            })
            facts.update(risk_binding.values)
            fact_provenance.update(risk_binding.provenance)
            risk_fact_audits.append(risk_binding.audit)
            summary = "；".join(
                value["method_value"] or value["project_value"] or f"{name}:未解析"
                for name, value in bindings.items()
            )
            finalized = (
                not unresolved
                and speed_resolution.approval is ReviewStatus.FINALIZED
                and bool(project_sources)
            )
            fact_provenance["operating_mode"] = self._fact_metadata(
                FactProvenance.PROJECT_INPUT,
                (
                    ReviewStatus.FINALIZED
                    if operating_mode.strip() and project_sources
                    else ReviewStatus.PENDING
                ),
                project_sources,
            )
            fact_provenance["method_scenario_dimensions"] = self._fact_metadata(
                FactProvenance.DERIVED,
                ReviewStatus.FINALIZED if finalized else ReviewStatus.PENDING,
                [*project_sources, *method_sources],
            )
            candidates.append(ScenarioCandidate(
                scenario_id=scenario_id,
                operating_scenario=facts.get("operating_scenario", operating_mode),
                situational_description=summary,
                situational_detailing=summary,
                facts=facts,
                operating_mode=operating_mode,
                context_resolution={
                    "speed_resolution": speed_resolution.to_dict(),
                    "speed_constraint": dict(speed_resolution.matched_envelope),
                    "dimension_bindings": bindings,
                    "risk_fact_binding": risk_binding.audit,
                    "dimension_compatibility": {
                        "status": constraint_evaluation.status.value,
                        "reason": constraint_evaluation.reason,
                        "matched_rule_ids": list(
                            constraint_evaluation.matched_rule_ids
                        ),
                    },
                    "function_phase_binding": candidate_function_phase_binding,
                },
                fact_provenance=fact_provenance,
                status=(ReviewStatus.FINALIZED if finalized else ReviewStatus.PENDING),
                sources=list(dict.fromkeys([
                    *project_sources, *speed_sources, *method_sources,
                    *risk_binding.sources,
                ])),
                rule_version=self.method.contract_version,
                review_reason=(
                    "" if finalized
                    else "Project facts, MethodContract dimension bindings or compatibility require review."
                ),
                source_scenario_id="METHOD_SCENARIO_ONTOLOGY",
                atomic_variant="method_dimensions",
                semantic_fingerprint=fingerprint,
                scenario_contract_version=SCENARIO_CONTRACT_VERSION,
                exposure_context=exposure_context,
            ))
        binding_coverage, binding_gaps = self._binding_coverage(
            candidates, list(functions or []), options,
        )
        return candidates, {
            "candidate_count": len(candidates),
            "atomic_candidate_count": len(candidates),
            "combination_strategy": (
                "primary_dimension_anchor_with_pending_unbound_dimensions"
            ),
            "dimension_compatibility_status": (
                "COMPILED_CONSTRAINTS"
                if self.method.scenario_model.constraint_rules
                else "PENDING_NO_COMPILED_RULES"
            ),
            "function_phase_binding_status": function_phase_binding["status"],
            "template_hash": self.method.metadata["template_hash"],
            "scenario_dimension_count": len(dimensions),
            "unresolved_binding_candidate_count": unresolved_count,
            "dropped_by_constraint_count": dropped_constraint_count,
            "rare_but_feasible_count": rare_constraint_count,
            "constraint_conflict_count": conflicting_constraint_count,
            "ego_speed_value_kph": (
                speed_resolution.resolved_value
                if speed_resolution.resolution_source == "ExplicitProjectInput"
                else None
            ),
            "ego_speed_constraint": dict(speed_resolution.matched_envelope),
            "operating_mode": operating_mode,
            "speed_source_status": speed_resolution.resolution_status.value,
            "speed_resolution": speed_resolution.to_dict(),
            "bound_risk_fact_count": sum(
                len(item["bound_fact_types"]) for item in risk_fact_audits
            ),
            "missing_risk_fact_types": sorted({
                fact_type for item in risk_fact_audits
                for fact_type in item["missing_fact_types"]
            }),
            "conflicting_risk_fact_types": sorted({
                fact_type for item in risk_fact_audits
                for fact_type in item["conflicting_fact_types"]
            }),
            "risk_fact_template_hash_mismatch_count": max(
                (item["template_hash_mismatch_count"] for item in risk_fact_audits),
                default=0,
            ),
            "scenario_contract_version": SCENARIO_CONTRACT_VERSION,
            "method_atom_count": len(self.atom_resolver.catalog),
            "normative_constraint_rule_count": len(
                self.method.scenario_model.constraint_rules
            ),
            "scenario_binding_coverage": binding_coverage,
            "scenario_binding_gaps": binding_gaps,
        }
