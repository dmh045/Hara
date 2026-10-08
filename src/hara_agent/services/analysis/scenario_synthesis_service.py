"""Governed, bounded Scenario atom synthesis over a compiled MethodContract.

The service deliberately excludes Exposure ratings from semantic ranking.  It
also never widens an empty ODD/semantic match back to the full atom catalog.
Legacy FUSA used both behaviours; neither is method authority in V13.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
import re
from typing import Any, Iterable

from hara_agent.contracts import (
    AnalyticalScenarioInstantiation, CandidateOrigin, CoverageLabel,
    MethodContract, SemanticCompatibility, ScenarioAtomCandidateSet, ScenarioBindingAuthority,
    ScenarioDimensionApplicability,
    ScenarioDimensionCandidate, ScenarioSynthesisAssessment,
    ScenarioSynthesisInput, ScenarioSynthesisStatus,
    SynthesisValidationStatus,
)
from hara_agent.models import (
    FactProvenance, MalfunctionCandidate, ReviewStatus, ScenarioCandidate, SourceRef,
)

from .scenario_constraint_service import ScenarioConstraintExecutor, ScenarioConstraintStatus
from .scenario_method_service import ScenarioMethodService
from .scenario_selection_quality import (
    normalize_object_category,
    ScenarioBindingPolicy, ScenarioCandidateRanker, ScenarioCoveragePlanner,
    ScenarioDimensionApplicabilityService, ScenarioSemanticQueryBuilder,
    ScenarioRefinementEvidencePolicy, ScenarioSemanticCompatibilityClassifier,
    ScenarioShortlistPolicy,
    ScenarioVariantDiversityValidator,
)


class ScenarioSynthesisValidationError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: Iterable[str] = (),
    ):
        super().__init__(message)
        self.code = code
        self.details = tuple(details)


class ConstrainedScenarioSynthesisService:
    """Create bounded candidate sets and materialize validated analytical children."""

    def __init__(self, method: MethodContract):
        self.method = method
        raw = method.metadata.get("scenario_atom_catalog", [])
        self.catalog = tuple(item for item in raw if isinstance(item, dict))
        self.by_id = {
            str(item.get("atom_id", "")): item
            for item in self.catalog if str(item.get("atom_id", ""))
        }
        self.dimensions = tuple(
            item.canonical_name for item in method.scenario_model.dimensions
        )
        self.constraint_executor = ScenarioConstraintExecutor()
        self.binding_policy = ScenarioBindingPolicy(self.by_id)
        self.applicability_service = ScenarioDimensionApplicabilityService()
        self.coverage_planner = ScenarioCoveragePlanner()
        self.ranker = ScenarioCandidateRanker()
        self.semantic_classifier = ScenarioSemanticCompatibilityClassifier()
        self.shortlist_policy = ScenarioShortlistPolicy()
        self.diversity_validator = ScenarioVariantDiversityValidator()
        self.scenario_method = ScenarioMethodService(method)
        if not self.dimensions or not self.catalog:
            raise ValueError("Scenario synthesis requires a compiled atom ontology")

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _speed_envelope(parent: ScenarioCandidate) -> tuple[float | None, float | None]:
        raw = parent.facts.get("ego_speed_constraint", {})
        if not isinstance(raw, dict):
            return None, None
        lower = raw.get("min_kph", raw.get("speed_min_kph"))
        upper = raw.get("max_kph", raw.get("speed_max_kph"))
        return (
            float(lower) if isinstance(lower, (int, float)) and not isinstance(lower, bool) else None,
            float(upper) if isinstance(upper, (int, float)) and not isinstance(upper, bool) else None,
        )

    @staticmethod
    def _speed_scope(value: Any) -> str:
        """Normalize an already structured scope; never classify prose."""
        normalized = re.sub(
            r"[^A-Z0-9]+", "_", str(value or "").strip().upper(),
        ).strip("_")
        for suffix in ("_SPEED_RANGE", "_SPEED", "_VEHICLE_SPEED"):
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        return normalized

    @classmethod
    def _envelope_scopes(cls, envelope: dict[str, Any]) -> set[str]:
        return {
            scope for scope in (
                cls._speed_scope(envelope.get("operating_mode")),
                cls._speed_scope(envelope.get("condition")),
            ) if scope
        }

    @staticmethod
    def _speed_source_refs(envelope: dict[str, Any]) -> list[dict[str, str]]:
        result = []
        for source in envelope.get("sources", []):
            if not isinstance(source, dict):
                continue
            result.append({
                "source_type": str(source.get("source_type", "")),
                "source_id": str(source.get("source_id", "")),
                "location": str(source.get("location", "")),
                "excerpt": str(source.get("excerpt", "")),
            })
        return result

    @classmethod
    def _speed_display_kind(cls, envelope: dict[str, Any]) -> str:
        source_text = " ".join(
            item.get("excerpt", "") for item in cls._speed_source_refs(envelope)
        )
        if re.search(r"\d+(?:\.\d+)?\s*(?:-|–|~|至|到)\s*\d", source_text):
            return "RANGE"
        if any(token in source_text for token in ("≤", "<=", "不高于", "最高", "最大")):
            return "UPPER_BOUND"
        minimum = envelope.get("speed_min_kph")
        return "UPPER_BOUND" if minimum in (None, 0, 0.0) else "RANGE"

    @classmethod
    def _speed_expression(cls, envelope: dict[str, Any]) -> str:
        minimum = envelope.get("speed_min_kph")
        maximum = envelope.get("speed_max_kph")
        if cls._speed_display_kind(envelope) == "UPPER_BOUND" and maximum is not None:
            return f"不高于{float(maximum):g} km/h"
        if minimum is not None and maximum is not None:
            return f"{float(minimum):g}–{float(maximum):g} km/h"
        if maximum is not None:
            return f"不高于{float(maximum):g} km/h"
        return f"不低于{float(minimum):g} km/h"

    @classmethod
    def _contextual_speed(
        cls, *, parent: ScenarioCandidate, project_context: dict[str, Any],
        query: dict[str, Any],
    ) -> dict[str, Any]:
        envelopes = [
            item for item in project_context.get("speed_envelopes", [])
            if isinstance(item, dict)
            and str(item.get("status", "")).upper() == "FINALIZED"
        ]
        if not envelopes:
            return {
                "status": "NO_FINALIZED_CONTEXT",
                "classification": "NO_CONTEXTUAL_SOURCE",
            }

        def envelope_range(item: dict[str, Any]) -> tuple[Any, Any]:
            return item.get("speed_min_kph"), item.get("speed_max_kph")

        scoped: list[dict[str, Any]] = []
        selected_context = ""
        match_basis = ""
        constraints = [
            item for item in query.get("function_speed_constraints", [])
            if isinstance(item, dict)
        ]
        for constraint in constraints:
            matching = [
                envelope for envelope in envelopes
                if envelope_range(envelope) == (
                    constraint.get("min_kph"), constraint.get("max_kph")
                )
            ]
            specific = [
                envelope for envelope in matching
                if cls._envelope_scopes(envelope) - {"ACTIVE", "GLOBAL", "PROJECT"}
            ]
            if specific:
                matching = specific
            if matching:
                scoped.extend(matching)
                match_basis = str(constraint.get("source_field", "FUNCTION_SPEED_CONSTRAINT"))
        scoped = list({id(item): item for item in scoped}.values())

        declared = [
            item for item in query.get("declared_operational_contexts", [])
            if isinstance(item, dict) and str(item.get("context", ""))
        ]
        specific_contexts = [
            cls._speed_scope(item["context"]) for item in declared
            if cls._speed_scope(item["context"]) not in {"ACTIVE", "GLOBAL", "PROJECT"}
        ]
        if not scoped and specific_contexts:
            for context in specific_contexts:
                matches = [
                    envelope for envelope in envelopes
                    if context in cls._envelope_scopes(envelope)
                ]
                if matches:
                    scoped.extend(matches)
                    selected_context = context
            if scoped:
                match_basis = "EXPLICIT_STRUCTURED_OPERATIONAL_CONTEXT"

        actions = set(map(str, query.get("action_categories", [])))
        compatible_contexts: tuple[str, ...] = ()
        if "ACTION_PARK" in actions:
            compatible_contexts = ("PARKING",)
        elif "ACTION_REVERSE" in actions:
            compatible_contexts = ("PARKING",)
        elif actions & {"ACTION_STOP", "ACTION_HOLD", "ACTION_ACCELERATE", "ACTION_TURN"}:
            compatible_contexts = ("CONTROL",)
        if not scoped and compatible_contexts:
            for context in compatible_contexts:
                matches = [
                    envelope for envelope in envelopes
                    if context in cls._envelope_scopes(envelope)
                ]
                if matches:
                    scoped.extend(matches)
                    selected_context = context
            if scoped:
                match_basis = "STRUCTURED_ACTION_CONTEXT_COMPATIBILITY"

        if not scoped:
            parent_scope = cls._speed_scope(parent.operating_mode)
            scoped = [
                envelope for envelope in envelopes
                if parent_scope in cls._envelope_scopes(envelope)
                and cls._envelope_scopes(envelope) <= {
                    parent_scope, "GLOBAL", "PROJECT",
                }
            ]
            selected_context = parent_scope
            if scoped:
                match_basis = "PARENT_STATE_LEVEL_ENVELOPE"
        if not scoped:
            parent_minimum, parent_maximum = cls._speed_envelope(parent)
            return {
                "status": "PARENT_FALLBACK",
                "classification": "PARENT_BROAD_RANGE_FALLBACK",
                "requested_contexts": specific_contexts,
                "min_kph": parent_minimum,
                "max_kph": parent_maximum,
            }

        scoped = list({id(item): item for item in scoped}.values())
        selected_scopes = sorted(set().union(*(
            cls._envelope_scopes(item) for item in scoped
        )))
        if not selected_context:
            selected_context = next(
                (item for item in selected_scopes if item not in {"ACTIVE", "GLOBAL", "PROJECT"}),
                selected_scopes[0] if selected_scopes else "",
            )

        ranges = {
            (item.get("speed_min_kph"), item.get("speed_max_kph"))
            for item in scoped
        }
        source_refs = [
            source for envelope in scoped for source in cls._speed_source_refs(envelope)
        ]
        expressions = list(dict.fromkeys(cls._speed_expression(item) for item in scoped))
        if len(ranges) != 1:
            return {
                "status": "SOURCE_CONFLICT",
                "classification": "SOURCE_CONFLICT",
                "selected_context": selected_context,
                "selected_scopes": selected_scopes,
                "match_basis": match_basis,
                "source_expressions": expressions,
                "source_refs": source_refs,
            }
        minimum, maximum = next(iter(ranges))
        parent_minimum, parent_maximum = cls._speed_envelope(parent)
        lowers = [float(value) for value in (minimum, parent_minimum) if value is not None]
        uppers = [float(value) for value in (maximum, parent_maximum) if value is not None]
        resolved_minimum = max(lowers) if lowers else None
        resolved_maximum = min(uppers) if uppers else None
        if (
            resolved_minimum is not None and resolved_maximum is not None
            and resolved_minimum > resolved_maximum
        ):
            return {
                "status": "SOURCE_CONFLICT",
                "classification": "SOURCE_CONFLICT",
                "selected_context": selected_context,
                "selected_scopes": selected_scopes,
                "match_basis": match_basis,
                "source_expressions": expressions,
                "source_refs": source_refs,
                "reason": "PARENT_AND_CONTEXT_SPEED_RANGES_DO_NOT_INTERSECT",
            }
        display_kinds = {cls._speed_display_kind(item) for item in scoped}
        broad_state_fallback = not set(selected_scopes) - {
            "ACTIVE", "GLOBAL", "PROJECT",
        }
        return {
            "status": "RESOLVED",
            "classification": (
                "PARENT_BROAD_RANGE_FALLBACK"
                if broad_state_fallback else "CONTEXTUAL_SPEED_CONSUMED"
            ),
            "selected_context": selected_context,
            "selected_scopes": selected_scopes,
            "match_basis": match_basis,
            "min_kph": resolved_minimum,
            "max_kph": resolved_maximum,
            "display_kind": next(iter(display_kinds)) if len(display_kinds) == 1 else "RANGE",
            "source_expressions": expressions,
            "source_refs": source_refs,
        }

    @classmethod
    def _effective_speed_envelope(
        cls, parent: ScenarioCandidate, query: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        contextual = query.get("contextual_speed", {})
        if isinstance(contextual, dict) and contextual.get("status") == "RESOLVED":
            return contextual.get("min_kph"), contextual.get("max_kph")
        return cls._speed_envelope(parent)

    @staticmethod
    def _speed_range(atom: dict[str, Any]) -> tuple[float | None, float | None] | None:
        raw = atom.get("speed_range_kph")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        return (
            float(raw[0]) if raw[0] is not None else None,
            float(raw[1]) if raw[1] is not None else None,
        )

    @classmethod
    def _speed_compatible(
        cls, atom: dict[str, Any], envelope: tuple[float | None, float | None],
    ) -> bool:
        atom_range = cls._speed_range(atom)
        if atom_range is None:
            return True
        lower_values = [item for item in (envelope[0], atom_range[0]) if item is not None]
        upper_values = [item for item in (envelope[1], atom_range[1]) if item is not None]
        lower = max(lower_values) if lower_values else None
        upper = min(upper_values) if upper_values else None
        return lower is None or upper is None or lower <= upper

    @staticmethod
    def _range_intersection(
        left: tuple[float | None, float | None] | None,
        right: tuple[float | None, float | None] | None,
    ) -> tuple[float | None, float | None] | None:
        if left is None or right is None:
            return None
        lowers = [value for value in (left[0], right[0]) if value is not None]
        uppers = [value for value in (left[1], right[1]) if value is not None]
        lower = max(lowers) if lowers else None
        upper = min(uppers) if uppers else None
        if lower is not None and upper is not None and lower > upper:
            return None
        return lower, upper

    @staticmethod
    def _project_slope_max(project_context: dict[str, Any]) -> float | None:
        explicit = project_context.get("slope_max_pct")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return float(explicit)
        text = json.dumps(project_context, ensure_ascii=False, sort_keys=True)
        matches = re.findall(
            r"(?:不超过|不高于|最大(?:坡度)?|至多|≤|<=)\s*(\d+(?:\.\d+)?)\s*%",
            text, re.IGNORECASE,
        )
        directional = re.findall(
            r"(?:上坡|下坡)\s*(\d+(?:\.\d+)?)\s*%",
            text, re.IGNORECASE,
        )
        bounds = [*map(float, matches)]
        if directional:
            bounds.append(max(map(float, directional)))
        return min(bounds) if bounds else None

    @classmethod
    def _slope_compatible(
        cls, atom: dict[str, Any], project_context: dict[str, Any],
    ) -> bool:
        project_max = cls._project_slope_max(project_context)
        if project_max is None:
            return True
        semantics = atom.get("physical_semantics", {})
        semantics = semantics if isinstance(semantics, dict) else {}
        slope = semantics.get("slope", {})
        slope = slope if isinstance(slope, dict) else {}
        raw_range = slope.get("range_pct")
        lower = raw_range[0] if isinstance(raw_range, list) and len(raw_range) == 2 else slope.get("pct_min")
        if not isinstance(lower, (int, float)) or isinstance(lower, bool):
            return True
        if float(lower) < project_max:
            return True
        if float(lower) > project_max:
            return False
        return bool(slope.get("min_inclusive", True))

    @staticmethod
    def _weather_compatible(
        atom: dict[str, Any], project_context: dict[str, Any],
    ) -> bool:
        label = str(atom.get("label", "")).casefold()
        context = json.dumps(project_context, ensure_ascii=False, sort_keys=True).casefold()
        snow_or_ice = ("snow", "ice", "雪", "冰")
        return not any(term in label for term in snow_or_ice) or any(
            term in context for term in snow_or_ice
        )

    def _candidate(
        self, atom: dict[str, Any], *, origin: CandidateOrigin,
        refs: Iterable[str], reason: str, validated: bool = False,
        binding_authority: str = ScenarioBindingAuthority.ANALYTICAL_SELECTION.value,
        template_relationship: str = "NONE",
        semantic_compatibility: SemanticCompatibility = SemanticCompatibility.UNKNOWN,
        ranking_scores: dict[str, float] | None = None,
    ) -> ScenarioDimensionCandidate:
        atom_id = str(atom.get("atom_id", "")).strip()
        canonical = str(atom.get("v2") or atom.get("v2_proper") or atom_id).strip()
        candidate = ScenarioDimensionCandidate(
            atom_id=atom_id,
            canonical_atom_id=canonical,
            dimensions=tuple(str(item) for item in atom.get("filled_dimensions", [])),
            label=str(atom.get("label", "")),
            source_asset=str(atom.get("source_asset", "")),
            source_rule=str(atom.get("source_rule", "")),
            source_tag=str(atom.get("source_tag", "")),
            candidate_origin=origin,
            supporting_context_refs=tuple(dict.fromkeys(str(item) for item in refs if item)),
            selection_reason=reason,
            method_semantics=deepcopy(atom.get("physical_semantics", {})),
            validation_status=(
                SynthesisValidationStatus.VALIDATED
                if validated else SynthesisValidationStatus.PENDING
            ),
            speed_range_kph=self._speed_range(atom),
            binding_authority=binding_authority,
            template_relationship=template_relationship,
            semantic_compatibility=semantic_compatibility,
            ranking_scores=dict(ranking_scores or {}),
        )
        return replace(
            candidate,
            semantic_family=self.shortlist_policy.semantic_family(
                candidate.dimensions[0] if len(candidate.dimensions) == 1 else "COMPOUND",
                candidate,
            ),
        )

    @staticmethod
    def _binding_atom_id(binding: dict[str, Any]) -> str:
        value = str(binding.get("atom_id") or binding.get("canonical_atom_id") or "").strip()
        return value.split("|", 1)[0].strip()

    @staticmethod
    def _source_ref_payload(source: Any) -> dict[str, str]:
        return {
            "source_hash": str(getattr(source, "source_hash", "")),
            "source_asset": str(getattr(source, "workbook", "")),
            "source_rule": (
                f"{getattr(source, 'sheet', '')}!{getattr(source, 'range', '')}"
            ).strip("!"),
            "source_excerpt": str(getattr(source, "raw_text", "")),
        }

    def _fm_template_evidence(
        self, malfunction: dict[str, Any], parent: ScenarioCandidate,
    ) -> dict[str, Any]:
        catalog = self.method.scenario_model.scenario_method.fm_template_catalog
        if catalog is None:
            return {}
        instance = parent.analysis_instance if isinstance(parent.analysis_instance, dict) else {}
        template_id = str(instance.get("source_template_id", "")).strip()
        template = next(
            (item for item in catalog.templates if item.template_id == template_id),
            None,
        )
        qualification = "PARENT_SOURCE_TEMPLATE" if template is not None else ""
        matched_by: tuple[str, ...] = ()
        matched_terms: tuple[str, ...] = ()
        decision_source_ref: dict[str, Any] = {}
        if template is None:
            causal_chain = malfunction.get("causal_chain", [])
            typed = MalfunctionCandidate(
                malfunction_id=str(malfunction.get("malfunction_id", "")),
                function_id=str(malfunction.get("function_id", "")),
                guideword=str(malfunction.get("guideword", "")),
                description=str(malfunction.get("description", "")),
                functional_effect=str(malfunction.get("functional_effect", "")),
                vehicle_level_hazard=str(malfunction.get("vehicle_level_hazard", "")),
                causal_chain=(
                    list(causal_chain) if isinstance(causal_chain, list) and len(causal_chain) >= 2
                    else ["MALFUNCTION", "VEHICLE_LEVEL_HAZARD"]
                ),
                component_category=str(malfunction.get("component_category", "")),
                failure_type=str(malfunction.get("failure_type", "")),
            )
            match = self.scenario_method.match_fm_template(typed)
            if not match.injectable:
                return {}
            template = match.template
            qualification = match.qualification_tier
            matched_by = match.matched_by
            matched_terms = match.matched_terms
            decision_source_ref = match.decision_source_ref or {}
        assert template is not None

        constraints = []
        for index, option in enumerate(template.required_scenarios, start=1):
            constraints.append({
                "source_option_id": f"{template.template_id}:OPTION:{index}",
                "label": option.label,
                "obj_type": option.obj_type,
                "obj_position": option.obj_position,
                "obj_distance_m": option.obj_distance_m,
                "obj_v_kph": option.obj_v_kph,
                "collision_type": option.collision_type,
                "source": self._source_ref_payload(option.source_ref),
            })
        active_option_id = str(instance.get("source_option_id", "")).strip()
        active_option = next(
            (item for item in constraints if item["source_option_id"] == active_option_id),
            {},
        )
        return {
            "template_id": template.template_id,
            "source": self._source_ref_payload(template.source_ref),
            "source_role": template.source_role,
            "qualification": qualification,
            "matched_by": list(matched_by),
            "matched_terms": list(matched_terms),
            "decision_source_ref": decision_source_ref,
            "applicable_semantics": {
                "keywords": list(template.match.keywords),
                "component_categories": list(template.match.component_categories),
                "failure_types": list(template.match.failure_types),
                "matching_semantics": template.match.matching_semantics,
            },
            "dimension_requirements_preferences": {
                "source_fields_by_dimension": {
                    "OBJECT": ["obj_type", "obj_position"],
                    "EGO_ACTION": ["collision_type"],
                },
                "derivation_basis": "FMTemplateScenario source fields; no new Method atom IDs",
            },
            "supported_interaction_categories": sorted({
                str(item.collision_type) for item in template.required_scenarios
            }),
            "source_governed_constraints": constraints,
            "active_option": active_option,
        }

    def global_compound_legality(
        self, *, atom: dict[str, Any], parent: ScenarioCandidate,
        project_context: dict[str, Any], query: dict[str, Any],
        applicability_by_dimension: dict[str, Any], exact_locks: dict[str, str],
        fm_template: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate one logical atom across every dimension that it fills."""
        atom_id = str(atom.get("atom_id", ""))
        filled = tuple(dict.fromkeys(map(str, atom.get("filled_dimensions", []))))
        if not filled or any(dimension not in self.dimensions for dimension in filled):
            return False, "COMPOUND_DIMENSION_UNAVAILABLE"
        if not self._speed_compatible(atom, self._effective_speed_envelope(parent, query)):
            return False, "ODD_SPEED_INCOMPATIBLE"
        if not self._slope_compatible(atom, project_context):
            return False, "ODD_SLOPE_INCOMPATIBLE"
        if not self._weather_compatible(atom, project_context):
            return False, "ODD_WEATHER_INCOMPATIBLE"
        for dimension in filled:
            applicability = applicability_by_dimension.get(dimension)
            if (
                applicability is None
                or applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE
            ):
                return False, f"COMPOUND_DIMENSION_NOT_APPLICABLE:{dimension}"
            if dimension in exact_locks and exact_locks[dimension] != atom_id:
                return False, f"COMPOUND_EXACT_LOCK_CONFLICT:{dimension}"
        if "WHERE" in filled:
            compatibility, reason = self.semantic_classifier.classify(
                dimension="WHERE", atom=atom, query=query,
                fm_template=fm_template, parent_atom_id="",
            )
            if compatibility is SemanticCompatibility.CONTRADICTED:
                return False, f"PROJECT_ODD_LOCATION_CONTRADICTION:{reason}"
        return True, "GLOBALLY_LEGAL"

    def hard_filter_decision(
        self, *, dimension: str, atom: dict[str, Any], parent: ScenarioCandidate,
        project_context: dict[str, Any], query: dict[str, Any],
        applicability: Any, applicability_by_dimension: dict[str, Any] | None = None,
        decision: Any, exact_locks: dict[str, str],
        fm_template: dict[str, Any],
    ) -> tuple[bool, str]:
        atom_id = str(atom.get("atom_id", ""))
        if dimension not in atom.get("filled_dimensions", []):
            return False, "DIMENSION_NOT_FILLED"
        if applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE:
            return False, "DIMENSION_NOT_APPLICABLE"
        applicability_by_dimension = applicability_by_dimension or {
            value: applicability for value in atom.get("filled_dimensions", [])
        }
        globally_legal, global_reason = self.global_compound_legality(
            atom=atom, parent=parent, project_context=project_context,
            query=query, applicability_by_dimension=applicability_by_dimension,
            exact_locks=exact_locks, fm_template=fm_template,
        )
        if not globally_legal:
            return False, global_reason

        compatibility, reason = self.semantic_classifier.classify(
            dimension=dimension, atom=atom, query=query,
            fm_template=fm_template, parent_atom_id=decision.parent_atom_id,
        )
        if compatibility is SemanticCompatibility.CONTRADICTED:
            return False, f"CONTRADICTED:{reason}"
        if (
            dimension == "EGO_DYNAMICS"
            and decision.authority is ScenarioBindingAuthority.RANGE_CONTAINMENT
            and compatibility is SemanticCompatibility.UNKNOWN
            and self._speed_range(atom)
        ):
            return True, "SUPPORTED:SPEED_RANGE_REFINEMENT"
        return True, f"{compatibility.value}:{reason}"

    def _traffic_representation_elsewhere(
        self, query: dict[str, Any], parent: ScenarioCandidate,
    ) -> dict[str, list[dict[str, Any]]]:
        relations = set(map(str, query.get("traffic_relations", [])))
        result: dict[str, list[dict[str, Any]]] = {}
        if not relations:
            return result
        envelope = self._effective_speed_envelope(parent, query)
        for atom in self.catalog:
            dimensions = tuple(map(str, atom.get("filled_dimensions", [])))
            if "TRAFFIC_PATTERN" in dimensions or not dimensions:
                continue
            if not self._speed_compatible(atom, envelope):
                continue
            categories = self.ranker.atom_categories(atom)
            for relation in sorted(relations & categories):
                result.setdefault(relation, []).append({
                    "atom_id": str(atom.get("atom_id", "")),
                    "canonical_atom_id": str(
                        atom.get("v2") or atom.get("v2_proper")
                        or atom.get("atom_id", "")
                    ),
                    "label": str(atom.get("label", "")),
                    "filled_dimensions": list(dimensions),
                    "source_asset": str(atom.get("source_asset", "")),
                    "source_rule": str(atom.get("source_rule", "")),
                })
        return result

    @staticmethod
    def _shortlist_diagnostics(
        ranked: Iterable[ScenarioDimensionCandidate],
        selected: Iterable[ScenarioDimensionCandidate],
    ) -> dict[str, int]:
        ranked_items = tuple(ranked)
        selected_items = tuple(selected)
        selected_ids = {item.atom_id for item in selected_items}
        dropped = [item for item in ranked_items if item.atom_id not in selected_ids]
        selected_high_families = {
            item.semantic_family for item in selected_items
            if float(item.ranking_scores.get("source_evidence_tier", 0)) >= 2
        }
        dropped_high_families = {
            item.semantic_family for item in dropped
            if float(item.ranking_scores.get("source_evidence_tier", 0)) >= 2
        }
        return {
            "ranked_pool_size": len(ranked_items),
            "selected_size": len(selected_items),
            "authoritative_below_cutoff": sum(
                float(item.ranking_scores.get("source_evidence_tier", 0)) >= 4
                or item.binding_authority
                != ScenarioBindingAuthority.ANALYTICAL_SELECTION.value
                for item in dropped
            ),
            "fm_template_below_cutoff": sum(
                item.template_relationship != "NONE" for item in dropped
            ),
            "exact_structured_source_below_cutoff": sum(
                bool(item.ranking_scores.get("structured_source_score", 0))
                or bool(item.ranking_scores.get("physical_semantics_score", 0))
                for item in dropped
            ),
            "high_evidence_family_below_cutoff": len(
                dropped_high_families - selected_high_families
            ),
        }

    def candidate_sets(
        self, *, malfunction: dict[str, Any], parent: ScenarioCandidate,
        assessment: dict[str, Any], project_context: dict[str, Any],
        query: dict[str, Any], fm_template: dict[str, Any],
    ) -> tuple[ScenarioAtomCandidateSet, ...]:
        envelope = self._effective_speed_envelope(parent, query)
        decisions = {
            dimension: self.binding_policy.decide(parent, dimension, envelope)
            for dimension in self.dimensions
        }
        applicability = {
            dimension: self.applicability_service.assess(
                dimension, query, decisions[dimension],
            )
            for dimension in self.dimensions
        }
        exact_locks = {
            dimension: decision.parent_atom_id
            for dimension, decision in decisions.items()
            if decision.parent_atom_id and not decision.refinable
        }
        variable_dimensions = tuple(
            dimension for dimension in self.dimensions
            if applicability[dimension].status is not ScenarioDimensionApplicability.NOT_APPLICABLE
            and dimension not in exact_locks
        )
        primary_dimensions, secondary_dimensions, _ = self.coverage_planner.axis_priority(
            query=query, dimensions=variable_dimensions,
        )
        bindings = parent.facts.get("method_scenario_dimensions", {})
        result = []
        ranked_by_dimension: dict[str, tuple[ScenarioDimensionCandidate, ...]] = {}
        for dimension in self.dimensions:
            catalog = [
                item for item in self.catalog
                if dimension in item.get("filled_dimensions", [])
            ]
            before = str(
                (bindings.get(dimension, {}) if isinstance(bindings, dict) else {}).get(
                    "resolution_status", "PENDING"
                )
            )
            decision = decisions[dimension]
            applicable = applicability[dimension]
            if dimension in exact_locks:
                atom = self.by_id[exact_locks[dimension]]
                legal, legal_reason = self.global_compound_legality(
                    atom=atom, parent=parent, project_context=project_context,
                    query=query, applicability_by_dimension=applicability,
                    exact_locks=exact_locks, fm_template=fm_template,
                )
                if not legal:
                    result.append(ScenarioAtomCandidateSet(
                        dimension=dimension, catalog_size=len(catalog),
                        hard_filtered_pool_size=0, candidates=(),
                        applicability=applicable, binding_decision=decision,
                        resolution_status_before=before, generation_status="METHOD_GAP",
                        reason=f"Exact binding is globally incompatible: {legal_reason}",
                        shortlist_budget=1, shortlist_policy="EXACT_FIXED",
                        shortlist_diagnostics={
                            "ranked_pool_size": 0, "selected_size": 0,
                            "authoritative_below_cutoff": 0,
                            "fm_template_below_cutoff": 0,
                            "exact_structured_source_below_cutoff": 0,
                            "high_evidence_family_below_cutoff": 0,
                        },
                        hard_filter_diagnostics={
                            "illegal_rejected": 1, "contradicted_rejected": 0,
                            "unknown_retained": 0, "supported_retained": 0,
                            "refinement_unsupported_rejected": 0,
                        },
                    ))
                    ranked_by_dimension[dimension] = ()
                    continue
                locked = self._candidate(
                    atom, origin=CandidateOrigin.DIRECT_PROJECT_BINDING,
                    refs=decision.source_refs,
                    reason=decision.basis, validated=True,
                    binding_authority=decision.authority.value,
                    ranking_scores={
                        "template_score": 0.0, "mechanism_score": 0.0,
                        "action_score": 0.0, "object_score": 0.0,
                        "traffic_relation_score": 0.0, "odd_score": 0.0,
                        "causal_score": 0.0, "lexical_score": 0.0,
                        "category_context_score": 0.0,
                        "structured_source_score": 0.0,
                        "physical_semantics_score": 0.0,
                        "source_evidence_tier": 4.0,
                        "final_rank_score": 0.0,
                    },
                    semantic_compatibility=SemanticCompatibility.SUPPORTED,
                )
                result.append(ScenarioAtomCandidateSet(
                    dimension=dimension, catalog_size=len(catalog),
                    hard_filtered_pool_size=1, candidates=(locked,),
                    applicability=applicable, binding_decision=decision,
                    locked_atom_ids=(locked.atom_id,),
                    resolution_status_before=before,
                    generation_status="LOCKED_EXACT_AUTHORITY",
                    reason=decision.basis,
                    shortlist_budget=1,
                    shortlist_policy="EXACT_FIXED",
                    shortlist_diagnostics={
                        "ranked_pool_size": 1, "selected_size": 1,
                        "authoritative_below_cutoff": 0,
                        "fm_template_below_cutoff": 0,
                        "exact_structured_source_below_cutoff": 0,
                        "high_evidence_family_below_cutoff": 0,
                    },
                    hard_filter_diagnostics={
                        "illegal_rejected": 0, "contradicted_rejected": 0,
                        "unknown_retained": 0, "supported_retained": 1,
                        "refinement_unsupported_rejected": 0,
                    },
                ))
                ranked_by_dimension[dimension] = (locked,)
                continue

            hard_pool = []
            hard_counts = {
                "illegal_rejected": 0, "contradicted_rejected": 0,
                "unknown_retained": 0, "supported_retained": 0,
                "refinement_unsupported_rejected": 0,
            }
            for atom in catalog:
                passed, hard_reason = self.hard_filter_decision(
                    dimension=dimension, atom=atom, parent=parent,
                    project_context=project_context, query=query,
                    applicability=applicable,
                    applicability_by_dimension=applicability, decision=decision,
                    exact_locks=exact_locks, fm_template=fm_template,
                )
                if passed:
                    hard_pool.append(atom)
                    key = (
                        "supported_retained"
                        if hard_reason.startswith("SUPPORTED:")
                        else "unknown_retained"
                    )
                    hard_counts[key] += 1
                elif hard_reason.startswith("CONTRADICTED:"):
                    hard_counts["contradicted_rejected"] += 1
                else:
                    hard_counts["illegal_rejected"] += 1
            corpus = [str(atom.get("label", "")) for atom in hard_pool]
            ranked = []
            for atom in hard_pool:
                compatibility, compatibility_reason = self.semantic_classifier.classify(
                    dimension=dimension, atom=atom, query=query,
                    fm_template=fm_template, parent_atom_id=decision.parent_atom_id,
                )
                scores, rank_reason = self.ranker.score(
                    dimension=dimension, atom=atom, query=query,
                    fm_template=fm_template, corpus_labels=corpus, odd_passed=True,
                )
                relationship = (
                    "SOURCE_TEMPLATE_COMPATIBLE"
                    if scores["template_score"] > 0 else "NONE"
                )
                atom_id = str(atom.get("atom_id", ""))
                origin = (
                    CandidateOrigin.DIRECT_PROJECT_BINDING
                    if atom_id == decision.parent_atom_id
                    else CandidateOrigin.METHOD_TEMPLATE
                    if relationship != "NONE"
                    else CandidateOrigin.BINDING_REFINEMENT
                    if decision.parent_atom_id and decision.refinable
                    else CandidateOrigin.ODD_CONSTRAINED_CATALOG
                    if dimension in {"WHERE", "ROAD", "EGO_X_ROAD", "EGO_DYNAMICS"}
                    else CandidateOrigin.HAZARD_CAUSAL_SEMANTIC_CANDIDATE
                )
                refs = [
                    "PROJECT.ODD", "MF.description", "MF.functional_effect",
                    "MF.vehicle_level_hazard", "HE.hazardous_event",
                    "CAUSAL.summary", "PARENT.scenario",
                    "METHOD.scenario_atom_catalog",
                ]
                if relationship != "NONE":
                    refs.append("METHOD.fm_scenario_template")
                candidate = self._candidate(
                    atom, origin=origin, refs=refs,
                    reason=(
                        f"{rank_reason}; {decision.basis}; hard Method/ODD/"
                        "compound/applicability filters passed"
                    ),
                    binding_authority=(
                        decision.authority.value
                        if atom_id == decision.parent_atom_id
                        else ScenarioBindingAuthority.ANALYTICAL_SELECTION.value
                    ),
                    template_relationship=relationship,
                    semantic_compatibility=compatibility,
                    ranking_scores=scores,
                )
                candidate = replace(
                    candidate,
                    semantic_family=self.shortlist_policy.semantic_family(
                        dimension, candidate,
                    ),
                    selection_reason=(
                        candidate.selection_reason
                        + f"; semantic_compatibility={compatibility.value}:"
                        + compatibility_reason
                    ),
                )
                if origin is CandidateOrigin.BINDING_REFINEMENT:
                    refinement_status, refinement_reason = (
                        ScenarioRefinementEvidencePolicy.classify(candidate, decision)
                    )
                    candidate = replace(
                        candidate,
                        selection_reason=(
                            candidate.selection_reason
                            + f"; refinement_evidence={refinement_status}:"
                            + refinement_reason
                        ),
                    )
                    if refinement_status == "REFINEMENT_UNSUPPORTED":
                        hard_counts["refinement_unsupported_rejected"] += 1
                        continue
                ranked.append(candidate)
            ranked.sort(key=lambda item: self.ranker.rank_key(
                item.atom_id, item.ranking_scores,
            ))
            ranked_by_dimension[dimension] = tuple(ranked)
            candidates, shortlist_budget, shortlist_name = self.shortlist_policy.select(
                dimension=dimension, ranked=ranked, applicability=applicable,
                exact=False, primary_dimensions=primary_dimensions,
                secondary_dimensions=secondary_dimensions, query=query,
            )
            explicit_requirement = bool(
                self.semantic_classifier._query_categories(dimension, query)
            )
            supported = any(
                candidate.semantic_compatibility is SemanticCompatibility.SUPPORTED
                for candidate in candidates
            )
            if applicable.status is ScenarioDimensionApplicability.NOT_APPLICABLE:
                status = "NOT_APPLICABLE"
            elif (
                applicable.status is ScenarioDimensionApplicability.REQUIRED
                and (not candidates or (explicit_requirement and not supported))
            ):
                status = "METHOD_GAP"
            elif not candidates:
                status = "OPTIONAL_NO_SUPPORTED_ATOM"
            else:
                status = "CANDIDATES_AVAILABLE"
            result.append(ScenarioAtomCandidateSet(
                dimension=dimension, catalog_size=len(catalog),
                hard_filtered_pool_size=len(hard_pool), candidates=candidates,
                applicability=applicable, binding_decision=decision,
                resolution_status_before=before, generation_status=status,
                reason=(
                    "Required semantic relation has no supported Method atom; "
                    "non-contradicted UNKNOWN candidates remain visible for review."
                    if status == "METHOD_GAP" else
                    applicable.reason if status == "NOT_APPLICABLE" else
                    "Candidates passed hard Method/ODD/compound/applicability filters "
                    "and were ranked by structured features plus BM25."
                ),
                shortlist_truncated=len(ranked) > len(candidates),
                shortlist_budget=shortlist_budget,
                shortlist_policy=shortlist_name,
                shortlist_diagnostics=self._shortlist_diagnostics(ranked, candidates),
                hard_filter_diagnostics=hard_counts,
            ))

        # Establish an atomic candidate space.  A logical compound is either
        # available in every filled dimension or absent everywhere.
        def globally_ranked(candidate: ScenarioDimensionCandidate) -> bool:
            return all(
                any(
                    value.atom_id == candidate.atom_id
                    for value in ranked_by_dimension.get(dimension, ())
                )
                for dimension in candidate.dimensions
                if dimension in self.dimensions
            )

        invalid_compounds = {
            candidate.atom_id
            for candidates in ranked_by_dimension.values()
            for candidate in candidates
            if len(candidate.dimensions) > 1 and not globally_ranked(candidate)
        }
        ranked_by_dimension = {
            dimension: tuple(
                candidate for candidate in candidates
                if candidate.atom_id not in invalid_compounds
            )
            for dimension, candidates in ranked_by_dimension.items()
        }

        refilled = []
        for item in result:
            candidates = [
                candidate for candidate in item.candidates
                if candidate.atom_id not in invalid_compounds
            ]
            selected_families = {
                candidate.semantic_family for candidate in candidates
                if float(candidate.ranking_scores.get("source_evidence_tier", 0)) >= 2
            }
            for candidate in ranked_by_dimension.get(item.dimension, ()):
                if (
                    float(candidate.ranking_scores.get("source_evidence_tier", 0)) >= 2
                    and candidate.semantic_family not in selected_families
                ):
                    candidates.append(candidate)
                    selected_families.add(candidate.semantic_family)
            refilled.append(replace(item, candidates=tuple(candidates)))
        result = refilled

        exposed_compounds = {
            candidate.atom_id
            for item in result for candidate in item.candidates
            if len(candidate.dimensions) > 1
        }
        closed = []
        for item in result:
            candidates = list(item.candidates)
            present = {candidate.atom_id for candidate in candidates}
            for candidate in ranked_by_dimension.get(item.dimension, ()):
                if candidate.atom_id in exposed_compounds and candidate.atom_id not in present:
                    candidates.append(candidate)
            order = {
                candidate.atom_id: index
                for index, candidate in enumerate(ranked_by_dimension.get(item.dimension, ()))
            }
            candidates.sort(key=lambda candidate: order.get(candidate.atom_id, len(order)))
            closed.append(replace(item, candidates=tuple(candidates)))
        result = closed

        # Closing paired dimensions can merge independently shortlisted
        # representatives.  Re-apply the non-intensity family cap globally so
        # compound expansion does not create generic intensity-only choices.
        family_removals: set[str] = set()
        for item in result:
            if (
                item.dimension != "EGO_DYNAMICS"
                or self.shortlist_policy._intensity_driven(query)
            ):
                continue
            family_counts: Counter[str] = Counter()
            for candidate in item.candidates:
                protected = (
                    candidate.binding_authority
                    != ScenarioBindingAuthority.ANALYTICAL_SELECTION.value
                    or candidate.template_relationship != "NONE"
                    or bool(candidate.ranking_scores.get("structured_source_score", 0))
                    or bool(candidate.ranking_scores.get("physical_semantics_score", 0))
                )
                if family_counts[candidate.semantic_family] >= 2 and not protected:
                    family_removals.add(candidate.atom_id)
                    continue
                family_counts[candidate.semantic_family] += 1
        if family_removals:
            result = [
                replace(
                    item,
                    candidates=tuple(
                        candidate for candidate in item.candidates
                        if candidate.atom_id not in family_removals
                    ),
                )
                for item in result
            ]

        membership = {
            item.dimension: {candidate.atom_id for candidate in item.candidates}
            for item in result
        }
        partial = {
            candidate.atom_id
            for item in result for candidate in item.candidates
            if len(candidate.dimensions) > 1 and any(
                candidate.atom_id not in membership.get(dimension, set())
                for dimension in candidate.dimensions
            )
        }
        if partial:
            raise RuntimeError(
                "Compound candidate closure invariant failed: " + ",".join(sorted(partial))
            )

        updated = []
        for item in result:
            status = item.generation_status
            if not item.candidates and item.applicability.status is ScenarioDimensionApplicability.REQUIRED:
                status = "METHOD_GAP"
            elif not item.candidates and item.applicability.status is ScenarioDimensionApplicability.OPTIONAL:
                status = "OPTIONAL_NO_SUPPORTED_ATOM"
            updated.append(replace(
                item, generation_status=status,
                shortlist_truncated=(
                    len(ranked_by_dimension.get(item.dimension, ())) > len(item.candidates)
                ),
                shortlist_diagnostics=self._shortlist_diagnostics(
                    ranked_by_dimension.get(item.dimension, ()), item.candidates,
                ),
            ))
        result = updated
        return tuple(result)

    def build_input(
        self, *, malfunction: dict[str, Any], parent: ScenarioCandidate,
        assessment: dict[str, Any], project_context: dict[str, Any],
        function: dict[str, Any] | None = None,
    ) -> ScenarioSynthesisInput:
        fm_template = self._fm_template_evidence(malfunction, parent)
        query = ScenarioSemanticQueryBuilder.build(
            malfunction=malfunction, parent=parent, assessment=assessment,
            project_context=project_context, fm_template=fm_template,
            function=function,
        )
        contextual_speed = self._contextual_speed(
            parent=parent, project_context=project_context, query=query,
        )
        query["contextual_speed"] = contextual_speed
        query["traffic_relations_represented_elsewhere"] = (
            self._traffic_representation_elsewhere(query, parent)
        )
        candidate_sets = self.candidate_sets(
            malfunction=malfunction, parent=parent, assessment=assessment,
            project_context=project_context, query=query,
            fm_template=fm_template,
        )
        coverage_plan = self.coverage_planner.plan(
            query=query, candidate_sets=candidate_sets,
        )
        hazard_id = str(assessment.get("hazardous_event_id", "")).strip()
        if not hazard_id:
            causal = assessment.get("causal_assessment", {})
            chain = causal.get("causal_chain", []) if isinstance(causal, dict) else []
            hazard_node = str(chain[-1]) if chain else "HAZARD"
            hazard_id = f"HE::{malfunction.get('malfunction_id','')}::{parent.scenario_id}::{hazard_node}"
        signature = {
            "method_hash": self.method.metadata.get("method_source_hash", ""),
            "malfunction": {
                key: malfunction.get(key, "") for key in (
                    "function_id", "guideword", "description", "functional_effect",
                    "vehicle_level_hazard", "component_category", "failure_type",
                )
            },
            "hazard": {
                "hazardous_event": assessment.get("hazardous_event", ""),
                "causal_status": assessment.get("causal_assessment", {}).get("status", "")
                if isinstance(assessment.get("causal_assessment"), dict) else "",
            },
            "parent": {
                "scenario_id": parent.scenario_id,
                "operating_mode": parent.operating_mode,
                "semantic_fingerprint": parent.semantic_fingerprint,
                "object_type": parent.facts.get("object_type", ""),
                "collision_type": parent.facts.get("collision_type", ""),
            },
            "candidate_sets": {
                item.dimension: [candidate.atom_id for candidate in item.candidates]
                for item in candidate_sets
            },
            "applicability": {
                item.dimension: item.applicability.status.value
                for item in candidate_sets
            },
            "coverage_plan": coverage_plan.to_dict(),
            "fm_template_id": fm_template.get("template_id", ""),
            "contextual_speed": contextual_speed,
        }
        group_id = "SYNTH-" + hashlib.sha256(
            self._canonical_json(signature).encode("utf-8")
        ).hexdigest()[:20].upper()
        return ScenarioSynthesisInput(
            malfunction_id=str(malfunction.get("malfunction_id", "")),
            parent_scenario_id=parent.scenario_id,
            hazardous_event_id=hazard_id,
            function_id=str(malfunction.get("function_id", "")),
            malfunction=dict(malfunction), parent_scenario=parent.to_dict(),
            causal_assessment=dict(assessment.get("causal_assessment", {})),
            project_context=dict(project_context),
            method_contract_hash=str(self.method.metadata.get("method_source_hash", "")),
            dimension_candidate_sets=candidate_sets,
            semantic_group_id=group_id,
            structured_semantic_query=query,
            coverage_plan=coverage_plan,
            fm_scenario_template=fm_template,
            contextual_speed=contextual_speed,
        )

    def logical_candidate_registry(
        self, synthesis_input: ScenarioSynthesisInput,
    ) -> dict[str, ScenarioDimensionCandidate]:
        """Return one entry per fully closed Provider-selectable Method atom."""
        candidate_sets = {
            item.dimension: item for item in synthesis_input.dimension_candidate_sets
        }
        registry: dict[str, ScenarioDimensionCandidate] = {}
        for candidate_set in synthesis_input.dimension_candidate_sets:
            for candidate in candidate_set.candidates:
                filled = tuple(
                    dimension for dimension in candidate.dimensions
                    if dimension in candidate_sets
                )
                if not filled or any(
                    candidate.atom_id not in {
                        item.atom_id for item in candidate_sets[dimension].candidates
                    }
                    for dimension in filled
                ):
                    continue
                registry.setdefault(candidate.atom_id, candidate)
        return dict(sorted(registry.items()))

    def logical_selection_bundles(
        self, synthesis_input: ScenarioSynthesisInput, *, limit: int = 96,
    ) -> tuple[tuple[str, ...], ...]:
        """Build a bounded set of conflict-free logical exact covers."""
        registry = self.logical_candidate_registry(synthesis_input)
        candidate_sets = {
            item.dimension: item for item in synthesis_input.dimension_candidate_sets
        }
        candidate_scores: dict[str, float] = {}
        for item in synthesis_input.dimension_candidate_sets:
            for candidate in item.candidates:
                score = (
                    100.0 * float(candidate.ranking_scores.get("source_evidence_tier", 0))
                    + float(candidate.ranking_scores.get("final_rank_score", 0))
                )
                candidate_scores[candidate.atom_id] = max(
                    candidate_scores.get(candidate.atom_id, float("-inf")), score,
                )

        locked_ids = tuple(dict.fromkeys(
            atom_id
            for item in synthesis_input.dimension_candidate_sets
            for atom_id in item.locked_atom_ids
        ))
        selected: list[str] = []
        occupied: set[str] = set()
        for atom_id in locked_ids:
            candidate = registry.get(atom_id)
            if candidate is None or occupied & set(candidate.dimensions):
                return ()
            selected.append(atom_id)
            occupied.update(candidate.dimensions)

        primary = tuple(synthesis_input.coverage_plan.primary_variation_dimensions)
        required = tuple(
            dimension for dimension in self.dimensions
            if candidate_sets[dimension].applicability.status
            is ScenarioDimensionApplicability.REQUIRED
        )
        optional_primary = tuple(
            dimension for dimension in primary
            if candidate_sets[dimension].applicability.status
            is ScenarioDimensionApplicability.OPTIONAL
        )
        targets = tuple(dict.fromkeys((*primary, *required, *optional_primary)))
        states: list[tuple[tuple[str, ...], frozenset[str], float]] = [(
            tuple(selected), frozenset(occupied),
            sum(candidate_scores.get(atom_id, 0.0) for atom_id in selected),
        )]

        def trim(
            values: Iterable[tuple[tuple[str, ...], frozenset[str], float]],
        ) -> list[tuple[tuple[str, ...], frozenset[str], float]]:
            unique = {}
            for atom_ids, dimensions, score in values:
                key = tuple(sorted(atom_ids))
                previous = unique.get(key)
                if previous is None or score > previous[2]:
                    unique[key] = (key, dimensions, score)
            ordered = sorted(unique.values(), key=lambda item: (-item[2], item[0]))
            return ordered[: max(limit * 4, 192)]

        for dimension in targets:
            next_states = []
            optional = (
                candidate_sets[dimension].applicability.status
                is ScenarioDimensionApplicability.OPTIONAL
            )
            for atom_ids, dimensions, score in states:
                if dimension in dimensions:
                    next_states.append((atom_ids, dimensions, score))
                    continue
                if optional:
                    next_states.append((atom_ids, dimensions, score))
                for candidate in registry.values():
                    filled = set(candidate.dimensions)
                    if dimension not in filled or dimensions & filled:
                        continue
                    next_states.append((
                        (*atom_ids, candidate.atom_id),
                        dimensions | filled,
                        score + candidate_scores.get(candidate.atom_id, 0.0),
                    ))
            states = trim(next_states)
            if not states:
                return ()

        valid = []
        fingerprints = set()
        primary_signatures = set()
        for atom_ids, dimensions, score in sorted(
            states, key=lambda item: (-item[2], item[0]),
        ):
            if any(dimension not in dimensions for dimension in required):
                continue
            try:
                expanded = self.expand_logical_atom_ids(synthesis_input, atom_ids)
            except ScenarioSynthesisValidationError:
                continue
            if self._selection_reasons(synthesis_input, expanded):
                continue
            fingerprint = self._canonical_json({
                dimension: [
                    self.by_id[atom_id].get("v2")
                    or self.by_id[atom_id].get("v2_proper") or atom_id
                    for atom_id in expanded[dimension]
                ]
                for dimension in self.dimensions
            })
            if fingerprint in fingerprints:
                continue
            signature = tuple(expanded.get(dimension, ()) for dimension in primary)
            if signature in primary_signatures and len(valid) < limit // 2:
                continue
            fingerprints.add(fingerprint)
            primary_signatures.add(signature)
            valid.append(tuple(sorted(atom_ids)))
            if len(valid) >= limit:
                break
        return tuple(valid)

    def logical_selection_assignments(
        self, synthesis_input: ScenarioSynthesisInput, *, limit: int = 24,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Build bounded whole-sibling assignments that satisfy the Coverage Plan."""
        desired = synthesis_input.coverage_plan.desired_variant_count
        bundles = self.logical_selection_bundles(synthesis_input, limit=96)
        if desired <= 0 or not bundles:
            return ()
        pool = bundles[:32]
        expanded = {
            bundle: self.expand_logical_atom_ids(synthesis_input, bundle)
            for bundle in pool
        }
        fingerprints = {
            bundle: self._canonical_json({
                dimension: [
                    self.by_id[atom_id].get("v2")
                    or self.by_id[atom_id].get("v2_proper") or atom_id
                    for atom_id in expanded[bundle].get(dimension, ())
                ]
                for dimension in self.dimensions
            })
            for bundle in pool
        }
        assignments = []
        for assignment in product(pool, repeat=desired):
            if len({fingerprints[bundle] for bundle in assignment}) != desired:
                continue
            if self.diversity_validator.reasons(
                synthesis_input.coverage_plan,
                (expanded[bundle] for bundle in assignment),
            ):
                continue
            assignments.append(assignment)
            if len(assignments) >= limit:
                break
        return tuple(assignments)

    def expand_logical_atom_ids(
        self, synthesis_input: ScenarioSynthesisInput,
        atom_ids: Iterable[str],
    ) -> dict[str, tuple[str, ...]]:
        """Expand each logical selection atomically across Method dimensions."""
        values = tuple(map(str, atom_ids))
        if len(values) != len(set(values)):
            duplicate = next(value for value in values if values.count(value) > 1)
            raise ScenarioSynthesisValidationError(
                f"DUPLICATE_LOGICAL_ATOM_ID:{duplicate}",
                f"Logical atom {duplicate} was selected more than once",
            )
        registry = self.logical_candidate_registry(synthesis_input)
        selected: dict[str, list[str]] = {dimension: [] for dimension in self.dimensions}
        conflicts: list[str] = []
        for atom_id in values:
            candidate = registry.get(atom_id)
            if candidate is None:
                code = (
                    f"INVENTED_LOGICAL_ATOM_ID:{atom_id}"
                    if atom_id not in self.by_id
                    else f"LOGICAL_ATOM_OUTSIDE_REGISTRY:{atom_id}"
                )
                raise ScenarioSynthesisValidationError(
                    code, f"Logical atom {atom_id} is not Provider-selectable",
                )
            for dimension in candidate.dimensions:
                if dimension not in selected:
                    raise ScenarioSynthesisValidationError(
                        f"UNKNOWN_DIMENSION:{dimension}",
                        f"Logical atom {atom_id} fills an unknown dimension",
                    )
                if selected[dimension] and atom_id not in selected[dimension]:
                    conflicts.append(
                        f"COMPOUND_ATOM_CONFLICT:{dimension}:"
                        f"{selected[dimension][0]},{atom_id}"
                    )
                selected[dimension].append(atom_id)
        if conflicts:
            reasons = sorted(set(conflicts))
            raise ScenarioSynthesisValidationError(
                self._primary_validation_reason(reasons), "; ".join(reasons),
                details=reasons,
            )
        return {
            dimension: tuple(atom_ids) for dimension, atom_ids in selected.items()
        }

    def _selection_reasons(
        self, synthesis_input: ScenarioSynthesisInput,
        selected: dict[str, tuple[str, ...]], *, partial: bool = False,
    ) -> list[str]:
        candidate_sets = {item.dimension: item for item in synthesis_input.dimension_candidate_sets}
        reasons: list[str] = []
        expected = set(self.dimensions)
        if not partial and set(selected) != expected:
            reasons.append("DIMENSION_KEY_SET_MISMATCH")
        for dimension, atom_ids in selected.items():
            candidate_set = candidate_sets.get(dimension)
            if candidate_set is None:
                reasons.append(f"UNKNOWN_DIMENSION:{dimension}")
                continue
            allowed = {item.atom_id for item in candidate_set.candidates}
            if len(atom_ids) != len(set(atom_ids)) or len(atom_ids) > 1:
                reasons.append(f"INVALID_ATOM_CARDINALITY:{dimension}")
            if candidate_set.locked_atom_ids and tuple(atom_ids) != candidate_set.locked_atom_ids:
                reasons.append(f"LOCKED_BINDING_CHANGED:{dimension}")
            for atom_id in atom_ids:
                if atom_id not in allowed:
                    atom = self.by_id.get(atom_id)
                    if atom is None:
                        reasons.append(f"INVENTED_ATOM_ID:{dimension}:{atom_id}")
                    elif dimension not in atom.get("filled_dimensions", []):
                        reasons.append(f"WRONG_DIMENSION:{dimension}:{atom_id}")
                    else:
                        reasons.append(f"ATOM_OUTSIDE_CANDIDATE_SET:{dimension}:{atom_id}")
                    continue
                atom = self.by_id.get(atom_id)
                if atom is None:
                    reasons.append(f"UNKNOWN_ATOM:{atom_id}")
                elif dimension not in atom.get("filled_dimensions", []):
                    reasons.append(f"WRONG_DIMENSION:{dimension}:{atom_id}")
        if partial:
            return sorted(set(reasons))
        for dimension in self.dimensions:
            if dimension not in selected:
                continue
            candidate_set = candidate_sets[dimension]
            applicability = candidate_set.applicability.status
            if not selected[dimension] and applicability is ScenarioDimensionApplicability.REQUIRED:
                reasons.append(f"REQUIRED_DIMENSION_EMPTY:{dimension}")
            if selected[dimension] and applicability is ScenarioDimensionApplicability.NOT_APPLICABLE:
                reasons.append(f"NOT_APPLICABLE_DIMENSION_SELECTED:{dimension}")
            decision = candidate_set.binding_decision
            if (
                decision.refinable and decision.parent_atom_id
                and selected[dimension]
                and selected[dimension] != (decision.parent_atom_id,)
            ):
                candidate = next(
                    (item for item in candidate_set.candidates if item.atom_id == selected[dimension][0]),
                    None,
                )
                refinement_status, _ = ScenarioRefinementEvidencePolicy.classify(
                    candidate, decision,
                ) if candidate is not None else ("REFINEMENT_UNSUPPORTED", "MISSING_CANDIDATE")
                if refinement_status == "REFINEMENT_UNSUPPORTED":
                    reasons.append(f"UNSUPPORTED_BINDING_REFINEMENT:{dimension}")
        selected_by_dimension = {
            dimension: set(atom_ids) for dimension, atom_ids in selected.items()
        }
        selected_union = {
            atom_id for atom_ids in selected_by_dimension.values() for atom_id in atom_ids
        }
        for atom_id in selected_union:
            atom = self.by_id.get(atom_id)
            if atom is None:
                continue
            filled = {
                str(item) for item in atom.get("filled_dimensions", [])
                if str(item) in expected
            }
            missing = sorted(
                dimension for dimension in filled
                if atom_id not in selected_by_dimension.get(dimension, set())
            )
            if missing:
                reasons.append(f"COMPOUND_ATOM_INCOMPLETE:{atom_id}:{','.join(missing)}")
            conflicts = sorted(
                dimension for dimension in filled
                if selected_by_dimension.get(dimension, set()) - {atom_id}
            )
            if conflicts:
                reasons.append(f"COMPOUND_ATOM_CONFLICT:{atom_id}:{','.join(conflicts)}")
        contextual_speed = synthesis_input.contextual_speed
        if contextual_speed.get("status") == "RESOLVED":
            envelope = (
                contextual_speed.get("min_kph"), contextual_speed.get("max_kph"),
            )
        else:
            parent = synthesis_input.parent_scenario
            parent_facts = parent.get("facts", {}) if isinstance(parent, dict) else {}
            envelope_raw = parent_facts.get("ego_speed_constraint", {})
            envelope = (
                envelope_raw.get("min_kph", envelope_raw.get("speed_min_kph")),
                envelope_raw.get("max_kph", envelope_raw.get("speed_max_kph")),
            ) if isinstance(envelope_raw, dict) else (None, None)
        for atom_id in selected_union:
            atom = self.by_id.get(atom_id)
            if atom is not None and not self._speed_compatible(atom, envelope):
                reasons.append(f"ODD_SPEED_INCOMPATIBLE:{atom_id}")
        bindings = {
            dimension: {
                "resolution_status": "RESOLVED" if atom_ids else "PENDING",
                "atom_id": atom_ids[0] if atom_ids else "",
                "method_value": atom_ids[0] if atom_ids else "",
            }
            for dimension, atom_ids in selected.items()
        }
        constraint = self.constraint_executor.evaluate(
            bindings, self.method.scenario_model.constraint_rules,
        )
        if constraint.status in {ScenarioConstraintStatus.DROP, ScenarioConstraintStatus.CONFLICT}:
            reasons.append(f"METHOD_CONSTRAINT:{constraint.status.value}:{constraint.reason}")
        return sorted(set(reasons))

    @staticmethod
    def _primary_validation_reason(reasons: list[str]) -> str:
        priority = (
            "INVENTED_ATOM_ID:", "WRONG_DIMENSION:",
            "ATOM_OUTSIDE_CANDIDATE_SET:", "UNKNOWN_DIMENSION:",
            "DIMENSION_KEY_SET_MISMATCH", "INVALID_ATOM_CARDINALITY:",
            "LOCKED_BINDING_CHANGED:", "REQUIRED_DIMENSION_EMPTY:",
            "NOT_APPLICABLE_DIMENSION_SELECTED:",
            "UNSUPPORTED_BINDING_REFINEMENT:",
            "COMPOUND_ATOM_CONFLICT:", "COMPOUND_ATOM_INCOMPLETE:",
            "ODD_SPEED_INCOMPATIBLE:", "METHOD_CONSTRAINT:",
        )
        return next(
            (reason for prefix in priority for reason in reasons if reason.startswith(prefix)),
            reasons[0],
        )

    @staticmethod
    def _same_physical_value(left: Any, right: Any) -> bool:
        if (
            isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)
        ):
            return float(left) == float(right)
        return str(left).strip().casefold() == str(right).strip().casefold()

    def _selected_template_option(
        self,
        synthesis_input: ScenarioSynthesisInput,
        selected_candidates: dict[str, ScenarioDimensionCandidate],
    ) -> tuple[dict[str, Any] | None, str, list[str]]:
        """Resolve exact or uniquely constrained FM option; never choose first."""

        template = synthesis_input.fm_scenario_template
        if not isinstance(template, dict) or not template:
            return None, "NO_APPLICABLE_FM_TEMPLATE", []
        active = template.get("active_option")
        if isinstance(active, dict) and active.get("source_option_id"):
            return active, "EXACT_ACTIVE_OPTION", [str(active["source_option_id"])]
        options = [
            item for item in template.get("source_governed_constraints", [])
            if isinstance(item, dict) and item.get("source_option_id")
        ]
        object_categories: set[str] = set()
        object_positions: set[str] = set()
        collision_values: set[str] = set()
        for candidate in selected_candidates.values():
            if "OBJECT" not in candidate.dimensions:
                continue
            source_atom = self.by_id.get(candidate.canonical_atom_id, {})
            object_categories.update(
                item for item in self.ranker.atom_categories(source_atom)
                if item.startswith("OBJECT_")
            )
            semantics = candidate.method_semantics
            semantics = semantics if isinstance(semantics, dict) else {}
            obj = semantics.get("object", {})
            obj = obj if isinstance(obj, dict) else {}
            if category := normalize_object_category(obj.get("type", "")):
                object_categories.add(category)
            if position := str(obj.get("position", "")).strip().casefold():
                object_positions.add(position)
            if collision := str(semantics.get("collision_type", "")).strip():
                resolution = self.scenario_method.risk_vocabulary.resolve(
                    field="collision_type", raw_value=collision,
                )
                if resolution.mapped:
                    collision_values.add(resolution.canonical_value)
        if not (object_categories or object_positions or collision_values):
            return None, "AMBIGUOUS_FM_TEMPLATE_OPTION", [
                str(item["source_option_id"]) for item in options
            ]
        matches = []
        for option in options:
            category = normalize_object_category(option.get("obj_type", ""))
            position = str(option.get("obj_position", "")).strip().casefold()
            collision = self.scenario_method.risk_vocabulary.resolve(
                field="collision_type", raw_value=option.get("collision_type", ""),
            )
            if object_categories and category not in object_categories:
                continue
            if object_positions and position not in object_positions:
                continue
            if collision_values and collision.canonical_value not in collision_values:
                continue
            matches.append(option)
        if len(matches) == 1:
            return matches[0], "UNIQUE_STRUCTURED_OPTION_MATCH", [
                str(matches[0]["source_option_id"])
            ]
        return None, "AMBIGUOUS_FM_TEMPLATE_OPTION", [
            str(item["source_option_id"]) for item in matches
        ]

    def _project_method_physical_facts(
        self, *, synthesis_input: ScenarioSynthesisInput,
        selected_candidates: dict[str, ScenarioDimensionCandidate],
        facts: dict[str, Any], provenance: dict[str, Any],
        scope: dict[str, str],
    ) -> dict[str, Any]:
        option, resolution, matching_ids = self._selected_template_option(
            synthesis_input, selected_candidates,
        )
        diagnostics: dict[str, Any] = {
            "option_resolution": resolution,
            "matching_option_ids": matching_ids,
            "projected_fields": [],
            "matching_existing_fields": [],
            "conflicts": [],
            "unmapped_values": [],
        }

        def set_fact(
            field: str, value: Any, *, source: SourceRef,
            origin: str, fact_provenance: str,
            selection_basis: str, extra: dict[str, Any] | None = None,
        ) -> bool:
            current = facts.get(field)
            if current not in (None, "") and not self._same_physical_value(current, value):
                diagnostics["conflicts"].append({
                    "field": field, "existing_value": current,
                    "method_value": value,
                })
                return False
            if current not in (None, ""):
                diagnostics["matching_existing_fields"].append(field)
            facts[field] = value
            provenance[field] = {
                "provenance": fact_provenance,
                "origin": origin,
                "approval": ReviewStatus.FINALIZED.value,
                "source_refs": [{
                    "source_type": source.source_type,
                    "source_id": source.source_id,
                    "location": source.location,
                    "excerpt": source.excerpt,
                }],
                "applicable_scope": scope,
                "validation_status": "VALIDATED",
                "selection_basis": selection_basis,
                **(extra or {}),
            }
            diagnostics["projected_fields"].append(field)
            return True

        option_source: SourceRef | None = None
        if option is not None:
            source_payload = option.get("source", {})
            source_payload = source_payload if isinstance(source_payload, dict) else {}
            option_source = SourceRef(
                "method_contract", synthesis_input.method_contract_hash,
                str(source_payload.get("source_rule", "")),
                str(source_payload.get("source_excerpt", option.get("label", ""))),
            )
            option_id = str(option.get("source_option_id", ""))
            template_id = str(
                synthesis_input.fm_scenario_template.get("template_id", "")
            )
            values = {
                "object_type": option.get("obj_type"),
                "object_position": option.get("obj_position"),
                "relative_distance_m": option.get("obj_distance_m"),
                "object_speed_kph": option.get("obj_v_kph"),
            }
            mappings: dict[str, Any] = {}
            for field, raw_field, target_field in (
                ("road_user_type", "obj_type", "road_user_type"),
                ("collision_type", "collision_type", "collision_type"),
            ):
                mapped = self.scenario_method.risk_vocabulary.resolve(
                    field=field, raw_value=option.get(raw_field, ""),
                )
                if mapped.mapped:
                    values[target_field] = mapped.canonical_value
                    mappings[target_field] = mapped.to_dict()
                else:
                    diagnostics["unmapped_values"].append({
                        "field": target_field, "raw_value": option.get(raw_field),
                    })
            for field, value in values.items():
                if value is None or value == "":
                    diagnostics["unmapped_values"].append({
                        "field": field, "raw_value": value,
                    })
                    continue
                set_fact(
                    field, value, source=option_source,
                    origin="METHOD_DEFINED",
                    fact_provenance=FactProvenance.METHOD_CONTRACT.value,
                    selection_basis=resolution,
                    extra={
                        "source_template_id": template_id,
                        "source_option_id": option_id,
                        "method_contract_hash": synthesis_input.method_contract_hash,
                        **(
                            {"risk_vocabulary_mapping": mappings[field]}
                            if field in mappings else {}
                        ),
                    },
                )
            speed = option.get("obj_v_kph")
            if (
                isinstance(speed, (int, float)) and not isinstance(speed, bool)
                and float(speed) == 0.0 and option_source is not None
            ):
                set_fact(
                    "object_longitudinal_direction", "STATIONARY",
                    source=option_source, origin=FactProvenance.DERIVED.value,
                    fact_provenance=FactProvenance.DERIVED.value,
                    selection_basis="OBJECT_SPEED_EXACT_ZERO",
                    extra={
                        "source_template_id": template_id,
                        "source_option_id": option_id,
                        "method_contract_hash": synthesis_input.method_contract_hash,
                        "inputs": ["object_speed_kph"],
                    },
                )

        directions: list[tuple[str, ScenarioDimensionCandidate]] = []
        for candidate in selected_candidates.values():
            semantics = candidate.method_semantics
            semantics = semantics if isinstance(semantics, dict) else {}
            dynamics = semantics.get("ego_dynamics", {})
            dynamics = dynamics if isinstance(dynamics, dict) else {}
            direction = str(dynamics.get("direction", "")).strip().upper()
            if direction in {"FORWARD", "REVERSE"}:
                directions.append((direction, candidate))
        unique_directions = {item[0] for item in directions}
        if len(unique_directions) == 1:
            direction, candidate = directions[0]
            atom_source = SourceRef(
                "method_contract", synthesis_input.method_contract_hash,
                candidate.source_rule, candidate.label,
            )
            set_fact(
                "ego_longitudinal_direction", direction,
                source=atom_source, origin="METHOD_DEFINED",
                fact_provenance=FactProvenance.METHOD_CONTRACT.value,
                selection_basis="SELECTED_ATOM_PHYSICAL_SEMANTICS",
                extra={
                    "source_atom_ids": sorted({item[1].atom_id for item in directions}),
                    "method_contract_hash": synthesis_input.method_contract_hash,
                },
            )
        elif len(unique_directions) > 1:
            diagnostics["conflicts"].append({
                "field": "ego_longitudinal_direction",
                "method_values": sorted(unique_directions),
            })
        return diagnostics

    def validate_provider_payload(
        self, synthesis_input: ScenarioSynthesisInput, payload: dict[str, Any],
    ) -> tuple[ScenarioSynthesisAssessment, ...]:
        if not isinstance(payload, dict) or set(payload) != {"variants"}:
            raise ScenarioSynthesisValidationError(
                "SCHEMA_TOP_LEVEL", "Provider output must contain only variants"
            )
        variants = payload.get("variants")
        desired = synthesis_input.coverage_plan.desired_variant_count
        if not isinstance(variants, list) or len(variants) != desired:
            raise ScenarioSynthesisValidationError(
                "COVERAGE_VARIANT_COUNT",
                f"Coverage Plan requires exactly {desired} variants",
            )
        expected_labels = [
            str(item.get("coverage_label", ""))
            for item in synthesis_input.coverage_plan.variant_intents
        ]
        assessments = []
        fingerprints: set[str] = set()
        for index, raw in enumerate(variants):
            if not isinstance(raw, dict) or set(raw) != {
                "coverage_label", "selected_atom_ids", "semantic_rationale", "context_refs",
            }:
                raise ScenarioSynthesisValidationError(
                    "SCHEMA_VARIANT_FIELDS", f"Variant {index} has invalid fields"
                )
            try:
                coverage = CoverageLabel(str(raw["coverage_label"]))
            except ValueError as exc:
                raise ScenarioSynthesisValidationError(
                    "INVALID_COVERAGE_LABEL", f"Variant {index} coverage label is invalid"
                ) from exc
            if coverage.value != expected_labels[index]:
                raise ScenarioSynthesisValidationError(
                    "COVERAGE_LABEL_PLAN_MISMATCH",
                    f"Variant {index} must implement {expected_labels[index]}",
                )
            selected_raw = raw["selected_atom_ids"]
            if not isinstance(selected_raw, list) or any(
                not isinstance(item, str) for item in selected_raw
            ):
                raise ScenarioSynthesisValidationError(
                    "SCHEMA_SELECTED_ATOM_IDS",
                    "selected_atom_ids must be an atom ID array",
                )
            selected = self.expand_logical_atom_ids(synthesis_input, selected_raw)
            rationale = raw["semantic_rationale"]
            refs = raw["context_refs"]
            if not isinstance(rationale, str) or not rationale.strip():
                raise ScenarioSynthesisValidationError(
                    "MISSING_RATIONALE", "Every variant requires a rationale"
                )
            if not isinstance(refs, list) or not refs or any(
                not isinstance(item, str) or not item.strip() for item in refs
            ):
                raise ScenarioSynthesisValidationError(
                    "INVALID_CONTEXT_REFS", "Every variant requires supplied context refs"
                )
            allowed_refs = {
                "PROJECT.ODD", "MF.description", "MF.functional_effect",
                "MF.vehicle_level_hazard", "HE.hazardous_event",
                "CAUSAL.summary", "PARENT.scenario", "METHOD.scenario_atom_catalog",
                "METHOD.fm_scenario_template",
            }
            if any(item not in allowed_refs for item in refs):
                raise ScenarioSynthesisValidationError(
                    "CONTEXT_REF_OUTSIDE_SUPPLIED_SET", "Provider invented a context ref"
                )
            reasons = self._selection_reasons(synthesis_input, selected)
            if reasons:
                raise ScenarioSynthesisValidationError(
                    self._primary_validation_reason(reasons), "; ".join(reasons),
                    details=reasons,
                )
            candidate_sets = {
                item.dimension: {candidate.atom_id: candidate.canonical_atom_id
                                 for candidate in item.candidates}
                for item in synthesis_input.dimension_candidate_sets
            }
            fingerprint = self._canonical_json({
                dimension: [candidate_sets[dimension][atom_id] for atom_id in atom_ids]
                for dimension, atom_ids in selected.items()
            })
            if fingerprint in fingerprints:
                raise ScenarioSynthesisValidationError(
                    "DUPLICATE_VARIANT", "Equivalent canonical atom sets are duplicated"
                )
            fingerprints.add(fingerprint)
            assessments.append(ScenarioSynthesisAssessment(
                semantic_group_id=synthesis_input.semantic_group_id,
                coverage_label=coverage, selected_atoms=selected,
                semantic_rationale=rationale.strip(), context_refs=tuple(refs),
                validation_status=SynthesisValidationStatus.VALIDATED,
                selection_authority="BOUNDED_PROVIDER_LOGICAL_ATOM_SELECTION",
            ))
        diversity_reasons = self.diversity_validator.reasons(
            synthesis_input.coverage_plan,
            (item.selected_atoms for item in assessments),
        )
        if diversity_reasons:
            raise ScenarioSynthesisValidationError(
                diversity_reasons[0], "; ".join(diversity_reasons),
                details=diversity_reasons,
            )
        return tuple(assessments)

    def materialize(
        self, *, synthesis_input: ScenarioSynthesisInput,
        assessment: ScenarioSynthesisAssessment, parent: ScenarioCandidate,
        provider_evidence: dict[str, Any],
    ) -> tuple[ScenarioCandidate, AnalyticalScenarioInstantiation]:
        reasons = self._selection_reasons(synthesis_input, assessment.selected_atoms)
        if reasons:
            raise ScenarioSynthesisValidationError(
                self._primary_validation_reason(reasons), "; ".join(reasons),
                details=reasons,
            )
        selected_candidates: dict[str, ScenarioDimensionCandidate] = {}
        candidate_sets = {item.dimension: item for item in synthesis_input.dimension_candidate_sets}
        for dimension, atom_ids in assessment.selected_atoms.items():
            by_id = {item.atom_id: item for item in candidate_sets[dimension].candidates}
            for atom_id in atom_ids:
                selected_candidates[atom_id] = by_id[atom_id]
        material = {
            "contract": synthesis_input.contract_version,
            "method_contract_hash": synthesis_input.method_contract_hash,
            "malfunction_id": synthesis_input.malfunction_id,
            "parent_scenario_id": synthesis_input.parent_scenario_id,
            "hazardous_event_id": synthesis_input.hazardous_event_id,
            "selected_atoms": {
                key: list(value) for key, value in sorted(assessment.selected_atoms.items())
            },
        }
        fingerprint = hashlib.sha256(
            self._canonical_json(material).encode("utf-8")
        ).hexdigest()
        child_id = f"SCN-ANALYTICAL-{fingerprint[:16].upper()}"
        bindings: dict[str, dict[str, Any]] = {}
        for dimension in self.dimensions:
            atom_ids = assessment.selected_atoms.get(dimension, ())
            candidate_set = candidate_sets[dimension]
            decision = candidate_set.binding_decision
            applicability = candidate_set.applicability
            if not atom_ids:
                not_applicable = (
                    applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE
                )
                bindings[dimension] = {
                    "project_value": "", "method_value": "",
                    "binding_status": "NOT_APPLICABLE" if not_applicable else "MISSING",
                    "resolution_status": "NOT_APPLICABLE" if not_applicable else "PENDING",
                    "unresolved_reason": (
                        applicability.reason if not_applicable else candidate_set.generation_status
                    ),
                    "candidate_atom_ids": [
                        item.atom_id for item in candidate_set.candidates
                    ],
                    "applicability_status": applicability.status.value,
                    "applicability_reason": applicability.reason,
                    "applicability_trigger_evidence": list(applicability.trigger_evidence),
                    "binding_authority": decision.authority.value,
                    "refinable": decision.refinable,
                    "parent_atom_id": decision.parent_atom_id,
                    "binding_source_refs": list(decision.source_refs),
                    "binding_basis": decision.basis,
                    "dimension_source": "MethodContract.scenario_model",
                }
                continue
            candidate = selected_candidates[atom_ids[0]]
            refined = bool(
                decision.parent_atom_id
                and candidate.atom_id != decision.parent_atom_id
            )
            project_envelope = decision.project_speed_envelope_kph
            child_range = candidate.speed_range_kph
            bindings[dimension] = {
                "project_value": str(
                    parent.facts.get("method_scenario_dimensions", {})
                    .get(dimension, {}).get("project_value", "")
                ),
                "method_value": f"{candidate.atom_id} | {candidate.label}",
                "binding_status": (
                    "REFINED" if refined else
                    "EXACT" if not decision.refinable else "ANALYTICAL"
                ),
                "resolution_status": "RESOLVED",
                "unresolved_reason": "", "candidate_atom_ids": list(atom_ids),
                "atom_id": candidate.atom_id,
                "canonical_atom_id": candidate.canonical_atom_id,
                "filled_dimensions": list(candidate.dimensions),
                "atom_provenance": {
                    "source_asset": candidate.source_asset,
                    "source_rule": candidate.source_rule,
                    "source_tag": candidate.source_tag,
                },
                "method_semantics": deepcopy(candidate.method_semantics),
                "selection_origin": candidate.candidate_origin.value,
                "selection_reason": assessment.semantic_rationale,
                "applicability_status": applicability.status.value,
                "applicability_reason": applicability.reason,
                "applicability_trigger_evidence": list(applicability.trigger_evidence),
                "binding_authority": (
                    ScenarioBindingAuthority.ANALYTICAL_SELECTION.value
                    if refined else decision.authority.value
                ),
                "refinable": decision.refinable,
                "parent_atom_id": decision.parent_atom_id,
                "parent_method_atom": decision.parent_atom_id,
                "child_refined_atom": candidate.atom_id if refined else "",
                "binding_source_refs": list(decision.source_refs),
                "binding_basis": decision.basis,
                "project_speed_envelope_kph": (
                    list(project_envelope) if project_envelope is not None else None
                ),
                "parent_speed_range_kph": (
                    list(decision.parent_speed_range_kph)
                    if decision.parent_speed_range_kph is not None else None
                ),
                "child_speed_range_kph": (
                    list(child_range) if child_range is not None else None
                ),
                "speed_intersection_kph": (
                    list(intersection)
                    if (
                        intersection := self._range_intersection(
                            project_envelope, child_range,
                        )
                    ) is not None else None
                ),
                "ranking_scores": dict(candidate.ranking_scores),
                "template_relationship": candidate.template_relationship,
                "dimension_source": "MethodContract.scenario_model",
            }
        method_source = SourceRef(
            "method_contract", synthesis_input.method_contract_hash,
            "scenario_model.scenario_atom_catalog", "bounded Scenario atom selection",
        )
        facts = deepcopy(parent.facts)
        facts["method_scenario_dimensions"] = bindings
        contextual_speed = deepcopy(synthesis_input.contextual_speed)
        if contextual_speed:
            facts["speed_context_resolution"] = contextual_speed
        if contextual_speed.get("status") == "RESOLVED":
            facts["ego_speed_constraint"] = {
                "min_kph": contextual_speed.get("min_kph"),
                "max_kph": contextual_speed.get("max_kph"),
                "unit": "km/h",
            }
        facts["scenario_atom_ids"] = list(dict.fromkeys(
            selected_candidates[atom_id].canonical_atom_id
            for dimension in self.dimensions
            for atom_id in assessment.selected_atoms.get(dimension, ())
        ))
        for dimension, fact_key in {
            "WHERE": "operating_scenario",
            "ROAD": "road_surface_conditions",
            "EGO_ACTION": "vehicle_state",
            "EGO_DYNAMICS": "ego_dynamics",
            "OBJECT": "scenario_object_atom",
            "TRAFFIC_PATTERN": "traffic_pattern",
            "EGO_X_ROAD": "ego_road_relation",
        }.items():
            atom_ids = assessment.selected_atoms.get(dimension, ())
            if atom_ids:
                facts[fact_key] = selected_candidates[atom_ids[0]].label
        scope = {
            "malfunction_id": synthesis_input.malfunction_id,
            "scenario_id": child_id,
            "parent_scenario_id": synthesis_input.parent_scenario_id,
            "hazardous_event_id": synthesis_input.hazardous_event_id,
        }
        provenance = deepcopy(parent.fact_provenance)
        if contextual_speed.get("status") == "RESOLVED":
            provenance["ego_speed_constraint"] = {
                "provenance": FactProvenance.PROJECT_INPUT.value,
                "origin": "PROJECT_DEFINED",
                "approval": ReviewStatus.FINALIZED.value,
                "source_refs": deepcopy(contextual_speed.get("source_refs", [])),
                "applicable_scope": scope,
                "selection_basis": "EXPLICIT_OPERATIONAL_CONTEXT_SPEED_ENVELOPE",
                "selected_context": contextual_speed.get("selected_context", ""),
            }
        provenance["scenario_atom_ids"] = {
            "provenance": FactProvenance.DERIVED.value,
            "origin": "METHOD_DEFINED",
            "approval": ReviewStatus.FINALIZED.value,
            "source_refs": [{
                "source_type": method_source.source_type,
                "source_id": method_source.source_id,
                "location": method_source.location,
                "excerpt": method_source.excerpt,
            }],
            "applicable_scope": scope,
            "selection_authority": assessment.selection_authority,
        }
        physical_projection = self._project_method_physical_facts(
            synthesis_input=synthesis_input,
            selected_candidates=selected_candidates,
            facts=facts, provenance=provenance, scope=scope,
        )
        speed_binding = bindings.get("EGO_DYNAMICS", {})
        speed_intersection = (
            speed_binding.get("speed_intersection_kph")
            if isinstance(speed_binding, dict) else None
        )
        if (
            isinstance(speed_intersection, list)
            and len(speed_intersection) == 2
            and any(value is not None for value in speed_intersection)
        ):
            parent_constraint = facts.get("ego_speed_constraint", {})
            parent_constraint = (
                deepcopy(parent_constraint) if isinstance(parent_constraint, dict) else {}
            )
            parent_constraint["min_kph"] = speed_intersection[0]
            parent_constraint["max_kph"] = speed_intersection[1]
            facts["ego_speed_constraint"] = parent_constraint
            speed_candidate = next((
                candidate for candidate in selected_candidates.values()
                if "EGO_DYNAMICS" in candidate.dimensions
                and candidate.speed_range_kph is not None
            ), None)
            speed_source = SourceRef(
                "method_contract", synthesis_input.method_contract_hash,
                speed_candidate.source_rule if speed_candidate else method_source.location,
                speed_candidate.label if speed_candidate else method_source.excerpt,
            )
            parent_metadata = provenance.get("ego_speed_constraint", {})
            parent_metadata = parent_metadata if isinstance(parent_metadata, dict) else {}
            parent_sources = deepcopy(parent_metadata.get("source_refs", []))
            if not parent_sources:
                parent_sources = [{
                    "source_type": "parent_scenario",
                    "source_id": parent.scenario_id,
                    "location": "facts.ego_speed_constraint",
                    "excerpt": self._canonical_json(parent.facts.get("ego_speed_constraint", {})),
                }]
            provenance["ego_speed_constraint"] = {
                "provenance": FactProvenance.DERIVED.value,
                "origin": "METHOD_DEFINED",
                "approval": ReviewStatus.FINALIZED.value,
                "source_refs": [*parent_sources, {
                    "source_type": speed_source.source_type,
                    "source_id": speed_source.source_id,
                    "location": speed_source.location,
                    "excerpt": speed_source.excerpt,
                }],
                "applicable_scope": scope,
                "validation_status": "VALIDATED",
                "selection_basis": "PROJECT_AND_METHOD_SPEED_RANGE_INTERSECTION",
                "source_atom_ids": [speed_candidate.atom_id] if speed_candidate else [],
                "project_speed_envelope_kph": speed_binding.get("project_speed_envelope_kph"),
                "child_speed_range_kph": speed_binding.get("child_speed_range_kph"),
                "speed_intersection_kph": list(speed_intersection),
                "method_contract_hash": synthesis_input.method_contract_hash,
            }
            physical_projection["projected_fields"].append("ego_speed_constraint")
        validations = (
            {"check": "candidate_membership", "status": "PASS"},
            {"check": "binding_refinement_policy", "status": "PASS"},
            {"check": "dimension_applicability", "status": "PASS"},
            {"check": "compound_atom_integrity", "status": "PASS"},
            {"check": "odd_speed_intersection", "status": "PASS"},
            {"check": "method_constraints", "status": "PASS"},
            {"check": "coverage_plan", "status": "PASS"},
            {"check": "sibling_non_trivial_diversity", "status": "PASS"},
            {
                "check": "method_physical_projection",
                "status": (
                    "PASS" if not physical_projection["conflicts"]
                    else "CONFLICT_RETAINED_PARENT"
                ),
            },
            {"check": "e_biased_ranking", "status": "NOT_USED"},
        )
        contextual_fact_refs = tuple(
            "PROJECT.SpeedEnvelope@" + str(item.get("location", ""))
            for item in contextual_speed.get("source_refs", [])
            if isinstance(item, dict) and str(item.get("location", ""))
        )
        instantiation = AnalyticalScenarioInstantiation(
            scenario_id=child_id,
            parent_scenario_id=synthesis_input.parent_scenario_id,
            malfunction_id=synthesis_input.malfunction_id,
            hazardous_event_id=synthesis_input.hazardous_event_id,
            selected_atoms=tuple(selected_candidates.values()),
            dimension_bindings=bindings,
            deterministic_validations=validations,
            provider_evidence=dict(provider_evidence),
            project_facts_used=tuple(dict.fromkeys((
                "PROJECT.ODD", "PARENT.scenario", *contextual_fact_refs,
            ))),
            method_facts_used=tuple(sorted({
                f"{item.source_asset}:{item.source_rule}"
                for item in selected_candidates.values()
            })),
        )
        instance = {
            "instance_id": child_id,
            "parent_scenario_id": synthesis_input.parent_scenario_id,
            "malfunction_id": synthesis_input.malfunction_id,
            "hazardous_event_id": synthesis_input.hazardous_event_id,
            "semantic_group_id": synthesis_input.semantic_group_id,
            "structured_semantic_query": deepcopy(
                synthesis_input.structured_semantic_query
            ),
            "contextual_speed": contextual_speed,
            "coverage_plan": synthesis_input.coverage_plan.to_dict(),
            "fm_scenario_template_id": synthesis_input.fm_scenario_template.get(
                "template_id", ""
            ),
            "selected_atoms": facts["scenario_atom_ids"],
            "dimension_bindings": bindings,
            "project_facts_used": list(instantiation.project_facts_used),
            "method_facts_used": list(instantiation.method_facts_used),
            "provider_evidence": provider_evidence,
            "deterministic_validations": list(validations),
            "method_physical_projection": physical_projection,
            "base_analysis_instance": deepcopy(parent.analysis_instance),
            "synthesis_version": synthesis_input.contract_version,
            "validation_status": "VALIDATED",
            "applicable_scope": scope,
        }
        detail = " | ".join(
            f"{dimension}={','.join(atom_ids) if atom_ids else 'PENDING'}"
            for dimension, atom_ids in assessment.selected_atoms.items()
        )
        child = replace(
            parent, scenario_id=child_id,
            operating_scenario=str(facts.get("operating_scenario", parent.operating_scenario)),
            situational_description=detail, situational_detailing=detail,
            facts=facts, context_resolution={
                **deepcopy(parent.context_resolution),
                "scenario_synthesis": {
                    "semantic_group_id": synthesis_input.semantic_group_id,
                    "coverage_label": assessment.coverage_label.value,
                    "status": ScenarioSynthesisStatus.METHOD_VALID.value,
                },
            },
            fact_provenance=provenance, status=ReviewStatus.PENDING,
            sources=list(dict.fromkeys([*parent.sources, method_source])),
            review_reason=(
                "Bounded analytical Scenario is method-valid; causal reuse/revalidation "
                "and any physical engineering assumptions remain separate gates."
            ),
            source_scenario_id=parent.scenario_id,
            atomic_variant=f"scenario_synthesis:{assessment.coverage_label.value}",
            semantic_fingerprint=fingerprint,
            analysis_instance=instance,
        )
        return child, instantiation


__all__ = [
    "ConstrainedScenarioSynthesisService", "ScenarioSynthesisValidationError",
]
