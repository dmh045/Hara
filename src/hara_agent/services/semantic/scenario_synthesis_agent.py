from __future__ import annotations

import json
import re
import time
from typing import Any

from hara_agent.contracts import ScenarioSynthesisAssessment, ScenarioSynthesisInput
from hara_agent.infrastructure.llm.protocol import LLMClient, LLMRequest
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
from hara_agent.services.analysis.scenario_synthesis_service import (
    ConstrainedScenarioSynthesisService, ScenarioSynthesisValidationError,
)


SCENARIO_SYNTHESIS_PROMPT_VERSION = "p5-f-logical-atom-selection-v4"


class BoundedScenarioSynthesisAgent:
    """Select only from deterministically narrowed Method atom candidates."""

    system_prompt = """You perform bounded semantic selection for a HARA analytical Scenario.
You are not an engineering-rule authority. Select each logical Method atom once, using only IDs in the supplied logical candidate registry.
Never invent IDs, context references, project facts, numeric physics values, source status, S/E/C, ASIL, or Exposure ratings.
Preserve bindings marked non-refinable; refinable parent bindings may be replaced only by supplied compatible candidates. The runtime expands every selected logical atom across all of its filled_dimensions.
Obey each supplied dimension applicability decision after that expansion: REQUIRED means exactly one atom, NOT_APPLICABLE means none, and OPTIONAL means zero or one atom.
Never repeat a compound atom per dimension. Return one selected_atom_ids array per variant. Do not select two logical atoms whose filled_dimensions overlap.
First choose exactly one whole object from coverage_valid_logical_atom_set_assignments. For each coverage label, copy selected_atom_ids exactly from that object's array: do not mix assignments or add, remove, or replace IDs. These whole assignments already satisfy the Coverage Plan and sibling primary-dimension diversity. A compound atom already fills every dimension named in its filled_dimensions; never add another atom for one of those dimensions merely for semantics, specificity, or sibling diversity.
Implement the supplied Scenario Coverage Plan exactly. Typical is the representative mechanism case; boundary is near a relevant project/Method/interaction boundary; extreme is more demanding but still project-valid. Never optimize risk, Exposure, S, E, C, or ASIL.
Sibling variants must differ on a primary variation dimension when more than one is requested; environment-only variation is invalid unless the Coverage Plan marks that environment dimension primary.
Return raw JSON matching the schema exactly, without Markdown."""

    def __init__(
        self, client: LLMClient, validator: ConstrainedScenarioSynthesisService,
    ):
        self.client = client
        self.validator = validator

    def _schema(self, synthesis_input: ScenarioSynthesisInput) -> dict[str, Any]:
        ids = list(self.validator.logical_candidate_registry(synthesis_input))
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False,
            "required": ["variants"],
            "properties": {
                "variants": {
                    "type": "array",
                    "minItems": synthesis_input.coverage_plan.desired_variant_count,
                    "maxItems": synthesis_input.coverage_plan.desired_variant_count,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": [
                            "coverage_label", "selected_atom_ids",
                            "semantic_rationale", "context_refs",
                        ],
                        "properties": {
                            "coverage_label": {
                                "type": "string",
                                "enum": [
                                    str(item["coverage_label"])
                                    for item in synthesis_input.coverage_plan.variant_intents
                                ],
                            },
                            "selected_atom_ids": {
                                "type": "array", "minItems": 1,
                                "maxItems": len(synthesis_input.dimension_candidate_sets),
                                "uniqueItems": True,
                                "items": {"type": "string", "enum": ids},
                            },
                            "semantic_rationale": {"type": "string", "minLength": 1},
                            "context_refs": {
                                "type": "array", "minItems": 1, "uniqueItems": True,
                                "items": {
                                    "type": "string", "enum": [
                                        "PROJECT.ODD", "MF.description",
                                        "MF.functional_effect", "MF.vehicle_level_hazard",
                                        "HE.hazardous_event", "CAUSAL.summary",
                                        "PARENT.scenario", "METHOD.scenario_atom_catalog",
                                        "METHOD.fm_scenario_template",
                                    ],
                                },
                            },
                        },
                    },
                },
            },
        }

    def _repair_constraints(
        self, synthesis_input: ScenarioSynthesisInput, repair_error: str,
    ) -> dict[str, Any] | None:
        match = re.search(
            r"COMPOUND_ATOM_CONFLICT:[^:]+:([^,\s]+),([^,\s.]+)",
            repair_error,
        )
        if not match:
            if "REQUIRED_DIMENSION_EMPTY:" in repair_error:
                return {
                    "correction": (
                        "Replace all variant arrays by copying one whole, unchanged "
                        "coverage_valid_logical_atom_set_assignments object."
                    ),
                }
            if any(code in repair_error for code in (
                "TRIVIAL_VARIANT_DIVERSITY", "DUPLICATE_VARIANT",
            )):
                return {
                    "correction": (
                        "Choose different complete, unchanged arrays from "
                        "one whole coverage_valid_logical_atom_set_assignments object; "
                        "do not choose each variant independently."
                    ),
                }
            return None
        registry = self.validator.logical_candidate_registry(synthesis_input)
        first, second = match.group(1), match.group(2)
        if first not in registry or second not in registry:
            return None
        keep, remove = sorted(
            (first, second),
            key=lambda atom_id: (-len(registry[atom_id].dimensions), atom_id),
        )
        return {
            "forbidden_together": [first, second],
            "correction": (
                f"Never return {first} and {second} in the same variant. "
                f"Prefer a whole supplied assignment containing {keep}, which fills "
                f"{list(registry[keep].dimensions)}, without {remove} in that variant. "
                "Do not edit an assignment array yourself."
            ),
        }

    def _user_payload(
        self, synthesis_input: ScenarioSynthesisInput, *, repair_error: str = "",
    ) -> dict[str, Any]:
        malfunction = synthesis_input.malfunction
        parent = synthesis_input.parent_scenario
        parent_facts = parent.get("facts", {}) if isinstance(parent, dict) else {}
        causal = synthesis_input.causal_assessment
        dimension_constraints = {}
        membership = {
            item.dimension: {candidate.atom_id for candidate in item.candidates}
            for item in synthesis_input.dimension_candidate_sets
        }
        logical_candidates: dict[str, dict[str, Any]] = {}
        for item in synthesis_input.dimension_candidate_sets:
            dimension_constraints[item.dimension] = {
                "locked_atom_ids": list(item.locked_atom_ids),
                "generation_status": item.generation_status,
                "applicability": item.applicability.to_dict(),
                "binding_decision": item.binding_decision.to_dict(),
                "catalog_size": item.catalog_size,
                "hard_filtered_pool_size": item.hard_filtered_pool_size,
                "shortlist_truncated": item.shortlist_truncated,
                "shortlist_budget": item.shortlist_budget,
                "shortlist_policy": item.shortlist_policy,
                "shortlist_diagnostics": item.shortlist_diagnostics,
                "hard_filter_diagnostics": item.hard_filter_diagnostics,
            }
            for candidate in item.candidates:
                if any(
                    candidate.atom_id not in membership.get(dimension, set())
                    for dimension in candidate.dimensions
                ):
                    continue
                logical = logical_candidates.setdefault(candidate.atom_id, {
                    "atom_id": candidate.atom_id,
                    "canonical_atom_id": candidate.canonical_atom_id,
                    "filled_dimensions": list(candidate.dimensions),
                    "label": candidate.label,
                    "speed_range_kph": list(candidate.speed_range_kph)
                    if candidate.speed_range_kph is not None else None,
                    "source_asset": candidate.source_asset,
                    "source_rule": candidate.source_rule,
                    "supporting_context_refs": list(candidate.supporting_context_refs),
                    "binding_authority": candidate.binding_authority,
                    "compact_physical_semantics": candidate.method_semantics,
                    "template_relationship": candidate.template_relationship,
                    "evidence_by_dimension": {},
                })
                logical["evidence_by_dimension"][item.dimension] = {
                    "candidate_origin": candidate.candidate_origin.value,
                    "selection_reason": candidate.selection_reason,
                    "ranking_scores": candidate.ranking_scores,
                    "semantic_compatibility": candidate.semantic_compatibility.value,
                    "semantic_family": candidate.semantic_family,
                }
        logical_dimensions = {
            atom_id: set(candidate["filled_dimensions"])
            for atom_id, candidate in logical_candidates.items()
        }
        for atom_id, candidate in logical_candidates.items():
            candidate["conflicts_with_atom_ids"] = sorted(
                other_id for other_id, dimensions in logical_dimensions.items()
                if other_id != atom_id
                and logical_dimensions[atom_id].intersection(dimensions)
            )
        labels = [
            str(item["coverage_label"])
            for item in synthesis_input.coverage_plan.variant_intents
        ]
        assignments = self.validator.logical_selection_assignments(synthesis_input)
        payload = {
            "semantic_group_id": synthesis_input.semantic_group_id,
            "identities": {
                "malfunction_id": synthesis_input.malfunction_id,
                "parent_scenario_id": synthesis_input.parent_scenario_id,
                "hazardous_event_id": synthesis_input.hazardous_event_id,
                "function_id": synthesis_input.function_id,
            },
            "malfunction": {
                key: malfunction.get(key, "") for key in (
                    "guideword", "description", "functional_effect",
                    "vehicle_level_hazard", "component_category", "failure_type",
                )
            },
            "hazard_and_causal_summary": {
                "hazardous_event": causal.get("hazardous_event", ""),
                "causal_chain": causal.get("causal_chain", []),
                "risk_dimension_changes": causal.get("risk_dimension_changes", []),
            },
            "parent_scenario": {
                "operating_scenario": parent.get("operating_scenario", ""),
                "operating_mode": parent.get("operating_mode", ""),
                "situational_description": parent.get("situational_description", ""),
                "facts": {
                    key: parent_facts.get(key) for key in (
                        "ego_speed_constraint", "object_type", "object_position",
                        "road_user_type", "collision_type", "vehicle_state",
                        "operating_scenario", "road_surface_conditions",
                    ) if key in parent_facts
                },
            },
            "project_odd": synthesis_input.project_context,
            "structured_semantic_query": synthesis_input.structured_semantic_query,
            "dimension_applicability": {
                item.dimension: item.applicability.to_dict()
                for item in synthesis_input.dimension_candidate_sets
            },
            "scenario_coverage_plan": synthesis_input.coverage_plan.to_dict(),
            "fm_scenario_template": synthesis_input.fm_scenario_template,
            "dimension_constraints": dimension_constraints,
            "logical_atom_candidates": [
                logical_candidates[atom_id] for atom_id in sorted(logical_candidates)
            ],
            "conflict_free_logical_atom_set_examples": [
                list(item) for item in self.validator.logical_selection_bundles(
                    synthesis_input
                )
            ],
            "coverage_valid_logical_atom_set_assignments": [
                {
                    label: list(atom_ids)
                    for label, atom_ids in zip(labels, assignment)
                }
                for assignment in assignments
            ],
            "required_output_contract": {
                "top_level_keys_exactly": ["variants"],
                "variant_keys_exactly": [
                    "coverage_label", "selected_atom_ids",
                    "semantic_rationale", "context_refs",
                ],
                "coverage_label_enum": [
                    str(item["coverage_label"])
                    for item in synthesis_input.coverage_plan.variant_intents
                ],
                "variant_count": synthesis_input.coverage_plan.desired_variant_count,
                "selected_atom_ids_value_type": (
                    "one non-overlapping exact cover; supplied examples show valid structure"
                ),
                "context_ref_enum": [
                    "PROJECT.ODD", "MF.description", "MF.functional_effect",
                    "MF.vehicle_level_hazard", "HE.hazardous_event",
                    "CAUSAL.summary", "PARENT.scenario",
                    "METHOD.scenario_atom_catalog", "METHOD.fm_scenario_template",
                ],
                "additional_properties": False,
            },
            "explicit_constraints": [
                "IDs must be present in the logical atom candidate registry",
                "use conflict-free set examples to avoid overlapping filled dimensions",
                "choose one whole coverage-valid assignment and copy every label array unchanged",
                "never select an atom together with any ID in its conflicts_with_atom_ids",
                "non-refinable locked atoms must remain unchanged",
                "dimension applicability is deterministic and cannot be changed",
                "variants must satisfy the supplied primary variation dimensions",
                "select every compound once; deterministic code expands filled_dimensions",
                "selected logical atoms must not overlap any filled dimension",
                "do not choose by S/E/C/ASIL or Exposure rating",
                "do not output numeric physics assumptions",
            ],
        }
        repair_constraints = self._repair_constraints(synthesis_input, repair_error)
        if repair_constraints is not None:
            payload["repair_constraints"] = repair_constraints
        return payload

    def _request(
        self, synthesis_input: ScenarioSynthesisInput, *, repair_error: str = "",
    ) -> LLMRequest:
        system = self.system_prompt
        if repair_error:
            repair_constraints = self._repair_constraints(synthesis_input, repair_error)
            system += (
                "\nREPAIR: The previous response was rejected by the deterministic validator: "
                + repair_error
                + ". Re-select from the exact same candidates and return a corrected JSON object."
            )
            if repair_constraints is not None:
                system += " " + repair_constraints["correction"]
        return LLMRequest(
            task="select_scenario_synthesis",
            system_prompt=system,
            user_prompt=json.dumps(
                self._user_payload(synthesis_input, repair_error=repair_error),
                ensure_ascii=False, sort_keys=True,
            ),
            schema_name="ScenarioSynthesisSelection",
            prompt_version=SCENARIO_SYNTHESIS_PROMPT_VERSION,
            metadata={
                "malfunction_id": synthesis_input.malfunction_id,
                "scenario_count": 1,
                "semantic_group_id": synthesis_input.semantic_group_id,
                "repair": bool(repair_error),
            },
            max_tokens=4096,
            response_schema=self._schema(synthesis_input),
        )

    @staticmethod
    def _bounded_raw_selection(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("variants"), list):
            return []
        result = []
        for index, variant in enumerate(payload["variants"][:3]):
            if not isinstance(variant, dict):
                continue
            raw_ids = variant.get("selected_atom_ids", [])
            ids = raw_ids[:12] if isinstance(raw_ids, list) else []
            result.append({
                "variant_index": index,
                "coverage_label": str(variant.get("coverage_label", "")),
                "selected_atom_ids": [str(item)[:128] for item in ids],
                "selection_truncated": isinstance(raw_ids, list) and len(raw_ids) > 12,
            })
        return result

    def _compound_canonicalization_count(
        self, synthesis_input: ScenarioSynthesisInput, payload: Any,
    ) -> int:
        registry = self.validator.logical_candidate_registry(synthesis_input)
        return sum(
            len(registry.get(atom_id).dimensions) > 1
            for variant in self._bounded_raw_selection(payload)
            for atom_id in variant["selected_atom_ids"]
            if registry.get(atom_id) is not None
        )

    def select(
        self, synthesis_input: ScenarioSynthesisInput,
    ) -> tuple[tuple[ScenarioSynthesisAssessment, ...], dict[str, Any]]:
        calls = []
        error = ""
        for attempt in range(2):
            started = time.monotonic()
            response = None
            try:
                response = self.client.complete_json(
                    self._request(synthesis_input, repair_error=error)
                )
                assessments = self.validator.validate_provider_payload(
                    synthesis_input, response.data,
                )
                usage = dict(response.usage)
                calls.append({
                    "attempt": attempt + 1,
                    "request_id": response.request_id,
                    "configured_model": str(getattr(getattr(self.client, "config", None), "model", "")),
                    "resolved_model": response.model,
                    "thinking": str(
                        getattr(getattr(self.client, "config", None), "scenario_thinking", "default")
                    ),
                    "finish_reason": str(usage.get("finish_reason", "")),
                    "reasoning_characters": int(usage.get("reasoning_characters", 0) or 0),
                    "latency_seconds": float(usage.get("latency_seconds", time.monotonic() - started)),
                    "usage": usage,
                    "schema_status": "PASS",
                    "deterministic_validation": "PASS",
                    "method_compound_canonicalizations": (
                        self._compound_canonicalization_count(
                            synthesis_input, response.data,
                        )
                    ),
                })
                return assessments, {
                    "semantic_group_id": synthesis_input.semantic_group_id,
                    "status": "PASS", "calls": calls,
                    "repairs": attempt, "failure_code": "",
                    "method_compound_canonicalizations": sum(
                        int(call.get("method_compound_canonicalizations", 0))
                        for call in calls
                    ),
                }
            except ProviderAttemptBudgetExceeded:
                # The hard Provider cap is an operational stop, not a
                # semantic candidate failure eligible for a repair call.
                raise
            except Exception as exc:
                code = (
                    exc.code if isinstance(exc, ScenarioSynthesisValidationError)
                    else type(exc).__name__
                )
                error = f"{code}: {exc}"
                call = {
                    "attempt": attempt + 1,
                    "configured_model": str(getattr(getattr(self.client, "config", None), "model", "")),
                    "resolved_model": str(response.model) if response is not None else "",
                    "thinking": str(
                        getattr(getattr(self.client, "config", None), "scenario_thinking", "default")
                    ),
                    "finish_reason": str(response.usage.get("finish_reason", "")) if response is not None else "",
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "schema_status": "PASS" if response is not None else "FAIL",
                    "deterministic_validation": "FAIL",
                    "failure_code": code,
                    "failure_reason": str(exc),
                    "failure_details": list(getattr(exc, "details", ())),
                }
                if response is not None:
                    call.update({
                        "request_id": response.request_id,
                        "reasoning_characters": int(
                            response.usage.get("reasoning_characters", 0) or 0
                        ),
                        "usage": dict(response.usage),
                        "response_shape": {
                            "top_level_type": type(response.data).__name__,
                            "top_level_keys": sorted(response.data)
                            if isinstance(response.data, dict) else [],
                            "variant_keys": sorted(response.data["variants"][0])
                            if (
                                isinstance(response.data, dict)
                                and isinstance(response.data.get("variants"), list)
                                and response.data["variants"]
                                and isinstance(response.data["variants"][0], dict)
                            ) else [],
                        },
                        "raw_logical_selection": self._bounded_raw_selection(
                            response.data
                        ),
                        "method_compound_canonicalizations": (
                            self._compound_canonicalization_count(
                                synthesis_input, response.data,
                            )
                        ),
                    })
                calls.append(call)
                if attempt == 1:
                    return (), {
                        "semantic_group_id": synthesis_input.semantic_group_id,
                        "status": "PENDING_SCENARIO_SYNTHESIS", "calls": calls,
                        "repairs": 1, "failure_code": code,
                        "failure_reason": str(exc),
                        "method_compound_canonicalizations": sum(
                            int(call.get("method_compound_canonicalizations", 0))
                            for call in calls
                        ),
                    }
        raise AssertionError("unreachable")


__all__ = [
    "BoundedScenarioSynthesisAgent", "SCENARIO_SYNTHESIS_PROMPT_VERSION",
]
