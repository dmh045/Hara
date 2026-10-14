"""Source-bound readiness checks for native FUSA v1 Exposure inputs.

This service does not implement an Exposure aggregation policy.  It validates
Scenario-to-atom bindings against the compiled MethodContract, then uses the
same ``ExposureMethodExecutor`` preview path as production scoring to prove
whether a still-unresolved, method-relevant dimension could change E.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import islice, product
from typing import Any

from hara_agent.contracts import CalculationStatus, MethodContract

from .deterministic_risk_executor import ExposureMethodExecutor


READY_STATUSES = frozenset({"READY_COMPLETE", "READY_METHOD_IRRELEVANT_GAPS"})
_FINALIZED = CalculationStatus.FINALIZED.value


def _status(result: Mapping[str, Any]) -> str:
    value = result.get("status", "")
    return str(getattr(value, "value", value))


class ExposureInputReadinessService:
    """Assess whether a supplied atom set is complete enough to finalize E.

    Relevance is limited to atom dimensions that the active compiled Exposure
    catalog can use.  This deliberately does not resurrect Function-context
    coverage as a FUSA v1 scoring gate.
    """

    def __init__(
        self, method: MethodContract, executor: ExposureMethodExecutor | None = None,
    ):
        if method.structured_risk_method is None:
            raise ValueError("Exposure readiness requires a structured MethodContract")
        self.method = method
        self.exposure = method.structured_risk_method.exposure
        self.executor = executor or ExposureMethodExecutor()
        self.by_id = {atom.atom_id: atom for atom in self.exposure.atoms}
        self.by_dimension: dict[str, tuple[str, ...]] = {}
        for atom in self.exposure.atoms:
            for dimension in atom.dimensions:
                self.by_dimension.setdefault(str(dimension), ())
                self.by_dimension[str(dimension)] = (
                    *self.by_dimension[str(dimension)], atom.atom_id,
                )

    @staticmethod
    def _bindings(scenario: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        value = scenario.get("method_scenario_dimensions", {})
        if not isinstance(value, Mapping):
            return {}
        return {
            str(name): dict(binding)
            for name, binding in value.items()
            if isinstance(binding, Mapping)
        }

    @staticmethod
    def _atom_ids(scenario: Mapping[str, Any]) -> list[str]:
        raw = scenario.get("scenario_atom_ids", ())
        if not isinstance(raw, (list, tuple)):
            return []
        return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))

    @staticmethod
    def _binding_atom_id(binding: Mapping[str, Any]) -> str:
        # The selected catalog record is the scoring identity. A V1/V2
        # cross-reference does not replace that record in scenario_atom_ids.
        return str(
            binding.get("atom_id") or binding.get("canonical_atom_id") or ""
        ).strip()

    @staticmethod
    def _is_resolved(binding: Mapping[str, Any]) -> bool:
        return str(binding.get("resolution_status", "")).upper() == "RESOLVED"

    @staticmethod
    def _contains(binding: Mapping[str, Any], token: str) -> bool:
        fields = (
            binding.get("resolution_status", ""),
            binding.get("binding_status", ""),
            binding.get("unresolved_reason", ""),
        )
        return any(token in str(value).upper() for value in fields)

    def _domain(self, scenario: Mapping[str, Any]) -> tuple[str, str]:
        category = str(scenario.get("component_category", "")).strip()
        matches = [
            rule for rule in self.exposure.domain_rules
            if category in rule.component_categories
        ]
        if len(matches) != 1:
            return "", "EXPOSURE_COMPONENT_DOMAIN_UNRESOLVED"
        return matches[0].domain.value, ""

    def _candidate_ids(self, dimension: str, binding: Mapping[str, Any]) -> list[str]:
        raw = binding.get("candidate_atom_ids", ())
        candidates = [str(item).strip() for item in raw] if isinstance(raw, (list, tuple)) else []
        if not candidates:
            candidates = list(self.by_dimension.get(dimension, ()))
        return list(dict.fromkeys(
            atom_id for atom_id in candidates
            if atom_id in self.by_id and dimension in self.by_id[atom_id].dimensions
        ))

    def _compatible_completion(self, atom_ids: list[str]) -> bool:
        """A compound atom cannot overwrite a different selected dimension."""
        dimensions: dict[str, str] = {}
        for atom_id in dict.fromkeys(atom_ids):
            for dimension in self.by_id[atom_id].dimensions:
                if dimension in dimensions and dimensions[dimension] != atom_id:
                    return False
                dimensions[dimension] = atom_id
        return True

    @staticmethod
    def _explicit_candidate_ids(binding: Mapping[str, Any]) -> list[str]:
        raw = binding.get("candidate_atom_ids", ())
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def _signature(result: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            _status(result),
            str(result.get("value", "")),
            str(result.get("requested_domain", "")),
            str(result.get("actual_domain", "")),
        )

    @staticmethod
    def _preview(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": _status(result),
            "value": str(result.get("value", "")),
            "requested_domain": str(result.get("requested_domain", "")),
            "actual_domain": str(result.get("actual_domain", "")),
            "dimension_fallback": bool(result.get("dimension_fallback", False)),
            "aggregation_rule": str(result.get("aggregation_rule", "")),
            "coupling": str(result.get("coupling", "")),
            "coupling_consumed": bool(result.get("coupling_consumed", False)),
            "atom_details": list(result.get("atom_details", [])),
            "pending_reason": str(result.get("pending_reason", "")),
        }

    def _result(
        self, *, status: str, reason_code: str, baseline: Mapping[str, Any],
        bindings: Mapping[str, Mapping[str, Any]], atom_ids: list[str],
        dimensions: list[dict[str, Any]], binding_issues: list[dict[str, Any]],
        unresolved_relevant: list[str], unresolved_irrelevant: list[str],
    ) -> dict[str, Any]:
        return {
            "readiness_version": "fusa-v1-exposure-input-readiness-v1",
            "status": status,
            "reason_code": reason_code,
            "baseline_exposure": self._preview(baseline),
            "scenario_atom_ids": atom_ids,
            "resolved_dimensions": sorted(
                dimension for dimension, binding in bindings.items()
                if self._is_resolved(binding)
            ),
            "unresolved_relevant_dimensions": sorted(set(unresolved_relevant)),
            "unresolved_irrelevant_dimensions": sorted(set(unresolved_irrelevant)),
            "binding_issues": binding_issues,
            "dimension_assessments": dimensions,
        }

    def assess(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        """Return a traceable, source-only readiness decision.

        The current atom set and every source-catalog candidate for an
        unresolved dimension are evaluated through ``preview``.  A dimension
        is blocking only when a permitted candidate produces a different
        native FUSA result, status, or Z/F selection.
        """
        baseline = self.executor.preview(dict(scenario), self.exposure)
        bindings = self._bindings(scenario)
        atom_ids = self._atom_ids(scenario)
        atom_id_set = set(atom_ids)
        dimensions: list[dict[str, Any]] = []
        binding_issues: list[dict[str, Any]] = []
        unresolved_relevant: list[str] = []
        unresolved_irrelevant: list[str] = []

        _, domain_error = self._domain(scenario)
        if domain_error:
            return self._result(
                status="PENDING_DOMAIN", reason_code=domain_error,
                baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                dimensions=dimensions, binding_issues=binding_issues,
                unresolved_relevant=unresolved_relevant,
                unresolved_irrelevant=unresolved_irrelevant,
            )

        if not atom_ids:
            binding_issues.append({
                "code": "SCENARIO_ATOM_SET_EMPTY",
                "dimensions": sorted(
                    dimension for dimension, binding in bindings.items()
                    if str(binding.get("applicability_status", "")).upper()
                    != "NOT_APPLICABLE"
                    and not self._is_resolved(binding)
                ),
            })
        for atom_id in atom_ids:
            if atom_id not in self.by_id:
                binding_issues.append({
                    "code": "SCENARIO_ATOM_NOT_IN_CATALOG", "atom_id": atom_id,
                    "dimensions": [],
                })

        for dimension, binding in bindings.items():
            if self._contains(binding, "CONFLICT"):
                return self._result(
                    status="SOURCE_CONFLICT", reason_code="EXPOSURE_SOURCE_CONFLICT",
                    baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                    dimensions=dimensions, binding_issues=[*binding_issues, {
                        "code": "BINDING_SOURCE_CONFLICT", "dimension": dimension,
                    }], unresolved_relevant=unresolved_relevant,
                    unresolved_irrelevant=unresolved_irrelevant,
                )
            if not self._is_resolved(binding):
                continue
            atom_id = self._binding_atom_id(binding)
            if not atom_id or atom_id not in self.by_id or atom_id not in atom_id_set:
                binding_issues.append({
                    "code": "RESOLVED_DIMENSION_ATOM_NOT_SUPPLIED",
                    "dimension": dimension, "atom_id": atom_id,
                })
                continue
            atom = self.by_id[atom_id]
            filled = binding.get("filled_dimensions", ())
            filled_dimensions = [str(item) for item in filled] if isinstance(filled, (list, tuple)) else []
            claimed = filled_dimensions or [dimension]
            if any(item not in atom.dimensions for item in claimed):
                binding_issues.append({
                    "code": "COMPOUND_FILL_NOT_SOURCE_DEFINED",
                    "dimension": dimension, "atom_id": atom_id,
                    "filled_dimensions": claimed,
                })

        if binding_issues:
            return self._result(
                status="PENDING_ATOM_BINDING",
                reason_code=(
                    "EXPOSURE_SCENARIO_ATOM_SET_EMPTY" if not atom_ids
                    else "EXPOSURE_ATOM_BINDING_INCOMPLETE"
                ),
                baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                dimensions=dimensions, binding_issues=binding_issues,
                unresolved_relevant=unresolved_relevant,
                unresolved_irrelevant=unresolved_irrelevant,
            )

        baseline_signature = self._signature(baseline)
        for dimension, binding in bindings.items():
            if self._is_resolved(binding):
                continue
            if str(binding.get("applicability_status", "")).upper() == "NOT_APPLICABLE":
                unresolved_irrelevant.append(dimension)
                dimensions.append({
                    "dimension": dimension,
                    "project_value": str(binding.get("project_value", "")),
                    "resolution_status": str(binding.get("resolution_status", "NOT_APPLICABLE")),
                    "binding_status": str(binding.get("binding_status", "NOT_APPLICABLE")),
                    "unresolved_reason": str(binding.get("applicability_reason", "")),
                    "candidate_atom_ids": [],
                    "candidate_atom_count": 0,
                    "could_change_exposure": False,
                    "change_witnesses": [],
                    "applicability_status": "NOT_APPLICABLE",
                })
                continue
            all_candidates = self._candidate_ids(dimension, binding)
            candidate_ids = [candidate_id for candidate_id in all_candidates
                             if self._compatible_completion([*atom_ids, candidate_id])]
            explicit_ambiguous = (
                self._contains(binding, "AMBIGUOUS")
                and len(self._explicit_candidate_ids(binding)) > 1
            )
            assessment = {
                "dimension": dimension,
                "project_value": str(binding.get("project_value", "")),
                "resolution_status": str(binding.get("resolution_status", "PENDING")),
                "binding_status": str(binding.get("binding_status", "")),
                "unresolved_reason": str(binding.get("unresolved_reason", "")),
                "candidate_atom_ids": candidate_ids,
                "candidate_atom_count": len(candidate_ids),
                "incompatible_compound_candidates": [value for value in all_candidates if value not in candidate_ids],
                "could_change_exposure": False,
                "change_witnesses": [],
            }
            for candidate_id in candidate_ids:
                if candidate_id in atom_id_set:
                    continue
                prospective = deepcopy(dict(scenario))
                prospective["scenario_atom_ids"] = [*atom_ids, candidate_id]
                result = self.executor.preview(prospective, self.exposure)
                if self._signature(result) != baseline_signature:
                    assessment["could_change_exposure"] = True
                    assessment["change_witnesses"].append({
                        "atom_id": candidate_id,
                        "result": self._preview(result),
                        "applicability_proof": ("NOT_ESTABLISHED_OPTIONAL_DIMENSION"
                            if binding.get("applicability_status") == "OPTIONAL" else "SOURCE_CANDIDATE_CONSTRAINTS"),
                    })
                    # One source-backed witness is enough to prove that a
                    # final E would be premature; keep audit artifacts compact.
                    break
            dimensions.append(assessment)
            if explicit_ambiguous and assessment["could_change_exposure"]:
                return self._result(
                    status="PENDING_AMBIGUOUS_ATOM_SET",
                    reason_code="EXPOSURE_ATOM_SET_AMBIGUOUS",
                    baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                    dimensions=dimensions, binding_issues=binding_issues,
                    unresolved_relevant=unresolved_relevant,
                    unresolved_irrelevant=unresolved_irrelevant,
                )
            if assessment["could_change_exposure"]:
                unresolved_relevant.append(dimension)
            else:
                unresolved_irrelevant.append(dimension)

        # Single-dimension previews do not prove joint invariance: two atoms
        # together can select a different native aggregation branch. Evaluate
        # the bounded product using the same executor, including omission.
        pools = [(item["dimension"], item["candidate_atom_ids"]) for item in dimensions
                 if item["candidate_atom_ids"] and not item["could_change_exposure"]]
        if not unresolved_relevant and len(pools) > 1:
            combinations = product(*[(None, *ids) for _, ids in pools])
            for index, combination in enumerate(islice(combinations, 257)):
                if index == 256:
                    return self._result(
                        status="PENDING_PROOF_BUDGET", reason_code="EXPOSURE_PROOF_BUDGET_EXCEEDED",
                        baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                        dimensions=[*dimensions, {"joint_preview": "PROOF_BUDGET_EXCEEDED", "preview_limit": 256}],
                        binding_issues=binding_issues, unresolved_relevant=[],
                        unresolved_irrelevant=unresolved_irrelevant,
                    )
                selected = list(dict.fromkeys([*atom_ids, *(value for value in combination if value)]))
                if not self._compatible_completion(selected):
                    continue
                result = self.executor.preview({**scenario, "scenario_atom_ids": selected}, self.exposure)
                if self._signature(result) != baseline_signature:
                    unresolved_relevant.extend(dimension for (dimension, _), value in zip(pools, combination) if value)
                    dimensions.append({"joint_preview": "EXPOSURE_CAN_CHANGE", "atom_ids": selected,
                                       "result": self._preview(result)})
                    break

        if unresolved_relevant:
            unresolved_irrelevant = [dimension for dimension in unresolved_irrelevant if dimension not in unresolved_relevant]
            applicability_unresolved = all(bindings.get(dimension, {}).get("applicability_status") == "OPTIONAL"
                                           for dimension in unresolved_relevant)
            return self._result(
                status="PENDING_DIMENSION_APPLICABILITY" if applicability_unresolved else "PENDING_RELEVANT_DIMENSION",
                reason_code="EXPOSURE_DIMENSION_APPLICABILITY_UNPROVEN" if applicability_unresolved else "EXPOSURE_RELEVANT_DIMENSION_UNRESOLVED",
                baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                dimensions=dimensions, binding_issues=binding_issues,
                unresolved_relevant=unresolved_relevant,
                unresolved_irrelevant=unresolved_irrelevant,
            )
        if _status(baseline) != _FINALIZED:
            return self._result(
                status="PENDING_ATOM_BINDING",
                reason_code=(str(baseline.get("pending_reason", ""))
                             or "EXPOSURE_ATOM_BINDING_INCOMPLETE"),
                baseline=baseline, bindings=bindings, atom_ids=atom_ids,
                dimensions=dimensions, binding_issues=binding_issues,
                unresolved_relevant=unresolved_relevant,
                unresolved_irrelevant=unresolved_irrelevant,
            )
        return self._result(
            status=("READY_METHOD_IRRELEVANT_GAPS" if unresolved_irrelevant else "READY_COMPLETE"),
            reason_code=("UNRESOLVED_DIMENSIONS_CANNOT_CHANGE_EXPOSURE"
                         if unresolved_irrelevant else "COMPLETE_RELEVANT_ATOM_SET"),
            baseline=baseline, bindings=bindings, atom_ids=atom_ids,
            dimensions=dimensions, binding_issues=binding_issues,
            unresolved_relevant=unresolved_relevant,
            unresolved_irrelevant=unresolved_irrelevant,
        )


__all__ = ["ExposureInputReadinessService", "READY_STATUSES"]
