from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hara_agent.contracts import (
    CategoricalPredicate, FactType, MethodContract, RangePredicate,
)
from hara_agent.models import (
    FactProvenance, ItemDefinitionFacts, MethodRiskFactBinding, ReviewStatus,
    RiskFact, SourceRef,
)


METHOD_FACT_KEYS = {item: item.value.casefold() for item in FactType}
METHOD_FACT_KEYS.update({
    FactType.SPEED_UNSPECIFIED: "speed_unspecified_kph",
    # EXPOSURE stores the method selector, not an E-level result.
    FactType.EXPOSURE: "exposure_method",
})

_BOOLEAN_FACT_TYPES = frozenset({
    FactType.DRIVER_IN_VEHICLE,
    FactType.DIRECT_CONTROL_AVAILABLE,
    FactType.INTERVENTION_AVAILABLE,
    FactType.REMOTE_INTERVENTION_AVAILABLE,
})


@dataclass(frozen=True)
class RiskFactBindingResult:
    values: dict[str, str | float | bool]
    provenance: dict[str, dict[str, Any]]
    sources: tuple[SourceRef, ...]
    audit: dict[str, Any]


class MethodRiskFactBindingCompiler:
    """Compile only exact ontology-name bindings; ambiguous names stay unbound."""

    def __init__(self, method: MethodContract):
        self.method = method
        self.template_hash = str(method.metadata["template_hash"])
        self.names: dict[str, FactType] = {}
        for fact_type, key in METHOD_FACT_KEYS.items():
            self.names[fact_type.value.casefold()] = fact_type
            self.names[key.casefold()] = fact_type

    def _sources(self, fact_type: FactType) -> list[SourceRef]:
        contract_sources = [
            source
            for spec in self.method.required_fact_specs
            if spec.fact_type is fact_type
            for source in spec.source_refs
        ]
        if fact_type is FactType.EXPOSURE:
            contract_sources = [
                source
                for rule in (
                    *self.method.exposure.duration_rules,
                    *self.method.exposure.frequency_rules,
                )
                for source in rule.source_refs
            ]
        return list(dict.fromkeys(
            SourceRef(
                "method_contract",
                self.template_hash,
                f"{source.sheet}!{source.range}",
                source.raw_text,
            )
            for source in contract_sources
        ))

    def compile(self, facts: list[RiskFact]) -> list[MethodRiskFactBinding]:
        bindings: list[MethodRiskFactBinding] = []
        for fact in facts:
            fact_type = self.names.get(fact.parameter.strip().casefold())
            if fact_type is None:
                continue
            sources = self._sources(fact_type)
            if not sources:
                continue
            bindings.append(MethodRiskFactBinding(
                source_fact_id=fact.fact_id,
                target_fact_type=fact_type.value,
                method_contract_hash=self.template_hash,
                source_refs=sources,
                provenance=FactProvenance.METHOD_CONTRACT,
                approval=(
                    ReviewStatus.FINALIZED
                    if fact.approval is ReviewStatus.FINALIZED
                    else ReviewStatus.PENDING
                ),
                binding_method="automatic_exact_ontology_name",
            ))
        return bindings


class MethodRiskFactBindingService:
    """Bind hash-matched canonical facts without semantic guessing.

    Context-specific facts take precedence over global facts. Equal-specificity
    conflicts are rejected and remain missing for scoring.
    """

    def __init__(self, method: MethodContract):
        self.method = method
        self.template_hash = str(method.metadata["template_hash"])
        self.compiler = MethodRiskFactBindingCompiler(method)
        rule_fact_types = {
            predicate.field
            for rule in (
                *method.severity.rules,
                *method.exposure.duration_rules,
                *method.exposure.frequency_rules,
                *method.controllability.criteria,
            )
            for predicate in rule.predicates
        }
        structured = method.structured_risk_method
        if structured is not None:
            for rule in structured.controllability_overrides:
                for condition in (*rule.all_of, *rule.any_of):
                    try:
                        rule_fact_types.add(FactType(condition.field.upper()))
                    except ValueError:
                        # Some structured inputs have no neutral FactType and
                        # remain directly scoped Scenario facts.
                        continue
        self.required = {
            item.fact_type: item for item in method.required_fact_specs
            if item.fact_type in rule_fact_types
        }
        # Exposure T/F selects which compiled rule family is executable.
        self.required.setdefault(FactType.EXPOSURE, None)
        self.allowed_values = self._allowed_values(method)
        self.numeric_fact_types = {
            item.fact_type for item in method.required_fact_specs if item.unit
        }
        self.numeric_fact_types.update(
            predicate.field
            for rule in (
                *method.severity.rules,
                *method.exposure.duration_rules,
                *method.exposure.frequency_rules,
                *method.controllability.criteria,
            )
            for predicate in rule.predicates
            if isinstance(predicate, RangePredicate)
        )

    @staticmethod
    def _allowed_values(method: MethodContract) -> dict[FactType, set[str]]:
        values: dict[FactType, set[str]] = {FactType.EXPOSURE: {"T", "F"}}
        rules = [
            *method.severity.rules,
            *method.exposure.duration_rules,
            *method.exposure.frequency_rules,
            *method.controllability.criteria,
        ]
        for rule in rules:
            for predicate in rule.predicates:
                if isinstance(predicate, CategoricalPredicate):
                    values.setdefault(predicate.field, set()).update(predicate.values)
        return values

    @staticmethod
    def _source_dict(source: SourceRef) -> dict[str, str]:
        return {
            "source_type": source.source_type,
            "source_id": source.source_id,
            "location": source.location,
            "excerpt": source.excerpt,
        }

    @staticmethod
    def _context_matches(context: dict[str, str], candidate: dict[str, Any]) -> bool:
        return all(
            str(candidate.get(key, "")).strip().casefold()
            == str(expected).strip().casefold()
            for key, expected in context.items()
        )

    def _normalize_value(
        self, fact_type: FactType, fact: RiskFact
    ) -> str | float | bool | None:
        if (
            fact_type is FactType.DRIVER_IN_VEHICLE
            and str(fact.context.get("allowed_driver_position", "")).casefold()
            == "outside_driver_seat"
            and str(fact.value).strip().casefold() in {"false", "0"}
            and not any(
                marker in " ".join(
                    source.excerpt.casefold() for source in fact.source_refs
                )
                for marker in (
                    "车外", "outside the vehicle", "outside vehicle",
                    "not in the vehicle",
                )
            )
        ):
            # The source says the driver is not in the driver seat. They may
            # still be elsewhere inside the vehicle, so this cannot prove the
            # Method predicate "driver_in_vehicle = false".
            return None
        if fact_type in _BOOLEAN_FACT_TYPES:
            if isinstance(fact.value, bool):
                return fact.value
            value = str(fact.value).strip().casefold()
            if value in {"true", "1"}:
                return True
            if value in {"false", "0"}:
                return False
            return None
        if fact_type in self.numeric_fact_types:
            if not isinstance(fact.value, (int, float)) or isinstance(fact.value, bool):
                return None
            required = self.required.get(fact_type)
            expected_unit = required.unit if required is not None else ""
            if expected_unit and fact.unit != expected_unit:
                return None
            return float(fact.value)
        value = str(fact.value).strip().upper()
        allowed = self.allowed_values.get(fact_type, set())
        return value if value and (not allowed or value in allowed) else None

    def bind(
        self,
        project_facts: ItemDefinitionFacts,
        candidate: dict[str, Any],
    ) -> RiskFactBindingResult:
        by_type: dict[FactType, list[tuple[RiskFact, MethodRiskFactBinding]]] = {}
        hash_mismatch = 0
        unknown_type = 0
        facts_by_id = {item.fact_id: item for item in project_facts.risk_facts}
        explicit_active_sources = {
            item.source_fact_id for item in project_facts.method_risk_fact_bindings
            if item.method_contract_hash == self.template_hash
        }
        automatic = [
            item for item in self.compiler.compile(project_facts.risk_facts)
            if item.source_fact_id not in explicit_active_sources
        ]
        bindings = [*project_facts.method_risk_fact_bindings, *automatic]
        driver_source_conflict = candidate.get("driver_configuration_source_conflict") is True
        for binding in bindings:
            if binding.method_contract_hash != self.template_hash:
                hash_mismatch += 1
                continue
            try:
                fact_type = FactType(binding.target_fact_type)
            except ValueError:
                unknown_type += 1
                continue
            if driver_source_conflict and fact_type is FactType.DRIVER_IN_VEHICLE:
                continue
            fact = facts_by_id[binding.source_fact_id]
            if fact_type not in self.required or not self._context_matches(
                fact.context, candidate
            ):
                continue
            by_type.setdefault(fact_type, []).append((fact, binding))

        values: dict[str, str | float | bool] = {}
        provenance: dict[str, dict[str, Any]] = {}
        sources: list[SourceRef] = []
        conflicts: list[str] = (
            [FactType.DRIVER_IN_VEHICLE.value] if driver_source_conflict else []
        )
        invalid: list[str] = []
        for fact_type, matches in by_type.items():
            specificity = max(len(item[0].context) for item in matches)
            selected = [item for item in matches if len(item[0].context) == specificity]
            normalized = [
                self._normalize_value(fact_type, fact) for fact, _ in selected
            ]
            if any(item is None for item in normalized):
                invalid.append(fact_type.value)
                continue
            distinct = {str(item) for item in normalized}
            if len(distinct) != 1:
                conflicts.append(fact_type.value)
                continue
            key = METHOD_FACT_KEYS[fact_type]
            normalized_value = normalized[0]
            if normalized_value is None:  # guarded above; keeps the type explicit
                continue
            values[key] = normalized_value
            selected_sources = list(dict.fromkeys(
                source
                for fact, binding in selected
                for source in [*fact.source_refs, *binding.source_refs]
            ))
            sources.extend(selected_sources)
            approvals = {
                status.value
                for fact, binding in selected
                for status in (fact.approval, binding.approval)
            }
            binding_authorities = {
                binding.provenance.value for _, binding in selected
            }
            provenance[key] = {
                "approval": (
                    "FINALIZED" if approvals == {"FINALIZED"} else "PENDING"
                ),
                "provenance": (
                    "HUMAN_CONFIRMATION"
                    if binding_authorities == {"HUMAN_CONFIRMATION"}
                    else "DERIVED"
                ),
                "source_refs": [self._source_dict(item) for item in selected_sources],
                "source_binding_kind": "METHOD_RISK_FACT_BINDING",
                "method_contract_hash": self.template_hash,
            }

        missing = sorted(
            fact_type.value for fact_type in self.required
            if METHOD_FACT_KEYS[fact_type] not in values
        )
        unique_sources = tuple(dict.fromkeys(sources))
        return RiskFactBindingResult(
            values=values,
            provenance=provenance,
            sources=unique_sources,
            audit={
                "bound_fact_types": sorted(
                    fact_type.value for fact_type in self.required
                    if METHOD_FACT_KEYS[fact_type] in values
                ),
                "missing_fact_types": missing,
                "conflicting_fact_types": sorted(conflicts),
                "invalid_fact_types": sorted(invalid),
                "template_hash_mismatch_count": hash_mismatch,
                "unknown_fact_type_count": unknown_type,
                "automatic_binding_count": len(automatic),
                "explicit_binding_count": len(bindings) - len(automatic),
            },
        )
