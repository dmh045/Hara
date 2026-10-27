from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from hara_agent.contracts import (
    NormativeStrength, ScenarioConstraintDisposition, ScenarioConstraintRule,
)


class ScenarioConstraintStatus(str, Enum):
    KEEP = "KEEP"
    KEEP_RARE = "KEEP_RARE"
    DROP = "DROP"
    CONFLICT = "CONFLICT"
    UNRESOLVED_NO_RULES = "UNRESOLVED_NO_RULES"
    PENDING_COMPATIBILITY = "PENDING_COMPATIBILITY"


@dataclass(frozen=True)
class ScenarioConstraintEvaluation:
    status: ScenarioConstraintStatus
    matched_rule_ids: tuple[str, ...]
    reason: str


class ScenarioConstraintExecutor:
    """Apply exact compiled cross-dimension constraints without rarity guessing."""

    @staticmethod
    def compound_atom_conflicts(selected: dict[str, set[str]], fills: dict[str, Sequence[str]]) -> list[str]:
        """A compiled compound Atom occupies all of its selected dimensions."""
        reasons = []
        for atom_id in sorted(set().union(*selected.values())):
            dimensions = set(fills.get(atom_id, ())) & selected.keys()
            missing = sorted(d for d in dimensions if atom_id not in selected[d])
            conflicts = sorted(d for d in dimensions if selected[d] - {atom_id})
            if missing:
                reasons.append(f"COMPOUND_ATOM_INCOMPLETE:{atom_id}:{','.join(missing)}")
            if conflicts:
                reasons.append(f"COMPOUND_ATOM_CONFLICT:{atom_id}:{','.join(conflicts)}")
        return reasons

    @staticmethod
    def physical_semantic_conflicts(atoms: Sequence[dict[str, Any]]) -> list[str]:
        """Selected atoms describing one road must share a possible slope.

        Bounds and endpoint semantics come from the compiled catalog. This is
        interval compatibility, not an Exposure rule or a new slope threshold.
        """
        lower, upper, lower_closed, upper_closed = float("-inf"), float("inf"), True, True
        participants = []
        for atom in atoms:
            slope = atom.get("physical_semantics", {}).get("slope", {})
            if not slope:
                continue
            bounds = slope.get("range_pct", [slope.get("pct_min"), slope.get("pct_max")])
            lo, hi = bounds
            if lo is None and hi is None:
                continue
            participants.append(str(atom["atom_id"]))
            if lo is not None and lo >= lower:
                closed = slope.get("min_inclusive", True)
                lower_closed = lower_closed and closed if lo == lower else closed
                lower = lo
            if hi is not None and hi <= upper:
                closed = slope.get("max_inclusive", True)
                upper_closed = upper_closed and closed if hi == upper else closed
                upper = hi
        if lower > upper or (lower == upper and not (lower_closed and upper_closed)):
            return ["INCOMPATIBLE_SLOPE_RANGES:" + ",".join(sorted(set(participants)))]
        return []

    @staticmethod
    def _value(binding: Any) -> str:
        if isinstance(binding, dict):
            return str(
                binding.get("method_value") or binding.get("project_value") or ""
            ).strip()
        return str(binding or "").strip()

    def evaluate(
        self,
        bindings: dict[str, Any],
        rules: Sequence[ScenarioConstraintRule],
    ) -> ScenarioConstraintEvaluation:
        executable = [
            rule for rule in rules
            if rule.executable and rule.normative_strength is NormativeStrength.NORMATIVE
        ]
        if not executable:
            return ScenarioConstraintEvaluation(
                ScenarioConstraintStatus.UNRESOLVED_NO_RULES, (),
                "No normative executable cross-dimension constraint is compiled.",
            )
        matched = [
            rule for rule in executable
            if all(
                self._value(bindings.get(predicate.dimension)) in predicate.values
                for predicate in rule.predicates
            )
        ]
        if not matched:
            return ScenarioConstraintEvaluation(
                ScenarioConstraintStatus.KEEP, (),
                "No compiled exclusion or rarity rule matches this combination.",
            )
        dispositions = {rule.disposition for rule in matched}
        impossible = dispositions & {
            ScenarioConstraintDisposition.PHYSICALLY_IMPOSSIBLE,
            ScenarioConstraintDisposition.SEMANTICALLY_INCOMPATIBLE,
        }
        if impossible and (
            ScenarioConstraintDisposition.EXPLICITLY_ALLOWED in dispositions
            or ScenarioConstraintDisposition.RARE_BUT_FEASIBLE in dispositions
        ):
            return ScenarioConstraintEvaluation(
                ScenarioConstraintStatus.CONFLICT,
                tuple(rule.rule_id for rule in matched),
                "Matched normative scenario constraints have conflicting dispositions.",
            )
        if impossible:
            return ScenarioConstraintEvaluation(
                ScenarioConstraintStatus.DROP,
                tuple(rule.rule_id for rule in matched),
                "; ".join(rule.reason for rule in matched if rule.disposition in {
                    ScenarioConstraintDisposition.PHYSICALLY_IMPOSSIBLE,
                    ScenarioConstraintDisposition.SEMANTICALLY_INCOMPATIBLE,
                }),
            )
        if ScenarioConstraintDisposition.RARE_BUT_FEASIBLE in dispositions:
            return ScenarioConstraintEvaluation(
                ScenarioConstraintStatus.KEEP_RARE,
                tuple(rule.rule_id for rule in matched),
                "Rare but feasible combinations remain in HARA and are handled by E.",
            )
        return ScenarioConstraintEvaluation(
            ScenarioConstraintStatus.KEEP,
            tuple(rule.rule_id for rule in matched),
            "The combination is explicitly allowed by the compiled method.",
        )

    @staticmethod
    def pending(
        unresolved_dimensions: Sequence[str], bindings: dict[str, Any],
    ) -> ScenarioConstraintEvaluation:
        reasons = {
            str(name): str(
                bindings.get(name, {}).get("unresolved_reason", "PENDING_BINDING")
            )
            for name in unresolved_dimensions
            if isinstance(bindings.get(name), dict)
        }
        return ScenarioConstraintEvaluation(
            ScenarioConstraintStatus.PENDING_COMPATIBILITY,
            (),
            "Compatibility remains pending until every required dimension is bound: "
            + ", ".join(
                f"{name}={reason}" for name, reason in sorted(reasons.items())
            ),
        )
