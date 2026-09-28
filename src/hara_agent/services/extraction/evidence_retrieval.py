from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .targeted_verification import block_value, normalize_source_text


DEFAULT_CONTEXT_CHARACTER_BUDGET = 5000


@dataclass(frozen=True)
class FactRetrievalSpec:
    """Deterministic retrieval contract for one Project Fact concept.

    Specifications contain terminology only. Expected values and gold source
    locators are deliberately not representable in this model.
    """

    fact_type: str
    aliases: tuple[str, ...]
    unit_hints: tuple[str, ...] = ()
    context_hints: tuple[str, ...] = ()
    section_hints: tuple[str, ...] = ()
    required: bool = True
    structural_kind: str = ""
    exclusion_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fact_type.strip():
            raise ValueError("FactRetrievalSpec.fact_type must not be empty")
        if not self.aliases:
            raise ValueError("FactRetrievalSpec.aliases must not be empty")
        if self.structural_kind not in {
            "", "OPERATIONAL_SPEED_CANDIDATE",
            "CATEGORICAL_ALLOWED_SET_CANDIDATE",
        }:
            raise ValueError("FactRetrievalSpec.structural_kind is unsupported")


@dataclass(frozen=True)
class ScoredEvidenceBlock:
    block_index: int
    block_id: str
    location: str
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FactRoutingDiagnostic:
    fact_type: str
    required: bool
    candidate_count: int
    selected_block_ids: tuple[str, ...]
    coverage_status: str
    top_scores: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RoutingDiagnostics:
    task: str
    max_characters: int
    selected_characters: int
    selected_block_ids: tuple[str, ...]
    facts: tuple[FactRoutingDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextAssemblyResult:
    block_ids: tuple[str, ...]
    source_blocks: tuple[dict[str, Any], ...]
    text: str
    selected_characters: int
    diagnostics: RoutingDiagnostics


class DeterministicEvidenceRetriever:
    """Global lexical/structural candidate retrieval with transparent scores."""

    def rank(
        self,
        blocks: Sequence[Any],
        specs: Sequence[FactRetrievalSpec],
    ) -> dict[str, list[ScoredEvidenceBlock]]:
        rankings: dict[str, list[ScoredEvidenceBlock]] = {}
        for spec in specs:
            candidates = []
            for index, block in enumerate(blocks):
                score, reasons = self._score(block, spec)
                if score <= 0:
                    continue
                candidates.append(ScoredEvidenceBlock(
                    block_index=index,
                    block_id=block_value(block, "block_id"),
                    location=block_value(block, "location"),
                    score=score,
                    reasons=tuple(reasons),
                ))
            rankings[spec.fact_type] = sorted(
                candidates,
                key=lambda item: (-item.score, item.block_index, item.block_id),
            )
        return rankings

    @classmethod
    def _score(cls, block: Any, spec: FactRetrievalSpec) -> tuple[int, list[str]]:
        text = normalize_source_text(block_value(block, "text"))
        location = normalize_source_text(block_value(block, "location"))
        kind = normalize_source_text(block_value(block, "kind"))
        alias_hits = [
            alias for alias in spec.aliases
            if normalize_source_text(alias) in text
        ]
        if not alias_hits and any(
            normalize_source_text(alias) in text
            for alias in spec.exclusion_aliases
        ):
            return 0, []
        if alias_hits:
            score = 12 + min(8, 2 * len(alias_hits))
            reasons = [f"alias:{alias}" for alias in alias_hits]
        else:
            score, reasons = cls._structural_score(text, location, kind, spec)
            if score <= 0:
                return 0, []
        context_hits = [
            hint for hint in spec.context_hints
            if normalize_source_text(hint) in text
        ]
        if context_hits:
            score += min(12, 3 * len(context_hits))
            reasons.extend(f"context:{hint}" for hint in context_hits)
        unit_hits = [
            hint for hint in spec.unit_hints
            if normalize_source_text(hint) in text
        ]
        if unit_hits:
            score += min(8, 4 * len(unit_hits))
            reasons.extend(f"unit:{hint}" for hint in unit_hits)
        section_hits = [
            hint for hint in spec.section_hints
            if normalize_source_text(hint) in text or normalize_source_text(hint) in location
        ]
        if section_hits:
            score += min(4, len(section_hits))
            reasons.extend(f"section:{hint}" for hint in section_hits)
        if kind == "table_row" or "table[" in location:
            score += 2
            reasons.append("structure:table_row")
        return score, reasons

    @staticmethod
    def _structural_score(
        text: str,
        location: str,
        kind: str,
        spec: FactRetrievalSpec,
    ) -> tuple[int, list[str]]:
        """Route shape-compatible candidates without deciding their meaning.

        Structural retrieval expands recall only. The semantic extraction and
        fail-closed normalizer remain responsible for accepting a project fact.
        """

        is_table_row = kind == "table_row" or "table[" in location
        if spec.structural_kind == "OPERATIONAL_SPEED_CANDIDATE":
            unit_hits = [
                hint for hint in spec.unit_hints
                if normalize_source_text(hint) in text
            ]
            numeric_constraint = re.search(
                r"(?:[<>]=?|[\u2264\u2265]|\d+(?:\.\d+)?\s*[-~～至]\s*)"
                r"\d+(?:\.\d+)?",
                text,
            )
            if not unit_hits or numeric_constraint is None:
                return 0, []
            reasons = ["structure:numeric_speed_constraint"]
            reasons.extend(f"unit:{hint}" for hint in unit_hits)
            return (8 if is_table_row else 6), reasons
        if spec.structural_kind == "CATEGORICAL_ALLOWED_SET_CANDIDATE":
            # Short table rows with a multi-value cell are candidates for an
            # allowed set. No domain wording or expected value is encoded here.
            member_separators = ("/", "／")
            if (
                is_table_row
                and any(item in text for item in member_separators)
                and text.count("|") == 1
                and len(text) <= 180
            ):
                return 8, ["structure:categorical_allowed_set"]
            conditional = any(item in text for item in (" if ", "when", "当", "且"))
            numeric_condition = bool(re.search(
                r"(?:[<>]=?|[\u2264\u2265])\s*\d+(?:\.\d+)?", text,
            ))
            if not is_table_row and conditional and numeric_condition:
                return 7, ["structure:conditional_context"]
            return 0, []
        return 0, []


class CoverageFirstContextAssembler:
    """Select one candidate per required fact before adding support or repeats."""

    _TABLE_LOCATION = re.compile(r"table\[(\d+)\]\.row\[(\d+)\]")

    def assemble(
        self,
        blocks: Sequence[Any],
        specs: Sequence[FactRetrievalSpec],
        rankings: dict[str, list[ScoredEvidenceBlock]],
        *,
        task: str,
        max_characters: int = DEFAULT_CONTEXT_CHARACTER_BUDGET,
    ) -> ContextAssemblyResult:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")
        selected: set[int] = set()
        primary_by_fact: dict[str, int] = {}
        total = 0

        def line_for(index: int) -> str:
            block = blocks[index]
            return (
                f"[{block_value(block, 'block_id')}] "
                f"{block_value(block, 'location')}: {block_value(block, 'text')}"
            )

        def add(index: int) -> bool:
            nonlocal total
            if index in selected:
                return True
            line = line_for(index)
            cost = len(line) + (1 if selected else 0)
            if total + cost > max_characters:
                return False
            selected.add(index)
            total += cost
            return True

        # Round 1: guarantee coverage opportunity for every required concept.
        for spec in specs:
            if not spec.required:
                continue
            candidates = rankings.get(spec.fact_type, [])
            if candidates and add(candidates[0].block_index):
                primary_by_fact[spec.fact_type] = candidates[0].block_index

        # Round 2: bounded table/header and adjacent support for covered facts.
        support_indices = []
        for index in primary_by_fact.values():
            support_indices.extend(self._support_indices(blocks, index))
        for index in dict.fromkeys(support_indices):
            add(index)

        # Round 3: use remaining budget by global score; duplicate hits are free.
        best_by_index: dict[int, ScoredEvidenceBlock] = {}
        for candidates in rankings.values():
            for candidate in candidates:
                current = best_by_index.get(candidate.block_index)
                if current is None or candidate.score > current.score:
                    best_by_index[candidate.block_index] = candidate
        for candidate in sorted(
            best_by_index.values(),
            key=lambda item: (-item.score, item.block_index, item.block_id),
        ):
            add(candidate.block_index)

        ordered_indices = sorted(selected)
        lines = [line_for(index) for index in ordered_indices]
        text = "\n".join(lines)
        source_blocks = tuple({
            "block_id": block_value(blocks[index], "block_id"),
            "kind": block_value(blocks[index], "kind"),
            "location": block_value(blocks[index], "location"),
            "text": block_value(blocks[index], "text"),
            "section_path": (
                list(blocks[index].get("section_path", []))
                if isinstance(blocks[index], dict)
                else list(getattr(blocks[index], "section_path", ()))
            ),
        } for index in ordered_indices)

        fact_diagnostics = []
        for spec in specs:
            candidates = rankings.get(spec.fact_type, [])
            selected_candidates = tuple(
                item.block_id for item in candidates if item.block_index in selected
            )
            if not candidates:
                status = "NO_CANDIDATE" if spec.required else "OPTIONAL_NO_CANDIDATE"
            elif candidates[0].block_index in selected:
                status = "COVERED"
            else:
                status = "BUDGET_EXCLUDED"
            fact_diagnostics.append(FactRoutingDiagnostic(
                fact_type=spec.fact_type,
                required=spec.required,
                candidate_count=len(candidates),
                selected_block_ids=selected_candidates,
                coverage_status=status,
                top_scores=tuple({
                    "block_id": item.block_id,
                    "location": item.location,
                    "score": item.score,
                    "reasons": list(item.reasons),
                } for item in candidates[:5]),
            ))
        diagnostics = RoutingDiagnostics(
            task=task,
            max_characters=max_characters,
            selected_characters=len(text),
            selected_block_ids=tuple(block_value(blocks[index], "block_id") for index in ordered_indices),
            facts=tuple(fact_diagnostics),
        )
        return ContextAssemblyResult(
            block_ids=diagnostics.selected_block_ids,
            source_blocks=source_blocks,
            text=text,
            selected_characters=len(text),
            diagnostics=diagnostics,
        )

    def _support_indices(self, blocks: Sequence[Any], index: int) -> list[int]:
        result: list[int] = []
        location = block_value(blocks[index], "location")
        table = self._TABLE_LOCATION.fullmatch(location)
        if table:
            table_id = table.group(1)
            header_location = f"table[{table_id}].row[1]"
            header = next(
                (
                    candidate_index for candidate_index, block in enumerate(blocks)
                    if block_value(block, "location") == header_location
                ),
                None,
            )
            if header is not None:
                result.append(header)
        if index > 0:
            result.append(index - 1)
        if index + 1 < len(blocks):
            result.append(index + 1)
        return result
