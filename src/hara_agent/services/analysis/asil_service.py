from __future__ import annotations

from hara_agent.contracts import MethodContract
from hara_agent.models import SourceRef


class MethodContractASILService:
    """Query the exact ASIL matrix compiled from the active template."""

    def __init__(self, method: MethodContract):
        if not method.engineering_rules_compiled:
            raise ValueError("MethodContract engineering rules are not compiled")
        self.method = method
        self.matrix = {
            (item.severity, item.exposure, item.controllability): item
            for item in method.asil.mappings
        }
        binding = method.asil.source_binding
        self.source = (
            f"method-contract:{method.metadata['template_hash']}#"
            f"{binding.sheet}!{binding.region}"
        )

    def _mapping(self, severity: str, exposure: str, controllability: str):
        key = tuple(
            str(value or "").strip().upper()
            for value in (severity, exposure, controllability)
        )
        if key not in self.matrix:
            raise ValueError(
                "Invalid S/E/C combination: "
                f"severity={severity!r}, exposure={exposure!r}, "
                f"controllability={controllability!r}"
            )
        return self.matrix[key]

    def determine(self, severity: str, exposure: str, controllability: str) -> str:
        return self._mapping(severity, exposure, controllability).result

    def partial_zero_decision(self, approved: dict[str, str]) -> tuple[str, list[SourceRef]]:
        """Prove the governed zero shortcut against every remaining matrix cell.

        Callers supply only finalized dimensions. Missing dimensions remain
        missing; no synthetic E0/C0/S0 values are created.
        """
        structured = self.method.structured_risk_method
        if structured is None or not any(
            approved.get(field) == zero
            for field, zero in (("severity", "S0"), ("exposure", "E0"), ("controllability", "C0"))
        ):
            return "", []
        levels = {
            "severity": self.method.asil.severity_levels,
            "exposure": self.method.asil.exposure_levels,
            "controllability": self.method.asil.controllability_levels,
        }
        if any(value not in levels[field] for field, value in approved.items()):
            return "", []
        matches = [row for row in self.matrix.values() if all(
            getattr(row, field) == value for field, value in approved.items()
        )]
        expected = 1
        for field, values in levels.items():
            if field not in approved:
                expected *= len(values)
        if len(matches) != expected or {row.result for row in matches} != {structured.asil_zero_short_circuit}:
            return "", []
        sources = list(dict.fromkeys(self.evidence_source(
            row.severity, row.exposure, row.controllability,
        ) for row in matches))
        return structured.asil_zero_short_circuit, sources

    def evidence_source(
        self, severity: str, exposure: str, controllability: str
    ) -> SourceRef:
        source = self._mapping(severity, exposure, controllability).source_ref
        return SourceRef(
            "method_contract",
            source.workbook,
            f"{source.sheet}!{source.range}",
            f"template_hash={source.template_hash}; source_hash={source.source_hash}",
        )
