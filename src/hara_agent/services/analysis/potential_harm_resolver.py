from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hara_agent.contracts import CalculationStatus, MethodContract
from hara_agent.models import EvidenceRecord, ReviewStatus, SourceRef


@dataclass(frozen=True)
class PotentialHarmResolution:
    potential_harm: str
    rule_id: str
    inputs_used: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]
    status: CalculationStatus
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "potential_harm": self.potential_harm,
            "rule_id": self.rule_id,
            "inputs_used": list(self.inputs_used),
            "evidence_refs": list(self.evidence_refs),
            "source_refs": [
                {
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "location": item.location,
                    "excerpt": item.excerpt,
                }
                for item in self.source_refs
            ],
            "status": self.status.value,
            "reason": self.reason,
        }


class PotentialHarmResolver:
    """Resolve harm semantics from the active MethodContract after Severity."""

    @staticmethod
    def _evidence_source(source: Any) -> SourceRef:
        if hasattr(source, "source_type"):
            return source
        return SourceRef(
            "method_contract",
            str(getattr(source, "template_hash", "")),
            f"{getattr(source, 'sheet', '')}!{getattr(source, 'range', '')}",
            str(getattr(source, "raw_text", "")),
        )

    @staticmethod
    def _severity_semantics(method: MethodContract) -> dict[str, tuple[str, SourceRef]]:
        return {
            level.level: (
                level.description.strip(),
                PotentialHarmResolver._evidence_source(level.source_ref),
            )
            for level in method.severity.scale.levels
            if level.description.strip()
        }

    @staticmethod
    def _method_rule_record(
        registry: Any, rule_id: str,
    ) -> EvidenceRecord | None:
        return next(
            (
                item for item in registry.records
                if item.evidence_ref.startswith("METHOD.SEVERITY.")
                and str(item.metadata.get("rule_id", "")) == rule_id
            ),
            None,
        )

    def resolve(
        self,
        *,
        method: MethodContract,
        severity_result: dict[str, Any],
        scenario: dict[str, Any],
        registry: Any,
    ) -> PotentialHarmResolution:
        severity = str(severity_result.get("severity_score", "")).strip()
        semantics = self._severity_semantics(method)
        if severity not in semantics:
            return PotentialHarmResolution(
                "", "", (), (), (), CalculationStatus.PENDING_METHOD_SEMANTICS,
                "MethodContract does not provide canonical harm semantics for the resolved Severity.",
            )
        rule_id = str(severity_result.get("engineering_rule_id", "")).split(",", 1)[0].strip()
        method_record = self._method_rule_record(registry, rule_id) if rule_id else None
        description, scale_source = semantics[severity]
        refs: list[str] = []
        sources: list[SourceRef] = []
        if method_record is not None:
            refs.append(method_record.evidence_ref)
            sources.extend(method_record.source_refs)
        else:
            # The scale source is still a real MethodContract source.  No
            # synthetic APPROVED_RULE ref is created when the rule provider
            # has no matching severity band.
            sources.append(scale_source)
        inputs_used = ["severity_score", "collision_type", "road_user_type"]
        for key in ("delta_v_kph", "impact_speed_kph", "relative_speed_kph"):
            if scenario.get(key) not in (None, ""):
                inputs_used.append(key)
                for ref in (f"DERIVED.{key}", f"SCN.{key}"):
                    record = registry.resolve_record(ref)
                    if record is not None:
                        refs.append(ref)
                        sources.extend(record.source_refs)
                        break
        for key in ("collision_type", "road_user_type"):
            if scenario.get(key) not in (None, ""):
                record = registry.resolve_record(f"SCN.{key}")
                if record is not None:
                    refs.append(record.evidence_ref)
                    sources.extend(record.source_refs)
        final = (
            severity_result.get("engineering_status") == ReviewStatus.FINALIZED.value
            and bool(description)
            and bool(refs)
        )
        return PotentialHarmResolution(
            # A calculated, conditional S still has the method's injury-class
            # meaning.  Availability of that meaning is not engineering approval.
            potential_harm=description,
            rule_id=rule_id,
            inputs_used=tuple(dict.fromkeys(inputs_used)),
            evidence_refs=tuple(dict.fromkeys(refs)),
            source_refs=tuple(dict.fromkeys(sources)),
            status=CalculationStatus.FINALIZED if final else CalculationStatus.PENDING_INPUT,
            reason=(
                "Resolved canonical potential-harm semantics from the active Severity scale."
                if final else
                "Conditional Severity class description from the active method; "
                "not a confirmed injury outcome or engineering approval."
            ),
        )
