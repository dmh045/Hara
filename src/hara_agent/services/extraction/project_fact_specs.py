from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from hara_agent.contracts import FactOrigin, FactType, RequiredFactSpec
from hara_agent.models import ProjectFactOutputType
from hara_agent.models.project_facts import PHYSICAL_QUANTITY_UNITS

from .evidence_retrieval import FactRetrievalSpec


@dataclass(frozen=True)
class RequiredProjectFactSpec:
    """Extraction schema compiled from Item facts and the Method Contract."""

    fact_type: str
    output_type: ProjectFactOutputType
    aliases: tuple[str, ...]
    unit_hints: tuple[str, ...]
    context_hints: tuple[str, ...]
    required_fields: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    structural_kind: str = ""
    exclusion_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fact_type or not self.aliases or not self.required_fields:
            raise ValueError("RequiredProjectFactSpec requires identity, aliases and fields")
        forbidden = {"expected_value", "gold", "source_block_ids", "source_location"}
        if forbidden & set(self.required_fields):
            raise ValueError("RequiredProjectFactSpec cannot encode answers or locators")

    def retrieval_spec(self) -> FactRetrievalSpec:
        return FactRetrievalSpec(
            fact_type=self.fact_type,
            aliases=self.aliases,
            unit_hints=self.unit_hints,
            context_hints=self.context_hints,
            structural_kind=self.structural_kind,
            exclusion_aliases=self.exclusion_aliases,
        )


def build_project_fact_spec_batches(
    operating_modes: Sequence[str],
    method_specs: Sequence[RequiredFactSpec],
) -> dict[str, tuple[RequiredProjectFactSpec, ...]]:
    """Build bounded extraction work without product/domain constants in Python."""

    modes = _unique_modes(operating_modes)
    speed_specs = (
        *tuple(_speed_spec(mode) for mode in modes),
        _contextual_speed_spec(modes),
    )
    non_source_extraction_facts = {
        # These are governance/engineering outputs, not Item Definition facts.
        FactType.EXPOSURE,
        FactType.AVOIDABILITY_PERCENT,
    }
    risk_specs = tuple(
        _method_risk_spec(spec)
        for spec in _unique_method_specs(method_specs)
        if spec.origin in {FactOrigin.PROJECT_FACT, FactOrigin.HUMAN_EVIDENCE}
        and spec.fact_type not in non_source_extraction_facts
        and spec.fact_type not in {FactType.FUNCTION, FactType.OUTPUT}
    )
    batches: dict[str, tuple[RequiredProjectFactSpec, ...]] = {}
    if speed_specs:
        batches["mode_speed_envelopes"] = speed_specs
    if risk_specs:
        batches["method_risk_facts"] = risk_specs
    # Names route source text only; no numeric or fault-stage defaults.
    aliases = {
        "longitudinal_acceleration_mps2": ("longitudinal acceleration", "纵向加速度"),
        "travel_to_impact_m": ("travel to impact", "碰撞前行驶距离"),
        "relative_distance_m": ("relative distance", "相对距离"),
        "model_valid_until_s": ("model valid until", "模型有效时间"),
    }
    batches["physical_motion_inputs"] = tuple(RequiredProjectFactSpec(
        key, ProjectFactOutputType.RISK_FACT, (key, *aliases[key]), units, (),
        ("status", "value", "unit", "context", "source_block_id", "source_excerpt"),
        constraints=("Only an explicit scalar source quantity; no conversion of command limits or normal envelopes into a fault bound.",),
    ) for key, units in PHYSICAL_QUANTITY_UNITS.items())
    return batches


def _speed_spec(mode: str) -> RequiredProjectFactSpec:
    normalized = _mode_identity(mode)
    slug = re.sub(r"[^a-z0-9]+", ".", normalized.casefold()).strip(".") or "mode"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return RequiredProjectFactSpec(
        fact_type=f"speed.{slug[:40]}.{digest}",
        output_type=ProjectFactOutputType.SPEED_ENVELOPE,
        aliases=(normalized,),
        unit_hints=("km/h", "kph", "kmh"),
        context_hints=(normalized,),
        required_fields=(
            "status", "operator", "value", "unit",
            "operating_mode", "source_block_id", "source_excerpt", "semantic_scope",
        ),
        structural_kind="OPERATIONAL_SPEED_CANDIDATE",
    )


def _contextual_speed_spec(modes: Sequence[str]) -> RequiredProjectFactSpec:
    """Recall source speed constraints whose context is not a mode label.

    Generic structural retrieval deliberately contains no product-language
    synonym table and no expected numeric values.
    """

    return RequiredProjectFactSpec(
        fact_type="speed.operational_context",
        output_type=ProjectFactOutputType.SPEED_ENVELOPE,
        aliases=("operational speed", "vehicle speed constraint"),
        unit_hints=("km/h", "kph", "kmh"),
        context_hints=(),
        required_fields=(
            "status", "operator", "value", "unit", "operating_mode",
            "condition", "semantic_scope", "source_block_id",
            "source_excerpt",
        ),
        structural_kind="OPERATIONAL_SPEED_CANDIDATE",
        exclusion_aliases=tuple(modes),
    )


def _method_risk_spec(spec: RequiredFactSpec) -> RequiredProjectFactSpec:
    canonical = spec.fact_type.value
    spaced = canonical.replace("_", " ").casefold()
    aliases = [canonical, spaced]
    if spec.fact_type is FactType.EXPOSURE:
        aliases.extend(("exposure method", "T/F"))
    structural_kind = (
        "CATEGORICAL_ALLOWED_SET_CANDIDATE"
        if spec.fact_type is FactType.DRIVER_IN_VEHICLE else ""
    )
    required_fields = (
        "status", "value", "unit", "context", "source_block_id",
    )
    if structural_kind:
        required_fields += ("semantic_scope", "source_excerpt")
    return RequiredProjectFactSpec(
        fact_type=canonical,
        output_type=ProjectFactOutputType.RISK_FACT,
        aliases=tuple(dict.fromkeys(aliases)),
        unit_hints=(spec.unit,) if spec.unit.strip() else (),
        context_hints=(),
        required_fields=required_fields,
        constraints=spec.constraints,
        structural_kind=structural_kind,
    )


def _unique_nonblank(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return tuple(result)


def _mode_identity(value: str) -> str:
    normalized = " ".join(str(value).split())
    head = re.split(r"[:：|]", normalized, maxsplit=1)[0].strip()
    return head or normalized


def _unique_modes(values: Sequence[str]) -> tuple[str, ...]:
    return _unique_nonblank(tuple(_mode_identity(value) for value in values))


def _unique_method_specs(
    specs: Sequence[RequiredFactSpec],
) -> tuple[RequiredFactSpec, ...]:
    result: list[RequiredFactSpec] = []
    seen: set[FactType] = set()
    for spec in specs:
        if spec.fact_type not in seen:
            seen.add(spec.fact_type)
            result.append(spec)
    return tuple(result)
