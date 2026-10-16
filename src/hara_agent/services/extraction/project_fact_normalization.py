from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from hara_agent.models import (
    ConstraintOperator, FactProvenance, ProjectFactOutputType, ReviewStatus,
    RiskFact, SourceRef, SpeedEnvelope,
)

from .project_fact_specs import RequiredProjectFactSpec
from hara_agent.models.project_facts import PHYSICAL_QUANTITY_UNITS, validated_physical_semantics


FOUND = "FOUND"
NOT_FOUND = "NOT_FOUND"
PHYSICAL_SEMANTICS_EXTRACTION_CONTRACT = (
    "For physical quantities preserve explicitly stated time/phase in an optional physical_semantics object. "
    "Allowed keys: analysis_value_semantic, time_role, impact_time_id, trajectory_id, branch_id, "
    "object_id, unit, assessment_reference (nonempty strings), model_valid_until_s (nonnegative seconds). "
    "Include physical_semantics_source_excerpt as an exact quote from the cited source. "
    "Copy explicit labels and identities; never invent a collision identity or assign a fault phase by default. "
    "Use {} when the source gives no phase. Normal performance, actuator capability and command limits "
    "are not fault motion bounds. Explicit natural-language statements of constant longitudinal "
    "acceleration after a fault may use POST_FAULT_CONSTANT_ACCELERATION with the complete source "
    "sentence, including qualifications. Deterministic normalization checks the phase, quantity and "
    "unit; an unsupported interpretation remains PENDING and never approves itself. "
)

# A bounded language normalization, not a physical or rating rule. Full sentences
# keep negation, hypothetical prefixes, limits and phase changes from being
# discarded by a keyword search. Unrecognized wording remains pending.
POST_FAULT_ACCELERATION_ZH_RULE = "physical-phase-post-fault-constant-acceleration-zh-v1"
_POST_FAULT_ACCELERATION_ZH = re.compile(
    r"(?:故障(?:发生)?后[，,]?(?:在(?:接触|碰撞)(?:前方|后方)?目标前[，,]?)?(?:车辆|本车|自车)"
    r"|(?:车辆|本车|自车)在故障(?:发生)?后)[，,]?"
    r"(?:沿(?:前进|倒车)方向)?以恒定(?:的)?纵向加速度(?:为|=)?"
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?P<unit>m/s(?:²|\^2|2))运动[。.]?"
)


def extracted_physical_semantics(
    raw: dict, sources: Sequence[SourceRef], *, normalization_evidence: list[dict] | None = None,
) -> tuple[dict, ReviewStatus]:
    """Normalize explicit source phases; unsupported interpretations stay pending.

    A citation validates the location, not an LLM's new engineering assertion.
    No default phase is assigned to an ordinary performance quantity.
    """
    semantics = validated_physical_semantics(raw.get("physical_semantics", {}))
    excerpt = raw.get("physical_semantics_source_excerpt", "")
    if excerpt and (not isinstance(excerpt, str) or not any(excerpt in s.excerpt for s in sources)):
        raise ValueError("INVALID_PHYSICAL_SEMANTICS: excerpt is not in cited evidence")
    texts = [excerpt] if excerpt else [s.excerpt for s in sources]
    unresolved = {key for key, value in semantics.items()
                  if not any(str(value) in text for text in texts)}
    if not unresolved:
        return semantics, ReviewStatus.FINALIZED
    if (unresolved == {"analysis_value_semantic"}
            and semantics["analysis_value_semantic"] == "POST_FAULT_CONSTANT_ACCELERATION"
            and raw.get("fact_type") == "longitudinal_acceleration_mps2"
            and raw.get("unit") in PHYSICAL_QUANTITY_UNITS["longitudinal_acceleration_mps2"]):
        # Inspect the full cited sentence, not a narrower phase excerpt which
        # could omit a surrounding condition or a conflicting phase.
        matches = [_POST_FAULT_ACCELERATION_ZH.fullmatch(re.sub(r"\s+", "", s.excerpt)) for s in sources]
        if matches and all(match and float(match["value"]) == raw.get("value") for match in matches):
            if normalization_evidence is not None:
                normalization_evidence.extend({
                    "field": "analysis_value_semantic", "canonical_value": semantics["analysis_value_semantic"],
                    "rule_id": POST_FAULT_ACCELERATION_ZH_RULE, "source_excerpt": source.excerpt,
                } for source in sources)
            return semantics, ReviewStatus.FINALIZED
    return semantics, ReviewStatus.PENDING


@dataclass(frozen=True)
class ProjectFactNormalizationFailure:
    fact_type: str
    code: str
    reason: str


@dataclass
class ProjectFactNormalizationResult:
    speed_envelopes: list[SpeedEnvelope] = field(default_factory=list)
    risk_facts: list[RiskFact] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    failures: list[ProjectFactNormalizationFailure] = field(default_factory=list)

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "speed_envelopes": [asdict(item) for item in self.speed_envelopes],
            "risk_facts": [asdict(item) for item in self.risk_facts],
            "coverage": self.coverage,
            "failures": [asdict(item) for item in self.failures],
        }


class ProjectFactNormalizer:
    """Fail-closed atomic normalizer; it never parses arbitrary prose."""

    OPERATOR_ALIASES = {
        "LT": ConstraintOperator.LT, "<": ConstraintOperator.LT,
        "LE": ConstraintOperator.LE, "<=": ConstraintOperator.LE,
        "≤": ConstraintOperator.LE, "≦": ConstraintOperator.LE,
        "â‰¤": ConstraintOperator.LE, "â‰¦": ConstraintOperator.LE,
        "EQ": ConstraintOperator.EQ, "=": ConstraintOperator.EQ,
        "GE": ConstraintOperator.GE, ">=": ConstraintOperator.GE,
        "≥": ConstraintOperator.GE, "≧": ConstraintOperator.GE,
        "â‰¥": ConstraintOperator.GE, "â‰§": ConstraintOperator.GE,
        "GT": ConstraintOperator.GT, ">": ConstraintOperator.GT,
        "＞": ConstraintOperator.GT, "ï¼ž": ConstraintOperator.GT,
        "RANGE": ConstraintOperator.RANGE,
    }
    UNIT_ALIASES = {
        "km/h": "km/h", "kph": "km/h", "kmh": "km/h",
        "m/sÂ²": "m/sÂ²", "m/s2": "m/sÂ²",
        "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
        "deg": "deg", "Â°": "deg",
    }
    def normalize(
        self,
        specs: Sequence[RequiredProjectFactSpec],
        raw_results: Any,
        source_blocks: Sequence[dict[str, str]],
        source_id: str,
        candidate_block_ids_by_fact: dict[str, set[str]] | None = None,
    ) -> ProjectFactNormalizationResult:
        result = ProjectFactNormalizationResult()
        if not isinstance(raw_results, list):
            for spec in specs:
                self._fail(result, spec.fact_type, "INVALID_COVERAGE", "results must be an array")
            return result
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in raw_results:
            if not isinstance(item, dict):
                self._fail(result, "", "INVALID_FACT_TYPE", "each result requires fact_type")
                continue
            candidate = dict(item)
            identity_resolution = "EXPLICIT"
            if not isinstance(candidate.get("fact_type"), str):
                self._fail(result, "", "INVALID_FACT_TYPE", "each result requires fact_type")
                continue
            candidate["_identity_resolution"] = identity_resolution
            by_type.setdefault(candidate["fact_type"], []).append(candidate)
        expected_types = {spec.fact_type for spec in specs}
        for unknown in sorted(set(by_type) - expected_types):
            self._fail(result, unknown, "INVALID_FACT_TYPE", "fact_type is not in the requested batch")
        block_map = {str(item.get("block_id", "")): item for item in source_blocks}
        seen_atomic: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        for spec in specs:
            candidates = by_type.get(spec.fact_type, [])
            if not candidates:
                self._fail(result, spec.fact_type, "INVALID_COVERAGE", "at least one FOUND/NOT_FOUND result is required")
                result.coverage.append({"fact_type": spec.fact_type, "status": "INVALID"})
                continue
            statuses = [str(item.get("status", "")).upper() for item in candidates]
            if NOT_FOUND in statuses:
                if len(candidates) != 1:
                    self._fail(
                        result, spec.fact_type, "INVALID_COVERAGE",
                        "NOT_FOUND must be the only result for a fact_type",
                    )
                    result.coverage.append({"fact_type": spec.fact_type, "status": "INVALID"})
                    continue
                result.coverage.append({
                    "fact_type": spec.fact_type,
                    "status": "NOT_FOUND_IN_EVIDENCE",
                    "source_truth_verified": False,
                })
                continue
            if any(status != FOUND for status in statuses):
                self._fail(result, spec.fact_type, "INVALID_COVERAGE", "status must be FOUND or NOT_FOUND")
                result.coverage.append({"fact_type": spec.fact_type, "status": "INVALID"})
                continue
            normalized_count = 0
            for candidate in candidates:
                missing = [name for name in spec.required_fields if candidate.get(name) is None]
                if missing:
                    self._fail(result, spec.fact_type, "MISSING_REQUIRED_FIELD", ",".join(missing))
                    continue
                try:
                    source = self._source_ref(
                        candidate, block_map, source_id, spec,
                        (candidate_block_ids_by_fact or {}).get(spec.fact_type),
                    )
                    fact = self._atomic_fact(spec, candidate, source)
                    scope, payload = self._identity(fact)
                    if scope in seen_atomic:
                        code = (
                            "DUPLICATE_FACT"
                            if seen_atomic[scope] == payload else "CONFLICTING_FACT"
                        )
                        raise ValueError(f"{code}: repeated atomic fact scope")
                    seen_atomic[scope] = payload
                    if isinstance(fact, SpeedEnvelope):
                        result.speed_envelopes.append(fact)
                    else:
                        result.risk_facts.append(fact)
                    normalized_count += 1
                except ValueError as error:
                    code, _, reason = str(error).partition(":")
                    self._fail(result, spec.fact_type, code, reason.strip() or code)
            result.coverage.append({
                "fact_type": spec.fact_type,
                "status": FOUND if normalized_count == len(candidates) else "ATOMICIZATION_FAILED",
                "normalized_count": normalized_count,
                "identity_resolution": candidates[0].get("_identity_resolution", "EXPLICIT"),
            })
        return result

    def _atomic_fact(self, spec, candidate, source):
        if spec.output_type is ProjectFactOutputType.SPEED_ENVELOPE:
            operator = self.OPERATOR_ALIASES.get(str(candidate.get("operator", "")))
            if operator is None:
                raise ValueError("INVALID_OPERATOR: unsupported operator")
            value = candidate.get("value")
            value_max = candidate.get("value_max")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("NON_ATOMIC_CANDIDATE: value must be one JSON number")
            if operator is ConstraintOperator.RANGE and (
                isinstance(value_max, bool) or not isinstance(value_max, (int, float))
            ):
                raise ValueError("NON_ATOMIC_CANDIDATE: RANGE requires one numeric value_max")
            if operator is not ConstraintOperator.RANGE and value_max is not None:
                raise ValueError("NON_ATOMIC_CANDIDATE: non-RANGE candidate cannot contain value_max")
            raw_unit = str(candidate.get("unit", ""))
            unit = self.UNIT_ALIASES.get(raw_unit)
            allowed_units = {self.UNIT_ALIASES.get(item, item) for item in spec.unit_hints}
            if unit is None or (spec.unit_hints and unit not in allowed_units):
                raise ValueError("INVALID_UNIT: unit is not allowed by the spec")
            mode = str(candidate.get("operating_mode", "")).strip()
            if not mode:
                raise ValueError("MISSING_CONTEXT: operating_mode is required")
            if spec.context_hints:
                mode = spec.context_hints[0]
            if operator is ConstraintOperator.RANGE:
                minimum, maximum = float(value), float(value_max)
            elif operator in {ConstraintOperator.LE, ConstraintOperator.LT, ConstraintOperator.EQ}:
                minimum, maximum = (
                    (float(value), float(value))
                    if operator is ConstraintOperator.EQ
                    else (0.0, float(value))
                )
            else:
                raise ValueError("INVALID_OPERATOR: speed lower-bound-only facts are unsupported")
            return SpeedEnvelope(
                mode, minimum, maximum, str(candidate.get("condition", "")),
                unit, [source], FactProvenance.PROJECT_INPUT,
                ReviewStatus.FINALIZED,
            )
        if spec.output_type is not ProjectFactOutputType.RISK_FACT:
            raise ValueError("INVALID_FACT_TYPE: unsupported output type")
        value = candidate.get("value")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("NON_ATOMIC_CANDIDATE: value must be one string or number")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("NON_ATOMIC_CANDIDATE: value must not be blank")
        raw_unit = str(candidate.get("unit", "")).strip()
        expected_unit = raw_unit if raw_unit in spec.unit_hints else (spec.unit_hints[0] if spec.unit_hints else "")
        if expected_unit and raw_unit != expected_unit:
            raise ValueError("INVALID_UNIT: unit must match the Method Contract")
        if not expected_unit and raw_unit:
            raise ValueError("INVALID_UNIT: unit is not defined by the Method Contract")
        context = candidate.get("context")
        if not isinstance(context, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in context.items()
        ):
            raise ValueError("MISSING_CONTEXT: context must be a string-to-string object")
        normalization_evidence = []
        semantics, semantic_approval = extracted_physical_semantics(
            {**candidate, "fact_type": spec.fact_type}, [source], normalization_evidence=normalization_evidence)
        identity = json.dumps(
            {
                "parameter": spec.fact_type,
                "value": value,
                "unit": expected_unit,
                "context": context,
                "source": asdict(source),
                "physical_semantics": semantics,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return RiskFact(
            fact_id="RF-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
            parameter=spec.fact_type,
            value=value,
            unit=expected_unit,
            context=dict(context),
            source_refs=[source],
            provenance=FactProvenance.PROJECT_INPUT,
            approval=semantic_approval,
            produced_by="validated_item_project_fact_extraction",
            physical_semantics=semantics,
            normalization_evidence=normalization_evidence,
        )

    @staticmethod
    def _source_ref(candidate, block_map, source_id, spec, allowed_block_ids=None):
        block_id = str(candidate.get("source_block_id", ""))
        block = block_map.get(block_id)
        if block is None:
            raise ValueError("UNKNOWN_SOURCE_REF: source_block_id was not routed")
        if allowed_block_ids is not None and block_id not in allowed_block_ids:
            raise ValueError("UNKNOWN_SOURCE_REF: source_block_id is support-only for this spec")
        excerpt = str(candidate.get("source_excerpt", ""))
        source_text = str(block.get("text", ""))
        has_exact_alias = any(
            alias.casefold() in source_text.casefold() for alias in spec.aliases
        )
        if not has_exact_alias and not spec.structural_kind:
            raise ValueError("UNKNOWN_SOURCE_REF: source block contains no exact spec alias")
        if spec.structural_kind:
            expected_scope = {
                "OPERATIONAL_SPEED_CANDIDATE": "OPERATIONAL_SPEED",
                "CATEGORICAL_ALLOWED_SET_CANDIDATE": "DRIVER_CONFIGURATION",
            }[spec.structural_kind]
            if str(candidate.get("semantic_scope", "")) != expected_scope:
                raise ValueError("INVALID_SEMANTIC_SCOPE: structural candidate was not semantically confirmed")
            if not excerpt:
                raise ValueError("UNKNOWN_SOURCE_REF: structural candidate requires an exact excerpt")
            if (
                spec.structural_kind == "OPERATIONAL_SPEED_CANDIDATE"
                and ("±" in excerpt or re.search(r"(?<!\d)[-−]\s*\d", excerpt))
            ):
                raise ValueError("NOT_OPERATIONAL_SPEED: signed capability range is not an operating speed")
        if excerpt and excerpt not in source_text:
            raise ValueError("UNKNOWN_SOURCE_REF: excerpt is not an exact substring of the source block")
        if excerpt and _POST_FAULT_ACCELERATION_ZH.fullmatch(re.sub(r"\s+", "", excerpt)):
            # The model's quote must not cut "假设" or another qualification
            # off the source sentence and thereby turn it into an assertion.
            boundaries = "。！？!?;；\n"
            spans = list(re.finditer(re.escape(excerpt), source_text))
            complete = any(
                (not source_text[:m.start()].strip(" \t") or source_text[:m.start()].rstrip(" \t")[-1] in boundaries)
                and (excerpt.rstrip()[-1] in boundaries or not source_text[m.end():].strip()
                     or source_text[m.end():].lstrip(" \t")[0] in boundaries)
                for m in spans)
            if not complete:
                excerpt = source_text
        return SourceRef(
            "item_definition", source_id, str(block.get("location", "")),
            excerpt or source_text,
        )

    @staticmethod
    def _identity(fact):
        if isinstance(fact, SpeedEnvelope):
            source_scope = tuple(
                (item.location, item.excerpt) for item in fact.sources
            )
            scope = (
                "speed", fact.operating_mode.casefold(),
                " ".join(fact.condition.split()).casefold(), source_scope,
            )
            payload = (fact.speed_min_kph, fact.speed_max_kph, fact.unit)
            return scope, payload
        source_scope = tuple(
            (item.location, item.excerpt) for item in fact.source_refs
        )
        scope = (
            "risk", fact.parameter, tuple(sorted(fact.context.items())),
            source_scope,
        )
        return scope, (fact.value, fact.unit, json.dumps(fact.physical_semantics, sort_keys=True))

    @staticmethod
    def _fail(result, fact_type, code, reason):
        result.failures.append(ProjectFactNormalizationFailure(fact_type, code, reason))
