from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from typing import Any, Callable, Sequence

from hara_agent.contracts import (
    FactOrigin, FactType, RequiredFactSpec, RiskFactResolution,
    RiskFactResolutionStatus,
)
from hara_agent.infrastructure.llm import LLMClient, LLMRequest, LLMSchemaContractError
from hara_agent.models import (
    FactProvenance, MalfunctionCandidate, ReviewStatus, RiskFact,
    ScenarioCandidate, SourceRef, evaluate_risk_eligibility_payload,
)
from hara_agent.models.project_facts import risk_fact_from_dict
from hara_agent.services.extraction.project_fact_normalization import (
    extracted_physical_semantics, PHYSICAL_SEMANTICS_EXTRACTION_CONTRACT,
)
from hara_agent.services.analysis.method_risk_fact_service import METHOD_FACT_KEYS
from hara_agent.services.analysis.scenario_physics import source_is_accepted_for


SCENARIO_RISK_FACT_BATCH_CACHE_VERSION = "scenario-risk-fact-batch-v2"


class ScenarioRiskFactAgent:
    """Interpret only template-requested scenario facts from exact evidence.

    The output is deliberately template-independent ``RiskFact`` data. Template
    identity is introduced later by ``MethodRiskFactBindingService``.
    HUMAN_EVIDENCE and DERIVED_FACT requirements are never guessed here.
    """

    PROMPT_VERSION = "scenario-risk-facts-v8-request-missing-inputs"

    def __init__(self, client: LLMClient, *, batch_max_results: int | None = None):
        self.client = client
        self.batch_max_results = batch_max_results or int(os.getenv(
            "HARA_SCENARIO_RISK_FACT_BATCH_MAX_RESULTS", "16"
        ))
        if self.batch_max_results <= 0:
            raise ValueError("HARA_SCENARIO_RISK_FACT_BATCH_MAX_RESULTS must be positive")

    @staticmethod
    def _allowed_values(spec: RequiredFactSpec) -> list[str]:
        values: list[str] = []
        for constraint in spec.constraints:
            values.extend(re.findall(r"'([^']+)'", constraint))
            match = re.search(r"\bEQ\s+([A-Z][A-Z0-9_]*)", constraint)
            if match:
                values.append(match.group(1))
        return list(dict.fromkeys(values))

    @staticmethod
    def _existing_input(pair, spec):
        """Keep scoped source/model inputs in their existing authority domain."""
        if spec.fact_type is FactType.SPEED_UNSPECIFIED:
            return False
        candidate = pair["candidate"]
        key = METHOD_FACT_KEYS[spec.fact_type]
        value = candidate.facts.get(key)
        allowed = ScenarioRiskFactAgent._allowed_values(spec)
        return (value is not None and value != "" and (not allowed or str(value).upper() in allowed)
            and source_is_accepted_for(
            candidate.fact_provenance.get(key, {}),
            malfunction_id=pair["malfunction_id"], scenario_id=pair["scenario_id"]))

    @staticmethod
    def _evidence(
        candidates: Sequence[ScenarioCandidate],
        malfunctions: Sequence[MalfunctionCandidate],
    ) -> tuple[list[dict[str, str]], dict[str, SourceRef]]:
        refs = list(dict.fromkeys(
            source
            for item in [*candidates, *malfunctions]
            for source in item.sources
            if (
                source.source_type == "item_definition"
                and source.source_id
                and source.excerpt
            )
        ))
        registry = {f"E{index:04d}": source for index, source in enumerate(refs, start=1)}
        return ([
            {
                "evidence_id": evidence_id,
                "source_type": source.source_type,
                "source_id": source.source_id,
                "location": source.location,
                "excerpt": source.excerpt,
            }
            for evidence_id, source in registry.items()
        ], registry)

    @staticmethod
    def _allowed_evidence_ids(
        pairs: Sequence[dict[str, Any]],
        registry: dict[str, SourceRef],
    ) -> dict[tuple[str, str], set[str]]:
        """Scope batch evidence to the malfunction/scenario pair that supplied it.

        The Provider receives a compact batch-wide evidence catalog for efficiency,
        but a fact for one pair must never cite an excerpt introduced only by another
        pair in the same batch.  Source identity is exact; no prose or domain
        inference is performed here.
        """
        evidence_id_by_source = {
            source: evidence_id for evidence_id, source in registry.items()
        }
        allowed: dict[tuple[str, str], set[str]] = {}
        for pair in pairs:
            pair_id = (
                str(pair["malfunction_id"]),
                str(pair["scenario_id"]),
            )
            sources = list(dict.fromkeys(
                source
                for item in (
                    pair["candidate"], pair["malfunction_candidate"],
                )
                for source in item.sources
                if (
                    source.source_type == "item_definition"
                    and source.source_id
                    and source.excerpt
                )
            ))
            allowed[pair_id] = {
                evidence_id_by_source[source]
                for source in sources if source in evidence_id_by_source
            }
        return allowed

    def interpret(
        self,
        assessments: Sequence[dict[str, Any]],
        candidates: Sequence[ScenarioCandidate],
        malfunctions: Sequence[MalfunctionCandidate],
        required_specs: Sequence[RequiredFactSpec],
        *,
        batch_cache: dict[str, Any] | None = None,
        on_batch: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> tuple[list[RiskFact], dict[str, Any]]:
        specs = [item for item in required_specs if item.origin is FactOrigin.SCENARIO_FACT]
        retained = [
            item for item in assessments
            if isinstance(item, dict) and evaluate_risk_eligibility_payload(
                item, allow_legacy_boolean_gate=True,
            ).eligible
        ]
        if not specs or not retained:
            return [], {
                "prompt_version": self.PROMPT_VERSION,
                "requested_pair_count": len(retained),
                "requested_fact_type_count": len(specs),
                "found_fact_count": 0,
                "missing_fact_count": len(retained) * len(specs),
                "llm_call_count": 0,
                "fact_resolutions": [],
            }

        candidate_by_id = {item.scenario_id: item for item in candidates}
        malfunction_by_id = {item.malfunction_id: item for item in malfunctions}
        pair_material = []
        for assessment in retained:
            scenario_id = str(assessment.get("scenario_id", ""))
            malfunction_id = str(assessment.get("malfunction_id", ""))
            if scenario_id not in candidate_by_id or malfunction_id not in malfunction_by_id:
                raise ValueError(
                    f"Risk fact interpretation foreign key missing: {malfunction_id}/{scenario_id}"
                )
            candidate = candidate_by_id[scenario_id]
            malfunction = malfunction_by_id[malfunction_id]
            pair_material.append({
                "malfunction_id": malfunction_id,
                "scenario_id": scenario_id,
                "malfunction": malfunction.description,
                "hazard": malfunction.vehicle_level_hazard,
                "scenario": candidate.situational_description,
                "scenario_detail": candidate.situational_detailing,
                "hazardous_event": str(assessment.get("hazardous_event", "")),
                "potential_harm": str(assessment.get("potential_harm", "")),
                "candidate": candidate,
                "malfunction_candidate": malfunction,
            })
        speed_specs = [
            item for item in specs if item.fact_type is FactType.SPEED_UNSPECIFIED
        ]
        road_user_specs = [
            item for item in specs if item.fact_type is FactType.ROAD_USER_TYPE
        ]
        primary_specs = [
            item for item in specs
            if item.fact_type not in {
                FactType.SPEED_UNSPECIFIED, FactType.ROAD_USER_TYPE,
            }
        ]
        facts, batch_audits, cache_hits = self._interpret_specs(
            pair_material, primary_specs, batch_cache=batch_cache, on_batch=on_batch,
        )
        collision_by_pair = {
            (pair["malfunction_id"], pair["scenario_id"]): pair["candidate"].facts["collision_type"]
            for pair in pair_material for spec in specs
            if spec.fact_type is FactType.COLLISION_TYPE and self._existing_input(pair, spec)
        }
        collision_by_pair.update({
            (
                item.context.get("malfunction_id", ""),
                item.context.get("scenario_id", ""),
            ): str(item.value).upper()
            for item in facts if item.parameter == FactType.COLLISION_TYPE.value
        })
        road_user_pairs = [
            pair for pair in pair_material
            if collision_by_pair.get((
                str(pair["malfunction_id"]), str(pair["scenario_id"]),
            )) == "VEHICLE_TO_ROAD_USER"
        ]
        road_facts, road_audits, road_cache_hits = self._interpret_specs(
            road_user_pairs, road_user_specs,
            batch_cache=batch_cache, on_batch=on_batch,
        )
        facts.extend(road_facts)
        batch_audits.extend(road_audits)
        cache_hits += road_cache_hits

        fact_by_key = {
            (
                item.context.get("malfunction_id", ""),
                item.context.get("scenario_id", ""),
                item.parameter,
            ): item for item in facts
        }
        not_found_keys = {
            tuple(key) for item in batch_audits
            for key in item.get("not_found_keys", [])
        }
        unresolved_keys = {
            tuple(key) for item in batch_audits
            for key in item.get("unresolved_contract_keys", [])
        }
        resolutions: list[RiskFactResolution] = []
        for pair in pair_material:
            pair_id = (
                str(pair["malfunction_id"]), str(pair["scenario_id"]),
            )
            collision = collision_by_pair.get(pair_id, "")
            for spec in specs:
                key = (*pair_id, spec.fact_type.value)
                fact = fact_by_key.get(key)
                if self._existing_input(pair, spec):
                    status = RiskFactResolutionStatus.FOUND
                    reason = "Existing scoped Scenario input; original provenance/approval retained, not re-extracted as an Item fact."
                    fact_id = "SCN." + METHOD_FACT_KEYS[spec.fact_type]
                elif fact is not None:
                    status = RiskFactResolutionStatus.FOUND
                    reason = "A source-grounded fact passed the bounded schema contract."
                    fact_id = fact.fact_id
                elif spec in speed_specs:
                    status = RiskFactResolutionStatus.UNRESOLVED_METHOD_SEMANTICS
                    reason = (
                        "The template speed variable is explicitly unresolved; an ego, "
                        "relative, impact, or delta-V meaning must be approved before binding."
                    )
                    fact_id = ""
                elif spec in road_user_specs and collision != "VEHICLE_TO_ROAD_USER":
                    if collision:
                        status = RiskFactResolutionStatus.NOT_APPLICABLE
                        reason = (
                            "ROAD_USER_TYPE is not required for the resolved non-road-user "
                            "collision configuration."
                        )
                    else:
                        status = RiskFactResolutionStatus.UNRESOLVED_DEPENDENCY
                        reason = (
                            "ROAD_USER_TYPE was not requested because COLLISION_TYPE is unresolved."
                        )
                    fact_id = ""
                elif key in not_found_keys:
                    status = RiskFactResolutionStatus.ABSENT_IN_EVIDENCE
                    reason = "No exact supporting excerpt was present in the supplied evidence."
                    fact_id = ""
                elif key in unresolved_keys:
                    status = RiskFactResolutionStatus.EXTRACTION_FAILED
                    reason = "The Provider result remained invalid after one bounded repair."
                    fact_id = ""
                else:
                    status = RiskFactResolutionStatus.UNRESOLVED_DEPENDENCY
                    reason = "A prerequisite fact required to request this fact is unresolved."
                    fact_id = ""
                resolutions.append(RiskFactResolution(
                    malfunction_id=pair_id[0], scenario_id=pair_id[1],
                    fact_type=spec.fact_type.value, status=status,
                    reason=reason, fact_id=fact_id,
                ))
        expected_count = len(pair_material) * len(specs)
        resolution_counts = {
            status.value: sum(item.status is status for item in resolutions)
            for status in RiskFactResolutionStatus
        }
        return facts, {
            "prompt_version": self.PROMPT_VERSION,
            "models": [item["model"] for item in batch_audits],
            "request_ids": [item["request_id"] for item in batch_audits],
            "usage": {"batches": [item["usage"] for item in batch_audits]},
            "requested_pair_count": len(retained),
            "requested_fact_type_count": len(specs),
            "candidate_fact_pair_count": expected_count,
            "llm_requested_fact_pair_count": (
                sum(not self._existing_input(pair, spec) for pair in pair_material for spec in primary_specs)
                + sum(not self._existing_input(pair, spec) for pair in road_user_pairs for spec in road_user_specs)
            ),
            "found_fact_count": len(facts),
            "existing_fact_count": sum(self._existing_input(pair, spec) for pair in pair_material for spec in specs),
            "missing_fact_count": expected_count - resolution_counts[RiskFactResolutionStatus.FOUND.value],
            "batch_count": len(batch_audits),
            "coverage_repair_count": sum(
                item["coverage_repair_count"] for item in batch_audits
            ),
            "llm_call_count": sum(item["llm_call_count"] for item in batch_audits),
            "batch_cache_hit_count": cache_hits,
            "validation_failure_count": sum(
                item.get("validation_failure_count", 0) for item in batch_audits
            ),
            "unresolved_contract_count": sum(
                item.get("unresolved_contract_count", 0) for item in batch_audits
            ),
            "unresolved_contract_keys": [
                key for item in batch_audits
                for key in item.get("unresolved_contract_keys", [])
            ],
            "resolution_counts": resolution_counts,
            "fact_resolutions": [item.to_dict() for item in resolutions],
            "facts": [asdict(item) for item in facts],
        }

    def _interpret_specs(
        self,
        pair_material: Sequence[dict[str, Any]],
        specs: Sequence[RequiredFactSpec],
        *,
        batch_cache: dict[str, Any] | None,
        on_batch: Callable[[str, dict[str, Any]], None] | None,
    ) -> tuple[list[RiskFact], list[dict[str, Any]], int]:
        if not pair_material or not specs:
            return [], [], 0
        groups = {}
        for pair in pair_material:
            needed = tuple(spec for spec in specs if not self._existing_input(pair, spec))
            if needed:
                key = tuple(spec.fact_type for spec in needed)
                groups.setdefault(key, (needed, []))[1].append(pair)
        facts: list[RiskFact] = []
        audits: list[dict[str, Any]] = []
        cache_hits = 0
        batches = [(pairs[start:start + max(1, self.batch_max_results // len(needed))], needed)
                   for needed, pairs in groups.values()
                   for start in range(0, len(pairs), max(1, self.batch_max_results // len(needed)))]
        for batch_pairs, specs in batches:
            cache_key = self._batch_cache_key(batch_pairs, specs)
            cached = self._load_batch_cache(
                (batch_cache or {}).get(cache_key), batch_pairs,
            )
            if cached is not None:
                batch_facts, batch_audit = cached
                cache_hits += 1
            else:
                batch_facts, batch_audit = self._interpret_batch(batch_pairs, specs)
                entry = {
                    "cache_version": SCENARIO_RISK_FACT_BATCH_CACHE_VERSION,
                    "cache_key": cache_key,
                    "facts": [asdict(item) for item in batch_facts],
                    "audit": batch_audit,
                }
                if batch_cache is not None:
                    batch_cache[cache_key] = entry
                if on_batch is not None:
                    on_batch(cache_key, entry)
            facts.extend(batch_facts)
            audits.append(batch_audit)
        return facts, audits, cache_hits

    def _batch_cache_key(
        self,
        pairs: Sequence[dict[str, Any]],
        specs: Sequence[RequiredFactSpec],
    ) -> str:
        config = getattr(self.client, "config", None)
        material = {
            "cache_version": SCENARIO_RISK_FACT_BATCH_CACHE_VERSION,
            "prompt_version": self.PROMPT_VERSION,
            "provider": getattr(config, "provider", type(self.client).__name__),
            "base_url": getattr(config, "base_url", ""),
            "model": getattr(config, "model", type(self.client).__name__),
            "pairs": [{
                **{
                    key: value for key, value in pair.items()
                    if key not in {"candidate", "malfunction_candidate"}
                },
                "candidate": asdict(pair["candidate"]),
                "malfunction_candidate": asdict(pair["malfunction_candidate"]),
            } for pair in pairs],
            "specs": [asdict(item) for item in specs],
        }
        serialized = json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_batch_cache(
        self,
        entry: object,
        pairs: Sequence[dict[str, Any]],
    ) -> tuple[list[RiskFact], dict[str, Any]] | None:
        if (
            not isinstance(entry, dict)
            or entry.get("cache_version") != SCENARIO_RISK_FACT_BATCH_CACHE_VERSION
        ):
            return None
        try:
            facts = [risk_fact_from_dict(item) for item in entry.get("facts", [])]
            audit = dict(entry["audit"])
        except (KeyError, TypeError, ValueError):
            return None
        expected_pairs = {
            (str(item["malfunction_id"]), str(item["scenario_id"]))
            for item in pairs
        }
        if any(
            (
                item.context.get("malfunction_id", ""),
                item.context.get("scenario_id", ""),
            ) not in expected_pairs
            or item.produced_by != self.PROMPT_VERSION
            or item.approval is not ReviewStatus.FINALIZED
            for item in facts
        ):
            return None
        return facts, audit

    def _interpret_batch(
        self,
        pairs: Sequence[dict[str, Any]],
        specs: Sequence[RequiredFactSpec],
    ) -> tuple[list[RiskFact], dict[str, Any]]:
        candidates = [item["candidate"] for item in pairs]
        malfunctions = [item["malfunction_candidate"] for item in pairs]
        evidence, registry = self._evidence(candidates, malfunctions)
        allowed_evidence_ids = self._allowed_evidence_ids(pairs, registry)
        fact_requests = []
        expected: set[tuple[str, str, str]] = set()
        spec_by_type = {item.fact_type.value: item for item in specs}
        for pair in pairs:
            context = {
                key: value for key, value in pair.items()
                if key not in {"candidate", "malfunction_candidate"}
            }
            for spec in specs:
                key = (
                    str(pair["malfunction_id"]),
                    str(pair["scenario_id"]),
                    spec.fact_type.value,
                )
                expected.add(key)
                fact_requests.append({
                    **context,
                    "fact_type": spec.fact_type.value,
                    "unit": spec.unit,
                    "allowed_values": self._allowed_values(spec),
                    "constraints": list(spec.constraints),
                })
        request = self._request(fact_requests, evidence, coverage_repair=False)
        response = None
        try:
            response = self.client.complete_json(request)
        except LLMSchemaContractError as error:
            # An invalid envelope has no validated results. Use the same single
            # coverage repair as missing/invalid result entries.
            facts, seen, not_found = [], set(), set()
            validation_failures = [f"primary response envelope: {error}"]
        else:
            facts, seen, not_found, validation_failures = self._parse_results(
                response.data, expected, spec_by_type, registry,
                allowed_evidence_ids,
            )
        missing = sorted(expected - seen)
        repair_response = None
        if missing:
            missing_set = set(missing)
            repair_requests = [
                item for item in fact_requests
                if (
                    str(item["malfunction_id"]),
                    str(item["scenario_id"]),
                    str(item["fact_type"]),
                ) in missing_set
            ]
            repair_request = self._request(
                repair_requests, evidence, coverage_repair=True,
            )
            try:
                repair_response = self.client.complete_json(repair_request)
            except LLMSchemaContractError as error:
                validation_failures.append(f"repair response envelope: {error}")
            else:
                repaired, repair_seen, repair_not_found, repair_failures = self._parse_results(
                    repair_response.data, missing_set, spec_by_type, registry,
                    allowed_evidence_ids,
                )
                facts.extend(repaired)
                seen.update(repair_seen)
                not_found.update(repair_not_found)
                validation_failures.extend(repair_failures)
        unresolved = sorted(expected - seen)
        audit_response = response or repair_response
        return facts, {
            "model": audit_response.model if audit_response else "",
            "request_id": audit_response.request_id if audit_response else "",
            "usage": {
                "primary": response.usage if response else {},
                "repair": repair_response.usage if repair_response else {},
            },
            "coverage_repair_count": int(bool(missing)),
            "llm_call_count": 1 + int(bool(missing)),
            "validation_failure_count": len(validation_failures),
            "validation_failures": validation_failures,
            "unresolved_contract_count": len(unresolved),
            "unresolved_contract_keys": [list(item) for item in unresolved],
            "not_found_keys": [list(item) for item in sorted(not_found)],
        }

    def _request(
        self,
        fact_requests: Sequence[dict[str, Any]],
        evidence: Sequence[dict[str, str]],
        *,
        coverage_repair: bool,
    ) -> LLMRequest:
        return LLMRequest(
            task="interpret_scenario_risk_facts",
            schema_name="ScenarioRiskFacts",
            prompt_version=self.PROMPT_VERSION,
            system_prompt=(
                "You are a bounded evidence interpreter. Return only facts explicitly supported by "
                "the supplied evidence excerpts. Do not estimate speed, frequency, duration, or "
                "avoidability. Do not use template identity or invent engineering rules."
                " Scenario/hazard descriptions may contain analytical hypotheses; they are not source excerpts. "
                "A source saying parking is supported does not establish a collision type. Return NOT_FOUND "
                "when the cited excerpt does not itself support the requested value."
                + PHYSICAL_SEMANTICS_EXTRACTION_CONTRACT
            ),
            user_prompt=(
                "Return JSON {\"results\":[...]}. Return exactly one result for every item in "
                "fact_requests. Do not return any unrequested identity. status is FOUND or NOT_FOUND. "
                "FOUND requires value, unit, and one or more evidence_ids. Categorical values must "
                "come from allowed_values; numeric values must be JSON numbers. Evidence IDs must "
                "come from evidence. No Markdown.\n"
                + (
                    "This is the single bounded coverage repair. Return only the explicitly "
                    "listed missing fact_requests.\n" if coverage_repair else ""
                )
                + f"fact_requests={json.dumps(fact_requests, ensure_ascii=False, separators=(',', ':'))}\n"
                f"evidence={json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
            ),
            metadata={
                "result_count": len(fact_requests),
                "coverage_repair": coverage_repair,
            },
            max_tokens=min(
                int(getattr(getattr(self.client, "config", None), "max_tokens", 4096)), 4096
            ),
        )

    def _parse_results(
        self,
        data: dict[str, Any],
        expected: set[tuple[str, str, str]],
        spec_by_type: dict[str, RequiredFactSpec],
        registry: dict[str, SourceRef],
        allowed_evidence_ids: dict[tuple[str, str], set[str]],
    ) -> tuple[
        list[RiskFact], set[tuple[str, str, str]], set[tuple[str, str, str]],
        list[str],
    ]:
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return [], set(), set(), ["response.results must be an array"]
        seen: set[tuple[str, str, str]] = set()
        not_found: set[tuple[str, str, str]] = set()
        facts: list[RiskFact] = []
        failures: list[str] = []
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for raw in raw_results:
            if not isinstance(raw, dict):
                failures.append("result must be an object")
                continue
            key = (
                str(raw.get("malfunction_id", "")),
                str(raw.get("scenario_id", "")),
                str(raw.get("fact_type", "")),
            )
            if key not in expected:
                failures.append(f"unexpected result identity: {key}")
                continue
            grouped.setdefault(key, []).append(raw)
        for key, items in grouped.items():
            if len(items) != 1:
                failures.append(f"duplicate result identity: {key}")
                continue
            raw = items[0]
            status = str(raw.get("status", "")).upper()
            if status == "NOT_FOUND":
                seen.add(key)
                not_found.add(key)
                continue
            try:
                if status != "FOUND":
                    raise ValueError(f"invalid status: {status!r}")
                spec = spec_by_type[key[2]]
                value = raw.get("value")
                allowed = self._allowed_values(spec)
                if allowed:
                    value = str(value).strip().upper()
                    if value not in allowed:
                        raise ValueError("value is outside the compiled ontology")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("numeric value must be a JSON number")
                unit = str(raw.get("unit", ""))
                if unit != spec.unit:
                    raise ValueError("unit does not match the compiled contract")
                evidence_ids = raw.get("evidence_ids")
                if not isinstance(evidence_ids, list) or not evidence_ids:
                    raise ValueError("FOUND requires evidence_ids")
                evidence_ids = [str(item) for item in evidence_ids]
                pair_allowed = allowed_evidence_ids.get((key[0], key[1]), set())
                disallowed = sorted(set(evidence_ids) - pair_allowed)
                if disallowed:
                    raise ValueError(
                        "evidence_ids are not scoped to this malfunction/scenario pair: "
                        f"{disallowed}"
                    )
                sources = list(dict.fromkeys(
                    registry[item] for item in evidence_ids
                ))
                normalization_evidence = []
                semantics, semantic_approval = extracted_physical_semantics(
                    raw, sources, normalization_evidence=normalization_evidence)
            except (KeyError, TypeError, ValueError) as error:
                failures.append(f"invalid result {key}: {error}")
                continue
            material = json.dumps(
                {"key": key, "value": value, "unit": unit, "physical_semantics": semantics},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            facts.append(RiskFact(
                fact_id=f"RF-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20].upper()}",
                parameter=spec.fact_type.value,
                value=value,
                unit=unit,
                context={"malfunction_id": key[0], "scenario_id": key[1]},
                source_refs=sources,
                provenance=FactProvenance.DERIVED,
                approval=semantic_approval,
                produced_by=self.PROMPT_VERSION,
                physical_semantics=semantics,
                normalization_evidence=normalization_evidence,
            ))
            seen.add(key)
        return facts, seen, not_found, failures
