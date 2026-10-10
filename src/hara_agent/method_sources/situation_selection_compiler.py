"""Compile normalized reference knowledge; runtime services never read YAML."""

from dataclasses import fields
import re

from hara_agent.contracts.situation_selection import (
    ExposureParityCase, MalfunctionSituationSelectionMethod,
    SituationAssessmentExample, SituationFamily, SituationMethodPrinciple,
    SituationReference, SituationSelectionRule, SituationSourceDocument,
)


def _record(cls, raw, *, tuples=(), refs=None):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
        raise ValueError(f"Invalid normalized {cls.__name__} fields")
    value = dict(raw)
    for key in tuples:
        if not isinstance(value.get(key), list):
            raise ValueError(f"{cls.__name__}.{key} must be a list")
        if len(set(value[key])) != len(value[key]) and key != "operands":
            raise ValueError(f"Duplicate {cls.__name__}.{key}")
        value[key] = tuple(value[key])
    if refs is not None:
        value["source_refs"] = refs(value.get("source_refs"))
    return cls(**value)


def compile_situation_selection(raw, *, asset, source_hash, atoms, failure_types):
    atom_ids = set(atoms)
    allowed = {
        "schema_version", "source_classification", "selection_policy", "documents",
        "functional_outputs", "families", "rules", "principles", "assessment_examples",
        "exposure_parity_cases",
    }
    if (
        not isinstance(raw, dict) or set(raw) != allowed
        or raw["schema_version"] != "1.0"
        or raw["source_classification"] != "VALIDATED_PROJECT_HARA_REFERENCE"
        or raw["selection_policy"] != "CONSERVATIVE_EVIDENCE_FIRST"
    ):
        raise ValueError("Invalid normalized malfunction situation selection schema")
    for key in allowed - {"schema_version", "source_classification", "selection_policy"}:
        if not isinstance(raw[key], list):
            raise ValueError(f"Situation selection {key} must be a list")
    documents = tuple(_record(SituationSourceDocument, x) for x in raw["documents"])
    by_doc = {d.document_key: d for d in documents}
    if len(by_doc) != len(documents) or not by_doc or any(
        not d.document_key or not d.filename
        or not re.fullmatch(r"[0-9a-fA-F]{64}", d.sha256)
        or d.source_kind not in {"REFERENCE_PROJECT", "REFERENCE_IMPLEMENTATION", "CURRENT_PROJECT_APPROVAL"}
        for d in documents
    ):
        raise ValueError("Invalid situation source documents")

    def references(values):
        if not isinstance(values, list) or not values:
            raise ValueError("Source references are required")
        result = tuple(_record(SituationReference, x, tuples=("rows",)) for x in values)
        for ref in result:
            if (
                ref.document_key not in by_doc or not ref.worksheet or not ref.columns
                or not ref.rows or any(type(n) is not int or n < 1 for n in ref.rows)
                or not ref.rule_text_or_summary or not ref.project_scope
                or ref.reference_status not in {"EXTRACTED_REFERENCE", "CONFIRMED_IN_SOURCE", "SOURCE_REVIEW", "CURRENT_PROJECT_APPROVED"}
                or ref.cross_version_mapping_status not in {"CROSS_SUPPORTED", "SINGLE_SOURCE", "SOURCE_SCOPE_REVIEW", "NOT_APPLICABLE"}
            ):
                raise ValueError("Invalid source reference provenance")
        return result

    families = tuple(_record(SituationFamily, x) for x in raw["families"])
    by_family = {f.family_id: f for f in families}
    if len(by_family) != len(families) or not by_family:
        raise ValueError("Duplicate or missing situation families")
    for f in families:
        if (
            not f.family_id or not f.mapping_source
            or f.source_version not in {"V1", "NONE"}
            or f.canonical_version not in {"V2", "NONE"}
            or f.mapping_status not in {"MAPPED", "SOURCE_FAMILY_ONLY", "MAPPING_REVIEW_REQUIRED", "VERSION_CONFLICT", "PROJECT_FACT_REQUIRED", "SOURCE_SCOPE_REVIEW"}
            or (f.source_id and (
                f.source_version != "V1" or atoms.get(f.source_id, {}).get("ver") != "V1"
            ))
            or (f.mapping_status == "MAPPED" and (
                not f.source_id or f.source_version != "V1" or f.canonical_version != "V2"
                or f.source_id not in atom_ids or f.candidate_canonical_id not in atom_ids
                or atoms.get(f.source_id, {}).get("v2") != f.candidate_canonical_id
                or atoms.get(f.candidate_canonical_id, {}).get("ver") != "V2"
            ))
            or (f.mapping_status != "MAPPED" and (
                f.candidate_canonical_id or f.canonical_version != "NONE"
            ))
            or (f.mapping_status == "SOURCE_FAMILY_ONLY" and f.source_id not in atom_ids)
        ):
            raise ValueError(f"Invalid situation version mapping: {f.family_id}")
    outputs = raw["functional_outputs"]
    if not outputs or any(not isinstance(x, str) or not x for x in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Invalid functional output vocabulary")
    rules = tuple(_record(
        SituationSelectionRule, x, refs=references,
        tuples=("source_failure_semantics", "canonical_failure_types", "supported_families"),
    ) for x in raw["rules"])
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError("Duplicate situation rule IDs")
    for r in rules:
        if (
            not r.rule_id or not r.rationale or r.functional_output not in outputs
            or not r.source_failure_semantics
            or not r.canonical_failure_types or not set(r.canonical_failure_types) <= failure_types
            or not set(r.supported_families) <= set(by_family)
            or r.function_scope not in {"WITHIN_FUNCTION", "OUTSIDE_FUNCTION", "UNRESOLVED"}
            or r.action not in {"INCLUDE", "EXCLUDE", "AUDIT"}
            or r.authority not in {"SOURCE_SCENARIO_GUIDANCE", "SOURCE_PROJECT_SPECIFIC", "SOURCE_UNRESOLVED", "APPROVED_PROJECT_CONSTRAINT"}
            or type(r.runtime_exclusion_authority) is not bool
            or r.approval not in {"REFERENCE_ONLY", "APPROVED"} or not r.project_scope
        ):
            raise ValueError(f"Invalid normalized selection rule {r.rule_id}")
        if r.action == "INCLUDE" and (
            r.authority != "SOURCE_SCENARIO_GUIDANCE" or not r.supported_families
            or r.function_scope == "UNRESOLVED"
            or r.project_scope != "CURRENT_PROJECT_IF_EVIDENCED"
            or any(x.reference_status not in {"EXTRACTED_REFERENCE", "CONFIRMED_IN_SOURCE"} for x in r.source_refs)
        ):
            raise ValueError("Include needs scoped positive source guidance")
        if r.action == "EXCLUDE":
            if (
                not r.runtime_exclusion_authority or r.approval != "APPROVED"
                or r.authority != "APPROVED_PROJECT_CONSTRAINT"
                or r.project_scope in {"REFERENCE_PROJECT_ONLY", "CURRENT_PROJECT_IF_EVIDENCED"}
                or not r.constraint_parameter
                or r.scenario_field not in {"operating_mode", "vehicle_state"}
                or not any(
                    by_doc[x.document_key].source_kind == "CURRENT_PROJECT_APPROVAL"
                    and x.reference_status == "CURRENT_PROJECT_APPROVED"
                    and x.project_scope == r.project_scope for x in r.source_refs
                )
            ):
                raise ValueError("Exclusion requires explicit current-project authority")
        elif r.runtime_exclusion_authority or r.constraint_parameter or r.scenario_field:
            raise ValueError("Non-exclusion rule cannot carry exclusion authority")
    principles = tuple(_record(SituationMethodPrinciple, x, refs=references) for x in raw["principles"])
    for p in principles:
        if (
            not p.rule_id or not p.summary or p.authority != "SOURCE_METHOD_PRINCIPLE"
            or p.trigger_semantics not in {"", "IMMEDIATE_VEHICLE_EFFECT", "SITUATION_TRIGGERED"}
            or p.exposure_domain not in {"", "Z", "F"}
            or bool(p.trigger_semantics) != bool(p.exposure_domain)
        ):
            raise ValueError("Invalid reference method principle")
    examples = tuple(_record(SituationAssessmentExample, x, refs=references) for x in raw["assessment_examples"])
    if any(not x.case_id or not x.summary or not x.reference_result or not x.allowed_use for x in examples):
        raise ValueError("Incomplete assessment reference example")
    parity = tuple(_record(ExposureParityCase, x, tuples=("operands",)) for x in raw["exposure_parity_cases"])
    levels = {"E0", "E1", "E2", "E3", "E4"}
    if any(
        not x.case_id or not x.operands or not set(x.operands) <= levels
        or x.expected not in levels or x.coupling not in {"", "independent", "coupled"}
        or x.source_document_key not in by_doc for x in parity
    ):
        raise ValueError("Invalid reference parity fixture")
    for records, key in ((principles, "rule_id"), (examples, "case_id"), (parity, "case_id")):
        if len({getattr(x, key) for x in records}) != len(records):
            raise ValueError("Duplicate reference record identity")
    return MalfunctionSituationSelectionMethod(
        schema_version=raw["schema_version"], selection_policy=raw["selection_policy"],
        source_asset=asset, source_hash=source_hash, documents=documents,
        families=families, rules=rules, principles=principles,
        assessment_examples=examples, exposure_parity_cases=parity,
    )
