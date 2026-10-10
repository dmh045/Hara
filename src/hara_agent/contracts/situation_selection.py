"""Source-governed situation guidance; distinct from physical FM options."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SituationReference:
    document_key: str
    worksheet: str
    rows: tuple[int, ...]
    columns: str
    rule_text_or_summary: str
    reference_status: str
    cross_version_mapping_status: str
    project_scope: str


@dataclass(frozen=True)
class SituationSourceDocument:
    document_key: str
    filename: str
    sha256: str
    source_kind: str


@dataclass(frozen=True)
class SituationFamily:
    family_id: str
    source_id: str
    source_version: str
    candidate_canonical_id: str
    canonical_version: str
    mapping_status: str
    mapping_source: str


@dataclass(frozen=True)
class SituationSelectionRule:
    rule_id: str
    functional_output: str
    source_failure_semantics: tuple[str, ...]
    canonical_failure_types: tuple[str, ...]
    function_scope: str
    supported_families: tuple[str, ...]
    authority: str
    action: str
    runtime_exclusion_authority: bool
    project_scope: str
    approval: str
    rationale: str
    source_refs: tuple[SituationReference, ...]
    constraint_parameter: str = ""
    scenario_field: str = ""


@dataclass(frozen=True)
class SituationMethodPrinciple:
    rule_id: str
    summary: str
    authority: str
    source_refs: tuple[SituationReference, ...]
    trigger_semantics: str = ""
    exposure_domain: str = ""


@dataclass(frozen=True)
class SituationAssessmentExample:
    case_id: str
    summary: str
    reference_result: str
    allowed_use: str
    source_refs: tuple[SituationReference, ...]


@dataclass(frozen=True)
class ExposureParityCase:
    case_id: str
    operands: tuple[str, ...]
    coupling: str
    expected: str
    source_document_key: str


@dataclass(frozen=True)
class MalfunctionSituationSelectionMethod:
    schema_version: str
    selection_policy: str
    source_asset: str
    source_hash: str
    documents: tuple[SituationSourceDocument, ...]
    families: tuple[SituationFamily, ...]
    rules: tuple[SituationSelectionRule, ...]
    principles: tuple[SituationMethodPrinciple, ...]
    assessment_examples: tuple[SituationAssessmentExample, ...]
    exposure_parity_cases: tuple[ExposureParityCase, ...]
