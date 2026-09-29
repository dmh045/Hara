from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import re
from typing import Any


_MACHINE_REASON_TOKENS = (
    "MISSING_RELATIVE_SPEED", "MISSING_ROAD_USER_TYPE", "MISSING_COLLISION_TYPE",
    "EXPOSURE_DIMENSION_COVERAGE", "CONTROLLABILITY_UNKNOWN_BRANCH_POLICY_UNSPECIFIED",
    "METHOD_BRANCH_UNRESOLVED", "PENDING_METHOD_SEMANTICS", "PENDING_UPSTREAM_RISK_VALUE",
    "NO_ITEM_FACT", "NO_EXPLICIT_METHOD_ALIAS", "AMBIGUOUS_BINDING", "RANGE_CONTAINMENT",
    "NOT_EVALUATED",
)
_DEBUG_TOKENS = ("candidate_atom_ids", "rule_id", "binding_status")
_RATIONAL_FIELDS = (
    "severity_rationale", "exposure_rationale", "controllability_rationale",
    "asil_rationale", "ftti_rationale",
)
_REVIEWER_PROSE_FIELDS = (
    "operational_scenario", "scenario_detail", "hazardous_event", "potential_harm",
)
_RAW_DIMENSION = re.compile(
    r"\b(?:WHERE|ROAD|EGO_ACTION|EGO_X_ROAD|TRAFFIC_PATTERN|EGO_DYNAMICS|OBJECT)\b",
    re.IGNORECASE,
)
_ATOM_ID = re.compile(
    r"\b(?:FA|PH|PU|FB|VD)\d{3,}\b|\bCN_[A-Z0-9_]+\b",
    re.IGNORECASE,
)
_ALLOWED_ASCII = re.compile(
    r"(?<![A-Za-z])(?:AVP|HMI|EPB|TTC|ASIL|FTTI|ISO|VDA|FUSA|km|h|S|E|C)(?![A-Za-z])",
    re.IGNORECASE,
)
_STATUS_REMARK = re.compile(
    r"S (?:已计算|待审试算|未计算)；"
    r"E (?:已计算|待审试算|未计算)；"
    r"C (?:已计算|待审试算|未计算)；"
    r"ASIL (?:已计算|待审试算|未计算)"
)


def _lengths(rows: list[Any], field: str) -> dict[str, float | int]:
    values = [str(getattr(row, field, "") or "") for row in rows]
    lengths = [len(value) for value in values]
    return {
        "coverage": sum(bool(value) for value in values),
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "max_length": max(lengths, default=0),
        "unique_count": len(set(values)),
    }


def _contains_unapproved_language(value: str) -> bool:
    stripped = _ALLOWED_ASCII.sub("", value)
    return bool(re.search(r"[A-Za-zÄÖÜäöüß]{2,}", stripped))


def audit_content_presentation(
    view_model: Any,
    potential_harm_path: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit reviewer prose and the main-to-detail projection boundary."""
    rows = list(view_model.rows)
    detail_rows = list(getattr(view_model, "scenario_details", ()) or ())
    metrics = dict(getattr(view_model, "projection_metrics", {}) or {})
    values = [str(value or "") for row in rows for value in row.to_dict().values()]
    prose_values = [
        str(getattr(row, field, "") or "")
        for row in rows for field in _REVIEWER_PROSE_FIELDS
    ]
    detail_prose_values = [
        str(getattr(row, field, "") or "")
        for row in detail_rows
        for field in ("operational_scenario", "scenario_detail", "hazardous_event")
    ]
    rationale_values = [
        str(getattr(row, field, "") or "")
        for row in rows for field in _RATIONAL_FIELDS
    ]
    all_reviewer_prose = [*prose_values, *detail_prose_values, *rationale_values]
    language_mix = sum(_contains_unapproved_language(value) for value in all_reviewer_prose)
    raw_dimension_leaks = sum(bool(_RAW_DIMENSION.search(value)) for value in all_reviewer_prose)
    atom_id_leaks = sum(bool(_ATOM_ID.search(value)) for value in all_reviewer_prose)
    raw_json = sum(value.lstrip().startswith(("{", "[")) for value in values)
    machine_leaks = sum(any(token in value for token in _MACHINE_REASON_TOKENS) for value in values)
    debug_leaks = sum(any(token in value for token in _DEBUG_TOKENS) for value in values)
    critical_fields = (
        "hazardous_event", "potential_harm", "operational_scenario", "scenario_detail",
        "severity", "severity_rationale", "exposure", "exposure_rationale",
        "controllability", "controllability_rationale", "asil", "asil_rationale",
        "ftti", "ftti_rationale",
    )
    critical_blanks = sum(
        not str(getattr(row, field, "") or "")
        for row in rows for field in critical_fields
    )
    scenario_duplicates = len(rows) - len({
        (row.malfunction_id, row.operational_scenario, row.hazardous_event)
        for row in rows
    })
    hazardous_event_duplicates = len(rows) - len({
        (row.malfunction_id, row.hazardous_event) for row in rows
    })
    child_count = int(metrics.get("child_scenario_rows", len(detail_rows)))
    group_count = int(metrics.get("main_hara_groups", len(rows)))
    child_duplication_in_main = int(
        child_count > group_count
        and int(metrics.get("grouping_reduction", -1)) != child_count - group_count
    )
    broad_speed_rows = int(metrics.get("broad_0_20_speed_rows", 0))
    contextual_speed_rows = int(metrics.get("contextual_speed_rows", 0))
    uniform_broad_speed_collapse = int(
        child_count >= 3
        and contextual_speed_rows > 0
        and int(metrics.get("distinct_visible_speed_expressions", 0)) == 1
        and broad_speed_rows == child_count
    )
    main_descriptions = [str(row.operational_scenario or "") for row in rows]
    natural_description_collapse = int(
        len(main_descriptions) >= 6
        and len(set(main_descriptions)) < max(4, (len(main_descriptions) + 3) // 4)
    )
    detail_values = [str(row.scenario_detail or "") for row in detail_rows]
    detail_counts = Counter(detail_values)
    dominant_detail_ratio = (
        max(detail_counts.values(), default=0) / len(detail_values)
        if detail_values else 0.0
    )
    detail_template_collapse = int(
        len(detail_values) >= 6 and dominant_detail_ratio > 0.8
    )
    failures = (
        raw_json, machine_leaks, debug_leaks, critical_blanks, language_mix,
        raw_dimension_leaks, atom_id_leaks, child_duplication_in_main,
        uniform_broad_speed_collapse,
        natural_description_collapse, detail_template_collapse,
    )
    return {
        "language_mode": "ZH_ENGINEERING",
        "main_hara_rows": len(rows),
        "scenario_detail_rows": len(detail_rows),
        "language_mix_count": language_mix,
        "raw_dimension_syntax_leakage_count": raw_dimension_leaks,
        "raw_atom_id_leakage_count": atom_id_leaks,
        "raw_json_leakage_count": raw_json,
        "raw_machine_status_leakage_count": machine_leaks,
        "debug_term_leakage_count": debug_leaks,
        "critical_blank_count": critical_blanks,
        "scenario_description": _lengths(rows, "operational_scenario"),
        "scenario_detail": _lengths(rows, "scenario_detail"),
        **{field: _lengths(rows, field) for field in _RATIONAL_FIELDS},
        "remark_duplicate_information_count": sum(
            bool(str(row.remark or "").strip())
            and not _STATUS_REMARK.fullmatch(str(row.remark).strip())
            for row in rows
        ),
        "scenario_duplicate_count": scenario_duplicates,
        "scenario_duplicate_ratio": round(scenario_duplicates / len(rows), 4) if rows else 0,
        "hazardous_event_duplicate_count": hazardous_event_duplicates,
        "hazardous_event_duplicate_ratio": round(hazardous_event_duplicates / len(rows), 4) if rows else 0,
        "child_duplication_in_main_count": child_duplication_in_main,
        "uniform_broad_speed_collapse_count": uniform_broad_speed_collapse,
        "natural_scenario_description_collapse_count": natural_description_collapse,
        "dominant_scenario_detail_ratio": round(dominant_detail_ratio, 4),
        "scenario_detail_template_collapse_count": detail_template_collapse,
        "hazardous_event_and_potential_harm_separated": all(
            row.hazardous_event != row.potential_harm for row in rows
        ),
        "projection": metrics,
        "potential_harm": {
            "resolved": sum(not str(row.potential_harm).startswith(("Pending", "待S评定")) for row in rows),
            "pending": sum(str(row.potential_harm).startswith(("Pending", "待S评定")) for row in rows),
            "path_classification": str(potential_harm_path.get("classification", "")),
        },
        "quality_gate": "PASS" if not any(failures) else "FAIL",
    }


def audit_potential_harm_path(state: Any, view_model: Any) -> dict[str, Any]:
    """Trace committed Potential Harm evidence through its report projection."""
    scoring_event = next(
        (
            item for item in reversed(state.audit_trail)
            if item.get("event") == "structured_risk_scoring_completed"
        ),
        {},
    )
    calculations = list(scoring_event.get("risk_calculation_inputs", []))
    by_pair = {
        (str(item.get("malfunction_id", "")), str(item.get("scenario_id", ""))): item
        for item in calculations if isinstance(item, Mapping)
    }
    risks = list(state.risk_results)
    resolver_results = [
        by_pair.get((risk.malfunction_id, risk.scenario_id), {}).get("potential_harm")
        for risk in risks
    ]
    invoked = [item for item in resolver_results if isinstance(item, Mapping)]
    status_reasons = Counter(
        f"{item.get('status', 'UNKNOWN')}: {item.get('reason', '')}" for item in invoked
    )
    resolved = sum(bool(str(risk.potential_harm or "").strip()) for risk in risks)
    rows_by_scenario = {row.scenario_id: row for row in view_model.rows}
    details_by_scenario = {
        row.scenario_id: row for row in getattr(view_model, "scenario_details", ())
    }
    persisted_projection_gap = any(
        bool(str(risk.potential_harm or "").strip())
        and (
            (row := rows_by_scenario.get(risk.scenario_id)) is not None
            and row.potential_harm != risk.potential_harm
        )
        and risk.scenario_id not in details_by_scenario
        for risk in risks
    )
    resolver_produced_but_not_persisted = any(
        bool(str(item.get("potential_harm", "") or "").strip())
        and not str(risk.potential_harm or "").strip()
        for risk, item in zip(risks, resolver_results, strict=True)
        if isinstance(item, Mapping)
    )
    severity_pending = sum(
        str(getattr(risk.severity.status, "value", risk.severity.status)) != "FINALIZED"
        for risk in risks
    )
    if len(invoked) != len(risks):
        classification = "RUNTIME_NOT_EXECUTED"
    elif resolver_produced_but_not_persisted:
        classification = "RESULT_NOT_PERSISTED"
    elif persisted_projection_gap:
        classification = "PROJECTION_NOT_MAPPED"
    elif resolved == 0 and severity_pending == len(risks):
        classification = "UPSTREAM_RISK_NOT_READY"
    elif resolved == 0:
        classification = "METHOD_SEMANTICS_UNAVAILABLE"
    elif resolved == len(risks):
        classification = "RESOLVED"
    elif all(
        str(getattr(risk.severity.status, "value", risk.severity.status)) != "FINALIZED"
        for risk in risks if not str(risk.potential_harm or "").strip()
    ):
        classification = "PARTIAL_UPSTREAM_RISK_PENDING"
    else:
        classification = "RUNTIME_DEFECT"
    return {
        "resolver_class": "PotentialHarmResolver",
        "resolver_entrypoint": "PotentialHarmResolver.resolve(method, severity_result, scenario, registry)",
        "resolver_invocation_count": len(invoked),
        "eligible_he_count": len(risks),
        "resolved_count": resolved,
        "pending_count": len(risks) - resolved,
        "not_invoked_count": len(risks) - len(invoked),
        "runtime_storage_path": "score_risks -> RiskAssessment.potential_harm",
        "committed_state_field": "HARAState.risk_results[].potential_harm",
        "projection_source_field": "RiskAssessment.potential_harm",
        "view_model_field": "HARAReportRowView.potential_harm",
        "excel_field": f"04_HARA!J6:J{5 + len(view_model.rows)}",
        "per_status_reason_counts": dict(sorted(status_reasons.items())),
        "upstream_severity_status_counts": dict(sorted(Counter(
            str(getattr(risk.severity.status, "value", risk.severity.status)) for risk in risks
        ).items())),
        "upstream_severity_review_reason_counts": dict(sorted(Counter(
            str(getattr(risk.severity, "review_reason", "")) for risk in risks
        ).items())),
        "resolver_execution_evidence": "HARAState.audit_trail[].risk_calculation_inputs[].potential_harm",
        "classification": classification,
        "runtime_to_projection_wiring_gap": classification in {"RESULT_NOT_PERSISTED", "PROJECTION_NOT_MAPPED"},
    }


__all__ = ["audit_content_presentation", "audit_potential_harm_path"]
