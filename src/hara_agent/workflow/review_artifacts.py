from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_ARTIFACT_FILES = {
    "function": "functions.jsonl",
    "guideword_assessment": "guideword_assessments.jsonl",
    "malfunction": "malfunctions.jsonl",
    "scenario_candidate": "scenario_candidates.jsonl",
    "scenario_feasibility": "scenario_feasibility.jsonl",
}
_IDENTITY_FIELDS = {
    "function": ("function_id",),
    "guideword_assessment": ("function_id", "guideword_id"),
    "malfunction": ("malfunction_id",),
    "scenario_candidate": ("scenario_id",),
    "scenario_feasibility": ("malfunction_id", "scenario_id"),
}
_SCENARIO_FACT_KEYS = (
    "operating_scenario", "operating_mode", "ego_speed_kph", "ego_speed_constraint",
    "weather_conditions", "road_surface_conditions", "vehicle_state",
    "object", "object_type", "object_position", "object_speed_kph",
    "road_user_type", "collision_type", "relative_distance_m", "WHERE", "ROAD", "EGO_ACTION",
    "EGO_X_ROAD", "TRAFFIC_PATTERN", "EGO_DYNAMICS", "OBJECT",
    "method_scenario_dimensions", "scenario_atom_ids",
)
_SCENARIO_CONTEXT_KEYS = (
    "ego_speed_kph", "speed_resolution", "speed_constraint",
    "dimension_bindings", "dimension_compatibility",
    "function_phase_binding", "risk_fact_binding", "analytical_scenario_instantiation",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _identity(kind: str, record: dict[str, Any]) -> tuple[str, ...] | None:
    fields = _IDENTITY_FIELDS[kind]
    values = tuple(str(record.get(field, "")).strip() for field in fields)
    return values if all(values) else None


def _compact_source_ref(value: Any) -> dict[str, Any]:
    """Keep a reviewable locator without duplicating governed source assets."""

    source = _jsonable(value)
    if not isinstance(source, dict):
        return {"reference": str(source)}
    excerpt = str(source.get("excerpt", ""))
    result = {
        key: source.get(key, "")
        for key in ("source_type", "source_id", "location")
        if source.get(key) not in (None, "")
    }
    if excerpt:
        result["excerpt_sha256"] = hashlib.sha256(
            excerpt.encode("utf-8")
        ).hexdigest()
        result["excerpt_chars"] = len(excerpt)
        result["excerpt_preview"] = excerpt[:512]
        result["excerpt_truncated"] = len(excerpt) > 512
    return result


def _compact_source_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_compact_source_ref(item) for item in value]


def _compact_review_value(value: Any) -> Any:
    """Recursively compact nested provenance source references."""

    value = _jsonable(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"sources", "source_refs"}:
                result["source_refs"] = _compact_source_refs(item)
            else:
                result[str(key)] = _compact_review_value(item)
        return result
    if isinstance(value, list):
        return [_compact_review_value(item) for item in value]
    return value


def _scenario_candidate_projection(
    value: Any, *, generated_for_malfunction_ids: Iterable[str],
) -> dict[str, Any] | None:
    """Build a compact read-only Scenario review projection.

    ScenarioCandidate remains canonical workflow data.  Only the review
    projection replaces repeated, full method-asset excerpts with locators,
    digests and bounded previews.
    """

    raw = _jsonable(value)
    if not isinstance(raw, dict):
        return None
    facts = raw.get("facts") if isinstance(raw.get("facts"), dict) else {}
    context = (
        raw.get("context_resolution")
        if isinstance(raw.get("context_resolution"), dict) else {}
    )
    provenance = (
        raw.get("fact_provenance")
        if isinstance(raw.get("fact_provenance"), dict) else {}
    )
    source_refs = _compact_source_refs(
        raw.get("source_refs", raw.get("sources", []))
    )
    compact_provenance: dict[str, Any] = {}
    for key in _SCENARIO_FACT_KEYS:
        item = provenance.get(key)
        if not isinstance(item, dict):
            continue
        compact_provenance[key] = {
            name: (
                _compact_source_refs(item[name])
                if name == "source_refs" else _compact_review_value(item[name])
            )
            for name in ("provenance", "approval", "resolution_status", "source_refs")
            if name in item
        }
        if "sources" in item and "source_refs" not in compact_provenance[key]:
            compact_provenance[key]["source_refs"] = _compact_source_refs(item["sources"])
    method_hashes = sorted({
        str(item.get("source_id", ""))
        for item in source_refs
        if item.get("source_type") == "method_contract" and item.get("source_id")
    })
    return {
        "projection_version": "scenario-review-v3",
        "scenario_id": raw.get("scenario_id", ""),
        "semantic_fingerprint": raw.get("semantic_fingerprint", ""),
        "operating_scenario": raw.get("operating_scenario", ""),
        "situational_description": raw.get("situational_description", ""),
        "situational_detailing": raw.get("situational_detailing", ""),
        "operating_mode": raw.get("operating_mode", facts.get("operating_mode", "")),
        "facts": {
            key: _compact_review_value(facts[key])
            for key in _SCENARIO_FACT_KEYS if key in facts
        },
        "context_resolution": {
            key: _compact_review_value(context[key])
            for key in _SCENARIO_CONTEXT_KEYS if key in context
        },
        "fact_provenance": compact_provenance,
        "source_refs": source_refs,
        "method_source_hash": method_hashes[0] if len(method_hashes) == 1 else "",
        "method_source_hashes": method_hashes if len(method_hashes) > 1 else [],
        "rule_version": raw.get("rule_version", ""),
        "scenario_contract_version": raw.get("scenario_contract_version", ""),
        "review_reason": raw.get("review_reason", ""),
        "source_scenario_id": raw.get("source_scenario_id", ""),
        "atomic_variant": raw.get("atomic_variant", ""),
        "analysis_instance": _compact_review_value(raw.get("analysis_instance", {})),
        "status": raw.get("status", ""),
        "generated_for_malfunction_ids": list(generated_for_malfunction_ids),
    }


def _expected_candidate_count(
    candidates: list[dict[str, Any]], malfunction_id: str,
) -> int:
    """Use isolated analytical instances when a malfunction has them."""
    scoped = [
        item for item in candidates
        if malfunction_id in item.get("generated_for_malfunction_ids", [])
    ]
    analytical = [
        item for item in scoped
        if isinstance(item.get("analysis_instance"), dict)
        and item["analysis_instance"].get("origin") == "SCENARIO_DEFINED"
    ]
    return len(analytical or scoped or candidates)


class ReviewArtifactWriter:
    """Append-only, idempotent read-model projection for an HARA run."""

    def __init__(self, run_id: str, root: str | Path = "runtime/review"):
        if not _SAFE_ID.fullmatch(str(run_id)):
            raise ValueError("review artifact run_id must contain only letters, digits, '-' or '_'")
        self.run_id = str(run_id)
        self.warnings: list[str] = []
        self._disabled = False
        self.directory = Path(root).expanduser().resolve() / self.run_id
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._disabled = True
            self._warn(f"could not create review artifact directory {self.directory}: {exc}")
        self._lock = threading.RLock()
        self._seen: dict[str, set[tuple[str, ...]]] = {
            kind: set() for kind in _ARTIFACT_FILES
        }
        if self._disabled:
            return
        self._load_seen()
        if not (self.directory / "summary.json").is_file():
            self._write_summary_payload({
                "run_id": self.run_id,
                "status": "IN_PROGRESS",
                "last_updated_at": _now(),
            })

    def _load_seen(self) -> None:
        for kind, filename in _ARTIFACT_FILES.items():
            path = self.directory / filename
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            self._warn(f"ignored malformed {filename}:{line_number}: {exc}")
                            continue
                        if isinstance(value, dict):
                            identity = _identity(kind, value)
                            if identity is not None:
                                self._seen[kind].add(identity)
            except OSError as exc:
                self._warn(f"could not inspect {path}: {exc}")

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[HARA] review artifact warning: {message}", flush=True)

    def _append(self, kind: str, value: Any) -> bool:
        if self._disabled:
            return False
        payload = _jsonable(value)
        if not isinstance(payload, dict):
            self._warn(f"{kind} artifact is not a JSON object")
            return False
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("recorded_at", _now())
        if "source_refs" not in payload and "sources" in payload:
            payload["source_refs"] = payload["sources"]
        if kind == "guideword_assessment":
            payload.setdefault("guideword_id", payload.get("guideword", ""))
        elif kind == "malfunction":
            payload.setdefault("guideword_id", payload.get("guideword", ""))
        identity = _identity(kind, payload)
        if identity is None:
            self._warn(f"{kind} artifact has no complete stable identity")
            return False
        with self._lock:
            if identity in self._seen[kind]:
                return False
            try:
                line = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), default=str,
                )
                with (self.directory / _ARTIFACT_FILES[kind]).open(
                    "a", encoding="utf-8", newline="\n",
                ) as stream:
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except (OSError, TypeError, ValueError) as exc:
                self._warn(f"could not append {kind} artifact: {exc}")
                return False
            self._seen[kind].add(identity)
            return True

    def record_function(self, function: Any) -> bool:
        return self._append("function", function)

    def record_guideword_assessment(self, assessment: Any) -> bool:
        return self._append("guideword_assessment", assessment)

    def record_malfunction(self, malfunction: Any) -> bool:
        return self._append("malfunction", malfunction)

    def record_scenario_candidate(
        self, scenario: Any, *, generated_for_malfunction_ids: Iterable[str] = (),
    ) -> bool:
        payload = _scenario_candidate_projection(
            scenario,
            generated_for_malfunction_ids=generated_for_malfunction_ids,
        )
        return self._append("scenario_candidate", payload)

    def write_scenario_binding_gaps(
        self, coverage: dict[str, Any], gaps: list[dict[str, Any]],
    ) -> None:
        """Write the deterministic Scenario binding gap review projection."""
        if self._disabled:
            return
        payload = {
            "artifact_version": "scenario-binding-gaps-v1",
            "run_id": self.run_id,
            "scenario_binding_coverage": _compact_review_value(coverage),
            "gaps": _compact_review_value(gaps),
            "recorded_at": _now(),
        }
        with self._lock:
            self._write_json_payload("scenario_binding_gaps.json", payload)

    def write_scenario_alias_proposals(self, payload: dict[str, Any]) -> None:
        """Write review-only alias proposals; never canonical Scenario truth."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("scenario alias proposal artifact is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("scenario_alias_proposals.json", value)

    def write_malfunction_situation_selection(self, payload: dict[str, Any]) -> None:
        """Persist preselection evidence and stage counts, including unresolved pass-through."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        value.update({"artifact_version": "malfunction-situation-selection-v1", "run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("malfunction_situation_selection.json", value)

    def write_scenario_coverage_proposals(self, payload: dict[str, Any]) -> None:
        """Write review-only Function coverage proposals; never runtime truth."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("scenario coverage proposal artifact is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("scenario_coverage_proposals.json", value)

    def write_confirmed_yaml_utilization(self, payload: dict[str, Any]) -> None:
        """Write offline confirmed-YAML utilization; never Scenario truth."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("confirmed YAML utilization artifact is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("confirmed_yaml_utilization.json", value)

    def write_fm_selector_semantic_audit(self, payload: dict[str, Any]) -> None:
        """Write a deterministic, offline FM selector review projection."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("FM selector semantic audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("fm_selector_semantic_audit.json", value)

    def write_fm_template_ambiguity_audit(self, payload: dict[str, Any]) -> None:
        """Write the deterministic FM template ambiguity trace."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("FM template ambiguity audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("fm_template_ambiguity_audit.json", value)

    def write_risk_execution_trace(self, payload: dict[str, Any]) -> None:
        """Write the read-only Scenario-to-risk execution projection."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("risk execution trace is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("risk_execution_trace.json", value)

    def write_exposure_binding_audit(self, payload: dict[str, Any]) -> None:
        """Write a read-only governed-atom Exposure input audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("exposure binding audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("exposure_binding_audit.json", value)

    def write_exposure_input_audit(self, payload: dict[str, Any]) -> None:
        """Write the committed-run Exposure input-completeness audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("exposure input audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("exposure_input_audit.json", value)

    def write_exposure_dimension_coverage_audit(self, payload: dict[str, Any]) -> None:
        """Write a read-only audit of formal Exposure dimension authority."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("exposure dimension coverage audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("exposure_dimension_coverage_audit.json", value)

    def write_severity_delta_v_semantic_audit(self, payload: dict[str, Any]) -> None:
        """Write a read-only Severity semantic/input-authority audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("severity DELTA_V semantic audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("severity_delta_v_semantic_audit.json", value)

    def write_hazardous_event_risk_context_audit(self, payload: dict[str, Any]) -> None:
        """Write the read-only Hazardous Event to Risk-input context audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("Hazardous Event risk context audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("hazardous_event_risk_context_audit.json", value)

    def write_controllability_branch_policy_audit(self, payload: dict[str, Any]) -> None:
        """Write the read-only selected-controllability branch-policy audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("Controllability branch audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("controllability_branch_policy_audit.json", value)

    def write_method_contract_parity_audit(self, payload: dict[str, Any]) -> None:
        """Write the read-only Template/YAML evaluator parity audit."""
        if self._disabled:
            return
        value = _compact_review_value(payload)
        if not isinstance(value, dict):
            self._warn("MethodContract parity audit is not a JSON object")
            return
        value.update({"run_id": self.run_id, "recorded_at": _now()})
        with self._lock:
            self._write_json_payload("method_contract_parity_audit.json", value)
            self._write_json_payload(
                "template_method_contract_audit.json",
                value.get("template_contract_inventory", {}),
            )
            self._write_json_payload(
                "yaml_method_contract_audit.json",
                value.get("yaml_contract_inventory", {}),
            )

    def record_scenario_feasibility(
        self, assessment: Any, *, function_id: str = "", guideword: str = "",
        audit: dict[str, Any] | None = None,
    ) -> bool:
        # ScenarioFeasibilityAssessment owns its review projection.  In
        # particular, final_retain is a derived eligibility result, not a
        # dataclass field that may be defaulted by this writer.
        to_dict = getattr(assessment, "to_dict", None)
        payload = _jsonable(to_dict() if callable(to_dict) else assessment)
        if isinstance(payload, dict):
            if function_id:
                payload["function_id"] = function_id
            if guideword:
                payload["guideword"] = guideword
            payload["final_retain"] = bool(
                payload.get("final_retain", payload.get("retained", False))
            )
            causal_assessment = payload.get("causal_assessment")
            causal_chain = (
                causal_assessment.get("causal_chain")
                if isinstance(causal_assessment, dict) else None
            )
            malfunction_id = str(payload.get("malfunction_id", ""))
            scenario_id = str(payload.get("scenario_id", ""))
            if (
                isinstance(causal_chain, list)
                and causal_chain
                and malfunction_id
                and scenario_id
            ):
                payload["hazardous_event_id"] = (
                    f"HE::{malfunction_id}::{scenario_id}::{causal_chain[-1]}"
                )
            if isinstance(audit, dict):
                selection = next(
                    (
                        item for item in audit.get("evidence_selection_audit", [])
                        if isinstance(item, dict)
                        and str(item.get("scenario_id", "")) == scenario_id
                    ),
                    None,
                )
                if isinstance(selection, dict):
                    for key in (
                        "mandatory_evidence_refs", "selected_context_evidence_refs",
                        "total_prompt_evidence_count", "candidate_evidence_count",
                        "selected_evidence_count", "dropped_evidence_count",
                        "evidence_selection_reason", "m_to_b_anchor_available",
                        "m_to_b_anchor_refs",
                    ):
                        if key in selection:
                            payload[key] = _jsonable(selection[key])
                salvage = next(
                    (
                        item for item in audit.get("item_salvage_audit", [])
                        if isinstance(item, dict)
                        and str(item.get("scenario_id", "")) == scenario_id
                    ),
                    None,
                )
                if isinstance(salvage, dict):
                    for key in (
                        "outcome", "error_stage", "repair_failed", "error_code",
                        "breakpoint_reason", "provider_request_id",
                    ):
                        if key in salvage:
                            payload[key] = _jsonable(salvage[key])
        return self._append("scenario_feasibility", payload)

    def write_summary(self, state: Any, *, status: str | None = None, error: str = "") -> None:
        if self._disabled:
            return
        with self._lock:
            self._write_summary_locked(state, status=status, error=error)

    def _write_summary_locked(
        self, state: Any, *, status: str | None = None, error: str = "",
    ) -> None:
        try:
            records = self.read_all()
            functions = records["function"]
            guidewords = records["guideword_assessment"]
            malfunctions = records["malfunction"]
            candidates = records["scenario_candidate"]
            feasibility = records["scenario_feasibility"]
            per_malfunction: dict[str, dict[str, Any]] = {}
            for item in malfunctions:
                malfunction_id = str(item.get("malfunction_id", ""))
                if malfunction_id:
                    per_malfunction.setdefault(
                        malfunction_id,
                        {
                            "expected_scenario_count": _expected_candidate_count(candidates, malfunction_id),
                            "total": 0, "feasible": 0, "infeasible": 0,
                            "breakpoints": {},
                        },
                    )
            for item in feasibility:
                malfunction_id = str(item.get("malfunction_id", ""))
                if not malfunction_id:
                    continue
                summary = per_malfunction.setdefault(
                    malfunction_id,
                    {
                        "expected_scenario_count": _expected_candidate_count(candidates, malfunction_id),
                        "total": 0, "feasible": 0, "infeasible": 0,
                        "breakpoints": {},
                    },
                )
                summary["total"] += 1
                if bool(item.get("final_retain", item.get("retained", False))):
                    summary["feasible"] += 1
                else:
                    summary["infeasible"] += 1
                breakpoint = str(item.get("breakpoint", "")).strip()
                if breakpoint:
                    summary["breakpoints"][breakpoint] = (
                        summary["breakpoints"].get(breakpoint, 0) + 1
                    )
            for item in per_malfunction.values():
                expected = int(item.get("expected_scenario_count", len(candidates)))
                observed = int(item.get("total", 0))
                item["assessment_status"] = (
                    "NOT_STARTED" if observed == 0 else
                    "COMPLETED" if expected > 0 and observed >= expected else
                    "PARTIAL"
                )
            payload = {
                "run_id": self.run_id,
                "stage": getattr(getattr(state, "stage", None), "value", ""),
                "status": status or (
                    "COMPLETED" if getattr(getattr(state, "stage", None), "value", "") == "complete"
                    else "IN_PROGRESS"
                ),
                "function_count": len(functions),
                "guideword_assessment_count": len(guidewords),
                "guideword_applicable_count": sum(bool(item.get("applicable")) for item in guidewords),
                "guideword_filtered_count": sum(not bool(item.get("applicable")) for item in guidewords),
                "malfunction_count": len(malfunctions),
                "scenario_candidate_count": len(candidates),
                "scenario_feasibility_count": len(feasibility),
                "scenario_feasible_count": sum(
                    bool(item.get("final_retain", item.get("retained", False))) for item in feasibility
                ),
                "scenario_infeasible_count": sum(
                    not bool(item.get("final_retain", item.get("retained", False))) for item in feasibility
                ),
                "scenario_assessment_completed_malfunction_count": sum(
                    item["assessment_status"] == "COMPLETED"
                    for item in per_malfunction.values()
                ),
                "scenario_assessment_partial_malfunction_count": sum(
                    item["assessment_status"] == "PARTIAL"
                    for item in per_malfunction.values()
                ),
                "scenario_assessment_not_started_malfunction_count": sum(
                    item["assessment_status"] == "NOT_STARTED"
                    for item in per_malfunction.values()
                ),
                "per_malfunction_summary": per_malfunction,
                "last_updated_at": _now(),
            }
            risk_trace = self._read_json_payload("risk_execution_trace.json")
            if isinstance(risk_trace, dict):
                payload["risk_execution_summary"] = {
                    "risk_stage_status": risk_trace.get("risk_stage_status", "NOT_REACHED"),
                    **dict(risk_trace.get("scenario_eligibility_summary", {})),
                    "rating_counts": self._risk_rating_counts(risk_trace),
                }
            if error:
                payload["error"] = error[:2000]
            self._write_summary_payload(payload)
        except Exception as exc:
            self._warn(f"could not write review summary: {exc}")

    def mark_failed(self, state: Any, error: Exception) -> None:
        self.write_summary(state, status="FAILED", error=f"{type(error).__name__}: {error}")

    def _write_summary_payload(self, payload: dict[str, Any]) -> None:
        self._write_json_payload("summary.json", payload)

    def _read_json_payload(self, filename: str) -> dict[str, Any] | None:
        path = self.directory / filename
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _risk_rating_counts(trace: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in ("severity", "exposure", "controllability", "asil"):
            values = [
                item.get(field, {}) for item in trace.get("assessments", [])
                if isinstance(item, dict) and isinstance(item.get(field), dict)
            ]
            result[field] = {
                "finalized": sum(value.get("status") == "FINALIZED" for value in values),
                "pending": sum(str(value.get("status", "")).startswith("PENDING") for value in values),
            }
        return result

    def _write_json_payload(self, filename: str, payload: dict[str, Any]) -> None:
        if self._disabled:
            return
        target = self.directory / filename
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(self.directory), text=True,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def read_all(self) -> dict[str, list[dict[str, Any]]]:
        return ReviewArtifactReader(self.run_id, self.directory.parent).read_all()


class ReviewArtifactReader:
    """Read-only reader that tolerates truncated/corrupt individual JSONL lines."""

    def __init__(self, run_id: str, root: str | Path = "runtime/review"):
        if not _SAFE_ID.fullmatch(str(run_id)):
            raise ValueError("review artifact run_id must contain only letters, digits, '-' or '_'")
        self.run_id = str(run_id)
        self.directory = Path(root).expanduser().resolve() / self.run_id
        self.warnings: list[str] = []

    def read_all(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _ARTIFACT_FILES}
        seen: dict[str, set[tuple[str, ...]]] = {
            kind: set() for kind in _ARTIFACT_FILES
        }
        for kind, filename in _ARTIFACT_FILES.items():
            path = self.directory / filename
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            self.warnings.append(f"ignored malformed {filename}:{line_number}: {exc}")
                            continue
                        if isinstance(value, dict):
                            identity = _identity(kind, value)
                            if identity is not None:
                                if identity in seen[kind]:
                                    self.warnings.append(
                                        f"ignored duplicate {filename}:{line_number} identity={identity}"
                                    )
                                    continue
                                seen[kind].add(identity)
                            result[kind].append(value)
            except OSError as exc:
                self.warnings.append(f"could not read {path}: {exc}")
        return result

    def summary(self) -> dict[str, Any]:
        path = self.directory / "summary.json"
        if not path.is_file():
            return {"run_id": self.run_id, "status": "NOT_FOUND"}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.warnings.append(f"could not read summary.json: {exc}")
            return {"run_id": self.run_id, "status": "CORRUPT"}
        return value if isinstance(value, dict) else {"run_id": self.run_id, "status": "CORRUPT"}


def _record_source(record: dict[str, Any]) -> str:
    sources = record.get("source_refs", record.get("sources", []))
    if not isinstance(sources, list) or not sources:
        return ""
    first = sources[0]
    if not isinstance(first, dict):
        return str(first)
    return ":".join(
        str(first.get(key, "")) for key in ("source_type", "source_id", "location")
        if first.get(key)
    )


def _assessment_feasible(record: dict[str, Any]) -> bool:
    return bool(record.get("final_retain", record.get("retained", False)))


def _assessment_status(*, expected: int, observed: int) -> str:
    if observed == 0:
        return "NOT_STARTED"
    if expected > 0 and observed >= expected:
        return "COMPLETED"
    return "PARTIAL"


def _summary_from_records(
    summary: dict[str, Any], records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Use JSONL as the read-only truth source if a persisted summary is stale."""

    result = dict(summary)
    candidates = records["scenario_candidate"]
    feasibility = records["scenario_feasibility"]
    per_malfunction: dict[str, dict[str, Any]] = {}
    for malfunction in records["malfunction"]:
        malfunction_id = str(malfunction.get("malfunction_id", ""))
        if malfunction_id:
            per_malfunction[malfunction_id] = {
                "expected_scenario_count": _expected_candidate_count(candidates, malfunction_id),
                "total": 0, "feasible": 0, "infeasible": 0, "breakpoints": {},
            }
    for assessment in feasibility:
        malfunction_id = str(assessment.get("malfunction_id", ""))
        if not malfunction_id:
            continue
        item = per_malfunction.setdefault(
            malfunction_id,
            {
                "expected_scenario_count": _expected_candidate_count(candidates, malfunction_id),
                "total": 0, "feasible": 0, "infeasible": 0, "breakpoints": {},
            },
        )
        item["total"] += 1
        if _assessment_feasible(assessment):
            item["feasible"] += 1
        else:
            item["infeasible"] += 1
        breakpoint = str(assessment.get("breakpoint", "")).strip()
        if breakpoint:
            item["breakpoints"][breakpoint] = item["breakpoints"].get(breakpoint, 0) + 1
    for item in per_malfunction.values():
        item["assessment_status"] = _assessment_status(
            expected=int(item["expected_scenario_count"]), observed=int(item["total"]),
        )
    result.update({
        "function_count": len(records["function"]),
        "guideword_assessment_count": len(records["guideword_assessment"]),
        "guideword_applicable_count": sum(
            bool(item.get("applicable")) for item in records["guideword_assessment"]
        ),
        "guideword_filtered_count": sum(
            not bool(item.get("applicable")) for item in records["guideword_assessment"]
        ),
        "malfunction_count": len(records["malfunction"]),
        "scenario_candidate_count": len(candidates),
        "scenario_feasibility_count": len(feasibility),
        "scenario_feasible_count": sum(_assessment_feasible(item) for item in feasibility),
        "scenario_infeasible_count": sum(not _assessment_feasible(item) for item in feasibility),
        "scenario_assessment_completed_malfunction_count": sum(
            item["assessment_status"] == "COMPLETED" for item in per_malfunction.values()
        ),
        "scenario_assessment_partial_malfunction_count": sum(
            item["assessment_status"] == "PARTIAL" for item in per_malfunction.values()
        ),
        "scenario_assessment_not_started_malfunction_count": sum(
            item["assessment_status"] == "NOT_STARTED" for item in per_malfunction.values()
        ),
        "per_malfunction_summary": per_malfunction,
    })
    return result


def _sample(records: list[dict[str, Any]], *, all_records: bool, limit: int) -> list[dict[str, Any]]:
    return records if all_records else records[:max(0, limit)]


def render_review(
    reader: ReviewArtifactReader,
    *,
    function_id: str = "",
    malfunction_id: str = "",
    scenario_id: str = "",
    all_records: bool = False,
    limit: int = 3,
    feasible: bool = False,
    infeasible: bool = False,
) -> str:
    """Render a read-only human review view without loading workflow state."""

    records = reader.read_all()
    summary = _summary_from_records(reader.summary(), records)
    if scenario_id:
        return _render_scenario(records, scenario_id)
    if malfunction_id:
        return _render_malfunction(
            records, malfunction_id, all_records=all_records, limit=limit,
            feasible=feasible, infeasible=infeasible,
        )
    if function_id:
        return _render_function(records, function_id, all_records=all_records, limit=limit)
    return _render_summary(summary, records)


def _render_summary(summary: dict[str, Any], records: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "RUN",
        "=" * 50,
        f"run_id: {summary.get('run_id', '')}",
        f"status: {summary.get('status', 'UNKNOWN')}",
        "",
        f"Functions: {summary.get('function_count', len(records['function']))}",
        f"Guideword assessments: {summary.get('guideword_assessment_count', len(records['guideword_assessment']))}",
        f"Malfunctions: {summary.get('malfunction_count', len(records['malfunction']))}",
        f"Scenario candidates: {summary.get('scenario_candidate_count', len(records['scenario_candidate']))}",
        f"Scenario assessments: {summary.get('scenario_feasibility_count', len(records['scenario_feasibility']))}",
        "",
        f"Feasible: {summary.get('scenario_feasible_count', 0)}",
        f"Infeasible: {summary.get('scenario_infeasible_count', 0)}",
    ]
    per_malfunction = summary.get("per_malfunction_summary", {})
    if per_malfunction:
        lines.extend([
            "", "Generated malfunctions: " + str(summary.get(
                "malfunction_count", len(per_malfunction),
            )),
            "Scenario assessment:",
            "  Completed: " + str(summary.get(
                "scenario_assessment_completed_malfunction_count", 0,
            )),
            "  Partial/in-progress: " + str(summary.get(
                "scenario_assessment_partial_malfunction_count", 0,
            )),
            "  Not started: " + str(summary.get(
                "scenario_assessment_not_started_malfunction_count", 0,
            )),
            "", "Assessment coverage by malfunction:",
        ])
        for malfunction_id, item in sorted(per_malfunction.items()):
            lines.append(
                f"  {malfunction_id}: {item.get('assessment_status', 'UNKNOWN')} "
                f"({item.get('total', 0)}/{item.get('expected_scenario_count', 0)} assessed; "
                f"{item.get('feasible', 0)} feasible)"
            )
        breakpoint_counts: dict[str, int] = {}
        for item in per_malfunction.values():
            for breakpoint, count in item.get("breakpoints", {}).items():
                breakpoint_counts[breakpoint] = breakpoint_counts.get(breakpoint, 0) + int(count)
        if breakpoint_counts:
            lines.extend(["", "Top breakpoints:"])
            lines.extend(
                f"  {name}: {count}"
                for name, count in sorted(breakpoint_counts.items(), key=lambda item: (-item[1], item[0]))
            )
    return "\n".join(lines)


def _render_function(
    records: dict[str, list[dict[str, Any]]], function_id: str,
    *, all_records: bool, limit: int,
) -> str:
    function = next((item for item in records["function"] if item.get("function_id") == function_id), None)
    if function is None:
        return f"FUNCTION {function_id}\n\nNot found"
    guidewords = [item for item in records["guideword_assessment"] if item.get("function_id") == function_id]
    malfunctions = [item for item in records["malfunction"] if item.get("function_id") == function_id]
    lines = [
        f"FUNCTION {function_id}", "=" * 50,
        f"Name: {function.get('name', '')}",
        f"Description: {function.get('description', '')}",
        f"Output: {function.get('output', '')}",
        f"Preconditions: {function.get('preconditions', [])}",
        f"Triggers: {function.get('triggers', [])}",
        f"ODD Constraints: {function.get('odd_constraints', [])}",
        f"Source: {_record_source(function)}",
        "", "GUIDEWORD ASSESSMENTS", "-" * 50,
        "Guideword | Applicable | Disposition | Rationale",
    ]
    for item in guidewords:
        lines.append(
            f"{item.get('guideword', item.get('guideword_id', ''))} | "
            f"{item.get('applicable', '')} | {item.get('disposition', '')} | {item.get('rationale', '')}"
        )
    lines.extend(["", "MALFUNCTIONS", "-" * 50])
    for malfunction in _sample(malfunctions, all_records=all_records, limit=limit):
        lines.extend([
            str(malfunction.get("malfunction_id", "")),
            f"  Guideword: {malfunction.get('guideword', '')}",
            f"  Description: {malfunction.get('description', '')}",
            f"  Functional Effect: {malfunction.get('functional_effect', '')}",
            f"  Vehicle-level Hazard: {malfunction.get('vehicle_level_hazard', '')}",
        ])
    return "\n".join(lines)


def _render_malfunction(
    records: dict[str, list[dict[str, Any]]], malfunction_id: str,
    *, all_records: bool, limit: int, feasible: bool, infeasible: bool,
) -> str:
    malfunction = next((item for item in records["malfunction"] if item.get("malfunction_id") == malfunction_id), None)
    all_assessments = [
        item for item in records["scenario_feasibility"]
        if item.get("malfunction_id") == malfunction_id
    ]
    expected_count = _expected_candidate_count(records["scenario_candidate"], malfunction_id)
    assessment_status = _assessment_status(
        expected=expected_count, observed=len(all_assessments),
    )
    assessments = list(all_assessments)
    if feasible:
        assessments = [item for item in assessments if _assessment_feasible(item)]
    if infeasible:
        assessments = [item for item in assessments if not _assessment_feasible(item)]
    feasible_count = sum(_assessment_feasible(item) for item in assessments)
    lines = [
        f"MALFUNCTION {malfunction_id}", "=" * 50,
        f"Function: {malfunction.get('function_id', '') if malfunction else ''}",
        f"Guideword: {malfunction.get('guideword', '') if malfunction else ''}",
        f"Description: {malfunction.get('description', '') if malfunction else ''}",
        f"Functional Effect: {malfunction.get('functional_effect', '') if malfunction else ''}",
        f"Vehicle-level Hazard: {malfunction.get('vehicle_level_hazard', '') if malfunction else ''}",
        "", "SCENARIO SUMMARY", "-" * 50,
        f"Assessment status: {assessment_status}",
        f"Coverage: {len(all_assessments)}/{expected_count}",
        f"Total shown: {len(assessments)}",
        f"Feasible: {feasible_count}",
        f"Infeasible: {len(assessments) - feasible_count}",
    ]
    breakpoint_counts: dict[str, int] = {}
    for item in assessments:
        breakpoint = str(item.get("breakpoint", ""))
        if breakpoint:
            breakpoint_counts[breakpoint] = breakpoint_counts.get(breakpoint, 0) + 1
    lines.append("Breakpoints: " + ", ".join(
        f"{name}={count}" for name, count in sorted(breakpoint_counts.items())
    ))
    shown = _sample(assessments, all_records=all_records, limit=limit)
    lines.extend(["", "SAMPLE SCENARIOS", "-" * 50])
    for item in shown:
        candidate = next(
            (value for value in records["scenario_candidate"] if value.get("scenario_id") == item.get("scenario_id")),
            {},
        )
        lines.extend(_scenario_lines(candidate, item))
    return "\n".join(lines)


def _scenario_lines(candidate: dict[str, Any], assessment: dict[str, Any]) -> list[str]:
    facts = candidate.get("facts", {}) if isinstance(candidate.get("facts"), dict) else {}
    chain = assessment.get("causal_chain", {})
    lines = [
        f"{assessment.get('scenario_id', candidate.get('scenario_id', ''))}",
        f"Operating Scenario: {candidate.get('operating_scenario', '')}",
        f"Mode: {candidate.get('operating_mode', facts.get('operating_mode', ''))}",
        f"Speed: {facts.get('ego_speed_kph', '')}",
        f"WHERE: {facts.get('WHERE', facts.get('where', ''))}",
        f"ROAD: {facts.get('ROAD', facts.get('road', ''))}",
        f"EGO_ACTION: {facts.get('EGO_ACTION', facts.get('ego_action', ''))}",
        f"OBJECT: {facts.get('OBJECT', facts.get('object', facts.get('object_type', '')))}",
    ]
    for hop in ("m_to_b", "b_to_i", "i_to_h"):
        value = chain.get(hop, {}) if isinstance(chain, dict) else {}
        lines.extend([
            f"{hop}:",
            f"  claim: {value.get('claim', '') if isinstance(value, dict) else ''}",
            f"  basis_type: {value.get('basis_type', '') if isinstance(value, dict) else ''}",
            f"  evidence_refs: {value.get('evidence_refs', []) if isinstance(value, dict) else []}",
        ])
    lines.extend([
        f"Breakpoint: {assessment.get('breakpoint', '')}",
        f"Reason: {assessment.get('breakpoint_reason', assessment.get('rationale', ''))}",
        "",
    ])
    return lines


def _render_scenario(records: dict[str, list[dict[str, Any]]], scenario_id: str) -> str:
    candidate = next((item for item in records["scenario_candidate"] if item.get("scenario_id") == scenario_id), {})
    assessments = [item for item in records["scenario_feasibility"] if item.get("scenario_id") == scenario_id]
    lines = ["SCENARIO CANDIDATE", "=" * 50]
    lines.extend([
        f"scenario_id: {scenario_id}",
        f"semantic_fingerprint: {candidate.get('semantic_fingerprint', '')}",
        f"Operating Scenario: {candidate.get('operating_scenario', '')}",
        f"Description: {candidate.get('situational_description', '')}",
        f"Detail: {candidate.get('situational_detailing', '')}",
        f"Facts: {candidate.get('facts', {})}",
        "", "MALFUNCTION ASSESSMENTS", "-" * 50,
    ])
    for assessment in assessments:
        lines.extend(_scenario_lines(candidate, assessment))
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["ReviewArtifactReader", "ReviewArtifactWriter"]
