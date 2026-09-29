"""Read-only source-coverage audit for ``HazardousEventRiskContext``.

This audit deliberately projects already compiled method semantics and persisted
run evidence.  It neither supplies RiskContext values nor reads raw method YAML
while the risk runtime is executing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from hara_agent.contracts import MethodContract, UnknownOverridePolicy

from .scenario_physics import TTC_CLOSING_FORMULA_IDENTITY


class RiskContextSourceCoverageAuditService:
    """Classify source availability before proposing any RiskContext change."""

    _FIELDS = (
        "road_user_type", "collision_type", "ego_speed_kph", "object_speed_kph",
        "relative_speed_kph", "impact_speed_kph", "relative_distance_m", "ttc_s",
        "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible", "direct_control_available",
        "vehicle_stability", "emergency_braking_available", "function_type",
        "has_remote_app",
    )
    _FACT_PARAMETER = {
        "road_user_type": "ROAD_USER_TYPE",
        "collision_type": "COLLISION_TYPE",
        "ego_speed_kph": "EGO_SPEED",
        "object_speed_kph": "OBJECT_SPEED",
        "relative_speed_kph": "RELATIVE_SPEED",
        "impact_speed_kph": "IMPACT_SPEED",
        "relative_distance_m": "RELATIVE_DISTANCE",
        "ttc_s": "TTC",
        "driver_in_vehicle": "DRIVER_IN_VEHICLE",
        "remote_intervention_available": "REMOTE_INTERVENTION_AVAILABLE",
        "other_road_user_avoidance_possible": "OTHER_ROAD_USER_AVOIDANCE_POSSIBLE",
        "direct_control_available": "DIRECT_CONTROL_AVAILABLE",
        "vehicle_stability": "VEHICLE_STABILITY",
        "emergency_braking_available": "EMERGENCY_BRAKING_AVAILABLE",
        "function_type": "FUNCTION_TYPE",
        "has_remote_app": "HAS_REMOTE_APP",
    }
    _DOCUMENT_TERMS = {
        "road_user_type": ("行人", "障碍物", "车辆"),
        "collision_type": ("碰撞",),
        "ego_speed_kph": ("车速", "km/h"),
        "object_speed_kph": ("障碍物", "前车"),
        "relative_speed_kph": ("车速", "km/h"),
        "impact_speed_kph": ("车速", "km/h"),
        "relative_distance_m": ("距离", "间距"),
        "ttc_s": ("距离", "车速"),
        "driver_in_vehicle": ("驾驶员", "人驾"),
        "remote_intervention_available": ("远程",),
        "other_road_user_avoidance_possible": ("避让",),
        "direct_control_available": ("控制", "接管"),
        "vehicle_stability": ("稳定",),
        "emergency_braking_available": ("紧急制动",),
        "function_type": ("AVP", "功能"),
        "has_remote_app": ("APP", "HMI"),
    }
    _SEVERITY_REQUIRED = {"road_user_type", "collision_type", "relative_speed_kph"}
    _TTC_FIELDS = {"relative_distance_m", "ttc_s"}

    def __init__(self, method: MethodContract):
        if method.structured_risk_method is None:
            raise ValueError("RiskContextSourceCoverageAuditService requires structured risk method")
        self.method = method
        self.structured = method.structured_risk_method

    @staticmethod
    def _source_ref(source: Any) -> dict[str, str]:
        return {
            "source": str(getattr(source, "workbook", "")),
            "location": "!".join(
                part for part in (
                    str(getattr(source, "sheet", "")),
                    str(getattr(source, "range", "")),
                ) if part
            ),
            "source_hash": str(getattr(source, "source_hash", "")),
        }

    @staticmethod
    def _context_key(value: dict[str, Any]) -> tuple[str, str]:
        return str(value.get("malfunction_id", "")), str(value.get("scenario_id", ""))

    @staticmethod
    def _finalized(value: dict[str, Any]) -> bool:
        return str(value.get("approval", value.get("status", ""))).upper() in {
            "FINALIZED", "APPROVED",
        }

    @classmethod
    def _document_signals(cls, document: Any, field: str) -> list[dict[str, str]]:
        terms = cls._DOCUMENT_TERMS[field]
        matches = []
        for block in getattr(document, "blocks", ()):
            text = str(getattr(block, "text", ""))
            if any(term.casefold() in text.casefold() for term in terms):
                matches.append({
                    "location": str(getattr(block, "location", "")),
                    "excerpt": text[:240],
                    "usability": "REJECTED_NON_EVENT_SPECIFIC",
                })
        return matches[:8]

    @classmethod
    def _risk_fact_is_source_grounded(
        cls, field: str, fact: dict[str, Any], document_blocks: dict[str, str],
    ) -> bool:
        """Require a deterministic field signal at the cited ItemDef location.

        A finalized schema record alone does not turn a generic ItemDef citation
        into evidence for an event-specific collision or road-user fact.
        """
        terms = cls._DOCUMENT_TERMS[field]
        for source in fact.get("source_refs", []):
            if not isinstance(source, dict):
                continue
            text = document_blocks.get(
                str(source.get("location", "")), str(source.get("excerpt", "")),
            )
            if any(term.casefold() in text.casefold() for term in terms):
                return True
        return False

    @staticmethod
    def _eligible_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
        assessments = trace.get("assessments", [])
        if not isinstance(assessments, list):
            raise ValueError("risk execution trace assessments must be a list")
        eligible = [
            item for item in assessments
            if isinstance(item, dict) and item.get("risk_scoring_invoked") is True
        ]
        if not eligible:
            raise ValueError("risk execution trace has no invoked RiskContext assessments")
        return eligible

    def _c_override_fields(self) -> set[str]:
        return {
            condition.field
            for rule in self.structured.controllability_overrides
            for condition in (*rule.all_of, *rule.any_of)
        }

    def _field_role(self, field: str) -> dict[str, Any]:
        c_fields = self._c_override_fields()
        severity_required = field in self._SEVERITY_REQUIRED
        c_override_required = field in c_fields
        ttc_conditionally_consumed = field in self._TTC_FIELDS
        return {
            "severity_consumer": "SeverityMethodExecutor" if severity_required else "",
            "controllability_consumer": (
                "StructuredControllabilityExecutor"
                if c_override_required or ttc_conditionally_consumed else ""
            ),
            "active_requirement": (
                "REQUIRED_BY_SEVERITY" if severity_required else
                "REQUIRED_BY_CONTROLLABILITY_OVERRIDE" if c_override_required else
                "CONDITIONALLY_REQUIRED_BY_TTC_BRANCH" if ttc_conditionally_consumed else
                "NOT_REQUIRED_BY_ACTIVE_METHOD"
            ),
        }

    def _raw_yaml_provenance(self) -> list[dict[str, Any]]:
        metadata = self.method.metadata
        severity_inventory = metadata.get("severity_source_inventory", [])
        result = [
            {
                "source": str(item.get("source", "")),
                "role": str(item.get("role", "")),
                "compiled_or_runtime_role": str(item.get("method_authority", "")),
                "runtime_consumer": str(item.get("runtime_consumer", "")),
                "classification": (
                    "EXAMPLE_OR_FALLBACK_ONLY"
                    if str(item.get("method_authority", "")).startswith("EXCLUDED")
                    else "COMPILED_METHOD_AUTHORITY"
                ),
            }
            for item in severity_inventory if isinstance(item, dict)
        ]
        policy = self.structured.controllability_branch_policy
        result.append({
            "source": str(policy.source_ref.workbook),
            "role": "SELECTED_CONTROLLABILITY_PROFILE_POLICY",
            "compiled_or_runtime_role": "COMPILED_METHOD_AUTHORITY",
            "runtime_consumer": "StructuredControllabilityExecutor",
            "classification": "COMPILED_METHOD_AUTHORITY",
        })
        result.extend({
            "source": str(item.get("source_asset", "")),
            "role": str(item.get("classification", "")),
            "compiled_or_runtime_role": "PROVENANCE_ONLY",
            "runtime_consumer": "NONE",
            "classification": "EXAMPLE_OR_FALLBACK_ONLY",
        } for item in metadata.get("scenario_coverage_knowledge_sources", [])
            if isinstance(item, dict) and item.get("source_asset"))
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for item in result:
            unique[(item["source"], item["role"])] = item
        return list(unique.values())

    def _classify(
        self,
        *,
        field: str,
        runtime_fact: dict[str, Any],
        source_risk_facts: list[dict[str, Any]],
        ambiguous_risk_facts: list[dict[str, Any]],
        scenario: dict[str, Any],
        binding_audit: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        if str(runtime_fact.get("status", "")) == "AVAILABLE":
            return "SOURCE_PRESENT_BOUND", "AUTO_RECOVERABLE", []
        if source_risk_facts:
            bound = set(binding_audit.get("bound_fact_types", []))
            fact_type = self._FACT_PARAMETER[field]
            if fact_type not in bound:
                return (
                    "SOURCE_PRESENT_BUT_NOT_BOUND", "NEEDS_PROJECT_BINDING",
                    ["RUNTIME_BINDING_DEFECT"],
                )
        if ambiguous_risk_facts:
            return (
                "SOURCE_PRESENT_BUT_AMBIGUOUS", "NEEDS_NEW_PROJECT_FACT",
                ["SOURCE_PRESENT_BUT_NOT_BOUND", "RUNTIME_BINDING_DEFECT"],
            )
        facts = scenario.get("facts", {}) if isinstance(scenario, dict) else {}
        provenance = scenario.get("fact_provenance", {}) if isinstance(scenario, dict) else {}
        if field in facts or field in provenance:
            return "RUNTIME_BINDING_DEFECT", "AUTO_RECOVERABLE", []
        if (
            field in self._TTC_FIELDS
            and self.structured.controllability_branch_policy.unknown_override_policy
            is UnknownOverridePolicy.UNSPECIFIED
        ):
            return "METHOD_RULE_PRESENT_BUT_NOT_APPLICABLE", "NEEDS_ENGINEERING_METHOD", []
        if field in self._TTC_FIELDS:
            return "TRUE_PROJECT_FACT_GAP", "NEEDS_NEW_PROJECT_FACT", []
        if field in self._SEVERITY_REQUIRED or field in self._c_override_fields():
            return "TRUE_PROJECT_FACT_GAP", "NEEDS_NEW_PROJECT_FACT", []
        return "NOT_REQUIRED_BY_ACTIVE_METHOD", "NOT_NEEDED", []

    def audit(
        self,
        *,
        state: dict[str, Any],
        trace: dict[str, Any],
        document: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Audit persisted evidence without changing the state, method, or trace."""
        eligible = self._eligible_trace(trace)
        typed = state.get("item_definition", {}).get("typed", {})
        if not isinstance(typed, dict):
            raise ValueError("checkpoint has no typed ItemDefinition facts")
        scenarios = {
            str(item.get("scenario_id", "")): item
            for item in state.get("scenarios", []) if isinstance(item, dict)
        }
        risk_facts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in typed.get("risk_facts", []):
            if not isinstance(item, dict) or not self._finalized(item):
                continue
            context = item.get("context", {})
            if not isinstance(context, dict):
                continue
            risk_facts[(
                str(context.get("malfunction_id", "")),
                str(context.get("scenario_id", "")),
                str(item.get("parameter", "")).upper(),
            )].append(item)
        scoring_event = next((
            item for item in reversed(state.get("audit_trail", []))
            if isinstance(item, dict) and item.get("event") == "structured_risk_scoring_completed"
        ), {})
        binding_audits = {
            self._context_key(item): item
            for item in scoring_event.get("risk_fact_binding_audits", [])
            if isinstance(item, dict)
        }
        source_path = Path(getattr(document, "source_path", ""))
        source_hash = ""
        if source_path.is_file():
            source_hash = sha256(source_path.read_bytes()).hexdigest()
        document_blocks = {
            str(getattr(block, "location", "")): str(getattr(block, "text", ""))
            for block in getattr(document, "blocks", ())
        }

        raw_yaml_provenance = self._raw_yaml_provenance()
        field_rows: list[dict[str, Any]] = []
        all_classifications: Counter[str] = Counter()
        all_recoverability: Counter[str] = Counter()
        existing_but_not_bound: list[dict[str, Any]] = []
        ambiguous_typed_records: list[dict[str, Any]] = []

        for field in self._FIELDS:
            role = self._field_role(field)
            rows = []
            runtime_statuses: Counter[str] = Counter()
            for assessment in eligible:
                pair = self._context_key(assessment)
                context = assessment.get("hazardous_event_risk_context", {})
                context = context if isinstance(context, dict) else {}
                runtime_fact = context.get(field, {})
                runtime_fact = runtime_fact if isinstance(runtime_fact, dict) else {}
                runtime_statuses[str(runtime_fact.get("status", "UNAVAILABLE"))] += 1
                parameter = self._FACT_PARAMETER[field]
                typed_source_facts = risk_facts.get((*pair, parameter), [])
                source_facts = [
                    item for item in typed_source_facts
                    if self._risk_fact_is_source_grounded(field, item, document_blocks)
                ]
                ambiguous_source_facts = [
                    item for item in typed_source_facts if item not in source_facts
                ]
                scenario = scenarios.get(pair[1], {})
                binding_audit = binding_audits.get(pair, {})
                classification, recovery, secondary = self._classify(
                    field=field,
                    runtime_fact=runtime_fact,
                    source_risk_facts=source_facts,
                    ambiguous_risk_facts=ambiguous_source_facts,
                    scenario=scenario,
                    binding_audit=binding_audit,
                )
                row = {
                    "malfunction_id": pair[0],
                    "scenario_id": pair[1],
                    "hazardous_event_id": str(assessment.get("hazardous_event_id", "")),
                    "classification": classification,
                    "secondary_classifications": secondary,
                    "recoverability": recovery,
                    "runtime_status": str(runtime_fact.get("status", "UNAVAILABLE")),
                    "runtime_reason": str(runtime_fact.get("reason", "")),
                }
                if typed_source_facts:
                    row["project_risk_fact_ids"] = [
                        str(item.get("fact_id", "")) for item in typed_source_facts
                    ]
                    row["project_risk_fact_sources"] = [
                        source for item in typed_source_facts
                        for source in item.get("source_refs", []) if isinstance(source, dict)
                    ]
                    row["project_risk_fact_source_grounding"] = (
                        "SOURCE_GROUNDED" if source_facts
                        else "CITED_ITEM_BLOCK_DOES_NOT_SUPPORT_FIELD_SEMANTICS"
                    )
                    row["binding_audit"] = {
                        key: binding_audit.get(key, [])
                        for key in (
                            "bound_fact_types", "missing_fact_types",
                            "automatic_binding_count", "explicit_binding_count",
                        )
                    }
                rows.append(row)
                all_classifications[classification] += 1
                all_recoverability[recovery] += 1
                if classification == "SOURCE_PRESENT_BUT_NOT_BOUND":
                    existing_but_not_bound.append({"field": field, **row})
                if classification == "SOURCE_PRESENT_BUT_AMBIGUOUS":
                    ambiguous_typed_records.append({"field": field, **row})

            classifications = Counter(item["classification"] for item in rows)
            recoverability = Counter(item["recoverability"] for item in rows)
            direct_scenario_count = sum(
                field in (scenarios.get(self._context_key(item)[1], {}).get("facts", {}) or {})
                for item in eligible
            )
            source_fact_count = sum(
                len(risk_facts.get((*self._context_key(item), self._FACT_PARAMETER[field]), []))
                for item in eligible
            )
            grounded_source_fact_count = sum(
                1 for item in eligible
                for fact in risk_facts.get((*self._context_key(item), self._FACT_PARAMETER[field]), [])
                if self._risk_fact_is_source_grounded(field, fact, document_blocks)
            )
            field_rows.append({
                "field": field,
                **role,
                "classification_counts": dict(sorted(classifications.items())),
                "recoverability_counts": dict(sorted(recoverability.items())),
                "source_coverage": {
                    "A_item_definition": {
                        "full_document_read": True,
                        "signals": self._document_signals(document, field),
                        "rule": "Document text is not an event-specific RiskContext fact without a typed binding.",
                    },
                    "B_project_facts_risk_facts": {
                        "finalized_context_specific_count": source_fact_count,
                        "source_grounded_count": grounded_source_fact_count,
                        "parameter": self._FACT_PARAMETER[field],
                    },
                    "C_scenario_fields": {
                        "direct_canonical_field_count": direct_scenario_count,
                    },
                    "D_hazardous_event_structured": {
                        "eligible_event_count": len(rows),
                        "writeback": "NOT_PERFORMED",
                    },
                    "E_compiled_method_contract": {
                        "active_requirement": role["active_requirement"],
                        "method_contract_hash": str(self.method.metadata.get("method_source_hash", "")),
                    },
                    "F_raw_yaml_provenance": {
                        "read_via": "YamlBaselineCompiler",
                        "runtime_authority": "COMPILED_METHOD_CONTRACT_ONLY",
                    },
                    "G_deterministic_derivation": (
                        {
                            "rule": "scenario_physics.derive_scenario_physics:TTC",
                            "status": "PRESENT_AND_WIRED_REQUIRES_DISTANCE_AND_POSITIVE_RELATIVE_SPEED",
                            "formula": TTC_CLOSING_FORMULA_IDENTITY,
                        } if field == "ttc_s" else {"status": "NO_FIELD_DERIVATION_RULE"}
                    ),
                    "H_adapters_resolvers": {
                        "method_risk_fact_binding_audit_records": len(binding_audits),
                        "automatic_binding_observed": bool(source_fact_count),
                    },
                    "I_runtime_final_risk_context": {
                        "status_counts": dict(sorted(runtime_statuses.items())),
                    },
                },
                "examples": [
                    item for item in rows
                    if item["classification"] != "NOT_REQUIRED_BY_ACTIVE_METHOD"
                ][:20],
            })

        method_gaps = [
            {
                "field": "EXPOSURE_DIMENSION_COVERAGE",
                "classification": "TRUE_METHOD_SEMANTICS_GAP",
                "recoverability": "NEEDS_ENGINEERING_METHOD",
                "evidence": "No selected compiled Function-to-dimension coverage rule is present.",
            },
            {
                "field": "controllability.iav_avp_v1.unknown_override_policy",
                "classification": "TRUE_METHOD_SEMANTICS_GAP",
                "recoverability": "NEEDS_ENGINEERING_METHOD",
                "evidence": "Selected profile has UNKNOWN override policy UNSPECIFIED; TTC is not entered.",
            },
        ]
        clarification = {
            "artifact_version": "engineering-clarification-package-p3a-v1",
            "run_id": str(state.get("run_id", "")),
            "method_contract_hash": str(self.method.metadata.get("method_source_hash", "")),
            "ec03_split": {
                "existing_but_not_bound": existing_but_not_bound,
                "rule_available_but_not_wired": [],
                "true_engineering_input": [
                    {
                        "field": item["field"],
                        "eligible_event_count": item["classification_counts"].get("TRUE_PROJECT_FACT_GAP", 0),
                        "required_action": "Provide an approved event-specific project fact; do not infer from a range, ontology, or example.",
                    }
                    for item in field_rows
                    if item["classification_counts"].get("TRUE_PROJECT_FACT_GAP", 0)
                ],
                "true_method_semantics": method_gaps,
            },
            "rejected_typed_source_records": ambiguous_typed_records,
            "no_project_fact_or_method_change_performed": True,
        }
        payload = {
            "artifact_version": "risk-context-source-coverage-audit-v1",
            "run_id": str(state.get("run_id", "")),
            "method_contract_hash": str(self.method.metadata.get("method_source_hash", "")),
            "item_definition_source": {
                "source_id": str(getattr(document, "source_id", "")),
                "path": str(source_path),
                "sha256": source_hash,
                "full_document_read": True,
                "block_count": len(getattr(document, "blocks", ())),
            },
            "audit_scope": {
                "eligible_hazardous_event_count": len(eligible),
                "provider_calls": 0,
                "runtime_yaml_read": 0,
                "raw_yaml_review": "PROVENANCE_ONLY_VIA_BASELINE_COMPILER",
                "method_contract_mutated": False,
                "risk_context_writeback": False,
                "scoring_invoked": False,
            },
            "source_authority_hierarchy": [
                {"rank": 1, "source": "direct Scenario/HazardousEvent fact", "accepted": True},
                {"rank": 2, "source": "explicit finalized ProjectFact/RiskFact", "accepted": True},
                {"rank": 3, "source": "selected compiled MethodContract deterministic rule", "accepted": True},
                {"rank": 4, "source": "approved deterministic physics", "accepted": True},
                {"rank": 5, "source": "speed envelope, ontology, raw example, or prose implication", "accepted": False},
            ],
            "field_inventory": [{"field": field, **self._field_role(field)} for field in self._FIELDS],
            "per_field": field_rows,
            "raw_yaml_provenance": raw_yaml_provenance,
            "method_semantics_gaps": method_gaps,
            "aggregate": {
                "eligible_hazardous_event_count": len(eligible),
                "classification_counts": dict(sorted(all_classifications.items())),
                "recoverability_counts": dict(sorted(all_recoverability.items())),
                "existing_but_not_bound_count": len(existing_but_not_bound),
                "typed_but_source_ambiguous_count": len(ambiguous_typed_records),
                "rule_available_but_not_wired_count": 0,
                "true_method_semantics_gap_count": len(method_gaps),
            },
            "actions": [
                {
                    "id": "P3A-01",
                    "classification": "RUNTIME_BINDING_DEFECT",
                    "action": "Reconcile finalized context-specific COLLISION_TYPE facts through the structured RiskContext binding boundary after source-validity review.",
                },
                {
                    "id": "P3A-02",
                    "classification": "TRUE_PROJECT_FACT_GAP",
                    "action": "Collect approved event-specific road-user, relative-speed, and controllability facts; do not promote ItemDef ranges or capabilities into event facts.",
                },
                {
                    "id": "P3A-03",
                    "classification": "TRUE_METHOD_SEMANTICS_GAP",
                    "action": "Obtain engineering authority for Exposure dimension coverage and the iav_avp_v1 UNKNOWN override transition before attempting E/C completion.",
                },
            ],
        }
        return payload, clarification
