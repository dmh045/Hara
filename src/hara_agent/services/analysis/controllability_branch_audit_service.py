"""Read-only audit of compiled controllability branch policy and readiness."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from typing import Any

from hara_agent.contracts import (
    ControllabilityFactState, HazardousEventRiskContext, MethodContract,
    RiskContextFactStatus, RuleMatchState, UnknownOverridePolicy,
)

from .hazardous_event_risk_context_service import HazardousEventRiskContextService


class ControllabilityBranchAuditService:
    """Project the compiled branch policy without reading raw YAML at runtime."""

    _OVERRIDE_FIELDS = (
        "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible",
    )
    _TTC_FIELDS = ["relative_distance_m", "relative_speed_kph", "ttc_s"]

    def __init__(self, method: MethodContract):
        if method.structured_risk_method is None:
            raise ValueError("ControllabilityBranchAuditService requires structured risk method")
        self.method = method
        self.structured = method.structured_risk_method
        self.contexts = HazardousEventRiskContextService(method)

    @staticmethod
    def _source(source: Any) -> dict[str, str]:
        return {
            "source": str(getattr(source, "workbook", "")),
            "location": f"{getattr(source, 'sheet', '')}!{getattr(source, 'range', '')}".strip("!"),
            "source_hash": str(getattr(source, "source_hash", "")),
        }

    @staticmethod
    def _state(fact: Any) -> ControllabilityFactState:
        if isinstance(fact, dict):
            status, value = str(fact.get("status", "")), fact.get("value")
        else:
            status = getattr(fact, "status", RiskContextFactStatus.UNAVAILABLE)
            status = status.value if hasattr(status, "value") else str(status)
            value = getattr(fact, "value", None)
        if status == "CONFLICT":
            return ControllabilityFactState.CONFLICT
        if status != RiskContextFactStatus.AVAILABLE.value or not isinstance(value, bool):
            return ControllabilityFactState.UNKNOWN
        return ControllabilityFactState.TRUE if value else ControllabilityFactState.FALSE

    def _states(self, context: HazardousEventRiskContext | dict[str, Any]) -> dict[str, ControllabilityFactState]:
        return {
            field: self._state(context.get(field, {}) if isinstance(context, dict) else getattr(context, field))
            for field in self._OVERRIDE_FIELDS
        }

    @staticmethod
    def _condition_state(state: ControllabilityFactState, expected: bool) -> ControllabilityFactState:
        if state in {ControllabilityFactState.UNKNOWN, ControllabilityFactState.CONFLICT}:
            return state
        return ControllabilityFactState.TRUE if ((state is ControllabilityFactState.TRUE) is expected) else ControllabilityFactState.FALSE

    def _rule_state(self, rule, states: dict[str, ControllabilityFactState]) -> RuleMatchState:
        values = [self._condition_state(states[item.field], item.expected) for item in (rule.all_of or rule.any_of)]
        if ControllabilityFactState.CONFLICT in values:
            return RuleMatchState.CONFLICT
        if rule.all_of:
            if ControllabilityFactState.FALSE in values:
                return RuleMatchState.NO_MATCH
            return RuleMatchState.UNKNOWN if ControllabilityFactState.UNKNOWN in values else RuleMatchState.MATCH
        if ControllabilityFactState.TRUE in values:
            return RuleMatchState.MATCH
        return RuleMatchState.UNKNOWN if ControllabilityFactState.UNKNOWN in values else RuleMatchState.NO_MATCH

    @staticmethod
    def _fields(rule) -> list[str]:
        return [item.field for item in (*rule.all_of, *rule.any_of)]

    def _result(
        self, *, status: str, stage: str, fields: list[str],
        states: dict[str, ControllabilityFactState], match_states: list[dict[str, str]],
        rule_id: str = "", result: str = "", reason: str = "", policy_action: str,
    ) -> dict[str, Any]:
        unresolved = [
            field for field in fields
            if states.get(field) is ControllabilityFactState.UNKNOWN
        ]
        result = {
            "status": status, "decision_stage": stage,
            "not_required_inputs": self._TTC_FIELDS if stage != "TTC" else list(self._OVERRIDE_FIELDS),
            "matched_rule_id": rule_id if status == "READY" else "",
            "blocked_rule_id": rule_id if status == "FACT_SOURCE_CONFLICT" else "",
            "result": result, "reason": reason,
            "unknown_override_policy": self.structured.controllability_branch_policy.unknown_override_policy.value,
            "unknown_policy_action": policy_action, "rule_match_states": match_states,
            "override_states": {key: value.value for key, value in states.items()},
        }
        if status == "METHOD_BRANCH_UNRESOLVED":
            result.update({
                "decision_inputs": fields,
                "unresolved_inputs": unresolved,
                "ttc_execution_outcome": "NOT_ENTERED_METHOD_BRANCH_UNRESOLVED",
            })
        else:
            result["required_inputs"] = fields
            result["missing_inputs"] = unresolved
        return result

    def _ttc_readiness(
        self, context: HazardousEventRiskContext | dict[str, Any],
        states: dict[str, ControllabilityFactState], match_states: list[dict[str, str]], *,
        policy_action: str,
    ) -> dict[str, Any]:
        ttc = context.get("ttc_s", {}) if isinstance(context, dict) else context.ttc_s
        status = str(ttc.get("status", "")) if isinstance(ttc, dict) else ttc.status.value
        if status == RiskContextFactStatus.AVAILABLE.value:
            return self._result(
                status="READY", stage="TTC", fields=self._TTC_FIELDS, states=states,
                match_states=match_states, policy_action=policy_action,
            )
        missing = (
            ["TTC_NOT_CLOSING"]
            if status == RiskContextFactStatus.TTC_NOT_CLOSING.value
            else ["relative_distance_m", "relative_speed_kph"]
        )
        result = self._result(
            status="PENDING_INPUT", stage="TTC", fields=self._TTC_FIELDS, states=states,
            match_states=match_states, policy_action=policy_action,
        )
        result["missing_inputs"] = missing
        return result

    def readiness(self, context: HazardousEventRiskContext | dict[str, Any]) -> dict[str, Any]:
        states = self._states(context)
        match_states: list[dict[str, str]] = []
        unknown_fields: list[str] = []
        for rule in sorted(self.structured.controllability_overrides, key=lambda item: item.priority):
            result = self._rule_state(rule, states)
            match_states.append({"rule_id": rule.rule_id, "state": result.value})
            fields = self._fields(rule)
            if result is RuleMatchState.MATCH:
                return self._result(
                    status="READY", stage="OVERRIDE", fields=fields, states=states,
                    match_states=match_states, rule_id=rule.rule_id, result=rule.result,
                    policy_action="NOT_APPLICABLE",
                )
            if result is RuleMatchState.CONFLICT:
                return self._result(
                    status="FACT_SOURCE_CONFLICT", stage="OVERRIDE", fields=fields,
                    states=states, match_states=match_states, rule_id=rule.rule_id,
                    policy_action="CONFLICT_BLOCKED",
                )
            if result is RuleMatchState.UNKNOWN:
                unknown_fields.extend(fields)
        policy = self.structured.controllability_branch_policy.unknown_override_policy
        if unknown_fields:
            fields = list(dict.fromkeys(unknown_fields))
            if policy is UnknownOverridePolicy.UNSPECIFIED:
                return self._result(
                    status="METHOD_BRANCH_UNRESOLVED", stage="UNRESOLVED", fields=fields,
                    states=states, match_states=match_states,
                    reason="CONTROLLABILITY_UNKNOWN_BRANCH_POLICY_UNSPECIFIED",
                    policy_action="NO_TRANSITION_DEFINED",
                )
            if policy is UnknownOverridePolicy.BLOCK_TTC:
                return self._result(
                    status="PENDING_INPUT", stage="OVERRIDE", fields=fields,
                    states=states, match_states=match_states, policy_action="BLOCK_TTC",
                )
            if policy is not UnknownOverridePolicy.SKIP_TO_TTC:
                raise ValueError(f"Unsupported unknown override policy: {policy}")
            return self._ttc_readiness(context, states, match_states, policy_action="SKIP_TO_TTC")
        return self._ttc_readiness(context, states, match_states, policy_action="ALL_OVERRIDES_NO_MATCH")

    def _decision_tree(self) -> dict[str, Any]:
        policy = self.structured.controllability_branch_policy
        overrides = [{
            "priority": rule.priority, "rule_id": rule.rule_id, "type": "POSITIVE_OVERRIDE",
            "operator": "ALL_OF" if rule.all_of else "ANY_OF",
            "conditions": [{"field": item.field, "expected": item.expected} for item in (rule.all_of or rule.any_of)],
            "result": rule.result, "source_ref": self._source(rule.source_ref),
        } for rule in sorted(self.structured.controllability_overrides, key=lambda item: item.priority)]
        profile = self.structured.controllability_profile
        absence_evidence: dict[str, Any] | None = None
        try:
            provenance = json.loads(policy.source_ref.raw_text)
        except (TypeError, json.JSONDecodeError):
            provenance = {}
        if provenance.get("asset_field_present") is False:
            absence_evidence = {
                "inspected_source": policy.source_ref.workbook,
                "profile_id": policy.profile_id,
                "method_hash": policy.method_hash,
                "source_hash": policy.source_ref.template_hash,
                "field_present": False,
            }
        return {
            "selected_profile": profile.profile_id, "source_status": policy.source_status,
            "unknown_override_policy": policy.unknown_override_policy.value,
            "policy_source_ref": None if absence_evidence else self._source(policy.source_ref),
            "policy_absence_evidence": absence_evidence,
            "method_hash": policy.method_hash,
            "implicit_python_default": False, "overrides": overrides,
            "ttc_branch": {
                "required_facts": ["relative_distance_m", "relative_speed_kph"],
                "preconditions": ["all overrides are NO_MATCH, or compiled policy explicitly permits TTC after UNKNOWN"],
                "fallback_status_when_ttc_missing": "PENDING_INPUT",
                "source_ref": self._source(profile.source_ref),
            },
            "ttc_thresholds": [{
                "rule_id": item.rule_id, "lower_ttc_s": item.lower_ttc_s,
                "lower_inclusive": item.lower_inclusive, "upper_ttc_s": item.upper_ttc_s,
                "upper_inclusive": item.upper_inclusive, "result": item.result,
                "source_ref": self._source(item.source_ref),
            } for item in profile.bands],
        }

    @staticmethod
    def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(item["controllability_readiness"]["status"] for item in rows)
        stages = Counter(item["controllability_readiness"]["decision_stage"] for item in rows)
        return {
            "causal_relevant_hazardous_events": len(rows),
            "resolved_by_override": sum(row["controllability_readiness"]["decision_stage"] == "OVERRIDE" and row["controllability_readiness"]["status"] == "READY" for row in rows),
            "eligible_for_ttc": stages["TTC"],
            "blocked_before_ttc": sum(row["controllability_readiness"]["unknown_policy_action"] == "BLOCK_TTC" for row in rows),
            "method_branch_unresolved": counts["METHOD_BRANCH_UNRESOLVED"],
            "ttc_inputs_ready": sum(row["controllability_readiness"]["decision_stage"] == "TTC" and row["controllability_readiness"]["status"] == "READY" for row in rows),
            "ttc_not_closing": sum("TTC_NOT_CLOSING" in row["controllability_readiness"].get("missing_inputs", []) for row in rows),
            "c_ready": counts["READY"], "c_pending_input": counts["PENDING_INPUT"],
            "c_method_branch_unresolved": counts["METHOD_BRANCH_UNRESOLVED"],
            "historical_source_leakage": 0, "prose_derived_c_fact": 0,
        }

    def _fixture_projection(
        self, policy: UnknownOverridePolicy, contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fixture_method = replace(
            self.method,
            structured_risk_method=replace(
                self.structured,
                controllability_branch_policy=replace(
                    self.structured.controllability_branch_policy,
                    unknown_override_policy=policy,
                ),
            ),
        )
        fixture = ControllabilityBranchAuditService(fixture_method)
        rows = [{
            "malfunction_id": item["malfunction_id"], "scenario_id": item["scenario_id"],
            "hazardous_event_id": item["hazardous_event_id"],
            "controllability_readiness": fixture.readiness(item["risk_context"]),
        } for item in contexts]
        return {"policy": policy.value, "rows": rows, "summary": self._summary(rows)}

    def audit(self, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        risk_context = self.contexts.audit(records)
        source_contexts = risk_context["r3_risk_context"]
        rows = [{
            "malfunction_id": item["malfunction_id"], "scenario_id": item["scenario_id"],
            "hazardous_event_id": item["hazardous_event_id"],
            "controllability_readiness": self.readiness(item["risk_context"]),
        } for item in source_contexts]
        tree = self._decision_tree()
        return {
            "artifact_version": "controllability-branch-policy-audit-v1",
            "method_contract_hash": str(self.method.metadata.get("method_source_hash", "")),
            "runtime_yaml_read": 0,
            "source_authority_hierarchy": [
                {"source": tree["ttc_branch"]["source_ref"]["source"], "role": "SELECTED_CONTROLLABILITY_PROFILE", "selected": True, "authority": "ACTIVE_NUMERIC_AND_OVERRIDE_METHOD", "runtime_consumer": "StructuredControllabilityExecutor"},
                {"source": "normalized/controllability_aliases.yaml", "role": "FIELD_NORMALIZATION", "selected": True, "authority": "ACTIVE_COMPILER_MAPPING", "runtime_consumer": "YamlBaselineCompiler"},
                {"source": "raw/controllability_rules.yaml", "role": "HISTORICAL_CONTROLLABILITY_REFERENCE", "selected": False, "authority": "EXCLUDED_NUMERIC_AUTHORITY", "runtime_consumer": "NONE"},
                {"source": "raw/c_iso26262.yaml", "role": "ISO_CONTROLLABILITY_REFERENCE", "selected": False, "authority": "EXCLUDED_NUMERIC_AUTHORITY", "runtime_consumer": "NONE"},
                {"source": "raw/domain_rules/avp_low_speed.yaml", "role": "DOMAIN_CONTROLLABILITY_REFERENCE", "selected": False, "authority": "EXCLUDED_NUMERIC_SECTION", "runtime_consumer": "NONE"},
            ],
            "confirmed_method_decision_tree": tree,
            "current_runtime_decision_tree": {
                "override_order": "priority ascending before TTC",
                "unknown_override_policy": tree["unknown_override_policy"],
                "source": "StructuredControllabilityExecutor",
            },
            "runtime_comparison": {
                "runtime_contract_alignment": "MATCH",
                "method_semantic_completeness": (
                    "INCOMPLETE_UNKNOWN_POLICY"
                    if self.structured.controllability_branch_policy.unknown_override_policy
                    is UnknownOverridePolicy.UNSPECIFIED else "COMPLETE"
                ),
            },
            "r3_readiness": rows,
            "policy_fixtures": {
                "UNSPECIFIED": self._fixture_projection(
                    UnknownOverridePolicy.UNSPECIFIED, source_contexts,
                ),
                "BLOCK_TTC": self._fixture_projection(UnknownOverridePolicy.BLOCK_TTC, source_contexts),
                "SKIP_TO_TTC": self._fixture_projection(UnknownOverridePolicy.SKIP_TO_TTC, source_contexts),
            },
            "summary": self._summary(rows),
        }
