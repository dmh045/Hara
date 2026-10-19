from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from hara_agent.contracts import CausalBreakpoint
from hara_agent.infrastructure.llm import (
    LLMClient, LLMEmptyOutputError, LLMJSONContractError,
    LLMOutputLimitError, LLMRequest,
    LLMSchemaContractError, LLMTimeoutError,
)
from hara_agent.models import (
    MalfunctionCandidate,
    ReviewStatus,
    ScenarioCandidate,
    ScenarioFeasibilityAssessment,
)
from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
from hara_agent.services.analysis.scenario_physics import causal_physics_conflicts

from .parsing import parse_confidence
from .scenario_contract import RISK_DIMENSION_VALUES
from .scenario_evidence import (
    SCENARIO_ASSESSMENT_CONTRACT_VERSION,
    EvidenceBasisType, FactRegistry, build_fact_registry, compile_causal_assessment,
    ScenarioEvidenceContractError, validate_evidence_contract,
)
from .scenario_batching import (
    DEFAULT_CAUSAL_EVIDENCE_BUDGET,
    DEFAULT_SCENARIO_BATCH_MAX_CHARS,
    DEFAULT_SCENARIO_BATCH_MAX_ITEMS,
    ScenarioAdaptiveBatchError, ScenarioCoverageContractError,
    ScenarioBatchSizeError,
    ScenarioSchemaContractError,
    build_scenario_batches,
    build_scenario_user_prompt,
    select_scenario_causal_evidence,
)


RECOVERABLE_REPAIR_PROVIDER_ERRORS = (
    LLMEmptyOutputError,
    LLMJSONContractError,
    LLMOutputLimitError,
    LLMSchemaContractError,
)
REPAIR_CONTRACT_ERRORS = (
    ScenarioCoverageContractError,
    ScenarioSchemaContractError,
    ScenarioEvidenceContractError,
)


class ScenarioRepairBoundaryError(ScenarioSchemaContractError):
    """A repair tried to downgrade an already-supported causal hop."""


class ScenarioSemanticConsistencyError(ScenarioSchemaContractError):
    """Parseable evidence contradicts the supplied physical world."""


class ScenarioFeasibilityAgent:
    """Assess each bounded candidate without treating generation as applicability proof."""

    PROMPT_VERSION = "scenario-feasibility-v26-dimension-evidence"
    ALLOWED_RISK_DIMENSIONS = frozenset(RISK_DIMENSION_VALUES)
    MAX_PROVIDER_OUTPUT_TOKENS = 16384
    paired_driver_requests = True
    SYSTEM_PROMPT = """ROLE AND EVIDENCE
你是 HARA 工程因果分类器。独立判断每个明确场景的 physical feasibility、functional relevance、causal relevance。
只能使用当前场景 causal_evidence_view 的精确 evidence_ref、固定 Malfunction 字段及共享上下文；共享上下文仅表示相同输入，不能复制其他分支的结论。
不得发明额外故障、道路参与者、恐慌、方向改变、道路条件、干预能力、碰撞参数或概率。场景名称不能替代结构化事实。source_operating_conditions 是来源范围；permitted_road_types 是允许的替代条件，不证明所有几何条件已被覆盖。selected_method_conditions 不得扩大项目 ODD。

CAUSAL CHAIN AND STAGE BOUNDARY
只分析 M→B→I→H，到 Hazardous Event 结束，不输出 Potential Harm、S/E/C、ASIL、FTTI 或安全目标。
Hazardous Event 是危险车辆状态与运行场景的组合，事故无须已经发生；缺少具体碰撞对象本身不能作为B_TO_I或I_TO_H断裂的唯一理由。已有运行条件下失去横纵向控制可构成危险状态，但不得写成与特定行人、车辆或设施必然碰撞。
M_TO_B 用 FINALIZED MF.description 和 MF.functional_effect 的 DIRECT_FACT 说明已有失效的直接行为。缺少当前场景触发条件应在 B_TO_I 检查，不得改写失效定义或将其改成 M_TO_B 缺口。
MF.vehicle_level_hazard 是上游因果主张，不能单独证明 I_TO_H。每个 claim 必须解释所引证据怎样支持这一跳。
DIRECT_FACT 只能引用 [kind=DIRECT_FACT]；DERIVED_PHYSICS 只能引用 [kind=DERIVED_PHYSICS] 的已有计算，不能把速度/距离原始事实伪装成已计算物理结论；APPROVED_RULE 只能引用 [kind=APPROVED_RULE]。一跳中不要混用不同 kind。
已验证且限于当前场景的分析事实可引用；未验证 SCENARIO_DEFINED 假设不能支持正向因果链。
conditional_physical_analysis 是程序已检查并实例化的条件性物理世界。它不是项目事实或工程批准；可用其中精确 SCN/DERIVED 引用证明“在这些假设成立时”的因果关系。不要因为假设未在 Item 原文逐字出现就拒绝条件分析。不得将该条件结论泛化为所有故障或真实能力上界。
只有三跳均有支持时 causally_relevant=true、breakpoint=NONE。否则在第一跳证据不足处停止，breakpoint 为 M_TO_B/B_TO_I/I_TO_H，该跳 basis_type=ASSUMPTION，claim 指明缺少的必要条件，evidence_refs 可为 []，省略后续跳。保留前面已成立的跳，不得为了格式修复而降低已有证据结论。
false 时 hazardous_event=""、risk_dimension_changes=[]。没有证据不等于证明不可能。positive 时 hazardous_event 非空，risk_dimension_changes 仍可为 []。

PHYSICAL CONSISTENCY
physical_interaction 与物理评分使用同一组对象、方向、阶段和模型。存在代表工况时只验证所列工况，不把它解释为所有故障轨迹的上界。不得忽略该模型或对未列出的对象/运动过程复用父级结论。
physical_interaction 是展示视图，其 JSON 路径不是证据 ID。引用所列条件时必须使用 physical_interaction.evidence_refs 中的完整 ID，或 causal_evidence_view 已列出的 ID；不得引用 physical_interaction.conditions.* 或 physical_interaction.branches.*。physically_possible 只是物理条件检查，不能据此跳过因果判断。
derived_physics_evidence 中列出本场景已注册的物理推导及其 kind。basis_type=DERIVED_PHYSICS 时仅引用 kind=DERIVED_PHYSICS 的完整 ID，不混入 SCN.collision_branches 或其他 DIRECT_FACT；代表工况的推导只能证明该工况，不能证明全部故障轨迹。
DERIVED.parameter_relations 是当前对象、故障后阶段和模型的条件性符号关系，可在其适用条件内引用。numeric_result_available=false 表示尚无唯一碰撞速度或时间；不得把公式当成已发生接触、数值结果或全部参数范围的证明。符号参数本身不否定条件性危险，但不满足适用条件或证据不足时仍返回具体未决断点。
前后左右以本车坐标为准；倒车时不要把后方对象改成前方，不要声称远离对象时距离正在缩小。
若主张与当前外部目标接近/接触，返回 physical_interaction={"target":"SCENARIO_OBJECT","object_position":"输入的方位","relation":"APPROACH或CONTACT","assessment_reference":"CURRENT_STATE或MODEL_TRAJECTORY"}；存在 object_id 时原样返回。这里只描述主张，不提供新的物理事实或批准。其他对象、阶段或车内非接触伤害返回 target="OTHER"，不得借当前目标的接近速度否定其他机制。
对象类别与运动状态分开：静止电车仍是电车。初始/正常车速不等于故障可达速度或碰撞速度。静止后溜车需要独立运动依据，不能直接用正常上限；静止 EPB 夹紧损坏不能直接当成外部碰撞伤害。

OUTPUT CONTRACT
输出 assessments 数组，每个请求 scenario_id 恰好一项。每项包含 scenario_id、physically_feasible、functionally_relevant、causally_relevant (JSON boolean)、breakpoint、causal_chain、risk_dimension_changes、hazardous_event、physical_interaction、confidence (0..1)、status (PENDING/FINALIZED)。physical_interaction 按上面的对象/阶段结构返回，无当前外部目标主张时返回 {"target":"OTHER"}，不可省略。
causal_chain 是对象而非数组，结构 {"m_to_b":{"claim":"...","basis_type":"DIRECT_FACT","evidence_refs":[]},"b_to_i":{...},"i_to_h":{...}}。所有返回的 hop 必须有非空 claim；示例只说明结构，不构成证据。
risk_dimension_changes 每项含 dimension、evidence_refs、reason；只选择当前事实明确需要重评的维度，使用用户提示中的枚举，不输出 risk_dimensions_changed 或 rationale。
只返回规定的必要字段，不重复输入、rationale或shared_causal_evidence。仅当四项都有明确证据时返回可选 dependency_metadata，保持 complete=false；driver_noninterference 分 enablement/trigger/interaction/hazard 四项，每项 {independent:boolean,evidence_refs:["精确ID"],reason:"简短依据"}。缺少任一项引用时省略该可选对象，不输出空引用或冗长的无法证明说明。输入 shared_causal_evidence 可用于共有步骤；差异步骤仍独立判断并返回完整链，程序自行记录复用。"""

    MACHINE_OUTPUT_RULE = """MACHINE OUTPUT RULE
Return exactly one JSON object {"assessments": [...]} with valid JSON strings, no markdown, comments, trailing commas or explanation outside JSON.
"""

    def __init__(
        self, client: LLMClient, *,
        batch_max_chars: int | None = None,
        batch_max_items: int | None = None,
        max_split_depth: int | None = None,
        causal_evidence_budget: int | None = None,
        response_checkpoint_dir: Path | None = None,
    ):
        self.client = client
        self.response_checkpoint = ResponseCheckpoint(response_checkpoint_dir) if response_checkpoint_dir else None
        self.prompt_version = self.PROMPT_VERSION
        self.assessment_contract_version = SCENARIO_ASSESSMENT_CONTRACT_VERSION
        self.schema_name = "ScenarioFeasibilityAssessmentList"
        self.system_prompt = self.SYSTEM_PROMPT
        config = getattr(client, "config", None)
        response_format = getattr(config, "scenario_response_format", "none")
        self.response_format = {"type": "json_object"} if response_format == "json_object" else None
        self.causal_evidence_budget = causal_evidence_budget or getattr(
            config, "scenario_causal_evidence_budget", DEFAULT_CAUSAL_EVIDENCE_BUDGET,
        )
        if not isinstance(self.causal_evidence_budget, int) or self.causal_evidence_budget <= 0:
            raise ValueError("scenario causal evidence budget must be a positive integer")
        self.batch_max_chars = batch_max_chars or getattr(
            config, "scenario_batch_max_chars", DEFAULT_SCENARIO_BATCH_MAX_CHARS,
        )
        self.batch_max_items = batch_max_items or getattr(
            config, "scenario_batch_max_items", DEFAULT_SCENARIO_BATCH_MAX_ITEMS,
        )
        self.max_split_depth = (
            max_split_depth if max_split_depth is not None
            else getattr(config, "scenario_max_split_depth", 8)
        )
        self.provider_max_tokens = self._provider_max_tokens()

    def _complete_json(self, request):
        request = replace(request, metadata={**request.metadata,
            "method_execution_identity": getattr(self, "method_execution_identity", {})})
        return (self.response_checkpoint.complete(self.client, request)
                if self.response_checkpoint else self.client.complete_json(request))

    def _provider_max_tokens(self) -> int:
        configured = int(getattr(
            getattr(self.client, "config", None), "max_tokens", 32768,
        ))
        if configured <= 0:
            raise ValueError("scenario provider max_tokens must be greater than 0")
        return min(configured, self.MAX_PROVIDER_OUTPUT_TOKENS)

    def assess(self, malfunction: MalfunctionCandidate,
               scenarios: list[ScenarioCandidate],
               project_registry: FactRegistry | None = None,
               ) -> tuple[list[ScenarioFeasibilityAssessment], dict[str, Any]]:
        ids = [item.scenario_id for item in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("场景候选ID重复")
        batches = build_scenario_batches(
            malfunction, scenarios,
            max_chars=self.batch_max_chars,
            max_items=self.batch_max_items,
            system_prompt=self.system_prompt + "\n" + self.MACHINE_OUTPUT_RULE,
            project_registry=project_registry,
            causal_evidence_budget=self.causal_evidence_budget,
        )
        batch_sizes = [
            len(self.system_prompt + self.MACHINE_OUTPUT_RULE) + len(build_scenario_user_prompt(
                malfunction, batch, project_registry,
                self.causal_evidence_budget,
            ))
            for batch in batches
        ]
        print(
            "[HARA] scenario batching "
            f"malfunction={malfunction.malfunction_id} scenario_count={len(scenarios)} "
            f"batch_count={len(batches)} max_batch_chars={max(batch_sizes, default=0)} "
            f"max_batch_items={max((len(batch) for batch in batches), default=0)}",
            file=sys.stderr,
            flush=True,
        )
        assessments: list[ScenarioFeasibilityAssessment] = []
        batch_usages, models, request_ids = [], [], []
        stats: dict[str, Any] = {
            "shared_context_batch_count": 0,
            "adaptive_split_count": 0,
            "adaptive_split_reasons": {},
            "initial_batch_provider_calls": 0,
            "actual_llm_calls": 0,
            "response_checkpoint_hits": 0,
            "retry_calls": 0,
            "output_limit_count": 0,
            "timeout_count": 0,
            "schema_error_count": 0,
            "coverage_error_count": 0,
            "item_contract_repair_count": 0,
            "item_contract_repair_failure_count": 0,
            "single_item_failures": 0,
            "valid_initial_count": 0,
            "invalid_initial_count": 0,
            "valid_items_salvaged": 0,
            "invalid_items_repaired": 0,
            "repair_success_count": 0,
            "repair_failed_count": 0,
            "repair_failure_by_code": {},
            "missing_ids": [],
            "unknown_ids": [],
            "duplicate_ids": [],
            "item_salvage_audit": [],
            "error_counts_by_code": {},
            "candidate_evidence_count": 0,
            "selected_evidence_count": 0,
            "causal_prompt_evidence_chars": 0,
            "causal_prompt_evidence_tokens": 0,
            "evidence_selection_reason": {},
            "dropped_evidence_count": 0,
            "mandatory_evidence_refs": [],
            "selected_context_evidence_refs": [],
            "total_prompt_evidence_count": 0,
            "evidence_selection_audit": [],
            "m_to_b_anchor_available": False,
            "m_to_b_anchor_refs": [],
            "leaf_items": [],
            "leaf_chars": [],
            "leaf_elapsed": [],
            "completion_tokens": [],
            "reasoning_characters": [],
            "transport_error_counts": {},
            "json_contract_errors": 0,
            "format_retry_calls": 0,
            "format_retry_successes": 0,
            "format_retry_failures": 0,
            "markdown_fence_normalizations": 0,
        }
        started = time.monotonic()
        partial_records = []
        for batch_index, batch in enumerate(batches, start=1):
            original_batch = batch
            anchor = next((a for a in assessments if
                a.causal_chain.get("m_to_b", {}).get("basis_type") == "DIRECT_FACT"
                and a.causal_chain["m_to_b"].get("evidence_refs")
                and all(ref.startswith("MF.") for ref in a.causal_chain["m_to_b"]["evidence_refs"])), None)
            if anchor is not None:
                batch = [replace(candidate, analysis_instance={**candidate.analysis_instance,
                    "shared_causal_evidence": {"reused_from": anchor.scenario_id,
                                               "m_to_b": anchor.causal_chain["m_to_b"]}})
                    if not candidate.analysis_instance.get("shared_causal_evidence") else candidate
                    for candidate in batch]
                if (len(batch) > 1 and len(self.system_prompt + self.MACHINE_OUTPUT_RULE)
                        + len(build_scenario_user_prompt(malfunction, batch, project_registry,
                                                        self.causal_evidence_budget)) > self.batch_max_chars):
                    batch = original_batch  # optional context must not create an extra request
            returned = self._assess_batch_adaptively(
                malfunction, batch,
                parent_batch=f"{batch_index}/{len(batches)}",
                split_path="root",
                depth=0,
                stats=stats,
                batch_usages=batch_usages,
                models=models,
                request_ids=request_ids,
                project_registry=project_registry,
            )
            by_id = {assessment.scenario_id: assessment for assessment in returned}
            for candidate in batch:
                shared = candidate.analysis_instance.get("shared_causal_evidence", {})
                if shared:
                    partial_records.append({"scenario_id": candidate.scenario_id,
                        "representative_scenario_id": shared["reused_from"], "kind": "partial_causal",
                        "shared_steps": ["m_to_b"], "remaining_steps": "independently validated",
                        "reused": by_id[candidate.scenario_id].causal_chain.get("m_to_b") == shared["m_to_b"]})
            assessments.extend(returned)
        self._validate(assessments, ids)
        order = {scenario_id: index for index, scenario_id in enumerate(ids)}
        assessments.sort(key=lambda item: order[item.scenario_id])
        elapsed = time.monotonic() - started
        print(
            "[HARA] scenario feasibility completed "
            f"malfunction={malfunction.malfunction_id} scenario_count={len(scenarios)} "
            f"batch_count={len(batches)} received={len(assessments)} "
            f"feasible={sum(item.retain for item in assessments)} "
            f"infeasible={sum(not item.retain for item in assessments)} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return assessments, {
            "task": "assess_scenario_feasibility",
            "malfunction_id": malfunction.malfunction_id,
            "prompt_version": self.prompt_version,
            "assessment_contract_version": self.assessment_contract_version,
            "schema_name": self.schema_name,
            "models": models,
            "request_ids": request_ids,
            "usage": {"batches": batch_usages},
            "partial_evidence_reuse_records": partial_records,
            "candidate_count": len(scenarios),
            "retained_count": sum(item.retain for item in assessments),
            "batch_count": len(batches),
            "shared_context_batch_count": stats["shared_context_batch_count"],
            "initial_batch_count": len(batches),
            "adaptive_split_count": stats["adaptive_split_count"],
            "leaf_batch_count": len(stats["leaf_items"]),
            "llm_calls": stats["actual_llm_calls"],
            "actual_llm_calls": stats["actual_llm_calls"],
            "response_checkpoint_hits": stats["response_checkpoint_hits"],
            "provider_calls_total": stats["actual_llm_calls"],
            "initial_batch_provider_calls": stats["initial_batch_provider_calls"],
            "item_repair_calls": stats["item_contract_repair_count"],
            "adaptive_split_calls": stats["adaptive_split_count"],
            "retry_calls": stats["retry_calls"],
            "batch_input_chars": stats["leaf_chars"],
            "initial_batch_input_chars": batch_sizes,
            "max_batch_chars": max(batch_sizes, default=0),
            "configured_batch_max_chars": self.batch_max_chars,
            "configured_batch_max_items": self.batch_max_items,
            "provider_output_token_limit": self.provider_max_tokens,
            "elapsed_seconds": round(elapsed, 3),
            **{key: stats[key] for key in (
                "output_limit_count", "timeout_count", "schema_error_count",
                "coverage_error_count", "item_contract_repair_count",
                "item_contract_repair_failure_count", "single_item_failures",
            )},
            "error_counts_by_code": dict(sorted(stats["error_counts_by_code"].items())),
            "candidate_evidence_count": stats["candidate_evidence_count"],
            "selected_evidence_count": stats["selected_evidence_count"],
            "mandatory_evidence_refs": list(stats["mandatory_evidence_refs"]),
            "selected_context_evidence_refs": list(stats["selected_context_evidence_refs"]),
            "total_prompt_evidence_count": stats["total_prompt_evidence_count"],
            "evidence_selection_audit": list(stats["evidence_selection_audit"]),
            "m_to_b_anchor_available": stats["m_to_b_anchor_available"],
            "m_to_b_anchor_refs": list(stats["m_to_b_anchor_refs"]),
            "causal_prompt_evidence_chars": stats["causal_prompt_evidence_chars"],
            "causal_prompt_evidence_tokens": stats["causal_prompt_evidence_tokens"],
            "evidence_selection_reason": dict(sorted(stats["evidence_selection_reason"].items())),
            "dropped_evidence_count": stats["dropped_evidence_count"],
            "expected_count": len(scenarios),
            "returned_count": len(assessments),
            "valid_initial_count": stats["valid_initial_count"],
            "invalid_initial_count": stats["invalid_initial_count"],
            "valid_items_salvaged": stats["valid_items_salvaged"],
            "salvaged_count": stats["valid_items_salvaged"],
            "invalid_items_repaired": stats["invalid_items_repaired"],
            "repair_attempted_count": stats["item_contract_repair_count"],
            "repair_success_count": stats["repair_success_count"],
            "repair_failed_count": stats["repair_failed_count"],
            "repair_failure_count": stats["repair_failed_count"],
            "repair_failure_by_code": dict(sorted(stats["repair_failure_by_code"].items())),
            "missing_ids": list(stats["missing_ids"]),
            "unknown_ids": list(stats["unknown_ids"]),
            "duplicate_ids": list(stats["duplicate_ids"]),
            "item_salvage_audit": list(stats["item_salvage_audit"]),
            "adaptive_split_used": bool(stats["adaptive_split_count"]),
            "adaptive_split_reason": dict(sorted(stats["adaptive_split_reasons"].items())),
            "prompt_tokens_total": sum(
                int(item.get("prompt_tokens", item.get("input_tokens", 0)) or 0)
                for item in batch_usages
            ),
            "completion_tokens_total": sum(
                int(item.get("completion_tokens", item.get("output_tokens", 0)) or 0)
                for item in batch_usages
            ),
            "leaf_items": stats["leaf_items"],
            "leaf_input_chars": stats["leaf_chars"],
            "leaf_elapsed_seconds": stats["leaf_elapsed"],
            "completion_tokens": stats["completion_tokens"],
            "reasoning_characters": stats["reasoning_characters"],
            "transport_error_counts": stats["transport_error_counts"],
            "transport_failures": 0,
            **{key: stats[key] for key in (
                "json_contract_errors", "format_retry_calls", "format_retry_successes",
                "format_retry_failures", "markdown_fence_normalizations",
            )},
        }

    def _assess_batch_adaptively(
        self,
        malfunction: MalfunctionCandidate,
        scenarios: list[ScenarioCandidate],
        *,
        parent_batch: str,
        split_path: str,
        depth: int,
        stats: dict[str, Any],
        batch_usages: list[dict[str, Any]],
        models: list[str],
        request_ids: list[str],
        project_registry: FactRegistry | None = None,
    ) -> list[ScenarioFeasibilityAssessment]:
        user_prompt = build_scenario_user_prompt(
            malfunction, scenarios, project_registry, self.causal_evidence_budget,
        )
        self._record_evidence_selection_stats(
            malfunction, scenarios, project_registry, stats,
        )
        input_chars = len(self.system_prompt + self.MACHINE_OUTPUT_RULE) + len(user_prompt)
        if input_chars > self.batch_max_chars and len(scenarios) > 1:
            raise ScenarioBatchSizeError(
                "Scenario child batch超过请求预算且已在Provider调用前阻断: "
                f"malfunction_id={malfunction.malfunction_id} batch={parent_batch} "
                f"split_path={split_path} estimated_chars={input_chars} "
                f"max_chars={self.batch_max_chars}"
            )
        if input_chars > self.batch_max_chars:
            print(
                "[HARA] scenario singleton exceeds batch target "
                f"malfunction={malfunction.malfunction_id} batch={parent_batch} "
                f"scenario_id={scenarios[0].scenario_id} "
                f"input_chars={input_chars} target_chars={self.batch_max_chars}",
                file=sys.stderr,
                flush=True,
            )
        print(
            "[HARA] scenario batch start "
            f"malfunction={malfunction.malfunction_id} batch={parent_batch} "
            f"split_path={split_path} split_depth={depth} "
            f"scenario_count={len(scenarios)} input_chars={input_chars}",
            file=sys.stderr,
            flush=True,
        )
        request = LLMRequest(
            task="assess_scenario_feasibility",
            system_prompt=self.system_prompt + "\n" + self.MACHINE_OUTPUT_RULE,
            user_prompt=user_prompt,
            schema_name=self.schema_name,
            prompt_version=self.prompt_version,
            metadata={
                "malfunction_id": malfunction.malfunction_id,
                "parent_batch": parent_batch,
                "split_path": split_path,
                "split_depth": depth,
                "scenario_count": len(scenarios),
                "assessment_contract_version": self.assessment_contract_version,
                "provider_response_constraint": "JSON_OBJECT" if self.response_format else "NONE",
            },
            max_tokens=self.provider_max_tokens,
            response_format=self.response_format,
        )
        batch_started = time.monotonic()
        stats["actual_llm_calls"] += 1
        if split_path == "root":
            stats["initial_batch_provider_calls"] += 1
        try:
            response = self._complete_json(request)
            if len(scenarios) == 2 and '"shared_context"' in user_prompt and not response.usage.get("response_checkpoint_hit"):
                stats["shared_context_batch_count"] += 1
            if response.usage.get("response_checkpoint_hit") and split_path == "root":
                stats["initial_batch_provider_calls"] -= 1
        except (LLMOutputLimitError, LLMTimeoutError) as exc:
            if isinstance(exc, LLMOutputLimitError):
                reason = "output_limit"
                stats["output_limit_count"] += 1
                diagnostics = exc.diagnostics
                if isinstance(diagnostics.get("reasoning_characters"), int):
                    stats["reasoning_characters"].append(diagnostics["reasoning_characters"])
                if isinstance(diagnostics.get("completion_tokens"), (int, float)):
                    stats["completion_tokens"].append(diagnostics["completion_tokens"])
            else:
                reason = "timeout"
                stats["timeout_count"] += 1
                stats["retry_calls"] += max(0, exc.attempts - 1)
            return self._split_or_fail(
                malfunction, scenarios, parent_batch=parent_batch,
                split_path=split_path, depth=depth, reason=reason,
                input_chars=input_chars, error=exc, stats=stats,
                batch_usages=batch_usages, models=models,
                request_ids=request_ids, project_registry=project_registry,
            )
        except (LLMJSONContractError, LLMSchemaContractError) as exc:
            stats["schema_error_count"] += 1
            return self._split_or_fail(
                malfunction, scenarios, parent_batch=parent_batch,
                split_path=split_path, depth=depth, reason="provider_schema",
                input_chars=input_chars, error=exc, stats=stats,
                batch_usages=batch_usages, models=models,
                request_ids=request_ids, project_registry=project_registry,
            )
        self._record_response_diagnostics(
            response, stats, batch_usages, models, request_ids,
        )
        scenario_by_id = {item.scenario_id: item for item in scenarios}
        raw = response.data.get("assessments")
        if isinstance(raw, list):
            return self._process_parseable_batch(
                malfunction, scenarios, raw, response=response,
                parent_batch=parent_batch,
                split_path=split_path, depth=depth, input_chars=input_chars,
                batch_started=batch_started, stats=stats,
                batch_usages=batch_usages, models=models,
                request_ids=request_ids, project_registry=project_registry,
            )
        envelope_error = ScenarioSchemaContractError(
            "LLM output must contain an assessments list"
        )
        stats["schema_error_count"] += 1
        self._record_contract_error(stats, envelope_error)
        return self._split_or_fail(
            malfunction, scenarios, parent_batch=parent_batch,
            split_path=split_path, depth=depth, reason="provider_schema",
            input_chars=input_chars, error=envelope_error, stats=stats,
            batch_usages=batch_usages, models=models,
            request_ids=request_ids, project_registry=project_registry,
        )
        try:
            if not isinstance(raw, list):
                raise ScenarioSchemaContractError("LLM输出缺少assessments数组")
            parsed = [self._parse(
                malfunction, item, scenario=scenario_by_id.get(str(item.get("scenario_id", "")).strip()),
                batch=parent_batch, split_path=split_path, split_depth=depth,
                project_registry=project_registry,
            ) for item in raw]
            self._validate(parsed, [item.scenario_id for item in scenarios])
        except ScenarioCoverageContractError as exc:
            stats["coverage_error_count"] += 1
            if len(scenarios) == 1:
                return self._repair_single_item_contract(
                    malfunction, scenarios[0], parent_batch=parent_batch,
                    split_path=split_path, depth=depth, reason="coverage_contract",
                    error=exc, stats=stats, batch_usages=batch_usages,
                    models=models, request_ids=request_ids,
                    project_registry=project_registry,
                    previous_assessment=self._previous_assessment(raw, scenarios[0]),
                )
            return self._split_or_fail(
                malfunction, scenarios, parent_batch=parent_batch,
                split_path=split_path, depth=depth, reason="coverage_contract",
                input_chars=input_chars, error=exc, stats=stats,
                batch_usages=batch_usages, models=models,
                request_ids=request_ids, project_registry=project_registry,
            )
        except (ScenarioSchemaContractError, ScenarioEvidenceContractError) as exc:
            stats["schema_error_count"] += 1
            self._record_contract_error(stats, exc)
            if len(scenarios) == 1:
                return self._repair_single_item_contract(
                    malfunction, scenarios[0], parent_batch=parent_batch,
                    split_path=split_path, depth=depth, reason="item_schema_contract",
                    error=exc, stats=stats, batch_usages=batch_usages,
                    models=models, request_ids=request_ids,
                    project_registry=project_registry,
                    previous_assessment=self._previous_assessment(raw, scenarios[0]),
                )
            return self._split_or_fail(
                malfunction, scenarios, parent_batch=parent_batch,
                split_path=split_path, depth=depth, reason="item_schema_contract",
                input_chars=input_chars, error=exc, stats=stats,
                batch_usages=batch_usages, models=models,
                request_ids=request_ids, project_registry=project_registry,
            )
        elapsed = time.monotonic() - batch_started
        stats["leaf_items"].append(len(scenarios))
        stats["leaf_chars"].append(input_chars)
        stats["leaf_elapsed"].append(round(elapsed, 3))
        print(
            "[HARA] scenario batch completed "
            f"malfunction={malfunction.malfunction_id} batch={parent_batch} "
            f"split_path={split_path} expected={len(scenarios)} received={len(parsed)} "
            f"elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return parsed

    def _repair_single_item_contract(
        self,
        malfunction: MalfunctionCandidate,
        scenario: ScenarioCandidate,
        *,
        parent_batch: str,
        split_path: str,
        depth: int,
        reason: str,
        error: Exception,
        stats: dict[str, Any],
        batch_usages: list[dict[str, Any]],
        models: list[str],
        request_ids: list[str],
        project_registry: FactRegistry | None,
        previous_assessment: dict[str, Any] | None,
    ) -> list[ScenarioFeasibilityAssessment]:
        stats["item_contract_repair_count"] += 1
        error_code = (
            error.code.value
            if isinstance(error, ScenarioEvidenceContractError)
            else type(error).__name__
        )
        error_detail = {
            "code": error_code,
            "reason": getattr(error, "reason", str(error))[:1200],
            "hop": getattr(error, "hop", ""),
            "invalid_evidence_refs": getattr(error, "invalid_evidence_refs", []),
        }
        targeted_correction = self._single_item_repair_instruction(error)
        allowed_breakpoints = [item.value for item in CausalBreakpoint]
        previous_payload = (
            json.dumps(previous_assessment, ensure_ascii=False, separators=(",", ":"))
            if previous_assessment is not None else "<unavailable>"
        )
        print(
            "[HARA] scenario single-item contract repair "
            f"malfunction={malfunction.malfunction_id} scenario={scenario.scenario_id} "
            f"batch={parent_batch} split_path={split_path} split_depth={depth} "
            f"reason={reason} error_code={error_code} attempt=1/1",
            file=sys.stderr,
            flush=True,
        )
        self._record_evidence_selection_stats(
            malfunction, [scenario], project_registry, stats,
        )
        user_prompt = build_scenario_user_prompt(
            malfunction, [scenario], project_registry, self.causal_evidence_budget,
        ) + (
            "\nSINGLE-ITEM CONTRACT REPAIR (attempt 1/1): Re-evaluate the same supplied "
            "Malfunction and Scenario. The previous result violated the declared contract. "
            "Return exactly one complete assessment for the requested scenario. Do not preserve "
            "an invalid field merely to resemble the previous response. Do not add facts, evidence "
            "references, actors, events, or engineering rules. If evidence cannot support a causal "
            "hop, return causally_relevant=false and identify the first unsupported breakpoint. "
            "Every returned hop object through the breakpoint must have a non-empty claim; an "
            "ASSUMPTION claim must name the unsupported condition. Omit hops after the breakpoint "
            "rather than returning blank objects.\n"
            "M_TO_B boundary: if finalized MF.description and MF.functional_effect anchors are "
            "shown, preserve m_to_b and cite those exact DIRECT_FACT refs. Do not change a valid "
            "m_to_b into an M_TO_B breakpoint because Scenario trigger/applicability context is "
            "absent. A downstream evidence gap must remain at B_TO_I. Do not use "
            "MF.vehicle_level_hazard as the m_to_b anchor.\n"
            f"AllowedBreakpointValues={json.dumps(allowed_breakpoints, separators=(',', ':'))}. "
            "The breakpoint value must exactly equal one of these strings; do not combine or "
            "rename enum values. Preserve valid fields from PreviousAssessment and change only "
            "the invalid field plus fields that must change to maintain cross-field consistency.\n"
            f"TargetedCorrection={targeted_correction}\n"
            f"PreviousContractError={json.dumps(error_detail, ensure_ascii=False, separators=(',', ':'))}\n"
            f"PreviousAssessment={previous_payload}"
        )
        request = LLMRequest(
            task="assess_scenario_feasibility",
            system_prompt=self.system_prompt + "\n" + self.MACHINE_OUTPUT_RULE,
            user_prompt=user_prompt,
            schema_name=self.schema_name,
            prompt_version=self.prompt_version,
            metadata={
                "malfunction_id": malfunction.malfunction_id,
                "parent_batch": parent_batch,
                "split_path": split_path,
                "split_depth": depth,
                "scenario_count": 1,
                "assessment_contract_version": self.assessment_contract_version,
                "provider_response_constraint": "JSON_OBJECT" if self.response_format else "NONE",
                "item_contract_repair": True,
                "item_contract_repair_attempt": 1,
            },
            max_tokens=self.provider_max_tokens,
            response_format=self.response_format,
        )
        repair_started = time.monotonic()
        stats["actual_llm_calls"] += 1
        repair_raw_assessment = None
        repair_parsed_assessment = None
        try:
            response = self._complete_json(request)
        except RECOVERABLE_REPAIR_PROVIDER_ERRORS as repair_error:
            return self._contain_repair_provider_failure(
                malfunction=malfunction,
                scenario=scenario,
                parent_batch=parent_batch,
                split_path=split_path,
                depth=depth,
                reason=reason,
                original_error=error,
                repair_error=repair_error,
                project_registry=project_registry,
                previous_assessment=previous_assessment,
                user_prompt=user_prompt,
                repair_started=repair_started,
                stats=stats,
                request_ids=request_ids,
            )
        except Exception as repair_error:
            # Provider and transport failures are not semantic contract
            # failures. Keep them retryable by the workflow instead of
            # converting them into a synthetic assessment.
            self._write_scenario_contract_debug_dump(
                malfunction=malfunction,
                scenario=scenario,
                error=error,
                project_registry=project_registry,
                provider_assessment=previous_assessment,
                repair_assessment=None,
                repair_parsed_assessment=None,
                repair_error=repair_error,
            )
            raise
        try:
            self._record_response_diagnostics(
                response, stats, batch_usages, models, request_ids,
            )
            raw = response.data.get("assessments")
            if not isinstance(raw, list) or len(raw) != 1:
                raise ScenarioCoverageContractError(
                    "single-item contract repair must return exactly one assessment"
                )
            repair_raw_assessment = raw[0]
            parsed = [self._parse(
                malfunction,
                raw[0],
                scenario=(
                    scenario
                    if str(raw[0].get("scenario_id", "")).strip() == scenario.scenario_id
                    else None
                ),
                batch=parent_batch,
                split_path=split_path + "C",
                split_depth=depth,
                project_registry=project_registry,
            )]
            repair_parsed_assessment = parsed[0].to_dict()
            self._validate(parsed, [scenario.scenario_id])
            self._enforce_repair_breakpoint_boundary(
                previous_assessment=previous_assessment,
                initial_error=error,
                repaired=parsed[0],
            )
        except REPAIR_CONTRACT_ERRORS as repair_error:
            self._record_contract_error(stats, repair_error)
            stats["item_contract_repair_failure_count"] += 1
            stats["repair_failed_count"] += 1
            repair_error_code = self._normalized_error_code(repair_error)
            self._record_repair_failure_code(stats, repair_error_code)
            stats["item_salvage_audit"].append({
                "scenario_id": scenario.scenario_id,
                "outcome": "repair_failed",
                "repair_failed": True,
                "error_stage": "repair_contract_error",
                "provider_request_id": request_ids[-1] if request_ids else "",
                "repair_provider_request_id": "",
                "repair_raw_available": repair_raw_assessment is not None,
                "repair_exception_class": type(repair_error).__name__,
                "repair_error_code": repair_error_code,
                "repair_attempt": 1,
                "raw_assessment": repair_raw_assessment,
                "parsed_assessment": repair_parsed_assessment,
                "error_code": repair_error_code,
                "error": str(repair_error)[:1200],
            })
            stats["single_item_failures"] += 1
            repair_elapsed = time.monotonic() - repair_started
            stats["leaf_items"].append(1)
            stats["leaf_chars"].append(
                len(self.system_prompt + self.MACHINE_OUTPUT_RULE) + len(user_prompt)
            )
            stats["leaf_elapsed"].append(round(repair_elapsed, 3))
            pending = self._pending_contract_failure(
                malfunction,
                scenario,
                reason=reason,
                original_error=error,
                repair_error=repair_error,
            )
            self._write_scenario_contract_debug_dump(
                malfunction=malfunction,
                scenario=scenario,
                error=error,
                project_registry=project_registry,
                provider_assessment=previous_assessment,
                repair_assessment=repair_raw_assessment,
                repair_parsed_assessment=repair_parsed_assessment,
                repair_error=repair_error,
            )
            print(
                "[HARA] scenario single-item contract repair exhausted "
                f"malfunction={malfunction.malfunction_id} "
                f"scenario={scenario.scenario_id} error_code={error_code} "
                f"attempt=1/1 disposition=PENDING elapsed={repair_elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            return [pending]
        except Exception:
            # Unexpected programmer errors are not Provider contract failures.
            # Preserve fail-fast behavior instead of manufacturing PENDING.
            raise
        self._write_scenario_contract_debug_dump(
            malfunction=malfunction,
            scenario=scenario,
            error=error,
            project_registry=project_registry,
            provider_assessment=previous_assessment,
            repair_assessment=repair_raw_assessment,
            repair_parsed_assessment=repair_parsed_assessment,
            repair_error=None,
        )
        stats["repair_success_count"] += 1
        stats["item_salvage_audit"].append({
            "scenario_id": scenario.scenario_id,
            "outcome": "repaired",
            "provider_request_id": request_ids[-1] if request_ids else "",
            "raw_assessment": repair_raw_assessment,
            "parsed_assessment": repair_parsed_assessment,
            "error_code": "",
            "error": "",
        })
        stats["leaf_items"].append(1)
        stats["leaf_chars"].append(
            len(self.system_prompt + self.MACHINE_OUTPUT_RULE) + len(user_prompt)
        )
        repair_elapsed = time.monotonic() - repair_started
        stats["leaf_elapsed"].append(round(repair_elapsed, 3))
        print(
            "[HARA] scenario single-item contract repair completed "
            f"malfunction={malfunction.malfunction_id} scenario={scenario.scenario_id} "
            f"error_code={error_code} attempt=1/1 elapsed={repair_elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return parsed

    def _contain_repair_provider_failure(
        self,
        *,
        malfunction: MalfunctionCandidate,
        scenario: ScenarioCandidate,
        parent_batch: str,
        split_path: str,
        depth: int,
        reason: str,
        original_error: Exception,
        repair_error: Exception,
        project_registry: FactRegistry | None,
        previous_assessment: dict[str, Any] | None,
        user_prompt: str,
        repair_started: float,
        stats: dict[str, Any],
        request_ids: list[str],
    ) -> list[ScenarioFeasibilityAssessment]:
        """Contain a provider machine-output failure at the item boundary."""

        repair_error_code = self._normalized_error_code(repair_error)
        self._record_contract_error(stats, repair_error)
        stats["item_contract_repair_failure_count"] += 1
        stats["repair_failed_count"] += 1
        self._record_repair_failure_code(stats, repair_error_code)
        diagnostics = getattr(repair_error, "diagnostics", {})
        repair_provider_request_id = str(
            diagnostics.get("request_id", "")
        ) if isinstance(diagnostics, dict) else ""
        repair_elapsed = time.monotonic() - repair_started
        stats["item_salvage_audit"].append({
            "scenario_id": scenario.scenario_id,
            "outcome": "repair_failed",
            "repair_failed": True,
            "error_stage": "repair_provider_error",
            "provider_request_id": request_ids[-1] if request_ids else "",
            "repair_provider_request_id": repair_provider_request_id,
            "repair_raw_available": False,
            "repair_exception_class": type(repair_error).__name__,
            "repair_error_code": repair_error_code,
            "repair_attempt": 1,
            "raw_assessment": None,
            "parsed_assessment": None,
            "error_code": repair_error_code,
            "error": str(repair_error)[:1200],
        })
        stats["single_item_failures"] += 1
        stats["leaf_items"].append(1)
        stats["leaf_chars"].append(
            len(self.system_prompt + self.MACHINE_OUTPUT_RULE) + len(user_prompt)
        )
        stats["leaf_elapsed"].append(round(repair_elapsed, 3))
        pending = self._pending_contract_failure(
            malfunction,
            scenario,
            reason=reason,
            original_error=original_error,
            repair_error=repair_error,
        )
        self._write_scenario_contract_debug_dump(
            malfunction=malfunction,
            scenario=scenario,
            error=original_error,
            project_registry=project_registry,
            provider_assessment=previous_assessment,
            repair_assessment=None,
            repair_parsed_assessment=None,
            repair_error=repair_error,
        )
        print(
            "[HARA] scenario single-item contract repair contained "
            f"malfunction={malfunction.malfunction_id} "
            f"scenario={scenario.scenario_id} "
            f"error_code={repair_error_code} exception={type(repair_error).__name__} "
            "repair_raw_available=false disposition=PENDING "
            f"elapsed={repair_elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return [pending]

    def _record_evidence_selection_stats(
        self,
        malfunction: MalfunctionCandidate,
        scenarios: list[ScenarioCandidate],
        project_registry: FactRegistry | None,
        stats: dict[str, Any],
    ) -> None:
        """Record selector telemetry for every prompt emitted by this agent."""

        for scenario in scenarios:
            selection = select_scenario_causal_evidence(
                malfunction, scenario, project_registry, self.causal_evidence_budget,
            )
            stats["candidate_evidence_count"] += selection.candidate_evidence_count
            stats["selected_evidence_count"] += selection.selected_evidence_count
            stats["causal_prompt_evidence_chars"] += selection.compact_chars
            stats["causal_prompt_evidence_tokens"] += selection.compact_tokens
            stats["dropped_evidence_count"] += selection.dropped_evidence_count
            for ref in selection.mandatory_evidence_refs:
                if ref not in stats["mandatory_evidence_refs"]:
                    stats["mandatory_evidence_refs"].append(ref)
            if selection.mandatory_evidence_refs:
                stats["m_to_b_anchor_available"] = all(
                    ref in selection.mandatory_evidence_refs
                    for ref in ("MF.description", "MF.functional_effect")
                )
                for ref in selection.mandatory_evidence_refs:
                    if ref not in stats["m_to_b_anchor_refs"]:
                        stats["m_to_b_anchor_refs"].append(ref)
            for ref in selection.selected_context_evidence_refs:
                if ref not in stats["selected_context_evidence_refs"]:
                    stats["selected_context_evidence_refs"].append(ref)
            stats["total_prompt_evidence_count"] += selection.total_prompt_evidence_count
            stats["evidence_selection_audit"].append({
                "scenario_id": scenario.scenario_id,
                "mandatory_evidence_refs": list(selection.mandatory_evidence_refs),
                "selected_context_evidence_refs": list(selection.selected_context_evidence_refs),
                "total_prompt_evidence_count": selection.total_prompt_evidence_count,
                "candidate_evidence_count": selection.candidate_evidence_count,
                "selected_evidence_count": selection.selected_evidence_count,
                "dropped_evidence_count": selection.dropped_evidence_count,
                "evidence_selection_reason": selection.evidence_selection_reason,
                "m_to_b_anchor_available": all(
                    ref in selection.mandatory_evidence_refs
                    for ref in ("MF.description", "MF.functional_effect")
                ),
                "m_to_b_anchor_refs": list(selection.mandatory_evidence_refs),
                "breakpoint_reason": "",
            })
            reason_counts = stats["evidence_selection_reason"]
            reason_counts[selection.evidence_selection_reason] = (
                reason_counts.get(selection.evidence_selection_reason, 0) + 1
            )

    def _process_parseable_batch(
        self,
        malfunction: MalfunctionCandidate,
        scenarios: list[ScenarioCandidate],
        raw: list[Any],
        *,
        response: Any,
        parent_batch: str,
        split_path: str,
        depth: int,
        input_chars: int,
        batch_started: float,
        stats: dict[str, Any],
        batch_usages: list[dict[str, Any]],
        models: list[str],
        request_ids: list[str],
        project_registry: FactRegistry | None,
    ) -> list[ScenarioFeasibilityAssessment]:
        """Parse a recoverable envelope and repair only invalid expected IDs."""

        parsed, repair_targets = self._salvage_items(
            malfunction, scenarios, raw,
            parent_batch=parent_batch, split_path=split_path, depth=depth,
            project_registry=project_registry, stats=stats,
            provider_request_id=response.request_id,
        )
        for scenario, original, item_error in repair_targets:
            stats["invalid_items_repaired"] += 1
            parsed.extend(self._repair_single_item_contract(
                malfunction, scenario,
                parent_batch=parent_batch,
                split_path=split_path + "I",
                depth=depth,
                reason=(
                    "item_schema_contract" if original is not None
                    else "coverage_contract"
                ),
                error=item_error,
                stats=stats,
                batch_usages=batch_usages,
                models=models,
                request_ids=request_ids,
                project_registry=project_registry,
                previous_assessment=original,
            ))
        try:
            self._validate(parsed, [item.scenario_id for item in scenarios])
        except ScenarioCoverageContractError as exc:
            stats["coverage_error_count"] += 1
            self._record_contract_error(stats, exc)
            return self._split_or_fail(
                malfunction, scenarios,
                parent_batch=parent_batch,
                split_path=split_path,
                depth=depth,
                reason="coverage_contract",
                input_chars=input_chars,
                error=exc,
                stats=stats,
                batch_usages=batch_usages,
                models=models,
                request_ids=request_ids,
                project_registry=project_registry,
            )
        elapsed = time.monotonic() - batch_started
        stats["leaf_items"].append(len(scenarios))
        stats["leaf_chars"].append(input_chars)
        stats["leaf_elapsed"].append(round(elapsed, 3))
        print(
            "[HARA] scenario batch completed "
            f"malfunction={malfunction.malfunction_id} batch={parent_batch} "
            f"split_path={split_path} expected={len(scenarios)} "
            f"received={len(raw)} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return parsed

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value and value not in values:
            values.append(value)

    def _salvage_items(
        self,
        malfunction: MalfunctionCandidate,
        scenarios: list[ScenarioCandidate],
        raw: list[Any],
        *,
        parent_batch: str,
        split_path: str,
        depth: int,
        project_registry: FactRegistry | None,
        stats: dict[str, Any],
        provider_request_id: str,
    ) -> tuple[
        list[ScenarioFeasibilityAssessment],
        list[tuple[ScenarioCandidate, dict[str, Any] | None, Exception]],
    ]:
        expected_by_id = {item.scenario_id: item for item in scenarios}
        valid_by_id: dict[str, ScenarioFeasibilityAssessment] = {}
        invalid_by_id: dict[str, tuple[dict[str, Any] | None, Exception]] = {}

        def audit_item(
            *,
            outcome: str,
            scenario_id: str,
            raw_item: Any,
            parsed_item: ScenarioFeasibilityAssessment | None = None,
            error: Exception | None = None,
        ) -> None:
            entry = {
                "scenario_id": scenario_id,
                "outcome": outcome,
                "error_stage": "initial_item_error" if error is not None else "",
                "repair_failed": False,
                "provider_request_id": provider_request_id,
                "raw_assessment": raw_item,
                "parsed_assessment": (
                    parsed_item.to_dict() if parsed_item is not None else None
                ),
                "error_code": (
                    error.code.value
                    if isinstance(error, ScenarioEvidenceContractError)
                    else type(error).__name__ if error is not None else ""
                ),
                "breakpoint_reason": (
                    getattr(error, "reason", str(error))[:1200]
                    if error is not None
                    else "; ".join(
                        parsed_item.causal_assessment.unsupported_links
                        if parsed_item is not None
                        and parsed_item.causal_assessment is not None
                        else ()
                    )
                ),
                "error": str(error)[:1200] if error is not None else "",
            }
            stats["item_salvage_audit"].append(entry)

        for raw_item in raw:
            if not isinstance(raw_item, dict):
                error = ScenarioSchemaContractError(
                    "Scenario assessment item must be a JSON object"
                )
                stats["invalid_initial_count"] += 1
                stats["schema_error_count"] += 1
                self._record_contract_error(stats, error)
                audit_item(outcome="invalid", scenario_id="", raw_item=raw_item, error=error)
                continue
            scenario_id = str(raw_item.get("scenario_id", "")).strip()
            if not scenario_id:
                error = ScenarioCoverageContractError(
                    "Scenario assessment is missing scenario_id"
                )
                stats["invalid_initial_count"] += 1
                stats["coverage_error_count"] += 1
                self._record_contract_error(stats, error)
                audit_item(outcome="missing_id", scenario_id="", raw_item=raw_item, error=error)
                continue
            if scenario_id not in expected_by_id:
                error = ScenarioCoverageContractError(
                    f"Scenario assessment has unknown scenario_id={scenario_id!r}"
                )
                stats["invalid_initial_count"] += 1
                stats["coverage_error_count"] += 1
                self._record_contract_error(stats, error)
                self._append_unique(stats["unknown_ids"], scenario_id)
                audit_item(outcome="unknown", scenario_id=scenario_id, raw_item=raw_item, error=error)
                continue
            if scenario_id in valid_by_id:
                error = ScenarioCoverageContractError(
                    f"duplicate assessment for scenario_id={scenario_id!r}"
                )
                stats["invalid_initial_count"] += 1
                stats["coverage_error_count"] += 1
                self._record_contract_error(stats, error)
                self._append_unique(stats["duplicate_ids"], scenario_id)
                audit_item(outcome="duplicate", scenario_id=scenario_id, raw_item=raw_item, error=error)
                continue
            try:
                parsed = self._parse(
                    malfunction, raw_item, scenario=expected_by_id[scenario_id],
                    batch=parent_batch, split_path=split_path, split_depth=depth,
                    project_registry=project_registry,
                )
            except REPAIR_CONTRACT_ERRORS as error:
                stats["invalid_initial_count"] += 1
                if isinstance(error, ScenarioEvidenceContractError):
                    self._record_contract_error(stats, error)
                    stats["schema_error_count"] += 1
                else:
                    self._record_contract_error(stats, error)
                    stats["schema_error_count"] += 1
                invalid_by_id.setdefault(scenario_id, (raw_item, error))
                audit_item(outcome="invalid", scenario_id=scenario_id, raw_item=raw_item, error=error)
                continue
            valid_by_id[scenario_id] = parsed
            stats["valid_initial_count"] += 1
            stats["valid_items_salvaged"] += 1
            audit_item(outcome="valid", scenario_id=scenario_id, raw_item=raw_item, parsed_item=parsed)

        repair_targets: list[tuple[ScenarioCandidate, dict[str, Any] | None, Exception]] = []
        for scenario in scenarios:
            if scenario.scenario_id in valid_by_id:
                continue
            if scenario.scenario_id in invalid_by_id:
                original, error = invalid_by_id[scenario.scenario_id]
            else:
                error = ScenarioCoverageContractError(
                    f"missing assessment for scenario_id={scenario.scenario_id!r}"
                )
                stats["coverage_error_count"] += 1
                self._record_contract_error(stats, error)
                self._append_unique(stats["missing_ids"], scenario.scenario_id)
                audit_item(outcome="missing", scenario_id=scenario.scenario_id, raw_item=None, error=error)
                original = None
            repair_targets.append((scenario, original, error))
        return [valid_by_id[item.scenario_id] for item in scenarios if item.scenario_id in valid_by_id], repair_targets

    @staticmethod
    def _record_contract_error(stats: dict[str, Any], error: Exception) -> None:
        code = ScenarioFeasibilityAgent._normalized_error_code(error)
        counts = stats.setdefault("error_counts_by_code", {})
        counts[code] = counts.get(code, 0) + 1

    @staticmethod
    def _record_repair_failure_code(stats: dict[str, Any], code: str) -> None:
        counts = stats.setdefault("repair_failure_by_code", {})
        counts[code] = counts.get(code, 0) + 1

    @staticmethod
    def _enforce_repair_breakpoint_boundary(
        *,
        previous_assessment: dict[str, Any] | None,
        initial_error: Exception,
        repaired: ScenarioFeasibilityAssessment,
    ) -> None:
        """Prevent repair from hiding a downstream failure as M_TO_B."""

        if getattr(initial_error, "hop", "") not in {"b_to_i", "i_to_h"}:
            return
        if not isinstance(previous_assessment, dict):
            return
        previous_chain = previous_assessment.get("causal_chain")
        previous_m_to_b = (
            previous_chain.get("m_to_b")
            if isinstance(previous_chain, dict)
            else None
        )
        if not isinstance(previous_m_to_b, dict):
            return
        previous_refs = previous_m_to_b.get("evidence_refs")
        if (
            not isinstance(previous_refs, list)
            or not previous_refs
            or str(previous_m_to_b.get("basis_type", "")) == EvidenceBasisType.ASSUMPTION.value
            or repaired.breakpoint != CausalBreakpoint.M_TO_B.value
        ):
            return
        raise ScenarioRepairBoundaryError(
            "repair cannot downgrade a previously supported m_to_b hop to M_TO_B; "
            "preserve m_to_b and classify the first downstream unsupported hop"
        )

    @staticmethod
    def _normalized_error_code(error: Exception) -> str:
        if isinstance(error, ScenarioEvidenceContractError):
            return error.code.value
        if isinstance(error, ScenarioRepairBoundaryError):
            return "REPAIR_BREAKPOINT_DOWNGRADE"
        if isinstance(error, ScenarioSemanticConsistencyError):
            return "PHYSICAL_SEMANTIC_CONFLICT"
        return {
            LLMEmptyOutputError: "LLM_EMPTY_OUTPUT_ERROR",
            LLMJSONContractError: "LLM_JSON_CONTRACT_ERROR",
            LLMOutputLimitError: "LLM_OUTPUT_LIMIT_ERROR",
            LLMSchemaContractError: "LLM_SCHEMA_CONTRACT_ERROR",
            ScenarioCoverageContractError: "SCENARIO_COVERAGE_CONTRACT_ERROR",
            ScenarioSchemaContractError: "SCENARIO_SCHEMA_CONTRACT_ERROR",
        }.get(type(error), type(error).__name__)

    @staticmethod
    def _write_scenario_contract_debug_dump(
        *,
        malfunction: MalfunctionCandidate,
        scenario: ScenarioCandidate,
        error: Exception,
        project_registry: FactRegistry | None,
        provider_assessment: Any,
        repair_assessment: Any,
        repair_parsed_assessment: Any,
        repair_error: Exception | None,
    ) -> None:
        """Persist contract-failure evidence only when explicitly enabled."""

        if os.getenv("HARA_SCENARIO_CONTRACT_DEBUG", "").strip() != "1":
            return
        try:
            registry = build_fact_registry(malfunction, scenario, project_registry)
            error_code = ScenarioFeasibilityAgent._normalized_error_code(error)
            payload = {
                "malfunction_id": malfunction.malfunction_id,
                "scenario_id": scenario.scenario_id,
                "semantic_fingerprint": scenario.semantic_fingerprint,
                "failed_hop": getattr(error, "hop", ""),
                "error_code": error_code,
                "claim": getattr(error, "claim", ""),
                "basis_type": getattr(error, "basis_type", ""),
                "invalid_evidence_refs": list(
                    getattr(error, "invalid_evidence_refs", [])
                ),
                **ScenarioFeasibilityAgent._evidence_kind_diagnostics(
                    error, registry,
                ),
                "scenario_fact_registry_snapshot": registry.snapshot(include_values=True),
                "provider_raw_assessment": provider_assessment,
                "single_item_repair_assessment": repair_assessment,
                "single_item_repair_parsed_assessment": repair_parsed_assessment,
                "repair_raw_available": repair_assessment is not None,
                "repair_error": (
                    {
                        "type": type(repair_error).__name__,
                        "message": str(repair_error),
                        "repair_raw_available": repair_assessment is not None,
                        "failed_hop": getattr(repair_error, "hop", ""),
                        "error_code": ScenarioFeasibilityAgent._normalized_error_code(
                            repair_error
                        ),
                        "claim": getattr(repair_error, "claim", ""),
                        "basis_type": getattr(repair_error, "basis_type", ""),
                        "invalid_evidence_refs": list(
                            getattr(repair_error, "invalid_evidence_refs", [])
                        ),
                        "provider_request_id": str(
                            getattr(repair_error, "diagnostics", {}).get(
                                "request_id", ""
                            )
                        ) if isinstance(
                            getattr(repair_error, "diagnostics", {}), dict
                        ) else "",
                    }
                    if repair_error is not None else None
                ),
            }
            directory = Path(os.getenv(
                "HARA_SCENARIO_CONTRACT_DEBUG_DIR",
                "runtime/scenario-contract-debug",
            ))
            directory.mkdir(parents=True, exist_ok=True)
            safe = lambda value: re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(value)
            ).strip("._-") or "unknown"
            path = directory / (
                f"{safe(malfunction.malfunction_id)}__{safe(scenario.scenario_id)}__"
                f"{time.time_ns()}__{safe(error_code)}.json"
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception as dump_error:
            print(
                "[HARA] scenario contract debug dump skipped "
                f"malfunction={malfunction.malfunction_id} "
                f"scenario={scenario.scenario_id} error={dump_error}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _evidence_kind_diagnostics(
        error: Exception, registry: FactRegistry,
    ) -> dict[str, Any]:
        basis_to_expected_kind = {
            "DIRECT_FACT_KIND_MISMATCH": "DIRECT_FACT",
            "DERIVED_PHYSICS_KIND_MISMATCH": "DERIVED_PHYSICS",
            "APPROVED_RULE_KIND_MISMATCH": "APPROVED_RULE",
        }
        error_code = ScenarioFeasibilityAgent._normalized_error_code(error)
        refs = list(getattr(error, "invalid_evidence_refs", []))
        resolved = []
        for ref in refs:
            record = registry.resolve(ref)
            resolved.append({
                "evidence_ref": ref,
                "resolved": record is not None,
                "kind": record.get("kind", "") if record else "",
                "evidence_role": record.get("evidence_role", "") if record else "",
                "source_namespace": ref.split(".", 1)[0] if "." in ref else ref,
            })
        return {
            "validator_expected_kind": basis_to_expected_kind.get(error_code, ""),
            "resolved_evidence": resolved,
            "actual_evidence_kinds": sorted({
                str(item["kind"]) for item in resolved if item["kind"]
            }),
        }

    @staticmethod
    def _single_item_repair_instruction(error: Exception) -> str:
        error_code = (
            error.code.value
            if isinstance(error, ScenarioEvidenceContractError)
            else type(error).__name__
        )
        hop = getattr(error, "hop", "")
        breakpoint = hop.upper() if hop in {
            "m_to_b", "b_to_i", "i_to_h",
        } else ""
        if error_code == "INVALID_CAUSAL_CHAIN_SHAPE":
            return (
                "Return causal_chain as one JSON object keyed only by m_to_b, b_to_i, "
                "and i_to_h. NEVER return causal_chain as an array, string, "
                "number, boolean, or null. Every included hop value must itself be a JSON "
                "object containing claim, basis_type, and evidence_refs."
            )
        if error_code in {"EMPTY_HOP_CLAIM", "REQUIRED_HOP_MISSING"}:
            return (
                "Every hop through the declared breakpoint must exist and contain a non-empty "
                "claim. If the hop is unsupported, use basis_type ASSUMPTION, name the missing "
                "condition in claim, use evidence_refs [], and set causally_relevant=false."
            )
        if error_code in {"INVALID_EVIDENCE_REF_SHAPE", "UNRESOLVED_EVIDENCE_REF"}:
            return (
                "evidence_refs must be a JSON string array containing only exact keys from the "
                "supplied fact_registry. If no allowed key supports a necessary hop, classify "
                "that hop as an ASSUMPTION instead of inventing a reference."
            )
        if error_code == "MISSING_EVIDENCE_REF" and hop == "m_to_b":
            return (
                "m_to_b is the Malfunction-definition transition. If MF.description and "
                "MF.functional_effect are present in the causal_evidence_view, cite both exact "
                "refs with basis_type DIRECT_FACT and preserve m_to_b. Do not lower the breakpoint "
                "to M_TO_B because the Scenario lacks a trigger or applicability detail; if the "
                "downstream interaction evidence is insufficient, preserve valid m_to_b and use "
                "breakpoint=B_TO_I with causally_relevant=false. Do not cite MF.vehicle_level_hazard "
                "as the m_to_b anchor."
            )
        if error_code in {
            "DERIVED_PHYSICS_KIND_MISMATCH", "DIRECT_FACT_KIND_MISMATCH",
            "APPROVED_RULE_KIND_MISMATCH", "MISSING_EVIDENCE_REF",
            "ASSUMPTION_IN_POSITIVE_CHAIN",
            "SELF_REFERENTIAL_CAUSAL_EVIDENCE",
        }:
            return (
                "Do not merely relabel an invalid evidence reference. basis_type must match the "
                "referenced fact_registry record kind. DERIVED_PHYSICS may cite only a record whose "
                "kind is DERIVED_PHYSICS; an environment or scenario label does not prove a concrete "
                "actor, object, collision, or harm. If no registered fact of the required kind "
                f"supports hop {hop!r}, set causally_relevant=false, breakpoint={breakpoint or '<the failed hop>'}, "
                "represent that hop as ASSUMPTION with evidence_refs [], clear risk_dimension_changes, "
                "hazardous_event, and omit later hops."
            )
        if error_code in {"INVALID_BREAKPOINT", "CAUSAL_BREAKPOINT_MISMATCH"}:
            return (
                "Use exactly one AllowedBreakpointValues enum. causally_relevant=true requires "
                "breakpoint=NONE; causally_relevant=false requires the enum for the first unsupported "
                "hop. I_TO_HARM is invalid and must never be returned."
            )
        return (
            "Rebuild the complete assessment against the declared machine schema and supplied "
            "fact_registry; correct the reported contract field without adding new facts."
        )

    @staticmethod
    def _pending_contract_failure(
        malfunction: MalfunctionCandidate,
        scenario: ScenarioCandidate,
        *,
        reason: str,
        original_error: Exception,
        repair_error: Exception,
    ) -> ScenarioFeasibilityAssessment:
        """Preserve one malformed pair as a reviewable unknown, never a result."""

        return ScenarioFeasibilityAssessment(
            malfunction_id=malfunction.malfunction_id,
            scenario_id=scenario.scenario_id,
            physically_feasible=False,
            functionally_relevant=False,
            causally_relevant=False,
            risk_dimensions_changed=[],
            rationale=(
                "LLM scenario contract remained invalid after one bounded repair; "
                f"classification was not accepted (failure_type={reason}, "
                f"original={type(original_error).__name__}, "
                f"repair={type(repair_error).__name__})."
            ),
            hazardous_event="",
            potential_harm="",
            status=ReviewStatus.PENDING,
            confidence=0.0,
            breakpoint="",
            causal_chain={},
            risk_dimension_changes=[],
            evidence_contract_version=SCENARIO_ASSESSMENT_CONTRACT_VERSION,
            causal_assessment=None,
        )

    @staticmethod
    def _previous_assessment(
        raw: Any, scenario: ScenarioCandidate,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, list):
            return None
        return next((
            item for item in raw
            if isinstance(item, dict)
            and str(item.get("scenario_id", "")).strip() == scenario.scenario_id
        ), None)

    @staticmethod
    def _record_response_diagnostics(
        response,
        stats: dict[str, Any],
        batch_usages: list[dict[str, Any]],
        models: list[str],
        request_ids: list[str],
    ) -> None:
        usage = response.usage
        if usage.get("response_checkpoint_hit"):
            stats["actual_llm_calls"] -= 1
            stats["response_checkpoint_hits"] = stats.get("response_checkpoint_hits", 0) + 1
        stats["actual_llm_calls"] += int(usage.get("format_retry_calls", 0))
        for metric in (
            "json_contract_errors", "format_retry_calls", "format_retry_successes",
            "format_retry_failures", "markdown_fence_normalizations",
        ):
            stats[metric] += int(usage.get(metric, 0))
        for category, count in usage.get("transient_error_counts", {}).items():
            stats["transport_error_counts"][category] = (
                stats["transport_error_counts"].get(category, 0) + int(count)
            )
        stats["retry_calls"] += max(0, int(usage.get("transport_attempts", 1)) - 1)
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(completion_tokens, (int, float)):
            stats["completion_tokens"].append(completion_tokens)
        reasoning_chars = usage.get("reasoning_characters")
        if isinstance(reasoning_chars, int):
            stats["reasoning_characters"].append(reasoning_chars)
        batch_usages.append(usage)
        models.append(response.model)
        request_ids.append(response.request_id)

    def _split_or_fail(
        self,
        malfunction: MalfunctionCandidate,
        scenarios: list[ScenarioCandidate],
        *,
        parent_batch: str,
        split_path: str,
        depth: int,
        reason: str,
        input_chars: int,
        error: Exception,
        stats: dict[str, Any],
        batch_usages: list[dict[str, Any]],
        models: list[str],
        request_ids: list[str],
        project_registry: FactRegistry | None,
    ) -> list[ScenarioFeasibilityAssessment]:
        if len(scenarios) == 1 or depth >= self.max_split_depth:
            stats["single_item_failures"] += int(len(scenarios) == 1)
            scenario_id = scenarios[0].scenario_id if len(scenarios) == 1 else "<multiple>"
            raise ScenarioAdaptiveBatchError(
                "Scenario adaptive batch无法继续拆分: "
                f"malfunction_id={malfunction.malfunction_id} scenario_id={scenario_id} "
                f"failure_type={reason} input_chars={input_chars} split_depth={depth}"
            ) from error
        midpoint = len(scenarios) // 2
        left, right = scenarios[:midpoint], scenarios[midpoint:]
        stats["adaptive_split_count"] += 1
        split_reasons = stats.setdefault("adaptive_split_reasons", {})
        split_reasons[reason] = split_reasons.get(reason, 0) + 1
        print(
            "[HARA] scenario batch split "
            f"malfunction={malfunction.malfunction_id} batch={parent_batch} "
            f"split_path={split_path} split_depth={depth} reason={reason} "
            f"scenario_count={len(scenarios)} left_count={len(left)} "
            f"right_count={len(right)} input_chars={input_chars}",
            file=sys.stderr,
            flush=True,
        )
        return self._assess_batch_adaptively(
            malfunction, left, parent_batch=parent_batch,
            split_path="L" if split_path == "root" else split_path + "L",
            depth=depth + 1, stats=stats, batch_usages=batch_usages,
            models=models, request_ids=request_ids,
            project_registry=project_registry,
        ) + self._assess_batch_adaptively(
            malfunction, right, parent_batch=parent_batch,
            split_path="R" if split_path == "root" else split_path + "R",
            depth=depth + 1, stats=stats, batch_usages=batch_usages,
            models=models, request_ids=request_ids,
            project_registry=project_registry,
        )

    @staticmethod
    def _deterministic_rationale(item: dict[str, Any]) -> str:
        """Render presentation text from validated machine-authority fields."""
        chain = item.get("causal_chain", {})
        chain = chain if isinstance(chain, dict) else {}
        claims = [
            str(chain.get(hop, {}).get("claim", "")).strip()
            for hop in ("m_to_b", "b_to_i", "i_to_h")
            if isinstance(chain.get(hop), dict)
            and str(chain.get(hop, {}).get("claim", "")).strip()
        ]
        if item.get("causally_relevant") is True:
            dimensions = [
                str(change.get("dimension", "")).strip()
                for change in item.get("risk_dimension_changes", [])
                if isinstance(change, dict) and str(change.get("dimension", "")).strip()
            ]
            suffix = (
                f" Risk dimensions requiring re-evaluation: {', '.join(dimensions)}."
                if dimensions else " No risk-dimension change is asserted."
            )
            return "Validated structured causal path: " + " -> ".join(claims) + "." + suffix
        breakpoint = str(item.get("breakpoint", "UNSPECIFIED")).strip() or "UNSPECIFIED"
        failed_hop = {
            "M_TO_B": "m_to_b", "B_TO_I": "b_to_i", "I_TO_H": "i_to_h",
        }.get(breakpoint)
        failed_claim = (
            str(chain.get(failed_hop, {}).get("claim", "")).strip()
            if failed_hop and isinstance(chain.get(failed_hop), dict) else ""
        )
        return (
            f"Structured causal validation stopped at {breakpoint}: "
            f"{failed_claim or 'the required transition lacks accepted evidence'}."
        )

    @staticmethod
    def _omit_empty_later_hops(item: dict) -> dict:
        """Treat empty post-breakpoint placeholders as omitted, without editing raw output."""
        hops = ("m_to_b", "b_to_i", "i_to_h", "h_to_harm")
        breakpoint = item.get("breakpoint")
        chain = item.get("causal_chain")
        if (item.get("causally_relevant") is not False
                or not isinstance(breakpoint, str)
                or breakpoint not in {"M_TO_B", "B_TO_I", "I_TO_H"}
                or not isinstance(chain, dict)):
            return item
        later = hops[hops.index(breakpoint.lower()) + 1:]
        empty = {
            name for name in later
            if isinstance((hop := chain.get(name)), dict)
            and set(hop) == {"claim", "basis_type", "evidence_refs"}
            and isinstance(hop["claim"], str) and not hop["claim"].strip()
            and hop["basis_type"] == "ASSUMPTION" and hop["evidence_refs"] == []
        }
        if not empty:
            return item
        return {**item, "causal_chain": {key: value for key, value in chain.items() if key not in empty}}

    @staticmethod
    def _parse(
        malfunction: MalfunctionCandidate,
        item: Any,
        *,
        scenario: ScenarioCandidate | None = None,
        batch: str = "unknown",
        split_path: str = "root",
        split_depth: int = 0,
        project_registry: FactRegistry | None = None,
    ) -> ScenarioFeasibilityAssessment:
        if not isinstance(item, dict):
            raise ScenarioSchemaContractError("Scenario assessment必须为JSON object")
        if any(field in item for field in ("edges", "supports", "mechanism_application")):
            raise ScenarioSchemaContractError(
                "Scenario assessment contains unsupported experimental contract fields"
            )
        if "risk_dimensions_changed" in item:
            raise ScenarioSchemaContractError(
                "risk_dimensions_changed属于v8旧schema；v9必须使用risk_dimension_changes"
            )
        boolean_fields = ("physically_feasible", "functionally_relevant", "causally_relevant")
        if any(not isinstance(item.get(field), bool) for field in boolean_fields):
            raise ScenarioSchemaContractError(
                "physically_feasible、functionally_relevant、causally_relevant必须为JSON boolean"
            )
        scenario_id = str(item.get("scenario_id", "")).strip()
        if scenario is None:
            raise ScenarioSchemaContractError(
                f"Scenario assessment引用未知scenario_id={scenario_id!r}"
            )
        item = ScenarioFeasibilityAgent._omit_empty_later_hops(item)
        causal = item["causally_relevant"]
        hazardous_event = str(item.get("hazardous_event", "")).strip()
        normalized_dimensions, _ = validate_evidence_contract(
            malfunction=malfunction, scenario=scenario, item=item,
            registry=build_fact_registry(malfunction, scenario, project_registry),
            prompt_version=ScenarioFeasibilityAgent.PROMPT_VERSION,
            batch=batch, split_path=split_path, split_depth=split_depth,
        )
        checks = (
            ((not causal and bool(normalized_dimensions)), "risk_dimensions_changed", normalized_dimensions, "[] when causally_relevant=false"),
            ((not causal and bool(hazardous_event)), "hazardous_event", hazardous_event, "empty string when causally_relevant=false"),
            ((causal and not hazardous_event), "hazardous_event", hazardous_event, "non-empty when causally_relevant=true"),
        )
        for violated, field, actual, expected in checks:
            if violated:
                print(
                    "[HARA] scenario invariant failure "
                    f"malfunction={malfunction.malfunction_id} scenario={scenario_id} "
                    f"causal={causal} violating_field={field} actual={actual!r} "
                    f"expected={expected} batch={batch} split_path={split_path} "
                    f"split_depth={split_depth}",
                    file=sys.stderr, flush=True,
                )
                raise ScenarioSchemaContractError(
                    "Scenario cross-field invariant violation: "
                    f"malfunction_id={malfunction.malfunction_id} scenario_id={scenario_id} "
                    f"field={field} causally_relevant={causal} actual={actual!r} "
                    f"expected={expected} batch={batch} split_path={split_path} "
                    f"split_depth={split_depth}"
                )
        interaction = item.get("physical_interaction", {})
        if (not isinstance(interaction, dict) or set(interaction) - {
                "target", "object_position", "object_id", "relation", "assessment_reference"}
                or any(not isinstance(value, str) for value in interaction.values())
                or (interaction and interaction.get("target") not in {"SCENARIO_OBJECT", "OTHER"})
                or (interaction.get("target") == "SCENARIO_OBJECT" and (
                    interaction.get("relation") not in {"APPROACH", "CONTACT", "NOT_CLOSING", "UNKNOWN"}
                    or interaction.get("object_position", "").upper() not in {"FRONT", "REAR", "LEFT", "RIGHT"}
                    or interaction.get("assessment_reference") not in {"CURRENT_STATE", "MODEL_TRAJECTORY", "OTHER"}))):
            raise ScenarioSchemaContractError("Invalid scoped physical_interaction claim")
        if causal:
            conflicts = causal_physics_conflicts(scenario, item)
            if conflicts:
                raise ScenarioSemanticConsistencyError("PHYSICAL_SEMANTIC_CONFLICT: " + ",".join(conflicts))
        causal_assessment = compile_causal_assessment(
            malfunction=malfunction,
            scenario=scenario,
            item=item,
            registry=build_fact_registry(malfunction, scenario, project_registry),
        )
        # Provider labels are ignored.  The typed causal/evidence compiler is
        # the approval gate for both retained and deterministically rejected
        # combinations.
        status = causal_assessment.review_status
        try:
            confidence = parse_confidence(
                item.get("confidence"),
                field_name=(
                    "Scenario confidence validation failed: "
                    f"malfunction={malfunction.malfunction_id} scenario={item.get('scenario_id', '')}"
                ),
            )
        except ValueError as exc:
            raise ScenarioSchemaContractError(str(exc)) from exc
        return ScenarioFeasibilityAssessment(
            malfunction_id=malfunction.malfunction_id,
            scenario_id=scenario_id,
            physically_feasible=item["physically_feasible"],
            functionally_relevant=item["functionally_relevant"],
            causally_relevant=item["causally_relevant"],
            risk_dimensions_changed=[value for value in normalized_dimensions if value],
            rationale=ScenarioFeasibilityAgent._deterministic_rationale(item),
            hazardous_event=hazardous_event,
            potential_harm="",
            status=status,
            confidence=confidence,
            breakpoint=str(item.get("breakpoint", "")),
            causal_chain=dict(item.get("causal_chain", {})),
            risk_dimension_changes=list(item.get("risk_dimension_changes", [])),
            evidence_contract_version=SCENARIO_ASSESSMENT_CONTRACT_VERSION,
            causal_assessment=causal_assessment,
            dependency_metadata=ScenarioFeasibilityAgent._dependencies(
                item, scenario, build_fact_registry(malfunction, scenario, project_registry)),
            physical_interaction=dict(interaction),
        )

    @staticmethod
    def _dependencies(item: dict, scenario: ScenarioCandidate, registry=None) -> dict:
        """Invalid optional reuse metadata never upgrades or rejects causality."""
        value = item.get("dependency_metadata", {})
        if not isinstance(value, dict):
            return {}
        local = value.get("driver_noninterference", {})
        valid_local = {}
        for aspect in ("enablement", "trigger", "interaction", "hazard"):
            record = local.get(aspect, {}) if isinstance(local, dict) else {}
            refs = record.get("evidence_refs", []) if isinstance(record, dict) else []
            if (registry and isinstance(refs, list) and refs and all(isinstance(ref, str)
                    and registry.resolve_record(ref) is not None
                    and registry.resolve_record(ref).approval_status.value == "FINALIZED" for ref in refs)
                    and isinstance(record.get("independent"), bool)
                    and isinstance(record.get("reason"), str) and record["reason"].strip()):
                valid_local[aspect] = {"independent": record["independent"],
                    "evidence_refs": sorted(set(refs)), "reason": record["reason"]}
        extra = {"driver_noninterference": valid_local} if len(valid_local) == 4 else {}
        fields = ("causal_evidence_fields", "physical_feasibility_fields",
                  "scenario_identity_fields", "noninterference_fields")
        allowed = set(scenario.facts) | {"driver_position", "allowed_driver_position", "driver_in_vehicle"}
        if (not isinstance(value, dict) or value.get("complete") is not True
                or not isinstance(value.get("child_subset_refinement"), str)
                or not value["child_subset_refinement"].strip()
                or any(not isinstance(value.get(key), list) or any(
                    not isinstance(field, str) or field not in allowed for field in value[key]
                ) for key in fields)):
            return {"complete": False, **extra} if value else {}
        return {"complete": True, "child_subset_refinement": value["child_subset_refinement"],
                **{key: sorted(set(value[key])) for key in fields}, **extra}

    def _validate(self, assessments: list[ScenarioFeasibilityAssessment], expected_ids: list[str]):
        actual = [item.scenario_id for item in assessments]
        if len(actual) != len(set(actual)):
            raise ValueError("场景评估存在重复项")
        missing = [item for item in expected_ids if item not in actual]
        extra = [item for item in actual if item not in expected_ids]
        if missing or extra:
            raise ScenarioCoverageContractError(
                f"场景评估覆盖不完整: missing={missing}, extra={extra}"
            )
