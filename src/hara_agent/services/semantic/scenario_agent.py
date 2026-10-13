from __future__ import annotations

import json
import os
import re
import sys
import time
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


class ScenarioFeasibilityAgent:
    """Assess each bounded candidate without treating generation as applicability proof."""

    PROMPT_VERSION = "scenario-feasibility-v16-source-scope"
    ALLOWED_RISK_DIMENSIONS = frozenset(RISK_DIMENSION_VALUES)
    MAX_PROVIDER_OUTPUT_TOKENS = 16384
    SYSTEM_PROMPT = """ROLE
你是汽车功能安全HARA工程因果相关性分类器，不是事故故事生成器。输入候选已通过确定性ODD过滤。你的任务是独立判断给定Malfunction在每个明确Scenario中，是否存在由已提供工程事实支持的可信因果链；不是先假设每个Scenario都有危险再寻找解释。负面结论是正常且预期的输出，不得按固定数量保留场景，也不得为了提高覆盖率强行生成Hazard。

STRICT FACT BOUNDARY
只能使用请求中明确提供的：1) Malfunction description、FunctionalEffect与VehicleLevelHazard；2) Scenario facts；3) 请求中明确给出的Item facts；4) 请求中明确给出的approved engineering rules；以及由这些事实直接支持的一阶工程后果。未知事实不得默认存在。不得为了建立危险链新增独立世界状态，包括未提供的特定交通参与者、后车、突然出现的行人或障碍物、天气、湿滑/低附着路面、坡道、驾驶员恐慌或误操作、远程操作员失误、通信/传感器/执行器/转向等新增故障、机械滑移、车辆前窜或侧滑。若这些事实明确存在于输入中才可使用。允许把输入已明确给出的VehicleLevelHazard作为危险状态依据，但不得擅自把泛化的碰撞可能性具体化为某个未提供的对象或事故类型。

CAUSAL CHAIN TEST
仅当以下链条每一跳连续且有输入依据时，causally_relevant才可为true：Malfunction(M) → direct system/vehicle behavior(B) → interaction with explicit Scenario facts(I) → Hazardous Event(H) → direct Potential Harm。M→B必须来自给定失效语义；B→I表示失效行为在当前运行场景、车辆状态、速度或其他明确条件下实际发生或受到实质影响，不要求场景预先列出碰撞对象；I→H不得依赖新发明的initiating event，但输入已给出的VehicleLevelHazard与当前运行条件共同形成危险车辆状态时，该状态本身即可构成Hazardous Event；H→Harm允许一阶、通用的工程伤害推理，但不得增加第二条新故障链或声称存在某个未提供的具体碰撞对象。可信因果可能性不要求事故100%必然发生，也不要求碰撞已经发生；必须有完整、工程上可信且由已知条件支持的路径。physical feasibility、functional relevance与causal relevance必须分别判断；physical=true或functional=true不推出causal=true。

HAZARD VERSUS HARM BOUNDARY
Hazardous Event是“危险车辆状态与运行场景的组合”，不是已经发生的事故或伤害。例如，在明确的Active运行状态和车速下发生非预期横纵向控制、无法退出车辆控制、制动/转向能力丧失或轨迹显著偏离，可在无需假设具体障碍物的情况下形成Hazardous Event。Potential Harm才描述该危险状态可直接导致的通用伤害类型。若输入没有具体对象，只能使用“可能发生碰撞/冲击并造成伤害”等通用表述，不得写成与特定行人、车辆或设施必然碰撞。缺少具体碰撞对象本身不能作为B_TO_I或I_TO_H断裂的唯一理由；只有当VehicleLevelHazard必须依赖某个未提供条件才存在时，才标记ASSUMPTION。

COUNTERFACTUAL TEST
设置causally_relevant=true前，内部检查：禁止加入任何输入未提供的新事件时，仅依靠当前明确事实，危险车辆状态是否仍能由Malfunction在该Scenario中合理产生？不要把“是否已存在具体碰撞对象”误作“危险状态是否存在”。若危险状态本身仍需一个未提供的独立条件才成立，必须返回causally_relevant=false、risk_dimensions_changed=[]、hazardous_event=""，并通过breakpoint及结构化hop说明M→B、B→I或I→H在哪一跳断裂。

RISK DIMENSION TEST
risk_dimensions_changed表示“当前Scenario中哪些明确事实需要下游重新评估风险维度”，不是相对于一个未提供的虚构基准，也不是列出所有涉及或理论上可能相关的维度。每个选择的dimension都必须在risk_dimension_changes中对应一个明确Scenario fact及因果解释；无法指出支持事实就不得选择。禁止blanket selection。因果链成立但当前事实不能证明某个canonical dimension需要变化时允许返回[]；这不会否定因果链，缺失的评分事实由后续评分质量门处理。severity只有在可信H→Harm链成立后才可因明确对象/条件改变；exposure可由明确的运行场景或暴露条件触发重新评估，具体E等级由后续MethodContract计算；controllability可由明确车速、车辆状态、driver position、direct control、remote monitoring或intervention channel等事实触发重新评估，不得假设恐慌；ftti/safe_state只可依据明确时间、距离、速度、干预通道或safe-state reachability事实。

CONTRACT ILLUSTRATIONS
无rear vehicle事实时，“unexpected braking→rear vehicle collision”无效。无driver panic/steering error事实时，“unexpected braking→driver panics→steering error→pedestrian collision”无效。相反，“active状态下的非预期横纵向控制→车辆轨迹不受预期控制”可构成危险车辆状态，通用Potential Harm可描述其可能造成碰撞/冲击伤害，但不得虚构具体碰撞对象；“failure to brake + explicit obstacle ahead + closing speed→distance continues decreasing→collision”则是对象和碰撞机制均明确的完整链。这些仅说明证据规则，不预设当前输入结论。

OUTPUT PRINCIPLES
不要返回rationale。可读说明由Python根据已验证的causal_chain、breakpoint和risk_dimension_changes确定性生成。confidence表示“当前分类判断由给定证据支持”的置信度，不是对生成故事详细程度的信心。不要生成无输入依据的概率、频率或可能性等级。此步骤只筛选候选，不直接计算S/E/C/ASIL。"""

    SYSTEM_PROMPT += """

ATOMIC SCENARIO AND FACT PRECEDENCE
Each Scenario is one atomic, internally consistent world state. Scenario labels and names are descriptive only and MUST NOT override structured facts. Structured explicit facts take precedence over approved derived facts, natural-language summaries, and labels. Do not infer a replacement numeric or state value from a Scenario name; in particular, a label containing "standstill" does not make ego_speed_kph zero when the structured field says otherwise.
source_operating_conditions preserves the source location and road context. permitted_road_types lists allowed alternatives, not simultaneously selected geometry and not proven equivalent conditions. Check whether their differences change the causal mechanism; do not claim all alternatives are covered by one example. selected_method_conditions refines the source world and must not replace or widen it. These context summaries do not add evidence references beyond the supplied fact registry.

BATCH INDEPENDENCE
Each assessment is an independent engineering classification. Do NOT compare scenarios with one another. Do NOT use another Scenario in the same batch as evidence. Do NOT normalize one Scenario relative to another, assume a progression or severity ladder from batch order, or carry facts between Scenarios. For each assessment use only the shared Malfunction facts, that Scenario's own explicit facts, and approved project or method facts. The Malfunction baseline is never another Scenario in the current batch; do not fabricate an unavailable baseline.
"""

    SYSTEM_PROMPT += """

STRUCTURED CAUSAL EVIDENCE CONTRACT
Readable rationale is not part of the Provider contract. Python renders it later from validated structured fields. Return breakpoint, a structured causal_chain with m_to_b, b_to_i, and i_to_h hops, and structured risk_dimension_changes. Every hop contains claim, basis_type, and evidence_refs. Cite only exact keys in that Scenario's read-only fact_registry. A fact's existence does not automatically prove a causal hop: the claim must explain how those cited facts support that exact transition. A causal claim without resolvable evidence cannot support causally_relevant=true.

DIRECT_FACT cites explicit facts. DERIVED_PHYSICS cites only a precomputed DERIVED fact in the registry; raw speed, distance, or actor facts do not license arbitrary vehicle dynamics or human behavior. APPROVED_RULE cites only a registry entry explicitly marked approved. ASSUMPTION identifies a necessary unsupported condition. If any hop needed for a true chain is an ASSUMPTION, return causally_relevant=false at the appropriate breakpoint. Do not reduce confidence to preserve an unsupported chain.

M_TO_B MALFUNCTION ANCHOR BOUNDARY
For m_to_b, use the finalized MF.description and MF.functional_effect evidence records when they are present in the causal_evidence_view. This hop explains what the already-defined Malfunction does; it does not re-evaluate whether the current Scenario satisfies a trigger, operating condition, location, weather, or other applicability context embedded in the Malfunction wording. Missing Scenario applicability context must not be converted into an M_TO_B breakpoint. If the malfunction anchors support M→B but downstream interaction evidence is insufficient, preserve m_to_b and stop at B_TO_I. MF.vehicle_level_hazard is an upstream causal claim and is not a mandatory m_to_b anchor.

For causally_relevant=true, all three hops are complete and breakpoint=NONE. For false, breakpoint identifies the first unsupported transition; do not fabricate negative facts or a completed hazard chain. risk_dimension_changes identifies only dimensions requiring downstream re-evaluation and supplies exact evidence_refs and a reason; do not calculate S/E/C/FTTI levels here. risk_dimensions_changed is not returned because Python derives it from validated risk_dimension_changes.

Every returned causal_chain hop object MUST contain a non-empty claim. At the false-case breakpoint, an ASSUMPTION claim must explicitly name the necessary unsupported condition; never return an empty claim as a placeholder. Omit hop objects after the breakpoint instead of returning blank hop objects.

MF.vehicle_level_hazard is an upstream causal claim, not independent proof of itself. It may provide context, but it MUST NOT be the sole evidence for i_to_h. The transition requires scenario evidence, bounded derived physics, or an approved rule that independently supports the transition. If no such evidence exists, stop at the first unsupported hop.

    causal_chain MUST be a JSON object, never an array, string, or null. Its exact container shape is {"m_to_b":{"claim":"...","basis_type":"...","evidence_refs":[]},"b_to_i":{"claim":"...","basis_type":"...","evidence_refs":[]},"i_to_h":{"claim":"...","basis_type":"...","evidence_refs":[]}}. The downstream harm result is not part of this response. This illustrates structure only; claims and evidence must still be derived solely from the supplied facts. For a false result, include every hop through the declared breakpoint and omit later hops.
"""

    SYSTEM_PROMPT += """

CAUSAL CONTEXT VIEW AND STAGE BOUNDARY
The complete EvidenceRegistry is retained by the deterministic validator but is not injected into this prompt. For each Scenario, use only the compact causal_evidence_view and the fixed Malfunction fields. The view is selected deterministically from P1 physical/interaction facts, P2 intervention/control facts, and P3 operating context; downstream P4 risk metadata is omitted unless explicitly marked causal. A validated analytical Scenario atom fact included in the view is a bounded Method-instantiated condition for that Scenario and may be cited only for that Scenario; it is not a Project Fact. Unvalidated analysis assumptions remain prohibited as positive causal evidence. Return exact evidence_refs from that compact view. This stage ends at Hazardous Event: do not return or calculate a harm result, S/E/C, ASIL, FTTI, or exposure combination.
"""

    MACHINE_OUTPUT_RULE = """MACHINE OUTPUT RULE
Return exactly one JSON object matching the requested schema. Do not use Markdown. Do not wrap the response in ```json or any code fence. Do not add explanatory text before or after the JSON. The first non-whitespace character must be { and the last non-whitespace character must be }. All strings must be valid JSON strings. Do not include comments or trailing commas. The top-level object must remain {"assessments": [...]}.
"""

    SYSTEM_PROMPT += """

CAUSAL DEPENDENCIES FOR SAFE REUSE
Return optional dependency_metadata with complete (boolean), causal_evidence_fields,
physical_feasibility_fields, scenario_identity_fields, noninterference_fields (arrays
of Scenario fact field names), and child_subset_refinement (a reason).
Only declare complete=true if all inputs to feasibility, function applicability and
M→B→I→H are enumerated. Driver position may be noninterfering only if changing it
does not change activation, triggers, operation, interaction or the existence of the
hazard; effects confined to downstream C may be evaluated separately. State the
engineering reason in child_subset_refinement. Absence from evidence_refs alone is
not proof of irrelevance. If uncertain return complete=false. Do not copy conclusions
between scenarios; the deterministic scheduler validates reuse separately.
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
        for batch_index, batch in enumerate(batches, start=1):
            assessments.extend(self._assess_batch_adaptively(
                malfunction, batch,
                parent_batch=f"{batch_index}/{len(batches)}",
                split_path="root",
                depth=0,
                stats=stats,
                batch_usages=batch_usages,
                models=models,
                request_ids=request_ids,
                project_registry=project_registry,
            ))
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
            "candidate_count": len(scenarios),
            "retained_count": sum(item.retain for item in assessments),
            "batch_count": len(batches),
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
                "provider_response_constraint": "NONE",
            },
            max_tokens=self.provider_max_tokens,
        )
        batch_started = time.monotonic()
        stats["actual_llm_calls"] += 1
        if split_path == "root":
            stats["initial_batch_provider_calls"] += 1
        try:
            response = self._complete_json(request)
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
                "provider_response_constraint": "NONE",
                "item_contract_repair": True,
                "item_contract_repair_attempt": 1,
            },
            max_tokens=self.provider_max_tokens,
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
            dependency_metadata=ScenarioFeasibilityAgent._dependencies(item, scenario),
        )

    @staticmethod
    def _dependencies(item: dict, scenario: ScenarioCandidate) -> dict:
        """Invalid optional reuse metadata never upgrades or rejects causality."""
        value = item.get("dependency_metadata", {})
        fields = ("causal_evidence_fields", "physical_feasibility_fields",
                  "scenario_identity_fields", "noninterference_fields")
        allowed = set(scenario.facts) | {"driver_position", "allowed_driver_position", "driver_in_vehicle"}
        if (not isinstance(value, dict) or value.get("complete") is not True
                or not isinstance(value.get("child_subset_refinement"), str)
                or not value["child_subset_refinement"].strip()
                or any(not isinstance(value.get(key), list) or any(
                    not isinstance(field, str) or field not in allowed for field in value[key]
                ) for key in fields)):
            return {}
        return {"complete": True, "child_subset_refinement": value["child_subset_refinement"],
                **{key: sorted(set(value[key])) for key in fields}}

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
