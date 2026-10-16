from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Sequence

from hara_agent.infrastructure.llm import LLMClient, LLMRequest
from hara_agent.services.extraction.project_fact_normalization import (
    ProjectFactNormalizationResult, ProjectFactNormalizer, PHYSICAL_SEMANTICS_EXTRACTION_CONTRACT,
)
from hara_agent.services.extraction.project_fact_specs import RequiredProjectFactSpec

from .item_supplement_agent import RoutedDocumentBlocks


class TargetedProjectFactExtractionAgent:
    """One bounded schema-guided call for one Project Fact category."""

    PROMPT_VERSION = "targeted-project-facts-v5-physical-semantics-atomic-speed-shape"

    def __init__(self, client: LLMClient, normalizer: ProjectFactNormalizer | None = None):
        self.client = client
        self.normalizer = normalizer or ProjectFactNormalizer()

    def extract(
        self,
        category: str,
        specs: Sequence[RequiredProjectFactSpec],
        routed: RoutedDocumentBlocks,
        source_id: str,
    ) -> tuple[ProjectFactNormalizationResult, dict]:
        diagnostics = {
            str(item.get("fact_type", "")): {
                str(block_id) for block_id in item.get("selected_block_ids", [])
            }
            for item in routed.diagnostics.get("facts", [])
            if isinstance(item, dict)
        }
        schema = [{
            "fact_type": item.fact_type,
            "output_type": item.output_type.value,
            "aliases": list(item.aliases),
            "unit_hints": list(item.unit_hints),
            "context_hints": list(item.context_hints),
            "required_fields": list(item.required_fields),
            "constraints": list(item.constraints),
            "structural_kind": item.structural_kind,
            "retrieval_candidate_block_ids": sorted(diagnostics.get(item.fact_type, set())),
        } for item in specs]
        request = LLMRequest(
            task=f"extract_targeted_project_facts:{category}",
            schema_name=f"TargetedProjectFacts:{category}",
            prompt_version=self.PROMPT_VERSION,
            system_prompt=(
                "For SPEED_ENVELOPE, an explicit activation, entry, transition, "
                "or operating condition that names the requested operating mode "
                "and states a speed inequality/range is valid evidence for that "
                "mode. Return FOUND for that atomic bound; do not require a section "
                "whose title says speed envelope. Select only the relevant bound "
                "when the same source block contains additional speed conditions. "
                "A structural speed candidate is FOUND only when its excerpt states "
                "an actual vehicle operating, entry, search, or control speed. "
                "Controller capability, calibration, signed/plus-minus ranges, and "
                "test tolerances are NOT_FOUND. Set semantic_scope to "
                "OPERATIONAL_SPEED for every structurally routed speed result. A "
                "speed spec may have multiple atomic FOUND results for distinct "
                "contextual envelopes, but the structural catch-all must not repeat "
                "a fact already mapped to a mode-specific spec. Every FOUND speed "
                "result must use JSON numbers: LE/LT/EQ has one numeric value and "
                "omits value_max; RANGE has one numeric value and one numeric "
                "value_max in the same result. Never return a range string/list or "
                "separate lower and upper results. For a structural speed spec with "
                "empty context_hints, operating_mode and condition must be concise, "
                "non-empty identifiers grounded only in the explicit source context. "
                "When one source expresses a stationary branch and a moving upper-"
                "bound branch for the same transition, return only the moving upper "
                "bound as the operating envelope; do not emit duplicate scopes. "
                "For RISK_FACT, return exactly one scalar value, the exact Method "
                "Contract unit (or an empty unit), and a string-to-string context object. "
                "For a structurally routed categorical allowed set, emit one atomic "
                "FOUND result per allowed member, add member-specific context, and "
                "set semantic_scope to DRIVER_CONFIGURATION. Never collapse the set "
                "into one string or claim an unscoped global Boolean. "
                "Constraints are validation boundaries, never expected answers. "
                "你是来源约束的原子项目事实抽取器。只可使用提供的带ID证据块，不得猜测、换算或补充常识。"
                "每个FOUND结果只能表达一个原子事实；数值必须是JSON number，运算符只能是LT/LE/EQ/GE/GT/RANGE。"
                "source_block_id必须逐字使用给定ID，source_excerpt若提供必须是该块原文的连续精确子串。"
                "相邻表头或邻行仅用于理解上下文；source_block_id必须取当前spec的retrieval_candidate_block_ids之一，"
                "非结构候选的正文必须包含当前spec的任一alias；不得选择support-only块。"
                + PHYSICAL_SEMANTICS_EXTRACTION_CONTRACT
            ),
            user_prompt=(
                "For mode-specific speed specs, copy the first context_hint into "
                "operating_mode and condition. For a structural speed spec whose "
                "context_hints are empty, derive both non-empty identifiers only "
                "from the explicit source context. "
                "严格返回JSON对象 {\"results\":[...]}。每个spec至少返回一项，status只能是FOUND或NOT_FOUND；"
                "仅在存在多个独立原子事实时可为同一spec返回多个FOUND，NOT_FOUND必须单独一项。"
                "FOUND需返回该spec的required_fields及fact_type/status；NOT_FOUND只返回fact_type/status。"
                "parameter使用aliases中的第一个英文标识；operating_mode和condition使用context_hints中的第一个英文标识；"
                "源文未明确control_mode或condition时返回空字符串，不要因此把明确的driver_location判为NOT_FOUND。"
                "不得把多个数值或多个驾驶员位置合并到同一项，不得返回解释或Markdown。\n"
                "specs=" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                + "\nevidence_blocks=\n" + routed.text
            ),
            metadata={"source_id": source_id, "block_ids": routed.block_ids, "category": category},
            max_tokens=min(int(getattr(getattr(self.client, "config", None), "max_tokens", 4096)), 4096),
        )
        started = time.monotonic()
        response = self.client.complete_json(request)
        normalized = self.normalizer.normalize(
            specs, response.data.get("results"), routed.source_blocks, source_id,
            diagnostics,
        )
        audit = {
            "task": request.task,
            "category": category,
            "prompt_version": request.prompt_version,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "selected_block_ids": routed.block_ids,
            "selected_source_blocks": list(routed.source_blocks),
            "routing_diagnostics": dict(routed.diagnostics),
            "raw_results": response.data.get("results"),
            "coverage": normalized.coverage,
            "normalization_failures": [asdict(item) for item in normalized.failures],
            "llm_call_count": 1,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return normalized, audit
