from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any

from hara_agent.infrastructure.llm import LLMClient, LLMOutputLimitError, LLMRequest
from hara_agent.services.extraction import (
    DEFAULT_CONTEXT_CHARACTER_BUDGET,
    CoverageFirstContextAssembler,
    DeterministicEvidenceRetriever,
    FactRetrievalSpec,
    RoutingDiagnostics,
)


@dataclass(frozen=True)
class RoutedDocumentBlocks:
    task: str
    block_ids: list[str]
    text: str
    source_blocks: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRoutingResult:
    routed: RoutedDocumentBlocks | None
    diagnostics: RoutingDiagnostics


class ItemEvidenceRouter:
    """Route blocks by schema semantics; routing never asserts an engineering fact."""

    ROUTES = {
        "odd_repair": (
            "odd", "运行", "模式", "速度", "km/h", "位置", "道路", "路面", "天气",
            "场地", "坡度", "operating", "speed", "location", "road", "weather",
        ),
    }

    def route(
        self,
        blocks: list[dict[str, Any]],
        task: str,
        max_characters: int = DEFAULT_CONTEXT_CHARACTER_BUDGET,
        required_specs: tuple[FactRetrievalSpec, ...] | None = None,
    ) -> RoutedDocumentBlocks | None:
        return self.retrieve(
            blocks,
            task,
            max_characters=max_characters,
            required_specs=required_specs,
        ).routed

    def retrieve(
        self,
        blocks: list[dict[str, Any]],
        task: str,
        max_characters: int = DEFAULT_CONTEXT_CHARACTER_BUDGET,
        required_specs: tuple[FactRetrievalSpec, ...] | None = None,
    ) -> EvidenceRoutingResult:
        if task == "project_evidence":
            if not required_specs:
                raise ValueError(
                    "project_evidence routing requires compiled fact specifications"
                )
            specs = required_specs
            rankings = DeterministicEvidenceRetriever().rank(blocks, specs)
            assembly = CoverageFirstContextAssembler().assemble(
                blocks,
                specs,
                rankings,
                task=task,
                max_characters=max_characters,
            )
            routed = RoutedDocumentBlocks(
                task=task,
                block_ids=list(assembly.block_ids),
                text=assembly.text,
                source_blocks=list(assembly.source_blocks),
                diagnostics=assembly.diagnostics.to_dict(),
            ) if assembly.block_ids else None
            return EvidenceRoutingResult(routed, assembly.diagnostics)
        if task != "odd_repair":
            raise ValueError(f"unknown Item Evidence routing task: {task}")
        return self._route_odd_blocks(blocks, task, max_characters)

    def _route_odd_blocks(
        self,
        blocks: list[dict[str, Any]],
        task: str,
        max_characters: int,
    ) -> EvidenceRoutingResult:
        keywords = self.ROUTES[task]
        matched = set()
        for index, block in enumerate(blocks):
            lowered = str(block.get("text", "")).lower()
            if any(keyword.lower() in lowered for keyword in keywords):
                matched.update({max(0, index - 1), index, min(len(blocks) - 1, index + 1)})
        selected, total = [], 0
        for index in sorted(matched):
            block = blocks[index]
            line = f"[{block.get('block_id')}] {block.get('location')}: {block.get('text')}"
            cost = len(line) + (1 if selected else 0)
            if total + cost > max_characters:
                break
            selected.append((str(block.get("block_id", "")), line))
            total += cost
        source_blocks = [{
            "block_id": str(blocks[index].get("block_id", "")),
            "kind": str(blocks[index].get("kind", "")),
            "location": str(blocks[index].get("location", "")),
            "text": str(blocks[index].get("text", "")),
            "section_path": list(blocks[index].get("section_path", [])),
        } for index in sorted(matched) if str(blocks[index].get("block_id", "")) in {
            item[0] for item in selected
        }]
        diagnostics = RoutingDiagnostics(
            task=task,
            max_characters=max_characters,
            selected_characters=len("\n".join(item[1] for item in selected)),
            selected_block_ids=tuple(item[0] for item in selected),
            facts=(),
        )
        routed = RoutedDocumentBlocks(
            task=task,
            block_ids=[item[0] for item in selected],
            text="\n".join(item[1] for item in selected),
            source_blocks=source_blocks,
            diagnostics=diagnostics.to_dict(),
        ) if selected else None
        return EvidenceRoutingResult(routed, diagnostics)


class ItemSupplementAgent:
    PROMPT_VERSION = "item-supplement-v3"

    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, routed: RoutedDocumentBlocks, source_id: str) -> tuple[dict[str, Any], dict]:
        if routed.task == "odd_repair":
            fields = (
                "operating_modes和odd；odd包含locations、road_types、weather_conditions、"
                "road_surfaces、speed_range_kph([min,max]或null)"
            )
        else:
            raise ValueError(f"未知局部抽取任务: {routed.task}")
        configured_limit = int(
            getattr(getattr(self.client, "config", None), "max_tokens", 32768)
        )
        requested_budget = int(os.getenv("HARA_ITEM_SUPPLEMENT_MAX_TOKENS", "4096"))
        if requested_budget <= 0:
            raise ValueError("HARA_ITEM_SUPPLEMENT_MAX_TOKENS必须大于0")
        initial_budget = min(requested_budget, configured_limit)
        request = LLMRequest(
            task=f"supplement_{routed.task}",
            system_prompt=(
                "你是Item Definition局部证据抽取Agent。只使用给定的带ID文档块，"
                "不得扩展到常识；没有证据的字段返回空数组或null。"
            ),
            user_prompt=(
                f"返回JSON对象，字段为{fields}。每个source_excerpt最多120字符，"
                "不得输出解释、Markdown或额外字段。"
                + "\nsource_id=" + source_id + "\n" + routed.text
            ),
            schema_name=f"ItemSupplement:{routed.task}",
            prompt_version=self.PROMPT_VERSION,
            metadata={"source_id": source_id, "block_ids": routed.block_ids},
            max_tokens=initial_budget,
        )
        started = time.monotonic()
        print(
            "[HARA] supplement start "
            f"task={routed.task} blocks={len(routed.block_ids)} "
            f"input_chars={len(request.system_prompt) + len(request.user_prompt)} "
            f"max_tokens={initial_budget}",
            file=sys.stderr,
            flush=True,
        )
        output_limit_retry = False
        llm_call_count = 1
        try:
            response = self.client.complete_json(request)
        except LLMOutputLimitError:
            first_attempt_elapsed = time.monotonic() - started
            retry_budget = min(configured_limit, max(initial_budget * 2, 8192))
            if retry_budget <= initial_budget:
                print(
                    "[HARA] supplement failed "
                    f"task={routed.task} elapsed={first_attempt_elapsed:.1f}s "
                    f"type=LLMOutputLimitError max_tokens={initial_budget}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            print(
                "[HARA] supplement output limit "
                f"task={routed.task} old_max_tokens={initial_budget} "
                f"new_max_tokens={retry_budget} "
                f"first_attempt_elapsed={first_attempt_elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            request = replace(request, max_tokens=retry_budget)
            output_limit_retry = True
            llm_call_count += 1
            response = self.client.complete_json(request)
        data = response.data
        elapsed_seconds = time.monotonic() - started
        print(
            "[HARA] supplement completed "
            f"task={routed.task} elapsed={elapsed_seconds:.1f}s "
            f"llm_calls={llm_call_count}",
            file=sys.stderr,
            flush=True,
        )
        return data, {
            "task": request.task,
            "prompt_version": request.prompt_version,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "selected_block_ids": routed.block_ids,
            "selected_source_blocks": list(routed.source_blocks or []),
            "routing_diagnostics": dict(routed.diagnostics or {}),
            "max_tokens": request.max_tokens,
            "output_limit_retry": output_limit_retry,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "llm_call_count": llm_call_count,
            "block_count": len(routed.block_ids),
            "input_characters": len(request.system_prompt) + len(request.user_prompt),
        }
