from __future__ import annotations

from dataclasses import replace
import json
import os
import re
import sys
import time
from typing import Any, Callable, Sequence

from hara_agent.infrastructure.llm import (
    LLMClient, LLMOutputLimitError, LLMRequest, LLMResponse,
)
from hara_agent.models import FunctionDefinition, ItemDefinitionFacts, ReviewStatus
from hara_agent.services.extraction import block_value, normalize_source_text
from hara_agent.services.validation import FunctionValidator

from .core_item_artifact_contract import (
    CORE_ITEM_ARTIFACTS_SCHEMA,
    CoreItemArtifactsContractError,
    normalize_core_item_artifacts,
)
from .function_agent import FunctionNormalizer
from .function_source_guard import validate_function_source_parity
from .item_definition_agent import ItemDefinitionNormalizer
from .parsing import REPORT_LANGUAGE_PROMPT, CONFIDENCE_PROMPT_CONTRACT


class ItemArtifactExtractionAgent:
    """Extract core Item facts and Functions once from the full document."""

    PROMPT_VERSION = "item-artifacts-v8-multilingual-function-scope-zh"
    SYSTEM_PROMPT = REPORT_LANGUAGE_PROMPT + """你是汽车功能安全HARA的相关项定义抽取Agent。只提取文档明确陈述的事实。
必须综合全文，不得把某个固定章节、标题编号或模板工作流中的章节示例当成唯一来源。
一次返回核心Item Definition和车辆级Functions。不得把标题、条件、步骤、后果或质量要求识别为Function；
不得用常识补写ODD、速度或功能。核心输出需保留功能的前置条件、触发、ODD约束、回退行为和后果；
复杂数值性能、驾驶员位置/控制通道及Exposure证据由MethodContract驱动的后续有界抽取处理，不得丢弃、猜测或从模板示例补写。
所有结论必须有可定位来源；证据不足时返回null或空数组并标记PENDING。"""

    FUNCTION_SOURCE_PROMPT = """Function truth-source rule:
- If the Item Definition contains one explicit table or numbered list headed
  '整车功能', 'vehicle-level function(s)', or an equivalent supported heading,
  that explicit source is the only membership authority for functions[].
- Extract every function from that source in source order. Do not add, omit,
  merge, split, or replace its members.
- For each Function, source_location and source_excerpt must identify its exact
  source table row/list item. The excerpt must be one continuous verbatim
  substring and must not concatenate evidence from other sections.
- Operating states, workflow steps, preconditions, parking-in/parking-out phases,
  and HMI interaction flows are not Functions.
- Other document content may enrich description, preconditions, triggers,
  odd_constraints, fallback_behavior, and consequences of an existing Function,
  but it must not change the Function member set.
- Only when no unique explicit Function table/list exists may functions[] be
  extracted from the full document under the existing rules.
- A profile's singular product/system/function name in a metadata form is not
  an enumerated vehicle-function list. In that case identify the supported
  functional actions and their outputs from the body; do not reduce the whole
  document to its product name, or treat headings/version history as functions.
- Except for an authoritative list requiring exact names, write the extracted
  semantic fields in Simplified Chinese; keep source excerpts verbatim."""

    def __init__(self, client: LLMClient, validator: FunctionValidator | None = None):
        self.client = client
        self.validator = validator or FunctionValidator()

    @staticmethod
    def _without_source_references(value: Any) -> Any:
        """Keep a source-only repair from changing extracted engineering facts."""

        if isinstance(value, dict):
            return {
                key: ItemArtifactExtractionAgent._without_source_references(item)
                for key, item in value.items()
                if key not in {"source_location", "source_excerpt"}
            }
        if isinstance(value, list):
            return [
                ItemArtifactExtractionAgent._without_source_references(item)
                for item in value
            ]
        return value

    @staticmethod
    def _excerpt_is_present(excerpt: str, document_text: str) -> bool:
        """Match source text across harmless DOCX/JSON formatting changes only.

        DOCX table cells and paragraphs introduce whitespace boundaries that an
        LLM JSON response may serialize as a regular space (or omit between CJK
        characters).  NFKC also reconciles full-width compatibility characters.
        No punctuation is discarded and no fuzzy similarity is accepted, so a
        paraphrase or an ellipsis-joined citation still fails closed.
        """

        if excerpt in document_text:
            return True
        normalized_excerpt = normalize_source_text(excerpt)
        normalized_document = normalize_source_text(document_text)
        if not normalized_excerpt:
            return False
        if normalized_excerpt in normalized_document:
            return True
        compact_excerpt = "".join(normalized_excerpt.split())
        compact_document = "".join(normalized_document.split())
        return bool(compact_excerpt) and compact_excerpt in compact_document

    @classmethod
    def _split_grounded_source(
        cls,
        source: Any,
        source_blocks: Sequence[Any],
    ) -> list[Any]:
        """Recover independently grounded sentences from a joined citation.

        Providers occasionally concatenate two verbatim sentences from
        different Item blocks.  The combined string is not a valid citation,
        but each sentence can be rebound deterministically to its real block.
        This is deliberately limited to explicit sentence boundaries; ellipsis
        joins and partial/fuzzy matches are not repaired.
        """

        fragments = [
            value.strip()
            for value in re.findall(r".+?(?:[。！？!?；;]+|$)", source.excerpt, re.DOTALL)
            if value.strip()
        ]
        if len(fragments) < 2:
            return []

        repaired = []
        for fragment in fragments:
            candidates = []
            for block in source_blocks:
                text = block_value(block, "text")
                if cls._excerpt_is_present(fragment, text):
                    candidates.append(block)
            if len(candidates) == 1:
                block = candidates[0]
                block_text = block_value(block, "text")
                exact_excerpt = fragment if fragment in block_text else block_text
                repaired.append(type(source)(
                    source_type=source.source_type,
                    source_id=source.source_id,
                    location=block_value(block, "location"),
                    excerpt=exact_excerpt,
                ))
                continue
            if candidates:
                return []
            fragment_source = type(source)(
                source_type=source.source_type,
                source_id=source.source_id,
                location=source.location,
                excerpt=fragment,
            )
            ordered = cls._ordered_omission_source(
                fragment_source, source_blocks,
            )
            if not ordered:
                ordered = cls._boundary_punctuation_omission_source(
                    fragment_source, source_blocks,
                )
            if len(ordered) != 1:
                return []
            repaired.extend(ordered)

        unique = []
        seen = set()
        for value in repaired:
            identity = (value.source_id, value.location, value.excerpt)
            if identity not in seen:
                seen.add(identity)
                unique.append(value)
        return unique

    @classmethod
    def _ordered_omission_source(
        cls,
        source: Any,
        source_blocks: Sequence[Any],
        *,
        require_unique: bool = True,
    ) -> list[Any]:
        """Bind a provider-shortened quote to one unambiguous source block.

        Every non-whitespace character in the proposed excerpt must occur in
        source order.  Strong exact anchors at both ends plus density limits
        prevent this from becoming fuzzy semantic matching.  The returned
        citation is the complete original block, never the shortened text.
        """

        excerpt = "".join(normalize_source_text(source.excerpt).split())
        if len(excerpt) < 16:
            return []

        candidates = []
        for block in source_blocks:
            block_text = block_value(block, "text")
            normalized_block = "".join(normalize_source_text(block_text).split())
            positions = []
            cursor = 0
            for character in excerpt:
                position = normalized_block.find(character, cursor)
                if position < 0:
                    positions = []
                    break
                positions.append(position)
                cursor = position + 1
            if not positions:
                continue
            span = positions[-1] - positions[0] + 1
            span_density = len(excerpt) / span
            block_coverage = len(excerpt) / max(1, len(normalized_block))
            prefix_anchor = 0
            for length in range(1, len(excerpt) + 1):
                if excerpt[:length] not in normalized_block:
                    break
                prefix_anchor = length
            suffix_anchor = 0
            for length in range(1, len(excerpt) + 1):
                if excerpt[-length:] not in normalized_block:
                    break
                suffix_anchor = length
            max_gap = max(
                (right - left - 1 for left, right in zip(positions, positions[1:])),
                default=0,
            )
            anchors_are_strong = (
                prefix_anchor >= 6
                and suffix_anchor >= 4
                and prefix_anchor + suffix_anchor >= 14
            )
            uniquely_bounded_one_sided_anchor = (
                require_unique
                and prefix_anchor >= 10
                and suffix_anchor >= 2
                and prefix_anchor + suffix_anchor >= 12
            )
            minimum_block_coverage = 0.28 if require_unique else 0.30
            if (
                not (anchors_are_strong or uniquely_bounded_one_sided_anchor)
                or span_density < 0.45
                or block_coverage < minimum_block_coverage
                or max_gap > 40
            ):
                continue
            candidates.append((
                prefix_anchor + suffix_anchor,
                span_density,
                block_coverage,
                block,
            ))

        if not candidates:
            return []
        if require_unique and len(candidates) != 1:
            return []
        candidates.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
        if len(candidates) > 1 and (
            candidates[0][0] == candidates[1][0]
            and abs(candidates[0][1] - candidates[1][1]) < 0.02
            and abs(candidates[0][2] - candidates[1][2]) < 0.02
        ):
            return []
        block = candidates[0][3]
        return [type(source)(
            source_type=source.source_type,
            source_id=source.source_id,
            location=block_value(block, "location"),
            excerpt=block_value(block, "text"),
        )]

    @classmethod
    def _boundary_punctuation_omission_source(
        cls,
        source: Any,
        source_blocks: Sequence[Any],
    ) -> list[Any]:
        """Combine an existing bounded omission with terminal punctuation repair.

        The only additional tolerance is removal of terminal punctuation.  The
        existing ordered-character, anchor, density, coverage, and gap checks
        still apply, and this variant requires exactly one qualifying block.
        """

        stripped_excerpt = source.excerpt.rstrip().rstrip("。！？!?；;：:，,、")
        if stripped_excerpt == source.excerpt.rstrip():
            return []
        stripped_source = type(source)(
            source_type=source.source_type,
            source_id=source.source_id,
            location=source.location,
            excerpt=stripped_excerpt,
        )
        return cls._ordered_omission_source(
            stripped_source,
            source_blocks,
            require_unique=True,
        )

    @classmethod
    def _unique_contiguous_anchor_source(
        cls,
        source: Any,
        source_blocks: Sequence[Any],
    ) -> list[Any]:
        """Rebind a boundary-punctuation variant to one exact source block.

        This repair is intentionally narrower than fuzzy matching.  It removes
        only terminal punctuation from the proposed citation and then requires
        the remaining, sufficiently long text to occur contiguously exactly
        once across all source blocks.  The complete original block is returned
        so model-generated punctuation and location text never become evidence.
        """

        normalized_excerpt = "".join(
            normalize_source_text(source.excerpt).split()
        )
        anchor = normalized_excerpt.rstrip("。！？!?；;：:，,、")
        if anchor == normalized_excerpt or len(anchor) < 12:
            return []

        matches: list[Any] = []
        match_count = 0
        for block in source_blocks:
            block_text = block_value(block, "text")
            normalized_block = "".join(normalize_source_text(block_text).split())
            occurrences = normalized_block.count(anchor)
            if occurrences:
                match_count += occurrences
                matches.append(block)

        if match_count != 1 or len(matches) != 1:
            return []
        block = matches[0]
        return [type(source)(
            source_type=source.source_type,
            source_id=source.source_id,
            location=block_value(block, "location"),
            excerpt=block_value(block, "text"),
        )]

    @classmethod
    def _explicit_ellipsis_range_source(
        cls,
        source: Any,
        source_blocks: Sequence[Any],
    ) -> list[Any]:
        """Expand an explicitly abbreviated multi-block citation.

        The text before and after an ellipsis must provide exact, sufficiently
        long boundary anchors in an unambiguous, short, forward block range.
        Every block in that range is then cited verbatim.  This supports Item
        lists split by DOCX paragraph boundaries without treating the model's
        abbreviated text as evidence.
        """

        parts = re.split(r"(?:\.{3,}|…+)", source.excerpt)
        if len(parts) != 2:
            return []
        prefix, suffix = (value.strip() for value in parts)
        leading_match = re.match(r".+?(?:[：:。！？!?；;])", prefix, re.DOTALL)
        leading_anchor = leading_match.group(0).strip() if leading_match else prefix
        trailing_fragments = [
            value.strip()
            for value in re.findall(r".+?(?:[。！？!?；;]+|$)", suffix, re.DOTALL)
            if value.strip()
        ]
        trailing_anchor = trailing_fragments[-1] if trailing_fragments else suffix
        if len(normalize_source_text(leading_anchor)) < 8:
            return []
        if len(normalize_source_text(trailing_anchor)) < 8:
            return []

        starts = [
            index for index, block in enumerate(source_blocks)
            if cls._excerpt_is_present(leading_anchor, block_value(block, "text"))
        ]
        ends = [
            index for index, block in enumerate(source_blocks)
            if cls._excerpt_is_present(trailing_anchor, block_value(block, "text"))
        ]
        ranges = [
            (start, end)
            for start in starts
            for end in ends
            if start < end and end - start + 1 <= 20
        ]
        if len(ranges) != 1:
            return []
        start, end = ranges[0]
        selected = source_blocks[start:end + 1]
        kinds = {block_value(block, "kind") for block in selected}
        if len(kinds) != 1:
            return []
        return [type(source)(
            source_type=source.source_type,
            source_id=source.source_id,
            location=block_value(block, "location"),
            excerpt=block_value(block, "text"),
        ) for block in selected if block_value(block, "text").strip()]

    @staticmethod
    def _ensure_source_grounded(
        facts: ItemDefinitionFacts,
        functions: list[FunctionDefinition],
        document_text: str,
        source_blocks: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        """Require exact evidence from the approved Item Definition input."""

        records = [("item_definition", facts)] + [
            (function.function_id, function) for function in functions
        ]
        repairs: list[dict[str, Any]] = []
        for identity, record in records:
            sources = record.sources
            if not sources:
                raise ValueError(f"{identity} missing Item Definition SourceRef")
            grounded_sources = []
            for source in sources:
                excerpt = source.excerpt.strip()
                if source.source_type != "item_definition":
                    raise ValueError(f"{identity} has non-Item source authority")
                if not source.location.strip() or not excerpt:
                    raise ValueError(f"{identity} has incomplete Item Definition SourceRef")
                if ItemArtifactExtractionAgent._excerpt_is_present(
                    excerpt, document_text,
                ):
                    grounded_sources.append(source)
                    continue
                repaired = ItemArtifactExtractionAgent._split_grounded_source(
                    source, source_blocks,
                )
                repair_kind = "sentence_split"
                if not repaired:
                    repaired = ItemArtifactExtractionAgent._ordered_omission_source(
                        source, source_blocks,
                    )
                    repair_kind = "ordered_omission"
                if not repaired:
                    repaired = (
                        ItemArtifactExtractionAgent._unique_contiguous_anchor_source(
                            source, source_blocks,
                        )
                    )
                    repair_kind = "unique_contiguous_anchor_rebind"
                if not repaired:
                    repaired = (
                        ItemArtifactExtractionAgent._boundary_punctuation_omission_source(
                            source, source_blocks,
                        )
                    )
                    repair_kind = "bounded_omission_boundary_punctuation"
                if not repaired:
                    repaired = ItemArtifactExtractionAgent._explicit_ellipsis_range_source(
                        source, source_blocks,
                    )
                    repair_kind = "explicit_ellipsis_range"
                if repaired:
                    grounded_sources.extend(repaired)
                    repairs.append({
                        "identity": identity,
                        "repair_kind": repair_kind,
                        "original_location": source.location,
                        "repaired_locations": [value.location for value in repaired],
                        "source_count": len(repaired),
                    })
                    continue
                raise ValueError(
                    f"{identity} source excerpt is not present in Item Definition; "
                    f"location={source.location!r} excerpt={excerpt[:160]!r}"
                )
            record.sources = grounded_sources
        return repairs

    def extract(
        self,
        document_text: str,
        source_id: str,
        source_blocks: Sequence[Any] = (),
        candidate_response: LLMResponse | None = None,
        candidate_recorder: Callable[[LLMResponse], None] | None = None,
    ) -> tuple[ItemDefinitionFacts, list[FunctionDefinition], dict[str, Any]]:
        if not document_text.strip():
            raise ValueError("Item Definition文本为空")
        configured_limit = int(getattr(getattr(self.client, "config", None), "max_tokens", 32768))
        initial_budget = min(
            int(os.getenv("HARA_ITEM_ARTIFACT_MAX_TOKENS", "16384")),
            configured_limit,
        )
        if initial_budget <= 0 or configured_limit <= 0:
            raise ValueError("Item Artifact token预算必须大于0")
        request = LLMRequest(
            task="extract_core_item_artifacts",
            system_prompt=self.SYSTEM_PROMPT + "\n\n" + self.FUNCTION_SOURCE_PROMPT,
            user_prompt=(
                "只返回JSON对象，顶层必须包含item_definition和functions。"
                "item_definition仅包含system_description、item_boundary、operating_modes、odd、"
                "source_location、source_excerpt、confidence、status；odd包含locations、road_types、"
                "weather_conditions、road_surfaces、speed_range_kph。functions每项包含function_id、name、"
                "output、description、preconditions、triggers、odd_constraints、fallback_behavior、"
                "consequences、source_location、source_excerpt、confidence、status。最多20个Function；"
                "operating_modes、preconditions、triggers、odd_constraints、consequences必须是JSON array"
                "或null，即使只有一条也不得返回string；fallback_behavior必须是string或null。"
                "每项列表最多10条，所有source_excerpt最多120字符，且必须是输入原文中连续、逐字相同的摘录。"
                "不得省略、改写或使用省略号。不得输出Markdown、解释或额外字段。\n\n"
                + CONFIDENCE_PROMPT_CONTRACT + "\n"
                + document_text + "\n" + REPORT_LANGUAGE_PROMPT
            ),
            schema_name="CoreItemArtifacts",
            prompt_version=self.PROMPT_VERSION,
            metadata={"source_id": source_id},
            max_tokens=initial_budget,
            response_schema=CORE_ITEM_ARTIFACTS_SCHEMA,
        )
        started = time.monotonic()
        output_limit_retry = False
        candidate_cache_hit = candidate_response is not None
        llm_call_count = 0
        if candidate_response is not None:
            response = candidate_response
        else:
            llm_call_count = 1
            try:
                response = self.client.complete_json(request)
            except LLMOutputLimitError:
                first_attempt_elapsed = time.monotonic() - started
                if initial_budget >= configured_limit:
                    raise
                output_limit_retry = True
                print(
                    "[HARA] core item artifact output limit "
                    f"old_max_tokens={initial_budget} new_max_tokens={configured_limit} "
                    f"first_attempt_elapsed={first_attempt_elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                request = replace(request, max_tokens=configured_limit)
                llm_call_count += 1
                response = self.client.complete_json(request)
            if candidate_recorder is not None:
                candidate_recorder(response)
        primary_response = response
        schema_repair_response = None
        schema_repair_count = 0
        try:
            normalized_data, contract_normalizations = normalize_core_item_artifacts(
                response.data
            )
        except CoreItemArtifactsContractError as error:
            print(
                "[HARA] core item schema repair "
                f"error_count={len(error.errors)} attempt=1/1",
                file=sys.stderr,
                flush=True,
            )
            repair_request = LLMRequest(
                task="repair_core_item_artifact_schema",
                system_prompt=(
                    "You repair JSON representation only. Preserve every engineering "
                    "statement, identifier, source location, source excerpt, confidence, "
                    "and status. Do not add, remove, merge, or reinterpret Functions. "
                    "Return exactly one raw JSON object matching CoreItemArtifacts."
                ),
                user_prompt=json.dumps({
                    "contract_errors": error.errors,
                    "invalid_payload": response.data,
                }, ensure_ascii=False),
                schema_name="CoreItemArtifacts",
                prompt_version="item-artifacts-schema-repair-v1",
                metadata={
                    "source_id": source_id,
                    "strict_no_format_retry": True,
                },
                max_tokens=request.max_tokens,
                response_schema=CORE_ITEM_ARTIFACTS_SCHEMA,
            )
            schema_repair_response = self.client.complete_json(repair_request)
            llm_call_count += 1
            if candidate_recorder is not None:
                candidate_recorder(schema_repair_response)
            normalized_data, contract_normalizations = normalize_core_item_artifacts(
                schema_repair_response.data
            )
            response = schema_repair_response
            schema_repair_count = 1

        # A schema-valid string can still confuse the function with its output.
        # Repair that field from the same source, without changing membership or
        # bypassing the existing domain/source validators below.
        output_repair_response = None
        repeated_outputs = [
            index for index, item in enumerate(normalized_data["functions"])
            if item["name"].strip() == item["output"].strip()
        ]
        if repeated_outputs:
            print("[HARA] core item function-output repair attempt=1/1",
                  file=sys.stderr, flush=True)
            output_repair_response = self.client.complete_json(LLMRequest(
                task="repair_core_item_function_outputs",
                system_prompt=(
                    "Repair only the output fields at the supplied function indices. "
                    "Describe the output signal, command, or state explicitly supported "
                    "by the Item, distinct from the function action/name. Do not invent "
                    "capabilities or merely append filler words. Preserve all other "
                    "fields, function identities, membership, order, sources and status. "
                    "Return one raw CoreItemArtifacts JSON object."
                ),
                user_prompt=json.dumps({
                    "function_indices": repeated_outputs,
                    "invalid_payload": normalized_data,
                    "item_definition_document": document_text,
                }, ensure_ascii=False),
                schema_name="CoreItemArtifacts",
                prompt_version="item-artifacts-output-repair-v1",
                metadata={"source_id": source_id, "strict_no_format_retry": True},
                max_tokens=request.max_tokens,
                response_schema=CORE_ITEM_ARTIFACTS_SCHEMA,
            ))
            llm_call_count += 1
            repaired_data, output_normalizations = normalize_core_item_artifacts(
                output_repair_response.data
            )
            expected = {
                **normalized_data,
                "functions": [dict(item) for item in normalized_data["functions"]],
            }
            if len(repaired_data["functions"]) == len(expected["functions"]):
                for index in repeated_outputs:
                    expected["functions"][index]["output"] = repaired_data["functions"][index]["output"]
            if repaired_data != expected:
                raise ValueError("function output repair altered non-output Item Definition content")
            if candidate_recorder is not None:
                candidate_recorder(output_repair_response)
            normalized_data = repaired_data
            contract_normalizations.extend(output_normalizations)
            response = output_repair_response

        def parse_entities(payload: dict[str, Any]):
            facts_value, warnings_value = ItemDefinitionNormalizer._parse_with_warnings(
                payload["item_definition"], source_id, source_blocks
            )
            functions_value = [
                FunctionNormalizer._parse(item, source_id)
                for item in payload["functions"]
            ]
            FunctionNormalizer._validate_unique(functions_value)
            self.validator.ensure_valid(functions_value)
            function_source_guard = validate_function_source_parity(
                functions_value, source_blocks, source_id=source_id,
            )
            return (
                facts_value,
                functions_value,
                warnings_value,
                function_source_guard,
            )

        facts, functions, warnings, function_source_guard = parse_entities(
            normalized_data
        )
        source_reference_repair_response = None
        source_reference_repair_count = 0
        source_reference_repair_normalizations: list[dict[str, Any]] = []
        try:
            source_reference_repairs = self._ensure_source_grounded(
                facts, functions, document_text, source_blocks,
            )
        except ValueError as error:
            if "source excerpt is not present in Item Definition" not in str(error):
                raise
            print(
                "[HARA] core item source-reference repair attempt=1/1",
                file=sys.stderr,
                flush=True,
            )
            source_repair_request = LLMRequest(
                task="repair_core_item_artifact_sources",
                system_prompt=(
                    "You repair source_location and source_excerpt fields only. "
                    "Preserve every other field, value, list order, identifier, confidence, "
                    "and status exactly. Each source_excerpt must be a continuous, verbatim "
                    "substring of the supplied Item Definition; do not paraphrase, abbreviate, "
                    "join fragments, or use ellipses. Return exactly one raw JSON object "
                    "matching CoreItemArtifacts."
                ),
                user_prompt=json.dumps({
                    "source_validation_error": str(error),
                    "invalid_payload": normalized_data,
                    "item_definition_document": document_text,
                }, ensure_ascii=False),
                schema_name="CoreItemArtifacts",
                prompt_version="item-artifacts-source-repair-v1",
                metadata={
                    "source_id": source_id,
                    "strict_no_format_retry": True,
                },
                max_tokens=request.max_tokens,
                response_schema=CORE_ITEM_ARTIFACTS_SCHEMA,
            )
            source_reference_repair_response = self.client.complete_json(source_repair_request)
            llm_call_count += 1
            if candidate_recorder is not None:
                candidate_recorder(source_reference_repair_response)
            repaired_data, source_reference_repair_normalizations = (
                normalize_core_item_artifacts(source_reference_repair_response.data)
            )
            if (
                self._without_source_references(repaired_data)
                != self._without_source_references(normalized_data)
            ):
                raise ValueError(
                    "source reference repair altered non-source Item Definition content"
                )
            facts, functions, warnings, function_source_guard = parse_entities(
                repaired_data
            )
            source_reference_repairs = self._ensure_source_grounded(
                facts, functions, document_text, source_blocks,
            )
            response = source_reference_repair_response
            source_reference_repair_count = 1

        # Provider labels are not approval authority.  The approved input plus
        # the deterministic schema/role/exact-source checks above are the gate.
        facts.status = ReviewStatus.FINALIZED if not warnings else ReviewStatus.PENDING
        for function in functions:
            function.status = ReviewStatus.FINALIZED
        elapsed_seconds = time.monotonic() - started
        return facts, functions, {
            "task": request.task,
            "prompt_version": request.prompt_version,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "primary_model": primary_response.model,
            "primary_request_id": primary_response.request_id,
            "primary_usage": primary_response.usage,
            "schema_repair_model": (
                schema_repair_response.model if schema_repair_response else ""
            ),
            "schema_repair_request_id": (
                schema_repair_response.request_id if schema_repair_response else ""
            ),
            "schema_repair_usage": (
                schema_repair_response.usage if schema_repair_response else {}
            ),
            "schema_repair_count": schema_repair_count,
            "function_output_repair_count": int(output_repair_response is not None),
            "function_output_repair_request_id": (
                output_repair_response.request_id if output_repair_response else ""
            ),
            "function_output_repair_usage": (
                output_repair_response.usage if output_repair_response else {}
            ),
            "source_reference_repair_model": (
                source_reference_repair_response.model
                if source_reference_repair_response else ""
            ),
            "source_reference_repair_request_id": (
                source_reference_repair_response.request_id
                if source_reference_repair_response else ""
            ),
            "source_reference_repair_usage": (
                source_reference_repair_response.usage
                if source_reference_repair_response else {}
            ),
            "source_reference_repair_count": source_reference_repair_count,
            "source_reference_repair_normalizations": (
                source_reference_repair_normalizations
            ),
            "contract_normalizations": contract_normalizations,
            "candidate_cache_hit": candidate_cache_hit,
            "function_source_guard": function_source_guard,
            "function_count": len(functions),
            "normalization_warnings": warnings,
            "source_reference_repairs": source_reference_repairs,
            "output_limit_retry": output_limit_retry,
            "max_tokens": request.max_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "llm_call_count": llm_call_count,
            "input_characters": len(request.system_prompt) + len(request.user_prompt),
        }
