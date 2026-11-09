from __future__ import annotations


import json
import re
import sys
import time
from copy import deepcopy
from dataclasses import replace

from hara_agent.contracts import Guideword
from hara_agent.infrastructure.llm import (
    LLMClient,
    LLMOutputLimitError,
    LLMRequest,
)
from hara_agent.models import (
    FunctionDefinition,
    GuidewordAssessment,
    GuidewordDisposition,
    ReviewStatus,
)

from .parsing import REPORT_LANGUAGE_PROMPT, CONFIDENCE_PROMPT_CONTRACT, parse_confidence
from .traceability import resolve_guideword_sources


class GuidewordApplicabilityAgent:
    """Assess every template guideword without generating a Cartesian product."""

    PROMPT_VERSION = "guideword-applicability-v12-repair-identity-shape-zh-v1"
    INITIAL_MAX_TOKENS = 4096
    OUTPUT_LIMIT_RETRY_MAX_TOKENS = 8192
    SYSTEM_PROMPT = REPORT_LANGUAGE_PROMPT + """你是汽车功能安全HAZOP分析助手。针对给定Function/Output逐项判断模板Guideword是否具有明确的功能语义和可形成的偏差。必须覆盖输入中的每个Guideword且只出现一次。不适用不是遗漏，必须给出具体理由；不得为了凑数量判为适用。结论必须区分：DOWNSTREAM_CANDIDATE（语义适用且存在可信的车辆级危害潜力）、NOT_APPLICABLE（偏差维度与该Function/Output语义不适用）、NO_CREDIBLE_HAZARD（偏差语义适用，但依据Item Definition无法形成可信的车辆级危害）。disposition只能是DOWNSTREAM_CANDIDATE、NOT_APPLICABLE、NO_CREDIBLE_HAZARD之一；PENDING不是disposition。证据完整性和ReviewStatus由系统确定，模型不得使用PENDING作为disposition。不确定时不得武断标记NO_CREDIBLE_HAZARD；证据不足时仍须在三种disposition中选择最合理的一个。"""
    COMPOSITE_OUTPUT_CONTRACT = """
For a composite Function output, evaluate every canonical_output_components entry.
Do not decide the Guideword from only an HMI message, duration, or any one partial
output. Include safety-relevant actuation, control, and action outputs. For
More/Less/No deviations, Less may include a missing partial output, insufficient
capability, incomplete execution, or insufficient quantity/duration.
"""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "guideword": {"type": "string"},
                        "applicable": {"type": "boolean"},
                        "disposition": {
                            "type": "string",
                            "enum": [
                                "DOWNSTREAM_CANDIDATE",
                                "NOT_APPLICABLE",
                                "NO_CREDIBLE_HAZARD",
                            ],
                        },
                        "rationale": {"type": "string", "maxLength": 240},
                        "confidence": {"type": "number"},
                    },
                    "required": ["guideword", "applicable", "disposition", "rationale", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["assessments"],
        "additionalProperties": False,
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def assess(self, function: FunctionDefinition, guidewords: list[str | Guideword]
               ) -> tuple[list[GuidewordAssessment], dict]:
        started = time.monotonic()
        contexts = [self._guideword_context(value) for value in guidewords]
        contexts = [value for value in contexts if value["name"]]
        expected_names = [value["name"] for value in contexts]
        expected_ids = [value["guideword_id"] for value in contexts]
        guideword_by_id = {value["guideword_id"]: value for value in contexts}
        if (
            not expected_ids
            or len(expected_ids) != len(set(expected_ids))
            or len(expected_names) != len(set(expected_names))
        ):
            raise ValueError("Guideword输入必须非空且唯一")
        response_schema = deepcopy(self.RESPONSE_SCHEMA)
        assessment_schema = response_schema["properties"]["assessments"]
        assessment_schema["minItems"] = len(expected_ids)
        assessment_schema["maxItems"] = len(expected_ids)
        assessment_schema["items"]["properties"]["guideword"]["enum"] = expected_ids
        function_context = {
            "function_id": function.function_id,
            "name": function.name,
            "output": function.output,
            "canonical_output_components": self._canonical_output_components(function.output),
            "description": function.description,
            "preconditions": function.preconditions,
            "triggers": function.triggers,
            "odd_constraints": function.odd_constraints,
            "fallback_behavior": function.fallback_behavior,
            "consequences": function.consequences,
            "source_evidence": [
                {
                    "location": source.location,
                    "excerpt": source.excerpt,
                }
                for source in function.sources
                if source.location or source.excerpt
            ],
        }
        request = LLMRequest(
            task="assess_guideword_applicability",
            system_prompt=self.SYSTEM_PROMPT + self.COMPOSITE_OUTPUT_CONTRACT,
            user_prompt=(
                "FunctionContext="
                + json.dumps(function_context, ensure_ascii=False)
                + "\nGuidewordDefinitions="
                + json.dumps(contexts, ensure_ascii=False)
                + "\nProviderIdentityContract: guideword is the stable guideword_id from GuidewordDefinitions, not the display name. "
                "Return each supplied ID exactly once. The system alone binds that ID back to its display name; do not generate guideword_id or guideword_name fields.\n"
                + "\n必须优先采用模板给出的Guideword description判断偏差维度；不得只按名称联想。"
                "在返回JSON前于同一次回答内逐项反证检查适用结论，尤其当几乎全部Guideword均适用时；"
                "不得为了全局覆盖率强制任何Guideword匹配Function。该自检不得产生额外顶层字段。\n"
                f"assessments必须恰好包含{len(expected_ids)}项：每个输入Guideword恰好一次，禁止遗漏、重复或新增Guideword。"
                "每个rationale只能用一句简洁工程理由说明适用性或disposition，最多240字符；"
                "不得长篇重复Function或Item Definition原文，不得使用Markdown，不得输出额外commentary。\n"
                "只返回一个JSON object，顶层必须且只能使用assessments字段："
                "{\"assessments\":[...]}。assessments每项包含guideword、applicable、rationale、"
                "disposition、confidence；guideword必须为string，applicable必须为boolean，"
                "disposition必须是DOWNSTREAM_CANDIDATE、NOT_APPLICABLE、NO_CREDIBLE_HAZARD之一。"
                "PENDING不是disposition，不得出现在disposition字段。"
                "DOWNSTREAM_CANDIDATE和NO_CREDIBLE_HAZARD的applicable必须为true，NOT_APPLICABLE必须为false；"
                "rationale必须为非空string且简洁，"
                "confidence必须为0.0到1.0的JSON number。必须逐项覆盖输入的全部Guidewords，不得省略rationale。"
                "source和status由系统从Function证据确定性传播，禁止生成source或status字段。"
                + CONFIDENCE_PROMPT_CONTRACT
            ),
            schema_name="GuidewordAssessmentList",
            prompt_version=self.PROMPT_VERSION,
            metadata={"function_id": function.function_id},
            max_tokens=self._initial_max_tokens(),
            response_schema=response_schema,
        )
        initial_budget = request.max_tokens or self.INITIAL_MAX_TOKENS
        configured_limit = self._configured_provider_limit()
        output_limit_retry_audit: dict[str, object] | None = None
        try:
            response = self.client.complete_json(request)
        except LLMOutputLimitError as error:
            retry_budget = min(configured_limit, self.OUTPUT_LIMIT_RETRY_MAX_TOKENS)
            output_limit_retry_audit = self._output_limit_retry_audit(
                function.function_id,
                initial_budget=initial_budget,
                retry_budget=retry_budget,
                error=error,
            )
            if retry_budget <= initial_budget:
                print(
                    "[HARA] guideword output-limit retry skipped "
                    f"function={function.function_id} "
                    f"initial_max_tokens={initial_budget} "
                    f"configured_limit={configured_limit} "
                    "reason=provider_limit_not_above_current_budget",
                    file=sys.stderr,
                    flush=True,
                )
                error.diagnostics.update({
                    "function_id": function.function_id,
                    "output_limit_retry": False,
                    "output_limit_retry_audit": output_limit_retry_audit,
                })
                raise
            print(
                "[HARA] guideword output-limit retry "
                f"function={function.function_id} "
                f"initial_max_tokens={initial_budget} "
                f"retry_max_tokens={retry_budget} attempt=2/2 "
                f"finish_reason={output_limit_retry_audit['finish_reason']} "
                f"prompt_tokens={output_limit_retry_audit['prompt_tokens']} "
                f"completion_tokens={output_limit_retry_audit['completion_tokens']}",
                file=sys.stderr,
                flush=True,
            )
            retry_metadata = dict(request.metadata)
            retry_metadata["_guideword_output_limit_retry_attempt"] = 1
            retry = replace(request, max_tokens=retry_budget, metadata=retry_metadata)
            try:
                response = self.client.complete_json(retry)
            except LLMOutputLimitError as retry_error:
                retry_error.diagnostics.update({
                    "function_id": function.function_id,
                    "output_limit_retry": True,
                    "output_limit_retry_audit": output_limit_retry_audit,
                })
                raise
            request = retry
        raw = response.data.get("assessments")
        if not isinstance(raw, list):
            raise ValueError("LLM输出缺少assessments数组")

        assessments: list[GuidewordAssessment] = []
        parse_errors: list[dict[str, object]] = []
        invalid_items: list[tuple[int, dict[str, object], dict[str, str], str]] = []
        item_salvage_audits: list[dict[str, object]] = []
        for idx, item in enumerate(raw):
            normalized_item, binding = self._bind_provider_identity(
                item, index=idx, guideword_by_id=guideword_by_id,
            )
            violation = self._applicability_disposition_violation(normalized_item)
            if violation:
                invalid_items.append((idx, normalized_item, binding, violation))
                continue
            assessment, parse_error = self._parse_item(function, normalized_item, idx)
            assessment.guideword_id = binding["guideword_id"]
            assessments.append(assessment)
            if parse_error is not None:
                parse_errors.append(parse_error)

        for index, _item, binding, violation in invalid_items:
            print(
                "[HARA] guideword item salvage "
                f"function={function.function_id} guideword_id={binding['guideword_id']} "
                f"index={index} attempt=1/1 reason={violation}",
                file=sys.stderr,
                flush=True,
            )
            repair_request = self._item_repair_request(
                function_context=function_context,
                binding=binding,
                violation=violation,
                max_tokens=request.max_tokens or self.INITIAL_MAX_TOKENS,
            )
            repair_response = self.client.complete_json(repair_request)
            repair_raw = repair_response.data.get("assessments")
            if not isinstance(repair_raw, list) or len(repair_raw) != 1:
                raise ValueError(
                    "GUIDEWORD_ITEM_REPAIR_CONTRACT_VIOLATION: "
                    f"function={function.function_id} guideword_id={binding['guideword_id']} "
                    "expected exactly one repaired assessment"
                )
            repaired_item, repaired_binding = self._bind_provider_identity(
                repair_raw[0], index=index,
                guideword_by_id={binding["guideword_id"]: binding},
            )
            repaired_violation = self._applicability_disposition_violation(repaired_item)
            if repaired_violation:
                raise ValueError(
                    "GUIDEWORD_ITEM_REPAIR_CONTRACT_VIOLATION: "
                    f"function={function.function_id} guideword_id={binding['guideword_id']} "
                    f"attempt=1/1 reason={repaired_violation}"
                )
            repaired_assessment, repair_parse_error = self._parse_item(
                function, repaired_item, index,
            )
            repaired_assessment.guideword_id = repaired_binding["guideword_id"]
            assessments.append(repaired_assessment)
            if repair_parse_error is not None:
                parse_errors.append(repair_parse_error)
            item_salvage_audits.append({
                "guideword_id": binding["guideword_id"],
                "display_name": binding["name"],
                "attempt": 1,
                "initial_violation": violation,
                "status": "REPAIRED",
                "request_id": repair_response.request_id,
                "usage": repair_response.usage,
            })

        actual = [item.guideword_id for item in assessments]
        if len(actual) != len(set(actual)):
            raise ValueError("Guideword identity duplicate: guideword_id")
        missing = [guideword_id for guideword_id in expected_ids if guideword_id not in actual]
        extra = [guideword_id for guideword_id in actual if guideword_id not in expected_ids]
        if missing or extra:
            raise ValueError(f"Guideword覆盖不完整: missing={missing}, extra={extra}")
        assessment_by_id = {item.guideword_id: item for item in assessments}
        assessments = [assessment_by_id[guideword_id] for guideword_id in expected_ids]
        complete_count = sum(item.is_semantically_complete for item in assessments)
        incomplete_count = len(assessments) - complete_count
        review_finalized_count = sum(item.status == ReviewStatus.FINALIZED for item in assessments)
        review_pending_count = sum(item.status == ReviewStatus.PENDING for item in assessments)
        applicable_count = sum(
            item.applicable and item.is_semantically_complete
            for item in assessments
        )
        downstream_candidate_count = sum(
            item.enters_downstream and item.is_semantically_complete
            for item in assessments
        )
        not_applicable_count = sum(
            item.disposition is GuidewordDisposition.NOT_APPLICABLE
            for item in assessments
        )
        no_credible_hazard_count = sum(
            item.disposition is GuidewordDisposition.NO_CREDIBLE_HAZARD
            for item in assessments
        )
        blanket_applicability = (
            len(expected_ids) > 1 and applicable_count == len(expected_ids)
        )
        near_blanket_applicability = (
            len(expected_ids) >= 4
            and downstream_candidate_count >= len(expected_ids) - 1
        )
        if blanket_applicability:
            print(
                "[HARA] guideword blanket applicability warning "
                f"function={function.function_id} applicable={applicable_count}/{len(expected_ids)} "
                "action=continue_to_downstream_causal_filters",
                file=sys.stderr,
                flush=True,
            )
        elapsed_seconds = time.monotonic() - started
        print(
            "[HARA] guideword assessment completed "
            f"function={function.function_id} expected={len(expected_ids)} "
            f"received={len(assessments)} complete={complete_count} "
            f"incomplete={incomplete_count} review_pending={review_pending_count} "
            f"review_finalized={review_finalized_count} parse_errors={len(parse_errors)} "
            f"elapsed={elapsed_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        metadata = {
            "task": request.task,
            "function_id": function.function_id,
            "prompt_version": request.prompt_version,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "coverage_count": len(assessments),
            "complete_count": complete_count,
            "incomplete_count": incomplete_count,
            "review_finalized_count": review_finalized_count,
            "review_pending_count": review_pending_count,
            # Backward-compatible review lifecycle counters.
            "finalized_count": review_finalized_count,
            "pending_count": review_pending_count,
            "parse_error_count": len(parse_errors),
            "applicable_count": applicable_count,
            "downstream_candidate_count": downstream_candidate_count,
            "not_applicable_count": not_applicable_count,
            "no_credible_hazard_count": no_credible_hazard_count,
            "blanket_applicability": blanket_applicability,
            "near_blanket_applicability": near_blanket_applicability,
            "additional_llm_calls": 1 if output_limit_retry_audit else 0,
            "output_limit_retry": bool(output_limit_retry_audit),
            "output_limit_retry_audit": output_limit_retry_audit,
            "item_salvage_attempted_count": len(invalid_items),
            "item_salvage_repaired_count": len(item_salvage_audits),
            "item_salvage": item_salvage_audits,
            "llm_call_count": (
                1 + (1 if output_limit_retry_audit else 0) + len(item_salvage_audits)
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        if parse_errors:
            metadata["parse_errors"] = parse_errors
        return assessments, metadata

    @staticmethod
    def _applicability_disposition_violation(item: dict[str, object]) -> str:
        applicable = item.get("applicable")
        disposition = str(item.get("disposition", "")).strip()
        if not isinstance(applicable, bool):
            return ""
        if applicable is False and disposition in {
            GuidewordDisposition.DOWNSTREAM_CANDIDATE.value,
            GuidewordDisposition.NO_CREDIBLE_HAZARD.value,
        }:
            return (
                f"applicable=false is incompatible with disposition={disposition}"
            )
        if applicable is True and disposition == GuidewordDisposition.NOT_APPLICABLE.value:
            return "applicable=true is incompatible with disposition=NOT_APPLICABLE"
        return ""

    def _item_repair_request(
        self,
        *,
        function_context: dict[str, object],
        binding: dict[str, str],
        violation: str,
        max_tokens: int,
    ) -> LLMRequest:
        response_schema = deepcopy(self.RESPONSE_SCHEMA)
        assessment_schema = response_schema["properties"]["assessments"]
        assessment_schema["minItems"] = 1
        assessment_schema["maxItems"] = 1
        assessment_schema["items"]["properties"]["guideword"]["enum"] = [
            binding["guideword_id"]
        ]
        guideword_context = {
            "guideword_id": binding["guideword_id"],
            "name": binding["name"],
            "description": binding["description"],
            "template_source": binding["template_source"],
        }
        return LLMRequest(
            task="assess_guideword_applicability_item_repair",
            system_prompt=self.SYSTEM_PROMPT + self.COMPOSITE_OUTPUT_CONTRACT,
            user_prompt=(
                "Repair exactly one malformed Guideword assessment. Do not return any sibling item.\n"
                "FunctionContext=" + json.dumps(function_context, ensure_ascii=False)
                + "\nGuidewordDefinition=" + json.dumps(guideword_context, ensure_ascii=False)
                + f"\nObservedContractViolation={violation}\n"
                "guideword must be exactly the supplied stable guideword_id. "
                "Return one JSON object with assessments containing exactly one item. "
                "The item must include the required guideword field exactly as "
                + json.dumps(binding["guideword_id"], ensure_ascii=False)
                + "; for example {\"guideword\":"
                + json.dumps(binding["guideword_id"], ensure_ascii=False)
                + ",\"applicable\":false,\"disposition\":\"NOT_APPLICABLE\","
                "\"rationale\":\"...\",\"confidence\":0.8}. "
                "Do not emit guideword_id or guideword_name fields. "
                "The only legal applicability/disposition combinations are: "
                "applicable=true with DOWNSTREAM_CANDIDATE or NO_CREDIBLE_HAZARD; "
                "applicable=false with NOT_APPLICABLE. Do not change the guideword identity."
            ),
            schema_name="GuidewordAssessmentItemRepair",
            prompt_version=self.PROMPT_VERSION,
            metadata={
                "function_id": str(function_context["function_id"]),
                "guideword_id": binding["guideword_id"],
                "item_repair": True,
            },
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def _initial_max_tokens(self) -> int:
        return min(self.INITIAL_MAX_TOKENS, self._configured_provider_limit())

    def _configured_provider_limit(self) -> int:
        configured = int(
            getattr(getattr(self.client, "config", None), "max_tokens", 32768)
        )
        if configured <= 0:
            raise ValueError("Guideword provider max_tokens必须大于0")
        return configured

    @staticmethod
    def _output_limit_retry_audit(
        function_id: str,
        *,
        initial_budget: int,
        retry_budget: int,
        error: LLMOutputLimitError,
    ) -> dict[str, object]:
        diagnostics = dict(error.diagnostics or {})
        return {
            "function_id": function_id,
            "initial_max_tokens": initial_budget,
            "retry_max_tokens": retry_budget,
            "attempt": 2,
            "failed_attempt": 1,
            "finish_reason": diagnostics.get("finish_reason", "length"),
            "prompt_tokens": diagnostics.get(
                "prompt_tokens", diagnostics.get("input_tokens")
            ),
            "completion_tokens": diagnostics.get(
                "completion_tokens", diagnostics.get("output_tokens")
            ),
        }

    @staticmethod
    def _guideword_context(value: str | Guideword) -> dict[str, str]:
        if isinstance(value, Guideword):
            return {
                "guideword_id": value.guideword_id.strip(),
                "name": value.name.strip(),
                "description": value.description.strip(),
                "template_source": (
                    f"{value.source_ref.sheet}!{value.source_ref.range}"
                ),
            }
        return {
            "guideword_id": str(value).strip(),
            "name": str(value).strip(),
            "description": "",
            "template_source": "legacy_untyped_input",
        }

    @staticmethod
    def _bind_provider_identity(
        item: object,
        *,
        index: int,
        guideword_by_id: dict[str, dict[str, str]],
    ) -> tuple[dict[str, object], dict[str, str]]:
        """Bind a provider-returned stable ID to the input contract display name."""
        if not isinstance(item, dict):
            raise ValueError(
                f"PROVIDER_GUIDEWORD_IDENTITY_VIOLATION: item={index} is not an object"
            )
        provider_id = str(item.get("guideword", "")).strip()
        if not provider_id:
            raise ValueError(
                f"PROVIDER_GUIDEWORD_IDENTITY_VIOLATION: item={index} missing guideword_id"
            )
        binding = guideword_by_id.get(provider_id)
        if binding is None:
            raise ValueError(
                "PROVIDER_GUIDEWORD_IDENTITY_VIOLATION: "
                f"item={index} unknown guideword_id={provider_id!r}"
            )
        explicit_id = str(item.get("guideword_id", "")).strip()
        if explicit_id and explicit_id != provider_id:
            raise ValueError(
                "PROVIDER_GUIDEWORD_IDENTITY_VIOLATION: "
                f"item={index} guideword={provider_id!r} guideword_id={explicit_id!r} mismatch"
            )
        provider_name = str(item.get("guideword_name", "")).strip()
        if provider_name and provider_name != binding["name"]:
            raise ValueError(
                "PROVIDER_GUIDEWORD_IDENTITY_VIOLATION: "
                f"item={index} guideword_id={provider_id!r} "
                f"name={provider_name!r} expected_name={binding['name']!r}"
            )
        normalized = dict(item)
        normalized["guideword"] = binding["name"]
        return normalized, binding

    @staticmethod
    def _canonical_output_components(output: str) -> list[str]:
        """Expose the existing canonical output as compact, deterministic parts."""
        text = str(output).strip()
        if not text:
            return []
        components = [
            item.strip(" -\t\u2022")
            for item in re.split(r"[\n;\uFF1B]+", text)
            if item.strip(" -\t\u2022")
        ]
        return components or [text]

    @classmethod
    def _parse_item(
        cls, function: FunctionDefinition, item: object, index: int,
    ) -> tuple[GuidewordAssessment, dict[str, object] | None]:
        raw_keys = sorted(str(key) for key in item) if isinstance(item, dict) else []
        guideword = str(item.get("guideword", "")).strip() if isinstance(item, dict) else ""
        raw_status = str(item.get("status", "")).strip() if isinstance(item, dict) else ""
        missing_fields = []
        if not guideword:
            missing_fields.append("guideword")
        if not isinstance(item, dict) or not isinstance(item.get("applicable"), bool):
            missing_fields.append("applicable")
        rationale = str(item.get("rationale", "")).strip() if isinstance(item, dict) else ""
        if not rationale:
            missing_fields.append("rationale")
        raw_disposition = (
            str(item.get("disposition", "")).strip() if isinstance(item, dict) else ""
        )
        raw_applicable = item.get("applicable") if isinstance(item, dict) else None
        disposition_repair = None
        if raw_disposition:
            try:
                GuidewordDisposition(raw_disposition)
            except ValueError:
                if raw_disposition == "PENDING" and raw_applicable is False:
                    disposition_repair = {
                        "field": "disposition",
                        "raw": raw_disposition,
                        "repaired": "NOT_APPLICABLE",
                        "reason": "applicable=false and disposition=PENDING is unambiguously NOT_APPLICABLE",
                    }
                    print(
                        "[HARA] guideword disposition deterministic repair "
                        f"function={function.function_id} index={index} "
                        f"guideword={guideword} raw=PENDING repaired=NOT_APPLICABLE",
                        file=sys.stderr, flush=True,
                    )
                else:
                    raise ValueError(
                        f"PROVIDER_INVALID_GUIDEWORD_DISPOSITION: "
                        f"disposition={raw_disposition!r} applicable={raw_applicable!r} "
                        f"function={function.function_id} guideword={guideword}; "
                        f"cannot deterministically repair"
                    )
        elif raw_applicable is False:
            disposition_repair = {
                "field": "disposition",
                "raw": "<missing>",
                "repaired": "NOT_APPLICABLE",
                "reason": "applicable=false and disposition missing is unambiguously NOT_APPLICABLE",
            }
            print(
                "[HARA] guideword disposition deterministic repair "
                f"function={function.function_id} index={index} "
                f"guideword={guideword} raw=<missing> repaired=NOT_APPLICABLE",
                file=sys.stderr, flush=True,
            )
        elif raw_applicable is True:
            raise ValueError(
                f"PROVIDER_INVALID_GUIDEWORD_DISPOSITION: "
                f"disposition=<missing> applicable=True "
                f"function={function.function_id} guideword={guideword}; "
                f"cannot distinguish DOWNSTREAM_CANDIDATE vs NO_CREDIBLE_HAZARD"
            )
        if missing_fields:
            print(
                "[HARA] guideword item validation failed "
                f"function={function.function_id} index={index} "
                f"guideword={guideword or '<missing>'} raw_keys={raw_keys} "
                f"missing_fields={missing_fields} raw_status={raw_status or '<missing>'}",
                file=sys.stderr,
                flush=True,
            )
        if any(field in missing_fields for field in ("guideword", "applicable")):
            if "guideword" in missing_fields:
                raise ValueError("GuidewordAssessment缺少guideword")
            raise ValueError("GuidewordAssessment.applicable必须为boolean")
        assessment = cls._parse(function, item, disposition_repair=disposition_repair)
        if "rationale" not in missing_fields and disposition_repair is None:
            return assessment, None
        parse_error = {
            "function_id": function.function_id,
            "index": index,
            "guideword": guideword,
            "field": "rationale" if "rationale" in missing_fields else "disposition",
            "error": "missing" if "rationale" in missing_fields else "provider_invalid_disposition_repaired",
            "raw_status": raw_status or "<missing>",
        }
        if disposition_repair is not None:
            parse_error["disposition_repair"] = disposition_repair
        print(
            "[HARA] guideword item degraded "
            f"function={function.function_id} index={index} guideword={guideword} "
            "status=PENDING reason=missing_rationale",
            file=sys.stderr,
            flush=True,
        )
        return assessment, parse_error

    @staticmethod
    def _parse(
        function: FunctionDefinition, item: object,
        disposition_repair: dict[str, object] | None = None,
    ) -> GuidewordAssessment:
        if not isinstance(item, dict):
            raise ValueError("assessments数组元素必须为object")
        if not isinstance(item.get("applicable"), bool):
            raise ValueError("Guideword applicable必须为boolean")
        rationale = str(item.get("rationale", "")).strip()
        disposition_text = str(item.get("disposition", "")).strip()
        if disposition_repair is not None:
            disposition_text = str(
                disposition_repair.get("repaired", disposition_text)
            ).strip()
        disposition = (
            GuidewordDisposition(disposition_text)
            if disposition_text else (
                GuidewordDisposition.DOWNSTREAM_CANDIDATE
                if item["applicable"] else GuidewordDisposition.NOT_APPLICABLE
            )
        )
        sources = resolve_guideword_sources(function)
        status = (
            ReviewStatus.FINALIZED
            if (
                rationale
                and sources
                and function.status is ReviewStatus.FINALIZED
            )
            else ReviewStatus.PENDING
        )
        return GuidewordAssessment(
            function_id=function.function_id,
            guideword=str(item.get("guideword", "")).strip(),
            applicable=item["applicable"],
            rationale=rationale,
            sources=sources,
            status=status,
            confidence=parse_confidence(
                item.get("confidence"),
                field_name=(
                    "Guideword confidence validation failed: "
                    f"function={function.function_id} guideword={item.get('guideword', '')}"
                ),
            ),
            disposition=disposition,
        )
