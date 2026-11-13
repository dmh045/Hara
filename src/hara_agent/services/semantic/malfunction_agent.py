from __future__ import annotations


import json
import re
import sys
from typing import Any

from hara_agent.contracts import FailureModeTaxonomyValue
from hara_agent.infrastructure.llm import LLMClient, LLMRequest
from hara_agent.models import (
    FunctionDefinition,
    GuidewordAssessment,
    MalfunctionCandidate,
    ReviewStatus,
)

from . import malfunction_guideword_gate
from .parsing import REPORT_LANGUAGE_PROMPT, CONFIDENCE_PROMPT_CONTRACT, parse_confidence
from .traceability import resolve_malfunction_sources
from hara_agent.services.extraction import normalize_source_text


# Preserve the established import surface for callers of this module.
MalfunctionGuidewordGateViolation = (
    malfunction_guideword_gate.MalfunctionGuidewordGateViolation
)


class _FailureTypeContractViolation(ValueError):
    """A configured canonical failure_type was omitted or not governed."""


class MalfunctionHazardAgent:
    """Generate traceable malfunction and vehicle-level hazard candidates."""

    PROMPT_VERSION = "malfunction-hazard-v9-explicit-candidate-fields"
    SYSTEM_PROMPT = REPORT_LANGUAGE_PROMPT + """你是汽车功能安全HARA分析助手。只处理已适用的Guideword，从给定功能的正常行为推导失效假设，不声称Item原文已定义这些故障。
每条候选只针对一个可区分的偏差量；转角与转向速率、制动力与驱动扭矩等不同量不能用“或/”合成一条。泊入/泊出等操作范围可以共享同一失效定义，但不证明各操作场景已覆盖，不机械展开操作×偏差的笛卡尔积。
derivation.subject须逐字来自所引用FunctionContext字段，description须使用该主体和quantity。仅有功能输出时分析输出偏差，不升级为执行器硬件损坏或独立控制通道故障；只有原文明确组件时才能引用它。分类枚举不是项目硬件证据。
derivation.basis说明功能与Guideword如何支持该偏差；原文未说明的实现、触发或能力放assumptions，不添加与来源否定/限制冲突的假设。没有可信车辆危险链时不得为凑Guideword数量编造危险。
functional_effect写行为影响；vehicle_level_hazard写车辆级危险状态，不写人员伤亡、评分或工程批准。保留前置条件、触发和限制。"""

    INJURY_TERMS = ("死亡", "致命伤", "骨折", "窒息", "fatality", "death", "injury")

    def __init__(
        self,
        client: LLMClient,
        *,
        component_categories: tuple[str, ...] = (),
        failure_types: tuple[FailureModeTaxonomyValue, ...] = (),
    ):
        self.client = client
        self.component_categories = tuple(component_categories)
        self.failure_types = tuple(failure_types)
        self.failure_type_ids = tuple(item.canonical_id for item in self.failure_types)
        if len(set(self.failure_type_ids)) != len(self.failure_type_ids):
            raise ValueError("Malfunction failure_type contract has duplicate canonical IDs")

    def _response_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "malfunction_id": {"type": "string"},
            "guideword": {"type": "string"},
            "description": {"type": "string"},
            "functional_effect": {"type": "string"},
            "vehicle_level_hazard": {"type": "string"},
            "causal_chain": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "confidence": {"type": "number"},
            "status": {"type": "string", "enum": ["PENDING"]},
            "derivation": {
                "type": "object",
                "properties": {
                    "function_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "subject": {"type": "string"},
                    "quantity": {"type": "string"},
                    "basis": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["function_refs", "subject", "quantity", "basis", "assumptions"],
                "additionalProperties": False,
            },
        }
        required = [
            "malfunction_id", "guideword", "description", "functional_effect",
            "vehicle_level_hazard", "causal_chain", "confidence", "status", "derivation",
        ]
        if self.component_categories:
            properties["component_category"] = {
                "type": "string", "enum": list(self.component_categories),
            }
            required.append("component_category")
        if self.failure_types:
            properties["failure_type"] = {
                "type": "string", "enum": list(self.failure_type_ids),
            }
            required.append("failure_type")
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

    def generate(self, function: FunctionDefinition,
                 assessments: list[GuidewordAssessment]
                 ) -> tuple[list[MalfunctionCandidate], dict[str, Any]]:
        applicable = [
            item for item in assessments
            if (
                item.applicable is True
                and item.is_semantically_complete
                and item.enters_downstream
                and item.status is ReviewStatus.FINALIZED
            )
        ]
        if any(item.function_id != function.function_id for item in assessments):
            raise ValueError("GuidewordAssessment与Function不一致")
        if not applicable:
            has_downstream_candidate = any(
                item.enters_downstream for item in assessments
            )
            has_semantically_applicable = any(item.applicable for item in assessments)
            if has_downstream_candidate:
                skip_reason = "no_complete_applicable_guidewords"
            elif has_semantically_applicable:
                skip_reason = "no_credible_hazard_guidewords"
            else:
                skip_reason = "no_applicable_guidewords"
            complete_count = sum(item.is_semantically_complete for item in assessments)
            incomplete_count = len(assessments) - complete_count
            print(
                "[HARA] malfunction skipped "
                f"function={function.function_id} reason={skip_reason} "
                f"complete={complete_count} incomplete={incomplete_count} "
                "applicable_complete=0",
                file=sys.stderr,
                flush=True,
            )
            return [], {
                "task": "derive_malfunctions_and_hazards",
                "function_id": function.function_id,
                "prompt_version": self.PROMPT_VERSION,
                "applicable_guidewords": 0,
                "candidate_count": 0,
                "llm_call_count": 0,
                "coverage_repair_count": 0,
                "skipped": True,
                "skip_reason": skip_reason,
                "complete_guidewords": complete_count,
                "incomplete_guidewords": incomplete_count,
            }
        request = self._request(function, applicable)
        response = self.client.complete_json(request)
        rejected = []
        candidates = self._parse_candidates(
            function, response.data, assessments, applicable, rejected=rejected,
        )
        self._validate(candidates, applicable, require_coverage=False)
        missing = self._missing_guidewords(candidates, applicable)
        repair_responses = []
        repair_rejected = []
        repair_words = sorted(set(missing) | {row["guideword"] for row in rejected})
        if repair_words:
            print(
                "[HARA] malfunction coverage repair "
                f"function={function.function_id} missing={missing} attempt=1/1",
                file=sys.stderr,
                flush=True,
            )
            repair_request = self._request(
                function,
                [item for item in applicable if item.guideword in set(repair_words)],
                coverage_repair=True,
                existing=candidates,
                rejected=rejected,
            )
            repair_response = self.client.complete_json(repair_request)
            repair_responses.append(repair_response)
            repaired = self._parse_candidates(
                function,
                repair_response.data,
                assessments,
                [item for item in applicable if item.guideword in set(repair_words)],
                rejected=repair_rejected,
            )
            self._validate(
                repaired,
                [item for item in applicable if item.guideword in set(repair_words)],
                require_coverage=False,
            )
            self._merge_candidates(function, candidates, repaired)

        remaining = sorted(set(self._missing_guidewords(candidates, applicable))
                           | {row["guideword"] for row in repair_rejected})
        unresolved_issues = []
        for guideword in remaining:
            target = [item for item in applicable if item.guideword == guideword]
            print(
                "[HARA] malfunction targeted coverage repair "
                f"function={function.function_id} guideword={guideword} attempt=1/1",
                file=sys.stderr,
                flush=True,
            )
            targeted_request = self._request(
                function,
                target,
                coverage_repair=True,
                existing=candidates,
                rejected=[row for row in repair_rejected if row["guideword"] == guideword],
            )
            targeted_response = self.client.complete_json(targeted_request)
            repair_responses.append(targeted_response)
            targeted = self._parse_candidates(
                function, targeted_response.data, assessments, target, rejected=unresolved_issues,
            )
            self._validate(targeted, target, require_coverage=False)
            self._merge_candidates(function, candidates, targeted)
        unresolved = self._missing_guidewords(candidates, applicable)
        if unresolved:
            raise ValueError(
                "适用Guideword缺少Malfunction候选: "
                f"function={function.function_id} missing={unresolved}; "
                "grouped and targeted coverage repair produced no valid candidate"
            )
        self._validate(candidates, applicable)
        return candidates, {
            "task": request.task,
            "function_id": function.function_id,
            "prompt_version": request.prompt_version,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "repair_model": repair_responses[0].model if repair_responses else "",
            "repair_request_id": repair_responses[0].request_id if repair_responses else "",
            "repair_usage": repair_responses[0].usage if repair_responses else {},
            "repair_request_ids": [item.request_id for item in repair_responses],
            "repair_usages": [item.usage for item in repair_responses],
            "applicable_guidewords": len(applicable),
            "candidate_count": len(candidates),
            "coverage_repair_count": len(repair_responses),
            "coverage_missing_before_repair": missing,
            "coverage_missing_after_grouped_repair": remaining,
            "llm_call_count": 1 + len(repair_responses),
            "rejected_candidates": rejected + repair_rejected + unresolved_issues,
            "unresolved_candidate_issues": unresolved_issues,
            "candidate_nature": "SOURCE_LINKED_ANALYSIS_PREMISE_NOT_ENGINEERING_APPROVAL",
        }

    @staticmethod
    def _function_context(function: FunctionDefinition) -> dict[str, Any]:
        context = {key: getattr(function, key) for key in (
            "name", "output", "description", "preconditions", "triggers",
            "odd_constraints", "fallback_behavior", "consequences",
        )}
        # The original source identity stays on FunctionDefinition/in audit.
        # The model needs the unabridged excerpt, not repeated file metadata.
        context.update({f"source[{index}]": source.excerpt
                        for index, source in enumerate(function.sources)})
        return {key: value for key, value in context.items() if value}

    @classmethod
    def _derivation(cls, function: FunctionDefinition, item: dict) -> dict:
        value = item.get("derivation")
        required = {"function_refs", "subject", "quantity", "basis", "assumptions"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("derivation requires function_refs, subject, quantity, basis, assumptions")
        context = cls._function_context(function)
        refs = value["function_refs"]
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(ref, str) or ref not in context or not context[ref] for ref in refs)):
            raise ValueError("derivation.function_refs must reference nonempty FunctionContext fields")
        if any(not isinstance(value[key], str) or not value[key].strip()
               for key in ("subject", "quantity", "basis")):
            raise ValueError("derivation subject, quantity and basis must be nonempty strings")
        assumptions = value["assumptions"]
        if not isinstance(assumptions, list) or any(not isinstance(v, str) or not v.strip() for v in assumptions):
            raise ValueError("derivation.assumptions must be an array of explicit assumptions")
        # Validate anchors, not free-form semantic plausibility. A source ID or
        # a classification label cannot establish a component in this project.
        values = [context[ref] for ref in refs]
        subject = normalize_source_text(value["subject"])
        if not any(subject in normalize_source_text(v if isinstance(v, str) else " ".join(v)) for v in values):
            raise ValueError("derivation.subject is absent from the referenced function/source text")
        quantity = value["quantity"].strip()
        if re.search(r"或|以及|、|[/／;；]|\b(?:or|and)\b", quantity, re.I):
            raise ValueError("derivation.quantity must name one deviation quantity; split distinct deviations")
        description = str(item.get("description", ""))
        if subject not in normalize_source_text(description) or normalize_source_text(quantity) not in normalize_source_text(description):
            raise ValueError("description must retain derivation.subject and quantity")
        if re.search(re.escape(quantity) + r"\s*(?:或|[/／]|\bor\b)", description, re.I):
            raise ValueError("description combines alternative deviation quantities")
        return {**value, "validation_status": "SOURCE_LINKED_ANALYSIS_PREMISE"}

    def _request(
        self,
        function: FunctionDefinition,
        applicable: list[GuidewordAssessment],
        *,
        coverage_repair: bool = False,
        existing: list[MalfunctionCandidate] | None = None,
        rejected: list[dict] | None = None,
    ) -> LLMRequest:
        guidewords = [item.guideword for item in applicable]
        classification_contract = ""
        if self.component_categories:
            classification_contract = (
                f"component_category必须从{list(self.component_categories)}中选择一个；"
                "它表示该失效直接所属的部件/能力类别，不得自行创造类别。"
            )
        failure_type_contract = ""
        if self.failure_types:
            taxonomy_context = [
                {"id": item.canonical_id, "description": item.description}
                for item in self.failure_types
            ]
            failure_type_contract = (
                "failure_type is required and must be exactly one canonical ID from "
                f"FailureTypeTaxonomy={json.dumps(taxonomy_context, ensure_ascii=False)}. "
                "Choose according to the malfunction's concrete semantics; do not map it mechanically from the guideword and do not invent a free-string value."
            )
        repair_instruction = ""
        if coverage_repair:
            repair_instruction = (
                "这是一次且仅一次的覆盖补全。只允许为MissingGuidewords生成候选；"
                "不得重复ExistingCandidates，不得返回其他Guideword。"
                f"\nMissingGuidewords={guidewords}"
                f"\nExistingCandidates={[(item.malfunction_id, item.guideword, item.description) for item in (existing or [])]}\n"
                f"\nRejectedCandidates={json.dumps(rejected or [], ensure_ascii=False)}\n"
            )
        schema = self._response_schema()
        candidate_schema = schema["properties"]["candidates"]["items"]
        return LLMRequest(
            task="derive_malfunctions_and_hazards",
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"FunctionContext={json.dumps(self._function_context(function), ensure_ascii=False, separators=(',', ':'))}\n"
                f"ApplicableGuidewords={guidewords}\n"
                f"GuidewordBasis={json.dumps({a.guideword: a.rationale for a in applicable}, ensure_ascii=False, separators=(',', ':'))}\n"
                "必须为ApplicableGuidewords中的每一个Guideword至少返回一个Malfunction候选；"
                "不得遗漏，不得增加列表外Guideword。"
                + repair_instruction +
                f"返回candidates数组；每项必须含{','.join(candidate_schema['required'])}。"
                "description、functional_effect、vehicle_level_hazard位于候选顶层；"
                f"derivation仅含{','.join(candidate_schema['properties']['derivation']['required'])}。"
                "derivation.function_refs引用FunctionContext的键，"
                "subject逐字引用对应功能主体，quantity为单个偏差量名称（不含单位），basis说明推导，assumptions单列补充假设。"
                "status只能为PENDING。"
                + ("、component_category。" if self.component_categories else "。")
                + (" failure_type is also required." if self.failure_types else "")
                + classification_contract +
                failure_type_contract +
                "causal_chain是至少两个非空字符串的数组，从功能偏差到车辆行为及危险状态。"
                "source由系统绑定；这只追溯功能依据，不表示故障或假设是原文事实。"
                + CONFIDENCE_PROMPT_CONTRACT
            ),
            schema_name="MalfunctionHazardCandidateList",
            prompt_version=self.PROMPT_VERSION,
            metadata={
                "function_id": function.function_id,
                "coverage_repair": coverage_repair,
                "guideword_count": len(guidewords),
            },
            # Reasoning and the final candidate JSON share the Provider cap.
            # Respect its configured allowance rather than truncating at 8k.
            max_tokens=int(getattr(getattr(self.client, "config", None), "max_tokens", 8192)),
            response_schema=schema,
        )

    def _parse_candidates(
        self,
        function: FunctionDefinition,
        data: dict[str, Any],
        assessments: list[GuidewordAssessment],
        allowed_assessments: list[GuidewordAssessment],
        *,
        rejected: list[dict] | None = None,
    ) -> list[MalfunctionCandidate]:
        raw = data.get("candidates")
        if not isinstance(raw, list):
            raise ValueError("LLM输出缺少candidates数组")
        assessment_by_guideword = {item.guideword: item for item in assessments}
        assessment_by_folded = {
            item.guideword.casefold(): item for item in assessments
        }
        allowed_guideword_ids = {
            item.guideword_id for item in allowed_assessments
        }
        candidates: list[MalfunctionCandidate] = []
        seen_ids: set[str] = set()
        seen_descriptions: set[str] = set()
        for index, item in enumerate(raw):
            guideword = (
                str(item.get("guideword", "")).strip()
                if isinstance(item, dict) else ""
            )
            assessment = assessment_by_guideword.get(guideword)
            normalized_item = dict(item) if isinstance(item, dict) else item
            if assessment is None and guideword:
                assessment = assessment_by_folded.get(guideword.casefold())
                if assessment is not None and isinstance(normalized_item, dict):
                    print(
                        "[HARA] malfunction candidate normalized "
                        f"function={function.function_id} index={index} "
                        f"field=guideword from={guideword!r} to={assessment.guideword!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    normalized_item["guideword"] = assessment.guideword
                    guideword = assessment.guideword
            if (
                isinstance(normalized_item, dict)
                and not str(normalized_item.get("malfunction_id", "")).strip()
            ):
                normalized_item["malfunction_id"] = (
                    f"{function.function_id}-{guideword or 'UNKNOWN'}-{index + 1}"
                )
                print(
                    "[HARA] malfunction candidate normalized "
                    f"function={function.function_id} index={index} "
                    "field=malfunction_id reason=blank_model_local_id",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                malfunction_guideword_gate.validate_malfunction_guideword_gate(
                    function_id=function.function_id,
                    guideword=guideword,
                    guideword_id=(assessment.guideword_id if assessment else ""),
                    assessment=assessment,
                    scheduled_guideword_ids=allowed_guideword_ids,
                )
                candidate = self._parse(function, normalized_item, assessment)
                try:
                    candidate.derivation = self._derivation(function, normalized_item)
                except ValueError as error:
                    if rejected is not None:
                        rejected.append({"guideword": guideword, "description": candidate.description, "reason": str(error)})
                    raise
                self._validate_failure_type(candidate)
                if (
                    self.component_categories
                    and candidate.component_category not in self.component_categories
                ):
                    print(
                        "[HARA] malfunction component classification unresolved "
                        f"function={function.function_id} "
                        f"malfunction={candidate.malfunction_id} "
                        f"value={candidate.component_category!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    candidate.component_category = ""
                hazard_lower = candidate.vehicle_level_hazard.lower()
                if any(term in hazard_lower for term in self.INJURY_TERMS):
                    raise ValueError("vehicle-level hazard contains an injury outcome")
                if candidate.malfunction_id in seen_ids:
                    original_id = candidate.malfunction_id
                    suffix = index + 1
                    while candidate.malfunction_id in seen_ids:
                        candidate.malfunction_id = f"{original_id}-{suffix}"
                        suffix += 1
                    print(
                        "[HARA] malfunction candidate normalized "
                        f"function={function.function_id} index={index} "
                        f"field=malfunction_id reason=duplicate_model_local_id "
                        f"from={original_id!r} to={candidate.malfunction_id!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                if candidate.description in seen_descriptions:
                    raise ValueError("duplicate malfunction description")
            except MalfunctionGuidewordGateViolation as error:
                print(
                    "[HARA] malfunction guideword gate violation "
                    f"function={function.function_id} index={index} "
                    f"guideword={guideword or '<missing>'} reason={error}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            except _FailureTypeContractViolation:
                raise
            except (TypeError, ValueError) as error:
                print(
                    "[HARA] malfunction candidate filtered "
                    f"function={function.function_id} index={index} "
                    f"guideword={guideword or '<missing>'} reason={error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            seen_ids.add(candidate.malfunction_id)
            seen_descriptions.add(candidate.description)
            candidates.append(candidate)
        return candidates

    def _validate_failure_type(self, candidate: MalfunctionCandidate) -> None:
        if not self.failure_types:
            return
        value = candidate.failure_type.strip()
        if not value:
            raise _FailureTypeContractViolation(
                "MALFUNCTION_FAILURE_TYPE_CONTRACT_VIOLATION: missing failure_type "
                f"malfunction={candidate.malfunction_id}"
            )
        if value not in self.failure_type_ids:
            raise _FailureTypeContractViolation(
                "MALFUNCTION_FAILURE_TYPE_CONTRACT_VIOLATION: invalid failure_type "
                f"malfunction={candidate.malfunction_id} value={value!r}"
            )

    @staticmethod
    def _merge_candidates(
        function: FunctionDefinition,
        existing: list[MalfunctionCandidate],
        additions: list[MalfunctionCandidate],
    ) -> None:
        """Keep valid partial repairs while preserving semantic uniqueness."""
        seen_ids = {item.malfunction_id for item in existing}
        seen_descriptions = {item.description for item in existing}
        for index, candidate in enumerate(additions, start=1):
            if candidate.description in seen_descriptions:
                print(
                    "[HARA] malfunction repair candidate filtered "
                    f"function={function.function_id} guideword={candidate.guideword} "
                    "reason=duplicate malfunction description across repair calls",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if candidate.malfunction_id in seen_ids:
                original_id = candidate.malfunction_id
                suffix = index
                while candidate.malfunction_id in seen_ids:
                    candidate.malfunction_id = f"{original_id}-R{suffix}"
                    suffix += 1
                print(
                    "[HARA] malfunction candidate normalized "
                    f"function={function.function_id} guideword={candidate.guideword} "
                    f"field=malfunction_id reason=duplicate_across_repair_calls "
                    f"from={original_id!r} to={candidate.malfunction_id!r}",
                    file=sys.stderr,
                    flush=True,
                )
            seen_ids.add(candidate.malfunction_id)
            seen_descriptions.add(candidate.description)
            existing.append(candidate)

    @staticmethod
    def _missing_guidewords(
        candidates: list[MalfunctionCandidate],
        applicable: list[GuidewordAssessment],
    ) -> list[str]:
        covered = {item.guideword for item in candidates}
        return sorted({item.guideword for item in applicable} - covered)

    @staticmethod
    def _parse(
        function: FunctionDefinition,
        item: Any,
        assessment: GuidewordAssessment | None = None,
    ) -> MalfunctionCandidate:
        if not isinstance(item, dict):
            raise ValueError("candidates数组元素必须为object")
        raw_chain = item.get("causal_chain")
        causal_chain, normalization = MalfunctionHazardAgent._normalize_causal_chain(
            raw_chain
        )
        if normalization:
            print(
                "[HARA] malfunction candidate normalized "
                f"function={function.function_id} "
                f"malfunction={item.get('malfunction_id', '')} "
                f"field=causal_chain mode={normalization}",
                file=sys.stderr,
                flush=True,
            )
        sources, source_origin = resolve_malfunction_sources(function, assessment)
        candidate = MalfunctionCandidate(
            malfunction_id=str(item.get("malfunction_id", "")).strip(),
            function_id=function.function_id,
            guideword=str(item.get("guideword", "")).strip(),
            description=str(item.get("description", "")).strip(),
            functional_effect=str(item.get("functional_effect", "")).strip(),
            vehicle_level_hazard=str(item.get("vehicle_level_hazard", "")).strip(),
            causal_chain=causal_chain,
            guideword_id=assessment.guideword_id if assessment is not None else "",
            component_category=str(item.get("component_category", "")).strip(),
            failure_type=str(item.get("failure_type", "")).strip(),
            sources=sources,
            status=ReviewStatus.PENDING,
            confidence=parse_confidence(
                item.get("confidence"),
                field_name=(
                    "Malfunction confidence validation failed: "
                    f"function={function.function_id} malfunction={item.get('malfunction_id', '')}"
                ),
            ),
        )
        if sources:
            print(
                "[HARA] malfunction source resolved "
                f"function={function.function_id} malfunction={candidate.malfunction_id} "
                f"guideword={candidate.guideword} source_count={len(sources)} "
                f"source_origin={source_origin}",
                file=sys.stderr,
                flush=True,
            )
        else:
            guideword_count = len(assessment.sources) if assessment is not None else 0
            print(
                "[HARA] malfunction source unresolved "
                f"function={function.function_id} malfunction={candidate.malfunction_id} "
                f"guideword={candidate.guideword} guideword_sources={guideword_count} "
                f"function_sources={len(function.sources)}",
                file=sys.stderr,
                flush=True,
            )
        return candidate

    @staticmethod
    def _normalize_causal_chain(value: Any) -> tuple[list[str], str]:
        if isinstance(value, list) and not isinstance(value, (str, bytes)):
            if any(not isinstance(item, str) for item in value):
                raise ValueError("Malfunction causal_chain must be an array of strings")
            return [item.strip() for item in value if item.strip()], ""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Malfunction causal_chain must be an array of strings")

        text = value.strip()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if (
                isinstance(decoded, list)
                and not isinstance(decoded, (str, bytes))
                and all(isinstance(item, str) for item in decoded)
            ):
                parts = [item.strip() for item in decoded if item.strip()]
                if len(parts) >= 2:
                    return parts, "json_array_string"

        parts = [
            item.strip()
            for item in re.split(r"\s*(?:--?>|=>|→|⇒|➜)\s*", text)
            if item.strip()
        ]
        if len(parts) >= 2:
            return parts, "explicit_arrow_string"
        raise ValueError(
            "Malfunction causal_chain string has no explicit segment boundary; "
            "expected an array of at least two strings"
        )

    def _validate(self, candidates: list[MalfunctionCandidate],
                  applicable: list[GuidewordAssessment], *,
                  require_coverage: bool = True):
        allowed = {item.guideword for item in applicable}
        allowed_by_name = {item.guideword: item for item in applicable}
        ids = [item.malfunction_id for item in candidates]
        descriptions = [item.description for item in candidates]
        if len(ids) != len(set(ids)) or len(descriptions) != len(set(descriptions)):
            raise ValueError("Malfunction候选存在重复ID或重复描述")
        invalid = sorted({item.guideword for item in candidates if item.guideword not in allowed})
        if invalid:
            raise ValueError(f"Malfunction使用了未判定适用的Guideword: {invalid}")
        covered = {item.guideword for item in candidates}
        missing = sorted(allowed - covered)
        if require_coverage and missing:
            raise ValueError(f"适用Guideword缺少Malfunction候选: {missing}")
        for item in candidates:
            assessment = allowed_by_name[item.guideword]
            malfunction_guideword_gate.validate_malfunction_guideword_gate(
                function_id=item.function_id,
                guideword=item.guideword,
                guideword_id=item.guideword_id,
                assessment=assessment,
                scheduled_guideword_ids={entry.guideword_id for entry in applicable},
            )
            hazard_lower = item.vehicle_level_hazard.lower()
            if any(term in hazard_lower for term in self.INJURY_TERMS):
                raise ValueError(f"Hazard不得直接写伤害结果: {item.malfunction_id}")
            if not item.sources or not item.sources[0].location:
                raise ValueError(f"Malfunction缺少来源位置: {item.malfunction_id}")
