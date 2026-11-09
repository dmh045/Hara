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


# Preserve the established import surface for callers of this module.
MalfunctionGuidewordGateViolation = (
    malfunction_guideword_gate.MalfunctionGuidewordGateViolation
)


class _FailureTypeContractViolation(ValueError):
    """A configured canonical failure_type was omitted or not governed."""


class MalfunctionHazardAgent:
    """Generate traceable malfunction and vehicle-level hazard candidates."""

    PROMPT_VERSION = "malfunction-hazard-v7-bounded-classification-zh-v1"
    SYSTEM_PROMPT = REPORT_LANGUAGE_PROMPT + """你是汽车功能安全HARA分析助手。只处理已判定适用的Guideword。Malfunction描述功能相对预期行为的偏差；functional_effect描述功能/车辆行为影响；vehicle_level_hazard描述可能造成伤害的车辆级危险状态，不得直接写人员伤亡。causal_chain必须明确从失效到车辆级危险状态的至少两段因果关系，并且必须是JSON字符串数组，不得返回单个字符串。不得复制无关子系统模板。"""

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
            "status": {"type": "string"},
        }
        required = [
            "malfunction_id", "guideword", "description", "functional_effect",
            "vehicle_level_hazard", "causal_chain", "confidence", "status",
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
        candidates = self._parse_candidates(
            function, response.data, assessments, applicable,
        )
        self._validate(candidates, applicable, require_coverage=False)
        missing = self._missing_guidewords(candidates, applicable)
        repair_responses = []
        if missing:
            print(
                "[HARA] malfunction coverage repair "
                f"function={function.function_id} missing={missing} attempt=1/1",
                file=sys.stderr,
                flush=True,
            )
            repair_request = self._request(
                function,
                [item for item in applicable if item.guideword in set(missing)],
                coverage_repair=True,
                existing=candidates,
            )
            repair_response = self.client.complete_json(repair_request)
            repair_responses.append(repair_response)
            repaired = self._parse_candidates(
                function,
                repair_response.data,
                assessments,
                [item for item in applicable if item.guideword in set(missing)],
            )
            self._validate(
                repaired,
                [item for item in applicable if item.guideword in set(missing)],
                require_coverage=False,
            )
            self._merge_candidates(function, candidates, repaired)

        remaining = self._missing_guidewords(candidates, applicable)
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
            )
            targeted_response = self.client.complete_json(targeted_request)
            repair_responses.append(targeted_response)
            targeted = self._parse_candidates(
                function, targeted_response.data, assessments, target,
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
        for candidate in candidates:
            candidate.status = ReviewStatus.FINALIZED
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
        }

    def _request(
        self,
        function: FunctionDefinition,
        applicable: list[GuidewordAssessment],
        *,
        coverage_repair: bool = False,
        existing: list[MalfunctionCandidate] | None = None,
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
            )
        return LLMRequest(
            task="derive_malfunctions_and_hazards",
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"Function={function.name}\nOutput={function.output}\nDescription={function.description}\n"
                f"ApplicableGuidewords={guidewords}\n"
                "必须为ApplicableGuidewords中的每一个Guideword至少返回一个Malfunction候选；"
                "不得遗漏，不得增加列表外Guideword。"
                + repair_instruction +
                "必须返回一个JSON object，顶层必须包含candidates字段，且candidates必须为JSON array。"
                "当前任务不需要额外顶层字段。不得使用malfunctions、results、items、candidate_list或hazards"
                "替代candidates；不得直接返回JSON array；不得在JSON前后添加说明文字。"
                "candidates每项包含malfunction_id、guideword、description、functional_effect、"
                "vehicle_level_hazard、causal_chain、confidence、status"
                + ("、component_category。" if self.component_categories else "。")
                + (" failure_type is also required." if self.failure_types else "")
                + classification_contract +
                failure_type_contract +
                "causal_chain必须是至少包含两个非空字符串的JSON array，例如"
                "[\"功能偏差\",\"车辆行为异常\",\"车辆级危险状态\"]；"
                "即使因果关系可写成一句话，也禁止把causal_chain写成string。"
                "source由系统从Function/Guideword证据确定性传播，禁止生成source字段。"
                + CONFIDENCE_PROMPT_CONTRACT
            ),
            schema_name="MalfunctionHazardCandidateList",
            prompt_version=self.PROMPT_VERSION,
            metadata={
                "function_id": function.function_id,
                "coverage_repair": coverage_repair,
                "guideword_count": len(guidewords),
            },
            max_tokens=8192,
            response_schema=self._response_schema(),
        )

    def _parse_candidates(
        self,
        function: FunctionDefinition,
        data: dict[str, Any],
        assessments: list[GuidewordAssessment],
        allowed_assessments: list[GuidewordAssessment],
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
