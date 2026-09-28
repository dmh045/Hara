from __future__ import annotations

import json
import http.client
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from hara_agent.config import LLMConfig

from .provider_budget import ProviderAttemptBudget
from .protocol import LLMRequest, LLMResponse


Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


class TransientLLMError(RuntimeError):
    """A timeout, throttling response or server error that may succeed on retry."""

    def __init__(self, message: str, category: str = "network_error"):
        super().__init__(message)
        self.category = category


class LLMOutputLimitError(ValueError):
    """The provider stopped before returning one complete JSON result."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class LLMEmptyOutputError(ValueError):
    """The provider returned no machine-readable content."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class LLMTimeoutError(RuntimeError):
    """A request exhausted its task-specific timeout attempts."""

    def __init__(self, message: str, *, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class LLMTransportError(RuntimeError):
    """Non-timeout transient transport failures exhausted bounded retries."""

    def __init__(self, message: str, *, attempts: int, category: str,
                 error_counts: dict[str, int]):
        super().__init__(message)
        self.attempts = attempts
        self.category = category
        self.error_counts = dict(error_counts)


class LLMQuotaExceededError(RuntimeError):
    """A non-transient provider quota exhaustion that must not be retried."""

    def __init__(self, *, provider_code: str, reset_at: str = ""):
        message = f"LLM account quota exceeded; provider_code={provider_code}"
        if reset_at:
            message += f"; reset_at={reset_at}"
        message += "; request was not retried because quota exhaustion is non-transient"
        super().__init__(message)
        self.provider_code = provider_code
        self.reset_at = reset_at


class LLMSchemaContractError(ValueError):
    """Complete JSON that does not satisfy the requested schema envelope."""


class LLMJSONContractError(ValueError):
    """A complete provider response is not valid machine JSON."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class OpenAICompatibleClient:
    """Minimal JSON client for OpenAI-compatible chat-completions endpoints."""

    THINKING_CAPABLE_PROVIDERS = {"volcengine-agent-plan"}

    def __init__(self, config: LLMConfig, transport: Optional[Transport] = None,
                 attempt_budget: ProviderAttemptBudget | None = None):
        config.validate()
        self.config = config
        self.transport = transport or self._http_transport
        self.attempt_budget = attempt_budget

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        endpoint = self.config.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if request.response_format is not None:
            body["response_format"] = request.response_format
        thinking_mode = self._resolve_thinking_mode(request.task)
        if thinking_mode in {"disabled", "enabled", "auto"}:
            body["thinking"] = {"type": thinking_mode}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        input_characters = len(request.system_prompt) + len(request.user_prompt)
        request_context = self._request_context(request)
        print(
            "[HARA] LLM start "
            f"task={request.task}{request_context} schema_name={request.schema_name} "
            f"input_chars={input_characters} max_tokens={body['max_tokens']} "
            f"thinking={thinking_mode} "
            f"timeout={self.config.timeout_seconds:.1f}s attempts={self.config.max_retries + 1}",
            file=sys.stderr,
            flush=True,
        )
        try:
            response, transport_diagnostics = self._send_with_retry(
                endpoint, headers, encoded_body, request=request,
            )
        except Exception as exc:
            print(
                "[HARA] LLM failed "
                f"task={request.task}{request_context} "
                f"elapsed={time.monotonic() - started:.1f}s "
                f"type={self._exception_type(exc)}",
                file=sys.stderr,
                flush=True,
            )
            raise
        latency_seconds = time.monotonic() - started
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            keys = sorted(str(key) for key in response)[:12]
            raise ValueError(f"LLM响应结构错误；顶层字段: {keys}") from exc
        finish_reason = str(choice.get("finish_reason", "unknown"))
        reasoning_content = message.get("reasoning_content")
        raw_usage = dict(response.get("usage", {}) or {})
        diagnostic_usage = {
            key: value for key, value in raw_usage.items()
            if key in {
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_tokens", "output_tokens", "reasoning_tokens",
            } and isinstance(value, (int, float))
        }
        reasoning_tokens = self._reasoning_tokens(raw_usage)
        if reasoning_tokens is not None:
            diagnostic_usage["reasoning_tokens"] = reasoning_tokens
        print(
            "[HARA] LLM diagnostic "
            f"task={request.task}{request_context} schema_name={request.schema_name} "
            f"finish_reason={finish_reason} content_chars={len(str(content or ''))} "
            f"reasoning_chars={len(str(reasoning_content or ''))} "
            f"max_tokens={body['max_tokens']} usage={diagnostic_usage}",
            file=sys.stderr,
            flush=True,
        )
        if finish_reason == "length":
            raise LLMOutputLimitError(
                "LLM输出达到长度上限，结构化结果不完整；finish_reason=length；"
                f"本次max_tokens={body['max_tokens']}。",
                diagnostics={
                    "input_characters": input_characters,
                    "content_characters": len(str(content or "")),
                    "reasoning_characters": len(str(reasoning_content or "")),
                    "finish_reason": finish_reason,
                    "max_tokens": body["max_tokens"],
                    "prompt_tokens": raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")),
                    "completion_tokens": raw_usage.get("completion_tokens", raw_usage.get("output_tokens")),
                },
            )
        if not str(content or "").strip():
            raise LLMEmptyOutputError(
                f"LLM返回空内容；finish_reason={finish_reason}；"
                f"本次max_tokens={body['max_tokens']}。",
                diagnostics={
                    "finish_reason": finish_reason,
                    "max_tokens": body["max_tokens"],
                    "prompt_tokens": raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")),
                    "completion_tokens": raw_usage.get("completion_tokens", raw_usage.get("output_tokens")),
                },
            )
        format_retry_attempt = int(request.metadata.get("_format_retry_attempt", 0))
        markdown_fence_normalizations = 0
        try:
            if request.task == "assess_scenario_feasibility":
                data, fence_removed = self._parse_scenario_json_content(
                    content, request=request, finish_reason=finish_reason,
                )
                markdown_fence_normalizations = int(fence_removed)
                if fence_removed:
                    print(
                        "[HARA] LLM JSON normalized "
                        f"task={request.task}{request_context} normalization=markdown_fence",
                        file=sys.stderr, flush=True,
                    )
            else:
                data = self._parse_json_content(content, schema_name=request.schema_name)
        except LLMJSONContractError as exc:
            diagnostics = exc.diagnostics
            diagnostics.setdefault("request_id", str(response.get("id", "")))
            diagnostics.setdefault("finish_reason", finish_reason)
            diagnostics.setdefault(
                "prompt_tokens", raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
            )
            diagnostics.setdefault(
                "completion_tokens", raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
            )
            print(
                "[HARA] LLM JSON contract failure "
                f"task={request.task}{request_context} schema_name={request.schema_name} "
                f"finish_reason={finish_reason} content_chars={diagnostics['content_chars']} "
                f"error_line={diagnostics['json_error_line']} "
                f"error_column={diagnostics['json_error_column']} "
                f"error_position={diagnostics['json_error_position']} "
                f"starts_with_fence={str(diagnostics['starts_with_markdown_fence']).lower()} "
                f"ends_with_fence={str(diagnostics['ends_with_markdown_fence']).lower()} "
                f"outer_fence_removed={str(diagnostics['outer_fence_removed']).lower()} "
                f"format_retry={format_retry_attempt + 1}/1",
                file=sys.stderr, flush=True,
            )
            if (
                request.task == "assess_scenario_feasibility"
                and format_retry_attempt == 0
                and not bool(request.metadata.get("strict_no_format_retry"))
            ):
                retry_metadata = dict(request.metadata)
                retry_metadata["_format_retry_attempt"] = 1
                retry = self.complete_json(LLMRequest(
                    task=request.task,
                    system_prompt=request.system_prompt + (
                        "\nFORMAT RETRY: The previous response was not valid machine JSON. "
                        "Re-evaluate the same supplied inputs and return exactly one raw JSON object. "
                        "Do not use Markdown or code fences. Do not add text before or after JSON. "
                        "Do not change the requested schema."
                    ),
                    user_prompt=request.user_prompt,
                    schema_name=request.schema_name,
                    prompt_version=request.prompt_version,
                    metadata=retry_metadata,
                    max_tokens=request.max_tokens,
                    response_schema=request.response_schema,
                    response_format=request.response_format,
                ))
                retry.usage["json_contract_errors"] = int(
                    retry.usage.get("json_contract_errors", 0)
                ) + 1
                retry.usage["format_retry_calls"] = int(
                    retry.usage.get("format_retry_calls", 0)
                ) + 1
                retry.usage["format_retry_successes"] = 1
                retry.usage["format_retry_failures"] = 0
                return retry
            diagnostics["format_retry_calls"] = format_retry_attempt
            diagnostics["format_retry_failures"] = int(format_retry_attempt > 0)
            raise
        self._log_parsed_envelope(request, data)
        data = self._normalize_schema_envelope(data, request.schema_name)
        schema_repair_count = 0
        try:
            self._validate_schema_envelope(data, request.schema_name)
        except LLMSchemaContractError as exc:
            self._log_schema_mismatch(request, data, exc)
            if request.task == "repair_schema_envelope":
                raise
            repaired = self._repair_list_schema_envelope_once(request, data)
            if repaired is None:
                raise
            data = repaired
            schema_repair_count = 1
        usage = raw_usage
        usage.update({
            "latency_seconds": round(latency_seconds, 3),
            "input_characters": len(request.system_prompt) + len(request.user_prompt),
            "max_tokens": body["max_tokens"],
            "finish_reason": finish_reason,
            "cache_hit": False,
            "transport_attempts": transport_diagnostics["attempts"],
            "transient_error_counts": transport_diagnostics["error_counts"],
            "schema_repair_count": schema_repair_count,
            "reasoning_characters": len(str(reasoning_content or "")),
            "json_contract_errors": 0,
            "format_retry_calls": 0,
            "format_retry_successes": 0,
            "format_retry_failures": 0,
            "markdown_fence_normalizations": markdown_fence_normalizations,
        })
        result = LLMResponse(
            data=data,
            model=str(response.get("model") or self.config.model),
            request_id=str(response.get("id", "")),
            usage=usage,
        )
        print(
            "[HARA] LLM completed "
            f"task={request.task}{request_context} elapsed={latency_seconds:.1f}s "
            f"finish_reason={finish_reason} "
            f"prompt_tokens={raw_usage.get('prompt_tokens', raw_usage.get('input_tokens', 'unknown'))} "
            f"completion_tokens={raw_usage.get('completion_tokens', raw_usage.get('output_tokens', 'unknown'))} "
            f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'}",
            file=sys.stderr,
            flush=True,
        )
        return result

    @staticmethod
    def _is_extraction_task(task: str) -> bool:
        return (
            task == "extract_core_item_artifacts"
            or task.startswith("repair_core_item_artifact")
            or task.startswith("supplement_")
            or task.startswith("extract_targeted_project_facts:")
        )

    def _resolve_thinking_mode(self, task: str) -> str:
        configured_mode = None
        if self._is_extraction_task(task):
            configured_mode = self.config.extraction_thinking
        elif task == "assess_guideword_applicability":
            configured_mode = self.config.guideword_thinking
        elif task == "derive_malfunctions_and_hazards":
            configured_mode = self.config.malfunction_thinking
        elif task in {
            "assess_scenario_feasibility",
            "interpret_scenario_risk_facts",
            "select_scenario_synthesis",
        }:
            configured_mode = self.config.scenario_thinking
        elif task == "repair_schema_envelope":
            configured_mode = "disabled"
        if configured_mode is None:
            return "default"
        if self.config.provider not in self.THINKING_CAPABLE_PROVIDERS:
            return "not_sent_provider_unsupported"
        return configured_mode

    @staticmethod
    def _normalize_schema_envelope(data: Any, schema_name: str) -> Any:
        if schema_name == "MalfunctionHazardCandidateList":
            return data
        if schema_name in {
            "ScenarioFeasibilityAssessmentList",
            "ScenarioFeasibilityAssessmentV2List",
            "ScenarioRiskFacts",
        } and isinstance(data, list):
            # Let the strict envelope validator route a complete direct array
            # through deterministic wrapping rather than rejecting it here.
            return data
        if schema_name != "GuidewordAssessmentList":
            if not isinstance(data, dict):
                raise ValueError("LLM JSON顶层必须为object")
            return data
        if isinstance(data, list):
            return {"assessments": data}
        if not isinstance(data, dict):
            raise ValueError("GuidewordAssessmentList顶层必须为object或assessment array")
        if isinstance(data.get("assessments"), list):
            return data
        alias = data.get("guideword_assessments")
        if isinstance(alias, list):
            return {"assessments": alias}
        for wrapper in ("data", "result", "GuidewordAssessmentList"):
            nested = data.get(wrapper)
            if isinstance(nested, dict) and isinstance(nested.get("assessments"), list):
                return {"assessments": nested["assessments"]}
        return data

    @staticmethod
    def _validate_schema_envelope(data: Any, schema_name: str) -> None:
        if schema_name == "CoreItemArtifacts":
            if not isinstance(data, dict):
                raise LLMSchemaContractError(
                    "CoreItemArtifacts顶层必须为JSON object"
                )
            if not isinstance(data.get("item_definition"), dict):
                raise LLMSchemaContractError(
                    "CoreItemArtifacts.item_definition类型必须为dict"
                )
            if not isinstance(data.get("functions"), list):
                raise LLMSchemaContractError(
                    "CoreItemArtifacts.functions类型必须为list"
                )
            return
        required_envelopes = {
            "GuidewordAssessmentList": ("assessments", list),
            "MalfunctionHazardCandidateList": ("candidates", list),
            "ScenarioFeasibilityAssessmentList": ("assessments", list),
            "ScenarioFeasibilityAssessmentV2List": ("assessments", list),
            "ScenarioRiskFacts": ("results", list),
        }
        contract = required_envelopes.get(schema_name)
        if contract is None:
            return
        required_key, expected_type = contract
        if not isinstance(data, dict):
            raise LLMSchemaContractError(f"{schema_name}顶层必须为JSON object")
        if required_key not in data:
            raise LLMSchemaContractError(
                f"{schema_name}必须包含{required_key}"
            )
        if not isinstance(data[required_key], expected_type):
            raise LLMSchemaContractError(
                f"{schema_name}.{required_key}类型必须为{expected_type.__name__}"
            )

    @staticmethod
    def _request_context(request: LLMRequest) -> str:
        metadata = request.metadata or {}
        values = []
        for output_key, metadata_key in (
            ("function", "function_id"),
            ("malfunction", "malfunction_id"),
            ("batch", "parent_batch"),
            ("split_path", "split_path"),
            ("split_depth", "split_depth"),
            ("scenario_count", "scenario_count"),
        ):
            value = metadata.get(metadata_key)
            if value is not None and value != "":
                values.append(f"{output_key}={value}")
        return (" " + " ".join(values)) if values else ""

    @classmethod
    def _log_parsed_envelope(cls, request: LLMRequest, data: Any) -> None:
        keys = sorted(str(key) for key in data)[:20] if isinstance(data, dict) else []
        print(
            "[HARA] LLM parsed envelope "
            f"task={request.task}{cls._request_context(request)} "
            f"schema_name={request.schema_name} "
            f"top_level_type={type(data).__name__} top_level_keys={keys}",
            file=sys.stderr,
            flush=True,
        )

    @classmethod
    def _log_schema_mismatch(
        cls, request: LLMRequest, data: Any, exc: BaseException,
    ) -> None:
        keys = sorted(str(key) for key in data)[:20] if isinstance(data, dict) else []
        expected = {
            "MalfunctionHazardCandidateList": "candidates",
            "ScenarioRiskFacts": "results",
        }.get(request.schema_name, "assessments")
        print(
            "[HARA] LLM schema mismatch "
            f"task={request.task}{cls._request_context(request)} "
            f"schema_name={request.schema_name} expected_key={expected} "
            f"actual_keys={keys} reason={cls._safe_reason(exc)}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _is_complete_malfunction_candidate(data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        required = {
            "malfunction_id", "guideword", "description", "functional_effect",
            "vehicle_level_hazard", "causal_chain", "confidence", "status",
        }
        if not required.issubset(data):
            return False
        text_fields = required - {"causal_chain", "confidence"}
        if any(not isinstance(data[field], str) or not data[field].strip() for field in text_fields):
            return False
        if not isinstance(data["causal_chain"], list):
            return False
        confidence = data["confidence"]
        return (
            not isinstance(confidence, bool)
            and isinstance(confidence, (int, float))
            and 0.0 <= float(confidence) <= 1.0
        )

    @classmethod
    def _candidate_collection_for_repair(
        cls, data: Any,
    ) -> list[dict[str, Any]] | None:
        if cls._is_complete_malfunction_candidate(data):
            return [data]
        if isinstance(data, list):
            if not data:
                return []
            if all(cls._is_complete_malfunction_candidate(item) for item in data):
                return data
            return None
        if not isinstance(data, dict):
            return None
        collections = [value for value in data.values() if isinstance(value, list)]
        candidate_collections = [
            value for value in collections
            if value and all(cls._is_complete_malfunction_candidate(item) for item in value)
        ]
        if len(candidate_collections) == 1:
            return candidate_collections[0]
        return None

    @staticmethod
    def _is_complete_list_item(data: Any, schema_name: str) -> bool:
        if not isinstance(data, dict):
            return False
        if schema_name == "GuidewordAssessmentList":
            return (
                isinstance(data.get("guideword"), str)
                and bool(data["guideword"].strip())
                and isinstance(data.get("applicable"), bool)
                and isinstance(data.get("rationale"), str)
                and bool(data["rationale"].strip())
                and "confidence" in data
                and isinstance(data.get("disposition"), str)
            )
        if schema_name in {
            "ScenarioFeasibilityAssessmentList",
            "ScenarioFeasibilityAssessmentV2List",
        }:
            return (
                isinstance(data.get("scenario_id"), str)
                and bool(data["scenario_id"].strip())
                and all(isinstance(data.get(field), bool) for field in (
                    "physically_feasible", "functionally_relevant", "causally_relevant",
                ))
                and "confidence" in data
            )
        if schema_name == "ScenarioRiskFacts":
            identity_complete = all(
                isinstance(data.get(field), str) and bool(data[field].strip())
                for field in ("malfunction_id", "scenario_id", "fact_type", "status")
            )
            if not identity_complete:
                return False
            status = data["status"].upper()
            if status == "NOT_FOUND":
                return True
            return (
                status == "FOUND"
                and "value" in data
                and isinstance(data.get("unit"), str)
                and isinstance(data.get("evidence_ids"), list)
            )
        return False

    @classmethod
    def _list_collection_for_repair(
        cls, data: Any, schema_name: str,
    ) -> list[dict[str, Any]] | None:
        if schema_name == "MalfunctionHazardCandidateList":
            return cls._candidate_collection_for_repair(data)
        if cls._is_complete_list_item(data, schema_name):
            return [data]
        if isinstance(data, list):
            if not data:
                return []
            if all(cls._is_complete_list_item(item, schema_name) for item in data):
                return data
            return None
        if not isinstance(data, dict):
            return None
        collections = [value for value in data.values() if isinstance(value, list)]
        candidates = [
            value for value in collections
            if value and all(cls._is_complete_list_item(item, schema_name) for item in value)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _repair_list_schema_envelope_once(
        self, original_request: LLMRequest, data: Any,
    ) -> dict[str, Any] | None:
        original_items = self._list_collection_for_repair(
            data, original_request.schema_name,
        )
        if original_items is None:
            return None
        envelope_key = {
            "MalfunctionHazardCandidateList": "candidates",
            "GuidewordAssessmentList": "assessments",
            "ScenarioFeasibilityAssessmentList": "assessments",
            "ScenarioFeasibilityAssessmentV2List": "assessments",
            "ScenarioRiskFacts": "results",
        }.get(original_request.schema_name)
        if envelope_key is None:
            return None
        context = self._request_context(original_request)
        print(
            "[HARA] schema repair started "
            f"task={original_request.task}{context} "
            f"schema_name={original_request.schema_name} expected_key={envelope_key} "
            "mode=deterministic_wrap attempt=1/1",
            file=sys.stderr,
            flush=True,
        )
        repaired = {envelope_key: original_items}
        try:
            self._validate_schema_envelope(repaired, original_request.schema_name)
            if repaired[envelope_key] != original_items:
                raise LLMSchemaContractError(
                    "schema repair changed semantic items"
                )
        except Exception as exc:
            print(
                "[HARA] schema repair failed "
                f"function={original_request.metadata.get('function_id', 'unknown')} "
                f"attempt=1/1 reason={self._safe_reason(exc)}",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            "[HARA] schema repair completed "
            f"function={original_request.metadata.get('function_id', 'unknown')} "
            "mode=deterministic_wrap attempt=1/1",
            file=sys.stderr,
            flush=True,
        )
        return repaired

    def _send_with_retry(
        self, endpoint: str, headers: dict[str, str], body: bytes, *, request: LLMRequest,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        task = request.task
        context = self._request_context(request)
        configured_attempts = self.config.max_retries + 1
        attempts = min(configured_attempts, 2) if task == "assess_scenario_feasibility" else configured_attempts
        error_counts: dict[str, int] = {}
        for attempt in range(1, attempts + 1):
            budget_attempt_id = (
                self.attempt_budget.begin(request, model=self.config.model)
                if self.attempt_budget is not None else None
            )
            attempt_started = time.monotonic()
            print(
                f"[HARA] LLM request attempt task={task}{context} "
                f"attempt={attempt}/{attempts} configured_timeout={self.config.timeout_seconds:.1f}s "
                f"configured_deadline={self.config.timeout_seconds:.1f}s attempt_started={attempt_started:.3f}",
                file=sys.stderr,
                flush=True,
            )
            try:
                response = self.transport(
                    endpoint, headers, body, self.config.timeout_seconds,
                )
            except (
                TimeoutError, socket.timeout, ConnectionError,
                http.client.RemoteDisconnected, http.client.IncompleteRead,
                ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                ssl.SSLError, TransientLLMError,
            ) as exc:
                if budget_attempt_id is not None:
                    self.attempt_budget.finish(budget_attempt_id, error=exc)
                category = (
                    exc.category if isinstance(exc, TransientLLMError)
                    else self._exception_type(exc)
                )
                error_counts[category] = error_counts.get(category, 0) + 1
                attempt_elapsed = time.monotonic() - attempt_started
                if attempt >= attempts:
                    print(
                        f"[HARA] LLM transient failure; giving up after {attempts} "
                        f"attempts task={task}{context} attempt={attempt}/{attempts} "
                        f"type={category} configured_deadline={self.config.timeout_seconds:.1f}s "
                        f"attempt_elapsed={attempt_elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if "timeout" in str(category).lower():
                        raise LLMTimeoutError(
                            f"LLM请求在{attempts}次尝试后仍超时: {exc}",
                            attempts=attempts,
                        ) from exc
                    raise LLMTransportError(
                        f"LLM请求在{attempts}次尝试后仍发生transport failure: {category}",
                        attempts=attempts, category=str(category), error_counts=error_counts,
                    ) from exc
                delay = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[HARA] LLM transient failure task={task}{context} "
                    f"attempt={attempt}/{attempts} type={category} retry={attempt + 1}/{attempts} "
                    f"after={delay:.1f}s attempt_elapsed={attempt_elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            except BaseException as exc:
                if budget_attempt_id is not None:
                    self.attempt_budget.finish(budget_attempt_id, error=exc)
                raise
            if budget_attempt_id is not None:
                self.attempt_budget.finish(budget_attempt_id, response=response)
            return response, {"attempts": attempt, "error_counts": error_counts}
        raise RuntimeError("LLM请求未执行")

    @staticmethod
    def _exception_type(exc: BaseException) -> str:
        if isinstance(exc, LLMQuotaExceededError):
            return "quota_exceeded"
        if isinstance(exc, TransientLLMError):
            return exc.category
        if isinstance(exc, http.client.RemoteDisconnected):
            return "remote_disconnect"
        if isinstance(exc, http.client.IncompleteRead):
            return "incomplete_read"
        if isinstance(exc, ConnectionResetError):
            return "connection_reset"
        if isinstance(exc, ConnectionAbortedError):
            return "connection_aborted"
        if isinstance(exc, BrokenPipeError):
            return "broken_pipe"
        if isinstance(exc, ssl.SSLError):
            return "ssl_error"
        if isinstance(exc, TimeoutError):
            return "TimeoutError"
        if isinstance(exc, socket.timeout):
            return "socket.timeout"
        if isinstance(exc, ConnectionError):
            return "ConnectionError"
        return type(exc).__name__

    @staticmethod
    def _safe_reason(exc: BaseException) -> str:
        reason = " ".join(str(exc).split())[:200]
        reason = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED]", reason)
        reason = re.sub(
            r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", reason,
        )
        return reason or "n/a"

    @staticmethod
    def _reasoning_tokens(usage: dict[str, Any]) -> int | float | None:
        direct = usage.get("reasoning_tokens")
        if isinstance(direct, (int, float)):
            return direct
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            nested = details.get("reasoning_tokens")
            if isinstance(nested, (int, float)):
                return nested
        return None

    @staticmethod
    def _parse_json_content(
        content: Any, *, schema_name: str = "",
    ) -> dict[str, Any] | list[Any]:
        if isinstance(content, (dict, list)):
            return content
        text = str(content or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = OpenAICompatibleClient._extract_json_value(text, allow_array=(
                schema_name == "GuidewordAssessmentList"
            ))
        if not isinstance(parsed, (dict, list)):
            raise ValueError("LLM JSON顶层必须为object或array")
        return parsed

    @classmethod
    def _parse_scenario_json_content(
        cls, content: Any, *, request: LLMRequest, finish_reason: str,
    ) -> tuple[dict[str, Any] | list[Any], bool]:
        if isinstance(content, (dict, list)):
            return content, False
        text = str(content or "").strip()
        starts_with_fence = bool(re.match(r"^```(?:json)?\s*(?:\r?\n)?", text, re.IGNORECASE))
        ends_with_fence = bool(re.search(r"(?:\r?\n)?```$", text))
        fence_removed = False
        candidate = text
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*(?:\r?\n)(.*?)(?:\r?\n)```",
            text, re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            candidate = fenced.group(1).strip()
            fence_removed = True
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            diagnostics = {
                "task": request.task,
                "schema_name": request.schema_name,
                "finish_reason": finish_reason,
                "content_chars": len(text),
                "malfunction_id": request.metadata.get("malfunction_id", ""),
                "batch": request.metadata.get("parent_batch", ""),
                "split_path": request.metadata.get("split_path", ""),
                "split_depth": request.metadata.get("split_depth", ""),
                "scenario_count": request.metadata.get("scenario_count", ""),
                "json_error_type": type(exc).__name__,
                "json_error_line": exc.lineno,
                "json_error_column": exc.colno,
                "json_error_position": exc.pos,
                "starts_with_markdown_fence": starts_with_fence,
                "ends_with_markdown_fence": ends_with_fence,
                "outer_fence_removed": fence_removed,
                "first_non_ws_char": text[:1],
                "last_non_ws_char": text[-1:] if text else "",
                "brace_balance": candidate.count("{") - candidate.count("}"),
                "bracket_balance": candidate.count("[") - candidate.count("]"),
            }
            raise LLMJSONContractError(
                "Scenario provider response违反machine JSON contract",
                diagnostics,
            ) from exc
        if not isinstance(parsed, (dict, list)):
            raise LLMJSONContractError(
                "Scenario JSON顶层必须为object或可验证的assessment array",
                {
                    "task": request.task, "schema_name": request.schema_name,
                    "finish_reason": finish_reason, "content_chars": len(text),
                    "malfunction_id": request.metadata.get("malfunction_id", ""),
                    "batch": request.metadata.get("parent_batch", ""),
                    "split_path": request.metadata.get("split_path", ""),
                    "split_depth": request.metadata.get("split_depth", ""),
                    "scenario_count": request.metadata.get("scenario_count", ""),
                    "json_error_type": "TopLevelTypeError", "json_error_line": 1,
                    "json_error_column": 1, "json_error_position": 0,
                    "starts_with_markdown_fence": starts_with_fence,
                    "ends_with_markdown_fence": ends_with_fence,
                    "outer_fence_removed": fence_removed,
                },
            )
        return parsed, fence_removed

    @staticmethod
    def _extract_json_value(text: str, *, allow_array: bool = False) -> dict[str, Any] | list[Any]:
        """Extract one complete top-level JSON value from provider prose."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{" and not (allow_array and character == "["):
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) or (allow_array and isinstance(value, list)):
                return value
        preview = " ".join(text.split())[:240]
        raise ValueError(f"LLM未返回有效JSON；响应摘要: {preview!r}")

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Backward-compatible object-only JSON salvage helper."""
        value = OpenAICompatibleClient._extract_json_value(text)
        if not isinstance(value, dict):
            raise ValueError("LLM JSON顶层必须为object")
        return value

    @staticmethod
    def _quota_error(detail: str) -> LLMQuotaExceededError | None:
        """Recognize quota exhaustion without making ordinary 429s permanent."""
        try:
            payload = json.loads(detail)
        except (json.JSONDecodeError, TypeError):
            return None
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return None
        provider_code = str(error.get("code") or "").strip()
        provider_message = str(error.get("message") or "").strip()
        normalized_code = re.sub(r"[^a-z0-9]", "", provider_code.lower())
        quota_codes = {
            "accountquotaexceeded",
            "insufficientquota",
            "quotaexceeded",
        }
        weekly_quota_message = "exceeded the weekly usage quota" in provider_message.lower()
        if normalized_code not in quota_codes and not weekly_quota_message:
            return None
        reset_match = re.search(
            r"\breset\s+at\s+(.+?)(?=\.\s+(?:We\b|Please\b)|$)",
            provider_message,
            flags=re.IGNORECASE,
        )
        reset_at = reset_match.group(1).strip() if reset_match else ""
        return LLMQuotaExceededError(
            provider_code=provider_code or "quota_exceeded",
            reset_at=reset_at,
        )

    @staticmethod
    def _http_transport(url: str, headers: dict[str, str], body: bytes,
                        timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        deadline = time.monotonic() + timeout
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                chunks = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TransientLLMError(
                            "LLM whole-request deadline exceeded", category="timeout"
                        )
                    raw_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                    if raw_socket is not None:
                        raw_socket.settimeout(remaining)
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                payload = b"".join(chunks).decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            quota_error = OpenAICompatibleClient._quota_error(detail) if exc.code == 429 else None
            if quota_error is not None:
                raise quota_error from exc
            if exc.code == 429 or 500 <= exc.code < 600:
                raise TransientLLMError(
                    f"LLM HTTP暂时错误 {exc.code}: {detail[:240]}",
                    category="http_429" if exc.code == 429 else "http_5xx",
                ) from exc
            raise RuntimeError(f"LLM HTTP错误 {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TransientLLMError(
                    f"LLM连接超时: {exc.reason}", category="timeout"
                ) from exc
            raise TransientLLMError(
                f"LLM连接失败: {exc.reason}", category="network_error"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TransientLLMError(
                f"LLM读取超时: {exc}", category="timeout"
            ) from exc
        except http.client.IncompleteRead as exc:
            raise TransientLLMError(
                "LLM response body incomplete "
                f"received={len(exc.partial)} remaining={exc.expected}",
                category="incomplete_read",
            ) from exc
        except http.client.RemoteDisconnected as exc:
            raise TransientLLMError("LLM remote disconnected", category="remote_disconnect") from exc
        except ConnectionResetError as exc:
            raise TransientLLMError("LLM connection reset", category="connection_reset") from exc
        except ConnectionAbortedError as exc:
            raise TransientLLMError("LLM connection aborted", category="connection_aborted") from exc
        except BrokenPipeError as exc:
            raise TransientLLMError("LLM broken pipe", category="broken_pipe") from exc
        except ssl.SSLError as exc:
            category = "timeout" if isinstance(exc, socket.timeout) else "ssl_error"
            raise TransientLLMError(f"LLM TLS failure: {type(exc).__name__}", category=category) from exc
        if not payload.strip():
            raise ValueError("LLM返回空响应，请检查endpoint和认证配置")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("LLM HTTP响应顶层必须为object")
        return parsed
