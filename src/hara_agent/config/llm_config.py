from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 180.0
    max_tokens: int = 32768
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    extraction_thinking: str = "disabled"
    guideword_thinking: str = "disabled"
    malfunction_thinking: str = "default"
    scenario_thinking: str = "default"
    scenario_batch_max_chars: int = 24000
    scenario_batch_max_items: int = 2
    scenario_max_split_depth: int = 8
    scenario_causal_evidence_budget: int = 10
    scenario_response_format: str = "none"

    @classmethod
    def from_env(cls, prefix: str = "HARA_LLM_") -> "LLMConfig":
        provider = os.getenv(f"{prefix}PROVIDER", "openai-compatible")
        return cls(
            provider=provider,
            base_url=os.getenv(f"{prefix}BASE_URL", ""),
            model=os.getenv(f"{prefix}MODEL", ""),
            api_key=os.getenv(f"{prefix}API_KEY", ""),
            timeout_seconds=float(os.getenv(f"{prefix}TIMEOUT_SECONDS", "180")),
            max_tokens=int(os.getenv(f"{prefix}MAX_TOKENS", "32768")),
            max_retries=int(os.getenv(f"{prefix}MAX_RETRIES", "2")),
            retry_backoff_seconds=float(os.getenv(f"{prefix}RETRY_BACKOFF_SECONDS", "1")),
            extraction_thinking=os.getenv(
                f"{prefix}EXTRACTION_THINKING", "disabled"
            ).strip().lower(),
            guideword_thinking=os.getenv(
                f"{prefix}GUIDEWORD_THINKING", "disabled"
            ).strip().lower(),
            malfunction_thinking=os.getenv(
                f"{prefix}MALFUNCTION_THINKING", "default"
            ).strip().lower(),
            scenario_thinking=os.getenv(
                f"{prefix}SCENARIO_THINKING", "default"
            ).strip().lower(),
            scenario_batch_max_chars=int(os.getenv(
                f"{prefix}SCENARIO_BATCH_MAX_CHARS", str(cls.scenario_batch_max_chars)
            )),
            scenario_batch_max_items=int(os.getenv(
                f"{prefix}SCENARIO_BATCH_MAX_ITEMS", str(cls.scenario_batch_max_items)
            )),
            scenario_max_split_depth=int(os.getenv(
                f"{prefix}SCENARIO_MAX_SPLIT_DEPTH", "8"
            )),
            scenario_causal_evidence_budget=int(os.getenv(
                f"{prefix}SCENARIO_CAUSAL_EVIDENCE_BUDGET", "10"
            )),
            scenario_response_format=os.getenv(
                f"{prefix}SCENARIO_RESPONSE_FORMAT",
                "json_object" if provider == "volcengine-agent-plan" else "none",
            ).strip().lower(),
        )

    def validate(self):
        supported_providers = {"openai-compatible", "volcengine-agent-plan"}
        if self.provider not in supported_providers:
            raise ValueError(f"不支持的LLM Provider: {self.provider}")
        if not self.base_url or not self.model or not self.api_key:
            raise ValueError("LLM配置必须包含base_url、model和api_key")
        if not self.base_url.lower().startswith(("http://", "https://")):
            raise ValueError("LLM base_url必须为HTTP(S)地址")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds必须大于0")
        if self.max_tokens <= 0:
            raise ValueError("LLM max_tokens必须大于0")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("LLM max_retries必须在0到5之间")
        if self.retry_backoff_seconds < 0:
            raise ValueError("LLM retry_backoff_seconds不得为负数")
        if self.extraction_thinking not in {"default", "disabled", "enabled", "auto"}:
            raise ValueError(
                "HARA_LLM_EXTRACTION_THINKING必须为default、disabled、enabled或auto"
            )
        if self.guideword_thinking not in {"default", "disabled", "enabled", "auto"}:
            raise ValueError(
                "HARA_LLM_GUIDEWORD_THINKING必须为default、disabled、enabled或auto"
            )
        if self.malfunction_thinking not in {"default", "disabled", "enabled", "auto"}:
            raise ValueError(
                "HARA_LLM_MALFUNCTION_THINKING必须为default、disabled、enabled或auto"
            )
        if self.scenario_thinking not in {"default", "disabled", "enabled", "auto"}:
            raise ValueError(
                "HARA_LLM_SCENARIO_THINKING必须为default、disabled、enabled或auto"
            )
        if self.scenario_batch_max_chars <= 0:
            raise ValueError("HARA_LLM_SCENARIO_BATCH_MAX_CHARS必须大于0")
        if self.scenario_batch_max_items <= 0:
            raise ValueError("HARA_LLM_SCENARIO_BATCH_MAX_ITEMS必须大于0")
        if self.scenario_causal_evidence_budget <= 0:
            raise ValueError(
                "HARA_LLM_SCENARIO_CAUSAL_EVIDENCE_BUDGET must be greater than 0"
            )
        if self.scenario_max_split_depth < 0:
            raise ValueError("HARA_LLM_SCENARIO_MAX_SPLIT_DEPTH不得小于0")
        if self.scenario_response_format not in {"none", "json_object"}:
            raise ValueError("HARA_LLM_SCENARIO_RESPONSE_FORMAT必须为none或json_object")
