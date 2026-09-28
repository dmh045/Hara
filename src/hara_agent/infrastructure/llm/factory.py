from __future__ import annotations

from hara_agent.config import LLMConfig

from .openai_compatible import OpenAICompatibleClient, Transport
from .provider_budget import ProviderAttemptBudget
from .protocol import LLMClient


def create_llm_client(config: LLMConfig, transport: Transport | None = None,
                      attempt_budget: ProviderAttemptBudget | None = None) -> LLMClient:
    config.validate()
    if config.provider in {"openai-compatible", "volcengine-agent-plan"}:
        return OpenAICompatibleClient(
            config, transport=transport, attempt_budget=attempt_budget,
        )
    raise ValueError(f"不支持的LLM Provider: {config.provider}")
