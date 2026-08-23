from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


API_STYLE_OLLAMA = "ollama"
API_STYLE_OPENAI_COMPATIBLE = "openai_compatible"
SUPPORTED_API_STYLES = {API_STYLE_OLLAMA, API_STYLE_OPENAI_COMPATIBLE}


class ChatModelConfigurationError(ValueError):
    pass


class MissingChatModelAPIKeyError(ChatModelConfigurationError):
    pass


class ChatModelLike(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any: ...

    def with_structured_output(self, schema: type, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ChatModelConfig:
    provider: str
    api_style: str
    model: str
    base_url: str
    api_key: str = ""
    api_key_env: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 0

    def validate(self) -> None:
        if not self.provider.strip():
            raise ChatModelConfigurationError("LLM provider cannot be empty.")
        if self.api_style not in SUPPORTED_API_STYLES:
            supported = ", ".join(sorted(SUPPORTED_API_STYLES))
            raise ChatModelConfigurationError(
                f"Unsupported LLM API style '{self.api_style}'. Expected one of: {supported}."
            )
        if not self.model.strip():
            raise ChatModelConfigurationError("LLM model cannot be empty.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ChatModelConfigurationError(
                "LLM base URL must be an absolute HTTP or HTTPS URL."
            )
        if self.api_style == API_STYLE_OPENAI_COMPATIBLE and not self.api_key.strip():
            variable = self.api_key_env or "LLM_API_KEY"
            raise MissingChatModelAPIKeyError(
                f"Missing API key. Set {variable} for {self.provider}."
            )
        if self.timeout_seconds <= 0:
            raise ChatModelConfigurationError("LLM timeout must be positive.")
        if self.max_retries < 0:
            raise ChatModelConfigurationError("LLM max retries cannot be negative.")

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_style": self.api_style,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(self.api_key.strip()),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def get_chat_model(config: ChatModelConfig) -> ChatModelLike:
    """Create a LangChain chat model without exposing provider details to RAG code."""
    config.validate()
    if config.api_style == API_STYLE_OLLAMA:
        return ChatOllama(
            model=config.model,
            base_url=config.base_url.rstrip("/"),
            temperature=0,
        )
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url.rstrip("/"),
        temperature=0,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
