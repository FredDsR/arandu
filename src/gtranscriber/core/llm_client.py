"""Unified LLM client using OpenAI SDK with configurable base_url.

Supports OpenAI, Ollama, and any OpenAI-compatible provider through
the OpenAI SDK's base_url parameter.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from gtranscriber.utils.text import GenerateResult, extract_thinking


class LLMProvider(Enum):
    """LLM provider types."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class LLMClient:
    """Unified LLM client using OpenAI SDK with configurable base_url.

    Supports OpenAI, Ollama, and any OpenAI-compatible provider.

    Examples:
        >>> # Using OpenAI
        >>> client = LLMClient(LLMProvider.OPENAI, "gpt-4", api_key="sk-...")
        >>> response = client.generate("What is the capital of France?")

        >>> # Using Ollama
        >>> client = LLMClient(LLMProvider.OLLAMA, "llama3.1:8b")
        >>> response = client.generate("What is the capital of France?")

        >>> # Using custom OpenAI-compatible endpoint
        >>> client = LLMClient(
        ...     LLMProvider.CUSTOM,
        ...     "model-name",
        ...     api_key="key",
        ...     base_url="http://localhost:8000/v1"
        ... )
    """

    # Default base URLs for known providers
    PROVIDER_URLS: ClassVar[dict[LLMProvider, str | None]] = {
        LLMProvider.OPENAI: None,  # Uses OpenAI default
        LLMProvider.OLLAMA: "http://localhost:11434/v1",
        LLMProvider.CUSTOM: None,  # Must be provided explicitly
    }

    def __init__(
        self,
        provider: LLMProvider,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            provider: The LLM provider type (openai, ollama, custom).
            model_id: The model identifier (e.g., "gpt-4", "llama3.1:8b").
            api_key: API key for authentication. Optional for Ollama.
            base_url: Custom base URL. If not provided, uses provider default.
        """
        self.provider = provider
        self.model_id = model_id

        # Determine base_url: explicit > provider default
        if base_url:
            self._base_url = base_url
        else:
            self._base_url = self.PROVIDER_URLS.get(provider)

        # For Ollama, api_key can be anything (not validated by server)
        if provider == LLMProvider.OLLAMA and not api_key:
            api_key = "ollama"

        self.client = OpenAI(api_key=api_key, base_url=self._base_url)

    @property
    def base_url(self) -> str | None:
        """Return the base URL being used."""
        return self._base_url

    def is_available(self) -> bool:
        """Check if the provider is available and responding.

        Returns:
            True if the provider is reachable and responding, False otherwise.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> GenerateResult:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt to send to the model.
            temperature: Sampling temperature (0.0-2.0). Higher values mean more
                random outputs. Defaults to 0.7.
            max_tokens: Maximum number of tokens to generate. If None, uses model default.
            system_prompt: Optional system prompt to set context.
            response_format: Optional response format constraint (e.g.,
                ``{"type": "json_object"}``). When provided, constrains the LLM
                to produce output matching this format.

        Returns:
            GenerateResult with content and optional thinking trace.

        Raises:
            openai.APIError: If the API request fails after retries.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        raw_content = message.content or ""

        # Source 1: API-level thinking (OpenAI o-series reasoning_content)
        api_thinking = getattr(message, "reasoning_content", None)

        # Source 2: Inline <think> tags (Ollama/Qwen3/DeepSeek)
        result = extract_thinking(raw_content)

        # Combine: API-level takes precedence, inline supplements
        thinking = api_thinking or result.thinking

        return GenerateResult(content=result.content, thinking=thinking)

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return (
            f"LLMClient(provider={self.provider.value}, "
            f"model_id={self.model_id!r}, "
            f"base_url={self._base_url!r})"
        )


def create_llm_client(
    provider: LLMProvider | str,
    model_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: The LLM provider. Can be an LLMProvider enum or string
            ("openai", "ollama", "custom").
        model_id: The model identifier.
        api_key: API key for authentication.
        base_url: Custom base URL for the API endpoint.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If provider string is not a valid provider type.

    Examples:
        >>> client = create_llm_client("ollama", "llama3.1:8b")
        >>> client = create_llm_client(LLMProvider.OPENAI, "gpt-4", api_key="sk-...")
    """
    if isinstance(provider, str):
        try:
            provider = LLMProvider(provider.lower())
        except ValueError as e:
            valid = [p.value for p in LLMProvider]
            raise ValueError(f"Invalid provider: {provider!r}. Must be one of {valid}") from e

    return LLMClient(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
    )
