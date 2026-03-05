"""Tests for LLM client."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from arandu.shared.llm_client import (
    LLMClient,
    LLMProvider,
    create_llm_client,
)
from arandu.utils.text import GenerateResult


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_initialization_openai(self, mocker: MockerFixture) -> None:
        """Test LLMClient initialization with OpenAI provider."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.OPENAI,
            model_id="gpt-4",
            api_key="sk-test-key",
        )

        assert client.provider == LLMProvider.OPENAI
        assert client.model_id == "gpt-4"
        assert client.base_url is None
        mock_openai.assert_called_once_with(api_key="sk-test-key", base_url=None)

    def test_initialization_ollama_default_api_key(self, mocker: MockerFixture) -> None:
        """Test LLMClient initialization with Ollama provider and default API key."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        assert client.provider == LLMProvider.OLLAMA
        assert client.model_id == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434/v1"
        mock_openai.assert_called_once_with(api_key="ollama", base_url="http://localhost:11434/v1")

    def test_initialization_ollama_custom_api_key(self, mocker: MockerFixture) -> None:
        """Test LLMClient initialization with Ollama provider and custom API key."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
            api_key="custom-key",
        )

        assert client.base_url == "http://localhost:11434/v1"
        mock_openai.assert_called_once_with(
            api_key="custom-key", base_url="http://localhost:11434/v1"
        )

    def test_initialization_custom_provider(self, mocker: MockerFixture) -> None:
        """Test LLMClient initialization with custom provider."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.CUSTOM,
            model_id="custom-model",
            api_key="custom-key",
            base_url="http://localhost:8000/v1",
        )

        assert client.provider == LLMProvider.CUSTOM
        assert client.model_id == "custom-model"
        assert client.base_url == "http://localhost:8000/v1"
        mock_openai.assert_called_once_with(
            api_key="custom-key", base_url="http://localhost:8000/v1"
        )

    def test_initialization_custom_base_url_override(self, mocker: MockerFixture) -> None:
        """Test that explicit base_url overrides provider default."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
            base_url="http://custom-ollama:11434/v1",
        )

        assert client.base_url == "http://custom-ollama:11434/v1"
        mock_openai.assert_called_once_with(
            api_key="ollama", base_url="http://custom-ollama:11434/v1"
        )

    def test_is_available_success(self, mocker: MockerFixture) -> None:
        """Test is_available returns True when models.list succeeds."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        assert client.is_available() is True
        mock_client.models.list.assert_called_once()

    def test_is_available_failure(self, mocker: MockerFixture) -> None:
        """Test is_available returns False when models.list fails."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Connection error")
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        assert client.is_available() is False

    def test_generate_basic(self, mocker: MockerFixture) -> None:
        """Test basic text generation."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        # Mock the response structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Paris is the capital of France."
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        result = client.generate("What is the capital of France?")

        assert isinstance(result, GenerateResult)
        assert result.content == "Paris is the capital of France."
        assert result.thinking is None
        mock_client.chat.completions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "llama3.1:8b"
        assert call_args.kwargs["temperature"] == 0.7
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["role"] == "user"
        assert call_args.kwargs["messages"][0]["content"] == "What is the capital of France?"

    def test_generate_with_system_prompt(self, mocker: MockerFixture) -> None:
        """Test text generation with system prompt."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        result = client.generate(
            "What is the capital of France?",
            system_prompt="You are a geography expert.",
        )

        assert result.content == "Test response"

        # Verify the call arguments include system prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a geography expert."
        assert messages[1]["role"] == "user"

    def test_generate_with_custom_temperature(self, mocker: MockerFixture) -> None:
        """Test text generation with custom temperature."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        client.generate("Test prompt", temperature=0.9)

        # Verify temperature is set correctly
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.9

    def test_generate_with_max_tokens(self, mocker: MockerFixture) -> None:
        """Test text generation with max_tokens limit."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        client.generate("Test prompt", max_tokens=100)

        # Verify max_tokens is set correctly
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 100

    def test_generate_with_response_format(self, mocker: MockerFixture) -> None:
        """Test that response_format is passed to API when provided."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        client.generate("Test prompt", response_format={"type": "json_object"})

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_generate_without_response_format(self, mocker: MockerFixture) -> None:
        """Test that response_format is omitted from API when None."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        client.generate("Test prompt")

        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" not in call_args.kwargs

    def test_generate_empty_content(self, mocker: MockerFixture) -> None:
        """Test text generation when response content is None."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        result = client.generate("Test prompt")

        assert result.content == ""
        assert result.thinking is None

    def test_generate_retry_on_failure(self, mocker: MockerFixture) -> None:
        """Test that generate retries on failure (tenacity decorator)."""
        # Skip retry delays for faster tests
        mocker.patch("tenacity.nap.time.sleep", return_value=None)

        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None

        mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            mock_response,
        ]
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        result = client.generate("Test prompt")

        # Should succeed after retries
        assert result.content == "Success"
        assert mock_client.chat.completions.create.call_count == 3

    def test_generate_extracts_thinking(self, mocker: MockerFixture) -> None:
        """Test that generate extracts thinking from <think> tags."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '<think>reasoning</think>{"key": 1}'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        result = client.generate("Test prompt")

        assert result.thinking == "reasoning"
        assert result.content == '{"key": 1}'

    def test_generate_no_thinking(self, mocker: MockerFixture) -> None:
        """Test that generate returns thinking=None for non-thinking models."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Plain text response"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        result = client.generate("Test prompt")

        assert result.thinking is None
        assert result.content == "Plain text response"

    def test_generate_empty_response_with_thinking(self, mocker: MockerFixture) -> None:
        """Test that generate handles None content with thinking extraction."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        result = client.generate("Test prompt")

        assert result.content == ""
        assert result.thinking is None

    def test_generate_ollama_reasoning_field(self, mocker: MockerFixture) -> None:
        """Test that generate extracts thinking from Ollama's 'reasoning' field."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"faithfulness": 0.9}'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = "Step 1: Check context..."
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        result = client.generate("Test prompt")

        assert result.thinking == "Step 1: Check context..."
        assert result.content == '{"faithfulness": 0.9}'

    def test_generate_model_extra_reasoning(self, mocker: MockerFixture) -> None:
        """Test that generate extracts thinking from model_extra dict."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"key": 1}'
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].message.reasoning = None
        mock_response.choices[0].message.model_extra = {"reasoning": "From extras"}
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )

        result = client.generate("Test prompt")

        assert result.thinking == "From extras"
        assert result.content == '{"key": 1}'

    def test_generate_reasoning_content_takes_precedence(self, mocker: MockerFixture) -> None:
        """Test that reasoning_content takes precedence over reasoning field."""
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"key": 1}'
        mock_response.choices[0].message.reasoning_content = "OpenAI reasoning"
        mock_response.choices[0].message.reasoning = "Ollama reasoning"
        mock_response.choices[0].message.model_extra = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OPENAI,
            model_id="o1-mini",
            api_key="sk-test",
        )

        result = client.generate("Test prompt")

        assert result.thinking == "OpenAI reasoning"

    def test_repr(self, mocker: MockerFixture) -> None:
        """Test string representation of LLMClient."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        repr_str = repr(client)

        assert "LLMClient" in repr_str
        assert "ollama" in repr_str
        assert "llama3.1:8b" in repr_str
        assert "http://localhost:11434/v1" in repr_str


class TestCreateLLMClient:
    """Tests for create_llm_client factory function."""

    def test_create_with_enum(self, mocker: MockerFixture) -> None:
        """Test creating client with LLMProvider enum."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        client = create_llm_client(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OLLAMA
        assert client.model_id == "llama3.1:8b"

    def test_create_with_string_lowercase(self, mocker: MockerFixture) -> None:
        """Test creating client with lowercase string provider."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        client = create_llm_client(
            provider="ollama",
            model_id="llama3.1:8b",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OLLAMA

    def test_create_with_string_uppercase(self, mocker: MockerFixture) -> None:
        """Test creating client with uppercase string provider."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        client = create_llm_client(
            provider="OPENAI",
            model_id="gpt-4",
            api_key="sk-test",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OPENAI

    def test_create_with_invalid_string(self, mocker: MockerFixture) -> None:
        """Test creating client with invalid string provider raises ValueError."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        with pytest.raises(ValueError) as exc_info:
            create_llm_client(
                provider="invalid_provider",
                model_id="some-model",
            )

        assert "Invalid provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_create_with_all_parameters(self, mocker: MockerFixture) -> None:
        """Test creating client with all parameters."""
        mocker.patch("arandu.shared.llm_client.OpenAI")

        client = create_llm_client(
            provider="custom",
            model_id="custom-model",
            api_key="custom-key",
            base_url="http://localhost:8000/v1",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.CUSTOM
        assert client.model_id == "custom-model"
        assert client.base_url == "http://localhost:8000/v1"


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_enum_values(self) -> None:
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.CUSTOM.value == "custom"

    def test_enum_from_string(self) -> None:
        """Test creating LLMProvider from string."""
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("ollama") == LLMProvider.OLLAMA
        assert LLMProvider("custom") == LLMProvider.CUSTOM

    def test_enum_invalid_string(self) -> None:
        """Test creating LLMProvider from invalid string raises ValueError."""
        with pytest.raises(ValueError):
            LLMProvider("invalid")
