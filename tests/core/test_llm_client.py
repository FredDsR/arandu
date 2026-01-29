"""Tests for LLM client."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from gtranscriber.core.llm_client import (
    LLMClient,
    LLMProvider,
    create_llm_client,
)


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_initialization_openai(self, mocker: MockerFixture) -> None:
        """Test LLMClient initialization with OpenAI provider."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        # Mock the response structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Paris is the capital of France."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        response = client.generate("What is the capital of France?")

        assert response == "Paris is the capital of France."
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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        response = client.generate(
            "What is the capital of France?",
            system_prompt="You are a geography expert.",
        )

        assert response == "Test response"

        # Verify the call arguments include system prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a geography expert."
        assert messages[1]["role"] == "user"

    def test_generate_with_custom_temperature(self, mocker: MockerFixture) -> None:
        """Test text generation with custom temperature."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
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
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
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

    def test_generate_empty_content(self, mocker: MockerFixture) -> None:
        """Test text generation when response content is None."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        response = client.generate("Test prompt")

        assert response == ""

    def test_generate_retry_on_failure(self, mocker: MockerFixture) -> None:
        """Test that generate retries on failure (tenacity decorator)."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success"

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

        response = client.generate("Test prompt")

        # Should succeed after retries
        assert response == "Success"
        assert mock_client.chat.completions.create.call_count == 3

    def test_repr(self, mocker: MockerFixture) -> None:
        """Test string representation of LLMClient."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        client = create_llm_client(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OLLAMA
        assert client.model_id == "llama3.1:8b"

    def test_create_with_string_lowercase(self, mocker: MockerFixture) -> None:
        """Test creating client with lowercase string provider."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        client = create_llm_client(
            provider="ollama",
            model_id="llama3.1:8b",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OLLAMA

    def test_create_with_string_uppercase(self, mocker: MockerFixture) -> None:
        """Test creating client with uppercase string provider."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        client = create_llm_client(
            provider="OPENAI",
            model_id="gpt-4",
            api_key="sk-test",
        )

        assert isinstance(client, LLMClient)
        assert client.provider == LLMProvider.OPENAI

    def test_create_with_invalid_string(self, mocker: MockerFixture) -> None:
        """Test creating client with invalid string provider raises ValueError."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        with pytest.raises(ValueError) as exc_info:
            create_llm_client(
                provider="invalid_provider",
                model_id="some-model",
            )

        assert "Invalid provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_create_with_all_parameters(self, mocker: MockerFixture) -> None:
        """Test creating client with all parameters."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

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
