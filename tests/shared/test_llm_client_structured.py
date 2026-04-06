"""Tests for LLMClient.generate_structured()."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from arandu.shared.llm_client import (
    LLMClient,
    LLMProvider,
    StructuredOutputError,
)


class SampleResponse(BaseModel):
    """Test model for structured output."""

    score: float
    rationale: str


def _make_llm_response(content: str) -> Mock:
    """Create a mock OpenAI chat completion response.

    Args:
        content: The text content for the response message.

    Returns:
        Mock object mimicking an OpenAI ChatCompletion response.
    """
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = content
    response.choices[0].message.reasoning_content = None
    response.choices[0].message.reasoning = None
    response.choices[0].message.model_extra = None
    return response


class TestGenerateStructured:
    """Tests for LLMClient.generate_structured()."""

    def _make_client(self, mocker: MockerFixture) -> tuple[LLMClient, Mock]:
        """Create an LLMClient with a mocked OpenAI backend.

        Args:
            mocker: pytest-mock fixture.

        Returns:
            Tuple of (LLMClient, mock_openai_client).
        """
        mock_openai = mocker.patch("arandu.shared.llm_client.OpenAI")
        mock_client = Mock()
        mock_openai.return_value = mock_client
        client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="qwen3:14b",
        )
        return client, mock_client

    def test_valid_json_parsed_as_model(self, mocker: MockerFixture) -> None:
        """Test that valid JSON response is parsed into the Pydantic model."""
        client, mock_backend = self._make_client(mocker)
        payload = json.dumps({"score": 0.85, "rationale": "Well supported"})
        mock_backend.chat.completions.create.return_value = _make_llm_response(payload)

        result = client.generate_structured("Rate this.", SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.score == 0.85
        assert result.rationale == "Well supported"

    def test_invalid_json_retries_then_succeeds(self, mocker: MockerFixture) -> None:
        """Test that invalid JSON triggers a retry and second attempt succeeds."""
        client, mock_backend = self._make_client(mocker)
        good_payload = json.dumps({"score": 0.5, "rationale": "OK"})
        mock_backend.chat.completions.create.side_effect = [
            _make_llm_response("not valid json {{{"),
            _make_llm_response(good_payload),
        ]

        result = client.generate_structured("Rate this.", SampleResponse, max_retries=1)

        assert isinstance(result, SampleResponse)
        assert result.score == 0.5
        assert mock_backend.chat.completions.create.call_count == 2

    def test_all_retries_exhausted_raises_error(self, mocker: MockerFixture) -> None:
        """Test that StructuredOutputError is raised after all retries fail."""
        client, mock_backend = self._make_client(mocker)
        mock_backend.chat.completions.create.return_value = _make_llm_response("not json at all")

        with pytest.raises(StructuredOutputError):
            client.generate_structured("Rate this.", SampleResponse, max_retries=2)

        # 1 initial + 2 retries = 3 calls
        assert mock_backend.chat.completions.create.call_count == 3

    def test_json_in_markdown_codeblock_extracted(self, mocker: MockerFixture) -> None:
        """Test that JSON wrapped in markdown codeblock fences is extracted."""
        client, mock_backend = self._make_client(mocker)
        wrapped = '```json\n{"score": 0.9, "rationale": "Excellent"}\n```'
        mock_backend.chat.completions.create.return_value = _make_llm_response(wrapped)

        result = client.generate_structured("Rate this.", SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.score == 0.9
        assert result.rationale == "Excellent"

    def test_pydantic_validation_error_triggers_retry(self, mocker: MockerFixture) -> None:
        """Test that a Pydantic validation error triggers a retry."""
        client, mock_backend = self._make_client(mocker)
        # Valid JSON but missing required field 'rationale'
        bad_payload = json.dumps({"score": 0.5})
        good_payload = json.dumps({"score": 0.7, "rationale": "Fixed"})
        mock_backend.chat.completions.create.side_effect = [
            _make_llm_response(bad_payload),
            _make_llm_response(good_payload),
        ]

        result = client.generate_structured("Rate this.", SampleResponse, max_retries=1)

        assert isinstance(result, SampleResponse)
        assert result.rationale == "Fixed"
        assert mock_backend.chat.completions.create.call_count == 2

    def test_system_prompt_forwarded(self, mocker: MockerFixture) -> None:
        """Test that system_prompt is forwarded to the underlying generate call."""
        client, mock_backend = self._make_client(mocker)
        payload = json.dumps({"score": 1.0, "rationale": "Perfect"})
        mock_backend.chat.completions.create.return_value = _make_llm_response(payload)

        client.generate_structured(
            "Rate this.",
            SampleResponse,
            system_prompt="You are a judge.",
        )

        call_args = mock_backend.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a judge."

    def test_temperature_forwarded(self, mocker: MockerFixture) -> None:
        """Test that temperature is forwarded to the underlying generate call."""
        client, mock_backend = self._make_client(mocker)
        payload = json.dumps({"score": 1.0, "rationale": "Perfect"})
        mock_backend.chat.completions.create.return_value = _make_llm_response(payload)

        client.generate_structured(
            "Rate this.",
            SampleResponse,
            temperature=0.1,
        )

        call_args = mock_backend.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.1

    def test_response_format_json_object_set(self, mocker: MockerFixture) -> None:
        """Test that response_format is set to json_object in the API call."""
        client, mock_backend = self._make_client(mocker)
        payload = json.dumps({"score": 0.5, "rationale": "OK"})
        mock_backend.chat.completions.create.return_value = _make_llm_response(payload)

        client.generate_structured("Rate this.", SampleResponse)

        call_args = mock_backend.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_zero_retries_fails_immediately(self, mocker: MockerFixture) -> None:
        """Test that max_retries=0 means only one attempt, no retries."""
        client, mock_backend = self._make_client(mocker)
        mock_backend.chat.completions.create.return_value = _make_llm_response("bad json")

        with pytest.raises(StructuredOutputError):
            client.generate_structured("Rate this.", SampleResponse, max_retries=0)

        assert mock_backend.chat.completions.create.call_count == 1
