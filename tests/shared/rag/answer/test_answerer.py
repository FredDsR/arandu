"""Tests for ``AnswererClient`` — the retry-with-temperature-bump contract."""

from __future__ import annotations

from unittest.mock import MagicMock

from arandu.shared.llm_client import StructuredOutputError
from arandu.shared.rag.answer.answerer import AnswererClient
from arandu.shared.rag.answer.schemas import AnswererOutput
from arandu.shared.rag.answer.settings import AnswererSettings


def _settings(temperature: float = 0.2) -> AnswererSettings:
    return AnswererSettings(temperature=temperature, _env_file=None)


class TestAnswererClient:
    def test_first_attempt_success(self) -> None:
        # Happy path: LLM returns valid structured output on the first try.
        client = MagicMock()
        client.generate_structured.return_value = AnswererOutput(
            abstained=False, answer="Maria mora em Itaqui.", rationale="Per P1."
        )
        ac = AnswererClient(llm_client=client, settings=_settings())

        output, meta = ac.answer(question="Onde Maria mora?", passage_texts=["P1: Itaqui."])

        assert output.abstained is False
        assert output.answer == "Maria mora em Itaqui."
        assert meta["attempts"] == 1
        assert meta["final_temperature"] == 0.2
        assert "fallback_reason" not in meta
        client.generate_structured.assert_called_once()

    def test_retries_with_bumped_temperature(self) -> None:
        # First two attempts raise StructuredOutputError; third succeeds.
        # Verify each call's temperature increases by 0.1 per the spec.
        client = MagicMock()
        client.generate_structured.side_effect = [
            StructuredOutputError("malformed"),
            StructuredOutputError("malformed"),
            AnswererOutput(abstained=True, answer=None, rationale="insufficient evidence"),
        ]
        ac = AnswererClient(llm_client=client, settings=_settings(temperature=0.2))

        output, meta = ac.answer(question="q", passage_texts=[])

        assert output.abstained is True
        assert meta["attempts"] == 3
        # Third attempt = 0.2 + 0.1 + 0.1 = 0.4.
        assert meta["final_temperature"] == pytest.approx(0.4)
        # Each call seen the cumulative bump.
        temps = [c.kwargs["temperature"] for c in client.generate_structured.call_args_list]
        assert temps == pytest.approx([0.2, 0.3, 0.4])

    def test_fallback_after_three_failed_attempts(self) -> None:
        # All three attempts fail. Spec §5.6 says return the abstained
        # fallback with `fallback_reason` recorded in meta.
        client = MagicMock()
        client.generate_structured.side_effect = StructuredOutputError("malformed")
        ac = AnswererClient(llm_client=client, settings=_settings())

        output, meta = ac.answer(question="q", passage_texts=[])

        assert output.abstained is True
        assert output.answer is None
        assert "3 retries" in output.rationale
        assert meta["attempts"] == 3
        assert meta["fallback_reason"] == output.rationale
        assert client.generate_structured.call_count == 3


# `pytest.approx` is imported lazily so the rest of the module can use
# plain `assert ==`; placing this near the bottom keeps test reads focused.
import pytest  # noqa: E402
