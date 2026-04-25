"""Tests for TranscriptionJudge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.judge import BaseJudge
from arandu.shared.judge.criterion import CriterionResponse
from arandu.shared.llm_client import LLMProvider
from arandu.transcription.judge import TranscriptionJudge, build_validator_client

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Long enough Portuguese text to exceed 30 wpm at 60s duration (>30 words)
_GOOD_PT_TEXT = (
    "O pescador mencionou a enchente que afetou a regiao no ultimo ano. "
    "Ele relatou que as aguas subiram rapidamente e inundaram as casas "
    "proximas ao rio. As familias tiveram que evacuar durante a noite "
    "e buscar abrigo em areas mais elevadas da comunidade."
)

_GOOD_EN_TEXT = (
    "The fisherman mentioned the flood that affected the region last year. "
    "He reported that the waters rose rapidly and flooded the houses near "
    "the river. The families had to evacuate during the night and seek "
    "shelter in higher areas of the community."
)


class TestTranscriptionJudge:
    def test_is_base_judge_subclass(self) -> None:
        judge = TranscriptionJudge()
        assert isinstance(judge, BaseJudge)

    def test_pipeline_has_heuristic_stage(self) -> None:
        judge = TranscriptionJudge()
        stages = judge._pipeline._stages
        assert len(stages) == 1
        assert stages[0].name == "heuristic_filter"
        assert stages[0].mode == "filter"

    def test_good_transcription_passes(self) -> None:
        judge = TranscriptionJudge()
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True

    def test_cjk_text_rejected(self) -> None:
        judge = TranscriptionJudge()
        result = judge.evaluate_transcription(
            text="\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\u3002" * 20,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "heuristic_filter"

    def test_evaluate_transcription_uses_init_language(self) -> None:
        judge = TranscriptionJudge(language="en")
        result = judge.evaluate_transcription(
            text=_GOOD_EN_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True

    def test_no_llm_client_needed(self) -> None:
        """TranscriptionJudge works without any LLM client."""
        judge = TranscriptionJudge()
        # Should not raise -- no LLM calls involved
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client that returns a passing score by default."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "test-model"
    client.generate_structured.return_value = CriterionResponse(score=1.0, rationale="clean")
    return client


class TestTranscriptionJudgeLLMStage:
    """LLM-stage pipeline assembly and evaluation behavior."""

    def test_pipeline_adds_llm_stage_when_client_provided(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        stages = judge._pipeline._stages
        assert [s.name for s in stages] == ["heuristic_filter", "llm_filter"]
        assert stages[1].mode == "filter"

    def test_llm_stage_passes_on_clean_text(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True
        assert "llm_filter" in result.stage_results
        llm_scores = result.stage_results["llm_filter"].criterion_scores
        assert set(llm_scores.keys()) == {"language_drift", "hallucination_loop"}
        # Two criteria => two LLM calls
        assert mock_llm_client.generate_structured.call_count == 2

    def test_llm_stage_rejects_on_drift(self, mock_llm_client: Any) -> None:
        def _structured_response(**kwargs: Any) -> CriterionResponse:
            prompt = kwargs.get("prompt", "")
            if "Language Drift" in prompt or "Linguística" in prompt:
                return CriterionResponse(score=0.0, rationale="fully English")
            return CriterionResponse(score=1.0, rationale="no hallucination")

        mock_llm_client.generate_structured.side_effect = _structured_response

        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "llm_filter"
        assert result.stage_results["llm_filter"].criterion_scores["language_drift"].passed is False

    def test_llm_stage_skipped_when_heuristic_rejects(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text="これはテストです。" * 20,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "heuristic_filter"
        assert "llm_filter" not in result.stage_results
        # LLM never called because heuristic filter short-circuited
        mock_llm_client.generate_structured.assert_not_called()

    def test_llm_criteria_receive_text_and_language(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(language="en", validator_client=mock_llm_client)
        judge.evaluate_transcription(text=_GOOD_EN_TEXT, duration_ms=60000)

        prompts_sent = [
            call.kwargs["prompt"] for call in mock_llm_client.generate_structured.call_args_list
        ]
        assert len(prompts_sent) == 2
        for prompt in prompts_sent:
            assert _GOOD_EN_TEXT in prompt
        # language_drift prompt must include the expected_language substitution
        drift_prompt = next(p for p in prompts_sent if "Language Drift" in p)
        assert "**Expected language:** en" in drift_prompt


class TestBuildValidatorClient:
    """Tests for the build_validator_client factory helper."""

    def test_infers_ollama_when_no_base_url(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "arandu.transcription.judge.get_llm_config",
            return_value=mocker.MagicMock(base_url=None),
        )
        mocker.patch("arandu.shared.llm_client.OpenAI")
        client = build_validator_client("qwen3:14b")
        assert client.provider == LLMProvider.OLLAMA
        assert client.model_id == "qwen3:14b"

    def test_infers_custom_when_base_url_from_env(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "arandu.transcription.judge.get_llm_config",
            return_value=mocker.MagicMock(base_url="https://example.test/v1"),
        )
        mocker.patch("arandu.shared.llm_client.OpenAI")
        client = build_validator_client("gemini-2.5-flash")
        assert client.provider == LLMProvider.CUSTOM
        assert client.base_url == "https://example.test/v1"

    def test_explicit_provider_overrides_inference(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "arandu.transcription.judge.get_llm_config",
            return_value=mocker.MagicMock(base_url="https://example.test/v1"),
        )
        mocker.patch("arandu.shared.llm_client.OpenAI")
        client = build_validator_client(
            "gpt-4o-mini", provider="openai", base_url="https://api.openai.com/v1"
        )
        assert client.provider == LLMProvider.OPENAI
        assert client.base_url == "https://api.openai.com/v1"


class TestJudgeTranscriptionCLI:
    """Tests for the judge-transcription CLI command."""

    def test_missing_validator_model_exits(self, mocker: MockerFixture, tmp_path: Any) -> None:
        """CLI exits without building a client when no model is configured."""
        from typer.testing import CliRunner

        from arandu.cli.app import app

        mocker.patch(
            "arandu.qa.config.get_judge_config",
            return_value=mocker.MagicMock(validator_model=None),
        )
        # If the guard works, build_validator_client is never reached
        build_spy = mocker.patch("arandu.transcription.judge.build_validator_client")

        input_dir = tmp_path / "results"
        input_dir.mkdir()
        (input_dir / "dummy_transcription.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(app, ["judge-transcription", str(input_dir)])
        assert result.exit_code == 1
        build_spy.assert_not_called()

    def test_env_var_fulfills_requirement(self, mocker: MockerFixture, tmp_path: Any) -> None:
        """ARANDU_JUDGE_VALIDATOR_MODEL satisfies the model requirement."""
        from typer.testing import CliRunner

        from arandu.cli.app import app

        mocker.patch(
            "arandu.qa.config.get_judge_config",
            return_value=mocker.MagicMock(
                validator_model="qwen3:14b",
                validator_provider=None,
                validator_base_url=None,
            ),
        )
        fake_client = mocker.MagicMock()
        fake_client.is_available.return_value = True
        fake_client.provider.value = "ollama"
        fake_client.base_url = "http://localhost:11434/v1"
        build_spy = mocker.patch(
            "arandu.transcription.judge.build_validator_client", return_value=fake_client
        )

        input_dir = tmp_path / "results"
        # Empty dir → CLI still proceeds past the validator-model check,
        # builds a client, then exits 1 on "no files found". The important
        # signal is that build_validator_client WAS called.
        input_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["judge-transcription", str(input_dir)])
        assert result.exit_code == 1
        build_spy.assert_called_once()
        call_kwargs = build_spy.call_args.kwargs
        assert call_kwargs["model_id"] == "qwen3:14b"

    def test_writes_verdict_back_into_record(self, mocker: MockerFixture, tmp_path: Any) -> None:
        """judge-transcription mutates the on-disk record and does not write an aggregate file."""
        import json as _json

        from typer.testing import CliRunner

        from arandu.cli.app import app
        from arandu.shared.judge.schemas import (
            CriterionScore,
            JudgePipelineResult,
            JudgeStepResult,
        )

        mocker.patch(
            "arandu.qa.config.get_judge_config",
            return_value=mocker.MagicMock(
                validator_model="qwen3:14b",
                validator_provider=None,
                validator_base_url=None,
            ),
        )
        fake_client = mocker.MagicMock()
        fake_client.is_available.return_value = True
        fake_client.provider.value = "ollama"
        fake_client.base_url = "http://localhost:11434/v1"
        mocker.patch("arandu.transcription.judge.build_validator_client", return_value=fake_client)

        pipeline_result = JudgePipelineResult(
            stage_results={
                "heuristic_filter": JudgeStepResult(
                    criterion_scores={
                        "script_match": CriterionScore(score=1.0, threshold=0.6, rationale="ok"),
                        "repetition": CriterionScore(score=0.9, threshold=0.5, rationale="ok"),
                        "segment_quality": CriterionScore(score=1.0, threshold=0.4, rationale="ok"),
                        "content_density": CriterionScore(score=0.8, threshold=0.4, rationale="ok"),
                    }
                )
            },
            passed=True,
        )
        evaluate_mock = mocker.patch(
            "arandu.transcription.judge.TranscriptionJudge.evaluate_transcription",
            return_value=pipeline_result,
        )

        input_dir = tmp_path / "outputs"
        input_dir.mkdir()
        sample_file = input_dir / "abc_transcription.json"
        sample_file.write_text(
            _json.dumps(
                {
                    "gdrive_id": "abc",
                    "name": "sample.mp4",
                    "mimeType": "video/mp4",
                    "parents": ["p1"],
                    "webContentLink": "https://x",
                    "transcription_text": _GOOD_PT_TEXT,
                    "detected_language": "pt",
                    "language_probability": 0.99,
                    "model_id": "whisper",
                    "compute_device": "cpu",
                    "processing_duration_sec": 1.0,
                    "transcription_status": "completed",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(app, ["judge-transcription", str(input_dir)])
        assert result.exit_code == 0, result.stdout
        evaluate_mock.assert_called_once()

        on_disk = _json.loads(sample_file.read_text())
        assert on_disk["is_valid"] is True
        assert on_disk["validation"]["passed"] is True
        assert set(
            on_disk["validation"]["stage_results"]["heuristic_filter"]["criterion_scores"].keys()
        ) == {"script_match", "repetition", "segment_quality", "content_density"}

        # No aggregate side-file produced.
        aggregate_candidates = list(input_dir.glob("judgements*.json"))
        assert aggregate_candidates == []
