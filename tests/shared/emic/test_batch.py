"""Tests for the emic-validity pre-pass batch (spec §5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.emic.batch import run_emic_prepass_batch
from arandu.shared.emic.schemas import EmicSourceScores
from arandu.shared.emic.settings import EmicPrepassSettings
from arandu.shared.judge.criterion import OrdinalCriterionResponse
from arandu.shared.judge.schemas import JudgePipelineResult

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _pair(question: str, *, approved: bool, bloom: str = "analyze") -> QAPairCEP:
    validation = JudgePipelineResult(stage_results={}, passed=approved)
    return QAPairCEP(
        question=question,
        answer="resposta situada do interlocutor",
        context="o interlocutor disse algo concreto sobre o rio",
        question_type="conceptual",
        confidence=0.9,
        bloom_level=bloom,
        validation=validation,
    )


def _write_cep_record(cep_outputs: Path, file_id: str, pairs: list[QAPairCEP]) -> None:
    record = QARecordCEP(
        source_gdrive_id=file_id,
        source_filename=f"{file_id}.mp4",
        transcription_text="t",
        qa_pairs=pairs,
        model_id="test-model",
        provider="ollama",
        total_pairs=len(pairs),
    )
    cep_outputs.mkdir(parents=True, exist_ok=True)
    record.save(cep_outputs / f"{file_id}_cep_qa.json")


@pytest.fixture
def mock_emic_client(mocker: MockerFixture) -> Any:
    """Patch the batch's LLM client builder; every call scores ordinal 2."""
    client = mocker.MagicMock()
    client.generate_structured.return_value = OrdinalCriterionResponse(
        score=2, rationale="reenquadramento institucional"
    )
    mocker.patch("arandu.shared.emic.batch._build_llm_client", return_value=client)
    return client


@pytest.fixture
def settings() -> EmicPrepassSettings:
    return EmicPrepassSettings(provider="ollama", model_id="test-model")


class TestEmicPrepassBatch:
    def test_scores_only_approved_pairs(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicPrepassSettings
    ) -> None:
        cep_outputs = tmp_path / "run1" / "cep" / "outputs"
        _write_cep_record(
            cep_outputs,
            "src1",
            [
                _pair("Q approved", approved=True),
                _pair("Q rejected", approved=False),
                _pair("Q approved 2", approved=True, bloom="evaluate"),
            ],
        )

        result = run_emic_prepass_batch("run1", settings=settings, base_dir=tmp_path)

        assert result.approved_pairs == 2  # the rejected pair is skipped
        assert result.scored_pairs == 2
        assert result.failed_pairs == 0
        assert result.sources == 1

        out = EmicSourceScores.load(
            tmp_path / "run1" / "emic_prepass" / "outputs" / "src1_cep_qa.json"
        )
        assert [s.pair_index for s in out.scores] == [0, 2]  # original indices preserved
        assert all(s.emic_score == 2 for s in out.scores)
        assert {s.bloom_level for s in out.scores} == {"analyze", "evaluate"}

    def test_missing_cep_stage_raises(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicPrepassSettings
    ) -> None:
        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            run_emic_prepass_batch("absent", settings=settings, base_dir=tmp_path)

    def test_resume_skips_completed_sources(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicPrepassSettings
    ) -> None:
        cep_outputs = tmp_path / "run2" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        run_emic_prepass_batch("run2", settings=settings, base_dir=tmp_path)
        calls_after_first = mock_emic_client.generate_structured.call_count
        assert calls_after_first == 1

        # Second run resumes: the source is already checkpointed, no new calls.
        second = run_emic_prepass_batch("run2", settings=settings, base_dir=tmp_path)
        assert mock_emic_client.generate_structured.call_count == calls_after_first
        assert second.scored_pairs == 0  # nothing re-scored on resume

    def test_rerun_rescores(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicPrepassSettings
    ) -> None:
        cep_outputs = tmp_path / "run3" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        run_emic_prepass_batch("run3", settings=settings, base_dir=tmp_path)
        run_emic_prepass_batch("run3", settings=settings, base_dir=tmp_path, rerun=True)
        assert mock_emic_client.generate_structured.call_count == 2

    def test_llm_error_records_failed_pair(
        self, tmp_path: Path, mocker: MockerFixture, settings: EmicPrepassSettings
    ) -> None:
        client = mocker.MagicMock()
        client.generate_structured.side_effect = RuntimeError("llm down")
        mocker.patch("arandu.shared.emic.batch._build_llm_client", return_value=client)

        cep_outputs = tmp_path / "run4" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        result = run_emic_prepass_batch("run4", settings=settings, base_dir=tmp_path)
        assert result.failed_pairs == 1
        assert result.scored_pairs == 0
        out = EmicSourceScores.load(
            tmp_path / "run4" / "emic_prepass" / "outputs" / "src1_cep_qa.json"
        )
        assert out.scores[0].emic_score is None
        assert out.scores[0].error is not None
