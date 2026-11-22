"""Shared fixtures for analysis-stage tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)
from arandu.shared.rag.schemas import AnswerRecord

if TYPE_CHECKING:
    from pathlib import Path


def make_answer(
    *,
    qa_pair_id: str = "src:chk_00:0",
    retriever_id: str = "bm25",
    is_answerable: bool = True,
    abstained: bool = False,
    abstention_score: float | None = 0.1,
    correctness_score: float | None = 0.9,
    faithfulness_score: float | None = 0.8,
    passage_coverage_score: float | None = 0.7,
    source_recovery_score: float | None = None,
) -> AnswerRecord:
    """Build a fully judged :class:`AnswerRecord` for analysis tests."""
    criterion_scores: dict[str, CriterionScore] = {
        "abstention": CriterionScore(score=abstention_score, threshold=0.7, rationale="r"),
        "answer_correctness": CriterionScore(score=correctness_score, threshold=0.6, rationale="r"),
        "answer_faithfulness": CriterionScore(
            score=faithfulness_score, threshold=0.6, rationale="r"
        ),
        "passage_coverage": CriterionScore(
            score=passage_coverage_score, threshold=0.5, rationale="r"
        ),
        "source_recovery": CriterionScore(
            score=source_recovery_score, threshold=0.5, rationale="r"
        ),
    }
    validation = JudgePipelineResult(
        stage_results={"answer_judge": JudgeStepResult(criterion_scores=criterion_scores)},
        passed=True,
    )
    return AnswerRecord(
        qa_pair_id=qa_pair_id,
        question="q",
        retriever_id=retriever_id,
        chunker_id="cep_4k",
        top_k=5,
        passages=[],
        elapsed_ms=1.0,
        is_answerable=is_answerable,
        answer_text=None if abstained else "answer",
        abstained=abstained,
        rationale="r",
        answerer_model="qwen3:14b",
        answerer_temperature=0.2,
        validation=validation,
    )


@pytest.fixture
def cep_outputs_dir(tmp_path: Path) -> Path:
    """A CEP outputs dir with one record carrying mixed Bloom + types."""
    out = tmp_path / "cep" / "outputs"
    out.mkdir(parents=True)
    pairs = [
        QAPairCEP(
            question="q1",
            answer="a1",
            context="c",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            chunk_id="chk_00",
        ),
        QAPairCEP(
            question="q2",
            answer="a2",
            context="c",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            chunk_id="chk_00",
        ),
    ]
    record = QARecordCEP(
        source_gdrive_id="src",
        source_filename="src.wav",
        transcription_text="…",
        qa_pairs=pairs,
        model_id="qwen3:14b",
        provider="ollama",
        total_pairs=len(pairs),
    )
    record.save(out / "src.json")
    return out
