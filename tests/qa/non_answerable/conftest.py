"""Shared fixtures for non-answerable benchmark tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _passed_validation() -> JudgePipelineResult:
    return JudgePipelineResult(
        stage_results={
            "qa_judge": JudgeStepResult(
                criterion_scores={
                    "faithfulness": CriterionScore(score=0.9, threshold=0.6, rationale="ok")
                }
            )
        },
        passed=True,
    )


def make_pair(
    *,
    question: str,
    bloom_level: str = "remember",
    question_type: str = "factual",
    chunk_id: str | None = "chk_00",
    validated: bool = True,
) -> QAPairCEP:
    """Build a CEP pair; ``validated=False`` leaves validation unset (ineligible)."""
    return QAPairCEP(
        question=question,
        answer="resposta",
        context="contexto de origem",
        question_type=question_type,
        confidence=0.9,
        bloom_level=bloom_level,
        chunk_id=chunk_id,
        validation=_passed_validation() if validated else None,
    )


def write_cep_record(
    cep_dir: Path,
    *,
    source_file_id: str,
    pairs: list[QAPairCEP],
    filename: str | None = None,
) -> None:
    """Write one ``QARecordCEP`` JSON file under ``cep_dir``."""
    cep_dir.mkdir(parents=True, exist_ok=True)
    record = QARecordCEP(
        source_gdrive_id=source_file_id,
        source_filename=f"{source_file_id}.wav",
        transcription_text="contexto de origem completo",
        qa_pairs=pairs,
        model_id="qwen3:14b",
        provider="ollama",
        total_pairs=len(pairs),
    )
    record.save(cep_dir / (filename or f"{source_file_id}.json"))


@pytest.fixture
def cep_dir(tmp_path: Path) -> Path:
    """A CEP outputs dir with validated pairs across two Bloom levels."""
    out = tmp_path / "results" / "run-x" / "cep" / "outputs"
    write_cep_record(
        out,
        source_file_id="src1",
        pairs=[
            make_pair(question="Em que ano Maria viu a enchente?", bloom_level="remember"),
            make_pair(
                question="Como Maria explica a cheia?",
                bloom_level="analyze",
                question_type="conceptual",
            ),
        ],
    )
    return out
