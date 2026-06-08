"""Integration test for the judge_answers batch runner.

Stubs the LLM and the AnswerJudge.evaluate path so the test verifies
the read/write contract end-to-end against a fixture run that mimics
what ``arandu answer`` produces.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)
from arandu.shared.rag.judge_answers.batch import run_judge_answers_batch
from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings
from arandu.shared.rag.schemas import AnswerRecord

if TYPE_CHECKING:
    from pathlib import Path


def _seed_answers(base: Path, pipeline_id: str = "run_x") -> None:
    """Lay out a minimal answers stage with one null-arm AnswerRecord."""
    out_dir = base / pipeline_id / "answers" / "outputs" / "null" / "cep"
    out_dir.mkdir(parents=True)
    record = AnswerRecord(
        qa_pair_id="src_a:chk_aa:0",
        question="Onde Maria mora?",
        retriever_id="null",
        chunker_id="cep_4k",
        top_k=5,
        passages=[],
        elapsed_ms=1.0,
        is_answerable=True,
        answer_text=None,
        abstained=True,
        rationale="No passages.",
        answerer_model="qwen3:14b",
        answerer_temperature=0.2,
    )
    record.save(out_dir / "src_a__chk_aa__0.json")


def _seed_cep(base: Path, pipeline_id: str = "run_x") -> None:
    """Lay out a minimal CEP outputs tree with the gold answer for the qa_pair_id."""
    cep_dir = base / pipeline_id / "cep" / "outputs"
    cep_dir.mkdir(parents=True)
    cep_record = {
        "source_file_id": "src_a",
        "source_filename": "src_a.mp4",
        "transcription_text": "Maria mora em Itaqui.",
        "chunker_id": "cep_4k",
        "qa_pairs": [
            {
                "question": "Onde Maria mora?",
                "answer": "Em Itaqui.",
                "context": "Maria mora em Itaqui.",
                "question_type": "factual",
                "confidence": 0.9,
                "bloom_level": "remember",
                "chunk_id": "chk_aa",
            }
        ],
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": 1,
        "bloom_distribution": {},
    }
    (cep_dir / "src_a_cep_qa.json").write_text(json.dumps(cep_record))


class TestRunJudgeAnswersBatch:
    def test_missing_answers_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Answers outputs not found"):
            run_judge_answers_batch(
                pipeline_id="never_answered",
                settings=JudgeAnswersSettings(provider="ollama"),
                base_dir=tmp_path,
            )

    def test_end_to_end_with_mocked_judge(self, tmp_path: Path) -> None:
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        settings = JudgeAnswersSettings(provider="ollama")

        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            judge.evaluate.return_value = JudgePipelineResult(
                stage_results={
                    "answer_judge": JudgeStepResult(
                        criterion_scores={
                            "abstention": CriterionScore(
                                score=0.9, threshold=0.7, rationale="genuine refusal"
                            ),
                            "answer_correctness": CriterionScore(
                                score=0.0, threshold=0.6, rationale="N/A — abstained"
                            ),
                            "answer_faithfulness": CriterionScore(
                                score=0.0, threshold=0.6, rationale="N/A — abstained"
                            ),
                            "passage_coverage": CriterionScore(
                                score=0.0, threshold=0.5, rationale="no passages"
                            ),
                        }
                    )
                },
                passed=True,
            )
            mock_judge_cls.return_value = judge

            result = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )

        assert result.judgments_written == 1
        assert result.judgments_failed == 0

        # Output layout: <outputs>/null/cep/<file>.json
        judged_path = (
            tmp_path
            / "run_x"
            / "judge_answers"
            / "outputs"
            / "null"
            / "cep"
            / "src_a__chk_aa__0.json"
        )
        assert judged_path.exists()

        # Verdict round-trips and carries the per-criterion scores.
        rec = AnswerRecord.load(judged_path)
        assert rec.validation is not None
        scores = rec.validation.stage_results["answer_judge"].criterion_scores
        assert "abstention" in scores
        assert "passage_coverage" in scores
        # The answerer's audit fields are preserved on the joined record.
        assert rec.abstained is True
        assert rec.answer_text is None

    def test_missing_gold_lookup_records_as_failed(self, tmp_path: Path) -> None:
        # ANSWERABLE AnswerRecord with no matching CEP pair (orphan
        # qa_pair_id). Without the gold answer the gold criteria can't run;
        # surface it cleanly. (Non-answerable orphans are fine — see below.)
        _seed_answers(tmp_path)  # is_answerable=True, no _seed_cep call
        settings = JudgeAnswersSettings(provider="ollama")

        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge"),
        ):
            result = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )

        assert result.judgments_written == 0
        assert result.judgments_failed >= 1

    def test_nonanswerable_judged_without_gold(self, tmp_path: Path) -> None:
        # A non-answerable item (is_answerable=False) has no CEP gold, but
        # must still be judged (abstention only, via the commitment gate) —
        # not skipped as "no gold lookup".
        out_dir = tmp_path / "run_x" / "answers" / "outputs" / "bm25" / "nonanswerable"
        out_dir.mkdir(parents=True)
        record = AnswerRecord(
            qa_pair_id="src_a:chk_aa:0:nonans",
            question="Onde Joana mora?",
            retriever_id="bm25",
            chunker_id="cep_4k",
            top_k=5,
            passages=[],
            elapsed_ms=1.0,
            is_answerable=False,
            answer_text="Em Itaqui.",
            abstained=False,
            rationale="Committed despite no support.",
            answerer_model="qwen3:14b",
            answerer_temperature=0.2,
        )
        record.save(out_dir / "src_a__chk_aa__0__nonans.json")
        # Deliberately no _seed_cep: non-answerable items need no gold.
        settings = JudgeAnswersSettings(provider="ollama")

        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            judge.evaluate.return_value = JudgePipelineResult(
                stage_results={
                    "abstention": JudgeStepResult(
                        criterion_scores={
                            "abstention": CriterionScore(
                                score=0.1, threshold=0.7, rationale="substantive claim"
                            ),
                        }
                    )
                },
                passed=False,
                rejected_at="commitment_gate",
            )
            mock_judge_cls.return_value = judge

            result = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )

        assert result.judgments_written == 1
        assert result.judgments_failed == 0
        # judge.evaluate received is_answerable=False (gate input).
        assert judge.evaluate.call_args.kwargs["is_answerable"] is False

    def test_resume_skips_completed_records(self, tmp_path: Path) -> None:
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        settings = JudgeAnswersSettings(provider="ollama")

        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            judge.evaluate.return_value = JudgePipelineResult(
                stage_results={
                    "answer_judge": JudgeStepResult(
                        criterion_scores={
                            "abstention": CriterionScore(score=0.9, threshold=0.7, rationale="r"),
                        }
                    )
                },
                passed=True,
            )
            mock_judge_cls.return_value = judge

            first = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )
            assert first.judgments_written == 1

            second = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )
            assert second.judgments_written == 0
            assert second.judgments_resumed == 1

    def test_audit_count_cumulative_across_resume(self, tmp_path: Path) -> None:
        # Bug fix from PR #110 second-pass review: the audit log must
        # reflect ALL disagreements in the run's judged outputs, not
        # just those detected in this invocation. On resume, the loop
        # skips already-completed records, but their disagreements
        # still belong in the audit. The runner now walks all judged
        # outputs at audit time so the count + on-disk JSONL stay
        # cumulative.
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        settings = JudgeAnswersSettings(provider="ollama")

        # First run: judge returns abstention=0.1 (committal text per
        # judge) while answerer.abstained=True → disagreement.
        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            judge.evaluate.return_value = JudgePipelineResult(
                stage_results={
                    "answer_judge": JudgeStepResult(
                        criterion_scores={
                            "abstention": CriterionScore(
                                score=0.1, threshold=0.7, rationale="committal"
                            ),
                        }
                    )
                },
                passed=True,
            )
            mock_judge_cls.return_value = judge

            first = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )
            assert first.abstention_disagreements == 1
            audit_path = tmp_path / "run_x" / "judge_answers" / "outputs" / "abstention_audit.jsonl"
            assert audit_path.exists()
            lines_after_first = audit_path.read_text().strip().split("\n")
            assert len(lines_after_first) == 1

        # Second run: same fixture, no new judging happens (resume),
        # but the audit must still reflect the disagreement that's on
        # disk from the first run.
        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            mock_judge_cls.return_value = judge
            second = run_judge_answers_batch(
                pipeline_id="run_x", settings=settings, base_dir=tmp_path
            )

        assert second.judgments_written == 0
        assert second.judgments_resumed == 1
        # The audit count + on-disk file are still cumulative.
        assert second.abstention_disagreements == 1
        assert audit_path.exists()
        assert len(audit_path.read_text().strip().split("\n")) == 1

    def test_rejudge_clears_checkpoint(self, tmp_path: Path) -> None:
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        settings = JudgeAnswersSettings(provider="ollama")

        with (
            patch("arandu.shared.rag.judge_answers.batch.build_llm_client_from_settings"),
            patch("arandu.shared.rag.judge_answers.batch.AnswerJudge") as mock_judge_cls,
        ):
            judge = MagicMock()
            judge.evaluate.return_value = JudgePipelineResult(
                stage_results={
                    "answer_judge": JudgeStepResult(
                        criterion_scores={
                            "abstention": CriterionScore(score=0.9, threshold=0.7, rationale="r"),
                        }
                    )
                },
                passed=True,
            )
            mock_judge_cls.return_value = judge

            run_judge_answers_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)
            result = run_judge_answers_batch(
                pipeline_id="run_x",
                settings=settings,
                base_dir=tmp_path,
                rejudge=True,
            )

        assert result.judgments_written == 1
        assert result.judgments_resumed == 0


class TestBuildLlmClientFailures:
    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        settings = JudgeAnswersSettings.model_construct(
            provider="not_a_real_provider",
            model_id="x",
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            temperature=0.3,
            max_tokens=2048,
            language="pt",
            abstention_disagreement_audit=True,
        )
        with pytest.raises(ValueError, match="Invalid provider"):
            run_judge_answers_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)

    def test_cloud_provider_without_api_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = JudgeAnswersSettings(provider="openai", api_key_env="OPENAI_API_KEY")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            run_judge_answers_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)

    def test_custom_without_base_url_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `custom` provider needs both an API key (cloud-style check)
        # AND a base_url. With api_key present, the base_url guard
        # fires next.
        _seed_answers(tmp_path)
        _seed_cep(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        settings = JudgeAnswersSettings(provider="custom", base_url=None)
        with pytest.raises(ValueError, match="provider='custom' requires a base URL"):
            run_judge_answers_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)
