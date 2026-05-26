"""Tests for the abstention-disagreement audit (spec §6.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)
from arandu.shared.rag.judge_answers.audit import (
    AbstentionDisagreement,
    detect_disagreement,
    write_audit_log,
)
from arandu.shared.rag.schemas import AnswerRecord

if TYPE_CHECKING:
    from pathlib import Path


def _judged_answer(
    abstained: bool,
    judge_score: float,
    threshold: float = 0.7,
    qa_pair_id: str = "src_a:chk_00:0",
) -> AnswerRecord:
    answer_text = None if abstained else "Maria mora em Itaqui."
    return AnswerRecord(
        qa_pair_id=qa_pair_id,
        question="q",
        retriever_id="bm25",
        chunker_id="cep_4k",
        top_k=5,
        passages=[],
        elapsed_ms=1.0,
        is_answerable=True,
        answer_text=answer_text,
        abstained=abstained,
        rationale="r",
        answerer_model="qwen3:14b",
        answerer_temperature=0.2,
        validation=JudgePipelineResult(
            stage_results={
                "answer_judge": JudgeStepResult(
                    criterion_scores={
                        "abstention": CriterionScore(
                            score=judge_score, threshold=threshold, rationale="r"
                        )
                    }
                )
            },
            passed=True,
        ),
    )


class TestDetectDisagreement:
    def test_agreement_returns_none(self) -> None:
        # answerer abstained AND judge says abstain (score >= 0.7) → agree.
        answer = _judged_answer(abstained=True, judge_score=0.9)
        assert detect_disagreement(answer) is None

    def test_agreement_on_non_abstention(self) -> None:
        # answerer committed AND judge says committed (score < 0.7) → agree.
        answer = _judged_answer(abstained=False, judge_score=0.2)
        assert detect_disagreement(answer) is None

    def test_answerer_abstains_judge_disagrees(self) -> None:
        # answerer flagged abstained=True, but the judge thinks the text
        # contains a substantive claim (score < threshold).
        answer = _judged_answer(abstained=True, judge_score=0.1)
        out = detect_disagreement(answer)
        assert out is not None
        assert out.disagreement_type == "answerer_abstains_judge_disagrees"
        assert out.answerer_abstained is True
        assert out.judge_score == 0.1

    def test_answerer_commits_judge_says_abstain(self) -> None:
        # Inverse: answerer committed to an answer, but the judge reads
        # the text as a hedge / refusal (score >= threshold).
        answer = _judged_answer(abstained=False, judge_score=0.9)
        out = detect_disagreement(answer)
        assert out is not None
        assert out.disagreement_type == "answerer_commits_judge_says_abstain"
        assert out.answerer_abstained is False

    def test_no_validation_returns_none(self) -> None:
        # Unjudged records (validation=None) can't contribute to the audit.
        answer = _judged_answer(abstained=True, judge_score=0.9)
        unjudged = answer.model_copy(update={"validation": None})
        assert detect_disagreement(unjudged) is None

    def test_judge_error_flagged_as_its_own_disagreement_type(self) -> None:
        # When the abstention criterion errored (score is None), the row
        # is emitted with disagreement_type="judge_error" so the auditor
        # sees the gap rather than silently treating None as "judge agrees".
        record = _judged_answer(abstained=False, judge_score=0.5)
        # Replace the score with an errored one.
        errored_validation = JudgePipelineResult(
            stage_results={
                "answer_judge": JudgeStepResult(
                    criterion_scores={
                        "abstention": CriterionScore(
                            score=None,
                            threshold=0.7,
                            rationale="",
                            error="LLM timeout",
                        )
                    }
                )
            },
            passed=True,
        )
        record = record.model_copy(update={"validation": errored_validation})

        out = detect_disagreement(record)
        assert out is not None
        assert out.disagreement_type == "judge_error"
        assert out.judge_score is None
        assert out.judge_threshold == 0.7

    def test_recorded_threshold_matches_decision_threshold(self) -> None:
        # If the abstention prompt config ships a non-default threshold,
        # both the decision AND the recorded `judge_threshold` field must
        # use that value (single source of truth — CriterionScore.threshold).
        # answerer abstained, score 0.85 with threshold 0.5 → judge also
        # says abstain → no disagreement. If the function ignored the
        # criterion's threshold and used a hardcoded 0.7, this would
        # falsely flag a disagreement.
        answer = _judged_answer(abstained=True, judge_score=0.85, threshold=0.5)
        assert detect_disagreement(answer) is None


class TestWriteAuditLog:
    def test_empty_disagreements_writes_nothing(self, tmp_path: Path) -> None:
        result = write_audit_log(tmp_path, [])
        assert result is None
        assert not (tmp_path / "abstention_audit.jsonl").exists()

    def test_writes_one_jsonl_row_per_disagreement(self, tmp_path: Path) -> None:
        rows = [
            AbstentionDisagreement(
                qa_pair_id="a:0",
                retriever_id="bm25",
                answerer_abstained=True,
                judge_score=0.1,
                judge_threshold=0.7,
                disagreement_type="answerer_abstains_judge_disagrees",
                answer_text=None,
                rationale="r1",
            ),
            AbstentionDisagreement(
                qa_pair_id="a:1",
                retriever_id="null",
                answerer_abstained=False,
                judge_score=0.95,
                judge_threshold=0.7,
                disagreement_type="answerer_commits_judge_says_abstain",
                answer_text="committed answer",
                rationale="r2",
            ),
        ]
        path = write_audit_log(tmp_path, rows)
        assert path is not None
        assert path.name == "abstention_audit.jsonl"
        # JSONL: one record per line.
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        # Round-trip via the schema.
        roundtrip_0 = AbstentionDisagreement.model_validate_json(lines[0])
        assert roundtrip_0.qa_pair_id == "a:0"
        assert roundtrip_0.disagreement_type == "answerer_abstains_judge_disagrees"
