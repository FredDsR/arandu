"""Tests for the emic-validity judge batch (spec §5)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.emic.batch import run_emic_judge_batch
from arandu.shared.emic.schemas import EmicSourceScores
from arandu.shared.emic.settings import EmicJudgeSettings
from arandu.shared.judge.criterion import OrdinalCriterionResponse
from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _pair(
    question: str, *, approved: bool, bloom: str = "analyze", judged: bool = True
) -> QAPairCEP:
    validation = JudgePipelineResult(stage_results={}, passed=approved) if judged else None
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
    mocker.patch("arandu.shared.emic.batch.build_llm_client_from_settings", return_value=client)
    return client


@pytest.fixture
def settings() -> EmicJudgeSettings:
    return EmicJudgeSettings(provider="ollama", model_id="test-model")


class TestEmicJudgeBatch:
    def test_scores_only_approved_pairs(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
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

        result = run_emic_judge_batch(
            "run1", settings=settings, base_dir=tmp_path, scope="approved"
        )

        assert result.approved_pairs == 2
        assert result.rejected_pairs == 1
        assert result.selected_pairs == 2  # the rejected pair is skipped
        assert result.skipped_pairs == 1
        assert result.scored_pairs == 2
        assert result.failed_pairs == 0
        assert result.sources == 1
        assert result.completed_sources == 1
        assert result.resumed_sources == 0
        assert result.failed_sources == 0
        assert result.unjudged_pairs == 0

        out = EmicSourceScores.load(
            tmp_path / "run1" / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert [s.pair_index for s in out.scores] == [0, 2]  # original indices preserved
        assert all(s.emic_score == 2 for s in out.scores)
        assert {s.bloom_level for s in out.scores} == {"analyze", "evaluate"}

    def test_scope_all_scores_every_pair_and_records_the_verdict(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        # The default scope. Every pair is scored regardless of the judge-qa
        # verdict, and the verdict rides along on each score so emic validity
        # can be cross-tabulated against approval downstream.
        cep_outputs = tmp_path / "run_all" / "cep" / "outputs"
        _write_cep_record(
            cep_outputs,
            "src1",
            [
                _pair("Q approved", approved=True),
                _pair("Q rejected", approved=False),
                _pair("Q unjudged", approved=False, judged=False),
            ],
        )

        result = run_emic_judge_batch("run_all", settings=settings, base_dir=tmp_path)

        assert result.scope == "all"
        assert result.selected_pairs == 3
        assert result.scored_pairs == 3
        assert result.skipped_pairs == 0
        assert result.approved_pairs == 1
        assert result.rejected_pairs == 1
        assert result.unjudged_pairs == 1
        assert mock_emic_client.generate_structured.call_count == 3

        out = EmicSourceScores.load(
            tmp_path / "run_all" / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert [s.pair_index for s in out.scores] == [0, 1, 2]
        assert [s.is_valid for s in out.scores] == [True, False, None]

    def test_scope_counters_reconcile(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        # The invariants the result docstring promises, asserted under both
        # scopes so a future counter change cannot drift from the docs.
        pairs = [
            _pair("A", approved=True),
            _pair("B", approved=True),
            _pair("C", approved=False),
            _pair("D", approved=False, judged=False),
        ]
        for idx, scope in enumerate(("all", "approved")):
            run_id = f"run_rec{idx}"
            _write_cep_record(tmp_path / run_id / "cep" / "outputs", "src1", list(pairs))

            result = run_emic_judge_batch(run_id, settings=settings, base_dir=tmp_path, scope=scope)

            assert result.selected_pairs == result.scored_pairs + result.failed_pairs
            assert result.selected_pairs + result.skipped_pairs == (
                result.approved_pairs + result.rejected_pairs + result.unjudged_pairs
            )
            assert result.approved_pairs == 2
            assert result.rejected_pairs == 1
            assert result.unjudged_pairs == 1

    @pytest.mark.parametrize("workers", [1, 4])
    def test_output_order_is_stable_across_worker_counts(
        self, tmp_path: Path, mocker: MockerFixture, workers: int
    ) -> None:
        # map_concurrent yields in completion order once workers > 1, so the
        # batch must restore pair order before persisting. A per-pair score
        # keyed off the question proves the score rides with the right pair and
        # not merely that the indices are sorted.
        client = mocker.MagicMock()
        client.generate_structured.side_effect = lambda prompt, **_: OrdinalCriterionResponse(
            score=int(prompt.split("MARK")[1][0]), rationale="r"
        )
        mocker.patch("arandu.shared.emic.batch.build_llm_client_from_settings", return_value=client)

        run_id = f"run_order{workers}"
        _write_cep_record(
            tmp_path / run_id / "cep" / "outputs",
            "src1",
            [_pair(f"MARK{n} pergunta", approved=True) for n in (1, 2, 3, 4, 5)],
        )

        result = run_emic_judge_batch(
            run_id,
            settings=EmicJudgeSettings(provider="ollama", model_id="m", workers=workers),
            base_dir=tmp_path,
        )

        assert result.scored_pairs == 5
        out = EmicSourceScores.load(
            tmp_path / run_id / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert [s.pair_index for s in out.scores] == [0, 1, 2, 3, 4]
        assert [s.emic_score for s in out.scores] == [1, 2, 3, 4, 5]

    def test_worker_exception_isolates_the_pair(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # JudgeCriterion.evaluate normally swallows failures into an error
        # score, so this covers the defensive branch: anything escaping the
        # criterion must fail one pair, not the whole source.
        mocker.patch(
            "arandu.shared.emic.batch.build_llm_client_from_settings",
            return_value=mocker.MagicMock(),
        )
        calls = {"n": 0}

        def _flaky(**_: Any) -> CriterionScore:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom outside the criterion")
            return CriterionScore(ordinal_score=4, scale="ordinal", threshold=0.0, rationale="r")

        mocker.patch(
            "arandu.shared.judge.criterion.OrdinalLLMCriterion.evaluate", side_effect=_flaky
        )

        _write_cep_record(
            tmp_path / "run_iso" / "cep" / "outputs",
            "src1",
            [_pair(f"Q{n}", approved=True) for n in range(3)],
        )

        result = run_emic_judge_batch(
            "run_iso",
            settings=EmicJudgeSettings(provider="ollama", model_id="m", workers=1),
            base_dir=tmp_path,
        )

        assert result.selected_pairs == 3
        assert result.failed_pairs == 1
        assert result.completed_sources == 1  # the source still got persisted
        out = EmicSourceScores.load(
            tmp_path / "run_iso" / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert len(out.scores) == 3
        failed = [s for s in out.scores if s.emic_score is None]
        assert len(failed) == 1
        assert "boom outside the criterion" in (failed[0].error or "")

    def test_missing_cep_stage_raises(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            run_emic_judge_batch("absent", settings=settings, base_dir=tmp_path)

    def test_resume_skips_completed_sources(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        cep_outputs = tmp_path / "run2" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        run_emic_judge_batch("run2", settings=settings, base_dir=tmp_path)
        calls_after_first = mock_emic_client.generate_structured.call_count
        assert calls_after_first == 1

        # Second run resumes: the source is already checkpointed, no new calls.
        second = run_emic_judge_batch("run2", settings=settings, base_dir=tmp_path)
        assert mock_emic_client.generate_structured.call_count == calls_after_first
        assert second.scored_pairs == 0  # nothing re-scored on resume
        assert second.approved_pairs == 0  # resumed sources are not re-counted
        assert second.completed_sources == 0
        assert second.resumed_sources == 1  # the prior run's source is accounted for
        assert second.sources == 1

    def test_rerun_rescores(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        cep_outputs = tmp_path / "run3" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        run_emic_judge_batch("run3", settings=settings, base_dir=tmp_path)
        run_emic_judge_batch("run3", settings=settings, base_dir=tmp_path, rerun=True)
        assert mock_emic_client.generate_structured.call_count == 2

    def test_llm_error_records_failed_pair(
        self, tmp_path: Path, mocker: MockerFixture, settings: EmicJudgeSettings
    ) -> None:
        client = mocker.MagicMock()
        client.generate_structured.side_effect = RuntimeError("llm down")
        mocker.patch("arandu.shared.emic.batch.build_llm_client_from_settings", return_value=client)

        cep_outputs = tmp_path / "run4" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        result = run_emic_judge_batch("run4", settings=settings, base_dir=tmp_path)
        assert result.failed_pairs == 1
        assert result.scored_pairs == 0
        out = EmicSourceScores.load(
            tmp_path / "run4" / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert out.scores[0].emic_score is None
        assert out.scores[0].error is not None

    def test_unjudged_pairs_are_skipped_and_flagged(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        # A run that was CEP-populated but never judge-qa'd: is_valid is None
        # for every pair, so nothing is scored and the result flags the gap
        # instead of reporting a clean 0/0 success.
        cep_outputs = tmp_path / "run5" / "cep" / "outputs"
        _write_cep_record(
            cep_outputs,
            "src1",
            [_pair("Q1", approved=False, judged=False), _pair("Q2", approved=False, judged=False)],
        )

        result = run_emic_judge_batch(
            "run5", settings=settings, base_dir=tmp_path, scope="approved"
        )

        assert result.unjudged_pairs == 2
        assert result.approved_pairs == 0
        assert result.selected_pairs == 0
        assert result.skipped_pairs == 2
        assert result.scored_pairs == 0
        assert mock_emic_client.generate_structured.call_count == 0  # nothing scored
        assert result.completed_sources == 1  # source still processed (empty scores)
        out = EmicSourceScores.load(
            tmp_path / "run5" / "emic_judge" / "outputs" / "src1_cep_qa.json"
        )
        assert out.scores == []

    def test_load_failure_counts_failed_source_and_marks_run_failed(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:

        cep_outputs = tmp_path / "run6" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "good", [_pair("Q", approved=True)])
        # A corrupt CEP artifact must be counted, not silently dropped.
        (cep_outputs / "broken_cep_qa.json").write_text("{not valid json", encoding="utf-8")

        result = run_emic_judge_batch("run6", settings=settings, base_dir=tmp_path)

        assert result.sources == 2
        assert result.completed_sources == 1
        assert result.failed_sources == 1
        assert not (tmp_path / "run6" / "emic_judge" / "outputs" / "broken_cep_qa.json").exists()

        metadata = json.loads(
            (tmp_path / "run6" / "emic_judge" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["status"] == "failed"  # a failed source marks the run FAILED

    def test_run_marked_completed_on_success(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:

        cep_outputs = tmp_path / "run7" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])

        run_emic_judge_batch("run7", settings=settings, base_dir=tmp_path)

        metadata = json.loads(
            (tmp_path / "run7" / "emic_judge" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["status"] == "completed"  # no longer stuck IN_PROGRESS

    def test_rerun_clears_stale_outputs(
        self, tmp_path: Path, mock_emic_client: Any, settings: EmicJudgeSettings
    ) -> None:
        cep_outputs = tmp_path / "run8" / "cep" / "outputs"
        _write_cep_record(cep_outputs, "src1", [_pair("Q", approved=True)])
        run_emic_judge_batch("run8", settings=settings, base_dir=tmp_path)

        # Simulate a stale output from a prior, larger corpus that is no longer
        # present in cep/outputs.
        outputs_dir = tmp_path / "run8" / "emic_judge" / "outputs"
        stale = outputs_dir / "removed_source_cep_qa.json"
        stale.write_text("{}", encoding="utf-8")

        run_emic_judge_batch("run8", settings=settings, base_dir=tmp_path, rerun=True)

        assert not stale.exists()  # --rerun purged the orphaned scores
        assert (outputs_dir / "src1_cep_qa.json").exists()  # live source re-scored
