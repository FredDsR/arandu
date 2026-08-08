"""Batch orchestrator for ``arandu emic-judge`` (spec §5).

Runs the ``emic_validity`` ordinal criterion over the CEP pairs of a populated
run and writes per-source ordinal scores under
``results/<id>/emic_judge/outputs/<source>.json``. The ``scope`` argument picks
the pairs: ``all`` (default) scores every pair and records its ``judge-qa``
verdict on the score, ``approved`` scores only canonically-approved ones.

These scores are the study's **measurement** of emic validity, not a
preliminary aid. The human annotation round (spec §6) rates a stratified
subsample and reports agreement with them (Krippendorff alpha over the raters,
weighted Cohen kappa of this judge against each annotator); it validates the
measurement rather than replacing it. The same scores also band the stratified
sample builder, but that is a downstream use, not their purpose.

The criterion is built standalone via ``OrdinalLLMCriterion.from_config`` — it
is not wired into the ``judge-qa`` pipeline (that, with a filter threshold, is
the separate ``emic-filter-stage`` task).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.qa.schemas import QARecordCEP
from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.emic.schemas import (
    EmicJudgeResult,
    EmicScope,
    EmicScore,
    EmicSourceScores,
)
from arandu.shared.emic.settings import EmicJudgeSettings
from arandu.shared.judge.criterion import OrdinalLLMCriterion
from arandu.shared.llm_client import build_llm_client_from_settings
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType
from arandu.utils.concurrency import map_concurrent
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.qa.schemas import QAPairCEP
    from arandu.shared.judge.schemas import CriterionScore

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "emic_judge_checkpoint.json"
EMIC_CRITERION_NAME = "emic_validity"


def run_emic_judge_batch(
    pipeline_id: str,
    *,
    settings: EmicJudgeSettings | None = None,
    base_dir: Path | None = None,
    rerun: bool = False,
    scope: EmicScope = "all",
) -> EmicJudgeResult:
    """Score the CEP pairs of ``pipeline_id`` for emic validity.

    Args:
        pipeline_id: Run identifier. The ``cep`` stage must be populated, and
            judged if ``scope`` is ``"approved"``.
        settings: Emic-judge LLM configuration. Defaults to
            :class:`EmicJudgeSettings` (reads ``ARANDU_EMIC_JUDGE_*``).
        base_dir: Override the project ``results/`` root.
        rerun: If True, clear the checkpoint so every source is re-scored.
        scope: Which pairs to score. ``"all"`` (default) scores every pair and
            records its ``judge-qa`` verdict, so emic validity can be
            cross-tabulated against approval; ``"approved"`` scores only
            canonically-approved pairs. Defaults to ``"all"`` because the emic
            score is the study's corpus-wide measurement, not a filter applied
            after ``judge-qa`` — at the cost of scoring the rejected pairs too.

    Returns:
        :class:`EmicJudgeResult` summary.

    Raises:
        FileNotFoundError: If the cep stage outputs aren't present.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved = settings if settings is not None else EmicJudgeSettings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    cep_outputs = base / pipeline_id / "cep" / "outputs"
    if not cep_outputs.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_outputs}. "
            f"Run `arandu generate-cep-qa` and `arandu judge-qa` first."
        )

    llm_client = build_llm_client_from_settings(resolved)
    criterion = OrdinalLLMCriterion.from_config(
        name=EMIC_CRITERION_NAME,
        prompts_dir=get_project_root() / "prompts" / "judge" / "criteria",
        language=resolved.language,
        llm_client=llm_client,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
    )

    results_mgr = ResultsManager(base, PipelineType.EMIC_JUDGE, pipeline_id=pipeline_id)
    results_mgr.create_run(
        resolved,
        input_source=str(cep_outputs),
        checkpoint_filename=CHECKPOINT_FILENAME,
    )
    checkpoint_path = results_mgr.run_dir / CHECKPOINT_FILENAME
    if rerun:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        # Resetting only the checkpoint would leave per-source outputs from a
        # prior run in outputs_dir. If the CEP stage was regenerated with a
        # different (e.g. smaller) corpus since, those orphaned files would be
        # globbed as live scores by the stratified sample builder. Clear them.
        for stale in results_mgr.outputs_dir.glob("*.json"):
            stale.unlink()
    checkpoint = CheckpointManager(checkpoint_path)

    def _score_pair(item: tuple[int, QAPairCEP]) -> CriterionScore:
        """Evaluate one pair. Runs on a worker thread; must not touch state."""
        _, pair = item
        return criterion.evaluate(
            context=pair.context,
            question=pair.question,
            answer=pair.answer,
        )

    cep_paths = sorted(cep_outputs.glob("*_cep_qa.json"))
    checkpoint.set_total_files(len(cep_paths))

    completed_sources = resumed_sources = failed_sources = 0
    selected = scored = failed = skipped = 0
    approved = rejected = unjudged = 0
    for path in cep_paths:
        ckpt_key = path.stem
        if checkpoint.is_completed(ckpt_key):
            resumed_sources += 1
            continue
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping %s: load failed: %s", path.name, exc)
            checkpoint.mark_failed(ckpt_key, f"load failed: {exc}")
            failed_sources += 1
            continue

        in_scope: list[tuple[int, QAPairCEP]] = []
        for idx, pair in enumerate(record.qa_pairs):
            if pair.is_valid is None:
                unjudged += 1  # never judged; cannot be canonically approved
            elif pair.is_valid:
                approved += 1
            else:
                rejected += 1

            # Under "approved" only canonically-approved pairs are scored; a
            # never-judged pair is not approved, so it is skipped too. Under
            # "all" every pair is scored and its verdict (including None) is
            # recorded on the score, which is what makes the emic-validity x
            # judge-approval cross-tabulation possible downstream.
            if scope == "approved" and pair.is_valid is not True:
                skipped += 1
                continue
            in_scope.append((idx, pair))

        selected += len(in_scope)
        # Workers only run `_score_pair`; the checkpoint write and the source
        # file save stay on this thread, so no locking is needed (same split as
        # judge-answers). Concurrency is per source rather than across the whole
        # corpus because a source's scores are persisted as one file: the loop
        # drains before writing, which costs a short tail per source but keeps
        # the per-source checkpoint honest.
        by_index: dict[int, EmicScore] = {}
        #
        # No `rate_limit_of`: it would be dead code here. The adaptive throttle
        # only sees exceptions that escape `fn`, but `JudgeCriterion.evaluate`
        # catches every exception and returns an error score
        # (judge/criterion.py), so an exhausted 429 budget never reaches this
        # layer. Passing the predicate would build a throttle that can only ever
        # record successes. The consequence is real and worth knowing before
        # raising workers against a metered provider: a rate-limited pair
        # becomes `emic_score=None` with the error recorded, counted in
        # `failed_pairs` and warned about, rather than being backed off and
        # retried.
        for (idx, pair), evaluation, error in map_concurrent(
            _score_pair,
            in_scope,
            workers=resolved.workers,
        ):
            if error is not None:
                # Defensive: JudgeCriterion.evaluate already converts failures
                # into an error score, so this branch means something outside
                # the criterion broke. Isolate the pair instead of losing the
                # whole source.
                logger.warning("Scoring pair %d of %s failed: %s", idx, path.name, error)
                failed += 1
                by_index[idx] = EmicScore(
                    pair_index=idx,
                    bloom_level=pair.bloom_level,
                    emic_score=None,
                    rationale="",
                    error=str(error),
                    is_valid=pair.is_valid,
                )
                continue
            if evaluation.ordinal_score is None:
                failed += 1
            else:
                scored += 1
            by_index[idx] = EmicScore(
                pair_index=idx,
                bloom_level=pair.bloom_level,
                emic_score=evaluation.ordinal_score,
                rationale=evaluation.rationale,
                error=evaluation.error,
                is_valid=pair.is_valid,
            )

        # map_concurrent yields in completion order once workers > 1; restore
        # pair order so the output file is stable across worker counts.
        scores = [by_index[i] for i in sorted(by_index)]

        EmicSourceScores(
            source_file_id=record.source_file_id,
            source_filename=record.source_filename,
            scores=scores,
        ).save(results_mgr.outputs_dir / f"{path.stem}.json")
        checkpoint.mark_completed(ckpt_key)
        completed_sources += 1

    if unjudged:
        logger.warning(
            "%d pair(s) had no judge verdict (is_valid is None); they were %s. "
            "The run may not have been fully judged via `arandu judge-qa`.",
            unjudged,
            "scored with a null verdict" if scope == "all" else "skipped",
        )

    # Mirror the judge-answers convention: record progress and finalize the run
    # so run_metadata flips to COMPLETED and the run enters the global index
    # (otherwise it is stuck IN_PROGRESS and invisible to get_latest_run /
    # list_runs). A load failure marks the run FAILED but is non-fatal.
    results_mgr.update_progress(completed_sources + resumed_sources, failed_sources, len(cep_paths))
    results_mgr.complete_run(success=(failed_sources == 0))

    logger.info(
        "Emic judge complete (scope=%s): %d/%d sources scored (%d resumed, %d failed), "
        "%d pairs selected, %d scored, %d failed, %d skipped; "
        "corpus split %d approved / %d rejected / %d unjudged.",
        scope,
        completed_sources,
        len(cep_paths),
        resumed_sources,
        failed_sources,
        selected,
        scored,
        failed,
        skipped,
        approved,
        rejected,
        unjudged,
    )
    return EmicJudgeResult(
        pipeline_id=pipeline_id,
        scope=scope,
        sources=len(cep_paths),
        completed_sources=completed_sources,
        resumed_sources=resumed_sources,
        failed_sources=failed_sources,
        selected_pairs=selected,
        scored_pairs=scored,
        failed_pairs=failed,
        skipped_pairs=skipped,
        approved_pairs=approved,
        rejected_pairs=rejected,
        unjudged_pairs=unjudged,
    )
