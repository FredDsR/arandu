"""Batch orchestrator for ``arandu emic-prepass`` (spec §5).

Runs the ``emic_validity`` ordinal criterion over the canonical-approved CEP
pairs of a populated run and writes per-source ordinal scores under
``results/<id>/emic_prepass/outputs/<source>.json``. These scores feed the
stratified sample builder (they bound the sampling bands; the human annotators
remain the ground truth).

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
from arandu.shared.emic.schemas import EmicPrepassResult, EmicScore, EmicSourceScores
from arandu.shared.emic.settings import EMIC_ENV_PREFIX, default_emic_settings
from arandu.shared.judge.criterion import OrdinalLLMCriterion
from arandu.shared.llm_client import LLMSettings, build_llm_client_from_settings
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "emic_prepass_checkpoint.json"
EMIC_CRITERION_NAME = "emic_validity"


def run_emic_prepass_batch(
    pipeline_id: str,
    *,
    settings: LLMSettings | None = None,
    base_dir: Path | None = None,
    rerun: bool = False,
) -> EmicPrepassResult:
    """Score the canonical-approved CEP pairs of ``pipeline_id`` for emic validity.

    Args:
        pipeline_id: Run identifier. The ``cep`` stage must be populated and
            judged (only pairs with ``is_valid`` are scored).
        settings: Emic-prepass LLM configuration. Defaults to
            :func:`default_emic_settings` (reads ``ARANDU_EMIC_PREPASS_*``).
        base_dir: Override the project ``results/`` root.
        rerun: If True, clear the checkpoint so every source is re-scored.

    Returns:
        :class:`EmicPrepassResult` summary.

    Raises:
        FileNotFoundError: If the cep stage outputs aren't present.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved = settings if settings is not None else default_emic_settings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    cep_outputs = base / pipeline_id / "cep" / "outputs"
    if not cep_outputs.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_outputs}. "
            f"Run `arandu generate-cep-qa` and `arandu judge-qa` first."
        )

    llm_client = build_llm_client_from_settings(resolved, env_prefix=EMIC_ENV_PREFIX)
    criterion = OrdinalLLMCriterion.from_config(
        name=EMIC_CRITERION_NAME,
        prompts_dir=get_project_root() / "prompts" / "judge" / "criteria",
        language=resolved.language,
        llm_client=llm_client,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
    )

    results_mgr = ResultsManager(base, PipelineType.EMIC_PREPASS, pipeline_id=pipeline_id)
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

    cep_paths = sorted(cep_outputs.glob("*_cep_qa.json"))
    checkpoint.set_total_files(len(cep_paths))

    completed_sources = resumed_sources = failed_sources = 0
    approved = scored = failed = unjudged = 0
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

        scores: list[EmicScore] = []
        for idx, pair in enumerate(record.qa_pairs):
            if pair.is_valid is None:
                unjudged += 1  # never judged; can't be canonically approved
                continue
            if not pair.is_valid:
                continue  # judged and rejected — only approved pairs are scored
            approved += 1
            result = criterion.evaluate(
                context=pair.context,
                question=pair.question,
                answer=pair.answer,
            )
            if result.ordinal_score is None:
                failed += 1
            else:
                scored += 1
            scores.append(
                EmicScore(
                    pair_index=idx,
                    bloom_level=pair.bloom_level,
                    emic_score=result.ordinal_score,
                    rationale=result.rationale,
                    error=result.error,
                )
            )

        EmicSourceScores(
            source_file_id=record.source_file_id,
            source_filename=record.source_filename,
            scores=scores,
        ).save(results_mgr.outputs_dir / f"{path.stem}.json")
        checkpoint.mark_completed(ckpt_key)
        completed_sources += 1

    if unjudged:
        logger.warning(
            "%d pair(s) had no judge verdict (is_valid is None) and were skipped; "
            "the run may not have been fully judged via `arandu judge-qa`.",
            unjudged,
        )

    # Mirror the judge-answers convention: record progress and finalize the run
    # so run_metadata flips to COMPLETED and the run enters the global index
    # (otherwise it is stuck IN_PROGRESS and invisible to get_latest_run /
    # list_runs). A load failure marks the run FAILED but is non-fatal.
    results_mgr.update_progress(completed_sources + resumed_sources, failed_sources, len(cep_paths))
    results_mgr.complete_run(success=(failed_sources == 0))

    logger.info(
        "Emic pre-pass complete: %d/%d sources scored (%d resumed, %d failed), "
        "%d approved pairs, %d scored, %d failed, %d unjudged.",
        completed_sources,
        len(cep_paths),
        resumed_sources,
        failed_sources,
        approved,
        scored,
        failed,
        unjudged,
    )
    return EmicPrepassResult(
        pipeline_id=pipeline_id,
        sources=len(cep_paths),
        completed_sources=completed_sources,
        resumed_sources=resumed_sources,
        failed_sources=failed_sources,
        approved_pairs=approved,
        scored_pairs=scored,
        failed_pairs=failed,
        unjudged_pairs=unjudged,
    )
