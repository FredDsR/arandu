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
import os
from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.qa.schemas import QARecordCEP
from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.emic.schemas import EmicPrepassResult, EmicScore, EmicSourceScores
from arandu.shared.emic.settings import EmicPrepassSettings
from arandu.shared.judge.criterion import OrdinalLLMCriterion
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "emic_prepass_checkpoint.json"
EMIC_CRITERION_NAME = "emic_validity"


def _build_llm_client(settings: EmicPrepassSettings) -> LLMClient:
    """Construct the unified LLMClient from emic-prepass settings."""
    try:
        provider_enum = LLMProvider(settings.provider)
    except ValueError as exc:
        raise ValueError(
            f"Unknown emic-prepass provider {settings.provider!r}. "
            f"Valid: {[p.value for p in LLMProvider]}."
        ) from exc

    api_key = os.environ.get(settings.api_key_env)
    if not api_key and provider_enum is not LLMProvider.OLLAMA:
        raise RuntimeError(
            f"emic-prepass provider {settings.provider!r} requires "
            f"{settings.api_key_env} to be set, or switch "
            f"ARANDU_EMIC_PREPASS_PROVIDER to 'ollama'."
        )
    if provider_enum is LLMProvider.CUSTOM and not settings.base_url:
        raise ValueError(
            "provider='custom' requires a base URL. Set "
            "ARANDU_EMIC_PREPASS_BASE_URL or pass base_url=... explicitly."
        )

    return LLMClient(
        provider=provider_enum,
        model_id=settings.model_id,
        api_key=api_key,
        base_url=settings.base_url,
    )


def run_emic_prepass_batch(
    pipeline_id: str,
    *,
    settings: EmicPrepassSettings | None = None,
    base_dir: Path | None = None,
    rerun: bool = False,
) -> EmicPrepassResult:
    """Score the canonical-approved CEP pairs of ``pipeline_id`` for emic validity.

    Args:
        pipeline_id: Run identifier. The ``cep`` stage must be populated and
            judged (only pairs with ``is_valid`` are scored).
        settings: Emic-prepass configuration. Defaults to
            :class:`EmicPrepassSettings` (reads ``ARANDU_EMIC_PREPASS_*``).
        base_dir: Override the project ``results/`` root.
        rerun: If True, clear the checkpoint so every source is re-scored.

    Returns:
        :class:`EmicPrepassResult` summary.

    Raises:
        FileNotFoundError: If the cep stage outputs aren't present.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved = settings if settings is not None else EmicPrepassSettings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    cep_outputs = base / pipeline_id / "cep" / "outputs"
    if not cep_outputs.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_outputs}. "
            f"Run `arandu generate-cep-qa` and `arandu judge-qa` first."
        )

    llm_client = _build_llm_client(resolved)
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
    if rerun and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = CheckpointManager(checkpoint_path)

    cep_paths = sorted(cep_outputs.glob("*_cep_qa.json"))
    checkpoint.set_total_files(len(cep_paths))

    approved = scored = failed = 0
    for path in cep_paths:
        ckpt_key = path.stem
        if checkpoint.is_completed(ckpt_key):
            continue
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping %s: load failed: %s", path.name, exc)
            checkpoint.mark_failed(ckpt_key, f"load failed: {exc}")
            continue

        scores: list[EmicScore] = []
        for idx, pair in enumerate(record.qa_pairs):
            if not pair.is_valid:
                continue  # only canonical-approved pairs enter the emic pre-pass
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

    logger.info(
        "Emic pre-pass complete: %d sources, %d approved pairs, %d scored, %d failed.",
        len(cep_paths),
        approved,
        scored,
        failed,
    )
    return EmicPrepassResult(
        pipeline_id=pipeline_id,
        sources=len(cep_paths),
        approved_pairs=approved,
        scored_pairs=scored,
        failed_pairs=failed,
    )
