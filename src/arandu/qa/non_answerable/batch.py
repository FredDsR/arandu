"""Batch driver for ``arandu generate-non-answerable`` (spec §7).

Samples validated CEP pairs (stratified by Bloom level), perturbs each
into a non-answerable twin via one LLM call + code-side verification,
and writes a :class:`NonAnswerableDataset` under
``results/<id>/non_answerable/outputs/dataset.json``.

Per-seed checkpointing (keyed by ``parent_qa_pair_id``) makes the ~400
LLM calls cost-safe to resume: a killed or re-run job skips seeds that
already succeeded or were already attempted-and-collided.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

from arandu.qa.non_answerable.corpus_index import SourceCorpusIndex, load_kg_node_set
from arandu.qa.non_answerable.perturbation import (
    perturb_to_non_answerable,
    stratified_seed_sample,
)
from arandu.qa.non_answerable.schemas import NonAnswerableDataset, NonAnswerableItem
from arandu.qa.non_answerable.settings import NonAnswerableSettings
from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "non_answerable_checkpoint.json"
_ITEMS_SUBDIR = "items"

# One perturbation per seed is the design (spec §7.7): ~400 seeds already
# give sufficient power, and a second swap on the same seed yields a
# near-duplicate. Recorded in the dataset for provenance; not configurable.
_PERTURBATIONS_PER_SEED = 1


class NonAnswerableBatchConfig(BaseModel):
    """Run-metadata snapshot for the non_answerable stage."""

    pipeline_id: str
    provider: str
    model_id: str
    language: str
    seeds_per_bloom: int
    rng_seed: int


class NonAnswerableBatchResult(BaseModel):
    """Summary of a completed non-answerable generation run."""

    pipeline_id: str
    run_dir: str
    dataset_path: str
    seed_count: int
    items_built: int  # newly perturbed this invocation
    dataset_items: int  # total items in dataset.json (incl. resumed)
    seeds_skipped_resumed: int
    seeds_failed: int
    success_rate: float


def run_generate_non_answerable_batch(
    pipeline_id: str,
    *,
    settings: NonAnswerableSettings | None = None,
    base_dir: Path | None = None,
    regenerate: bool = False,
) -> NonAnswerableBatchResult:
    """Generate the non-answerable benchmark for ``pipeline_id``.

    Args:
        pipeline_id: Run identifier. The ``cep`` stage must be populated;
            ``kg`` and ``transcription`` outputs are consulted for the
            absence checks (their absence weakens, but does not abort,
            verification - logged as a warning).
        settings: Generation config. Defaults to
            :class:`NonAnswerableSettings` (reads ``ARANDU_NONANSWERABLE_*``).
        base_dir: Override the project ``results/`` root.
        regenerate: If True, clear the checkpoint + previously written
            items before running so every seed is re-perturbed.

    Returns:
        :class:`NonAnswerableBatchResult` summary.

    Raises:
        FileNotFoundError: If the CEP stage outputs aren't present.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved = settings if settings is not None else NonAnswerableSettings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    cep_dir = base / pipeline_id / "cep" / "outputs"
    if not cep_dir.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_dir}. "
            f"Run `arandu generate-cep-qa` + `arandu judge-qa` first."
        )

    kg_graphml_dir = base / pipeline_id / "kg" / "outputs" / "atlas_output" / "kg_graphml"
    kg_node_set = _load_kg_nodes(kg_graphml_dir)
    corpus_index = SourceCorpusIndex(base / pipeline_id / "transcription" / "outputs")
    logger.info("Absence gates: %d KG nodes, %d corpus spans.", len(kg_node_set), len(corpus_index))

    llm_client = _build_llm_client(resolved)

    config = NonAnswerableBatchConfig(
        pipeline_id=pipeline_id,
        provider=resolved.provider,
        model_id=resolved.model_id,
        language=resolved.language,
        seeds_per_bloom=resolved.seeds_per_bloom,
        rng_seed=resolved.rng_seed,
    )
    results_mgr = ResultsManager(base, PipelineType.NON_ANSWERABLE, pipeline_id=pipeline_id)
    results_mgr.create_run(
        config, input_source=str(cep_dir), checkpoint_filename=CHECKPOINT_FILENAME
    )

    items_dir = results_mgr.outputs_dir / _ITEMS_SUBDIR
    checkpoint_path = results_mgr.run_dir / CHECKPOINT_FILENAME
    if regenerate:
        _reset_state(checkpoint_path, items_dir)
    items_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(checkpoint_path)

    seeds = stratified_seed_sample(
        cep_dir, seeds_per_bloom=resolved.seeds_per_bloom, rng_seed=resolved.rng_seed
    )
    checkpoint.set_total_files(len(seeds))

    built = 0
    resumed = 0
    failed = 0
    for seed in seeds:
        key = seed.parent_qa_pair_id
        if checkpoint.is_completed(key) or key in checkpoint.state.failed_files:
            resumed += 1
            continue
        item = perturb_to_non_answerable(
            seed,
            kg_node_set=kg_node_set,
            corpus_index=corpus_index,
            llm=llm_client,
            language=resolved.language,
            base_temperature=resolved.base_temperature,
            max_retries=resolved.retry_max,
        )
        if item is None:
            checkpoint.mark_failed(key, "no valid swap after retries")
            failed += 1
            continue
        item_path = items_dir / f"{_safe_name(key)}.json"
        item_path.write_text(item.model_dump_json(indent=2), encoding="utf-8")
        checkpoint.mark_completed(key)
        built += 1

    items = _collect_items(items_dir)
    success_rate = len(items) / len(seeds) if seeds else 0.0
    dataset = NonAnswerableDataset(
        items=items,
        seed_cep_dataset=str(cep_dir),
        kg_artifact=str(kg_graphml_dir),
        seed_count=len(seeds),
        perturbations_per_seed=_PERTURBATIONS_PER_SEED,
        success_rate=success_rate,
        rng_seed=resolved.rng_seed,
    )
    dataset_path = results_mgr.outputs_dir / "dataset.json"
    dataset.save(dataset_path)

    results_mgr.update_progress(built + resumed, failed, len(seeds))
    results_mgr.complete_run(success=True)

    return NonAnswerableBatchResult(
        pipeline_id=pipeline_id,
        run_dir=str(results_mgr.run_dir),
        dataset_path=str(dataset_path),
        seed_count=len(seeds),
        items_built=built,
        dataset_items=len(items),
        seeds_skipped_resumed=resumed,
        seeds_failed=failed,
        success_rate=success_rate,
    )


def _load_kg_nodes(kg_graphml_dir: Path) -> set[str]:
    """Union the normalized node labels across every ``*.graphml`` in the dir."""
    if not kg_graphml_dir.exists():
        logger.warning(
            "KG graphml dir absent at %s; KG absence gate will be empty.", kg_graphml_dir
        )
        return set()
    nodes: set[str] = set()
    for path in sorted(kg_graphml_dir.glob("*.graphml")):
        nodes |= load_kg_node_set(path)
    return nodes


def _collect_items(items_dir: Path) -> list[NonAnswerableItem]:
    """Load every persisted item, sorted by ``qa_pair_id`` for determinism."""
    items: list[NonAnswerableItem] = []
    for path in sorted(items_dir.glob("*.json")):
        try:
            items.append(NonAnswerableItem.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable item %s: %s", path, exc)
    items.sort(key=lambda it: it.qa_pair_id)
    return items


def _reset_state(checkpoint_path: Path, items_dir: Path) -> None:
    """Clear checkpoint + previously written items for a clean regenerate."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if items_dir.exists():
        for path in items_dir.glob("*.json"):
            path.unlink()


def _safe_name(qa_pair_id: str) -> str:
    """Turn a composite id into a filesystem-safe stem (colons → underscores)."""
    return qa_pair_id.replace(":", "_").replace("/", "_")


def _build_llm_client(settings: NonAnswerableSettings) -> LLMClient:
    """Construct the unified LLMClient from settings (mirrors the judge stage)."""
    try:
        provider_enum = LLMProvider(settings.provider)
    except ValueError as exc:
        raise ValueError(
            f"Unknown non_answerable provider {settings.provider!r}. "
            f"Valid: {[p.value for p in LLMProvider]}."
        ) from exc

    api_key = os.environ.get(settings.api_key_env)
    if not api_key and provider_enum is not LLMProvider.OLLAMA:
        raise RuntimeError(
            f"non_answerable provider {settings.provider!r} requires "
            f"{settings.api_key_env} to be set. Either set it, or switch "
            f"ARANDU_NONANSWERABLE_PROVIDER to 'ollama'."
        )
    if provider_enum is LLMProvider.CUSTOM and not settings.base_url:
        raise ValueError(
            "provider='custom' requires a base URL. Set ARANDU_NONANSWERABLE_BASE_URL "
            "or pass base_url=... explicitly."
        )

    return LLMClient(
        provider=provider_enum,
        model_id=settings.model_id,
        api_key=api_key,
        base_url=settings.base_url,
    )
