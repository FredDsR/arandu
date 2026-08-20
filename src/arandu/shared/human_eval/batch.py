"""Batch orchestrator for ``arandu build-human-eval-sample`` (spec §5).

Builds the in-frame pool from the CEP records (keeping the pairs ``judge-qa``
approved, dropping out-of-frame Bloom levels), runs the deterministic
Bloom-stratified sampler, and persists the 120-pair sample + a provenance
manifest under ``results/<id>/human_eval/outputs/``.

Revised 2026-08-19: the pool used to be built from the emic-judge scores, which
put that run on the annotation critical path. Only the emic band needed it, and
the band is gone. See
``docs/superpowers/specs/2026-08-19-bloom-only-sampling-design.md``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from arandu.qa.schemas import QARecordCEP
from arandu.shared.config import ResultsConfig
from arandu.shared.human_eval.sampling import (
    FRAME_BLOOM_LEVELS,
    PER_CELL,
    PoolEntry,
    all_cell_ids,
    build_sample,
    population_by_cell,
)
from arandu.shared.human_eval.schemas import HumanEvalSampleConfig, SampleItem, SampleManifest
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_FILENAME = "sample.jsonl"
MANIFEST_FILENAME = "sample_manifest.json"


def _pool_sha256(pool: list[PoolEntry]) -> str:
    """Hash the full in-frame pool (incl. payload) for reproducibility provenance.

    Canonical JSON per entry, sorted, so the digest is order-independent and
    changes if any payload text (segment/question/answer) drifts -- not just the
    ids -- letting an auditor detect a CEP regeneration under the same pair ids.
    """
    lines = sorted(e.model_dump_json() for e in pool)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def run_build_sample_batch(
    pipeline_id: str,
    *,
    seed: int,
    base_dir: Path | None = None,
    per_cell: int = PER_CELL,
) -> SampleManifest:
    """Build the stratified human-comparison sample for ``pipeline_id``.

    Args:
        pipeline_id: Run identifier. The ``cep`` stage must be populated and
            judged; the ``emic_judge`` stage is NOT read.
        seed: RNG seed for the deterministic selection (recorded in the run
            metadata and the manifest).
        base_dir: Override the project ``results/`` root.
        per_cell: Pairs to draw per cell (default 30 -> 120 total).

    Returns:
        The :class:`SampleManifest` describing the build.

    Raises:
        FileNotFoundError: If the cep stage outputs are absent.
        ValueError: If a stratification cell has fewer than ``per_cell`` pairs,
            if no approved in-frame pair exists at all, or if two CEP records
            share a ``source_file_id`` and collide a ``pair_id``.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    cep_outputs = base / pipeline_id / PipelineType.CEP.value / "outputs"
    if not cep_outputs.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_outputs}. "
            f"Run `arandu generate-cep-qa --id {pipeline_id}` first. The emic-judge stage "
            f"is not needed to build the sample."
        )

    pool: list[PoolEntry] = []
    seen_pair_ids: set[str] = set()
    excluded_not_approved = 0
    excluded_bloom: dict[str, int] = {}
    for cep_path in sorted(cep_outputs.glob("*_cep_qa.json")):
        record = QARecordCEP.load(cep_path)
        for pair_index, pair in enumerate(record.qa_pairs):
            # The study's frame is the corpus judge-qa approved. The verdict is
            # read from the CEP record, which is the authoritative copy: a
            # judge-qa re-run (e.g. a threshold change) is picked up here rather
            # than leaving the frame pinned to a stale snapshot.
            if pair.is_valid is not True:
                excluded_not_approved += 1
                continue
            if pair.bloom_level not in FRAME_BLOOM_LEVELS:
                excluded_bloom[pair.bloom_level] = excluded_bloom.get(pair.bloom_level, 0) + 1
                continue
            pair_id = f"{record.source_file_id}:{pair_index}"
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Duplicate pair_id {pair_id!r} while pooling {cep_path.name}; two CEP "
                    f"records share source_file_id {record.source_file_id!r}, so the join key "
                    f"the annotations resolve through is not unique. Clean "
                    f"results/{pipeline_id}/cep/outputs/ and re-run."
                )
            seen_pair_ids.add(pair_id)
            pool.append(
                PoolEntry(
                    pair_id=pair_id,
                    source_file_id=record.source_file_id,
                    pair_index=pair_index,
                    segment=pair.context,
                    question=pair.question,
                    answer=pair.answer,
                    bloom_level=pair.bloom_level,
                )
            )

    if not pool:
        raise ValueError(
            f"No in-frame approved pairs found for {pipeline_id!r} "
            f"({excluded_not_approved} not judge-approved, "
            f"{sum(excluded_bloom.values())} out-of-frame-Bloom excluded). Check that "
            f"`arandu generate-cep-qa` + `arandu judge-qa` ran and produced approved, "
            f"in-frame ({', '.join(FRAME_BLOOM_LEVELS)}) pairs."
        )

    population = population_by_cell(pool)
    pool_hash = _pool_sha256(pool)

    results_mgr = ResultsManager(base, PipelineType.HUMAN_EVAL, pipeline_id=pipeline_id)
    results_mgr.create_run(
        HumanEvalSampleConfig(seed=seed, per_cell=per_cell),
        input_source=str(cep_outputs),
    )

    # build_sample may raise InsufficientCellError (a ValueError); let it
    # propagate after marking the run failed so the metadata reflects reality.
    try:
        items = build_sample(pool, seed=seed, per_cell=per_cell)
    except ValueError:
        results_mgr.complete_run(success=False, error="insufficient pairs in a cell")
        raise

    _write_sample(results_mgr.outputs_dir / SAMPLE_FILENAME, items)
    manifest = SampleManifest(
        pipeline_id=pipeline_id,
        seed=seed,
        total_items=len(items),
        per_cell=per_cell,
        cell_counts=dict.fromkeys(all_cell_ids(), per_cell),
        population_by_cell=population,
        excluded_not_approved=excluded_not_approved,
        excluded_bloom=excluded_bloom,
        pool_sha256=pool_hash,
    )
    manifest.save(results_mgr.outputs_dir / MANIFEST_FILENAME)

    results_mgr.update_progress(len(items), 0, len(items))
    results_mgr.complete_run(success=True)

    logger.info(
        "Built human-eval sample: %d items across %d cells (pool=%d, excluded "
        "not-approved=%d, bloom=%d).",
        len(items),
        len(manifest.cell_counts),
        len(pool),
        excluded_not_approved,
        sum(excluded_bloom.values()),
    )
    return manifest


def _write_sample(path: Path, items: list[SampleItem]) -> None:
    """Write the sample as JSONL (one :class:`SampleItem` per line)."""
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json())
            fh.write("\n")
