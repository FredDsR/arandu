"""Batch orchestrator for ``arandu build-human-eval-sample`` (spec §5).

Builds the in-frame pool from the emic pre-pass outputs (dropping null-score
and out-of-frame-Bloom pairs), joins each pair's CEP payload (segment +
question + answer), runs the deterministic stratified sampler, and persists the
80-pair sample + a provenance manifest under
``results/<id>/human_eval/outputs/``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from arandu.qa.schemas import QARecordCEP
from arandu.shared.config import ResultsConfig
from arandu.shared.emic.schemas import EmicSourceScores
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
    ids/scores -- letting an auditor detect a CEP regeneration under the same
    pair ids.
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
        pipeline_id: Run identifier. Both the ``emic_prepass`` and ``cep``
            stages must be populated.
        seed: RNG seed for the deterministic selection (recorded in the run
            metadata and the manifest).
        base_dir: Override the project ``results/`` root.
        per_cell: Pairs to draw per cell (default 10 -> 80 total).

    Returns:
        The :class:`SampleManifest` describing the build.

    Raises:
        FileNotFoundError: If the emic_prepass or cep stage outputs are absent.
        ValueError: If a stratification cell has fewer than ``per_cell`` pairs,
            or a referenced CEP pair cannot be resolved.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    emic_outputs = base / pipeline_id / PipelineType.EMIC_PREPASS.value / "outputs"
    cep_outputs = base / pipeline_id / PipelineType.CEP.value / "outputs"
    if not emic_outputs.exists():
        raise FileNotFoundError(
            f"Emic pre-pass outputs not found for pipeline_id {pipeline_id!r}: {emic_outputs}. "
            f"Run `arandu emic-prepass --id {pipeline_id}` first."
        )
    if not cep_outputs.exists():
        raise FileNotFoundError(
            f"CEP outputs not found for pipeline_id {pipeline_id!r}: {cep_outputs}. "
            f"The sample payload (segment/question/answer) is joined from the CEP records."
        )

    pool: list[PoolEntry] = []
    seen_pair_ids: set[str] = set()
    excluded_none = 0
    excluded_bloom: dict[str, int] = {}
    for emic_path in sorted(emic_outputs.glob("*_cep_qa.json")):
        scores = EmicSourceScores.load(emic_path)
        cep_path = cep_outputs / emic_path.name
        if not cep_path.exists():
            raise ValueError(
                f"Emic scores reference {emic_path.name} but the matching CEP record "
                f"{cep_path} is missing; cannot build the annotation payload."
            )
        record = QARecordCEP.load(cep_path)
        for score in scores.scores:
            if score.emic_score is None:
                excluded_none += 1
                continue
            if score.bloom_level not in FRAME_BLOOM_LEVELS:
                excluded_bloom[score.bloom_level] = excluded_bloom.get(score.bloom_level, 0) + 1
                continue
            if score.pair_index >= len(record.qa_pairs):
                raise ValueError(
                    f"pair_index {score.pair_index} out of range for {cep_path.name} "
                    f"({len(record.qa_pairs)} pairs); emic/cep stages are out of sync."
                )
            pair_id = f"{scores.source_file_id}:{score.pair_index}"
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Duplicate pair_id {pair_id!r} while pooling {emic_path.name}; the "
                    f"emic_prepass outputs likely contain a stale or duplicate file for this "
                    f"source. Clean results/{pipeline_id}/emic_prepass/outputs/ and re-run."
                )
            seen_pair_ids.add(pair_id)
            pair = record.qa_pairs[score.pair_index]
            pool.append(
                PoolEntry(
                    pair_id=pair_id,
                    source_file_id=scores.source_file_id,
                    pair_index=score.pair_index,
                    segment=pair.context,
                    question=pair.question,
                    answer=pair.answer,
                    bloom_level=score.bloom_level,
                    emic_score=score.emic_score,
                )
            )

    if not pool:
        raise ValueError(
            f"No in-frame approved pairs found for {pipeline_id!r} "
            f"({excluded_none} null-score, {sum(excluded_bloom.values())} out-of-frame-Bloom "
            f"excluded). Check that `arandu judge-qa` + `arandu emic-prepass` ran and produced "
            f"scored, in-frame ({', '.join(FRAME_BLOOM_LEVELS)}) pairs."
        )

    population = population_by_cell(pool)
    pool_hash = _pool_sha256(pool)

    results_mgr = ResultsManager(base, PipelineType.HUMAN_EVAL, pipeline_id=pipeline_id)
    results_mgr.create_run(
        HumanEvalSampleConfig(seed=seed, per_cell=per_cell),
        input_source=str(emic_outputs),
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
        excluded_none_score=excluded_none,
        excluded_bloom=excluded_bloom,
        pool_sha256=pool_hash,
    )
    manifest.save(results_mgr.outputs_dir / MANIFEST_FILENAME)

    results_mgr.update_progress(len(items), 0, len(items))
    results_mgr.complete_run(success=True)

    logger.info(
        "Built human-eval sample: %d items across %d cells (pool=%d, excluded none=%d, bloom=%d).",
        len(items),
        len(manifest.cell_counts),
        len(pool),
        excluded_none,
        sum(excluded_bloom.values()),
    )
    return manifest


def _write_sample(path: Path, items: list[SampleItem]) -> None:
    """Write the sample as JSONL (one :class:`SampleItem` per line)."""
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json())
            fh.write("\n")
