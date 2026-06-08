"""Deterministic stratified sampler for the human-comparison study (spec §5).

Pure selection logic with no I/O: given an in-frame pool of approved pairs
(each carrying its emic pre-pass ordinal score and annotation payload), build
the 80-pair sample stratified as 4 Bloom levels x 2 emic bands x 10 pairs.
Frame construction (dropping null-score and out-of-frame-Bloom pairs, joining
the CEP payload) happens upstream in ``batch.py``.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arandu.shared.human_eval.schemas import SampleItem

if TYPE_CHECKING:
    from collections.abc import Iterable

# The four Bloom levels the agreement study stratifies over (spec §5). The CEP
# generator can also emit ``apply`` / ``create``; those are out-of-frame and
# dropped during pool construction (see batch.py).
FRAME_BLOOM_LEVELS: tuple[str, ...] = ("remember", "understand", "analyze", "evaluate")
BANDS: tuple[str, ...] = ("duvidosa", "limpa")
PER_CELL: int = 10
DUBIOUS_MAX_SCORE: int = 3  # emic_score <= 3 -> duvidosa; >= 4 -> limpa


class PoolEntry(BaseModel):
    """One in-frame approved pair eligible for sampling.

    Attributes:
        pair_id: Stable key ``"{source_file_id}:{pair_index}"``.
        source_file_id: Source interview id.
        pair_index: Index into the source ``QARecordCEP.qa_pairs``.
        segment: Source transcript segment (annotation payload).
        question: The generated question.
        answer: The generated answer.
        bloom_level: Bloom level; must be one of :data:`FRAME_BLOOM_LEVELS`.
        emic_score: Ordinal emic score {1..5} (banding key; never None here).
    """

    pair_id: str
    source_file_id: str
    pair_index: int = Field(..., ge=0)
    segment: str
    question: str
    answer: str
    bloom_level: str
    emic_score: int = Field(..., ge=1, le=5)


def band_for(emic_score: int) -> str:
    """Return the emic band for an ordinal score (``duvidosa`` <=3, ``limpa`` >=4)."""
    return "duvidosa" if emic_score <= DUBIOUS_MAX_SCORE else "limpa"


def cell_id_for(bloom_level: str, band: str) -> str:
    """Return the ``"{bloom_level}:{band}"`` cell key."""
    return f"{bloom_level}:{band}"


def all_cell_ids() -> list[str]:
    """Return the 8 cell ids in a stable order (Bloom x band)."""
    return [cell_id_for(bloom, band) for bloom in FRAME_BLOOM_LEVELS for band in BANDS]


def population_by_cell(pool: Iterable[PoolEntry]) -> dict[str, int]:
    """Count in-frame pool entries per cell (all 8 cells present, zero-filled)."""
    counts = dict.fromkeys(all_cell_ids(), 0)
    for entry in pool:
        counts[cell_id_for(entry.bloom_level, band_for(entry.emic_score))] += 1
    return counts


class InsufficientCellError(ValueError):
    """Raised when a stratification cell has fewer than ``PER_CELL`` pairs."""


def build_sample(pool: list[PoolEntry], seed: int, *, per_cell: int = PER_CELL) -> list[SampleItem]:
    """Build the stratified sample deterministically.

    Bands each pool entry, groups into the 8 cells (4 Bloom x 2 bands), and
    draws exactly ``per_cell`` from each with a seeded RNG. The pool is sorted
    by ``pair_id`` first, so the result is independent of input/file order:
    same seed + same pool always yields the same sample. ``duvidosa`` is thus
    over-sampled to 50/50 against ``limpa`` regardless of population skew.

    Args:
        pool: In-frame approved pairs (Bloom in :data:`FRAME_BLOOM_LEVELS`,
            non-null score). Frame filtering is the caller's responsibility.
        seed: RNG seed; recorded in the manifest for reproducibility.
        per_cell: Pairs to draw per cell (default 10 -> 80 total).

    Returns:
        ``8 * per_cell`` :class:`SampleItem` objects, grouped by cell in
        :func:`all_cell_ids` order with ``slot_id`` 0..per_cell-1 per cell.

    Raises:
        InsufficientCellError: If any cell has fewer than ``per_cell`` entries;
            the message names the cell, its available count, and remediation.
    """
    by_cell: dict[str, list[PoolEntry]] = {cid: [] for cid in all_cell_ids()}
    for entry in pool:
        by_cell[cell_id_for(entry.bloom_level, band_for(entry.emic_score))].append(entry)

    rng = random.Random(seed)
    items: list[SampleItem] = []
    for cell_id in all_cell_ids():
        entries = sorted(by_cell[cell_id], key=lambda e: e.pair_id)
        if len(entries) < per_cell:
            raise InsufficientCellError(
                f"Cell {cell_id!r} has only {len(entries)} approved pair(s) but {per_cell} "
                f"are required. Remediate by approving more pairs (larger CEP/judge pool) "
                f"or revisiting the emic bands; cells are never back-filled from other cells."
            )
        chosen = rng.sample(entries, per_cell)
        for slot_id, entry in enumerate(chosen):
            items.append(
                SampleItem(
                    pair_id=entry.pair_id,
                    source_file_id=entry.source_file_id,
                    pair_index=entry.pair_index,
                    segment=entry.segment,
                    question=entry.question,
                    answer=entry.answer,
                    bloom_level=entry.bloom_level,
                    emic_prepass_score=entry.emic_score,
                    cell_id=cell_id,
                    slot_id=slot_id,
                )
            )
    return items
