"""Deterministic stratified sampler for the human-comparison study (spec §5).

Pure selection logic with no I/O: given an in-frame pool of approved pairs
(each carrying its emic pre-pass ordinal score and annotation payload), build
the 80-pair sample stratified as 4 Bloom levels x 2 emic bands x 10 pairs.
Frame construction (dropping null-score and out-of-frame-Bloom pairs, joining
the CEP payload) happens upstream in ``batch.py``.
"""

from __future__ import annotations

import hashlib
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


def _selection_key(seed: int, pair_id: str) -> str:
    """Return the seeded selection key for a pair.

    A SHA-256 of ``"{seed}:{pair_id}"`` gives a uniform pseudo-random ordering
    keyed by the seed. Unlike :func:`random.sample`, this is stable across
    Python versions/platforms and depends only on the pair itself (not on other
    cells' sizes or iteration order), so reproducibility is airtight.
    """
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def build_sample(pool: list[PoolEntry], seed: int, *, per_cell: int = PER_CELL) -> list[SampleItem]:
    """Build the stratified sample deterministically.

    Bands each pool entry, groups into the 8 cells (4 Bloom x 2 bands), and
    deterministically draws ``per_cell`` from each by ordering the cell's
    entries on a seeded SHA-256 key (:func:`_selection_key`) and taking the
    first ``per_cell``. The ordering depends only on the seed and each pair's
    id, so the result is independent of input/file order AND of other cells'
    sizes, and is stable across Python versions: same seed + same pool always
    yields the same sample. ``duvidosa`` is over-sampled to 50/50 against
    ``limpa`` regardless of population skew.

    Args:
        pool: In-frame approved pairs (Bloom in :data:`FRAME_BLOOM_LEVELS`,
            non-null score). Frame filtering is the caller's responsibility.
        seed: Selection seed; recorded in the manifest for reproducibility.
        per_cell: Pairs to draw per cell (default 10 -> 80 total).

    Returns:
        ``8 * per_cell`` :class:`SampleItem` objects, grouped by cell in
        :func:`all_cell_ids` order with ``slot_id`` 0..per_cell-1 per cell
        (slot order is the deterministic selection-key rank).

    Raises:
        InsufficientCellError: If any cell has fewer than ``per_cell`` entries;
            the message names the cell, its available count, and remediation.
    """
    by_cell: dict[str, list[PoolEntry]] = {cid: [] for cid in all_cell_ids()}
    for entry in pool:
        by_cell[cell_id_for(entry.bloom_level, band_for(entry.emic_score))].append(entry)

    items: list[SampleItem] = []
    for cell_id in all_cell_ids():
        entries = by_cell[cell_id]
        if len(entries) < per_cell:
            raise InsufficientCellError(
                f"Cell {cell_id!r} has only {len(entries)} approved pair(s) but {per_cell} "
                f"are required. Remediate by approving more pairs (larger CEP/judge pool) "
                f"or revisiting the emic bands; cells are never back-filled from other cells."
            )
        # pair_id is the tiebreaker so the order is total even on a (vanishingly
        # unlikely) key collision.
        ordered = sorted(entries, key=lambda e: (_selection_key(seed, e.pair_id), e.pair_id))
        for slot_id, entry in enumerate(ordered[:per_cell]):
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
