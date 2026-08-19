"""Deterministic Bloom-stratified sampler for the human-comparison study (spec §5).

Pure selection logic with no I/O: given an in-frame pool of judge-approved pairs
carrying the annotation payload, build the 120-pair sample stratified as 4 Bloom
levels x 30 pairs. Frame construction (dropping non-approved and out-of-frame
Bloom pairs, reading the CEP payload) happens upstream in ``batch.py``.

Revised 2026-08-19: stratification was 4 Bloom x 2 emic bands x 15. The band came
from the emic judge's score, which put that run on the annotation critical path;
it was removed and the sample is now built from the CEP records alone. See
``docs/superpowers/specs/2026-08-19-bloom-only-sampling-design.md``.
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
# dropped during pool construction (see batch.py), never made into a fifth cell.
FRAME_BLOOM_LEVELS: tuple[str, ...] = ("remember", "understand", "analyze", "evaluate")
#: Pairs drawn per cell. 4 cells x 30 = the 120 pairs the annotators accepted.
PER_CELL: int = 30


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
            This is the stratification cell.
    """

    pair_id: str
    source_file_id: str
    pair_index: int = Field(..., ge=0)
    segment: str
    question: str
    answer: str
    bloom_level: str


def all_cell_ids() -> list[str]:
    """Return the 4 cell ids in frame order.

    A cell *is* a Bloom level. There is no composite key any more, so there is
    also no ``cell_id_for`` helper: use the entry's ``bloom_level`` directly.
    """
    return list(FRAME_BLOOM_LEVELS)


def population_by_cell(pool: Iterable[PoolEntry]) -> dict[str, int]:
    """Count in-frame pool entries per cell (all 4 cells present, zero-filled).

    Args:
        pool: In-frame entries. Every ``bloom_level`` must be in
            :data:`FRAME_BLOOM_LEVELS`; filtering is the caller's job and an
            out-of-frame level raises ``KeyError`` rather than inventing a cell.

    Returns:
        Count per Bloom level, every level present.
    """
    counts = dict.fromkeys(all_cell_ids(), 0)
    for entry in pool:
        counts[entry.bloom_level] += 1
    return counts


class InsufficientCellError(ValueError):
    """Raised when a stratification cell has fewer than ``per_cell`` pairs."""


def _selection_key(seed: int, pair_id: str) -> str:
    """Return the seeded selection key for a pair.

    A SHA-256 of ``"{seed}:{pair_id}"`` gives a uniform pseudo-random ordering
    keyed by the seed. Unlike :func:`random.sample`, this is stable across
    Python versions/platforms and depends only on the pair itself (not on other
    cells' sizes or iteration order), so reproducibility is airtight.
    """
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def build_sample(pool: list[PoolEntry], seed: int, *, per_cell: int = PER_CELL) -> list[SampleItem]:
    """Build the Bloom-stratified sample deterministically.

    Groups the pool into the 4 Bloom cells and deterministically draws
    ``per_cell`` from each by ordering the cell's entries on a seeded SHA-256 key
    (:func:`_selection_key`) and taking the first ``per_cell``. The ordering
    depends only on the seed and each pair's id, so the result is independent of
    input/file order AND of other cells' sizes, and is stable across Python
    versions: same seed + same pool always yields the same sample.

    Within a cell the draw is uniform, so the sample mirrors that cell's
    population. This is the accepted cost of dropping the emic band, which used
    to over-sample the ``duvidosa`` half to 50/50: the frame is the approved
    corpus, so easy pairs dominate and agreement is high partly by construction.

    Args:
        pool: In-frame approved pairs (Bloom in :data:`FRAME_BLOOM_LEVELS`).
            Frame filtering is the caller's responsibility.
        seed: Selection seed; recorded in the manifest for reproducibility.
        per_cell: Pairs to draw per cell (default :data:`PER_CELL` -> 120 total).

    Returns:
        ``4 * per_cell`` :class:`SampleItem` objects, grouped by cell in
        :func:`all_cell_ids` order with ``slot_id`` 0..per_cell-1 per cell
        (slot order is the deterministic selection-key rank).

    Raises:
        InsufficientCellError: If any cell has fewer than ``per_cell`` entries;
            the message names the cell, its available count, and remediation.
    """
    by_cell: dict[str, list[PoolEntry]] = {cid: [] for cid in all_cell_ids()}
    for entry in pool:
        by_cell[entry.bloom_level].append(entry)

    items: list[SampleItem] = []
    for cell_id in all_cell_ids():
        entries = by_cell[cell_id]
        if len(entries) < per_cell:
            raise InsufficientCellError(
                f"Bloom cell {cell_id!r} has only {len(entries)} approved pair(s) but "
                f"{per_cell} are required. Remediate by approving more pairs (larger "
                f"CEP/judge pool) or by lowering --per-cell; cells are never back-filled "
                f"from other cells, because that would silently unbalance the design."
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
                    slot_id=slot_id,
                )
            )
    return items
