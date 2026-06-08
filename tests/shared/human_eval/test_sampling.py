"""Tests for the pure stratified sampler (spec §5)."""

from __future__ import annotations

import pytest

from arandu.shared.human_eval.sampling import (
    BANDS,
    FRAME_BLOOM_LEVELS,
    InsufficientCellError,
    PoolEntry,
    band_for,
    build_sample,
)

DUBIOUS_SCORE = 2
CLEAN_SCORE = 5


def _entry(idx: int, bloom: str, score: int) -> PoolEntry:
    return PoolEntry(
        pair_id=f"src:{idx}",
        source_file_id="src",
        pair_index=idx,
        segment=f"segment {idx}",
        question=f"q {idx}",
        answer=f"a {idx}",
        bloom_level=bloom,
        emic_score=score,
    )


def _pool(per_cell_available: int) -> list[PoolEntry]:
    """Build a pool with ``per_cell_available`` entries in each of the 8 cells."""
    pool: list[PoolEntry] = []
    idx = 0
    for bloom in FRAME_BLOOM_LEVELS:
        for score in (DUBIOUS_SCORE, CLEAN_SCORE):
            for _ in range(per_cell_available):
                pool.append(_entry(idx, bloom, score))
                idx += 1
    return pool


class TestBandFor:
    @pytest.mark.parametrize(
        ("score", "band"),
        [(1, "duvidosa"), (3, "duvidosa"), (4, "limpa"), (5, "limpa")],
    )
    def test_boundary(self, score: int, band: str) -> None:
        assert band_for(score) == band


class TestBuildSample:
    def test_stratification_exact_ten_per_cell(self) -> None:
        sample = build_sample(_pool(15), seed=42)
        assert len(sample) == 80  # 8 cells x 10
        counts: dict[str, int] = {}
        for item in sample:
            counts[item.cell_id] = counts.get(item.cell_id, 0) + 1
        assert len(counts) == 8
        assert all(c == 10 for c in counts.values())
        # 50/50 band balance regardless of (here equal) population
        duvidosa = sum(1 for i in sample if i.cell_id.endswith("duvidosa"))
        limpa = sum(1 for i in sample if i.cell_id.endswith("limpa"))
        assert duvidosa == 40
        assert limpa == 40

    def test_slot_ids_zero_to_nine_per_cell(self) -> None:
        sample = build_sample(_pool(15), seed=7)
        by_cell: dict[str, list[int]] = {}
        for item in sample:
            by_cell.setdefault(item.cell_id, []).append(item.slot_id)
        for slots in by_cell.values():
            assert sorted(slots) == list(range(10))

    def test_oversamples_dubious_against_skewed_population(self) -> None:
        # Population skewed 20 limpa / 11 duvidosa per bloom, yet the sample is 50/50.
        pool: list[PoolEntry] = []
        idx = 0
        for bloom in FRAME_BLOOM_LEVELS:
            for _ in range(11):
                pool.append(_entry(idx, bloom, DUBIOUS_SCORE))
                idx += 1
            for _ in range(20):
                pool.append(_entry(idx, bloom, CLEAN_SCORE))
                idx += 1
        sample = build_sample(pool, seed=1)
        assert sum(1 for i in sample if i.cell_id.endswith("duvidosa")) == 40
        assert sum(1 for i in sample if i.cell_id.endswith("limpa")) == 40

    def test_reproducible_same_seed_same_pool(self) -> None:
        pool = _pool(15)
        a = build_sample(pool, seed=99)
        b = build_sample(pool, seed=99)
        assert [(i.pair_id, i.cell_id, i.slot_id) for i in a] == [
            (i.pair_id, i.cell_id, i.slot_id) for i in b
        ]

    def test_order_independent_reproducibility(self) -> None:
        pool = _pool(15)
        shuffled = list(reversed(pool))
        a = build_sample(pool, seed=5)
        b = build_sample(shuffled, seed=5)
        # Selection is sorted by pair_id first, so input order cannot change it.
        assert {i.pair_id for i in a} == {i.pair_id for i in b}

    def test_different_seed_selects_different_pairs(self) -> None:
        pool = _pool(30)  # plenty of slack so two seeds almost surely differ
        a = {i.pair_id for i in build_sample(pool, seed=1)}
        b = {i.pair_id for i in build_sample(pool, seed=2)}
        assert a != b

    def test_insufficient_cell_raises_named_error(self) -> None:
        pool = _pool(15)
        # Drop one cell (analyze:limpa) down to 9 available.
        pool = [
            p
            for p in pool
            if not (p.bloom_level == "analyze" and band_for(p.emic_score) == "limpa")
        ]
        pool += [_entry(900 + k, "analyze", CLEAN_SCORE) for k in range(9)]
        with pytest.raises(InsufficientCellError, match="analyze:limpa"):
            build_sample(pool, seed=1)

    def test_bands_constant_is_two(self) -> None:
        assert BANDS == ("duvidosa", "limpa")
