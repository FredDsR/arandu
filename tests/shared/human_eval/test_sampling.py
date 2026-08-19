"""Tests for the pure Bloom-stratified sampler (spec §5, revised 2026-08-19)."""

from __future__ import annotations

import pytest

from arandu.shared.human_eval.sampling import (
    FRAME_BLOOM_LEVELS,
    PER_CELL,
    InsufficientCellError,
    PoolEntry,
    all_cell_ids,
    build_sample,
    population_by_cell,
)


def _entry(idx: int, bloom: str) -> PoolEntry:
    return PoolEntry(
        pair_id=f"src:{idx}",
        source_file_id="src",
        pair_index=idx,
        segment=f"segment {idx}",
        question=f"q {idx}",
        answer=f"a {idx}",
        bloom_level=bloom,
    )


def _pool(per_bloom: int) -> list[PoolEntry]:
    """Build a pool with ``per_bloom`` entries in each of the 4 Bloom cells."""
    pool: list[PoolEntry] = []
    idx = 0
    for bloom in FRAME_BLOOM_LEVELS:
        for _ in range(per_bloom):
            pool.append(_entry(idx, bloom))
            idx += 1
    return pool


class TestCells:
    def test_there_are_exactly_four_cells_and_they_are_the_bloom_levels(self) -> None:
        assert all_cell_ids() == list(FRAME_BLOOM_LEVELS)
        assert len(all_cell_ids()) == 4

    def test_default_per_cell_yields_the_agreed_120(self) -> None:
        """120 is the load the three annotators accepted; the default must produce it."""
        assert PER_CELL * len(FRAME_BLOOM_LEVELS) == 120

    def test_population_is_counted_per_bloom_and_zero_filled(self) -> None:
        pool = [_entry(0, "remember"), _entry(1, "remember"), _entry(2, "analyze")]
        assert population_by_cell(pool) == {
            "remember": 2,
            "understand": 0,
            "analyze": 1,
            "evaluate": 0,
        }


class TestBuildSample:
    def test_stratification_is_balanced_across_the_four_cells(self) -> None:
        sample = build_sample(_pool(40), seed=42, per_cell=30)
        assert len(sample) == 120
        counts: dict[str, int] = {}
        for item in sample:
            counts[item.bloom_level] = counts.get(item.bloom_level, 0) + 1
        assert counts == dict.fromkeys(FRAME_BLOOM_LEVELS, 30)

    def test_slot_ids_cover_the_range_once_per_cell(self) -> None:
        sample = build_sample(_pool(10), seed=7, per_cell=5)
        by_cell: dict[str, list[int]] = {}
        for item in sample:
            by_cell.setdefault(item.bloom_level, []).append(item.slot_id)
        assert len(by_cell) == 4
        for slots in by_cell.values():
            assert sorted(slots) == list(range(5))

    def test_sample_mirrors_the_population_within_a_cell(self) -> None:
        """No band means no oversampling: a cell is drawn from its own pool only."""
        pool = _pool(8)
        sample = build_sample(pool, seed=3, per_cell=8)
        assert len(sample) == 32
        assert {i.pair_id for i in sample} == {e.pair_id for e in pool}

    def test_reproducible_same_seed_same_pool(self) -> None:
        pool = _pool(10)
        a = build_sample(pool, seed=99, per_cell=5)
        b = build_sample(pool, seed=99, per_cell=5)
        assert [(i.pair_id, i.bloom_level, i.slot_id) for i in a] == [
            (i.pair_id, i.bloom_level, i.slot_id) for i in b
        ]

    def test_order_independent_reproducibility(self) -> None:
        pool = _pool(10)
        a = build_sample(pool, seed=5, per_cell=5)
        b = build_sample(list(reversed(pool)), seed=5, per_cell=5)
        assert {i.pair_id for i in a} == {i.pair_id for i in b}

    def test_different_seed_selects_different_pairs(self) -> None:
        pool = _pool(30)
        a = {i.pair_id for i in build_sample(pool, seed=1, per_cell=5)}
        b = {i.pair_id for i in build_sample(pool, seed=2, per_cell=5)}
        assert a != b

    def test_cell_selection_independent_of_other_cells(self) -> None:
        base = _pool(10)
        extra = [_entry(1000 + k, "remember") for k in range(20)]
        a = {
            i.pair_id for i in build_sample(base, seed=3, per_cell=5) if i.bloom_level == "evaluate"
        }
        b = {
            i.pair_id
            for i in build_sample([*base, *extra], seed=3, per_cell=5)
            if i.bloom_level == "evaluate"
        }
        assert a == b

    def test_insufficient_cell_raises_naming_the_bloom_level(self) -> None:
        pool = [e for e in _pool(10) if e.bloom_level != "analyze"]
        pool += [_entry(900 + k, "analyze") for k in range(4)]
        with pytest.raises(InsufficientCellError, match="analyze"):
            build_sample(pool, seed=1, per_cell=5)

    def test_insufficient_cell_message_names_the_counts(self) -> None:
        pool = [e for e in _pool(10) if e.bloom_level != "evaluate"]
        pool += [_entry(900 + k, "evaluate") for k in range(4)]
        with pytest.raises(InsufficientCellError) as excinfo:
            build_sample(pool, seed=1, per_cell=5)
        message = str(excinfo.value)
        assert "4" in message
        assert "5" in message

    def test_the_sample_item_carries_no_emic_score(self) -> None:
        """The regression that keeps the emic judge off the critical path."""
        sample = build_sample(_pool(5), seed=1, per_cell=5)
        assert "emic_score" not in sample[0].model_dump()
        assert "cell_id" not in sample[0].model_dump()
