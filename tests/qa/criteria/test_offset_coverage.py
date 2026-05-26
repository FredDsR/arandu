"""Tests for the deterministic ``offset_coverage`` criterion (spec §6.3)."""

from __future__ import annotations

from arandu.qa.criteria.offset_coverage import OffsetCoverageCriterion, offset_coverage
from arandu.shared.chunking.schemas import Chunk


def _chunk(file_id: str, start: int, end: int, chunker_id: str = "cep_4k") -> Chunk:
    return Chunk(
        chunk_id=f"{file_id}_{start}_{end}",
        source_file_id=file_id,
        chunker_id=chunker_id,
        start_char=start,
        end_char=end,
    )


class TestOffsetCoverageFunction:
    def test_full_overlap_returns_one(self) -> None:
        gold = _chunk("src_a", 0, 100)
        retrieved = [_chunk("src_a", 0, 100)]
        assert offset_coverage(gold, retrieved) == 1.0

    def test_half_overlap_returns_half(self) -> None:
        gold = _chunk("src_a", 0, 100)
        retrieved = [_chunk("src_a", 0, 50)]
        assert offset_coverage(gold, retrieved) == 0.5

    def test_no_overlap_returns_zero(self) -> None:
        gold = _chunk("src_a", 0, 50)
        retrieved = [_chunk("src_a", 100, 200)]
        assert offset_coverage(gold, retrieved) == 0.0

    def test_cross_file_overlap_excluded(self) -> None:
        # Same numeric range, different source_file_id → no overlap.
        # Without this guard, char-range arithmetic would silently
        # cross file boundaries since the offsets are file-relative.
        gold = _chunk("src_a", 0, 100)
        retrieved = [_chunk("src_b", 0, 100)]
        assert offset_coverage(gold, retrieved) == 0.0

    def test_union_across_multiple_retrieved(self) -> None:
        # Two retrieved chunks each covering half of the gold → 100%.
        gold = _chunk("src_a", 0, 100)
        retrieved = [_chunk("src_a", 0, 50), _chunk("src_a", 50, 100)]
        assert offset_coverage(gold, retrieved) == 1.0

    def test_empty_retrieved_returns_zero(self) -> None:
        gold = _chunk("src_a", 0, 100)
        assert offset_coverage(gold, []) == 0.0

    def test_zero_length_gold_safe(self) -> None:
        # Defensive: the Chunk schema forbids end_char <= start_char so
        # this can't be constructed via the model. Exercise the path by
        # bypassing validation via model_construct.
        gold = Chunk.model_construct(
            chunk_id="x",
            source_file_id="src_a",
            chunker_id="cep_4k",
            start_char=0,
            end_char=0,
        )
        # Without the empty-gold guard this would ZeroDivisionError.
        assert offset_coverage(gold, [_chunk("src_a", 0, 100)]) == 0.0


class TestOffsetCoverageCriterion:
    def test_evaluate_returns_score_in_rationale(self) -> None:
        c = OffsetCoverageCriterion()
        gold = _chunk("src_a", 0, 100)
        retrieved = [_chunk("src_a", 0, 50)]
        score = c.evaluate(gold_chunk=gold, retrieved_passages=retrieved)
        assert score.score == 0.5
        assert "offset_coverage=0.500" in score.rationale
        assert score.threshold == 0.5

    def test_evaluate_records_in_file_count_in_rationale(self) -> None:
        # The rationale should help auditors spot when most retrieved
        # passages are from the wrong source file (the cross-file guard
        # excludes them from the overlap calc).
        c = OffsetCoverageCriterion()
        gold = _chunk("src_a", 0, 100)
        retrieved = [
            _chunk("src_a", 0, 50),  # in-file
            _chunk("src_b", 0, 50),  # wrong file
            _chunk("src_b", 50, 100),  # wrong file
        ]
        score = c.evaluate(gold_chunk=gold, retrieved_passages=retrieved)
        assert "1/3 retrieved passages share" in score.rationale

    def test_missing_kwargs_recorded_as_error(self) -> None:
        # Base class catches the KeyError and stuffs it into CriterionScore.error.
        c = OffsetCoverageCriterion()
        score = c.evaluate()  # no kwargs
        assert score.score is None
        assert score.error is not None
