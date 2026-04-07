"""Tests for SegmentQualityCriterion."""

from __future__ import annotations

from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.schemas import TranscriptionSegment
from arandu.transcription.criteria.segment_quality import SegmentQualityCriterion


class TestSegmentQualityCriterion:
    """Tests for segment pattern analysis criterion."""

    def test_implements_judge_criterion_protocol(self) -> None:
        """Test that SegmentQualityCriterion satisfies JudgeCriterion protocol."""
        from arandu.shared.judge.criterion import JudgeCriterion

        criterion = SegmentQualityCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_name_and_threshold(self) -> None:
        """Test default name and threshold."""
        criterion = SegmentQualityCriterion()
        assert criterion.name == "segment_quality"
        assert criterion.threshold == 0.4

    def test_empty_segments_returns_neutral(self) -> None:
        """Test with no segments returns neutral score."""
        criterion = SegmentQualityCriterion()
        result = criterion.evaluate(segments=[])

        assert isinstance(result, CriterionScore)
        assert result.score == 0.5
        assert result.passed is True

    def test_normal_segments(self) -> None:
        """Test with normal, varied segment durations."""
        criterion = SegmentQualityCriterion()
        segments = [
            TranscriptionSegment(text="Hello", start=0.0, end=1.5),
            TranscriptionSegment(text="how are you", start=1.5, end=3.2),
            TranscriptionSegment(text="I'm fine", start=3.2, end=5.1),
            TranscriptionSegment(text="thanks", start=5.1, end=6.0),
        ]
        result = criterion.evaluate(segments=segments)

        assert result.score == 1.0
        assert result.passed is True

    def test_suspicious_uniform_intervals(self) -> None:
        """Test detection of suspicious uniform 1-second intervals."""
        criterion = SegmentQualityCriterion()
        segments = [
            TranscriptionSegment(text=f"Segment {i}", start=float(i), end=float(i) + 0.5)
            for i in range(7)
        ]
        result = criterion.evaluate(segments=segments)

        assert result.score is not None
        assert result.score < 1.0
        assert "suspicious_uniform_intervals" in result.rationale

    def test_empty_text_segments(self) -> None:
        """Test detection of empty segments."""
        criterion = SegmentQualityCriterion()
        segments = [
            TranscriptionSegment(text="", start=0.0, end=1.0),
            TranscriptionSegment(text="", start=1.0, end=2.0),
            TranscriptionSegment(text="Some text", start=2.0, end=3.0),
            TranscriptionSegment(text="", start=3.0, end=4.0),
        ]
        result = criterion.evaluate(segments=segments)

        assert result.score is not None
        assert result.score < 1.0
        assert "high_empty_segments" in result.rationale

    def test_custom_suspicious_uniform_intervals(self) -> None:
        """Test with custom suspicious_uniform_intervals threshold."""
        criterion = SegmentQualityCriterion(suspicious_uniform_intervals=3)
        # 4 consecutive uniform intervals should trigger with threshold=3
        segments = [
            TranscriptionSegment(text=f"Seg {i}", start=float(i), end=float(i) + 0.5)
            for i in range(5)
        ]
        result = criterion.evaluate(segments=segments)
        assert "suspicious_uniform_intervals" in result.rationale

    def test_custom_max_empty_segment_ratio(self) -> None:
        """Test with custom max_empty_segment_ratio."""
        criterion = SegmentQualityCriterion(max_empty_segment_ratio=0.1)
        segments = [
            TranscriptionSegment(text="", start=0.0, end=1.0),
            TranscriptionSegment(text="Text", start=2.0, end=3.0),
            TranscriptionSegment(text="More text", start=4.0, end=5.0),
            TranscriptionSegment(text="Even more", start=6.0, end=7.0),
            TranscriptionSegment(text="Still more", start=8.0, end=9.0),
        ]
        # 1/5 = 0.2 > 0.1 threshold
        result = criterion.evaluate(segments=segments)
        assert "high_empty_segments" in result.rationale

    def test_custom_uniform_interval_tolerance(self) -> None:
        """Test with custom uniform_interval_tolerance."""
        criterion = SegmentQualityCriterion(uniform_interval_tolerance=0.3)
        # Intervals of 1.2 seconds are within 1.0 +/- 0.3 tolerance
        segments = [
            TranscriptionSegment(text=f"Seg {i}", start=i * 1.2, end=i * 1.2 + 0.5)
            for i in range(7)
        ]
        result = criterion.evaluate(segments=segments)
        assert "suspicious_uniform_intervals" in result.rationale

    def test_error_returns_criterion_score_with_error(self) -> None:
        """Test that exceptions are caught and returned as error CriterionScore."""
        criterion = SegmentQualityCriterion()
        result = criterion.evaluate(segments="not_a_list")

        assert result.score is None
        assert result.error is not None
        assert result.passed is False
