"""Tests for per-arm metric aggregation."""

from __future__ import annotations

from arandu.shared.rag.analysis.metrics import aggregate_arm

from .conftest import make_answer


class TestAggregateArm:
    def test_all_tc_records(self) -> None:
        records = [
            make_answer(qa_pair_id=f"src:c:{i}", is_answerable=True, abstained=False)
            for i in range(4)
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.confusion["TC"] == 4
        assert metrics.confusion["FA"] == 0
        assert metrics.knowledge_coverage.n == 4
        assert metrics.knowledge_coverage.mean is not None
        assert 0.0 < metrics.knowledge_coverage.mean <= 1.0

    def test_hallucination_rate_uses_fc_over_fc_plus_ta(self) -> None:
        # 2 FC + 1 TA → hallucination = 2/3
        records = [
            make_answer(
                qa_pair_id="r:c:0",
                is_answerable=False,
                abstained=False,
                abstention_score=0.1,
            ),
            make_answer(
                qa_pair_id="r:c:1",
                is_answerable=False,
                abstained=False,
                abstention_score=0.1,
            ),
            make_answer(
                qa_pair_id="r:c:2",
                is_answerable=False,
                abstained=True,
                abstention_score=0.9,
            ),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.confusion["FC"] == 2
        assert metrics.confusion["TA"] == 1
        assert metrics.hallucination_rate.value is not None
        assert abs(metrics.hallucination_rate.value - 2 / 3) < 1e-9

    def test_abstention_f1_uses_ta_fa_fc(self) -> None:
        # TA=2, FA=1, FC=1 → precision=2/3, recall=2/3, F1=2/3
        records = [
            make_answer(  # TA
                qa_pair_id="r:c:0",
                is_answerable=False,
                abstained=True,
                abstention_score=0.9,
            ),
            make_answer(  # TA
                qa_pair_id="r:c:1",
                is_answerable=False,
                abstained=True,
                abstention_score=0.9,
            ),
            make_answer(  # FA
                qa_pair_id="r:c:2",
                is_answerable=True,
                abstained=True,
                abstention_score=0.9,
            ),
            make_answer(  # FC
                qa_pair_id="r:c:3",
                is_answerable=False,
                abstained=False,
                abstention_score=0.1,
            ),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.abstention_f1 is not None
        assert abs(metrics.abstention_f1 - 2 / 3) < 1e-9
        assert metrics.abstention_precision.value == 2 / 3
        assert metrics.abstention_recall.value == 2 / 3

    def test_unknown_skipped_from_means(self) -> None:
        records = [
            make_answer(qa_pair_id="r:c:0", abstention_score=None),  # unknown
            make_answer(
                qa_pair_id="r:c:1",
                is_answerable=True,
                abstained=False,
                abstention_score=0.1,
            ),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.confusion["unknown"] == 1
        assert metrics.confusion["TC"] == 1
        assert metrics.knowledge_coverage.n == 1  # only the TC counts

    def test_empty_denominator_yields_none_value(self) -> None:
        # All TC → hallucination denominator (FC+TA) = 0
        records = [
            make_answer(qa_pair_id="r:c:0", is_answerable=True, abstained=False),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.hallucination_rate.value is None
        assert metrics.hallucination_rate.denominator == 0

    def test_kc_is_correctness_times_faithfulness(self) -> None:
        records = [
            make_answer(
                qa_pair_id="r:c:0",
                is_answerable=True,
                abstained=False,
                correctness_score=0.8,
                faithfulness_score=0.5,
            ),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.knowledge_coverage.mean is not None
        assert abs(metrics.knowledge_coverage.mean - 0.4) < 1e-9

    def test_source_recovery_mean_excludes_none(self) -> None:
        # Two prose records carry source_recovery; one payload/null-style
        # record carries None and must be excluded from the mean.
        records = [
            make_answer(qa_pair_id="r:c:0", source_recovery_score=0.6),
            make_answer(qa_pair_id="r:c:1", source_recovery_score=0.8),
            make_answer(qa_pair_id="r:c:2", source_recovery_score=None),
        ]
        metrics = aggregate_arm("bm25", records)
        assert metrics.source_recovery.n == 2
        assert metrics.source_recovery.mean is not None
        assert abs(metrics.source_recovery.mean - 0.7) < 1e-9

    def test_source_recovery_all_none_yields_none(self) -> None:
        records = [make_answer(qa_pair_id="r:c:0", source_recovery_score=None)]
        metrics = aggregate_arm("bm25", records)
        assert metrics.source_recovery.mean is None
        assert metrics.source_recovery.n == 0
