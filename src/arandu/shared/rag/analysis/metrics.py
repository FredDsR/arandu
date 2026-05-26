"""Per-arm metrics over judged AnswerRecords (spec §8.2 + §8.6).

Aggregates a per-arm set of judged records into the confusion-matrix
counts (TA/TC/FA/FC), the headline rates (hallucination,
over-cautiousness, abstention F1, KC), and Wilson 95% CIs for each
proportion.

Per-Bloom + per-question-type cross-cuts are computed by re-running
the same aggregator over filtered subsets.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arandu.shared.rag.analysis.classifier import classify_record
from arandu.shared.rag.analysis.wilson import wilson_ci

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arandu.shared.rag.schemas import AnswerRecord


# Criterion names persisted by the judge_answers stage.
_CORRECTNESS = "answer_correctness"
_FAITHFULNESS = "answer_faithfulness"
_PASSAGE_COVERAGE = "passage_coverage"


class ProportionMetric(BaseModel):
    """A binary proportion with Wilson 95% CI."""

    numerator: int = Field(..., ge=0)
    denominator: int = Field(..., ge=0)
    value: float | None
    ci_lower: float
    ci_upper: float


class MeanMetric(BaseModel):
    """A scalar mean over a population, with sample size."""

    mean: float | None
    n: int = Field(..., ge=0)


class ArmMetrics(BaseModel):
    """Per-arm metric block for one slice (joint / Bloom / type)."""

    arm: str
    slice_name: str
    confusion: dict[str, int]  # {"TA","TC","FA","FC","unknown"} → counts
    hallucination_rate: ProportionMetric
    over_cautiousness_rate: ProportionMetric
    abstention_precision: ProportionMetric
    abstention_recall: ProportionMetric
    abstention_f1: float | None
    answer_correctness: MeanMetric
    answer_faithfulness: MeanMetric
    knowledge_coverage: MeanMetric
    passage_coverage: MeanMetric


def aggregate_arm(
    arm: str,
    records: Iterable[AnswerRecord],
    *,
    slice_name: str = "joint",
) -> ArmMetrics:
    """Aggregate one arm's judged records into a :class:`ArmMetrics` block.

    Args:
        arm: Retriever id (``"bm25"``, ``"atlas_rag"``, …). Pure label;
            used as ``ArmMetrics.arm``.
        records: Judged :class:`AnswerRecord` instances for this arm.
            Records whose abstention judge errored or didn't run are
            counted as ``"unknown"`` in the confusion matrix and
            EXCLUDED from rate/mean denominators (so a single judge
            failure doesn't skew the headline numbers).
        slice_name: Slice label (``"joint"``, ``"bloom=remember"``,
            …). Echoed onto the result so tables can group by it.

    Returns:
        Populated :class:`ArmMetrics` block. All rate metrics carry
        Wilson 95% CIs; means use record-count denominators.
    """
    confusion: Counter[str] = Counter()
    correctness: list[float] = []
    faithfulness: list[float] = []
    kc_values: list[float] = []
    passage_cov: list[float] = []

    for record in records:
        label = classify_record(record)
        confusion[label] += 1
        if label == "unknown":
            continue
        if label == "TC":
            corr = _criterion_score(record, _CORRECTNESS)
            faith = _criterion_score(record, _FAITHFULNESS)
            if corr is not None:
                correctness.append(corr)
            if faith is not None:
                faithfulness.append(faith)
            if corr is not None and faith is not None:
                kc_values.append(corr * faith)
        cov = _criterion_score(record, _PASSAGE_COVERAGE)
        if cov is not None:
            passage_cov.append(cov)

    ta = confusion.get("TA", 0)
    tc = confusion.get("TC", 0)
    fa = confusion.get("FA", 0)
    fc = confusion.get("FC", 0)

    return ArmMetrics(
        arm=arm,
        slice_name=slice_name,
        confusion={
            "TA": ta,
            "TC": tc,
            "FA": fa,
            "FC": fc,
            "unknown": confusion.get("unknown", 0),
        },
        hallucination_rate=_proportion(fc, fc + ta),
        over_cautiousness_rate=_proportion(fa, fa + tc),
        abstention_precision=_proportion(ta, ta + fa),
        abstention_recall=_proportion(ta, ta + fc),
        abstention_f1=_f1(ta=ta, fa=fa, fc=fc),
        answer_correctness=_mean(correctness),
        answer_faithfulness=_mean(faithfulness),
        knowledge_coverage=_mean(kc_values),
        passage_coverage=_mean(passage_cov),
    )


def _proportion(numerator: int, denominator: int) -> ProportionMetric:
    """Build a :class:`ProportionMetric` with Wilson 95% CI.

    Empty denominator yields ``value=None`` and a ``(0.0, 0.0)`` band
    — the report layer renders these as "n/a" rather than misleading
    zero rates.
    """
    if denominator == 0:
        return ProportionMetric(
            numerator=numerator,
            denominator=0,
            value=None,
            ci_lower=0.0,
            ci_upper=0.0,
        )
    value = numerator / denominator
    lower, upper = wilson_ci(numerator, denominator)
    return ProportionMetric(
        numerator=numerator,
        denominator=denominator,
        value=value,
        ci_lower=lower,
        ci_upper=upper,
    )


def _mean(values: list[float]) -> MeanMetric:
    """Build a :class:`MeanMetric`; empty input → ``mean=None``."""
    if not values:
        return MeanMetric(mean=None, n=0)
    return MeanMetric(mean=sum(values) / len(values), n=len(values))


def _f1(*, ta: int, fa: int, fc: int) -> float | None:
    """Abstention F1 from TA/FA/FC counts; ``None`` when undefined."""
    precision_denom = ta + fa
    recall_denom = ta + fc
    if precision_denom == 0 or recall_denom == 0:
        return None
    precision = ta / precision_denom
    recall = ta / recall_denom
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _criterion_score(record: AnswerRecord, name: str) -> float | None:
    """Pull ``name``'s score from the record's validation, or ``None`` on miss."""
    if record.validation is None:
        return None
    for step in record.validation.stage_results.values():
        criterion_score = step.criterion_scores.get(name)
        if criterion_score is None:
            continue
        return criterion_score.score
    return None
