"""Shared helpers for the judge-qa threshold / cohort analysis scripts.

These tools inspect a completed run's judge-qa verdicts (``cep/outputs``) and the
RAG judged answers (``judge_answers/outputs``) without re-running any stage. The
judge-qa criteria are 5-point Likert rubrics mapped to ``[0, 1]`` (anchors in
``ANCHORS``), so ``pass`` at a threshold ``t`` is ``score >= t`` for every
evaluated criterion. A pair carrying fewer criteria (remember pairs skip
``informativeness`` + ``self_containedness`` by design) is judged on whatever it
carries.

Used by ``analyze_judge_thresholds``, ``analyze_qa_cohort_rag_outcome``,
``analyze_cep_gate_flip`` and ``plot_judge_score_distributions``. Run those as
modules from the repo root so this import resolves, e.g.
``uv run python -m analysis.analyze_judge_thresholds``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.qa.schemas import iter_cep_pairs
from arandu.shared.rag.analysis.metrics import aggregate_arm

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from arandu.shared.rag.schemas import AnswerRecord

CRITERIA = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
ANCHORS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLD = 0.625
CEP_STAGE = "cep_validation"

# qa_pair_id -> {criterion_name: score or None}; None means the criterion errored
# or produced no score (treated as a fail at any threshold).
PairScores = dict[str, dict[str, float | None]]


def load_pair_scores(cep_dir: Path) -> PairScores:
    """Read per-criterion judge-qa scores for every CEP pair.

    The ``qa_pair_id`` is derived exactly as the retrieve loader and analysis
    cross-cut map derive it (``"<file_id>:<chunk_id or 'none'>:<idx>"``), so ids
    join cleanly onto the ``judge_answers`` records.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``.

    Returns:
        Mapping ``qa_pair_id -> {criterion: score or None}`` over the criteria
        the judge actually evaluated for that pair.
    """
    out: PairScores = {}
    for ref in iter_cep_pairs(cep_dir):
        scores: dict[str, float | None] = {}
        if ref.pair.validation is not None:
            step = ref.pair.validation.stage_results.get(CEP_STAGE)
            if step is not None:
                for name, cs in step.criterion_scores.items():
                    scores[name] = None if cs.error is not None else cs.score
        out[ref.qa_pair_id] = scores
    return out


def approved_ids(scores: PairScores, threshold: float) -> set[str]:
    """qa_pair_ids whose every evaluated criterion scores >= ``threshold``.

    A pair with an unscored/errored criterion (score None) or no criteria at all
    is excluded, mirroring ``CriterionScore.passed`` (error/None never passes).
    """
    return {
        pid
        for pid, crit in scores.items()
        if crit and all(s is not None and s >= threshold for s in crit.values())
    }


def snap_to_anchor(score: float) -> float:
    """Return the Likert anchor (0/.25/.5/.75/1) nearest to ``score``."""
    return min(ANCHORS, key=lambda a: abs(a - score))


def answered_share(confusion: Mapping[str, int]) -> float | None:
    """Share of answerable records an arm committed to, ``TC / (TC + FA)``.

    ``TA`` and ``FC`` only occur on non-answerable records, so they are left
    out: counting ``TA`` as an answer would credit correct refusals.

    Args:
        confusion: Per-label counts from ``ArmMetrics.confusion``.

    Returns:
        The share in ``[0, 1]``, or ``None`` when there is no answerable record.
    """
    committed = confusion.get("TC", 0)
    total = committed + confusion.get("FA", 0)
    return committed / total if total else None


def format_score(value: float | None) -> str:
    """Format a metric for a fixed-width table, ``n/a`` when undefined."""
    return "n/a" if value is None else f"{value:.3f}"


def outcome_table(name: str, ids: set[str], by_arm: Mapping[str, list[AnswerRecord]]) -> list[str]:
    """Render the per-arm RAG outcome restricted to one cohort of pairs.

    Args:
        name: Cohort label, used as the table heading and the metrics slice.
        ids: The cohort's ``qa_pair_id`` set.
        by_arm: Judged answer records grouped by arm.

    Returns:
        The table lines. Arms with no answerable record in the cohort are left
        out.
    """
    lines = [
        f"\n### {name}  (|answerable| = {len(ids)})",
        f"{'arm':<20} {'n':>6} {'KC':>7} {'Correct':>8} {'PassCov':>8} {'answered%':>10}",
    ]
    for arm in sorted(by_arm):
        records = [r for r in by_arm[arm] if r.is_answerable and r.qa_pair_id in ids]
        if not records:
            continue
        metrics = aggregate_arm(arm, records, slice_name=name)
        share = answered_share(metrics.confusion)
        pct = "n/a" if share is None else f"{100 * share:.0f}%"
        lines.append(
            f"{arm:<20} {len(records):>6} {format_score(metrics.knowledge_coverage.mean):>7} "
            f"{format_score(metrics.answer_correctness.mean):>8} "
            f"{format_score(metrics.passage_coverage.mean):>8} {pct:>10}"
        )
    return lines
