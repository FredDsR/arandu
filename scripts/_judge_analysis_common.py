"""Shared helpers for the judge-qa threshold / cohort analysis scripts.

These tools inspect a completed run's judge-qa verdicts (``cep/outputs``) and the
RAG judged answers (``judge_answers/outputs``) without re-running any stage. The
judge-qa criteria are 5-point Likert rubrics mapped to ``[0, 1]`` (anchors in
``ANCHORS``), so ``pass`` at a threshold ``t`` is ``score >= t`` for every
evaluated criterion. A pair carrying fewer criteria (remember pairs skip
``informativeness`` + ``self_containedness`` by design) is judged on whatever it
carries.

Used by ``analyze_judge_thresholds``, ``analyze_qa_cohort_rag_outcome`` and
``plot_judge_score_distributions``. Run those as modules from the repo root so
this import resolves, e.g. ``uv run python -m scripts.analyze_judge_thresholds``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.qa.schemas import QARecordCEP

if TYPE_CHECKING:
    from pathlib import Path

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
    for path in sorted(cep_dir.glob("*.json")):
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            continue
        for idx, pair in enumerate(record.qa_pairs):
            seg = pair.chunk_id or "none"
            pid = f"{record.source_file_id}:{seg}:{idx}"
            scores: dict[str, float | None] = {}
            if pair.validation is not None:
                step = pair.validation.stage_results.get(CEP_STAGE)
                if step is not None:
                    for name, cs in step.criterion_scores.items():
                        scores[name] = None if cs.error is not None else cs.score
            out[pid] = scores
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
