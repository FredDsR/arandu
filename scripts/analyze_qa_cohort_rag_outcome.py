"""RAG outcome on the just-below-threshold (0.5) judge-qa cohort (read-only).

The judge scores are quantized to 5 Likert anchors, so there is no continuous
"near-threshold" band; the meaningful borderline set is the pairs admitted only
at threshold 0.5 (minimum criterion score == 0.5). This characterizes how each
RAG arm performs on that cohort vs the approved set, attributed by the criterion
that scored 0.5.

Interpretation caveat (why this is DESCRIPTIVE, not "the judge is too strict"):
the 0.5 anchors are legitimate exclusion reasons, and RAG-answerability does not
test them. self_containedness=0.5 ("partly needs context") is confounded because
the arms retrieve the context; informativeness=0.5 ("moderately useful/generic")
measures answer knowledge-value, which is orthogonal to answerability (generic
questions are simply easier). Higher RAG scores on the excluded cohort therefore
confirm the exclusions are well-motivated rather than mistaken.

Run from the repo root:

    uv run python -m scripts.analyze_qa_cohort_rag_outcome --id thesis-run-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.shared.rag.analysis.metrics import aggregate_arm
from scripts._judge_analysis_common import DEFAULT_THRESHOLD, load_pair_scores

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import AnswerRecord


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _build_cohorts(
    scores: dict[str, dict[str, float | None]], threshold: float
) -> dict[str, set[str]]:
    """Partition pairs into approved, the 0.5 cohort, and pure-criterion subsets."""
    approved: set[str] = set()
    cohort05: set[str] = set()
    sc_only: set[str] = set()
    inf_only: set[str] = set()
    for pid, raw in scores.items():
        if not raw or any(v is None for v in raw.values()):
            continue
        sc = {k: float(v) for k, v in raw.items() if v is not None}
        vals = list(sc.values())
        if all(v >= threshold for v in vals):
            approved.add(pid)
        elif all(v >= 0.5 for v in vals) and min(vals) == 0.5:
            cohort05.add(pid)
            if sc.get("self_containedness") == 0.5 and all(
                v >= 0.75 for k, v in sc.items() if k != "self_containedness"
            ):
                sc_only.add(pid)
            if sc.get("informativeness") == 0.5 and all(
                v >= 0.75 for k, v in sc.items() if k != "informativeness"
            ):
                inf_only.add(pid)
    return {
        f"APPROVED (>={threshold:.3f})": approved,
        "0.5 COHORT (all)": cohort05,
        "0.5 via self_containedness only": sc_only,
        "0.5 via informativeness only": inf_only,
    }


def _report(name: str, ids: set[str], by_arm: dict[str, list[AnswerRecord]]) -> None:
    print(f"\n### {name}  (|answerable| = {len(ids)})")
    print(f"{'Arm':<20} {'n_ans':>6} {'KC':>7} {'Correct':>8} {'PassCov':>8} {'answered%':>10}")
    for arm in sorted(by_arm):
        recs = [r for r in by_arm[arm] if r.is_answerable and r.qa_pair_id in ids]
        m = aggregate_arm(arm, recs, slice_name=name)
        conf = m.confusion
        answered = conf.get("TC", 0) + conf.get("TA", 0)
        denom = sum(conf.get(k, 0) for k in ("TC", "TA", "FC", "FA"))
        pct = f"{100 * answered / denom:.0f}%" if denom else "n/a"
        print(
            f"{arm:<20} {len(recs):>6} {_fmt(m.knowledge_coverage.mean):>7} "
            f"{_fmt(m.answer_correctness.mean):>8} {_fmt(m.passage_coverage.mean):>8} {pct:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG outcome on the 0.5 judge-qa cohort.")
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-01).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Approval threshold."
    )
    args = parser.parse_args()

    base = Path(args.results_root) / args.id
    scores = load_pair_scores(base / "cep" / "outputs")
    by_arm, _, _ = _group_by_arm(base / "judge_answers" / "outputs")
    cohorts = _build_cohorts(scores, args.threshold)
    for name, ids in cohorts.items():
        _report(name, ids, by_arm)


if __name__ == "__main__":
    main()
