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

    uv run python -m analysis.analyze_qa_cohort_rag_outcome --id thesis-run-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis._judge_common import DEFAULT_THRESHOLD, load_pair_scores, outcome_table
from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.utils.console import console


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


def main() -> None:
    """Entry point for the 0.5-cohort outcome report."""
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
    lines: list[str] = []
    for name, ids in cohorts.items():
        lines += outcome_table(name, ids, by_arm)
    console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)


if __name__ == "__main__":
    main()
