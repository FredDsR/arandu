"""Judge-qa threshold sensitivity + score-distribution diagnostic (read-only).

Because the RAG chain is evaluated over every generated CEP pair, this recomputes
the approved set at several thresholds from the stored per-criterion scores and
re-aggregates the cross-arm metrics (reusing the production aggregators), then
prints the per-criterion score distribution. It answers two questions for the
results chapter:

  1. Is the cross-arm ranking robust to the judge-qa pass threshold?
  2. Where do the criterion scores sit relative to the threshold (is it in a
     dense region or an empty gap)?

Run from the repo root:

    uv run python -m scripts.analyze_judge_thresholds --id thesis-run-01
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.shared.rag.analysis.metrics import aggregate_arm
from scripts._judge_analysis_common import (
    ANCHORS,
    CRITERIA,
    DEFAULT_THRESHOLD,
    approved_ids,
    load_pair_scores,
    snap_to_anchor,
)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _sensitivity(cep_dir: Path, judged_dir: Path, thresholds: list[float]) -> None:
    scores = load_pair_scores(cep_dir)
    by_arm, _, _ = _group_by_arm(judged_dir)
    arms = sorted(by_arm)
    print("=" * 74)
    print("Cross-arm joint metrics vs judge-qa threshold")
    print("=" * 74)
    for t in thresholds:
        approved = approved_ids(scores, t)
        print(f"\n### threshold = {t}  (approved answerable = {len(approved)})")
        header = f"{'Arm':<20} {'KC':>7} {'Halluc':>7} {'OverC':>7} {'AbstF1':>7} {'PassCov':>8}"
        print(header)
        for arm in arms:
            kept = [r for r in by_arm[arm] if (not r.is_answerable) or (r.qa_pair_id in approved)]
            m = aggregate_arm(arm, kept, slice_name="joint")
            print(
                f"{arm:<20} {_fmt(m.knowledge_coverage.mean):>7} "
                f"{_fmt(m.hallucination_rate.value):>7} "
                f"{_fmt(m.over_cautiousness_rate.value):>7} {_fmt(m.abstention_f1):>7} "
                f"{_fmt(m.passage_coverage.mean):>8}"
            )


def _distribution(cep_dir: Path, threshold: float) -> None:
    scores = load_pair_scores(cep_dir)
    print("\n" + "=" * 74)
    print(f"Judge-qa score distribution per criterion (threshold {threshold})")
    print("=" * 74)
    for crit in CRITERIA:
        vals = [s[crit] for s in scores.values() if crit in s and s[crit] is not None]
        if not vals:
            print(f"\n{crit}: (no scores)")
            continue
        n = len(vals)
        below = sum(1 for v in vals if v < threshold)
        hist: Counter[float] = Counter(snap_to_anchor(v) for v in vals)
        mean = sum(vals) / n
        print(f"\n{crit}: n={n}  mean={mean:.3f}  <{threshold} = {below} ({100 * below / n:.1f}%)")
        peak = max(hist.values())
        for a in ANCHORS:
            c = hist.get(a, 0)
            bar = "#" * round(50 * c / peak) if peak else ""
            gap = "  <- threshold in gap" if a == 0.5 and threshold > 0.5 else ""
            print(f"  {a:>4}: {c:>5} {bar}{gap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge-qa threshold + distribution diagnostic.")
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-01).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument(
        "--thresholds",
        default="0.5,0.625,0.7",
        help="Comma-separated thresholds for the sensitivity sweep.",
    )
    args = parser.parse_args()

    base = Path(args.results_root) / args.id
    cep_dir = base / "cep" / "outputs"
    judged_dir = base / "judge_answers" / "outputs"
    thresholds = [float(t) for t in args.thresholds.split(",")]

    _sensitivity(cep_dir, judged_dir, thresholds)
    _distribution(cep_dir, DEFAULT_THRESHOLD)


if __name__ == "__main__":
    main()
