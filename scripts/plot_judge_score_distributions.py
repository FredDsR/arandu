"""Plot judge-qa score distributions per criterion, marking the pass threshold.

Renders one bar panel per criterion over the 5 Likert anchors, colouring
below-threshold anchors as rejected, and exports a static image. Shows the
quantization and whether the threshold falls in an empty gap.

Requires the ``report`` extra (plotly + kaleido). Run from the repo root:

    uv run --extra report python -m scripts.plot_judge_score_distributions \
        --id thesis-run-01 --out results/thesis-run-01/analysis/judge_score_distributions.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts._judge_analysis_common import (
    ANCHORS,
    CRITERIA,
    DEFAULT_THRESHOLD,
    load_pair_scores,
    snap_to_anchor,
)

_REJECT = "#c0392b"
_PASS = "#27ae60"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot judge-qa score distributions.")
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-01).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: results/<id>/analysis/judge_score_distributions.png).",
    )
    args = parser.parse_args()

    base = Path(args.results_root) / args.id
    scores = load_pair_scores(base / "cep" / "outputs")
    out = Path(args.out) if args.out else base / "analysis" / "judge_score_distributions.png"

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(CRITERIA),
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )
    for i, crit in enumerate(CRITERIA):
        vals = [s[crit] for s in scores.values() if crit in s and s[crit] is not None]
        hist: Counter[float] = Counter(snap_to_anchor(v) for v in vals)
        heights = [hist.get(a, 0) for a in ANCHORS]
        colours = [_REJECT if a < args.threshold else _PASS for a in ANCHORS]
        n = len(vals)
        below = sum(h for a, h in zip(ANCHORS, heights, strict=True) if a < args.threshold)
        row, col = divmod(i, 2)
        fig.add_trace(
            go.Bar(
                x=[str(a) for a in ANCHORS],
                y=heights,
                marker_color=colours,
                text=heights,
                textposition="outside",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        pct = 100 * below / n if n else 0.0
        fig.layout.annotations[
            i
        ].text = f"{crit}  (n={n}, <{args.threshold} = {below} = {pct:.0f}%)"

    fig.update_layout(
        title_text=(
            f"judge-qa score distributions ({args.id}) - 5-point Likert; "
            f"threshold {args.threshold} between the 0.5 and 0.75 anchors"
        ),
        template="plotly_white",
        height=720,
        width=1100,
        bargap=0.35,
    )
    fig.update_xaxes(title_text="score anchor")
    fig.update_yaxes(title_text="pairs")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
