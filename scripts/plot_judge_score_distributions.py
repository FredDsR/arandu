"""Plot judge-qa score distributions per criterion, marking the pass threshold.

Two styles:

- ``bars`` (default): one bar panel per criterion over the 5 Likert anchors,
  colouring below-threshold anchors as rejected. Faithful to the raw data.
- ``density``: a Gaussian-KDE curve per criterion with a vertical threshold line
  and a shaded reject region. NOTE the scores are quantized to 5 anchors, so the
  density is a SMOOTHED view of those 5 points, not evidence of a continuum; the
  caveat is annotated on the figure.

Requires the ``report`` extra (plotly + kaleido). Run from the repo root:

    uv run --extra report python -m scripts.plot_judge_score_distributions \
        --id thesis-run-01 --style density
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
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
_KDE_BANDWIDTH = 0.08


def _criterion_values(scores: dict[str, dict[str, float | None]], crit: str) -> list[float]:
    return [s[crit] for s in scores.values() if crit in s and s[crit] is not None]


def _gaussian_kde(samples: list[float], grid: np.ndarray, bw: float) -> np.ndarray:
    """Simple Gaussian KDE (no scipy): mean of per-sample normal kernels."""
    s = np.asarray(samples, dtype=float)[:, None]
    diff = (grid[None, :] - s) / bw
    kernels = np.exp(-0.5 * diff**2) / (bw * np.sqrt(2 * np.pi))
    return kernels.mean(axis=0)


def _plot_bars(scores: dict, threshold: float, run_id: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=list(CRITERIA), vertical_spacing=0.16, horizontal_spacing=0.1
    )
    for i, crit in enumerate(CRITERIA):
        vals = _criterion_values(scores, crit)
        hist: Counter[float] = Counter(snap_to_anchor(v) for v in vals)
        heights = [hist.get(a, 0) for a in ANCHORS]
        colours = [_REJECT if a < threshold else _PASS for a in ANCHORS]
        n = len(vals)
        below = sum(h for a, h in zip(ANCHORS, heights, strict=True) if a < threshold)
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
        fig.layout.annotations[i].text = f"{crit}  (n={n}, <{threshold} = {below} = {pct:.0f}%)"
    fig.update_layout(
        title_text=(
            f"judge-qa score distributions ({run_id}) - 5-point Likert; "
            f"threshold {threshold} between the 0.5 and 0.75 anchors"
        ),
        template="plotly_white",
        height=720,
        width=1100,
        bargap=0.35,
    )
    fig.update_xaxes(title_text="score anchor")
    fig.update_yaxes(title_text="pairs")
    return fig


def _plot_hist(scores: dict, threshold: float, run_id: str) -> go.Figure:
    """Fine-binned histogram (no smoothing): shows the raw quantized spikes."""
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=list(CRITERIA), vertical_spacing=0.16, horizontal_spacing=0.1
    )
    for i, crit in enumerate(CRITERIA):
        vals = _criterion_values(scores, crit)
        row, col = divmod(i, 2)
        if not vals:
            continue
        fig.add_trace(
            go.Histogram(
                x=vals,
                xbins={"start": -0.0125, "end": 1.0125, "size": 0.025},
                histnorm="probability density",
                marker_color="#2c3e50",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.add_vrect(
            x0=-0.05,
            x1=threshold,
            fillcolor=_REJECT,
            opacity=0.08,
            line_width=0,
            row=row + 1,
            col=col + 1,
        )
        fig.add_vline(
            x=threshold,
            line={"color": _REJECT, "width": 1.5, "dash": "dash"},
            row=row + 1,
            col=col + 1,
        )
        n = len(vals)
        below = sum(1 for v in vals if v < threshold)
        pct = 100 * below / n if n else 0.0
        fig.layout.annotations[i].text = f"{crit}  (n={n}, <{threshold} = {below} = {pct:.0f}%)"
    fig.update_layout(
        title_text=(
            f"judge-qa score histogram ({run_id}) - raw, bin=0.025 (no smoothing); "
            f"dashed = threshold {threshold}"
        ),
        template="plotly_white",
        height=720,
        width=1100,
    )
    fig.update_xaxes(title_text="score", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="density")
    return fig


def _plot_density(scores: dict, threshold: float, run_id: str) -> go.Figure:
    grid = np.linspace(-0.05, 1.05, 300)
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=list(CRITERIA), vertical_spacing=0.16, horizontal_spacing=0.1
    )
    for i, crit in enumerate(CRITERIA):
        vals = _criterion_values(scores, crit)
        row, col = divmod(i, 2)
        if not vals:
            continue
        density = _gaussian_kde(vals, grid, _KDE_BANDWIDTH)
        n = len(vals)
        below = sum(1 for v in vals if v < threshold)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                line={"color": "#2c3e50", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(44,62,80,0.12)",
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        # Shade the reject region (score < threshold) and mark the threshold line.
        fig.add_vrect(
            x0=grid[0],
            x1=threshold,
            fillcolor=_REJECT,
            opacity=0.08,
            line_width=0,
            row=row + 1,
            col=col + 1,
        )
        fig.add_vline(
            x=threshold,
            line={"color": _REJECT, "width": 1.5, "dash": "dash"},
            row=row + 1,
            col=col + 1,
        )
        pct = 100 * below / n if n else 0.0
        fig.layout.annotations[i].text = f"{crit}  (n={n}, <{threshold} = {below} = {pct:.0f}%)"
    fig.update_layout(
        title_text=(
            f"judge-qa score density ({run_id}) - dashed = threshold {threshold}; "
            "scores quantized to {0,.25,.5,.75,1} (KDE smoothed, bw="
            f"{_KDE_BANDWIDTH})"
        ),
        template="plotly_white",
        height=720,
        width=1100,
    )
    fig.update_xaxes(title_text="score", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="density")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot judge-qa score distributions.")
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-01).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--style", choices=["bars", "density", "hist"], default="bars")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: results/<id>/analysis/judge_score_<style>.png).",
    )
    args = parser.parse_args()

    base = Path(args.results_root) / args.id
    scores = load_pair_scores(base / "cep" / "outputs")
    names = {
        "bars": "judge_score_distributions",
        "density": "judge_score_density",
        "hist": "judge_score_hist",
    }
    out = Path(args.out) if args.out else base / "analysis" / f"{names[args.style]}.png"

    builders = {"bars": _plot_bars, "density": _plot_density, "hist": _plot_hist}
    fig = builders[args.style](scores, args.threshold, args.id)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out))
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
