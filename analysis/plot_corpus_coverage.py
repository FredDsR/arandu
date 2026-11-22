"""Plot corpus concentration curves and yield-by-length quartiles (read-only).

Reads the JSON report produced by ``analysis.analyze_corpus_coverage`` and writes
a two-panel figure (PDF + 200 dpi PNG):

- Panel (a): cumulative share of transcribed text, useful QA pairs, and graph
  nodes as interviews are sorted from largest to smallest.
- Panel (b): useful QA pairs per chunk across interview-length quartiles.

Run from the repo root:

    uv run python -m analysis.plot_corpus_coverage \\
        --input results/thesis-run-02/analysis/corpus_coverage.json \\
        --output results/thesis-run-02/analysis/corpus_coverage.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = {"text": "#2a78d6", "useful": "#eb6834", "nodes": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def concentration_curve(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative share coordinates sorted in decreasing order.

    Args:
        values: Non-negative numeric sequence.

    Returns:
        Tuple ``(x, y)`` of cumulative item fraction and cumulative value share.
    """
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    x = np.arange(0, len(arr) + 1) / max(1, len(arr))
    total = float(arr.sum())
    y = np.concatenate([[0.0], np.cumsum(arr) / total if total > 0 else np.zeros(len(arr))])
    return x, y


def plot_coverage_figure(data: dict[str, Any], output_pdf: Path) -> None:
    """Render the two-panel corpus coverage figure to PDF and PNG.

    Args:
        data: Parsed JSON dictionary produced by ``analysis.analyze_corpus_coverage``.
        output_pdf: Destination PDF path (a sibling ``.png`` is also written).
    """
    rows = data["interviews"]["rows"]
    nodes_by_file = {r["file"]: r["nodes"] for r in data["graph"]["rows"]}
    text = [float(r["chars"]) for r in rows]
    useful = [float(r["useful"]) for r in rows]
    gnodes = [float(nodes_by_file.get(r["file"], 0)) for r in rows]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    styles = [
        ("text", text, "Transcribed text", "-"),
        ("useful", useful, "Useful pairs", "--"),
        ("nodes", gnodes, "Graph nodes", ":"),
    ]
    for key, vals, label, ls in styles:
        x, y = concentration_curve(vals)
        ax_left.plot(x * 100, y * 100, color=SERIES[key], lw=2, ls=ls, label=label)
    ax_left.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax_left.axvline(25, color=GRID, lw=0.8, zorder=0)
    ax_left.set_xlabel("Interviews, sorted by the quantity (%)")
    ax_left.set_ylabel("Cumulative share (%)")
    ax_left.set_xlim(0, 100)
    ax_left.set_ylim(0, 101)
    ax_left.grid(axis="y", color=GRID, lw=0.6)
    ax_left.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax_left.set_title(
        f"(a) Concentration across the {len(rows)} interviews",
        fontsize=8,
        color=INK,
        loc="left",
    )
    for spine in ("top", "right"):
        ax_left.spines[spine].set_visible(False)

    bands = data["interviews"]["length_bands"]
    names = list(bands)
    ypc = [bands[name]["yield_per_chunk"] for name in names]
    xs = np.arange(len(names))
    bars = ax_right.bar(xs, ypc, width=0.6, color=SERIES["useful"])
    for x, bar in zip(xs, bars, strict=True):
        ax_right.text(
            x,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            color=INK,
        )
    labels = [
        f"{name}\n{bands[name]['chars_min'] / 1000:.1f}-{bands[name]['chars_max'] / 1000:.1f}k"
        for name in names
    ]
    ax_right.set_xticks(xs, labels)
    ax_right.set_ylabel("Useful pairs per chunk")
    ax_right.set_ylim(0, max(1.05, max(ypc, default=1.0) * 1.2))
    ax_right.grid(axis="y", color=GRID, lw=0.6)
    ax_right.set_axisbelow(True)
    ax_right.set_title("(b) Yield by interview length quartile", fontsize=8, color=INK, loc="left")
    for spine in ("top", "right"):
        ax_right.spines[spine].set_visible(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point for plotting corpus coverage."""
    parser = argparse.ArgumentParser(
        description="Plot corpus concentration and quartile yield from corpus_coverage.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON output from analysis.analyze_corpus_coverage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination PDF path (a 200 dpi PNG is saved alongside it).",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    plot_coverage_figure(data, args.output)


if __name__ == "__main__":
    main()
