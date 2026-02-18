"""Static figure builders for publication-quality exports.

Creates publication-ready figures using Matplotlib and Seaborn that can be
exported to PNG, SVG, and PDF formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .style import BLOOM_COLORS, CATEGORICAL_COLORS, get_matplotlib_style

if TYPE_CHECKING:
    from pathlib import Path

    from .collector import RunReport

# Set Agg backend for headless environments
matplotlib.use("Agg")


def configure_matplotlib() -> None:
    """Configure matplotlib with publication-quality settings."""
    style_config = get_matplotlib_style()
    plt.rcParams.update(style_config)


def save_figure(fig: matplotlib.figure.Figure, path: Path, format: str = "png") -> None:
    """Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure object.
        path: Output file path.
        format: Output format (png, svg, pdf).
    """
    fig.savefig(path, format=format, bbox_inches="tight", dpi=300)
    plt.close(fig)


def create_bloom_distribution_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save a Bloom taxonomy distribution bar chart.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    pipeline_ids = []
    bloom_data = {"remember": [], "understand": [], "analyze": [], "evaluate": []}

    for report in reports:
        if report.cep_records:
            pipeline_ids.append(report.pipeline_id)
            total_bloom = {"remember": 0, "understand": 0, "analyze": 0, "evaluate": 0}
            for cep_record in report.cep_records:
                for level, count in cep_record.bloom_distribution.items():
                    if level in total_bloom:
                        total_bloom[level] += count

            for level in bloom_data:
                bloom_data[level].append(total_bloom[level])

    if not pipeline_ids:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pipeline_ids))
    width = 0.6
    bottom = np.zeros(len(pipeline_ids))

    for level in ["remember", "understand", "analyze", "evaluate"]:
        ax.bar(
            x,
            bloom_data[level],
            width,
            label=level.capitalize(),
            bottom=bottom,
            color=BLOOM_COLORS[level],
        )
        bottom += np.array(bloom_data[level])

    ax.set_xlabel("Pipeline ID")
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title("Bloom's Taxonomy Distribution by Run")
    ax.set_xticks(x)
    ax.set_xticklabels(pipeline_ids, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, output_path / f"bloom_distribution.{format}", format)


def create_validation_scores_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save validation score box plots.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    criterion_data = {c: [] for c in criteria}

    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                if hasattr(qa_pair, "validation") and qa_pair.validation:
                    for criterion in criteria:
                        score = getattr(qa_pair.validation, criterion, None)
                        if score is not None:
                            criterion_data[criterion].append(score)

    if not any(criterion_data.values()):
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [criterion_data[c] for c in criteria if criterion_data[c]]
    labels = [c.replace("_", " ").title() for c in criteria if criterion_data[c]]

    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        notch=False,
        showmeans=True,
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], CATEGORICAL_COLORS, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score")
    ax.set_title("Validation Score Distributions by Criterion")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    save_figure(fig, output_path / f"validation_scores_boxplot.{format}", format)


def create_confidence_distribution_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save confidence score histogram.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    confidence_scores = []
    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                if hasattr(qa_pair, "confidence_score") and qa_pair.confidence_score:
                    confidence_scores.append(qa_pair.confidence_score)

    if not confidence_scores:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidence_scores, bins=30, color=CATEGORICAL_COLORS[0], alpha=0.7, edgecolor="black")

    # Add KDE overlay
    if len(confidence_scores) > 1:
        from scipy import stats

        density = stats.gaussian_kde(confidence_scores)
        xs = np.linspace(min(confidence_scores), max(confidence_scores), 200)
        ax2 = ax.twinx()
        ax2.plot(xs, density(xs), color=CATEGORICAL_COLORS[1], linewidth=2, label="KDE")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper left")

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("QA Pair Confidence Score Distribution")
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, output_path / f"confidence_distribution.{format}", format)


def create_transcription_quality_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save transcription quality multi-panel histogram.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    quality_scores = {
        "Overall": [],
        "Script Match": [],
        "Repetition": [],
        "Segment Quality": [],
        "Content Density": [],
    }

    for report in reports:
        for record in report.transcription_records:
            if record.transcription_quality:
                q = record.transcription_quality
                quality_scores["Overall"].append(q.overall_score)
                quality_scores["Script Match"].append(q.script_match_score)
                quality_scores["Repetition"].append(q.repetition_score)
                quality_scores["Segment Quality"].append(q.segment_quality_score)
                quality_scores["Content Density"].append(q.content_density_score)

    if not quality_scores["Overall"]:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Transcription Quality Metrics", fontsize=14, fontweight="bold")

    for ax, (name, scores), color in zip(
        axes.flat, quality_scores.items(), CATEGORICAL_COLORS, strict=False
    ):
        if scores:
            ax.hist(scores, bins=20, color=color, alpha=0.7, edgecolor="black")
            ax.set_title(name)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.3)

    # Hide the last subplot if not needed
    axes.flat[-1].axis("off")

    plt.tight_layout()
    save_figure(fig, output_path / f"transcription_quality.{format}", format)


def create_multihop_ratio_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save multi-hop ratio pie chart.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    multihop_count = 0
    single_hop_count = 0

    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                if hasattr(qa_pair, "is_multi_hop") and qa_pair.is_multi_hop is not None:
                    if qa_pair.is_multi_hop:
                        multihop_count += 1
                    else:
                        single_hop_count += 1

    if multihop_count + single_hop_count == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        [multihop_count, single_hop_count],
        labels=["Multi-hop", "Single-hop"],
        autopct="%1.1f%%",
        colors=[CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1]],
        startangle=90,
    )
    ax.set_title("Multi-hop vs Single-hop Questions")

    save_figure(fig, output_path / f"multihop_ratio.{format}", format)


def create_run_comparison_figure(
    reports: list[RunReport], output_path: Path, format: str = "png"
) -> None:
    """Create and save run comparison bar chart.

    Args:
        reports: List of RunReport objects to visualize.
        output_path: Directory to save the figure.
        format: Output format (png, svg, pdf).
    """
    configure_matplotlib()

    pipeline_ids = []
    success_rates = []

    for report in reports:
        if report.transcription_metadata or report.cep_metadata:
            metadata = report.cep_metadata or report.transcription_metadata
            if metadata and metadata.success_rate is not None:
                pipeline_ids.append(report.pipeline_id)
                success_rates.append(metadata.success_rate)

    if not pipeline_ids:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(pipeline_ids, success_rates, color=CATEGORICAL_COLORS[0], alpha=0.7)
    ax.set_xlabel("Pipeline ID")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate Comparison Across Runs")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    save_figure(fig, output_path / f"run_comparison.{format}", format)


def export_all_figures(
    reports: list[RunReport], output_dir: Path, format: str = "png"
) -> list[Path]:
    """Export all available figures for the given reports.

    Args:
        reports: List of RunReport objects to visualize.
        output_dir: Directory to save all figures.
        format: Output format (png, svg, pdf).

    Returns:
        List of paths to generated figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    figure_functions = [
        create_bloom_distribution_figure,
        create_validation_scores_figure,
        create_confidence_distribution_figure,
        create_transcription_quality_figure,
        create_multihop_ratio_figure,
        create_run_comparison_figure,
    ]

    for func in figure_functions:
        try:
            func(reports, output_dir, format)
            # Determine the expected filename
            func_name = func.__name__.replace("create_", "").replace("_figure", "")
            expected_file = output_dir / f"{func_name}.{format}"
            if expected_file.exists():
                generated_files.append(expected_file)
        except Exception:
            # Skip figures that fail to generate
            pass

    return generated_files
