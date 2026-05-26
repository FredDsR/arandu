"""Emit ``report.json`` + ``tables.md`` for an analysis run (spec §8.8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arandu.shared.rag.analysis.metrics import ArmMetrics, MeanMetric, ProportionMetric

if TYPE_CHECKING:
    from pathlib import Path


class AnalysisReport(BaseModel):
    """Top-level run report.

    Attributes:
        pipeline_id: The run id this analysis was computed against.
        joint: Per-arm joint-benchmark metrics keyed by arm id.
        by_bloom: ``{arm: {bloom_level: ArmMetrics}}`` cross-cut.
        by_question_type: ``{arm: {question_type: ArmMetrics}}`` cross-cut.
    """

    pipeline_id: str
    joint: dict[str, ArmMetrics] = Field(default_factory=dict)
    by_bloom: dict[str, dict[str, ArmMetrics]] = Field(default_factory=dict)
    by_question_type: dict[str, dict[str, ArmMetrics]] = Field(default_factory=dict)


def write_report(outputs_dir: Path, report: AnalysisReport) -> tuple[Path, Path]:
    """Write ``report.json`` and ``tables.md`` under ``outputs_dir``.

    Returns ``(json_path, tables_path)``. The two file paths align so
    downstream consumers (thesis chapter, dashboards) can read either
    form without separate config.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / "report.json"
    tables_path = outputs_dir / "tables.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tables_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, tables_path


def _render_markdown(report: AnalysisReport) -> str:
    """Render Markdown tables matching the spec §8.8 mock layout."""
    parts: list[str] = [
        f"# RAG Analysis Report — {report.pipeline_id}",
        "",
        _table_1_joint(report.joint),
    ]
    if report.by_bloom:
        parts.extend(["", _table_2_bloom(report.by_bloom)])
    if report.by_question_type:
        parts.extend(["", _table_question_type(report.by_question_type)])
    return "\n".join(parts) + "\n"


def _table_1_joint(joint: dict[str, ArmMetrics]) -> str:
    """Table 1 — per-arm joint-benchmark metrics."""
    header = (
        "## Table 1 — Per-arm joint-benchmark metrics\n"
        "\n"
        "| Arm | n_TA+TC | n_TA+FC | KC ↑ | Hallucination ↓ | Over-caution ↓ "
        "| Abstention F1 ↑ | Passage cov (judge) ↑ |\n"
        "|-----|---------|---------|------|-----------------|----------------"
        "|-----------------|-----------------------|"
    )
    rows = [header]
    for arm in sorted(joint):
        m = joint[arm]
        n_answerable = m.confusion.get("TC", 0) + m.confusion.get("FA", 0)
        n_nonanswerable = m.confusion.get("TA", 0) + m.confusion.get("FC", 0)
        rows.append(
            f"| {arm} | {n_answerable} | {n_nonanswerable} "
            f"| {_fmt_mean(m.knowledge_coverage.mean)} "
            f"| {_fmt_prop_ci(m.hallucination_rate)} "
            f"| {_fmt_prop_ci(m.over_cautiousness_rate)} "
            f"| {_fmt_mean(m.abstention_f1)} "
            f"| {_fmt_mean(m.passage_coverage.mean)} |"
        )
    return "\n".join(rows)


def _table_2_bloom(by_bloom: dict[str, dict[str, ArmMetrics]]) -> str:
    """Table 2 — Bloom-stratified KC."""
    all_levels = sorted({level for arm in by_bloom.values() for level in arm})
    header = "## Table 2 — Bloom-stratified Knowledge Coverage (KC)\n\n"
    header += "| Arm | " + " | ".join(all_levels) + " |\n"
    header += "|" + "-----|" * (len(all_levels) + 1)
    rows = [header]
    for arm in sorted(by_bloom):
        cells = [
            _fmt_mean(by_bloom[arm].get(level, _empty()).knowledge_coverage.mean)
            for level in all_levels
        ]
        rows.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _table_question_type(by_type: dict[str, dict[str, ArmMetrics]]) -> str:
    """Table — question-type-stratified KC."""
    all_types = sorted({qt for arm in by_type.values() for qt in arm})
    header = "## Table 3 — Question-type-stratified Knowledge Coverage (KC)\n\n"
    header += "| Arm | " + " | ".join(all_types) + " |\n"
    header += "|" + "-----|" * (len(all_types) + 1)
    rows = [header]
    for arm in sorted(by_type):
        cells = [
            _fmt_mean(by_type[arm].get(qt, _empty()).knowledge_coverage.mean) for qt in all_types
        ]
        rows.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _fmt_prop_ci(p: ProportionMetric) -> str:
    """Render a ProportionMetric as ``value [lo, hi]`` or ``n/a``."""
    if p.value is None:
        return "n/a"
    return f"{p.value:.3f} [{p.ci_lower:.3f}, {p.ci_upper:.3f}]"


def _fmt_mean(v: float | None) -> str:
    """Render a scalar mean as 3 decimals or ``n/a``."""
    return "n/a" if v is None else f"{v:.3f}"


def _empty() -> ArmMetrics:
    """Empty ArmMetrics fallback used for missing strata."""
    zero_prop = ProportionMetric(numerator=0, denominator=0, value=None, ci_lower=0.0, ci_upper=0.0)
    zero_mean = MeanMetric(mean=None, n=0)
    return ArmMetrics(
        arm="",
        slice_name="",
        confusion={"TA": 0, "TC": 0, "FA": 0, "FC": 0, "unknown": 0},
        hallucination_rate=zero_prop,
        over_cautiousness_rate=zero_prop,
        abstention_precision=zero_prop,
        abstention_recall=zero_prop,
        abstention_f1=None,
        answer_correctness=zero_mean,
        answer_faithfulness=zero_mean,
        knowledge_coverage=zero_mean,
        passage_coverage=zero_mean,
    )
