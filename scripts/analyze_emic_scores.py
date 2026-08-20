"""Descriptive analysis of the emic-validity judge's scores (read-only).

Aggregates ``results/<id>/emic_judge/outputs/*.json`` into the tables the
results chapter needs: the ordinal distribution over the whole corpus, the same
distribution split by Bloom level, by ``judge-qa`` verdict and by source
interview, plus a run-health block. This is the automatic measurement only; the
human-annotation agreement (Krippendorff alpha, weighted Cohen kappa, AC2) is
the separate ``emic-analysis`` work and is deliberately absent here.

Pairs whose LLM call errored carry ``emic_score=None``. They are counted under
run health and excluded from every mean, median and percentage, so a dead
sidecar reads as a failure rather than as low emic validity.

Run from the repo root:

    uv run python -m scripts.analyze_emic_scores --id thesis-run-01
    uv run python -m scripts.analyze_emic_scores --id thesis-run-01 --out emic.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from pydantic import ValidationError

from arandu.shared.emic.schemas import EmicSourceScores
from arandu.shared.schemas import BloomLevel
from arandu.utils.console import console
from arandu.utils.logger import print_error, print_success

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arandu.shared.emic.schemas import EmicScore

ANCHORS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The ordinal emic-validity labels, in ruler order."""

CANONICAL_BLOOM_ORDER: tuple[str, ...] = get_args(BloomLevel)
STAGE = "emic_judge"
UNKNOWN_SCOPE = "unknown"
VERDICT_LABELS: dict[str, str] = {
    "approved": "approved",
    "rejected": "rejected",
    "unjudged": "unjudged",
}


@dataclass(frozen=True)
class Distribution:
    """Summary of one set of ordinal scores.

    Attributes:
        n: Number of pairs carrying an ordinal score.
        counts: Pairs per anchor, always keyed by every anchor in ``1..5``.
        mean: Arithmetic mean, or ``None`` when ``n`` is 0.
        median: Median, or ``None`` when ``n`` is 0.
        p25: First quartile, or ``None`` when ``n`` is 0.
        p75: Third quartile, or ``None`` when ``n`` is 0.
        pct_ge_4: Share of scores at 4 or 5, or ``None`` when ``n`` is 0.
    """

    n: int
    counts: dict[int, int]
    mean: float | None
    median: float | None
    p25: float | None
    p75: float | None
    pct_ge_4: float | None


@dataclass(frozen=True)
class Health:
    """Coverage and failure counters for the run being analysed.

    Attributes:
        scope: The run's scope (``all``, ``approved`` or ``unknown``).
        sources_read: Output files parsed.
        sources_unreadable: Output files that failed to parse.
        pairs_total: Pairs found across the parsed files.
        pairs_scored: Pairs carrying an ordinal score.
        pairs_failed: Pairs whose LLM call errored.
    """

    scope: str
    sources_read: int
    sources_unreadable: int
    pairs_total: int
    pairs_scored: int
    pairs_failed: int


@dataclass(frozen=True)
class SourceRow:
    """One source interview's distribution.

    Attributes:
        file_id: Source interview id.
        filename: Original media filename.
        dist: The source's score distribution.
    """

    file_id: str
    filename: str
    dist: Distribution


@dataclass(frozen=True)
class EmicAnalysis:
    """Everything the report renders.

    Attributes:
        health: Coverage and failure counters.
        overall: Distribution over every scored pair.
        by_bloom: Distribution per Bloom level, in canonical ladder order.
        by_verdict: Distribution per ``judge-qa`` verdict bucket.
        by_source: Per-source distributions, worst mean first.
    """

    health: Health
    overall: Distribution
    by_bloom: dict[str, Distribution]
    by_verdict: dict[str, Distribution]
    by_source: list[SourceRow]


def distribution(values: Iterable[int]) -> Distribution:
    """Summarise a set of ordinal scores.

    Args:
        values: Ordinal scores in ``1..5``. Errored pairs must be filtered out
            by the caller; this function has no notion of a missing score.

    Returns:
        The :class:`Distribution`. An empty input yields zero counts and no
        central tendency rather than a zero mean.
    """
    data = sorted(values)
    counts = dict.fromkeys(ANCHORS, 0)
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    if not data:
        return Distribution(0, counts, None, None, None, None, None)
    if len(data) == 1:
        only = float(data[0])
        return Distribution(1, counts, only, only, only, only, float(data[0] >= 4))
    p25, _, p75 = statistics.quantiles(data, n=4, method="inclusive")
    return Distribution(
        n=len(data),
        counts=counts,
        mean=statistics.fmean(data),
        median=statistics.median(data),
        p25=p25,
        p75=p75,
        pct_ge_4=sum(1 for value in data if value >= 4) / len(data),
    )


def _verdict_bucket(score: EmicScore) -> str:
    """Map a pair's ``judge-qa`` verdict to its cross-tab row."""
    if score.is_valid is None:
        return "unjudged"
    return "approved" if score.is_valid else "rejected"


def _bloom_sort_key(level: str) -> tuple[int, str]:
    """Order Bloom levels by the canonical ladder, unknown levels last."""
    if level in CANONICAL_BLOOM_ORDER:
        return (CANONICAL_BLOOM_ORDER.index(level), "")
    return (len(CANONICAL_BLOOM_ORDER), level)


def build_analysis(sources: list[EmicSourceScores], unreadable: int, scope: str) -> EmicAnalysis:
    """Aggregate per-source scores into every table of the report.

    Args:
        sources: Parsed per-source score files.
        unreadable: Output files that failed to parse, for the health block.
        scope: The run's scope, as resolved by :func:`resolve_scope`.

    Returns:
        The assembled :class:`EmicAnalysis`.
    """
    scored: list[int] = []
    failed = 0
    total = 0
    bloom_values: dict[str, list[int]] = {}
    verdict_values: dict[str, list[int]] = {bucket: [] for bucket in VERDICT_LABELS}
    rows: list[SourceRow] = []

    for source in sources:
        source_values: list[int] = []
        for score in source.scores:
            total += 1
            # Register the level even for an errored pair: a Bloom level whose
            # every pair failed must show up as n=0, not vanish from the table.
            level_values = bloom_values.setdefault(score.bloom_level, [])
            if score.emic_score is None:
                failed += 1
                continue
            scored.append(score.emic_score)
            source_values.append(score.emic_score)
            level_values.append(score.emic_score)
            verdict_values[_verdict_bucket(score)].append(score.emic_score)
        rows.append(
            SourceRow(
                file_id=source.source_file_id,
                filename=source.source_filename,
                dist=distribution(source_values),
            )
        )

    # A source whose every pair errored has no mean; sort it last rather than
    # letting it masquerade as the worst-scoring interview.
    rows.sort(key=lambda row: (row.dist.mean is None, row.dist.mean or 0.0, row.filename))

    return EmicAnalysis(
        health=Health(
            scope=scope,
            sources_read=len(sources),
            sources_unreadable=unreadable,
            pairs_total=total,
            pairs_scored=len(scored),
            pairs_failed=failed,
        ),
        overall=distribution(scored),
        by_bloom={
            level: distribution(bloom_values[level])
            for level in sorted(bloom_values, key=_bloom_sort_key)
        },
        by_verdict={bucket: distribution(values) for bucket, values in verdict_values.items()},
        by_source=rows,
    )


def load_source_scores(outputs_dir: Path) -> tuple[list[EmicSourceScores], int]:
    """Read every per-source score file under ``outputs_dir``.

    Args:
        outputs_dir: ``results/<id>/emic_judge/outputs``.

    Returns:
        The parsed files (sorted by path) and the number that failed to parse.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not outputs_dir.is_dir():
        raise FileNotFoundError(
            f"No emic-judge outputs at {outputs_dir}. Run `arandu emic-judge --id <id>` first."
        )
    sources: list[EmicSourceScores] = []
    unreadable = 0
    for path in sorted(outputs_dir.glob("*.json")):
        try:
            sources.append(EmicSourceScores.load(path))
        except (OSError, ValidationError):
            unreadable += 1
    return sources, unreadable


def resolve_scope(run_dir: Path, sources: list[EmicSourceScores]) -> str:
    """Determine which pairs the run was allowed to score.

    Prefers the ``run_metadata.json`` snapshot, which is written once per run.
    Falls back to the ``scope`` persisted on the outputs themselves, and only
    accepts it when every file agrees: files written before the field existed
    carry ``None``, and mixing them with a later run's outputs makes the scope
    unknowable.

    Args:
        run_dir: ``results/<id>/emic_judge``.
        sources: The parsed per-source score files.

    Returns:
        ``"all"``, ``"approved"`` or ``"unknown"``.
    """
    metadata_path = run_dir / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scope = metadata["config"]["config_values"]["scope"]
    except (OSError, ValueError, KeyError, TypeError):
        scope = None
    if scope in ("all", "approved"):
        return str(scope)
    scopes = {source.scope for source in sources}
    if len(scopes) == 1 and (only := scopes.pop()) is not None:
        return only
    return UNKNOWN_SCOPE


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _iqr(dist: Distribution) -> str:
    if dist.p25 is None or dist.p75 is None:
        return "n/a"
    return f"{dist.p25:.1f}-{dist.p75:.1f}"


def _share(count: int, total: int) -> str:
    return "n/a" if total == 0 else f"{count / total * 100:.1f}%"


def _health_table(health: Health) -> list[str]:
    return [
        "## Run health",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Scope | {health.scope} |",
        f"| Sources read | {health.sources_read} |",
        f"| Sources unreadable | {health.sources_unreadable} |",
        f"| Pairs found | {health.pairs_total} |",
        f"| Pairs scored | {health.pairs_scored} |",
        f"| Pairs failed (LLM error) | {health.pairs_failed} "
        f"({_share(health.pairs_failed, health.pairs_total)}) |",
        "",
    ]


def _overall_table(dist: Distribution) -> list[str]:
    lines = [
        "## Score distribution",
        "",
        "| Score | n | % |",
        "|-------|---|---|",
    ]
    lines += [
        f"| {anchor} | {dist.counts[anchor]} | {_share(dist.counts[anchor], dist.n)} |"
        for anchor in ANCHORS
    ]
    lines += [
        f"| **total** | **{dist.n}** | |",
        "",
        f"mean {_num(dist.mean)} | median {_num(dist.median)} | IQR {_iqr(dist)} "
        f"| >=4 {_pct(dist.pct_ge_4)}",
        "",
    ]
    return lines


def _breakdown_table(title: str, label: str, rows: list[tuple[str, Distribution]]) -> list[str]:
    anchors_header = " | ".join(str(anchor) for anchor in ANCHORS)
    lines = [
        title,
        "",
        f"| {label} | n | mean | median | IQR | >=4 | {anchors_header} |",
        f"|{'---|' * (6 + len(ANCHORS))}",
    ]
    for name, dist in rows:
        counts = " | ".join(str(dist.counts[anchor]) for anchor in ANCHORS)
        lines.append(
            f"| {name} | {dist.n} | {_num(dist.mean)} | {_num(dist.median)} | {_iqr(dist)} "
            f"| {_pct(dist.pct_ge_4)} | {counts} |"
        )
    lines.append("")
    return lines


def _verdict_rows(analysis: EmicAnalysis) -> list[tuple[str, Distribution]]:
    """Pick the cross-tab rows the run's scope can actually support."""
    if analysis.health.scope == "approved":
        return [(VERDICT_LABELS["approved"], analysis.by_verdict["approved"])]
    return [(label, analysis.by_verdict[bucket]) for bucket, label in VERDICT_LABELS.items()]


def _source_table(rows: list[SourceRow]) -> list[str]:
    lines = [
        "## By source",
        "",
        "| Source | n | mean | median | IQR | >=4 |",
        "|--------|---|------|--------|-----|-----|",
    ]
    lines += [
        f"| {row.filename} | {row.dist.n} | {_num(row.dist.mean)} | {_num(row.dist.median)} "
        f"| {_iqr(row.dist)} | {_pct(row.dist.pct_ge_4)} |"
        for row in rows
    ]
    lines.append("")
    return lines


def render_markdown(analysis: EmicAnalysis, pipeline_id: str) -> str:
    """Render every table as Markdown.

    Args:
        analysis: The aggregated analysis.
        pipeline_id: Run identifier, for the heading.

    Returns:
        The Markdown document, also used verbatim as the console output.
    """
    lines = [
        f"# Emic-validity scores - {pipeline_id}",
        "",
        "Automatic judge only; no human-agreement coefficients. Pairs whose LLM call "
        "errored are counted under run health and excluded from every statistic below.",
        "",
    ]
    lines += _health_table(analysis.health)
    lines += _overall_table(analysis.overall)
    lines += _breakdown_table("## By Bloom level", "Bloom level", list(analysis.by_bloom.items()))
    lines += _breakdown_table(
        "## Emic score x judge-qa verdict", "Verdict", _verdict_rows(analysis)
    )
    if analysis.health.scope == "approved":
        lines += [
            "Scope is `approved`: pairs the judge-qa step rejected or never judged were "
            "never scored, so they are omitted here rather than reported as zero.",
            "",
        ]
    elif analysis.health.scope == UNKNOWN_SCOPE:
        lines += [
            "Scope is unknown (no `run_metadata.json` and the outputs disagree), so the "
            "verdict rows may be empty because those pairs were out of scope rather than "
            "absent from the corpus.",
            "",
        ]
    lines += _source_table(analysis.by_source)
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, aggregate the run and emit the tables."""
    parser = argparse.ArgumentParser(
        description="Descriptive analysis of the emic-validity judge's scores."
    )
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-01).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument("--out", default=None, help="Optional path to write the Markdown to.")
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.id / STAGE
    try:
        sources, unreadable = load_source_scores(run_dir / "outputs")
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc

    analysis = build_analysis(sources, unreadable, resolve_scope(run_dir, sources))
    rendered = render_markdown(analysis, args.id)
    # soft_wrap: Rich word-wrap would fold wide table rows and corrupt the Markdown.
    console.print(rendered, markup=False, highlight=False, soft_wrap=True)
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print_success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
