#!/usr/bin/env python3
"""Generate a human-readable report from a Knowledge Graph pipeline run.

Reads atlas-rag output artifacts (extraction JSONLs, triple CSVs, concept
CSVs, GraphML metadata) and prints a structured summary with Rich tables.

Usage:
    uv run python scripts/kg_report.py <run_dir> [--json]

Arguments:
    run_dir     Path to a pipeline run directory (e.g. results/test-kg-02).
                Must contain kg/outputs/atlas_output/ with the standard
                atlas-rag directory structure.

Options:
    --json      Output raw metrics as JSON instead of Rich tables.
                Useful for piping into jq or storing for later comparison.

Examples:
    # Pretty-print report with Rich tables
    uv run python scripts/kg_report.py results/test-kg-02

    # Export as JSON
    uv run python scripts/kg_report.py results/test-kg-02 --json

    # Compare two runs
    diff <(uv run python scripts/kg_report.py results/test-kg-02 --json) \\
         <(uv run python scripts/kg_report.py results/test-kg-03 --json)

Expected directory layout inside <run_dir>::

    <run_dir>/kg/outputs/atlas_output/
    ├── kg_extraction/          JSONL files from triple extraction
    ├── triples_csv/            Triple/text node and edge CSVs
    │   └── missing_concepts_*  Input for concept generation
    ├── concepts/               Concept shard CSVs
    ├── concept_csv/            Final merged concept CSVs
    └── kg_graphml/             GraphML + metadata JSON

Reported metrics:
    - Graph summary (nodes, edges, density, average degree)
    - Node breakdown by type (entity, event, relation, text)
    - Edge breakdown (triple, text)
    - Top 15 relation predicates with frequency and share
    - Concept generation coverage
    - Extraction file inventory
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class KGReport:
    """Aggregated metrics from a KG pipeline run."""

    run_id: str = ""
    model_id: str = ""
    provider: str = ""
    language: str = ""
    total_documents: int = 0

    # Extraction
    extraction_records: int = 0
    extraction_files: list[tuple[str, int, int]] = field(default_factory=list)

    # Triple nodes / edges
    triple_node_count: int = 0
    triple_node_types: Counter = field(default_factory=Counter)
    triple_edge_count: int = 0
    relation_freq: Counter = field(default_factory=Counter)
    unique_relations: int = 0

    # Text nodes / edges
    text_node_count: int = 0
    text_edge_count: int = 0

    # Concepts
    concept_input_count: int = 0
    concept_output_count: int = 0
    concept_types: Counter = field(default_factory=Counter)
    concept_coverage_pct: float = 0.0

    # Final graph
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_density: float = 0.0
    avg_degree: float = 0.0
    graphml_size_mb: float = 0.0


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _count_csv_rows(path: Path) -> int:
    """Return data-row count (excludes header) for a CSV file."""
    with path.open(newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _read_csv_column(path: Path, col: int) -> list[str]:
    """Return all values from a single CSV column (skipping header)."""
    values: list[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) > col:
                values.append(row[col].strip())
    return values


def _read_metadata(atlas_output: Path) -> dict:
    """Read graph metadata JSON if present."""
    graphml_dir = atlas_output / "kg_graphml"
    if not graphml_dir.exists():
        return {}
    for f in graphml_dir.glob("*.metadata.json"):
        return json.loads(f.read_text())
    return {}


def _find_single_glob(directory: Path, pattern: str) -> Path | None:
    """Return first match for a glob pattern, or None."""
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def collect_report(run_dir: Path) -> KGReport:
    """Collect all metrics from a KG pipeline run directory.

    Args:
        run_dir: Path to the pipeline run (e.g. ``results/test-kg-02``).

    Returns:
        Populated KGReport dataclass.
    """
    report = KGReport(run_id=run_dir.name)
    atlas_output = run_dir / "kg" / "outputs" / "atlas_output"

    if not atlas_output.exists():
        console.print(f"[red]atlas_output not found at {atlas_output}[/red]")
        sys.exit(1)

    # --- Metadata ---
    meta = _read_metadata(atlas_output)
    if meta:
        report.model_id = meta.get("model_id", "")
        report.provider = meta.get("provider", "")
        report.language = meta.get("language", "")
        report.total_documents = meta.get("total_documents", 0)
        report.graph_nodes = meta.get("total_nodes", 0)
        report.graph_edges = meta.get("total_edges", 0)

    # --- Extraction files ---
    kg_ext = atlas_output / "kg_extraction"
    if kg_ext.exists():
        for f in sorted(kg_ext.glob("*.json")):
            size = f.stat().st_size
            lines = [ln for ln in f.read_text().strip().split("\n") if ln.strip()]
            report.extraction_files.append((f.name, len(lines), size))
            report.extraction_records += len(lines)

    # --- Triple nodes ---
    tn = _find_single_glob(atlas_output / "triples_csv", "triple_nodes_*_without_emb.csv")
    if tn:
        report.triple_node_count = _count_csv_rows(tn)
        report.triple_node_types = Counter(_read_csv_column(tn, 1))

    # --- Triple edges ---
    te = _find_single_glob(atlas_output / "triples_csv", "triple_edges_*_without_emb.csv")
    if te:
        report.triple_edge_count = _count_csv_rows(te)
        relations = _read_csv_column(te, 2)
        report.relation_freq = Counter(r.lower() for r in relations)
        report.unique_relations = len(report.relation_freq)

    # --- Text nodes / edges ---
    txn = _find_single_glob(atlas_output / "triples_csv", "text_nodes_*.csv")
    if txn:
        report.text_node_count = _count_csv_rows(txn)

    txe = _find_single_glob(atlas_output / "triples_csv", "text_edges_*.csv")
    if txe:
        report.text_edge_count = _count_csv_rows(txe)

    # --- Concept input ---
    mc = _find_single_glob(atlas_output / "triples_csv", "missing_concepts_*_from_json.csv")
    if mc:
        report.concept_input_count = _count_csv_rows(mc)

    # --- Concept output ---
    shard = atlas_output / "concepts" / "concept_shard_0.csv"
    if shard.exists():
        with shard.open(newline="") as f:
            rows = list(csv.reader(f))
        report.concept_output_count = len(rows)
        report.concept_types = Counter(r[2].strip() for r in rows if len(r) >= 3)

    # --- Coverage ---
    if report.concept_input_count > 0:
        report.concept_coverage_pct = report.concept_output_count / report.concept_input_count * 100

    # --- Graph file ---
    graphml_dir = atlas_output / "kg_graphml"
    if graphml_dir.exists():
        gml = _find_single_glob(graphml_dir, "*.graphml")
        if gml:
            report.graphml_size_mb = gml.stat().st_size / (1024 * 1024)

    # --- Derived metrics ---
    n = report.graph_nodes
    e = report.graph_edges
    if n > 1:
        report.graph_density = (2 * e) / (n * (n - 1))
    if n > 0:
        report.avg_degree = (2 * e) / n

    return report


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_report(report: KGReport) -> None:
    """Render a KGReport to the console using Rich tables and panels."""
    # --- Header ---
    console.print()
    console.print(
        Panel(
            f"[bold]{report.run_id}[/bold]\n"
            f"Model: {report.model_id} ({report.provider})  |  "
            f"Language: {report.language}  |  "
            f"Documents: {report.total_documents}",
            title="KG Pipeline Report",
            border_style="blue",
        )
    )

    # --- Graph summary ---
    graph_table = Table(title="Graph Summary", show_header=False, border_style="cyan")
    graph_table.add_column("Metric", style="bold")
    graph_table.add_column("Value", justify="right")
    graph_table.add_row("Nodes", f"{report.graph_nodes:,}")
    graph_table.add_row("Edges", f"{report.graph_edges:,}")
    graph_table.add_row("Density", f"{report.graph_density:.6f}")
    graph_table.add_row("Average degree", f"{report.avg_degree:.2f}")
    graph_table.add_row("GraphML size", f"{report.graphml_size_mb:.1f} MB")
    console.print(graph_table)

    # --- Node breakdown ---
    node_table = Table(title="Node Breakdown", border_style="green")
    node_table.add_column("Type", style="bold")
    node_table.add_column("Triple Nodes", justify="right")
    node_table.add_column("Concept Nodes", justify="right")
    for ntype in sorted(set(list(report.triple_node_types) + list(report.concept_types))):
        node_table.add_row(
            ntype,
            f"{report.triple_node_types.get(ntype, 0):,}",
            f"{report.concept_types.get(ntype, 0):,}",
        )
    node_table.add_row(
        "text",
        f"{report.text_node_count:,}",
        "—",
        style="dim",
    )
    node_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{report.triple_node_count:,}[/bold]",
        f"[bold]{report.concept_output_count:,}[/bold]",
    )
    console.print(node_table)

    # --- Edge breakdown ---
    edge_table = Table(title="Edge Breakdown", show_header=False, border_style="green")
    edge_table.add_column("Type", style="bold")
    edge_table.add_column("Count", justify="right")
    edge_table.add_row("Triple edges", f"{report.triple_edge_count:,}")
    edge_table.add_row("Text edges", f"{report.text_edge_count:,}")
    console.print(edge_table)

    # --- Relations ---
    title = f"Top 15 Relations ({report.unique_relations:,} unique)"
    rel_table = Table(title=title, border_style="yellow")
    rel_table.add_column("#", justify="right", style="dim")
    rel_table.add_column("Predicate")
    rel_table.add_column("Count", justify="right")
    rel_table.add_column("Share", justify="right")
    for i, (rel, count) in enumerate(report.relation_freq.most_common(15), 1):
        pct = count / report.triple_edge_count * 100 if report.triple_edge_count else 0
        rel_table.add_row(str(i), rel, f"{count:,}", f"{pct:.1f}%")
    console.print(rel_table)

    # --- Concepts ---
    concept_table = Table(title="Concept Generation", show_header=False, border_style="magenta")
    concept_table.add_column("Metric", style="bold")
    concept_table.add_column("Value", justify="right")
    concept_table.add_row("Input nodes", f"{report.concept_input_count:,}")
    concept_table.add_row("Output concepts", f"{report.concept_output_count:,}")
    concept_table.add_row("Coverage", f"{report.concept_coverage_pct:.1f}%")
    concept_table.add_row(
        "Missing", f"{report.concept_input_count - report.concept_output_count:,}"
    )
    console.print(concept_table)

    # --- Extraction files ---
    ext_table = Table(title="Extraction Files", border_style="dim")
    ext_table.add_column("File")
    ext_table.add_column("Records", justify="right")
    ext_table.add_column("Size", justify="right")
    for name, count, size in report.extraction_files:
        ext_table.add_row(name, str(count), f"{size / 1024:.1f} KB")
    ext_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{report.extraction_records}[/bold]",
        "",
    )
    console.print(ext_table)
    console.print()


def dump_json(report: KGReport) -> None:
    """Dump report as JSON to stdout."""
    data = {
        "run_id": report.run_id,
        "model_id": report.model_id,
        "provider": report.provider,
        "language": report.language,
        "total_documents": report.total_documents,
        "graph": {
            "nodes": report.graph_nodes,
            "edges": report.graph_edges,
            "density": report.graph_density,
            "avg_degree": report.avg_degree,
            "graphml_size_mb": round(report.graphml_size_mb, 1),
        },
        "triple_nodes": {
            "total": report.triple_node_count,
            "by_type": dict(report.triple_node_types.most_common()),
        },
        "triple_edges": {
            "total": report.triple_edge_count,
            "unique_relations": report.unique_relations,
            "top_15": [
                {"predicate": r, "count": c} for r, c in report.relation_freq.most_common(15)
            ],
        },
        "text_nodes": report.text_node_count,
        "text_edges": report.text_edge_count,
        "concepts": {
            "input": report.concept_input_count,
            "output": report.concept_output_count,
            "coverage_pct": round(report.concept_coverage_pct, 1),
            "by_type": dict(report.concept_types.most_common()),
        },
        "extraction_records": report.extraction_records,
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a report from a KG pipeline run.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the pipeline run directory (e.g. results/test-kg-02)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of Rich tables",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        console.print(f"[red]Run directory not found: {args.run_dir}[/red]")
        sys.exit(1)

    report = collect_report(args.run_dir)

    if args.json:
        dump_json(report)
    else:
        print_report(report)


if __name__ == "__main__":
    main()
