#!/usr/bin/env python3
"""Compute structural graph metrics from a knowledge-graph GraphML file.

Loads the GraphML with NetworkX and reports nodes/edges, density,
degree distribution, weakly/strongly connected components, clustering
coefficients, and (with --full) shortest-path metrics on the largest
weakly connected component.

Usage:
    uv run python scripts/kg_structural_metrics.py <graphml_path> [--full] [--json]

Examples:
    uv run python scripts/kg_structural_metrics.py \\
        results/test-kg-04/kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml

Notes:
    - The graph is directed. Density / strongly-connected use the
      directed graph; clustering and shortest-path metrics use the
      undirected projection.
    - --full enables avg shortest path + diameter on the largest WCC,
      which is O(V*E) per node and slow on dense graphs (~minutes for
      ~10k nodes).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arandu.kg.atlas_backend import (
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG,
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN,
)

# All known forms of the synthesized event-participation predicate, across
# the languages we relabel. Used to split edges into atlas-rag-synthesized
# vs LLM-extracted relations regardless of which language the run used.
SYNTHESIZED_PARTICIPATION_PREDICATES: frozenset[str] = frozenset(
    {SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN}
    | set(SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG.values())
)


def _component_stats(g: nx.DiGraph) -> dict[str, int | float]:
    wcc = list(nx.weakly_connected_components(g))
    scc = list(nx.strongly_connected_components(g))
    largest_wcc = max(wcc, key=len) if wcc else set()
    largest_scc = max(scc, key=len) if scc else set()
    return {
        "weakly_connected_components": len(wcc),
        "largest_wcc_size": len(largest_wcc),
        "largest_wcc_fraction": len(largest_wcc) / g.number_of_nodes()
        if g.number_of_nodes()
        else 0.0,
        "strongly_connected_components": len(scc),
        "largest_scc_size": len(largest_scc),
    }


def _degree_stats(g: nx.DiGraph) -> dict[str, float]:
    in_degs = [d for _, d in g.in_degree()]
    out_degs = [d for _, d in g.out_degree()]
    total_degs = [d for _, d in g.degree()]
    return {
        "avg_in_degree": statistics.fmean(in_degs) if in_degs else 0.0,
        "avg_out_degree": statistics.fmean(out_degs) if out_degs else 0.0,
        "avg_total_degree": statistics.fmean(total_degs) if total_degs else 0.0,
        "median_total_degree": float(statistics.median(total_degs)) if total_degs else 0.0,
        "stdev_total_degree": statistics.pstdev(total_degs) if total_degs else 0.0,
        "max_total_degree": float(max(total_degs)) if total_degs else 0.0,
        "min_total_degree": float(min(total_degs)) if total_degs else 0.0,
    }


def _clustering_stats(g_undir: nx.Graph) -> dict[str, float]:
    return {
        "transitivity": nx.transitivity(g_undir),
        "average_clustering": nx.average_clustering(g_undir),
    }


def _edge_type_breakdown(g: nx.DiGraph) -> dict[str, int]:
    """Count edges by atlas-rag's ``type`` attribute.

    Atlas-rag tags edges with one of: ``Relation`` (entity↔entity / event↔
    entity triples), ``Source`` (text-mention edges), ``Concept`` (concept
    linking). Unknown types get bucketed under ``other``.
    """
    counts = {"Relation": 0, "Source": 0, "Concept": 0, "other": 0}
    for _, _, data in g.edges(data=True):
        t = data.get("type")
        counts[t if t in counts else "other"] += 1
    return counts


def _relation_split_stats(g: nx.DiGraph) -> dict[str, int | float]:
    """Within ``type=Relation`` edges, split synthesized vs LLM-extracted.

    Atlas-rag synthesizes ``(Event, "is participated by", Entity)`` edges
    from the LLM's event_entity_relation_dict (no predicate field). Every
    other Relation-typed edge carries an LLM-chosen predicate. Text and
    Concept edges are excluded — they're structural, not semantic.
    """
    synthesized = 0
    extracted = 0
    for _, _, data in g.edges(data=True):
        if data.get("type") != "Relation":
            continue
        rel = data.get("relation")
        if rel in SYNTHESIZED_PARTICIPATION_PREDICATES:
            synthesized += 1
        else:
            extracted += 1

    total = synthesized + extracted
    return {
        "synthesized_participation_edges": synthesized,
        "extracted_relation_edges": extracted,
        "relation_edges_total": total,
        "synthesized_share_of_relations": synthesized / total if total else 0.0,
        "extracted_share_of_relations": extracted / total if total else 0.0,
    }


def _shortest_path_stats_largest_wcc(g: nx.DiGraph) -> dict[str, float]:
    wcc = max(nx.weakly_connected_components(g), key=len)
    sub = g.subgraph(wcc).to_undirected()
    eccentricities = nx.eccentricity(sub)
    diameter = max(eccentricities.values())
    avg_shortest_path = nx.average_shortest_path_length(sub)
    return {
        "largest_wcc_diameter": float(diameter),
        "largest_wcc_avg_shortest_path": avg_shortest_path,
    }


def compute_metrics(graphml: Path, full: bool) -> dict:
    t0 = time.perf_counter()
    g: nx.DiGraph = nx.read_graphml(str(graphml))
    load_seconds = time.perf_counter() - t0

    n = g.number_of_nodes()
    e = g.number_of_edges()
    metrics: dict = {
        "graphml_path": str(graphml),
        "graphml_size_mb": round(graphml.stat().st_size / (1024 * 1024), 2),
        "nodes": n,
        "edges": e,
        "is_directed": g.is_directed(),
        "density": nx.density(g),
        "load_seconds": round(load_seconds, 2),
    }
    metrics.update(_component_stats(g))
    metrics.update(_degree_stats(g))
    metrics.update(_clustering_stats(g.to_undirected()))
    metrics["edges_by_type"] = _edge_type_breakdown(g)
    metrics.update(_relation_split_stats(g))

    if full:
        t0 = time.perf_counter()
        metrics.update(_shortest_path_stats_largest_wcc(g))
        metrics["shortest_path_seconds"] = round(time.perf_counter() - t0, 2)

    return metrics


def render_table(metrics: dict, console: Console) -> None:
    console.print(
        Panel.fit(
            f"[bold]{Path(metrics['graphml_path']).name}[/bold]\n"
            f"{metrics['graphml_size_mb']} MB | "
            f"loaded in {metrics['load_seconds']}s",
            title="Structural Metrics",
        )
    )

    basic = Table(title="Basic", show_header=False)
    basic.add_column(style="bold")
    basic.add_column(justify="right")
    basic.add_row("Nodes", f"{metrics['nodes']:,}")
    basic.add_row("Edges", f"{metrics['edges']:,}")
    basic.add_row("Directed", str(metrics["is_directed"]))
    basic.add_row("Density", f"{metrics['density']:.6f}")
    console.print(basic)

    comp = Table(title="Components", show_header=False)
    comp.add_column(style="bold")
    comp.add_column(justify="right")
    comp.add_row("Weakly connected components", f"{metrics['weakly_connected_components']:,}")
    comp.add_row(
        "Largest WCC size",
        f"{metrics['largest_wcc_size']:,} ({metrics['largest_wcc_fraction'] * 100:.1f}%)",
    )
    comp.add_row("Strongly connected components", f"{metrics['strongly_connected_components']:,}")
    comp.add_row("Largest SCC size", f"{metrics['largest_scc_size']:,}")
    console.print(comp)

    deg = Table(title="Degree distribution", show_header=False)
    deg.add_column(style="bold")
    deg.add_column(justify="right")
    deg.add_row("Avg in-degree", f"{metrics['avg_in_degree']:.2f}")
    deg.add_row("Avg out-degree", f"{metrics['avg_out_degree']:.2f}")
    deg.add_row("Avg total degree", f"{metrics['avg_total_degree']:.2f}")
    deg.add_row("Median total degree", f"{metrics['median_total_degree']:.2f}")
    deg.add_row("Stdev total degree", f"{metrics['stdev_total_degree']:.2f}")
    deg.add_row("Max total degree", f"{metrics['max_total_degree']:.0f}")
    deg.add_row("Min total degree", f"{metrics['min_total_degree']:.0f}")
    console.print(deg)

    clu = Table(title="Clustering (undirected projection)", show_header=False)
    clu.add_column(style="bold")
    clu.add_column(justify="right")
    clu.add_row("Transitivity (global)", f"{metrics['transitivity']:.4f}")
    clu.add_row("Average clustering", f"{metrics['average_clustering']:.4f}")
    console.print(clu)

    edge_types = metrics["edges_by_type"]
    total_edges = metrics["edges"]
    et = Table(title="Edges by type")
    et.add_column("Type", style="bold")
    et.add_column("Edges", justify="right")
    et.add_column("Share", justify="right")
    for label, key in (
        ("Relation (triples)", "Relation"),
        ("Source (text mention)", "Source"),
        ("Concept (concept link)", "Concept"),
        ("other", "other"),
    ):
        count = edge_types[key]
        if count or key != "other":
            share = (count / total_edges * 100) if total_edges else 0.0
            et.add_row(label, f"{count:,}", f"{share:.1f}%")
    console.print(et)

    if metrics["relation_edges_total"]:
        rel = Table(title="Within Relation edges: synthesized vs LLM-extracted")
        rel.add_column("Source", style="bold")
        rel.add_column("Edges", justify="right")
        rel.add_column("Share of Relation", justify="right")
        rel.add_row(
            "atlas-rag synthesized",
            f"{metrics['synthesized_participation_edges']:,}",
            f"{metrics['synthesized_share_of_relations'] * 100:.1f}%",
        )
        rel.add_row(
            "LLM extracted",
            f"{metrics['extracted_relation_edges']:,}",
            f"{metrics['extracted_share_of_relations'] * 100:.1f}%",
        )
        console.print(rel)

    if "largest_wcc_diameter" in metrics:
        sp = Table(title="Shortest paths (largest WCC, undirected)", show_header=False)
        sp.add_column(style="bold")
        sp.add_column(justify="right")
        sp.add_row("Diameter", f"{metrics['largest_wcc_diameter']:.0f}")
        sp.add_row("Avg shortest path", f"{metrics['largest_wcc_avg_shortest_path']:.2f}")
        sp.add_row("Compute time", f"{metrics['shortest_path_seconds']}s")
        console.print(sp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graphml", type=Path, help="Path to the GraphML file")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also compute diameter + avg shortest path on the largest WCC (slow)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Rich tables")
    args = parser.parse_args()

    console = Console()
    if not args.graphml.exists():
        console.print(f"[red]GraphML not found: {args.graphml}[/red]")
        sys.exit(2)

    metrics = compute_metrics(args.graphml, full=args.full)

    if args.json:
        console.print_json(json.dumps(metrics))
    else:
        render_table(metrics, console)


if __name__ == "__main__":
    main()
