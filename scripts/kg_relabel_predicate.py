#!/usr/bin/env python3
"""Relabel atlas-rag's hardcoded event-participation predicate in a GraphML.

Equivalent to the in-pipeline sweep
``AtlasRagConstructor._relabel_synthesized_event_participation`` but
applied directly to an already-built GraphML file. Use this on graphs
produced before the in-pipeline relabel landed.

Usage:
    uv run python scripts/kg_relabel_predicate.py <graphml> [--lang pt] [--dry-run]

The original file is moved to ``<graphml>.bak`` before the rewrite. Pass
``--dry-run`` to count occurrences without modifying anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import networkx as nx
from rich.console import Console

from arandu.kg.atlas_backend import (
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG,
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graphml", type=Path, help="Path to the GraphML file")
    parser.add_argument(
        "--lang",
        default="pt",
        help=(
            "Target language whose predicate replaces 'is participated by'. "
            f"Available: {sorted(SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matches without rewriting the file",
    )
    args = parser.parse_args()

    console = Console()

    target = SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG.get(args.lang)
    if not target:
        console.print(
            f"[red]No predicate mapping for language '{args.lang}'. "
            f"Available: {sorted(SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG)}[/red]"
        )
        sys.exit(2)

    if not args.graphml.exists():
        console.print(f"[red]GraphML not found: {args.graphml}[/red]")
        sys.exit(2)

    console.print(f"Loading [bold]{args.graphml}[/bold] ...")
    g: nx.DiGraph = nx.read_graphml(str(args.graphml))

    matches = 0
    for _, _, data in g.edges(data=True):
        if data.get("relation") == SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN:
            if not args.dry_run:
                data["relation"] = target
            matches += 1

    console.print(
        f"Found [bold]{matches:,}[/bold] edges with relation "
        f"'{SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN}' → '{target}'"
    )

    if args.dry_run:
        console.print("[yellow]Dry-run: file not modified[/yellow]")
        return

    if matches == 0:
        console.print("[green]Nothing to rewrite — file untouched[/green]")
        return

    backup = args.graphml.with_suffix(args.graphml.suffix + ".bak")
    console.print(f"Backing up to [bold]{backup}[/bold]")
    shutil.copy2(args.graphml, backup)

    console.print("Writing rewritten GraphML ...")
    nx.write_graphml(g, str(args.graphml), infer_numeric_types=True)
    console.print(f"[green]Done. {matches:,} edges relabeled.[/green]")


if __name__ == "__main__":
    main()
