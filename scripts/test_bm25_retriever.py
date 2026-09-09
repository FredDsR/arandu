#!/usr/bin/env python3
"""Smoke test for the BM25Retriever — builds a real on-disk index and queries it.

Unlike the pytest suite (which uses ``tmp_path`` and discards artifacts), this
script writes ``bm25.pkl`` + ``manifest.json`` to a directory you can inspect
afterwards. Useful for sanity-checking the index layout, the manifest schema,
and the SHA-256 integrity field on a real Portuguese fixture.

Usage:
    uv run python scripts/test_bm25_retriever.py
    uv run python scripts/test_bm25_retriever.py --out /tmp/bm25_demo
    uv run python scripts/test_bm25_retriever.py --language en
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arandu.shared.chunking.resolver import ChunkResolver
from arandu.shared.chunking.schemas import Chunk
from arandu.shared.rag.retrievers.bm25 import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    BM25Retriever,
)

PT_CORPUS = [
    "A enchente de 2024 alagou completamente a cidade de Itaqui.",
    "Maria contou sobre a perda da casa e dos animais.",
    "O rio Uruguai subiu três metros em poucas horas.",
    "A enchente foi terrível e o rio Uruguai transbordou na madrugada.",
    "Joao perdeu o gado e teve que recomeçar a vida em outra cidade.",
]
PT_QUERIES = ["enchente", "Maria animais", "rio Uruguai madrugada"]

EN_CORPUS = [
    "The 2024 flood completely submerged the town of Itaqui.",
    "Maria spoke about losing her house and her animals.",
    "The Uruguay river rose three meters in just a few hours.",
    "The flood was terrible and the Uruguay river overflowed at dawn.",
    "Joao lost his cattle and had to start a new life in another town.",
]
EN_QUERIES = ["flood", "Maria animals", "Uruguay river dawn"]

SOURCE_ID = "src_demo_001"


def build_chunks(sentences: list[str], chunker_id: str) -> tuple[list[Chunk], ChunkResolver]:
    """Chunk-per-sentence over a single joined source document."""
    full_text = " ".join(sentences)
    chunks: list[Chunk] = []
    pos = 0
    for i, sent in enumerate(sentences):
        chunks.append(
            Chunk(
                chunk_id=f"chk_{i:013d}",
                source_file_id=SOURCE_ID,
                chunker_id=chunker_id,
                start_char=pos,
                end_char=pos + len(sent),
            )
        )
        pos += len(sent) + 1  # +1 for the joining space
    resolver = ChunkResolver(text_loader=lambda _fid: full_text)
    return chunks, resolver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=Path("scratch") / "bm25_demo",
        help="Output directory for the index (default: scratch/bm25_demo).",
    )
    p.add_argument(
        "--language", choices=("pt", "en"), default="pt", help="Tokenizer language."
    )
    p.add_argument(
        "--top-k", type=int, default=3, help="Top-K passages to retrieve per query."
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Do not wipe the output directory before building.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    sentences = PT_CORPUS if args.language == "pt" else EN_CORPUS
    queries = PT_QUERIES if args.language == "pt" else EN_QUERIES
    chunker_id = f"bm25_demo_{args.language}"
    index_dir: Path = args.out / chunker_id / "bm25"

    if index_dir.exists() and not args.keep:
        console.print(f"[yellow]Wiping existing {index_dir}[/yellow]")
        shutil.rmtree(index_dir)

    chunks, resolver = build_chunks(sentences, chunker_id)

    console.print(
        Panel.fit(
            f"corpus_size={len(chunks)}  language={args.language}  "
            f"chunker_id={chunker_id}\nindex_dir={index_dir}",
            title="BM25 demo — config",
        )
    )

    console.print("\n[bold cyan]1. Building index[/bold cyan]")
    BM25Retriever.build_index(
        chunks=chunks,
        resolver=resolver,
        index_dir=index_dir,
        chunker_id=chunker_id,
        language=args.language,
    )

    # Show what landed on disk
    files = sorted(index_dir.iterdir())
    files_table = Table(title="Files written")
    files_table.add_column("path", style="cyan")
    files_table.add_column("size (bytes)", style="magenta", justify="right")
    for f in files:
        files_table.add_row(str(f.relative_to(args.out)), str(f.stat().st_size))
    console.print(files_table)

    # Show the manifest
    manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text())
    console.print(
        Panel(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            title=f"{MANIFEST_FILENAME}",
        )
    )

    console.print("\n[bold cyan]2. Loading retriever from disk[/bold cyan]")
    retriever = BM25Retriever(
        index_dir=index_dir, chunker_id=chunker_id, language=args.language
    )
    console.print(
        f"  retriever_id = [bold green]{retriever.retriever_id}[/bold green]"
    )

    console.print("\n[bold cyan]3. Running queries[/bold cyan]")
    for q in queries:
        results = retriever.retrieve(q, top_k=args.top_k)
        table = Table(title=f"query: {q!r}  (top_k={args.top_k})")
        table.add_column("rank", justify="right")
        table.add_column("score", justify="right")
        table.add_column("chunk_id", style="cyan")
        table.add_column("text", style="white")
        for p in results:
            # Walk back to the source text via the chunks list (chunk_id = index suffix).
            idx = int(p.chunk_id.split("_")[-1])
            snippet = sentences[idx]
            table.add_row(
                str(p.rank),
                f"{p.score:.4f}",
                p.chunk_id,
                snippet if len(snippet) < 70 else snippet[:67] + "...",
            )
        console.print(table)

    console.print(
        f"\n[bold green]Done.[/bold green] Inspect artifacts at: [bold]{index_dir}[/bold]"
    )
    console.print(
        f"  Clean up with: [dim]rm -rf {args.out}[/dim]"
    )


if __name__ == "__main__":
    main()
