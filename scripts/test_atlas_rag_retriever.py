#!/usr/bin/env python3
"""Smoke test for the AtlasRagRetriever — builds + queries against a real KG.

Two modes:

- `--provider gemini` (default) — Gemini for BOTH the LLMGenerator (NER during
  retrieval) AND the sentence encoder (via Gemini's OpenAI-compatible embeddings
  endpoint). Designed for local iteration: small fixed cost, no GPU needed,
  reasonable wall time (~minutes for `test-kg-04`).
- `--provider ollama` — atlas-rag's LLMGenerator goes through ollama; sentence
  encoder uses local sentence-transformers (CUDA-friendly). Intended for the
  PCAD end-to-end run.

The KG must already exist on disk (e.g. `results/test-kg-04/kg/outputs/atlas_output/`
produced by a prior `arandu build-kg`). If the `precompute/` subdir is missing
(or `--rebuild-index` is passed), Phase 1 runs first.

Usage:
    # Gemini (default; both LLM + embeddings via Gemini)
    GEMINI_API_KEY=... uv run --extra kg python scripts/test_atlas_rag_retriever.py
    GEMINI_API_KEY=... uv run --extra kg python scripts/test_atlas_rag_retriever.py \\
        --kg-dir results/test-kg-04/kg/outputs/atlas_output \\
        --llm-model gemini-2.5-flash \\
        --embedding-model text-embedding-004 \\
        --rebuild-index

    # Ollama (intended for PCAD)
    uv run --extra kg python scripts/test_atlas_rag_retriever.py \\
        --provider ollama \\
        --llm-base-url http://localhost:11434/v1 \\
        --llm-model qwen3:14b \\
        --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arandu.shared.embeddings import GeminiEmbedder
from arandu.shared.rag.retrievers.atlas_rag import (
    PRECOMPUTE_DIR_NAME,
    AtlasRagRetriever,
)

# A few canned PT questions covering the kinds of recall the CEP benchmark uses.
SMOKE_QUESTIONS: list[str] = [
    "Em que ano ocorreu a enchente que afetou Itaqui?",
    "Quem participou da entrevista em Barra de Pelotas em 30-07-2025?",
    "Qual foi a primeira coisa que Maria perdeu na enchente?",
    "Como o rio Uruguai se comportou durante a enchente de 2024?",
    "Onde fica a estrada que foi reconstruída segundo D. Silvia?",
]


def make_sentence_transformers_encoder(model_name: str) -> object:
    """Fallback encoder for the ollama/PCAD path.

    Imports atlas-rag lazily so the Gemini-only run path doesn't pay the
    sentence-transformers / atlas-rag import cost.
    """
    from atlas_rag.vectorstore.embedding_model import SentenceTransformerEmbeddingModel

    return SentenceTransformerEmbeddingModel(model_name)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--kg-dir",
        type=Path,
        default=Path("results/test-kg-04/kg/outputs/atlas_output"),
        help="atlas-rag KG outputs dir (default: test-kg-04).",
    )
    p.add_argument(
        "--provider",
        choices=("gemini", "ollama"),
        default="gemini",
        help="LLM backend for the NER step during retrieval.",
    )
    p.add_argument("--llm-model", default="gemini-2.5-flash", help="LLM model id.")
    p.add_argument(
        "--llm-base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="OpenAI-compatible base URL (Gemini default; pass ollama URL on PCAD).",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-004",
        help="Embedding model id. Gemini: `text-embedding-004` / `gemini-embedding-001`. "
        "Ollama mode falls back to sentence-transformers using this string.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Top-K passages to retrieve per query.")
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Run `build_index` even if `precompute/` is already populated.",
    )
    p.add_argument(
        "--keyword",
        default="transcriptions.json",
        help="atlas-rag filename pattern (matches existing test-kg-04).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]GEMINI_API_KEY not set.[/red]")
            raise SystemExit(2)
    else:
        api_key = os.environ.get("ARANDU_OLLAMA_API_KEY", "ollama")

    llm_client = OpenAI(api_key=api_key, base_url=args.llm_base_url)

    # Embedder: Gemini for `--provider gemini`, sentence-transformers otherwise.
    # The embedder client gets a generous max_retries since the openai SDK
    # honours Gemini's Retry-After header on 429s; this lets the pacing layer
    # be a soft hint without us hand-coding backoff.
    if args.provider == "gemini":
        embedder_client = OpenAI(api_key=api_key, base_url=args.llm_base_url, max_retries=8)
        encoder = GeminiEmbedder(embedder_client, args.embedding_model)
    else:
        encoder = make_sentence_transformers_encoder(args.embedding_model)

    console.print(
        Panel.fit(
            f"provider={args.provider}  llm_model={args.llm_model}  "
            f"embedding_model={args.embedding_model}\n"
            f"kg_dir={args.kg_dir}",
            title="AtlasRagRetriever smoke",
        )
    )

    precompute_dir = args.kg_dir / PRECOMPUTE_DIR_NAME
    needs_build = args.rebuild_index or not (precompute_dir / "manifest.json").exists()

    if needs_build:
        console.print(
            "\n[bold cyan]Phase 1: build_index[/bold cyan] (slow — encoding nodes/edges/texts)"
        )
        t0 = time.perf_counter()
        AtlasRagRetriever.build_index(
            kg_outputs_dir=args.kg_dir,
            sentence_encoder=encoder,
            sentence_encoder_model=args.embedding_model,
            keyword=args.keyword,
            include_events=True,
            include_concept=True,
        )
        console.print(f"  done in {time.perf_counter() - t0:.1f}s")
    else:
        console.print(
            "\n[cyan]Skipping Phase 1 — `precompute/manifest.json` already present "
            "(use --rebuild-index to force).[/cyan]"
        )

    console.print("\n[bold cyan]Phase 2: instantiate retriever[/bold cyan]")
    t0 = time.perf_counter()
    retriever = AtlasRagRetriever(
        kg_outputs_dir=args.kg_dir,
        llm_client=llm_client,
        llm_model_id=args.llm_model,
        sentence_encoder=encoder,
        sentence_encoder_model=args.embedding_model,
        keyword=args.keyword,
    )
    # Disable atlas-rag's LLM-based edge filtering for the local smoke. The
    # filter calls the LLM to keep only "relevant" edges; Gemini's output
    # format/parsing seems to differ enough from atlas-rag's expectation that
    # the filter returns nothing, producing an empty personalization dict and
    # a PageRank ZeroDivisionError. Raw edge similarity is good enough for
    # this smoke; revisit when running on PCAD with ollama if needed.
    retriever._inner.inference_config.is_filter_edges = False
    console.print(
        f"  retriever_id={retriever.retriever_id} (init in {time.perf_counter() - t0:.1f}s)"
    )

    console.print("\n[bold cyan]Phase 3: retrieve[/bold cyan]")
    for q in SMOKE_QUESTIONS:
        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        table = Table(title=f"{q!r}  (top_k={args.top_k}, {elapsed_ms:.0f} ms)")
        table.add_column("rank", justify="right")
        table.add_column("score", justify="right")
        table.add_column("passage_id (chunk_id)", style="cyan")
        for p in results:
            table.add_row(str(p.rank), f"{p.score:.4f}", p.chunk_id)
        console.print(table)

    console.print(
        f"\n[bold green]Done.[/bold green] Precompute lives at: [bold]{precompute_dir}[/bold]"
    )


if __name__ == "__main__":
    main()
