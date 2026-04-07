#!/usr/bin/env python3
"""Smoke test for the shared judge pipeline against real LLM.

Runs QAJudge on a small sample of CEP QA pairs to verify the full
pipeline works end-to-end with a real provider.

Usage:
    # With Gemini (OpenAI-compatible)
    GEMINI_API_KEY=your-key uv run python scripts/test_judge.py

    # With Ollama
    uv run python scripts/test_judge.py --provider ollama --model qwen3:14b

    # Custom sample size
    uv run python scripts/test_judge.py --files 2 --pairs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from arandu.qa.cep.judge import QAJudge
from arandu.qa.config import CEPConfig
from arandu.qa.schemas import QAPairCEP
from arandu.shared.llm_client import create_llm_client

console = Console()

DEFAULT_CEP_DIR = Path("results/test-cep-01/cep/outputs")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Smoke test the judge pipeline.")
    parser.add_argument("--provider", default="custom", help="LLM provider (default: custom)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model ID")
    parser.add_argument("--base-url", default=None, help="Base URL override")
    parser.add_argument("--cep-dir", type=Path, default=DEFAULT_CEP_DIR, help="CEP outputs dir")
    parser.add_argument("--files", type=int, default=2, help="Number of QA files to sample")
    parser.add_argument("--pairs", type=int, default=2, help="Max pairs per file")
    parser.add_argument("--output", type=Path, default=None, help="Save results as JSON")
    args = parser.parse_args()

    # Resolve base URL
    base_url = args.base_url
    if base_url is None and args.provider == "custom":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]Set GEMINI_API_KEY or use --provider ollama[/red]")
            sys.exit(1)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        os.environ["OPENAI_API_KEY"] = api_key

    if args.provider == "ollama":
        base_url = base_url or "http://localhost:11434/v1"

    # Create client and judge
    client = create_llm_client(provider=args.provider, model_id=args.model, base_url=base_url)
    judge = QAJudge(validator_client=client, cep_config=CEPConfig())

    # Sample files
    qa_files = sorted(args.cep_dir.glob("*_cep_qa.json"))[: args.files]
    if not qa_files:
        console.print(f"[red]No QA files found in {args.cep_dir}[/red]")
        sys.exit(1)

    console.print(
        f"\nRunning judge on [bold]{len(qa_files)}[/bold] files, "
        f"up to [bold]{args.pairs}[/bold] pairs each\n"
    )

    # Evaluate
    all_results: list[dict] = []

    for qa_file in qa_files:
        data = json.loads(qa_file.read_text())
        context = data.get("transcription_text") or data.get("context", "")
        all_pairs = data.get("qa_pairs", [])

        # Sample one pair per Bloom level for diversity
        seen_levels: set[str] = set()
        pairs: list[dict] = []
        for p in all_pairs:
            level = p.get("bloom_level", "")
            if level not in seen_levels and len(pairs) < args.pairs:
                pairs.append(p)
                seen_levels.add(level)

        console.print(f"[bold cyan]{qa_file.name}[/bold cyan]")

        for pair_data in pairs:
            qa = QAPairCEP(**pair_data)
            result = judge.validate(qa, context)

            table = Table(show_header=True, border_style="dim", width=90)
            table.add_column("Criterion", width=22)
            table.add_column("Score", justify="right", width=7)
            table.add_column("Threshold", justify="right", width=10)
            table.add_column("Pass", justify="center", width=6)
            table.add_column("Rationale", width=40)

            if result.validation:
                for stage_result in result.validation.stage_results.values():
                    for name, cs in stage_result.criterion_scores.items():
                        if cs.error:
                            table.add_row(
                                name, "—", f"{cs.threshold:.2f}",
                                "[red]ERR[/red]", cs.error[:40],
                            )
                        else:
                            status = "[green]Yes[/green]" if cs.passed else "[red]No[/red]"
                            table.add_row(
                                name,
                                f"{cs.score:.2f}" if cs.score is not None else "—",
                                f"{cs.threshold:.2f}", status,
                                (cs.rationale or "")[:40],
                            )

            console.print(f"  Q: {qa.question[:70]}...")
            console.print(f"  Bloom: {qa.bloom_level}  |  Valid: {result.is_valid}")
            console.print(table)
            console.print()

            all_results.append({
                "file": qa_file.name,
                "question": qa.question,
                "answer": qa.answer,
                "bloom_level": qa.bloom_level,
                "is_valid": result.is_valid,
                "validation": result.validation.model_dump() if result.validation else None,
            })

    # Save JSON results
    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        console.print(f"Results saved to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
