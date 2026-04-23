#!/usr/bin/env python3
"""Smoke test for the TranscriptionJudge heuristic pipeline.

Runs TranscriptionJudge on a sample of transcription files to verify the
heuristic filter pipeline (script_match, repetition, content_density,
segment_quality) works end-to-end. No LLM needed.

Usage:
    uv run python scripts/test_transcription_judge.py
    uv run python scripts/test_transcription_judge.py \\
        --input-dir results/test-kg-03/transcription/outputs
    uv run python scripts/test_transcription_judge.py --files 5 --language pt
    uv run python scripts/test_transcription_judge.py --output judgements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from arandu.shared.schemas import EnrichedRecord
from arandu.transcription.judge import TranscriptionJudge

console = Console()

DEFAULT_INPUT_DIR = Path("results/test-kg-03/transcription/outputs")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Smoke test TranscriptionJudge on sample files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_transcription.json files",
    )
    parser.add_argument(
        "--files", type=int, default=3, help="Number of transcription files to sample"
    )
    parser.add_argument("--language", default="pt", help="Expected transcription language")
    parser.add_argument("--output", type=Path, default=None, help="Save results as JSON")
    args = parser.parse_args()

    json_files = sorted(args.input_dir.glob("*_transcription.json"))[: args.files]
    if not json_files:
        console.print(f"[red]No transcription files found in {args.input_dir}[/red]")
        sys.exit(1)

    console.print(
        f"\nRunning judge on [bold]{len(json_files)}[/bold] files "
        f"(language=[bold]{args.language}[/bold])\n"
    )

    judge = TranscriptionJudge(language=args.language)

    all_results: list[dict] = []
    pass_counter: Counter[str] = Counter()
    fail_counter: Counter[str] = Counter()

    for json_path in json_files:
        data = json.loads(json_path.read_text())
        record = EnrichedRecord(**data)

        result = judge.evaluate_transcription(
            text=record.transcription_text,
            duration_ms=record.duration_milliseconds,
            segments=record.segments or [],
        )

        table = Table(show_header=True, border_style="dim", width=90)
        table.add_column("Criterion", width=22)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Threshold", justify="right", width=10)
        table.add_column("Pass", justify="center", width=6)
        table.add_column("Rationale", width=40)

        criteria_dump: dict[str, dict] = {}

        for stage_result in result.stage_results.values():
            for name, cs in stage_result.criterion_scores.items():
                criteria_dump[name] = {
                    "score": cs.score,
                    "threshold": cs.threshold,
                    "passed": cs.passed,
                    "error": cs.error,
                    "rationale": cs.rationale,
                }
                if cs.error:
                    table.add_row(
                        name,
                        "—",
                        f"{cs.threshold:.2f}",
                        "[red]ERR[/red]",
                        cs.error[:40],
                    )
                    fail_counter[name] += 1
                else:
                    status = "[green]Yes[/green]" if cs.passed else "[red]No[/red]"
                    table.add_row(
                        name,
                        f"{cs.score:.2f}" if cs.score is not None else "—",
                        f"{cs.threshold:.2f}",
                        status,
                        (cs.rationale or "")[:40],
                    )
                    if cs.passed:
                        pass_counter[name] += 1
                    else:
                        fail_counter[name] += 1

        preview = record.transcription_text[:70].replace("\n", " ")
        console.print(f"[bold cyan]{json_path.name}[/bold cyan]")
        console.print(f"  Text: {preview}...")
        console.print(
            f"  Duration: {record.duration_milliseconds}ms  |  "
            f"Segments: {len(record.segments or [])}  |  "
            f"Passed: {result.passed}"
        )
        console.print(table)
        console.print()

        all_results.append(
            {
                "file": json_path.name,
                "file_id": record.file_id,
                "duration_ms": record.duration_milliseconds,
                "num_segments": len(record.segments or []),
                "passed": result.passed,
                "rejected_at": result.rejected_at,
                "criteria": criteria_dump,
            }
        )

    summary = Table(
        title="Summary (per criterion)",
        show_header=True,
        border_style="dim",
        width=60,
    )
    summary.add_column("Criterion", width=24)
    summary.add_column("Pass", justify="right", width=8)
    summary.add_column("Fail", justify="right", width=8)
    summary.add_column("Rate", justify="right", width=10)

    all_criteria = sorted(set(pass_counter) | set(fail_counter))
    for name in all_criteria:
        p = pass_counter[name]
        f = fail_counter[name]
        total = p + f
        rate = f"{100 * p / total:.0f}%" if total else "—"
        summary.add_row(name, str(p), str(f), rate)
    console.print(summary)

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        console.print(f"\nResults saved to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
