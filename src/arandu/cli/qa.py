"""QA generation CLI commands."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated, Any

import typer
from rich.table import Table

from arandu.utils.console import console
from arandu.utils.logger import print_error, print_info, print_success, print_warning


def generate_cep_qa(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing transcription JSON files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Output directory for CEP QA dataset JSON files."),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="LLM provider: openai, ollama, custom.")
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id", "-m", help="Model ID for QA generation (e.g., llama3.1:8b, gpt-4)."
        ),
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", "-w", help="Number of parallel workers.")
    ] = None,
    questions: Annotated[
        int | None,
        typer.Option("--questions", help="Number of QA pairs to generate per document (1-50)."),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="LLM temperature for generation (0.0-2.0)."),
    ] = None,
    ollama_url: Annotated[
        str | None, typer.Option("--ollama-url", help="Ollama API base URL.")
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Custom base URL for OpenAI-compatible endpoints."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language for prompts: 'pt' (Portuguese) or 'en' (English). Default: pt",
        ),
    ] = None,
    bloom_dist: Annotated[
        str | None,
        typer.Option(
            "--bloom-dist",
            help="Bloom level distribution as 'level:weight,...' "
            "(e.g., 'remember:0.2,understand:0.3,analyze:0.3,evaluate:0.2').",
        ),
    ] = None,
    export_jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl/--no-jsonl", help="Also export QA pairs to JSONL format for KGQA training."
        ),
    ] = False,
    pipeline_id: Annotated[
        str | None,
        typer.Option("--id", help="Pipeline ID. Auto-resolves transcription outputs."),
    ] = None,
) -> None:
    """Generate CEP (cognitive scaffolding) QA pairs from transcriptions.

    Uses the Cognitive Elicitation Pipeline (CEP) with:
    - Module I: Bloom Scaffolding (question generation by cognitive level)
    - Module II: Reasoning & Grounding (reasoning traces and multi-hop detection)

    Questions are distributed across Bloom taxonomy levels (remember, understand,
    analyze, evaluate) to create cognitively scaffolded QA datasets.

    Use ``arandu judge-qa`` to evaluate generated pairs with LLM-as-a-Judge.

    Examples:
        # Basic CEP generation (default: Portuguese)
        arandu generate-cep-qa results/ -o cep_dataset/

        # Adjust Bloom level distribution
        arandu generate-cep-qa results/ \\
            --bloom-dist "remember:0.1,understand:0.3,analyze:0.4,evaluate:0.2"

        # Export to JSONL for KGQA training
        arandu generate-cep-qa results/ --jsonl
    """
    from arandu.qa.batch import run_batch_cep_generation
    from arandu.qa.config import CEPConfig, QAConfig

    # Load configs with defaults from environment variables
    qa_config = QAConfig()
    cep_config = CEPConfig()

    # Build QA config overrides dict for CLI args
    qa_overrides: dict[str, Any] = {}
    if provider is not None:
        qa_overrides["provider"] = provider
    if model_id is not None:
        qa_overrides["model_id"] = model_id
    if ollama_url is not None:
        qa_overrides["ollama_url"] = ollama_url
    if base_url is not None:
        qa_overrides["base_url"] = base_url
    if questions is not None:
        qa_overrides["questions_per_document"] = questions
    if temperature is not None:
        qa_overrides["temperature"] = temperature
    if output_dir is not None:
        qa_overrides["output_dir"] = output_dir
    if workers is not None:
        qa_overrides["workers"] = workers

    if qa_overrides:
        qa_config = QAConfig.model_validate({**qa_config.model_dump(), **qa_overrides})

    # Build CEP config overrides dict for CLI args
    cep_overrides: dict[str, Any] = {}
    if language is not None:
        cep_overrides["language"] = language

    # Parse Bloom distribution if provided
    if bloom_dist is not None:
        try:
            dist_dict = {}
            for item in bloom_dist.split(","):
                level, weight = item.strip().split(":")
                dist_dict[level.strip()] = float(weight.strip())
            cep_overrides["bloom_distribution"] = dist_dict
            cep_overrides["bloom_levels"] = list(dist_dict.keys())
        except ValueError as e:
            print_error(f"Invalid bloom-dist format: {e}")
            print_error("Expected format: 'level:weight,level:weight,...'")
            raise typer.Exit(code=1) from e

    if cep_overrides:
        try:
            cep_config = CEPConfig.model_validate({**cep_config.model_dump(), **cep_overrides})
        except ValueError as e:
            print_error(f"Invalid CEP configuration: {e}")
            raise typer.Exit(code=1) from e

    # Validate configs
    if qa_config.workers < 1:
        print_error("Number of workers must be at least 1")
        raise typer.Exit(code=1)

    if qa_config.questions_per_document < 1 or qa_config.questions_per_document > 50:
        print_error("Number of questions must be between 1 and 50")
        raise typer.Exit(code=1)

    valid_languages = {"en", "pt"}
    if cep_config.language not in valid_languages:
        print_error(
            f"Invalid language: {cep_config.language!r}. Must be one of {sorted(valid_languages)}"
        )
        raise typer.Exit(code=1)

    # Display configuration
    console.print("\n[bold]CEP QA Generation Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {qa_config.output_dir}")
    console.print(f"[cyan]Provider:[/cyan] {qa_config.provider}")
    console.print(f"[cyan]Model:[/cyan] {qa_config.model_id}")
    console.print(f"[cyan]Workers:[/cyan] {qa_config.workers}")
    console.print(f"[cyan]Questions per document:[/cyan] {qa_config.questions_per_document}")
    console.print(f"[cyan]Language:[/cyan] {cep_config.language}")
    console.print(f"[cyan]Bloom Levels:[/cyan] {', '.join(cep_config.bloom_levels)}")
    console.print(f"[cyan]Bloom Distribution:[/cyan] {cep_config.bloom_distribution}")
    console.print(f"[cyan]Reasoning Traces:[/cyan] {cep_config.enable_reasoning_traces}")
    console.print(f"[cyan]Export JSONL:[/cyan] {export_jsonl}")
    if qa_config.provider == "ollama":
        console.print(f"[cyan]Ollama URL:[/cyan] {qa_config.ollama_url}")
    console.print()

    try:
        run_batch_cep_generation(
            input_dir,
            qa_config.output_dir,
            qa_config,
            cep_config,
            qa_config.workers,
            pipeline_id=pipeline_id,
        )

        # Export to JSONL if requested
        if export_jsonl:
            from arandu.qa.schemas import QARecordCEP

            console.print("\n[cyan]Exporting to JSONL format...[/cyan]")
            for json_file in qa_config.output_dir.glob("*_cep_qa.json"):
                try:
                    record = QARecordCEP.load(json_file)
                    jsonl_file = json_file.with_suffix(".jsonl")
                    record.to_jsonl(jsonl_file)
                    console.print(f"  Exported: {jsonl_file.name}")
                except Exception as e:
                    print_warning(f"Failed to export {json_file.name}: {e}")

        print_success("CEP QA generation completed!")

    except Exception as e:
        print_error(f"CEP QA generation failed: {e}")
        raise typer.Exit(code=1) from e


def judge_qa(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing *_cep_qa.json files to judge.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=(
                "LLM provider: openai, ollama, custom. Falls back to "
                "ARANDU_JUDGE_VALIDATOR_PROVIDER, then inferred from "
                "ARANDU_LLM_BASE_URL (custom when set, else ollama)."
            ),
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model ID for judge evaluation. Falls back to "
                "ARANDU_JUDGE_VALIDATOR_MODEL when not provided."
            ),
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help=(
                "Custom base URL for OpenAI-compatible endpoints. Falls back to "
                "ARANDU_JUDGE_VALIDATOR_BASE_URL, then ARANDU_LLM_BASE_URL."
            ),
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language for judge prompts (pt or en)."),
    ] = "pt",
    files: Annotated[
        int | None,
        typer.Option("--files", help="Maximum number of QA files to sample."),
    ] = None,
    pairs: Annotated[
        int | None,
        typer.Option("--pairs", help="Maximum QA pairs to judge per file."),
    ] = None,
    rejudge: Annotated[
        bool,
        typer.Option(
            "--rejudge/--resume",
            help=(
                "--rejudge re-evaluates every sampled pair from scratch. "
                "--resume (default) skips pairs that already carry a "
                "``validation`` payload, so a re-submission after a wall "
                "hit only judges the unjudged remainder."
            ),
        ),
    ] = False,
) -> None:
    """Judge CEP QA pairs using LLM-as-a-Judge evaluation.

    Evaluates sampled pairs on faithfulness, Bloom calibration, informativeness,
    and self-containedness using the QAJudge pipeline. Each judged pair is
    persisted back into its ``*_cep_qa.json`` file by populating the pair's
    ``validation`` field. ``is_valid`` is derived from
    ``validation.passed`` automatically. No aggregate side-file is
    produced — run a downstream analytics script for cross-record reports.

    Validator model and provider come from ``--model`` / ``--provider`` /
    ``--base-url`` when supplied, otherwise from
    ``ARANDU_JUDGE_VALIDATOR_MODEL`` / ``ARANDU_JUDGE_VALIDATOR_PROVIDER``
    / ``ARANDU_JUDGE_VALIDATOR_BASE_URL`` (with ``ARANDU_LLM_BASE_URL`` as
    a final fallback for custom endpoints), matching ``judge-transcription``.

    By default the command resumes: any pair whose ``validation`` is already
    populated is skipped. Pass ``--rejudge`` to force a fresh pass over
    every sampled pair (e.g. after changing the validator model).

    Examples:
        # Defaults from ARANDU_JUDGE_* env vars
        arandu judge-qa cep_dataset/

        # Explicit Ollama
        arandu judge-qa cep_dataset/ --provider ollama --model qwen3:14b

        # OpenAI-compatible custom endpoint
        arandu judge-qa cep_dataset/ --provider custom --model gemini-2.5-flash \\
            --base-url https://generativelanguage.googleapis.com/v1beta/openai/

        arandu judge-qa cep_dataset/ --files 2 --pairs 3
        arandu judge-qa cep_dataset/ --rejudge
    """
    from arandu.qa.cep.judge import QAJudge
    from arandu.qa.config import CEPConfig, get_judge_config
    from arandu.qa.schemas import QAPairCEP, QARecordCEP
    from arandu.transcription.judge import build_validator_client

    judge_config = get_judge_config()
    resolved_model = model or judge_config.validator_model
    if not resolved_model:
        print_error(
            "Validator model is required. Pass --model or set "
            "ARANDU_JUDGE_VALIDATOR_MODEL in your environment / .env."
        )
        raise typer.Exit(code=1)

    try:
        client = build_validator_client(
            model_id=resolved_model,
            provider=provider or judge_config.validator_provider,
            base_url=base_url or judge_config.validator_base_url,
        )
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    cep_config = CEPConfig(language=language)
    judge = QAJudge(validator_client=client, cep_config=cep_config)

    # Find QA files
    qa_files = sorted(input_dir.glob("*_cep_qa.json"))
    if not qa_files:
        print_error(f"No CEP QA files found in {input_dir}")
        raise typer.Exit(code=1)

    if files is not None:
        qa_files = qa_files[:files]

    mode_label = "rejudge" if rejudge else "resume"
    print_info(
        f"Judging [bold]{len(qa_files)}[/bold] QA files (mode: {mode_label})"
        + (f", up to [bold]{pairs}[/bold] pairs each" if pairs else "")
    )
    console.print()

    total_valid = 0
    total_judged = 0
    total_skipped = 0

    for qa_file in qa_files:
        try:
            record = QARecordCEP.model_validate_json(qa_file.read_text())
        except Exception as e:
            print_error(f"Failed to read {qa_file.name}: {e}")
            continue

        context = record.transcription_text
        all_pairs = record.qa_pairs

        # Sample diverse pairs by Bloom level first, then fill remaining slots
        seen_levels: set[str] = set()
        sampled_indices: list[int] = []
        for i, p in enumerate(all_pairs):
            if pairs is not None and len(sampled_indices) >= pairs:
                break
            if p.bloom_level not in seen_levels:
                sampled_indices.append(i)
                seen_levels.add(p.bloom_level)
        if pairs is not None:
            for i in range(len(all_pairs)):
                if len(sampled_indices) >= pairs:
                    break
                if i not in sampled_indices:
                    sampled_indices.append(i)
        elif pairs is None:
            sampled_indices = list(range(len(all_pairs)))

        console.print(f"[bold cyan]{qa_file.name}[/bold cyan]")

        updated_pairs: list[QAPairCEP] = list(all_pairs)
        file_judged = 0
        file_valid = 0
        file_skipped = 0

        for idx in sampled_indices:
            qa = updated_pairs[idx]
            if not rejudge and qa.validation is not None:
                # Resume mode — pair already carries a verdict.
                if qa.is_valid:
                    file_valid += 1
                    total_valid += 1
                file_skipped += 1
                total_skipped += 1
                continue
            try:
                validated = judge.validate(qa, context)
                updated_pairs[idx] = validated
                file_judged += 1
                total_judged += 1
                if validated.is_valid:
                    file_valid += 1
                    total_valid += 1
                _render_qa_verdict(validated)
            except Exception as e:
                print_warning(f"Failed to judge pair in {qa_file.name}: {e}")
                continue

        # Persist updated record back to disk. ``resolved_model`` (not the
        # raw CLI option) is the actual model the judge ran with —
        # ``model`` may be None when the value came from
        # ARANDU_JUDGE_VALIDATOR_MODEL, and writing None would clobber an
        # existing validator_model_id on the record.
        record.qa_pairs = updated_pairs
        if resolved_model is not None:
            record.validator_model_id = resolved_model
        # Count pairs that *passed* validation, not just pairs that have a
        # verdict — the schema field is documented as "Number of pairs
        # passing validation" and validation_rate is computed off it.
        record.validated_pairs = sum(1 for p in record.qa_pairs if p.is_valid)
        qa_file.write_text(record.model_dump_json(indent=2, by_alias=True))

        console.print(
            f"  [dim]{qa_file.name}: judged {file_judged}, "
            f"resumed (skipped) {file_skipped}, valid {file_valid}, "
            f"persisted {record.validated_pairs}/{len(record.qa_pairs)} validated[/dim]"
        )
        console.print()

    console.print(f"[bold]Total pairs judged:[/bold] {total_judged}")
    console.print(f"[green]Valid:[/green] {total_valid}")
    console.print(f"[red]Invalid:[/red] {total_judged + total_skipped - total_valid}")
    if total_skipped:
        console.print(f"[dim]Resumed (already judged, skipped):[/dim] {total_skipped}")
    console.print()


def _render_qa_verdict(validated: Any) -> None:
    """Render a single validated QA pair's verdict to the console."""
    table = Table(show_header=True, border_style="dim", width=90)
    table.add_column("Criterion", width=22)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Pass", justify="center", width=6)
    table.add_column("Rationale", width=40)

    if validated.validation:
        for stage_result in validated.validation.stage_results.values():
            for name, cs in stage_result.criterion_scores.items():
                if cs.error:
                    table.add_row(
                        name, "\u2014", f"{cs.threshold:.2f}", "[red]ERR[/red]", cs.error[:40]
                    )
                else:
                    status = "[green]Yes[/green]" if cs.passed else "[red]No[/red]"
                    table.add_row(
                        name,
                        f"{cs.score:.2f}" if cs.score is not None else "\u2014",
                        f"{cs.threshold:.2f}",
                        status,
                        (cs.rationale or "")[:40],
                    )

    console.print(f"  Q: {validated.question[:70]}...")
    console.print(f"  Bloom: {validated.bloom_level}  |  Valid: {validated.is_valid}")
    console.print(table)
    console.print()
