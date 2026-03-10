"""QA generation CLI commands."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated, Any

import typer

from arandu.utils.console import console
from arandu.utils.logger import print_error, print_success, print_warning


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
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Enable LLM-as-a-Judge validation. Default: enabled",
        ),
    ] = True,
    validator_model: Annotated[
        str | None,
        typer.Option("--validator-model", help="Model ID for LLM-as-a-Judge validation."),
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
    - Module III: LLM-as-a-Judge Validation (optional quality evaluation)

    Questions are distributed across Bloom taxonomy levels (remember, understand,
    analyze, evaluate) to create cognitively scaffolded QA datasets.

    Examples:
        # Basic CEP generation (default: Portuguese, validation enabled)
        arandu generate-cep-qa results/ -o cep_dataset/

        # Disable validation for faster processing
        arandu generate-cep-qa results/ --no-validate

        # Use custom validator model
        arandu generate-cep-qa results/ --validator-model gpt-4

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
    if not validate:
        cep_overrides["enable_validation"] = False
    if validator_model is not None:
        cep_overrides["validator_model_id"] = validator_model

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
    console.print(f"[cyan]Validation Enabled:[/cyan] {cep_config.enable_validation}")
    if cep_config.enable_validation:
        console.print(f"[cyan]Validator Model:[/cyan] {cep_config.validator_model_id}")
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
