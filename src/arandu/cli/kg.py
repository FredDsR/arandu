"""Knowledge graph construction CLI commands."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from arandu.utils.console import console
from arandu.utils.logger import print_error, print_success


def build_kg(
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
        typer.Option("--output-dir", "-o", help="Output directory for knowledge graph artifacts."),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="LLM provider: openai, ollama, custom.")
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id",
            "-m",
            help="Model ID for KG construction (e.g., llama3.1:8b, qwen3:14b).",
        ),
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="KGC backend: atlas (AutoSchemaKG).")
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language for prompts: 'pt' (Portuguese) or 'en' (English).",
        ),
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
    no_concepts: Annotated[
        bool,
        typer.Option("--no-concepts", help="Skip concept generation (only extract triples)."),
    ] = False,
    backend_option: Annotated[
        list[str] | None,
        typer.Option("--backend-option", help="Backend-specific option as KEY=VALUE (repeatable)."),
    ] = None,
    pipeline_id: Annotated[
        str | None,
        typer.Option("--id", help="Pipeline ID. Auto-resolves transcription outputs."),
    ] = None,
) -> None:
    """Build a knowledge graph from transcription records.

    Extracts entity-relation triples from transcription text and builds a
    corpus-level knowledge graph using the configured backend (default: atlas-rag).

    Examples:
        # Basic KG construction with Ollama
        arandu build-kg results/ --provider ollama -m qwen3:14b -l pt

        # Skip concept generation for faster processing
        arandu build-kg results/ --no-concepts

        # Pass backend-specific options
        arandu build-kg results/ --backend-option chunk_size=4096

        # Use a specific pipeline ID
        arandu build-kg results/ --id my-pipeline
    """
    from arandu.kg.batch import run_batch_kg_construction
    from arandu.kg.config import KGConfig

    # Load config with defaults from environment variables
    kg_config = KGConfig()

    # Build overrides dict from CLI args
    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["provider"] = provider
    if model_id is not None:
        overrides["model_id"] = model_id
    if backend is not None:
        overrides["backend"] = backend
    if ollama_url is not None:
        overrides["ollama_url"] = ollama_url
    if base_url is not None:
        overrides["base_url"] = base_url
    if temperature is not None:
        overrides["temperature"] = temperature
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if language is not None:
        overrides["language"] = language

    # Parse backend-specific options (KEY=VALUE pairs)
    parsed_backend_options: dict[str, Any] = {}
    if backend_option:
        for opt in backend_option:
            if "=" not in opt:
                print_error(f"Invalid backend option format: {opt!r}. Expected KEY=VALUE.")
                raise typer.Exit(code=1)
            key, value = opt.split("=", 1)
            # Try to parse numeric values
            try:
                parsed_backend_options[key.strip()] = int(value.strip())
            except ValueError:
                try:
                    parsed_backend_options[key.strip()] = float(value.strip())
                except ValueError:
                    parsed_backend_options[key.strip()] = value.strip()

    if no_concepts:
        parsed_backend_options["include_concept"] = False

    if parsed_backend_options:
        overrides["backend_options"] = parsed_backend_options

    if overrides:
        try:
            kg_config = KGConfig.model_validate({**kg_config.model_dump(), **overrides})
        except (ValueError, ValidationError) as e:
            print_error(f"Invalid KG configuration: {e}")
            raise typer.Exit(code=1) from e

    # Display configuration
    console.print("\n[bold]Knowledge Graph Construction Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {kg_config.output_dir}")
    console.print(f"[cyan]Backend:[/cyan] {kg_config.backend}")
    console.print(f"[cyan]Provider:[/cyan] {kg_config.provider}")
    console.print(f"[cyan]Model:[/cyan] {kg_config.model_id}")
    console.print(f"[cyan]Language:[/cyan] {kg_config.language}")
    if kg_config.backend_options:
        console.print(f"[cyan]Backend Options:[/cyan] {kg_config.backend_options}")
    if kg_config.provider == "ollama":
        console.print(f"[cyan]Ollama URL:[/cyan] {kg_config.ollama_url}")
    console.print()

    try:
        run_batch_kg_construction(
            input_dir,
            kg_config.output_dir,
            kg_config,
            pipeline_id=pipeline_id,
        )

        print_success("Knowledge graph construction completed!")

    except Exception as e:
        print_error(f"Knowledge graph construction failed: {e}")
        raise typer.Exit(code=1) from e
