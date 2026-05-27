"""Arandu CLI application.

Main entry point for the Arandu application using Typer for CLI
and Rich for visual feedback.
"""

from __future__ import annotations

from typing import Annotated

import typer

from arandu import __version__
from arandu.utils.console import console
from arandu.utils.logger import setup_logging

# Initialize Typer app
app = typer.Typer(
    name="arandu",
    help="Composable pipelines for ethnographic knowledge elicitation.",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]Arandu[/bold] version {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Print version and exit."""
    setup_logging()


# Register commands from submodules
from arandu.cli.answer import answer  # noqa: E402
from arandu.cli.chunking import chunk  # noqa: E402
from arandu.cli.judge_answers import judge_answers  # noqa: E402
from arandu.cli.kg import build_kg, kg_build_retriever_index, kg_link_passages  # noqa: E402
from arandu.cli.manage import (  # noqa: E402
    enrich_metadata,
    info,
    list_runs,
    rebuild_index,
    refresh_auth,
    replicate,
    run_info,
)
from arandu.cli.non_answerable import generate_non_answerable  # noqa: E402
from arandu.cli.qa import generate_cep_qa, judge_qa  # noqa: E402
from arandu.cli.rag_analysis import rag_analysis  # noqa: E402
from arandu.cli.report import report, serve_report  # noqa: E402
from arandu.cli.retrieve import retrieve  # noqa: E402
from arandu.cli.transcribe import (  # noqa: E402
    batch_transcribe,
    drive_transcribe,
    judge_transcription,
    transcribe,
)

app.command()(chunk)
app.command()(transcribe)
app.command()(drive_transcribe)
app.command()(batch_transcribe)
app.command()(refresh_auth)
app.command()(generate_cep_qa)
app.command()(judge_transcription)
app.command()(judge_qa)
app.command()(build_kg)
app.command()(kg_link_passages)
app.command()(kg_build_retriever_index)
app.command()(retrieve)
app.command()(answer)
app.command()(judge_answers)
app.command()(rag_analysis)
app.command()(generate_non_answerable)
app.command()(replicate)
app.command()(info)
app.command()(list_runs)
app.command()(run_info)
app.command()(rebuild_index)
app.command()(enrich_metadata)
app.command()(report)
app.command()(serve_report)


if __name__ == "__main__":
    app()
