"""CLI command: ``arandu answer`` — drive the Answerer over a populated retrieve stage.

Iterates every ``RetrievalRecord`` under ``results/<id>/retrieve/outputs/``
and emits one ``AnswerRecord`` per record under
``results/<id>/answers/outputs/<arm>/<source>/``.

The Answerer is held constant across arms — same LLM, same prompt, same
temperature — so the cross-arm benchmark isolates retrieval quality
from generation quality (spec §5.1).
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.rag.answer.batch import run_answer_batch
from arandu.shared.rag.answer.settings import AnswererSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def answer(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The retrieve/ stage must already "
                "be populated; the answerer reads its RetrievalRecord JSON files."
            ),
        ),
    ],
) -> None:
    """Run the Answerer LLM over every RetrievalRecord in a populated run.

    Reads ``results/<id>/retrieve/outputs/<arm>/<source>/*.json`` and
    emits one ``AnswerRecord`` per record under
    ``results/<id>/answers/outputs/<arm>/<source>/`` (same layout,
    different stage).

    Answerer configuration is read from ``ARANDU_ANSWERER_*`` env vars;
    see :class:`AnswererSettings` for fields. The same configuration
    applies to every arm in this run — the methodological constant
    that makes cross-arm comparison fair.
    """
    settings = AnswererSettings()
    print_info(f"Run: {pipeline_id}")
    print_info(
        f"Answerer: provider={settings.provider}, model={settings.model_id}, "
        f"temperature={settings.temperature}, language={settings.language}"
    )

    try:
        result = run_answer_batch(pipeline_id=pipeline_id, settings=settings)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except (RuntimeError, ValueError) as exc:
        print_error(f"Invalid answer configuration: {exc}")
        raise typer.Exit(code=1) from exc

    if result.answers_failed:
        print_warning(
            f"Failed answers: {result.answers_failed} (check logs for per-record errors)."
        )
    if result.answers_resumed:
        print_info(f"Resumed: {result.answers_resumed} record(s) already completed.")
    print_success(f"Wrote {result.answers_written} AnswerRecord(s) to {result.run_dir}/outputs/")
