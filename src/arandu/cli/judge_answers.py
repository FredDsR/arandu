"""CLI command: ``arandu judge-answers`` — run the 4-criterion judge over AnswerRecords.

Reads ``results/<id>/answers/outputs/<arm>/<source>/*.json`` and emits
judged copies under ``results/<id>/judge_answers/outputs/<arm>/<source>/``
each carrying the original AnswerRecord fields plus a populated
``validation`` field with per-criterion scores.

Emits ``abstention_audit.jsonl`` alongside outputs when the answerer's
``abstained`` flag disagrees with the abstention judge's verdict
(spec §6.4 disagreement signal).
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.rag.judge_answers.batch import run_judge_answers_batch
from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def judge_answers(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The answers/, cep/, and chunk/ stages "
                "must already be populated. The judge needs the CEP records for "
                "gold answers + chunks, and the chunk/ + passage_offsets sidecar "
                "for the deterministic offset_coverage criterion."
            ),
        ),
    ],
    rejudge: Annotated[
        bool,
        typer.Option(
            "--rejudge",
            help=(
                "Discard the existing checkpoint and re-run every criterion. "
                "Use after editing prompts or threshold configs."
            ),
        ),
    ] = False,
) -> None:
    """Run the 4-criterion judge over every AnswerRecord in a populated run.

    Persists per-criterion verdicts via `JudgeResultMixin.validation` on
    each `AnswerRecord` copy under `judge_answers/outputs/<arm>/<source>/`.
    The deterministic `offset_coverage` heuristic runs alongside the four
    LLM criteria so analysis can compare semantic vs literal retrieval-
    coverage signals.

    Judge LLM configuration is read from `ARANDU_JUDGE_ANSWERS_*` env
    vars; see :class:`JudgeAnswersSettings` for fields.
    """
    settings = JudgeAnswersSettings()
    print_info(f"Run: {pipeline_id}")
    print_info(
        f"Judge LLM: provider={settings.provider}, model={settings.model_id}, "
        f"language={settings.language}, temperature={settings.temperature}"
    )
    if rejudge:
        print_warning("--rejudge: clearing checkpoint; every record will be re-judged.")

    try:
        result = run_judge_answers_batch(
            pipeline_id=pipeline_id, settings=settings, rejudge=rejudge
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except (RuntimeError, ValueError) as exc:
        print_error(f"Invalid judge_answers configuration: {exc}")
        raise typer.Exit(code=1) from exc

    if result.judgments_failed:
        print_warning(
            f"Failed judgments: {result.judgments_failed} (check logs for per-record errors)."
        )
    if result.judgments_resumed:
        print_info(f"Resumed: {result.judgments_resumed} record(s) already judged.")
    if result.abstention_disagreements:
        print_warning(
            f"Abstention disagreements: {result.abstention_disagreements} "
            f"(see abstention_audit.jsonl under outputs/)."
        )
    print_success(
        f"Wrote {result.judgments_written} judged AnswerRecord(s) to {result.run_dir}/outputs/"
    )
