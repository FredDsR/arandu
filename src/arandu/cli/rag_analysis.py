"""CLI command: ``arandu rag-analysis`` — emit cross-arm comparison tables.

Aggregates judged :class:`AnswerRecord` artifacts produced by ``arandu
judge-answers`` into per-arm TA/TC/FA/FC counts, Wilson-95% CIs, and
Bloom + question-type cross-cuts; writes ``report.json`` and
``tables.md`` under ``results/<id>/analysis/outputs/``.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.rag.analysis.batch import run_rag_analysis_batch
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def rag_analysis(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The judge_answers/ stage must be "
                "populated. The cep/ stage outputs are consulted for Bloom + "
                "question-type cross-cuts; absence is non-fatal (joint table "
                "only)."
            ),
        ),
    ],
) -> None:
    """Aggregate judged answers and emit ``report.json`` + ``tables.md``.

    Output structure under ``results/<id>/analysis/outputs/``:

    - ``report.json`` — :class:`AnalysisReport` with per-arm
      confusion matrices, Wilson CIs, and stratified breakdowns.
    - ``tables.md`` — human-readable Markdown tables matching the
      thesis chapter's reporting layout.
    """
    print_info(f"Run: {pipeline_id}")
    try:
        result = run_rag_analysis_batch(pipeline_id=pipeline_id)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if result.records_unreadable:
        print_warning(
            f"Skipped {result.records_unreadable} unreadable judged record(s); check logs."
        )
    print_info(f"Arms aggregated: {', '.join(result.arms) or '(none)'}")
    print_info(f"Judged records loaded: {result.records_loaded}")
    print_success(f"Wrote {result.report_json}")
    print_success(f"Wrote {result.tables_md}")
