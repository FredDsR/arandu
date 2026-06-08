"""CLI command: ``arandu emic-prepass`` — ordinal emic-validity scoring of approved CEP pairs.

Runs the ``emic_validity`` ordinal criterion over the canonical-approved pairs
of a populated run and writes per-source ordinal scores under
``results/<id>/emic_prepass/outputs/<source>.json``. The scores bound the
sampling bands for the stratified human-comparison sample (the annotators are
the ground truth; this is a sampling aid).
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.emic.batch import run_emic_prepass_batch
from arandu.shared.emic.settings import EmicPrepassSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def emic_prepass(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The cep/ stage must be populated and "
                "judged; only canonical-approved pairs (is_valid) are scored."
            ),
        ),
    ],
    rerun: Annotated[
        bool,
        typer.Option(
            "--rerun/--resume",
            help=(
                "--rerun discards the checkpoint and re-scores every source. "
                "--resume (default) skips sources already completed."
            ),
        ),
    ] = False,
) -> None:
    """Score canonical-approved CEP pairs for emic validity (ordinal 1-5).

    Builds the ``emic_validity`` ordinal criterion standalone (not a judge-qa
    pipeline stage) and runs it over each approved pair's segment + question +
    answer, persisting per-source ``EmicSourceScores`` for the stratified
    sample builder.

    LLM configuration is read from ``ARANDU_EMIC_PREPASS_*`` env vars; see
    :class:`EmicPrepassSettings`. The score is a sampling aid, not ground
    truth — the human annotators are the reference.
    """
    settings = EmicPrepassSettings()
    print_info(f"Run: {pipeline_id}")
    print_info(
        f"Emic LLM: provider={settings.provider}, model={settings.model_id}, "
        f"language={settings.language}, temperature={settings.temperature}"
    )
    if rerun:
        print_warning("--rerun: clearing checkpoint; every source will be re-scored.")

    try:
        result = run_emic_prepass_batch(pipeline_id=pipeline_id, settings=settings, rerun=rerun)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except (RuntimeError, ValueError) as exc:
        print_error(f"Invalid emic-prepass configuration: {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        print_error(f"I/O error during emic pre-pass: {exc}")
        raise typer.Exit(code=1) from exc

    if result.failed_sources:
        print_warning(
            f"{result.failed_sources} source(s) failed to load and were skipped (see logs)."
        )
    if result.unjudged_pairs:
        print_warning(
            f"{result.unjudged_pairs} pair(s) had no judge verdict and were skipped; "
            "the run may not have been fully judged (`arandu judge-qa`)."
        )
    if result.failed_pairs:
        print_warning(f"{result.failed_pairs} approved pair(s) errored while scoring (see logs).")
    print_success(
        f"Scored {result.scored_pairs}/{result.approved_pairs} approved pairs this run "
        f"across {result.completed_sources} new source(s); "
        f"{result.resumed_sources} resumed, {result.failed_sources} failed "
        f"({result.sources} total)."
    )
