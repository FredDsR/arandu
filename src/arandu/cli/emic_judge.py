"""CLI command ``arandu emic-judge``: ordinal emic-validity scoring of CEP pairs.

Runs the ``emic_validity`` ordinal criterion over the canonical-approved pairs
of a populated run and writes per-source ordinal scores under
``results/<id>/emic_judge/outputs/<source>.json``. These scores are the study's
measurement of emic validity; the human annotation round validates them by
agreement (spec §6). The stratified human-comparison sample is drawn
independently from the CEP records; these scores rejoin it at analysis time
by joining on ``pair_id``, not by shaping which pairs are drawn.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.emic.batch import run_emic_judge_batch
from arandu.shared.emic.schemas import EmicScope  # noqa: TC001 (Typer needs runtime access)
from arandu.shared.emic.settings import EmicJudgeSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def emic_judge(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The cep/ stage must be populated, and "
                "judged when --scope approved."
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
    scope: Annotated[
        EmicScope,
        typer.Option(
            "--scope",
            help=(
                "all (default) scores every pair and records its judge-qa verdict, "
                "enabling the emic-validity x approval cross-tabulation. approved "
                "scores only canonically-approved pairs (cheaper; skips rejected "
                "and never-judged ones)."
            ),
        ),
    ] = "all",
) -> None:
    """Score CEP pairs for emic validity (ordinal 1-5).

    Builds the ``emic_validity`` ordinal criterion standalone (not a judge-qa
    pipeline stage) and runs it over each in-scope pair's segment + question +
    answer, persisting per-source ``EmicSourceScores``. Each score carries the
    pair's ``judge-qa`` verdict, so ``--scope all`` supports cross-tabulating
    emic validity against approval.

    The resulting scores are the study's measurement of emic validity. The
    human annotation round (spec §6) rates a stratified subsample and reports
    agreement with them; it validates the measurement rather than replacing it.

    LLM configuration is read from ``ARANDU_EMIC_JUDGE_*`` env vars; see
    :class:`EmicJudgeSettings`. Because the model defines the instrument under
    test, a run feeding the agreement study must pin the model the thesis
    reports.
    """
    # Built inside the try: an out-of-range ARANDU_EMIC_JUDGE_* value raises
    # pydantic.ValidationError (a ValueError), which the handler below turns
    # into a readable message instead of a raw traceback in the SLURM log.
    print_info(f"Run: {pipeline_id} | scope: {scope}")
    if rerun:
        print_warning("--rerun: clearing checkpoint; every source will be re-scored.")

    try:
        settings = EmicJudgeSettings()
        print_info(
            f"Emic LLM: provider={settings.provider}, model={settings.model_id}, "
            f"language={settings.language}, temperature={settings.temperature}, "
            f"workers={settings.workers}"
        )
        if settings.workers > 1:
            print_info(
                f"Scoring {settings.workers} pairs concurrently "
                "(ARANDU_EMIC_JUDGE_WORKERS). Match this with server-side slots "
                "(OLLAMA_NUM_PARALLEL) and the per-slot VRAM budget."
            )
        result = run_emic_judge_batch(
            pipeline_id=pipeline_id, settings=settings, rerun=rerun, scope=scope
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except (RuntimeError, ValueError) as exc:
        # Covers both invalid ARANDU_EMIC_JUDGE_* values (pydantic
        # ValidationError is a ValueError) and the unjudged-corpus preflight.
        print_error(f"Cannot run the emic judge: {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        print_error(f"I/O error during the emic judge run: {exc}")
        raise typer.Exit(code=1) from exc

    if result.failed_sources:
        print_warning(
            f"{result.failed_sources} source(s) failed to load and were skipped (see logs)."
        )
    if result.unjudged_pairs:
        disposition = "scored with a null verdict" if scope == "all" else "skipped"
        print_warning(
            f"{result.unjudged_pairs} pair(s) had no judge verdict and were {disposition}; "
            "the run may not have been fully judged (`arandu judge-qa`)."
        )
    if result.failed_pairs:
        print_warning(f"{result.failed_pairs} pair(s) errored while scoring (see logs).")
    print_success(
        f"Scored {result.scored_pairs}/{result.selected_pairs} selected pairs this run "
        f"across {result.completed_sources} new source(s); "
        f"{result.resumed_sources} resumed, {result.failed_sources} failed "
        f"({result.sources} total)."
    )
    print_info(
        f"Corpus split seen: {result.approved_pairs} approved, "
        f"{result.rejected_pairs} rejected, {result.unjudged_pairs} unjudged "
        f"({result.skipped_pairs} skipped by scope)."
    )
