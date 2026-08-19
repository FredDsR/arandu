"""CLI command: ``arandu build-human-eval-sample`` -- Bloom-stratified sample (spec §5).

Builds the human-comparison study sample (4 Bloom levels x ``--per-cell``) from a
run's CEP records and writes ``sample.jsonl`` + ``sample_manifest.json`` under
``results/<id>/human_eval/outputs/``. The emic-judge stage is not read.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.human_eval.batch import run_build_sample_batch
from arandu.shared.human_eval.sampling import FRAME_BLOOM_LEVELS, PER_CELL
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def build_human_eval_sample(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help="Pipeline ID for the run. The cep stage must be populated and judged.",
        ),
    ],
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="RNG seed for the deterministic selection (recorded in the manifest).",
        ),
    ],
    per_cell: Annotated[
        int,
        typer.Option(
            "--per-cell",
            min=1,
            help="Pairs drawn per Bloom cell. 4 cells, so the total is 4x this.",
        ),
    ] = PER_CELL,
) -> None:
    """Build the Bloom-stratified human-comparison sample for a run.

    Pools the pairs ``judge-qa`` approved from the run's CEP records, groups them
    by Bloom level, and draws ``--per-cell`` from each of the four levels with a
    fixed seed. The sample is what the human annotators rate, so agreement with
    the emic judge's measurement can be reported (spec §6); the judge's scores
    join back at analysis time on ``pair_id``, which is why this command does not
    read the emic_judge stage and does not wait for that run.
    """
    total = per_cell * len(FRAME_BLOOM_LEVELS)
    print_info(f"Run: {pipeline_id} | seed: {seed} | {per_cell}/cell x 4 cells = {total} pairs")

    try:
        manifest = run_build_sample_batch(pipeline_id=pipeline_id, seed=seed, per_cell=per_cell)
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        print_error(f"Could not build a balanced sample: {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        print_error(f"I/O error building the human-eval sample: {exc}")
        raise typer.Exit(code=1) from exc

    excluded_bloom_total = sum(manifest.excluded_bloom.values())
    if excluded_bloom_total or manifest.excluded_not_approved:
        print_warning(
            f"Excluded from frame: {manifest.excluded_not_approved} not judge-approved "
            f"pair(s), {excluded_bloom_total} out-of-frame-Bloom pair(s) "
            f"({manifest.excluded_bloom or 'none'})."
        )
    print_success(
        f"Built {manifest.total_items} pairs across {len(manifest.cell_counts)} Bloom cells "
        f"({manifest.per_cell}/cell). Sample + manifest under "
        f"results/{pipeline_id}/human_eval/outputs/."
    )
