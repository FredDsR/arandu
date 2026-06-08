"""CLI command: ``arandu build-human-eval-sample`` — stratified 80-pair sample (spec §5).

Builds the human-comparison study sample (4 Bloom x 2 emic bands x 10) from the
emic pre-pass scores of a run, joining each pair's CEP payload, and writes
``sample.jsonl`` + ``sample_manifest.json`` under
``results/<id>/human_eval/outputs/``.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.human_eval.batch import run_build_sample_batch
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def build_human_eval_sample(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The emic_prepass and cep stages must both be populated."
            ),
        ),
    ],
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="RNG seed for the deterministic selection (recorded in the manifest).",
        ),
    ],
) -> None:
    """Build the stratified human-comparison sample (80 pairs) for a run.

    Pools the canonical-approved pairs scored by ``arandu emic-prepass``, bands
    them by emic validity (duvidosa <=3 / limpa >=4), and draws 10 pairs from
    each of the 8 cells (4 Bloom levels x 2 bands) with a fixed seed. The
    emic score is a sampling aid to cover the bands, not ground truth; the
    human annotators remain the reference. Writes the sample + manifest under
    ``results/<id>/human_eval/outputs/``.
    """
    print_info(f"Run: {pipeline_id} | seed: {seed}")

    try:
        manifest = run_build_sample_batch(pipeline_id=pipeline_id, seed=seed)
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
    if manifest.excluded_none_score or excluded_bloom_total:
        print_warning(
            f"Excluded from frame: {manifest.excluded_none_score} null-score pair(s), "
            f"{excluded_bloom_total} out-of-frame-Bloom pair(s) "
            f"({manifest.excluded_bloom or 'none'})."
        )
    print_success(
        f"Built {manifest.total_items} pairs across {len(manifest.cell_counts)} cells "
        f"({manifest.per_cell}/cell). Sample + manifest under "
        f"results/{pipeline_id}/human_eval/outputs/."
    )
