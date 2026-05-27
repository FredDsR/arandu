"""CLI command: ``arandu generate-non-answerable`` - build the non-answerable benchmark.

Samples validated CEP pairs, perturbs each into a non-answerable twin via
one LLM entity-swap call + code-side verification, and writes a
``NonAnswerableDataset`` under ``results/<id>/non_answerable/outputs/dataset.json``.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.qa.non_answerable.batch import run_generate_non_answerable_batch
from arandu.qa.non_answerable.settings import NonAnswerableSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def generate_non_answerable(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The cep/ stage must be populated "
                "(post judge-qa); kg/ and transcription/ outputs are consulted "
                "for the entity absence checks."
            ),
        ),
    ],
    seeds_per_bloom: Annotated[
        int | None,
        typer.Option("--seeds-per-bloom", help="Target seeds per Bloom level. Default 100."),
    ] = None,
    rng_seed: Annotated[
        int | None,
        typer.Option("--rng-seed", help="Sampler seed for reproducibility. Default 42."),
    ] = None,
    regenerate: Annotated[
        bool,
        typer.Option(
            "--regenerate",
            help="Discard checkpoint + prior items and re-perturb every seed.",
        ),
    ] = False,
) -> None:
    """Generate a non-answerable benchmark from validated CEP + KG.

    Each item keeps a ``parent_qa_pair_id`` link to its answerable twin so
    the analysis stage can run paired comparisons. Generation is fully
    automated (no hand-curation); the LLM proposes a same-type entity swap
    and code verifies the replacement is absent from both the KG and the
    source corpus.

    LLM + sampling config comes from ``ARANDU_NONANSWERABLE_*`` env vars
    (see :class:`NonAnswerableSettings`); ``--seeds-per-bloom`` / ``--rng-seed``
    override the corresponding settings for this run.
    """
    settings = NonAnswerableSettings()
    if seeds_per_bloom is not None:
        settings.seeds_per_bloom = seeds_per_bloom
    if rng_seed is not None:
        settings.rng_seed = rng_seed

    print_info(f"Run: {pipeline_id}")
    print_info(
        f"Perturbation LLM: provider={settings.provider}, model={settings.model_id}, "
        f"language={settings.language}"
    )
    print_info(
        f"Sampling: {settings.seeds_per_bloom}/Bloom level, rng_seed={settings.rng_seed}, "
        f"retries={settings.retry_max}"
    )
    if regenerate:
        print_warning("--regenerate: clearing checkpoint; every seed will be re-perturbed.")

    try:
        result = run_generate_non_answerable_batch(
            pipeline_id=pipeline_id, settings=settings, regenerate=regenerate
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except (RuntimeError, ValueError) as exc:
        print_error(f"Invalid non_answerable configuration: {exc}")
        raise typer.Exit(code=1) from exc

    if result.seeds_skipped_resumed:
        print_info(f"Resumed: {result.seeds_skipped_resumed} seed(s) already processed.")
    if result.seeds_failed:
        print_warning(
            f"Skipped {result.seeds_failed} seed(s) with no valid swap after retries "
            f"(no perturbable entity, or all candidates collided)."
        )
    if result.items_built:
        print_info(f"Newly perturbed this run: {result.items_built} item(s).")
    print_info(
        f"Dataset: {result.dataset_items} item(s) from {result.seed_count} seed(s) "
        f"(success rate {result.success_rate:.1%})."
    )
    print_success(f"Wrote {result.dataset_path}")
