"""CLI commands for the emic annotation instrument (spec §3).

Three flat commands, matching the project convention (the CLI has no
``add_typer`` anywhere):

- ``emic-annotation-build``: offline, deterministic, no secret.
- ``emic-annotation-push``: create the Label Studio project.
- ``emic-annotation-pull``: fetch the annotations back.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.annotation.build import run_build_annotation
from arandu.shared.annotation.ruler import RulerNotSignedOffError
from arandu.utils.logger import print_error, print_info, print_success

logger = logging.getLogger(__name__)


def emic_annotation_build(
    pipeline_id: Annotated[
        str,
        typer.Option("--id", help="Pipeline ID. The human_eval stage must be populated."),
    ],
    seed: Annotated[
        int,
        typer.Option("--seed", help="Shuffle seed for the presentation order."),
    ],
) -> None:
    """Build the Label Studio artifacts for a run's human-eval sample.

    Offline and deterministic: reads ``sample.jsonl`` and the signed-off emic
    ruler, and writes ``labeling_config.xml``, ``tasks.json`` and
    ``manifest.json`` under ``results/<id>/annotation/outputs/``. No network and
    no credential is involved, so the artifacts can be audited for blinding and
    anchor fidelity before anything reaches an annotator.
    """
    print_info(f"Run: {pipeline_id} | seed: {seed}")

    try:
        manifest = run_build_annotation(pipeline_id=pipeline_id, seed=seed)
    except RulerNotSignedOffError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        print_error(f"Could not build the annotation instrument: {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        print_error(f"I/O error building the annotation instrument: {exc}")
        raise typer.Exit(code=1) from exc

    print_success(
        f"Built {manifest.total_items} blinded tasks (seed {manifest.seed}). "
        f"Artifacts under results/{pipeline_id}/annotation/outputs/."
    )
