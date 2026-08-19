"""CLI commands for the emic annotation instrument (spec §3).

Three flat commands, matching the project convention (the CLI has no
``add_typer`` anywhere):

- ``emic-annotation-build``: offline, deterministic, no secret.
- ``emic-annotation-push``: create the Label Studio project.
- ``emic-annotation-pull``: fetch the annotations back.
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

from arandu.shared.annotation.build import run_build_annotation
from arandu.shared.annotation.client import LabelStudioError, build_client_from_settings
from arandu.shared.annotation.pull import run_pull_annotation
from arandu.shared.annotation.push import run_push_annotation
from arandu.shared.annotation.ruler import RULER_PATH, RulerNotSignedOffError
from arandu.shared.annotation.settings import LabelStudioSettings
from arandu.utils.logger import print_error, print_info, print_success, print_warning

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
    ruler, and writes ``labeling_config.xml``, ``expert_instruction.html``,
    ``tasks.json`` and ``manifest.json`` under
    ``results/<id>/annotation/outputs/``. No network and no credential is
    involved, so the artifacts can be audited for blinding and anchor fidelity
    before anything reaches an annotator.
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
    except KeyError as exc:
        # render_labeling_config indexes the ruler directly, so a ruler missing
        # a section surfaces here. A raw traceback would say nothing about which
        # file to fix.
        print_error(
            f"The emic ruler is missing a required key ({exc}). The labeling config cannot be "
            f"rendered without it. Ruler: {RULER_PATH}"
        )
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


def emic_annotation_push(
    pipeline_id: Annotated[
        str,
        typer.Option("--id", help="Pipeline ID with a built annotation stage."),
    ],
    title: Annotated[
        str | None,
        typer.Option("--title", help="Project title. Defaults to one naming the run."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Create another project even though one is recorded."),
    ] = False,
) -> None:
    """Create the Label Studio project for a built annotation stage.

    Uploads only what ``emic-annotation-build`` already wrote to disk, so the
    annotators see exactly the artifacts an auditor can review. Reads the
    instance URL and token from ``ARANDU_LABEL_STUDIO_URL`` and
    ``ARANDU_LABEL_STUDIO_TOKEN``.
    """
    try:
        settings = LabelStudioSettings()
    except ValueError as exc:
        print_error(
            "Label Studio is not configured. Set ARANDU_LABEL_STUDIO_URL and "
            f"ARANDU_LABEL_STUDIO_TOKEN in your environment or .env file. ({exc})"
        )
        raise typer.Exit(code=1) from exc

    print_info(f"Run: {pipeline_id} | instance: {settings.url}")
    client = build_client_from_settings(settings)

    try:
        project_id = run_push_annotation(
            pipeline_id=pipeline_id, client=client, title=title, force=force
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        print_error(f"Could not push the annotation project: {exc}")
        raise typer.Exit(code=1) from exc
    except LabelStudioError as exc:
        print_error(f"Label Studio rejected the push: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    print_success(
        f"Created project {project_id} at {settings.url}/projects/{project_id}. "
        f"Invite the annotators and they can start."
    )


def emic_annotation_pull(
    pipeline_id: Annotated[
        str,
        typer.Option("--id", help="Pipeline ID with a pushed annotation stage."),
    ],
    export_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Read a JSON export downloaded from the UI instead of using the network.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    project_id: Annotated[
        int | None,
        typer.Option(
            "--project-id",
            help=(
                "Project to pull from. Required when a --force re-push recorded several. "
                "Not usable together with -f."
            ),
        ),
    ] = None,
) -> None:
    """Pull the annotations into per-annotator label files.

    Writes ``results/<id>/annotation/outputs/labels/<annotator_id>.jsonl`` with
    anonymous ids (``A1``, ``A2``, ``A3``). No email is ever requested from the
    API or written to any artifact. With ``-f`` the export is read from disk and
    no credential is needed.

    ``labels/`` is left holding exactly what this pull wrote: files from an
    earlier, wider pull that this export does not cover are removed and
    reported, so the agreement analysis never reads a mix of two pulls.
    """
    client = None
    settings = None
    if export_file is None:
        try:
            settings = LabelStudioSettings()
        except ValueError as exc:
            print_error(
                "Label Studio is not configured. Set ARANDU_LABEL_STUDIO_URL and "
                f"ARANDU_LABEL_STUDIO_TOKEN, or pass -f with a downloaded export. ({exc})"
            )
            raise typer.Exit(code=1) from exc
        client = build_client_from_settings(settings)

    source = str(export_file) if export_file is not None else (settings.url if settings else "")
    print_info(f"Run: {pipeline_id} | source: {source}")

    try:
        summary = run_pull_annotation(
            pipeline_id=pipeline_id,
            client=client,
            export_file=export_file,
            project_id=project_id,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except KeyError as exc:
        print_error(f"Manifest and Label Studio project are out of sync: {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        print_error(f"Could not pull the annotations: {exc}")
        raise typer.Exit(code=1) from exc
    except LabelStudioError as exc:
        print_error(f"Label Studio rejected the export request: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if client is not None:
            client.close()

    if summary.stale_removed:
        print_warning(
            f"{summary.stale_removed} label file(s) from an earlier pull were removed: this "
            f"export did not cover them. labels/ now holds only what this pull wrote."
        )
    if summary.skipped:
        print_warning(
            f"{summary.skipped} annotation(s) were skipped in Label Studio and carry no "
            f"rating. They are not counted below and those pairs still need rating."
        )
    if not summary.annotators:
        print_info(f"No annotations yet ({summary.total_items} task(s) waiting).")
        return
    progress = ", ".join(
        f"{alias}: {done}/{summary.total_items}" for alias, done in summary.annotators.items()
    )
    print_success(f"Pulled annotations. {progress}")
