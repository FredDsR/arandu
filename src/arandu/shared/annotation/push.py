"""Create the Label Studio project from the built artifacts (spec §3.2).

The build is the auditable artifact; this step only transports it. Nothing is
computed here that the build did not already write to disk, so what the
annotators see is exactly what an auditor reviewed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from arandu.shared.annotation.build import CONFIG_FILENAME, MANIFEST_FILENAME, TASKS_FILENAME
from arandu.shared.annotation.schemas import AnnotationManifest
from arandu.shared.config import ResultsConfig
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.annotation.client import LabelStudioClient

logger = logging.getLogger(__name__)


def _outputs_dir(base: Path, pipeline_id: str) -> Path:
    """Return the annotation stage's outputs directory for ``pipeline_id``."""
    return base / pipeline_id / PipelineType.ANNOTATION.value / "outputs"


def run_push_annotation(
    pipeline_id: str,
    *,
    client: LabelStudioClient,
    base_dir: Path | None = None,
    title: str | None = None,
    force: bool = False,
) -> int:
    """Create the annotation project and import its tasks.

    Args:
        pipeline_id: Run identifier. The ``annotation`` stage must be built.
        client: Label Studio transport.
        base_dir: Override the project ``results/`` root.
        title: Project title. Defaults to one naming the run.
        force: Create a second project even though one is recorded. Both ids are
            kept in ``project_ids`` so a duplicate is a recorded fact.

    Returns:
        The created project id.

    Raises:
        FileNotFoundError: If the build artifacts are absent.
        ValueError: If a project already exists and ``force`` is false, or the
            task count disagrees with the manifest.
        LabelStudioError: On any API failure.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    outputs = _outputs_dir(base, pipeline_id)
    manifest_path = outputs / MANIFEST_FILENAME
    config_path = outputs / CONFIG_FILENAME
    tasks_path = outputs / TASKS_FILENAME
    if not (manifest_path.exists() and config_path.exists() and tasks_path.exists()):
        raise FileNotFoundError(
            f"Annotation artifacts not found for pipeline_id {pipeline_id!r}: {outputs}. "
            f"Run `arandu emic-annotation-build --id {pipeline_id} --seed <n>` first."
        )

    manifest = AnnotationManifest.load(manifest_path)
    if manifest.project_id is not None and not force:
        raise ValueError(
            f"Run {pipeline_id!r} is already pushed as Label Studio project "
            f"{manifest.project_id}. Pushing again would create a duplicate project and "
            f"split the annotators across two. Use --force only if that is intended."
        )

    tasks: list[dict[str, Any]] = json.loads(tasks_path.read_text(encoding="utf-8"))
    if len(tasks) != manifest.total_items:
        raise ValueError(
            f"{TASKS_FILENAME} holds {len(tasks)} tasks but the manifest declares "
            f"{manifest.total_items}. The build is inconsistent; rebuild before pushing."
        )

    project_title = title or f"Validade êmica ({pipeline_id})"
    project_id = client.create_project(project_title, config_path.read_text(encoding="utf-8"))
    imported = client.import_tasks(project_id, tasks)

    manifest.project_id = project_id
    manifest.project_ids = [*manifest.project_ids, project_id]
    manifest.save(manifest_path)

    logger.info(
        "Pushed %d task(s) to Label Studio project %d (%s).", imported, project_id, project_title
    )
    return project_id
