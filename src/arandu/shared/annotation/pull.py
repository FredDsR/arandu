"""Fetch annotations back and anonymize the annotators (spec §3.3, §8).

Anonymity is structural here rather than by convention. Label Studio identifies
users by email, but the export carries a numeric ``completed_by``, so this
module maps that integer straight to ``A1``/``A2``/``A3`` and never requests an
email from the API at all. The number-to-alias map is written beside the labels,
not inside ``labels/``, so the directory that feeds the agreement analysis holds
nothing identifying.

An unknown ``task_id`` aborts the whole pull. A desynced manifest means the join
is wrong, and silently dropping the row would leave a plausible-looking labels
file that mislabels an unknown share of the sample.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from arandu.shared.annotation.build import MANIFEST_FILENAME
from arandu.shared.annotation.schemas import AnnotationLabel, AnnotationManifest
from arandu.shared.config import ResultsConfig
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.annotation.client import LabelStudioClient

logger = logging.getLogger(__name__)

LABELS_DIRNAME = "labels"
ANNOTATOR_MAP_FILENAME = "annotator_map.json"
SCORE_FIELD = "score"
RATIONALE_FIELD = "rationale"


class PullSummary(BaseModel):
    """Outcome of a pull.

    Attributes:
        project_id: Label Studio project the export came from; ``0`` for a
            file-based pull.
        annotators: Completed-annotation count per anonymous annotator id.
        total_items: Tasks in the project (the denominator for progress).
    """

    project_id: int
    annotators: dict[str, int]
    total_items: int


def anonymize(user_ids: list[int]) -> dict[int, str]:
    """Map Label Studio user ids to stable anonymous aliases.

    Sorted by user id so the mapping is deterministic and reproducible from the
    export alone: the same project always yields the same ``A1``/``A2``/``A3``.

    Args:
        user_ids: Numeric Label Studio user ids, one per completed annotation.

    Returns:
        Mapping from each distinct user id to its anonymous alias.
    """
    return {user_id: f"A{rank}" for rank, user_id in enumerate(sorted(set(user_ids)), start=1)}


def _completed_by_id(annotation: dict[str, Any]) -> int:
    """Return the numeric user id, tolerating the expanded object form.

    Some export shapes inline the user as an object. Only the ``id`` is read; the
    email is deliberately ignored so it cannot reach any artifact.

    Args:
        annotation: One Label Studio annotation payload.

    Returns:
        The numeric ``completed_by`` user id.

    Raises:
        ValueError: If no numeric id can be found.
    """
    raw = annotation.get("completed_by")
    if isinstance(raw, dict):
        raw = raw.get("id")
    if not isinstance(raw, int):
        raise ValueError(
            f"Annotation has no numeric completed_by: {raw!r}. Cannot attribute the label."
        )
    return raw


def _extract_score(annotation: dict[str, Any]) -> int:
    """Read the ordinal score out of the annotation result.

    The option label is ``"<score> - <anchor>"``, so the score is the leading
    integer. A label that does not parse is an error: guessing would silently
    corrupt the measurement the whole study reports.

    Args:
        annotation: One Label Studio annotation payload.

    Returns:
        The ordinal emic-validity score, in ``1..5``.

    Raises:
        ValueError: If the annotation carries no score region, or the choice
            label does not start with an integer.
    """
    for region in annotation.get("result", []):
        if region.get("from_name") != SCORE_FIELD:
            continue
        choices = region.get("value", {}).get("choices") or []
        if not choices:
            break
        head = str(choices[0]).split(" - ", maxsplit=1)[0].strip()
        if not head.isdigit():
            raise ValueError(
                f"Could not read an ordinal score from the choice {choices[0]!r}. The "
                f"labeling config and this parser have diverged."
            )
        return int(head)
    raise ValueError(
        f"Annotation carries no {SCORE_FIELD!r} region. The labeling config and this "
        f"parser have diverged."
    )


def _extract_rationale(annotation: dict[str, Any]) -> str | None:
    """Read the optional free-text justification, or ``None`` if absent.

    Args:
        annotation: One Label Studio annotation payload.

    Returns:
        The rationale text, or ``None`` if the annotator left it blank.
    """
    for region in annotation.get("result", []):
        if region.get("from_name") != RATIONALE_FIELD:
            continue
        texts = region.get("value", {}).get("text") or []
        if texts and str(texts[0]).strip():
            return str(texts[0])
    return None


def _load_export(
    manifest: AnnotationManifest,
    client: LabelStudioClient | None,
    export_file: Path | None,
    pipeline_id: str,
) -> tuple[list[dict[str, Any]], int]:
    """Return the export payload and the project id it came from.

    Args:
        manifest: The build's manifest, used to find the pushed project id.
        client: Label Studio transport. Ignored when ``export_file`` is given.
        export_file: A JSON export downloaded from the UI, if given.
        pipeline_id: Run identifier, used only for error messages.

    Returns:
        A tuple of the raw export payload and the project id it came from
        (``0`` for a file-based pull).

    Raises:
        ValueError: If neither a client nor a file is given, if the file is
            not a JSON list, or if the run was never pushed.
    """
    if export_file is not None:
        data = json.loads(export_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{export_file} is not a Label Studio JSON export (expected a list).")
        return data, 0
    if client is None:
        raise ValueError("Pass either a client or export_file; there is nothing to read from.")
    if manifest.project_id is None:
        raise ValueError(
            f"Run {pipeline_id!r} has no Label Studio project. Run "
            f"`arandu emic-annotation-push --id {pipeline_id}` first, or pass a downloaded "
            f"export with -f."
        )
    return client.export_annotations(manifest.project_id), manifest.project_id


def run_pull_annotation(
    pipeline_id: str,
    *,
    client: LabelStudioClient | None = None,
    export_file: Path | None = None,
    base_dir: Path | None = None,
) -> PullSummary:
    """Pull annotations into per-annotator label files.

    Args:
        pipeline_id: Run identifier with a built (and usually pushed) stage.
        client: Label Studio transport. Ignored when ``export_file`` is given.
        export_file: A JSON export downloaded from the UI. Reading it touches no
            network and needs no credential.
        base_dir: Override the project ``results/`` root.

    Returns:
        A :class:`PullSummary` with per-annotator completion counts.

    Raises:
        FileNotFoundError: If the annotation stage was never built.
        ValueError: If the run was never pushed, no source was given, or a
            score cannot be parsed.
        KeyError: If the export references a task this build does not know. The
            manifest and the project are out of sync and no label from that
            project can be trusted.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    outputs = base / pipeline_id / PipelineType.ANNOTATION.value / "outputs"
    manifest_path = outputs / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Annotation manifest not found for pipeline_id {pipeline_id!r}: {manifest_path}. "
            f"Run `arandu emic-annotation-build --id {pipeline_id} --seed <n>` first."
        )

    manifest = AnnotationManifest.load(manifest_path)
    export, project_id = _load_export(manifest, client, export_file, pipeline_id)

    # Resolve every task_id before writing anything: a desync must not leave a
    # half-written labels directory behind.
    pending: list[tuple[int, AnnotationLabel]] = []
    user_ids: list[int] = []
    for task in export:
        task_id = task.get("data", {}).get("task_id")
        if not isinstance(task_id, int):
            raise ValueError(f"Export task {task.get('id')!r} carries no integer task_id.")
        pair_id = manifest.pair_id_for(task_id)
        for annotation in task.get("annotations", []):
            user_id = _completed_by_id(annotation)
            user_ids.append(user_id)
            pending.append(
                (
                    user_id,
                    AnnotationLabel(
                        pair_id=pair_id,
                        annotator_id="",  # filled once the alias map is known
                        score=_extract_score(annotation),
                        rationale=_extract_rationale(annotation),
                        timestamp=str(annotation.get("created_at", "")),
                    ),
                )
            )

    aliases = anonymize(user_ids)
    by_annotator: dict[str, list[AnnotationLabel]] = {}
    for user_id, label in pending:
        alias = aliases[user_id]
        by_annotator.setdefault(alias, []).append(label.model_copy(update={"annotator_id": alias}))

    labels_dir = outputs / LABELS_DIRNAME
    labels_dir.mkdir(parents=True, exist_ok=True)
    for alias, labels in sorted(by_annotator.items()):
        with (labels_dir / f"{alias}.jsonl").open("w", encoding="utf-8") as fh:
            for label in sorted(labels, key=lambda item: item.pair_id):
                fh.write(label.model_dump_json())
                fh.write("\n")

    # Beside the labels, never inside: `labels/` is what feeds the agreement
    # analysis and must hold nothing that links an alias to a person.
    (outputs / ANNOTATOR_MAP_FILENAME).write_text(
        json.dumps({str(uid): alias for uid, alias in sorted(aliases.items())}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = PullSummary(
        project_id=project_id,
        annotators={alias: len(labels) for alias, labels in sorted(by_annotator.items())},
        total_items=manifest.total_items,
    )
    logger.info(
        "Pulled annotations for %s: %s of %d task(s) per annotator.",
        pipeline_id,
        summary.annotators,
        summary.total_items,
    )
    return summary
