"""Fetch annotations back and anonymize the annotators (spec §3.3, §8).

Anonymity is structural here rather than by convention. Label Studio identifies
users by email, but the export carries a numeric ``completed_by``, so this
module maps that integer straight to ``A1``/``A2``/``A3`` and never requests an
email from the API at all. The number-to-alias map is written beside the labels,
not inside ``labels/``, so the directory that feeds the agreement analysis holds
nothing identifying.

Aliases are anchored, not recomputed: an existing ``annotator_map.json`` is read
back and every binding in it preserved, so a progress pull run week after week
keeps ``A1`` pointing at the same person as somebody submits their first rating.

An unknown ``task_id`` aborts the whole pull. A desynced manifest means the join
is wrong, and silently dropping the row would leave a plausible-looking labels
file that mislabels an unknown share of the sample. A cancelled annotation is
the opposite case: nobody rated that task yet, so it is counted and skipped
rather than treated as a broken export.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from arandu.shared.annotation.build import LABELS_DIRNAME, MANIFEST_FILENAME
from arandu.shared.annotation.schemas import AnnotationLabel, AnnotationManifest
from arandu.shared.config import ResultsConfig
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.annotation.client import LabelStudioClient

logger = logging.getLogger(__name__)

__all__ = [
    "ANNOTATOR_MAP_FILENAME",
    "LABELS_DIRNAME",
    "PullSummary",
    "anonymize",
    "run_pull_annotation",
]

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
        skipped: Annotations the annotator cancelled (Label Studio's Skip
            button) or left with no regions. They carry no rating, so they are
            counted here instead of silently vanishing from the progress
            numbers.
    """

    project_id: int
    annotators: dict[str, int]
    total_items: int
    skipped: int = 0


def _alias_rank(alias: str) -> int:
    """Return the numeric rank in an ``A<n>`` alias, or ``0`` if unparseable."""
    tail = alias[1:]
    return int(tail) if alias.startswith("A") and tail.isdigit() else 0


def anonymize(user_ids: list[int], existing: dict[int, str] | None = None) -> dict[int, str]:
    """Map Label Studio user ids to stable anonymous aliases.

    ``existing`` is preserved verbatim: an alias, once written, belongs to that
    user id forever. Ranking over only the ids present in one export would
    reassign aliases whenever somebody submits their first rating, and a later
    pull would rewrite ``A1.jsonl`` with a different person's labels. Ids not
    already bound are appended in sorted order, so a pull with no prior map is
    still fully determined by the export.

    Args:
        user_ids: Numeric Label Studio user ids, one per completed annotation.
        existing: Bindings recorded by an earlier pull, if any.

    Returns:
        Mapping from each distinct user id to its anonymous alias, containing
        every binding in ``existing`` plus one per newly seen user id.
    """
    aliases = dict(existing or {})
    next_rank = max((_alias_rank(alias) for alias in aliases.values()), default=0) + 1
    for user_id in sorted(set(user_ids) - set(aliases)):
        aliases[user_id] = f"A{next_rank}"
        next_rank += 1
    return aliases


def _load_annotator_map(path: Path) -> dict[int, str]:
    """Read a previously written ``annotator_map.json``, or ``{}`` if absent.

    Args:
        path: Location of the map written beside ``labels/``.

    Returns:
        The recorded id-to-alias bindings, with the JSON string keys parsed back
        to ints.

    Raises:
        ValueError: If the file exists but is not an object of id-to-alias
            entries. Guessing here would break the stability it exists to
            provide.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object of user id to alias entries.")
    try:
        return {int(key): str(value) for key, value in data.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} carries a non-integer user id key; the alias map is unusable and the "
            f"aliases it pins cannot be preserved."
        ) from exc


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


def _is_unrated(annotation: dict[str, Any]) -> bool:
    """Return whether the annotation carries no rating at all.

    Label Studio's Skip button is on by default and writes a real annotation
    with ``was_cancelled: true`` and an empty ``result``. That is a task nobody
    rated, not a broken export, so it must not abort the pull: the labels are
    irrecoverable without reconvening the annotators, and a permanent parse
    failure would block every other annotator's ratings too.

    Args:
        annotation: One Label Studio annotation payload.

    Returns:
        ``True`` if the annotation was cancelled or holds no regions.
    """
    return bool(annotation.get("was_cancelled")) or not annotation.get("result")


def _extract_score(annotation: dict[str, Any]) -> int:
    """Read the ordinal score out of the annotation result.

    The option label is ``"<score> - <anchor>"``, so the score is the leading
    integer. A label that does not parse is an error: guessing would silently
    corrupt the measurement the whole study reports. Callers must filter
    unrated annotations with :func:`_is_unrated` first.

    Args:
        annotation: One Label Studio annotation payload.

    Returns:
        The ordinal emic-validity score, in ``1..5``.

    Raises:
        ValueError: If the annotation carries no score region, if the region is
            present but empty, or if the choice label does not start with an
            integer.
    """
    for region in annotation.get("result", []):
        if region.get("from_name") != SCORE_FIELD:
            continue
        choices = region.get("value", {}).get("choices") or []
        if not choices:
            raise ValueError(
                f"The {SCORE_FIELD!r} region is present but carries no choice. The labeling "
                f"config and this parser have diverged."
            )
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


def _resolve_project_id(
    manifest: AnnotationManifest, project_id: int | None, pipeline_id: str
) -> int:
    """Pick the project to export from, refusing to guess between several.

    A ``--force`` re-push leaves more than one live project recorded, and each
    may hold real ratings. Pulling the newest would silently discard the older
    project's labels; merging both would double-count anyone who annotated in
    each. Neither has a safe default, so the operator chooses.

    Args:
        manifest: The build's manifest, holding every recorded project id.
        project_id: Operator's explicit choice, if given.
        pipeline_id: Run identifier, used only for error messages.

    Returns:
        The project id to export from.

    Raises:
        ValueError: If the run was never pushed, or if several projects are
            recorded and none was named.
    """
    if project_id is not None:
        return project_id
    if manifest.project_id is None:
        raise ValueError(
            f"Run {pipeline_id!r} has no Label Studio project. Run "
            f"`arandu emic-annotation-push --id {pipeline_id}` first, or pass a downloaded "
            f"export with -f."
        )
    if len(manifest.project_ids) > 1:
        recorded = ", ".join(str(pid) for pid in manifest.project_ids)
        raise ValueError(
            f"Run {pipeline_id!r} has more than one Label Studio project recorded: {recorded}. "
            f"Each may hold real ratings, so pulling one of them silently would lose the "
            f"others' labels and merging them would double-count anyone who annotated in "
            f"more than one. Re-run with `--project-id <n>` naming the project to pull."
        )
    return manifest.project_id


def _load_export(
    manifest: AnnotationManifest,
    client: LabelStudioClient | None,
    export_file: Path | None,
    pipeline_id: str,
    project_id: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Return the export payload and the project id it came from.

    Args:
        manifest: The build's manifest, used to find the pushed project id.
        client: Label Studio transport. Ignored when ``export_file`` is given.
        export_file: A JSON export downloaded from the UI, if given.
        pipeline_id: Run identifier, used only for error messages.
        project_id: Explicit project to export from, overriding the manifest.

    Returns:
        A tuple of the raw export payload and the project id it came from
        (``0`` for a file-based pull).

    Raises:
        ValueError: If neither a client nor a file is given, if the file is not
            a JSON list, if the run was never pushed, or if several projects
            are recorded and none was named.
    """
    if export_file is not None:
        data = json.loads(export_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{export_file} is not a Label Studio JSON export (expected a list).")
        return data, 0
    if client is None:
        raise ValueError("Pass either a client or export_file; there is nothing to read from.")
    resolved = _resolve_project_id(manifest, project_id, pipeline_id)
    return client.export_annotations(resolved), resolved


def run_pull_annotation(
    pipeline_id: str,
    *,
    client: LabelStudioClient | None = None,
    export_file: Path | None = None,
    base_dir: Path | None = None,
    project_id: int | None = None,
) -> PullSummary:
    """Pull annotations into per-annotator label files.

    Args:
        pipeline_id: Run identifier with a built (and usually pushed) stage.
        client: Label Studio transport. Ignored when ``export_file`` is given.
        export_file: A JSON export downloaded from the UI. Reading it touches no
            network and needs no credential.
        base_dir: Override the project ``results/`` root.
        project_id: Project to export from, required when a ``--force`` re-push
            left more than one recorded.

    Returns:
        A :class:`PullSummary` with per-annotator completion counts.

    Raises:
        FileNotFoundError: If the annotation stage was never built.
        ValueError: If the run was never pushed, several projects are recorded
            and none was named, no source was given, a score cannot be parsed,
            or one annotator rated the same pair twice.
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
    export, source_project_id = _load_export(manifest, client, export_file, pipeline_id, project_id)

    # Resolve every task_id before writing anything: a desync must not leave a
    # half-written labels directory behind.
    pending: list[tuple[int, AnnotationLabel]] = []
    user_ids: list[int] = []
    skipped = 0
    for task in export:
        task_id = task.get("data", {}).get("task_id")
        if not isinstance(task_id, int):
            raise ValueError(
                f"Export task {task.get('id')!r} carries no integer task_id. This looks like a "
                f"JSON_MIN export, which drops the task data the join needs; download the "
                f"export with exportType=JSON instead."
            )
        pair_id = manifest.pair_id_for(task_id)
        for annotation in task.get("annotations", []):
            if _is_unrated(annotation):
                skipped += 1
                continue
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

    map_path = outputs / ANNOTATOR_MAP_FILENAME
    aliases = anonymize(user_ids, _load_annotator_map(map_path))
    by_annotator: dict[str, list[AnnotationLabel]] = {}
    seen: set[tuple[str, str]] = set()
    for user_id, label in pending:
        alias = aliases[user_id]
        if (alias, label.pair_id) in seen:
            raise ValueError(
                f"Annotator {alias} rated pair {label.pair_id!r} more than once. Two ratings "
                f"of the same pair by the same person cannot both be the measurement; resolve "
                f"the duplicate annotation in Label Studio before pulling."
            )
        seen.add((alias, label.pair_id))
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
    map_path.write_text(
        json.dumps({str(uid): alias for uid, alias in sorted(aliases.items())}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = PullSummary(
        project_id=source_project_id,
        annotators={alias: len(labels) for alias, labels in sorted(by_annotator.items())},
        total_items=manifest.total_items,
        skipped=skipped,
    )
    logger.info(
        "Pulled annotations for %s: %s of %d task(s) per annotator, %d skipped.",
        pipeline_id,
        summary.annotators,
        summary.total_items,
        summary.skipped,
    )
    return summary
