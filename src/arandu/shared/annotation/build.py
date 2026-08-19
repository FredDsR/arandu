"""Offline builder for the emic annotation instrument (spec §3.1, §5).

This is the auditable half: no network, no secret, fully deterministic. Given a
run's ``human_eval`` sample and the signed-off ruler, it writes the Label Studio
labeling config, the project instructions HTML, the blinded tasks, and the
manifest that holds the ``task_id -> pair_id`` join.

The join stays here because ``pair_id`` is ``"{source_file_id}:{pair_index}"``:
shipping it would let an attentive annotator group pairs from the same interview
and read the stratification off the instrument.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from arandu.shared.annotation.labeling_config import (
    render_expert_instruction,
    render_labeling_config,
)
from arandu.shared.annotation.ruler import load_ruler, require_signed_off, ruler_sha256
from arandu.shared.annotation.schemas import (
    AnnotationBuildConfig,
    AnnotationManifest,
    AnnotationTask,
)
from arandu.shared.config import ResultsConfig
from arandu.shared.human_eval.batch import MANIFEST_FILENAME as SAMPLE_MANIFEST_FILENAME
from arandu.shared.human_eval.batch import SAMPLE_FILENAME
from arandu.shared.human_eval.schemas import SampleItem, SampleManifest
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "labeling_config.xml"
#: The full ruler, rendered for Label Studio's project instructions modal.
#:
#: A second artifact rather than a second thing for `push` to render: the build
#: is the auditable half, and both annotator-facing surfaces have to be readable
#: from disk before anything is created server-side.
INSTRUCTION_FILENAME = "expert_instruction.html"
TASKS_FILENAME = "tasks.json"
MANIFEST_FILENAME = "manifest.json"
LABELS_DIRNAME = "labels"


def _shuffle_key(seed: int, pair_id: str) -> str:
    """Return the seeded ordering key for a pair.

    Same construction as the sampler's selection key: a SHA-256 of
    ``"{seed}:{pair_id}"`` is stable across Python versions and platforms and
    depends only on the pair itself, so the order is reproducible from the
    manifest alone.
    """
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def shuffle_order(pair_ids: list[str], seed: int) -> list[str]:
    """Return ``pair_ids`` in the deterministic presentation order.

    The sample arrives grouped by stratification cell. Presenting it that way
    would let the annotator calibrate within a block of same-band pairs, so the
    order is broken here, at the source, with a recorded seed. Label Studio's own
    shuffle is per-annotator, random, and not recordable, which is why it is not
    used.
    """
    return sorted(pair_ids, key=lambda pid: (_shuffle_key(seed, pid), pid))


def _load_sample(path: Path) -> list[SampleItem]:
    """Read ``sample.jsonl`` into sample items."""
    with path.open(encoding="utf-8") as fh:
        return [SampleItem.model_validate_json(line) for line in fh if line.strip()]


def run_build_annotation(
    pipeline_id: str,
    *,
    seed: int,
    base_dir: Path | None = None,
    ruler_path: Path | None = None,
) -> AnnotationManifest:
    """Build the annotation artifacts for ``pipeline_id``.

    Args:
        pipeline_id: Run identifier. The ``human_eval`` stage must be populated.
        seed: Shuffle seed, recorded in the manifest.
        base_dir: Override the project ``results/`` root.
        ruler_path: Override the ruler location (tests and audits).

    Returns:
        The :class:`AnnotationManifest` describing the build.

    Raises:
        RulerNotSignedOffError: If the ruler gate is still open.
        FileNotFoundError: If the sample or its manifest is absent.
        ValueError: If the sample count disagrees with the sample manifest, if
            the sample repeats a ``pair_id``, or if a previous build of this run
            has already been pushed or already has pulled labels.
    """
    # Gate first: nothing is written while the anchors are unreviewed.
    ruler = load_ruler(ruler_path)
    require_signed_off(ruler)

    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    sample_outputs = base / pipeline_id / PipelineType.HUMAN_EVAL.value / "outputs"
    sample_path = sample_outputs / SAMPLE_FILENAME
    sample_manifest_path = sample_outputs / SAMPLE_MANIFEST_FILENAME
    if not sample_path.exists() or not sample_manifest_path.exists():
        raise FileNotFoundError(
            f"Human-eval sample not found for pipeline_id {pipeline_id!r}: {sample_path}. "
            f"Run `arandu build-human-eval-sample --id {pipeline_id} --seed <n>` first."
        )

    annotation_outputs = base / pipeline_id / PipelineType.ANNOTATION.value / "outputs"
    existing_manifest_path = annotation_outputs / MANIFEST_FILENAME
    if existing_manifest_path.exists():
        existing = AnnotationManifest.load(existing_manifest_path)
        if existing.project_id is not None:
            raise ValueError(
                f"Run {pipeline_id!r} was already pushed as Label Studio project "
                f"{existing.project_id}. Rebuilding would rewrite the task_id -> pair_id join "
                f"while annotators work against the old one, silently mislabelling every "
                f"pull. Create a new run id instead."
            )

    # The push guard above keys on project_id, which stays None on the file-mode
    # pull path (a project created by hand in the UI, then `emic-annotation-pull
    # -f`). Pulled labels are the other half of the same join, so their presence
    # forbids a rebuild just as a live project does.
    existing_labels = sorted((annotation_outputs / LABELS_DIRNAME).glob("*.jsonl"))
    if existing_labels:
        raise ValueError(
            f"Run {pipeline_id!r} already has pulled labels "
            f"({', '.join(path.name for path in existing_labels)}). Rebuilding would rewrite "
            f"the task_id -> pair_id join those labels were resolved through, silently "
            f"invalidating every pair_id in them with nothing marking the divergence. "
            f"Create a new run id instead."
        )

    sample_manifest = SampleManifest.load(sample_manifest_path)
    items = _load_sample(sample_path)
    if len(items) != sample_manifest.total_items:
        raise ValueError(
            f"{SAMPLE_FILENAME} holds {len(items)} pairs but {SAMPLE_MANIFEST_FILENAME} "
            f"declares {sample_manifest.total_items}. The sample stage is inconsistent; "
            f"rebuild it before annotating."
        )

    # Keying by pair_id collapses duplicates, so the count has to be re-checked
    # after the dict is built: without this, a repeated pair_id would silently
    # drop a pair and the reduced count would be recorded as authoritative.
    by_pair_id = {item.pair_id: item for item in items}
    if len(by_pair_id) != sample_manifest.total_items:
        raise ValueError(
            f"{SAMPLE_FILENAME} holds {len(items)} pairs but only {len(by_pair_id)} distinct "
            f"pair_id(s); {SAMPLE_MANIFEST_FILENAME} declares "
            f"{sample_manifest.total_items}. A duplicated pair_id would silently shrink the "
            f"instrument; rebuild the sample before annotating."
        )

    ordered_pair_ids = shuffle_order(list(by_pair_id), seed=seed)

    tasks: list[AnnotationTask] = []
    task_map: dict[str, str] = {}
    for task_id, pair_id in enumerate(ordered_pair_ids):
        item = by_pair_id[pair_id]
        tasks.append(
            AnnotationTask(
                task_id=task_id,
                segment=item.segment,
                question=item.question,
                answer=item.answer,
            )
        )
        task_map[str(task_id)] = pair_id

    results_mgr = ResultsManager(base, PipelineType.ANNOTATION, pipeline_id=pipeline_id)
    results_mgr.create_run(
        AnnotationBuildConfig(seed=seed, total_items=len(tasks)),
        input_source=str(sample_outputs),
    )

    outputs = results_mgr.outputs_dir
    (outputs / CONFIG_FILENAME).write_text(render_labeling_config(ruler), encoding="utf-8")
    (outputs / INSTRUCTION_FILENAME).write_text(render_expert_instruction(ruler), encoding="utf-8")
    (outputs / TASKS_FILENAME).write_text(
        json.dumps(
            [task.to_label_studio() for task in tasks],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = AnnotationManifest(
        pipeline_id=pipeline_id,
        seed=seed,
        total_items=len(tasks),
        per_cell=sample_manifest.per_cell,
        pool_sha256=sample_manifest.pool_sha256,
        ruler_sha256=ruler_sha256(ruler_path),
        task_map=task_map,
    )
    manifest.save(outputs / MANIFEST_FILENAME)

    results_mgr.update_progress(len(tasks), 0, len(tasks))
    results_mgr.complete_run(success=True)

    logger.info(
        "Built annotation instrument: %d tasks, seed=%d, ruler=%s.",
        len(tasks),
        seed,
        manifest.ruler_sha256[:12],
    )
    return manifest
