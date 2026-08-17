"""Schemas for the emic annotation instrument (spec §4, §5, §8).

The blinding contract of spec §5 is enforced here rather than downstream:
:class:`AnnotationTask` forbids extra fields, so an attempt to smuggle
``bloom_level`` or ``pair_id`` into the payload is a validation error at
construction, not a silent leak into a live project.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AnnotationBuildConfig(BaseModel):
    """Config snapshotted into ``run_metadata.json`` for an annotation build.

    Attributes:
        seed: Shuffle seed (recorded so the task order is reproducible).
        total_items: Number of tasks built.
    """

    seed: int
    total_items: int


class AnnotationTask(BaseModel):
    """One blinded task as Label Studio receives it.

    These four fields are the complete payload. ``pair_id`` is deliberately
    absent: it is ``"{source_file_id}:{pair_index}"``, so shipping it would let
    an attentive annotator group pairs from the same interview and infer the
    stratification. The join lives in :class:`AnnotationManifest`.

    Attributes:
        task_id: Opaque 0-based index into the shuffled order.
        segment: Source transcript segment.
        question: The generated question.
        answer: The generated answer.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(..., ge=0)
    segment: str
    question: str
    answer: str

    def to_label_studio(self) -> dict[str, dict[str, object]]:
        """Return the Label Studio import shape (``{"data": {...}}``)."""
        return {"data": self.model_dump()}


class AnnotationManifest(BaseModel):
    """Provenance, reproducibility, and the ``task_id -> pair_id`` join.

    Attributes:
        pipeline_id: Run identifier.
        seed: Shuffle seed.
        total_items: Number of tasks built.
        per_cell: Pairs per stratification cell, copied from the sample manifest
            for cross-checking.
        pool_sha256: Copied from the sample manifest (ties this build to the
            exact sample it came from).
        ruler_sha256: Hash of the ruler the labeling config was rendered from.
            A ruler edit after the push would otherwise be invisible.
        task_map: ``str(task_id) -> pair_id``. JSON object keys are strings, so
            the int is stringified on the way out and parsed on the way back.
        project_id: Label Studio project created by ``push``; ``None`` until
            then.
        project_ids: Every project ever created from this build, in order. A
            ``--force`` re-push appends rather than overwriting, so a duplicate
            project is a recorded fact instead of a lost one.
    """

    pipeline_id: str
    seed: int
    total_items: int
    per_cell: int
    pool_sha256: str
    ruler_sha256: str
    task_map: dict[str, str]
    project_id: int | None = None
    project_ids: list[int] = Field(default_factory=list)

    def pair_id_for(self, task_id: int) -> str:
        """Resolve a ``task_id`` back to its canonical ``pair_id``.

        Raises:
            KeyError: If the task is not in this build. That means the manifest
                and the Label Studio project are out of sync, which invalidates
                every label pulled from it.
        """
        key = str(task_id)
        if key not in self.task_map:
            raise KeyError(
                f"task_id {task_id} is not in this build's task_map "
                f"({self.total_items} tasks). The manifest and the Label Studio project "
                f"are out of sync; do not trust labels pulled from it."
            )
        return self.task_map[key]

    def save(self, path: str | Path) -> None:
        """Write the manifest to JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AnnotationManifest:
        """Load a manifest from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class AnnotationLabel(BaseModel):
    """One annotator's rating of one pair, keyed back to the canonical pair.

    Attributes:
        pair_id: Canonical ``"{source_file_id}:{pair_index}"``.
        annotator_id: Anonymous stable id (``A1``, ``A2``, ``A3``).
        score: Ordinal emic-validity rating {1..5}.
        rationale: Optional free-text justification.
        timestamp: Label Studio ``created_at`` for the annotation, verbatim.
    """

    pair_id: str
    annotator_id: str
    score: int = Field(..., ge=1, le=5)
    rationale: str | None = None
    timestamp: str
