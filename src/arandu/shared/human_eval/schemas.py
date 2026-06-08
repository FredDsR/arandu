"""Schemas for the stratified human-comparison sample (spec §5)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class HumanEvalSampleConfig(BaseModel):
    """Config snapshotted into ``run_metadata.json`` for a sample build.

    Attributes:
        seed: RNG seed driving the deterministic selection.
        per_cell: Pairs drawn per stratification cell.
    """

    seed: int
    per_cell: int


class SampleItem(BaseModel):
    """One pair selected into the 80-pair human-comparison sample.

    Carries the blinded annotation payload (segment + question + answer) plus
    the stratification bookkeeping. Deliberately excludes ``tacit_inference``
    and the canonical judge scores; further blinding (hiding ``bloom_level`` /
    ``emic_prepass_score`` from the annotator) is the annotation instrument's
    responsibility, not this artifact's.

    Attributes:
        pair_id: Stable canonical key ``"{source_file_id}:{pair_index}"``.
        source_file_id: Source interview id (joins back to the CEP record).
        pair_index: Index into the source ``QARecordCEP.qa_pairs``.
        segment: Source transcript segment the QA pair was generated from.
        question: The generated question.
        answer: The generated answer.
        bloom_level: Bloom level (stratification dimension).
        emic_prepass_score: Ordinal emic band hint {1..5} from the pre-pass.
        cell_id: ``"{bloom_level}:{band}"`` stratification cell.
        slot_id: 0-based slot within the cell (0..9).
    """

    pair_id: str
    source_file_id: str
    pair_index: int = Field(..., ge=0)
    segment: str
    question: str
    answer: str
    bloom_level: str
    emic_prepass_score: int = Field(..., ge=1, le=5)
    cell_id: str
    slot_id: int = Field(..., ge=0)


class SampleManifest(BaseModel):
    """Provenance + reproducibility record for a built sample.

    Attributes:
        pipeline_id: Run identifier.
        seed: RNG seed (makes the selection reproducible).
        total_items: Number of items in the sample (8 cells x per_cell).
        per_cell: Target pairs per cell.
        cell_counts: Selected count per ``cell_id`` (each equals ``per_cell``).
        population_by_cell: In-frame available count per ``cell_id`` (the pool
            each cell was sampled from).
        excluded_none_score: Approved pairs dropped for a null emic score.
        excluded_bloom: Approved pairs dropped per out-of-frame Bloom level
            (``apply`` / ``create``), keyed by level.
        pool_sha256: Hash of the in-frame pool keys + scores (provenance).
    """

    pipeline_id: str
    seed: int
    total_items: int
    per_cell: int
    cell_counts: dict[str, int]
    population_by_cell: dict[str, int]
    excluded_none_score: int = 0
    excluded_bloom: dict[str, int] = Field(default_factory=dict)
    pool_sha256: str

    def save(self, path: str | Path) -> None:
        """Write the manifest to JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        """Load a manifest from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
