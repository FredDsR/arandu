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
    """One pair selected into the human-comparison sample.

    Carries the blinded annotation payload (segment + question + answer) plus the
    stratification bookkeeping. Deliberately excludes ``tacit_inference`` and the
    canonical judge scores; further blinding (hiding ``bloom_level`` from the
    annotator) is the annotation instrument's responsibility, not this artifact's.

    Carries **no emic score**. The sample is built from the CEP records alone so
    the emic-judge run stays off the annotation critical path; the judge's scores
    join back at analysis time on ``pair_id``. Re-adding the field would restore
    the dependency silently (revised 2026-08-19).

    Attributes:
        pair_id: Stable canonical key ``"{source_file_id}:{pair_index}"``. This is
            the join key to the emic-judge outputs at analysis time.
        source_file_id: Source interview id (joins back to the CEP record).
        pair_index: Index into the source ``QARecordCEP.qa_pairs``.
        segment: Source transcript segment the QA pair was generated from.
        question: The generated question.
        answer: The generated answer.
        bloom_level: Bloom level. This is the stratification cell.
        slot_id: 0-based slot within the cell (``0..per_cell-1``).
    """

    pair_id: str
    source_file_id: str
    pair_index: int = Field(..., ge=0)
    segment: str
    question: str
    answer: str
    bloom_level: str
    slot_id: int = Field(..., ge=0)


class SampleManifest(BaseModel):
    """Provenance + reproducibility record for a built sample.

    Attributes:
        pipeline_id: Run identifier.
        seed: RNG seed (makes the selection reproducible).
        total_items: Number of items in the sample (4 cells x per_cell).
        per_cell: Target pairs per cell.
        cell_counts: Selected count per Bloom level (each equals ``per_cell``).
        population_by_cell: In-frame available count per Bloom level (the pool
            each cell was sampled from). Keys are the Bloom levels themselves,
            not the ``"{bloom}:{band}"`` composites used before 2026-08-19.
        excluded_not_approved: Pairs dropped because ``judge-qa`` did not approve
            them. **Not comparable to the pre-2026-08-19 counts:** the builder
            used to walk the emic-judge outputs and could only count pairs the
            judge had scored, whereas it now walks the CEP records and counts
            every non-approved pair in the corpus, so the number is much larger
            for the same run.
        excluded_bloom: Approved pairs dropped per out-of-frame Bloom level
            (``apply`` / ``create``), keyed by level.
        pool_sha256: Hash of the in-frame pool entries incl. payload
            (provenance). Not comparable across the two designs either: the pool
            model no longer carries ``emic_score``.
    """

    pipeline_id: str
    seed: int
    total_items: int
    per_cell: int
    cell_counts: dict[str, int]
    population_by_cell: dict[str, int]
    excluded_not_approved: int = 0
    excluded_bloom: dict[str, int] = Field(default_factory=dict)
    pool_sha256: str

    def save(self, path: str | Path) -> None:
        """Write the manifest to JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        """Load a manifest from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
