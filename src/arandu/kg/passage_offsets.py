"""Atlas-rag passage offset mapping — Phase C spec §3.8.

Atlas-rag's KG extraction emits passages whose ``original_text`` is a chunk
of the source transcription prefixed with an injected
``[Contexto da Entrevista]…[Transcrição]\\n`` header (the same metadata
prelude atlas-rag adds at construction time). This module maps each such
passage back to a ``(start_char, end_char)`` span in the original
``EnrichedRecord.transcription_text``, so atlas-rag passages can be
expressed in the same coordinate space as BM25 / NetworkX chunks.

Atlas-rag does not emit a stable ``passage_id``; this module synthesises one
as ``<source_file_id>:<chunk_index>``, where the index is the position of
the record among same-``id`` records in JSONL order. The synthesis is
deterministic given the JSONL layout.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field

from arandu.shared.config import ResultsConfig
from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


PASSAGE_OFFSETS_FILENAME = "passage_offsets.json"
_HEADER_END_MARKER = "[Transcrição]\n"
_WHITESPACE_RE = re.compile(r"\s+")


class PassageOffset(BaseModel):
    """One atlas-rag passage's location in source ``EnrichedRecord`` space."""

    passage_id: str = Field(..., description="Synthesised id: '<source_file_id>:<chunk_index>'.")
    source_file_id: str = Field(..., description="EnrichedRecord file_id this chunk came from.")
    start_char: int = Field(..., ge=0, description="Inclusive start offset in transcription_text.")
    end_char: int = Field(..., gt=0, description="Exclusive end offset in transcription_text.")
    chunker_id: Literal["atlas_8k"] = Field(
        default="atlas_8k", description="Fixed identifier of the atlas-rag chunker view."
    )


class PassageOffsetSidecar(BaseModel):
    """A sidecar mapping every atlas-rag passage in a KG run to source offsets."""

    kg_run_id: str = Field(..., description="The pipeline_id whose KG produced these passages.")
    offsets: list[PassageOffset] = Field(default_factory=list)
    unmatched: list[str] = Field(
        default_factory=list,
        description="passage_ids whose chunk text could not be located in the source.",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def save(self, path: Path) -> None:
        """Serialize this sidecar to ``path`` as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a sidecar from ``path``."""
        return cls.model_validate_json(path.read_text())


class _AtlasPassage(BaseModel):
    """In-process view of a single atlas-rag JSONL record."""

    passage_id: str
    source_file_id: str
    text: str


def _iter_atlas_passages(kg_extraction_dir: Path) -> Iterator[_AtlasPassage]:
    """Yield one ``_AtlasPassage`` per JSONL record across all files in ``kg_extraction_dir``.

    The ``passage_id`` is synthesised as ``<id>:<chunk_index>`` where
    ``chunk_index`` is the running count of records with the same ``id``
    encountered so far across the whole directory (JSONL files are read in
    sorted filename order so the synthesis is deterministic).
    """
    if not kg_extraction_dir.exists():
        return
    chunk_idx: Counter[str] = Counter()
    for jsonl_path in sorted(kg_extraction_dir.glob("*.json")):
        # Stream line-by-line so the mapper stays memory-friendly on large KG runs
        # (a single extraction file can hold thousands of records at thesis scale).
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable line in %s", jsonl_path)
                    continue
                file_id = rec.get("id")
                text = rec.get("original_text")
                if not file_id or not text:
                    continue
                idx = chunk_idx[file_id]
                chunk_idx[file_id] += 1
                yield _AtlasPassage(
                    passage_id=f"{file_id}:{idx}",
                    source_file_id=file_id,
                    text=text,
                )


def _strip_atlas_header(text: str) -> str:
    """Strip the atlas-rag-injected ``[Contexto…][Transcrição]\\n`` header.

    If the marker is absent (passage was indexed without a header), return
    the text unchanged.
    """
    idx = text.find(_HEADER_END_MARKER)
    if idx == -1:
        return text
    return text[idx + len(_HEADER_END_MARKER) :]


def _load_source_text(transcription_dir: Path, file_id: str) -> str | None:
    """Load ``EnrichedRecord.transcription_text`` for ``file_id`` from disk.

    The transcription stage today writes ``<file_id>_transcription.json``;
    older / future runs may use the bare ``<file_id>.json``. Try both before
    treating the source as missing.

    Returns ``None`` if no matching file exists or the file fails schema
    validation — callers treat that as an orphan passage (source dropped or
    drifted after KG construction).
    """
    for candidate in (
        transcription_dir / f"{file_id}_transcription.json",
        transcription_dir / f"{file_id}.json",
    ):
        if candidate.exists():
            try:
                record = EnrichedRecord.model_validate_json(candidate.read_text())
            except Exception as exc:
                logger.warning("Skipping invalid EnrichedRecord %s: %s", candidate, exc)
                return None
            return record.transcription_text
    return None


def _find_normalized(source: str, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``source`` ignoring whitespace differences.

    Returns ``(start_char, end_char)`` in the original (un-normalised)
    ``source`` such that ``source[start:end]`` is the whitespace-equivalent
    of ``needle``, or ``None`` if no match exists.

    Implementation note: builds an index from normalised positions back to
    original positions, runs a string find on the normalised forms, then
    maps the result back.
    """
    if not needle.strip():
        return None

    # Build normalised source + per-char map back to original offsets.
    normalised_chars: list[str] = []
    original_offsets: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(source):
        if ch.isspace():
            if prev_was_space:
                continue
            normalised_chars.append(" ")
            original_offsets.append(i)
            prev_was_space = True
        else:
            normalised_chars.append(ch)
            original_offsets.append(i)
            prev_was_space = False
    normalised_source = "".join(normalised_chars).strip()

    # Normalise the needle the same way.
    normalised_needle = _WHITESPACE_RE.sub(" ", needle).strip()
    if not normalised_needle:
        return None

    # Account for the strip() on the normalised source.
    leading_strip = (
        len(normalised_source) - len(normalised_source.lstrip())
        if normalised_source != normalised_source.lstrip()
        else 0
    )
    # original_offsets indexes into normalised_chars pre-strip; after strip we need
    # to skip any leading-space entry. The pre-strip normalised source only ever
    # has a leading space if the source itself started with whitespace; track that.
    pre_strip = "".join(normalised_chars)
    leading_offset = len(pre_strip) - len(pre_strip.lstrip())

    found_at = normalised_source.find(normalised_needle)
    if found_at == -1:
        return None

    start_in_pre_strip = found_at + leading_offset + leading_strip
    end_in_pre_strip_exclusive = start_in_pre_strip + len(normalised_needle)
    if end_in_pre_strip_exclusive > len(original_offsets):
        return None

    start_char = original_offsets[start_in_pre_strip]
    # End offset: original position of the last char + 1
    end_char = original_offsets[end_in_pre_strip_exclusive - 1] + 1
    return start_char, end_char


def link_passages(
    pipeline_id: str,
    base_dir: Path | None = None,
    output: Path | None = None,
) -> PassageOffsetSidecar:
    """Map atlas-rag passages in run ``pipeline_id`` to source offsets.

    Args:
        pipeline_id: The run identifier. Inputs are resolved to
            ``<base_dir>/<pipeline_id>/transcription/outputs/`` (source
            ``EnrichedRecord`` files) and
            ``<base_dir>/<pipeline_id>/kg/outputs/atlas_output/kg_extraction/``
            (atlas-rag JSONL files).
        base_dir: Project ``results/`` root. Defaults to ``ResultsConfig().base_dir``.
        output: Override the default sidecar path
            (``<base_dir>/<pipeline_id>/kg/outputs/passage_offsets.json``).

    Returns:
        The ``PassageOffsetSidecar`` written to disk.

    Raises:
        FileNotFoundError: If the transcription / kg outputs / atlas-rag
            extraction directory is missing for ``pipeline_id``. Failing fast
            on a missing ``kg_extraction/`` is intentional — silently writing
            an empty sidecar would be a false-success artifact for callers
            (wrong backend, incomplete KG run, moved outputs).
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    transcription_dir = base / pipeline_id / "transcription" / "outputs"
    kg_outputs_dir = base / pipeline_id / "kg" / "outputs"
    kg_extraction_dir = kg_outputs_dir / "atlas_output" / "kg_extraction"

    if not transcription_dir.exists():
        raise FileNotFoundError(
            f"transcription outputs not found for pipeline_id {pipeline_id!r}: {transcription_dir}"
        )
    if not kg_outputs_dir.exists():
        raise FileNotFoundError(
            f"kg outputs not found for pipeline_id {pipeline_id!r}: {kg_outputs_dir}"
        )
    if not kg_extraction_dir.exists():
        raise FileNotFoundError(
            f"atlas-rag kg_extraction directory not found for pipeline_id "
            f"{pipeline_id!r}: {kg_extraction_dir}. The KG stage either used a "
            f"non-atlas backend, was interrupted before extraction, or its "
            f"outputs were moved."
        )

    offsets: list[PassageOffset] = []
    unmatched: list[str] = []
    source_text_cache: dict[str, str | None] = {}

    for passage in _iter_atlas_passages(kg_extraction_dir):
        if passage.source_file_id not in source_text_cache:
            source_text_cache[passage.source_file_id] = _load_source_text(
                transcription_dir, passage.source_file_id
            )
        source_text = source_text_cache[passage.source_file_id]
        if source_text is None:
            logger.info(
                "Skipping passage %s — source EnrichedRecord missing",
                passage.passage_id,
            )
            unmatched.append(passage.passage_id)
            continue

        needle = _strip_atlas_header(passage.text)
        # If the atlas record carried only the injected header (degenerate
        # chunk), `needle` is empty / whitespace. `str.find("")` returns 0,
        # which would yield `end_char=0` and fail `PassageOffset.end_char (gt=0)`
        # validation, aborting the whole run. Treat as unmatched instead.
        if not needle.strip():
            logger.warning(
                "Skipping passage %s — chunk text is empty after header strip",
                passage.passage_id,
            )
            unmatched.append(passage.passage_id)
            continue
        idx = source_text.find(needle)
        if idx != -1:
            offsets.append(
                PassageOffset(
                    passage_id=passage.passage_id,
                    source_file_id=passage.source_file_id,
                    start_char=idx,
                    end_char=idx + len(needle),
                )
            )
            continue

        # Whitespace-normalised fallback (atlas-rag occasionally re-flows whitespace).
        normalised = _find_normalized(source_text, needle)
        if normalised is not None:
            start_char, end_char = normalised
            offsets.append(
                PassageOffset(
                    passage_id=passage.passage_id,
                    source_file_id=passage.source_file_id,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            continue

        unmatched.append(passage.passage_id)

    sidecar = PassageOffsetSidecar(
        kg_run_id=pipeline_id,
        offsets=offsets,
        unmatched=unmatched,
        generated_at=datetime.now(UTC),
    )
    out_path = output if output is not None else kg_outputs_dir / PASSAGE_OFFSETS_FILENAME
    sidecar.save(out_path)
    return sidecar
