"""CLI command: ``arandu chunk`` — build ChunkSets across one or more chunker views."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from arandu.shared.chunking.registry import KNOWN_CHUNKER_IDS, get_chunker
from arandu.shared.chunking.schemas import Chunk, ChunkSet
from arandu.shared.schemas import EnrichedRecord
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


def chunk(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of EnrichedRecord JSON files (transcriptions).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Output directory for ChunkSet JSON files."),
    ] = Path("chunks"),
    views: Annotated[
        list[str] | None,
        typer.Option(
            "--view",
            help=(
                "Chunker view ID; repeatable. Known: "
                + ", ".join(KNOWN_CHUNKER_IDS)
                + ". Defaults to cep_4k."
            ),
        ),
    ] = None,
    language: Annotated[
        str, typer.Option("--language", "-l", help="Language hint (ISO 639-1).")
    ] = "pt",
) -> None:
    """Build ChunkSets across one or more chunker views.

    Reads every ``*.json`` file in ``input_dir`` as an EnrichedRecord. For each
    record, slices ``transcription_text`` with every requested chunker view and
    writes a ChunkSet to ``output_dir/{source_file_id}.json``.
    """
    selected_views: list[str] = views or ["cep_4k"]

    unknown = [v for v in selected_views if v not in KNOWN_CHUNKER_IDS]
    if unknown:
        print_error(f"Unknown chunker_id(s): {unknown}. Known: {list(KNOWN_CHUNKER_IDS)}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print_warning(f"No JSON files found in {input_dir}")
        return

    print_info(
        f"Chunking {len(json_files)} record(s) into views: {selected_views}; language={language}"
    )

    chunkers = {view_id: get_chunker(view_id) for view_id in selected_views}

    written = 0
    skipped = 0
    for json_file in json_files:
        try:
            record = EnrichedRecord.model_validate_json(json_file.read_text())
        except ValidationError as exc:
            logger.debug("Skipping non-EnrichedRecord file %s: %s", json_file, exc)
            skipped += 1
            continue

        text = record.transcription_text
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        views_payload: dict[str, list[Chunk]] = {
            view_id: chunker.chunk(text, source_file_id=record.file_id)
            for view_id, chunker in chunkers.items()
        }

        chunk_set = ChunkSet(
            source_file_id=record.file_id,
            source_filename=record.name,
            source_text_sha256=sha,
            views=views_payload,
            generated_at=datetime.now(UTC),
        )
        out_path = output_dir / f"{record.file_id}.json"
        chunk_set.save(out_path)
        written += 1

    if skipped:
        print_warning(f"Skipped {skipped} unreadable / non-EnrichedRecord file(s).")
    print_success(f"Wrote {written} ChunkSet(s) to {output_dir}")
