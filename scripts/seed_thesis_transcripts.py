#!/usr/bin/env python3
"""Seed a fresh thesis run with the transcripts that can be reused.

The thesis corpus (``catalog-final-run.csv``) overlaps an existing run's
transcripts by ``file_id``. This copies ONLY the transcripts whose ``file_id``
appears in the new catalog into a fresh run directory and writes a transcription
checkpoint marking them complete, so a subsequent ``arandu transcribe --id
<target>`` skips them and transcribes only the genuinely new files.

Why only the overlap (not the whole source run): downstream stages discover
inputs by globbing the transcription outputs directory (``qa/batch.py``,
``chunking/batch.py``), NOT by re-reading the catalog. Copying transcripts whose
``file_id`` is absent from the new catalog would leak those files into the
thesis corpus. ``ResultsManager.replicate`` copies the entire source run and is
therefore unsafe here.

Example:
    uv run python scripts/seed_thesis_transcripts.py \
        --source-run test-kg-04 --catalog input/catalog-final-run.csv \
        --target-id thesis-run-01 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import sys
from pathlib import Path

# Audio/video MIME types the transcription loader accepts. Mirrors
# ``AUDIO_VIDEO_MIME_TYPES`` in ``arandu.transcription.batch``.
AUDIO_VIDEO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/m4a",
    "audio/aac",
    "video/mp4",
    "video/quicktime",
    "video/mpeg",
    "video/avi",
    "video/x-msvideo",
    "video/x-matroska",
}


def load_catalog_av_ids(catalog: Path) -> set[str]:
    """Return the set of transcribable (audio/video) file_ids in a catalog.

    Args:
        catalog: Path to the catalog CSV (``gdrive_id``/``file_id`` + ``mime_type``).

    Returns:
        The file_ids whose MIME type is in ``AUDIO_VIDEO_MIME_TYPES``.
    """
    ids: set[str] = set()
    with catalog.open(newline="") as handle:
        for row in csv.DictReader(handle):
            file_id = row.get("gdrive_id") or row.get("file_id")
            mime = (row.get("mime_type") or "").strip()
            if file_id and mime in AUDIO_VIDEO_MIME_TYPES:
                ids.add(file_id)
    return ids


def load_completed_ids(checkpoint: Path) -> set[str]:
    """Return the completed file_ids from a transcription checkpoint.

    Args:
        checkpoint: Path to ``transcription/checkpoint.json``.

    Returns:
        The ``completed_files`` set.
    """
    data = json.loads(checkpoint.read_text())
    completed = data.get("completed_files") or []
    return set(completed)


def main() -> int:
    """Seed reusable transcripts into a fresh thesis run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run", default="test-kg-04", help="Run id to reuse transcripts from."
    )
    parser.add_argument("--catalog", type=Path, default=Path("input/catalog-final-run.csv"))
    parser.add_argument("--target-id", required=True, help="New thesis run id to create.")
    parser.add_argument("--base-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true", help="Report counts; copy nothing.")
    args = parser.parse_args()

    src_trans = args.base_dir / args.source_run / "transcription"
    src_outputs = src_trans / "outputs"
    src_checkpoint = src_trans / "checkpoint.json"
    for path in (src_outputs, src_checkpoint, args.catalog):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    catalog_ids = load_catalog_av_ids(args.catalog)
    completed_ids = load_completed_ids(src_checkpoint)
    reusable = sorted(catalog_ids & completed_ids)
    new_to_do = sorted(catalog_ids - completed_ids)

    print(f"catalog transcribable (audio/video): {len(catalog_ids)}")
    print(f"source '{args.source_run}' completed:  {len(completed_ids)}")
    print(f"reusable (copy):                       {len(reusable)}")
    print(f"new to transcribe after seeding:       {len(new_to_do)}")

    tgt_trans = args.base_dir / args.target_id / "transcription"
    tgt_outputs = tgt_trans / "outputs"
    tgt_checkpoint = tgt_trans / "checkpoint.json"

    if args.dry_run:
        print(f"\n[dry-run] would create {tgt_outputs} and copy {len(reusable)} transcripts")
        print(
            f"[dry-run] would write {tgt_checkpoint} "
            f"(completed={len(reusable)}, total={len(catalog_ids)})"
        )
        return 0

    if tgt_trans.exists():
        print(f"ERROR: target already exists: {tgt_trans} (refusing to overwrite)", file=sys.stderr)
        return 1

    tgt_outputs.mkdir(parents=True, exist_ok=False)
    copied = 0
    missing: list[str] = []
    for file_id in reusable:
        src_file = src_outputs / f"{file_id}_transcription.json"
        if not src_file.exists():
            missing.append(file_id)
            continue
        shutil.copy2(src_file, tgt_outputs / src_file.name)
        copied += 1

    now = datetime.datetime.now(datetime.UTC).isoformat()
    checkpoint = {
        "completed_files": reusable,
        "failed_files": {},
        "total_files": len(catalog_ids),
        "started_at": now,
        "last_updated": now,
    }
    tgt_checkpoint.write_text(json.dumps(checkpoint, indent=2))

    print(f"\nCopied {copied} transcripts to {tgt_outputs}")
    if missing:
        print(
            f"WARNING: {len(missing)} reusable ids had no transcript file in source: {missing[:10]}"
        )
    print(
        f"Wrote checkpoint {tgt_checkpoint} (completed={len(reusable)}, total={len(catalog_ids)})"
    )
    print(
        f"\nNext: arandu transcribe --id {args.target_id} "
        f"--model openai/whisper-large-v3-turbo  # only the {len(new_to_do)} new files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
