#!/usr/bin/env python3
"""Migrate a run's ``chunk_id`` namespace onto the canonical coordinate space.

Before ``EnrichedRecord.transcription_text`` was normalized, the ``chunk`` stage
chunked the raw transcription while CEP generation chunked the same text
``.strip()``ed. Whisper prefixes its output with a space, so every stage
boundary sat one character past generation's and the two produced disjoint
``chunk_id`` namespaces: 0 of 2670 pairs in ``thesis-run-01`` resolved against
their persisted ``ChunkSet``.

This rewrites a run's ``chunk`` stage in the canonical space, by re-chunking the
canonical text with the real chunker rather than shifting offsets
arithmetically, and remaps every consumer that referenced the old ids. Design:
``docs/superpowers/specs/2026-09-09-chunk-id-coordinate-space-normalization-design.md``.

Usage:
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --dry-run
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --verify

Writing requires the run to be a clone (``arandu replicate`` provenance in its
``pipeline.json``); ``--dry-run`` and ``--verify`` are read-only and are not
gated, so both can be rehearsed against the frozen original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console

from arandu.qa.schemas import QARecordCEP
from arandu.shared.chunking.registry import get_chunker
from arandu.shared.chunking.schemas import Chunk, ChunkSet
from arandu.shared.config import get_results_config
from arandu.shared.io import resolve_transcription_path
from arandu.shared.schemas import EnrichedRecord, PipelineMetadata


@dataclass(frozen=True)
class SourceTexts:
    """The two readings of one transcription file the migration needs.

    After the ``EnrichedRecord`` validator landed, the raw text is no longer
    observable through the schema: loading a record already returns it stripped.
    The migration still needs the raw lead to shift the atlas passage offsets
    and to assert the rewrite is a pure shift, so it reads the file both ways.

    Attributes:
        canonical: Text as every offset in the pipeline refers to it.
        raw: Text exactly as stored on disk, still carrying the whitespace the
            schema strips. Every pre-migration offset refers to this reading,
            so it is what the content-preservation assertions compare against.
        lead_ws: Count of leading whitespace characters the schema stripped.
        filename: The record's source media filename, available to callers.
            ``rewrite_chunk_sets`` does not use it: the rewritten ChunkSet keeps
            the old ChunkSet's own ``source_filename``, per the migration's
            invariant that filenames are untouched.
    """

    canonical: str
    raw: str
    lead_ws: int
    filename: str


def load_source_texts(transcription_dir: Path, file_id: str) -> SourceTexts:
    """Read one transcription both canonically and raw.

    Args:
        transcription_dir: The run's ``transcription/outputs`` directory.
        file_id: Source file identifier.

    Returns:
        Both readings, plus the stripped leading-whitespace count.

    Raises:
        FileNotFoundError: If no transcription file exists for ``file_id``. This
            aborts rather than skipping: a ChunkSet with no transcription is an
            inconsistency the migration must not paper over.
    """
    path = resolve_transcription_path(transcription_dir, file_id)
    if path is None:
        raise FileNotFoundError(f"No transcription for {file_id!r} under {transcription_dir}")

    payload = path.read_text(encoding="utf-8")
    raw = json.loads(payload)["transcription_text"]
    record = EnrichedRecord.model_validate_json(payload)
    return SourceTexts(
        canonical=record.transcription_text,
        raw=raw,
        lead_ws=len(raw) - len(raw.lstrip()),
        filename=record.name,
    )


def _assert_pure_shift(path: Path, stale: list[Chunk], fresh: list[Chunk], lead_ws: int) -> None:
    """Abort unless the fresh chunking is the stale one shifted by ``lead_ws``.

    This is the safety rail on re-chunking rather than shifting arithmetically,
    and it doubles as the content-preservation proof: identical spans over the
    same underlying text resolve to identical strings, the sole exception being
    the first chunk, which loses the stripped leading whitespace.

    Args:
        path: ChunkSet path, for the error message.
        stale: Chunks currently on disk.
        fresh: Chunks produced from the canonical text.
        lead_ws: Leading whitespace the schema stripped.

    Raises:
        ValueError: If the count or any span fails to match the expected shift.
    """
    if len(stale) != len(fresh):
        raise ValueError(
            f"{path}: chunk count changed under normalization "
            f"({len(stale)} -> {len(fresh)}); refusing to write"
        )
    for old, new in zip(stale, fresh, strict=True):
        expected = (max(old.start_char - lead_ws, 0), old.end_char - lead_ws)
        if (new.start_char, new.end_char) != expected:
            raise ValueError(
                f"{path}: expected chunk span {expected}, got "
                f"({new.start_char}, {new.end_char}); refusing to write"
            )


def _sha256(text: str) -> str:
    """Return the hex sha256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _classify_chunk_set(
    path: Path, recorded_sha: str, texts: SourceTexts
) -> Literal["migrated", "stale"]:
    """Say which coordinate space a ChunkSet's recorded hash puts it in.

    Args:
        path: ChunkSet path, for the error message.
        recorded_sha: The ChunkSet's ``source_text_sha256``.
        texts: Both readings of the ChunkSet's source transcription.

    Returns:
        ``"migrated"`` when the hash is the canonical text's, so the file is
        already in the target space and must be left alone, or ``"stale"`` when
        it is the raw text's, so the file is the migration's input.

    Raises:
        ValueError: If the hash is neither, which means the ChunkSet was not
            built from this transcription at all.
    """
    if recorded_sha == _sha256(texts.canonical):
        return "migrated"
    if recorded_sha == _sha256(texts.raw):
        return "stale"
    raise ValueError(
        f"{path}: source_text_sha256 matches neither the raw nor the canonical "
        f"reading of {texts.filename!r}, so this ChunkSet was not built from "
        f"that transcription; refusing to write"
    )


def rewrite_chunk_sets(run_dir: Path, *, dry_run: bool) -> tuple[dict[str, str], dict[str, int]]:
    """Rewrite every ChunkSet in the run's canonical coordinate space.

    A ChunkSet whose recorded ``source_text_sha256`` is already the canonical
    text's is left alone and contributes no id mapping, which makes a rerun over
    a fully migrated run a no-op. Such a file also reports a lead of 0, so the
    sidecar is not shifted a second time. Recovering a migration that failed
    part way through is a re-clone, never a rerun.

    Args:
        run_dir: The run directory (``<results base>/<run id>``).
        dry_run: When true, compute everything but write nothing.

    Returns:
        A tuple of the ``old chunk_id -> new chunk_id`` map and the
        ``file_id -> leading whitespace still to strip`` map.

    Raises:
        ValueError: If a ChunkSet file holds views other than the one its
            directory names, if its recorded hash matches neither reading of
            its transcription, or if the rewrite is not a pure shift.
    """
    chunk_outputs = run_dir / "chunk" / "outputs"
    transcription_dir = run_dir / "transcription" / "outputs"
    id_map: dict[str, str] = {}
    lead_by_file: dict[str, int] = {}

    for view_dir in sorted(p for p in chunk_outputs.iterdir() if p.is_dir()):
        view_id = view_dir.name
        chunker = get_chunker(view_id)
        for path in sorted(view_dir.glob("*.json")):
            stale_set = ChunkSet.load(path)
            if set(stale_set.views) != {view_id}:
                raise ValueError(
                    f"{path}: expected only the {view_id!r} view, found {sorted(stale_set.views)}"
                )

            texts = load_source_texts(transcription_dir, stale_set.source_file_id)
            state = _classify_chunk_set(path, stale_set.source_text_sha256, texts)
            if state == "migrated":
                # setdefault, not assignment: a stale view of the same file
                # elsewhere in the run still needs its real lead, and letting
                # the nonzero lead win makes the sidecar check abort loudly on
                # a half-migrated run instead of skipping it.
                lead_by_file.setdefault(stale_set.source_file_id, 0)
                continue
            lead_by_file[stale_set.source_file_id] = texts.lead_ws

            stale = stale_set.view(view_id)
            fresh = chunker.chunk(texts.canonical, source_file_id=stale_set.source_file_id)
            _assert_pure_shift(path, stale, fresh, texts.lead_ws)

            for old, new in zip(stale, fresh, strict=True):
                id_map[old.chunk_id] = new.chunk_id

            if not dry_run:
                ChunkSet(
                    source_file_id=stale_set.source_file_id,
                    source_filename=stale_set.source_filename,
                    source_text_sha256=_sha256(texts.canonical),
                    views={view_id: fresh},
                    generated_at=stale_set.generated_at,
                ).save(path)

    return id_map, lead_by_file


PASSAGE_STAGES: tuple[str, ...] = ("retrieve", "answers", "judge_answers")


def remap_bm25_manifests(run_dir: Path, id_map: dict[str, str], *, dry_run: bool) -> int:
    """Remap ``chunk_ids`` in every BM25 index manifest.

    The manifest's ``chunk_ids`` are positionally aligned with the pickled BM25
    corpus, and its ``sha256`` covers ``bm25.pkl`` rather than the manifest
    itself, so rewriting the ids leaves the index valid and the pickle untouched.
    Ids are remapped by lookup, not by position, so a reordered manifest cannot
    silently mis-map.

    Args:
        run_dir: The run directory.
        id_map: Old to new ``chunk_id``.
        dry_run: When true, count but write nothing.

    Returns:
        The number of ids remapped.
    """
    remapped = 0
    for manifest_path in sorted((run_dir / "retrieve" / "indexes").glob("bm25_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stale_ids = manifest.get("chunk_ids")
        if not stale_ids:
            continue

        fresh_ids = [id_map.get(cid, cid) for cid in stale_ids]
        changed = sum(1 for old, new in zip(stale_ids, fresh_ids, strict=True) if old != new)
        if not changed:
            continue

        remapped += changed
        if not dry_run:
            manifest["chunk_ids"] = fresh_ids
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return remapped


def remap_passage_chunk_ids(
    run_dir: Path, id_map: dict[str, str], *, dry_run: bool
) -> tuple[int, int]:
    """Remap ``passages[].chunk_id`` across the retrieval-shaped stages.

    Every passage is visited exactly once and looked up against the original
    value, so a new id that happens to collide with some other old id cannot be
    mapped twice. Ids absent from the map are left alone, which is what
    preserves the foreign namespaces without naming a single arm: ``atlas_rag``
    and ``khop_passage`` use ``<file_id>:<index>`` and ``khop_triple`` uses
    ``triple:<sha>``.

    Args:
        run_dir: The run directory.
        id_map: Old to new ``chunk_id``.
        dry_run: When true, count but write nothing.

    Returns:
        A tuple of (files touched, passage references remapped).
    """
    files = 0
    refs = 0
    for stage in PASSAGE_STAGES:
        stage_outputs = run_dir / stage / "outputs"
        if not stage_outputs.is_dir():
            continue
        for path in sorted(stage_outputs.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            passages = payload.get("passages")
            if not isinstance(passages, list):
                continue

            touched = 0
            for passage in passages:
                fresh = id_map.get(passage.get("chunk_id"))
                if fresh is not None and fresh != passage["chunk_id"]:
                    passage["chunk_id"] = fresh
                    touched += 1

            if not touched:
                continue
            files += 1
            refs += touched
            if not dry_run:
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return files, refs


def _assert_offset_preserved(
    passage_id: str, texts: SourceTexts, old_span: tuple[int, int], new_span: tuple[int, int]
) -> None:
    """Abort unless the shifted span resolves to the text the old span did.

    This is the sidecar's counterpart to :func:`_assert_pure_shift`, and the
    reason the sidecar is not mutated by unchecked arithmetic: at migration time
    both readings of the transcription are in hand, so the old span's text is
    still observable and can be compared with the new span's. The one licensed
    difference is the whitespace the schema stripped: a span reaching into the
    lead loses it, exactly as the first chunk does.

    Args:
        passage_id: Sidecar entry id, for the error message.
        texts: Both readings of the entry's source transcription.
        old_span: ``(start_char, end_char)`` as persisted, in raw space.
        new_span: ``(start_char, end_char)`` to be written, in canonical space.

    Raises:
        ValueError: If the two spans resolve to different strings.
    """
    old_start, old_end = old_span
    expected = texts.raw[max(old_start, texts.lead_ws) : old_end]
    actual = texts.canonical[new_span[0] : new_span[1]]
    if actual != expected:
        raise ValueError(
            f"passage_offsets.json: {passage_id} shifted from {old_span} to "
            f"{new_span} resolves to different text ({len(expected)} chars "
            f"before, {len(actual)} after); refusing to write"
        )


def shift_passage_offsets(run_dir: Path, lead_by_file: dict[str, int], *, dry_run: bool) -> int:
    """Shift the atlas passage-offset sidecar into the canonical space.

    The sidecar's spans are expressed against ``EnrichedRecord.transcription_text``,
    which the schema validator just moved by the stripped leading whitespace. The
    synthesized ``passage_id`` is index-based, so ids stay stable and only spans
    move.

    Every shifted span is checked against the text the old span resolved to
    before anything is written, and a ``source_file_id`` the ``chunk`` stage
    never covered is an abort rather than a skip: the sidecar comes from the
    ``kg`` stage, which selects its input files independently, and an offset
    silently left in the old space would stay wrong forever.

    Args:
        run_dir: The run directory.
        lead_by_file: ``file_id`` to leading whitespace still to strip, as
            returned by :func:`rewrite_chunk_sets`. A file mapped to 0 is a
            legitimate no-op; a file absent from the map is an inconsistency.
        dry_run: When true, count but write nothing.

    Returns:
        The number of offsets shifted.

    Raises:
        ValueError: If an entry names a file the ``chunk`` stage never covered,
            or if a shifted span does not resolve to the same text.
    """
    path = run_dir / "kg" / "outputs" / "passage_offsets.json"
    if not path.exists():
        return 0

    transcription_dir = run_dir / "transcription" / "outputs"
    texts_by_file: dict[str, SourceTexts] = {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    shifted = 0
    for offset in payload.get("offsets", []):
        file_id = offset["source_file_id"]
        if file_id not in lead_by_file:
            raise ValueError(
                f"passage_offsets.json: {offset['passage_id']} names source file "
                f"{file_id!r}, which has no ChunkSet in this run, so its lead is "
                f"unknown; refusing to write"
            )
        lead = lead_by_file[file_id]
        if lead == 0:
            continue

        if file_id not in texts_by_file:
            texts_by_file[file_id] = load_source_texts(transcription_dir, file_id)
        texts = texts_by_file[file_id]

        old_span = (offset["start_char"], offset["end_char"])
        # end_char is constrained gt=0 on PassageOffset; clamp so a degenerate
        # one-character span cannot make the sidecar unloadable.
        new_span = (max(old_span[0] - lead, 0), max(old_span[1] - lead, 1))
        _assert_offset_preserved(offset["passage_id"], texts, old_span, new_span)

        offset["start_char"], offset["end_char"] = new_span
        shifted += 1

    if shifted and not dry_run:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return shifted


# Offset-derived chunk_ids are a 16-char lowercase sha1 prefix. Foreign
# namespaces are shaped differently on purpose: atlas_rag and khop_passage use
# "<file_id>:<index>", khop_triple uses "triple:<sha>". Only the offset-derived
# ones are expected to resolve against a ChunkSet.
_OFFSET_DERIVED_ID = re.compile(r"^[0-9a-f]{16}$")


def _known_chunk_ids(run_dir: Path) -> set[str]:
    """Collect every chunk_id present in the run's ChunkSets."""
    known: set[str] = set()
    for path in sorted((run_dir / "chunk" / "outputs").rglob("*.json")):
        for chunks in ChunkSet.load(path).views.values():
            known.update(chunk.chunk_id for chunk in chunks)
    return known


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of :func:`verify`.

    Attributes:
        failures: Human-readable failures. Empty means the run is consistent.
        resolved: CEP pairs whose ``chunk_id`` resolves against a ChunkSet.
        total: CEP pairs inspected. Zero is itself a failure: there is nothing
            to resolve, so the headline check never ran.
    """

    failures: list[str]
    resolved: int
    total: int


def verify(run_dir: Path) -> VerificationResult:
    """Check that a migrated run is internally consistent.

    Runs after the rewrite, so it cannot compare against the pre-migration
    state. Content preservation is asserted during the rewrite instead, by
    :func:`_assert_pure_shift` for the ChunkSets and by
    :func:`_assert_offset_preserved` for the atlas sidecar. What is checked
    here:

    1. Every CEP pair's ``chunk_id`` resolves against its file's ChunkSet. This
       is the headline: 2670 of 2670 in ``thesis-run-02``, against 0 of 2670
       before. The fraction is reported, never hard-coded, and a run with no
       pairs at all fails rather than passing vacuously.
    2. Every ChunkSet's ``source_text_sha256`` matches its canonical text.
    3. Re-chunking the canonical text reproduces the ids already on disk, so the
       artifact really is what the fixed pipeline would produce.
    4. No offset-derived id referenced by a BM25 manifest or by
       ``passages[].chunk_id`` dangles.
    5. Every atlas passage offset lies within its canonical text.

    Args:
        run_dir: The run directory.

    Returns:
        The failures found plus the resolved-pair fraction. An empty failure
        list means the run is consistent.
    """
    failures: list[str] = []
    transcription_dir = run_dir / "transcription" / "outputs"

    chunk_outputs = run_dir / "chunk" / "outputs"
    for view_dir in sorted(p for p in chunk_outputs.iterdir() if p.is_dir()):
        view_id = view_dir.name
        chunker = get_chunker(view_id)
        for path in sorted(view_dir.glob("*.json")):
            chunk_set = ChunkSet.load(path)
            texts = load_source_texts(transcription_dir, chunk_set.source_file_id)

            expected_sha = _sha256(texts.canonical)
            if chunk_set.source_text_sha256 != expected_sha:
                failures.append(
                    f"{path.name}: source_text_sha256 does not match the canonical text"
                )

            on_disk = [c.chunk_id for c in chunk_set.view(view_id)]
            recomputed = [
                c.chunk_id
                for c in chunker.chunk(texts.canonical, source_file_id=chunk_set.source_file_id)
            ]
            if on_disk != recomputed:
                failures.append(
                    f"{path.name}: re-chunking the canonical text does not reproduce "
                    f"the persisted chunk_ids"
                )

    known = _known_chunk_ids(run_dir)

    cep_outputs = run_dir / "cep" / "outputs"
    cep_files = sorted(cep_outputs.glob("*_cep_qa.json")) if cep_outputs.is_dir() else []
    resolved = 0
    total = 0
    for path in cep_files:
        record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        for pair in record.qa_pairs:
            total += 1
            if pair.chunk_id is None:
                failures.append(f"{path.name}: a pair carries no chunk_id")
            elif pair.chunk_id in known:
                resolved += 1
            else:
                failures.append(f"{path.name}: chunk_id {pair.chunk_id} resolves to no chunk")
    if total == 0:
        failures.append(
            f"no CEP pairs found under {cep_outputs}, so nothing proves the "
            f"chunk_id namespace resolves"
        )
    elif resolved != total:
        failures.append(f"CEP pairs resolving against a ChunkSet: {resolved}/{total}")

    for manifest_path in sorted((run_dir / "retrieve" / "indexes").glob("bm25_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for cid in manifest.get("chunk_ids", []):
            if _OFFSET_DERIVED_ID.match(cid) and cid not in known:
                failures.append(f"{manifest_path.parent.name}/manifest.json: dangling {cid}")

    for stage in PASSAGE_STAGES:
        stage_outputs = run_dir / stage / "outputs"
        if not stage_outputs.is_dir():
            continue
        for path in sorted(stage_outputs.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            passages = payload.get("passages")
            if not isinstance(passages, list):
                continue
            for passage in passages:
                cid = passage.get("chunk_id", "")
                if _OFFSET_DERIVED_ID.match(cid) and cid not in known:
                    failures.append(f"{stage}/{path.name}: dangling passage chunk_id {cid}")

    sidecar = run_dir / "kg" / "outputs" / "passage_offsets.json"
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        for offset in payload.get("offsets", []):
            texts = load_source_texts(transcription_dir, offset["source_file_id"])
            if offset["end_char"] > len(texts.canonical):
                failures.append(
                    f"passage_offsets.json: {offset['passage_id']} ends past the "
                    f"canonical text ({offset['end_char']} > {len(texts.canonical)})"
                )

    return VerificationResult(failures=failures, resolved=resolved, total=total)


def clone_provenance(run_dir: Path) -> str | None:
    """Return the pipeline this run was replicated from, or ``None``.

    ``arandu replicate`` stamps ``PipelineMetadata.replicated_from`` into the
    run's ``pipeline.json`` (``shared/results_manager.py``), which is the only
    on-disk evidence that a run is a clone rather than an original. The
    migration has no ``.bak``, so writing to an original would destroy the
    frozen pre-correction record with nothing to recover from.

    Args:
        run_dir: The run directory.

    Returns:
        The source pipeline id, or ``None`` when the run carries no replication
        provenance or has no ``pipeline.json`` at all.
    """
    path = run_dir / "pipeline.json"
    if not path.exists():
        return None
    replication = PipelineMetadata.load(path).replicated_from
    return None if replication is None else replication.source_pipeline_id


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", required=True, help="Pipeline/run ID under the results base dir")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory. Defaults to ARANDU_RESULTS_BASE_DIR, then ./results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change, write nothing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check an already-migrated run instead of migrating it",
    )
    parser.add_argument(
        "--allow-original",
        action="store_true",
        help=(
            "Write to a run that was not produced by `arandu replicate`. "
            "There is no backup: only pass this deliberately."
        ),
    )
    args = parser.parse_args()

    console = Console()
    base_dir = args.results_dir or get_results_config().base_dir
    run_dir = base_dir / args.id
    if not (run_dir / "chunk" / "outputs").is_dir():
        console.print(f"[red]No chunk stage under {run_dir}[/red]")
        sys.exit(2)

    if args.verify:
        result = verify(run_dir)
        console.print(f"  CEP pairs resolving against a ChunkSet: {result.resolved}/{result.total}")
        if result.failures:
            console.print(f"[red]{len(result.failures)} check(s) failed:[/red]")
            for failure in result.failures:
                console.print(f"  [red]{failure}[/red]")
            sys.exit(1)
        console.print(f"[green]{args.id} is consistent in the canonical space[/green]")
        return

    provenance = clone_provenance(run_dir)
    if not args.dry_run and not args.allow_original and provenance is None:
        console.print(
            f"[red]{args.id} carries no `replicated_from` provenance, so it may be "
            f"an original rather than a clone.[/red]\n"
            f"Clone it first ([cyan]arandu replicate {args.id} --id <new>[/cyan]) and "
            f"migrate the clone, or pass [cyan]--allow-original[/cyan] to write here "
            f"anyway. This script keeps no backup."
        )
        sys.exit(2)

    # Always run the whole pass read-only first. The writing steps have no
    # cross-file atomicity, so every logic-level abort must happen before the
    # first byte is written, not part way through the 214 ChunkSets.
    id_map, lead_by_file = rewrite_chunk_sets(run_dir, dry_run=True)
    manifest_ids = remap_bm25_manifests(run_dir, id_map, dry_run=True)
    files, refs = remap_passage_chunk_ids(run_dir, id_map, dry_run=True)
    offsets = shift_passage_offsets(run_dir, lead_by_file, dry_run=True)

    if not args.dry_run:
        id_map, lead_by_file = rewrite_chunk_sets(run_dir, dry_run=False)
        manifest_ids = remap_bm25_manifests(run_dir, id_map, dry_run=False)
        files, refs = remap_passage_chunk_ids(run_dir, id_map, dry_run=False)
        offsets = shift_passage_offsets(run_dir, lead_by_file, dry_run=False)

    label = "would remap" if args.dry_run else "remapped"
    console.print(f"[bold]{args.id}[/bold] ({'dry run' if args.dry_run else 'applied'})")
    console.print(f"  replicated from: {provenance or 'nothing (not a clone)'}")
    console.print(f"  chunk_ids {label}: {len(id_map)} across {len(lead_by_file)} files")
    console.print(f"  bm25 manifest ids {label}: {manifest_ids}")
    console.print(f"  passage references {label}: {refs} in {files} files")
    console.print(f"  atlas offsets {'would shift' if args.dry_run else 'shifted'}: {offsets}")
    if not args.dry_run:
        console.print(f"\nNow run: [cyan]--id {args.id} --verify[/cyan]")


if __name__ == "__main__":
    main()
