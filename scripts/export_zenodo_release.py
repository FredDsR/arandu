"""Package an Arandu pipeline run and judge-retest replicas for Zenodo release.

Preserves the native ``ResultsManager`` layout (``results/<run-id>/<stage>/outputs/``)
so researchers can both reproduce reported analyses directly and re-run individual
pipeline stages (e.g., answering and judging with another LLM) without format
conversion.

Usage:
    uv run python scripts/export_zenodo_release.py \\
        --run-dir results/thesis-run-02 \\
        --out-dir results/zenodo-release \\
        --docs-dir ../dissertacao-tex/submissoes/lre/zenodo \\
        --replica r1=results/intrajudge/thesis-run-02-round-1.tar.gz \\
        --replica r2=results/intrajudge/thesis-run-03-round-1.tar.gz \\
        --replica r3=results/intrajudge/thesis-run-03-round-2.tar.gz \\
        --gate-replica r3-gate=results/thesis-run-03 \\
        --replica-answers-overlay \\
            r1:dense_bgem3_256,dense_mpnet128=results/thesis-run-02/judge_answers \\
        --replica-answers-overlay \\
            r2:dense_bgem3_256,dense_mpnet128=results/thesis-run-02-dense-retest-r2/judge_answers \\
        --replica-answers-overlay \\
            r3:dense_bgem3_256,dense_mpnet128=results/thesis-run-02-dense-retest-r3/judge_answers
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from arandu.utils.logger import print_error, print_info, print_success, print_warning

app = typer.Typer(help="Package an Arandu run and test-retest replicas for Zenodo release.")

PUBLICATION_STEPS: tuple[str, ...] = (
    "transcription",
    "chunk",
    "cep",
    "kg",
    "non_answerable",
    "retrieve",
    "answers",
    "judge_answers",
    "emic_judge",
    "judge_qa",
)

ARM_STEPS: frozenset[str] = frozenset({"retrieve", "answers", "judge_answers"})
GDRIVE_SCRUB_STEPS: frozenset[str] = frozenset({"transcription", "chunk", "cep"})
DEFAULT_ARMS: tuple[str, ...] = (
    "null",
    "bm25",
    "khop_passage",
    "khop_triple",
    "atlas_rag",
    "oracle",
    "dense_bgem3_256",
    "dense_mpnet128",
)
DOC_FILENAMES: tuple[str, ...] = ("README.md", "DATA_USE_AGREEMENT.md", "LICENSE")
TEXT_SUFFIXES: frozenset[str] = frozenset({".json", ".jsonl", ".csv", ".graphml", ".txt", ".md"})

_HOST_PATH_RE = re.compile(r"/(?:home|users|scratch|tmp)/[^\s\"']*?/(results|cache)/")
_GENERIC_HOME_RE = re.compile(r"/home/[^/\s\"']+/")


def sanitize_host_path(value: str) -> str:
    """Replace absolute host/cluster prefixes with portable relative paths.

    Args:
        value: Raw string that may contain local or SLURM filesystem paths.

    Returns:
        Sanitized string with portable relative ``results/`` or ``cache/`` paths.
    """
    cleaned = _HOST_PATH_RE.sub(r"\1/", value)
    return _GENERIC_HOME_RE.sub("./", cleaned)


def sanitize_run_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Scrub hostnames, usernames, and cluster paths from a ``run_metadata.json`` dict.

    Args:
        data: Parsed ``run_metadata.json`` dictionary.

    Returns:
        A new dictionary with sensitive host fields scrubbed while preserving
        model, hardware, timing, and configuration provenance.
    """
    cleaned = json.loads(json.dumps(data))

    execution = cleaned.get("execution")
    if isinstance(execution, dict):
        if execution.get("hostname") is not None:
            execution["hostname"] = "redacted"
        if execution.get("username") is not None:
            execution["username"] = "redacted"

    for key in ("output_directory", "checkpoint_file", "input_source"):
        val = cleaned.get(key)
        if isinstance(val, str):
            cleaned[key] = sanitize_host_path(val)

    config = cleaned.get("config")
    if isinstance(config, dict):
        env_vars = config.get("environment_variables")
        if isinstance(env_vars, dict):
            config["environment_variables"] = {
                k: sanitize_host_path(v) if isinstance(v, str) else v for k, v in env_vars.items()
            }

    return cleaned


def scrub_gdrive_paths(obj: Any) -> Any:
    """Recursively clear ``source_gdrive_path`` fields inside a JSON structure.

    Args:
        obj: Decoded JSON value (dict, list, or primitive).

    Returns:
        Structure with ``source_gdrive_path`` set to ``None`` wherever present.
    """
    if isinstance(obj, dict):
        return {
            k: (None if k == "source_gdrive_path" else scrub_gdrive_paths(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub_gdrive_paths(item) for item in obj]
    return obj


def load_id_map(id_map_path: Path | None) -> list[tuple[str, str]]:
    """Load ordered ``(source, replacement)`` pairs from an identification map JSON.

    Accepts either a flat mapping ``{"Original Name": "Pseudonym"}`` or an
    object ``{"replacements": {"Original Name": "Pseudonym"}}``. Keys are
    sorted longest-first so multi-word full names match before shorter tokens.

    Args:
        id_map_path: Optional path to a JSON file defining participant replacements.

    Returns:
        List of ``(source, replacement)`` tuples ordered by descending source length.

    Raises:
        ValueError: If the JSON file does not match either supported schema.
    """
    if id_map_path is None:
        return []

    raw = json.loads(id_map_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "replacements" in raw and isinstance(raw["replacements"], dict):
        mapping = raw["replacements"]
    elif isinstance(raw, dict) and all(isinstance(v, str) for v in raw.values()):
        mapping = raw
    else:
        raise ValueError(
            f"Unsupported id-map schema in {id_map_path}: expected dict[str, str] "
            "or {'replacements': dict[str, str]}."
        )

    pairs = [(str(k), str(v)) for k, v in mapping.items() if k]
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def apply_replacements_to_text(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply case-sensitive substring replacements in order.

    Args:
        text: Input string.
        replacements: Ordered ``(old, new)`` pairs.

    Returns:
        Text with replacements applied.
    """
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def apply_replacements_to_json(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    """Recursively apply text replacements to every string value in a JSON object.

    Args:
        obj: Decoded JSON value.
        replacements: Ordered ``(old, new)`` pairs.

    Returns:
        JSON structure with string values replaced.
    """
    if isinstance(obj, str):
        return apply_replacements_to_text(obj, replacements)
    if isinstance(obj, dict):
        return {k: apply_replacements_to_json(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [apply_replacements_to_json(item, replacements) for item in obj]
    return obj


def filter_arm_checkpoint(
    checkpoint_data: dict[str, Any], selected_arms: set[str]
) -> dict[str, Any]:
    """Trim an arm-indexed checkpoint to only the selected retrieval arms.

    Args:
        checkpoint_data: Parsed checkpoint dictionary.
        selected_arms: Set of arm names to retain.

    Returns:
        Filtered checkpoint dictionary with updated ``total_files``.
    """
    completed = [
        key
        for key in checkpoint_data.get("completed_files", [])
        if isinstance(key, str) and key.split("::", 1)[0] in selected_arms
    ]
    failed = {
        key: val
        for key, val in checkpoint_data.get("failed_files", {}).items()
        if isinstance(key, str) and key.split("::", 1)[0] in selected_arms
    }
    updated = dict(checkpoint_data)
    updated["completed_files"] = completed
    updated["failed_files"] = failed
    updated["total_files"] = len(completed) + len(failed)
    return updated


def filter_transcription_checkpoint(
    checkpoint_data: dict[str, Any], valid_ids: set[str]
) -> dict[str, Any]:
    """Trim a transcription checkpoint to only the quality-approved file IDs.

    Args:
        checkpoint_data: Parsed transcription checkpoint dictionary.
        valid_ids: Set of approved recording file IDs.

    Returns:
        Filtered checkpoint dictionary with ``total_files`` equal to ``len(valid_ids)``.
    """
    completed = [key for key in checkpoint_data.get("completed_files", []) if key in valid_ids]
    updated = dict(checkpoint_data)
    updated["completed_files"] = completed
    updated["failed_files"] = {}
    updated["total_files"] = len(completed)
    return updated


def parse_arms_option(arms_option: str, available_arms: set[str]) -> set[str]:
    """Resolve the ``--arms`` CLI option into a set of arm names.

    Args:
        arms_option: Either ``"all"`` or a comma-separated list of arm names.
        available_arms: Arms present on disk in ``retrieve/outputs/``.

    Returns:
        Set of selected arm names.

    Raises:
        ValueError: If a requested arm is not found in ``available_arms``.
    """
    cleaned = arms_option.strip()
    if cleaned.lower() == "all":
        return set(available_arms)

    requested = {item.strip() for item in cleaned.split(",") if item.strip()}
    missing = requested - available_arms
    if missing:
        raise ValueError(
            f"Requested arm(s) not present in retrieve/outputs: {sorted(missing)}. "
            f"Available: {sorted(available_arms)}"
        )
    return requested


def parse_replica_specs(replica_specs: list[str]) -> list[tuple[str, Path]]:
    """Parse repeatable ``name=path`` replica CLI arguments.

    Args:
        replica_specs: List of ``name=path`` strings.

    Returns:
        List of ``(replica_name, resolved_path)`` tuples.

    Raises:
        ValueError: If any specification lacks ``=`` or points to a missing path.
    """
    parsed: list[tuple[str, Path]] = []
    for spec in replica_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --replica value '{spec}': expected 'name=path'.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not name:
            raise ValueError(f"Empty replica name in '{spec}'.")
        if not path.exists():
            raise ValueError(f"Replica source does not exist: {path}")
        parsed.append((name, path))
    return parsed


def parse_overlay_specs(
    overlay_specs: list[str],
) -> list[tuple[str, set[str] | None, Path]]:
    """Parse repeatable ``name[:arm1,arm2]=path`` answer-overlay CLI arguments.

    Args:
        overlay_specs: List of ``name[:arms]=path`` strings pointing to a
            ``judge_answers`` directory whose arm outputs should be merged into
            an already-staged replica.

    Returns:
        List of ``(replica_name, selected_arms_or_none, resolved_path)`` tuples.

    Raises:
        ValueError: If any specification is malformed or points to a missing directory.
    """
    parsed: list[tuple[str, set[str] | None, Path]] = []
    for spec in overlay_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --replica-answers-overlay value '{spec}': expected 'name[:arms]=path'."
            )
        lhs, raw_path = spec.split("=", 1)
        if ":" in lhs:
            name, arms_str = lhs.split(":", 1)
            arm_set: set[str] | None = {a.strip() for a in arms_str.split(",") if a.strip()}
            if not arm_set:
                raise ValueError(f"Empty arm list in overlay '{spec}'.")
        else:
            name, arm_set = lhs, None
        name = name.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not name:
            raise ValueError(f"Empty replica name in overlay '{spec}'.")
        if not path.is_dir():
            raise ValueError(f"Overlay judge_answers directory does not exist: {path}")
        parsed.append((name, arm_set, path))
    return parsed


def _write_json(path: Path, data: Any) -> None:
    """Write formatted UTF-8 JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_and_transform_file(
    src: Path,
    dst: Path,
    *,
    scrub_gdrive: bool,
    replacements: list[tuple[str, str]],
) -> None:
    """Copy a file to ``dst``, optionally scrubbing paths and applying text replacements."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()

    if suffix == ".json" and (scrub_gdrive or replacements):
        data = json.loads(src.read_text(encoding="utf-8"))
        if scrub_gdrive:
            data = scrub_gdrive_paths(data)
        if replacements:
            data = apply_replacements_to_json(data, replacements)
        _write_json(dst, data)
        return

    if replacements and suffix in TEXT_SUFFIXES:
        text = src.read_text(encoding="utf-8")
        dst.write_text(apply_replacements_to_text(text, replacements), encoding="utf-8")
        return

    shutil.copy2(src, dst)


def stage_canonical_run(
    run_dir: Path,
    stage_root: Path,
    selected_arms: set[str],
    replacements: list[tuple[str, str]],
) -> dict[str, Any]:
    """Copy and sanitize a canonical pipeline run into ``stage_root/<run_id>/``.

    Args:
        run_dir: Path to the source pipeline run (e.g. ``results/thesis-run-02``).
        stage_root: Parent staging directory where ``<run_id>/`` is created.
        selected_arms: Retrieval/answer/judge arms to retain.
        replacements: Ordered ``(old, new)`` participant replacement pairs.

    Returns:
        Summary statistics dictionary for the staged run.
    """
    run_id = run_dir.name
    dst_run = stage_root / run_id
    dst_run.mkdir(parents=True, exist_ok=True)

    included_steps: list[str] = []
    step_file_counts: dict[str, int] = {}
    valid_transcript_ids: set[str] = set()

    for step in PUBLICATION_STEPS:
        src_step = run_dir / step
        if not src_step.is_dir():
            continue
        included_steps.append(step)
        dst_step = dst_run / step
        dst_step.mkdir(parents=True, exist_ok=True)
        count = 0

        # 1. Sanitize run_metadata.json
        src_meta = src_step / "run_metadata.json"
        if src_meta.is_file():
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
            _write_json(dst_step / "run_metadata.json", sanitize_run_metadata(meta))
            count += 1

        # 2. Copy step outputs with stage-specific filtering
        src_outputs = src_step / "outputs"
        dst_outputs = dst_step / "outputs"
        dst_outputs.mkdir(parents=True, exist_ok=True)
        scrub_gdrive = step in GDRIVE_SCRUB_STEPS

        if step == "transcription" and src_outputs.is_dir():
            for tr_file in sorted(src_outputs.glob("*_transcription.json")):
                record = json.loads(tr_file.read_text(encoding="utf-8"))
                if record.get("is_valid") is not True:
                    continue
                file_id = tr_file.name.removesuffix("_transcription.json")
                valid_transcript_ids.add(file_id)
                if scrub_gdrive:
                    record = scrub_gdrive_paths(record)
                if replacements:
                    record = apply_replacements_to_json(record, replacements)
                _write_json(dst_outputs / tr_file.name, record)
                count += 1

        elif step in ARM_STEPS and src_outputs.is_dir():
            for entry in sorted(src_outputs.iterdir()):
                if entry.is_dir():
                    if entry.name not in selected_arms:
                        continue
                    for item in sorted(entry.rglob("*")):
                        if item.is_file():
                            rel = item.relative_to(src_outputs)
                            _copy_and_transform_file(
                                item,
                                dst_outputs / rel,
                                scrub_gdrive=False,
                                replacements=replacements,
                            )
                            count += 1
                elif entry.is_file():
                    _copy_and_transform_file(
                        entry,
                        dst_outputs / entry.name,
                        scrub_gdrive=False,
                        replacements=replacements,
                    )
                    count += 1

        elif src_outputs.is_dir():
            for item in sorted(src_outputs.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(src_outputs)
                if step == "kg":
                    if item.name == "logging.txt":
                        continue
                    if "precompute" in rel.parts and item.name != "manifest.json":
                        continue
                _copy_and_transform_file(
                    item,
                    dst_outputs / rel,
                    scrub_gdrive=scrub_gdrive,
                    replacements=replacements,
                )
                count += 1

        # 3. Copy and trim checkpoint files (skipping .bak backups)
        for ckpt in sorted(src_step.glob("*checkpoint*.json")):
            if ".bak" in ckpt.name:
                continue
            ckpt_data = json.loads(ckpt.read_text(encoding="utf-8"))
            if step == "transcription":
                ckpt_data = filter_transcription_checkpoint(ckpt_data, valid_transcript_ids)
            elif step in ARM_STEPS:
                ckpt_data = filter_arm_checkpoint(ckpt_data, selected_arms)
            _write_json(dst_step / ckpt.name, ckpt_data)
            count += 1

        step_file_counts[step] = count

    # 4. Write updated pipeline.json
    src_pipeline = run_dir / "pipeline.json"
    if src_pipeline.is_file():
        pipeline_data = json.loads(src_pipeline.read_text(encoding="utf-8"))
        pipeline_data["steps_run"] = included_steps
        _write_json(dst_run / "pipeline.json", pipeline_data)

    return {
        "run_id": run_id,
        "included_steps": included_steps,
        "selected_arms": sorted(selected_arms),
        "valid_transcripts": len(valid_transcript_ids),
        "step_file_counts": step_file_counts,
    }


def stage_replicas(
    replicas: list[tuple[str, Path]],
    stage_root: Path,
    replacements: list[tuple[str, str]],
) -> dict[str, int]:
    """Extract, sanitize, and stage judge-retest replicas under ``stage_root/judge-retest/``.

    Args:
        replicas: List of ``(replica_name, source_path)`` where ``source_path``
            is either a directory or a ``.tar.gz`` archive.
        stage_root: Parent staging directory.
        replacements: Ordered ``(old, new)`` participant replacement pairs.

    Returns:
        Mapping of replica name to number of staged files.
    """
    retest_root = stage_root / "judge-retest"
    retest_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for name, src in replicas:
        dst_rep = retest_root / name
        dst_rep.mkdir(parents=True, exist_ok=True)
        file_count = 0

        with tempfile.TemporaryDirectory() as tmp:
            if src.is_dir():
                source_dir = src
            else:
                tmp_dir = Path(tmp)
                with tarfile.open(src, "r:*") as tf:
                    tf.extractall(tmp_dir, filter="data")
                source_dir = tmp_dir

            for item in sorted(source_dir.rglob("*")):
                if not item.is_file():
                    continue
                if ".bak" in item.name or item.name == "logging.txt":
                    continue
                rel = item.relative_to(source_dir)
                dst = dst_rep / rel
                if item.name == "run_metadata.json" or (
                    rel.parts[0:2] == ("judge_qa", "history") and item.suffix == ".json"
                ):
                    meta = json.loads(item.read_text(encoding="utf-8"))
                    _write_json(dst, sanitize_run_metadata(meta))
                else:
                    scrub_gdrive = rel.parts[0] in GDRIVE_SCRUB_STEPS if rel.parts else False
                    _copy_and_transform_file(
                        item,
                        dst,
                        scrub_gdrive=scrub_gdrive,
                        replacements=replacements,
                    )
                file_count += 1

        counts[name] = file_count

    return counts


GATE_REPLICA_STEPS: tuple[str, ...] = ("cep", "judge_qa")


def stage_gate_replicas(
    gate_replicas: list[tuple[str, Path]],
    stage_root: Path,
    replacements: list[tuple[str, str]],
) -> dict[str, int]:
    """Stage gate-only replicas (``cep/`` and ``judge_qa/``) under ``stage_root/judge-retest/``.

    Args:
        gate_replicas: List of ``(replica_name, source_path)`` tuples where
            ``source_path`` is a run directory or ``.tar.gz`` archive.
        stage_root: Parent staging directory.
        replacements: Ordered ``(old, new)`` participant replacement pairs.

    Returns:
        Mapping of replica name to number of staged files.
    """
    retest_root = stage_root / "judge-retest"
    retest_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for name, src in gate_replicas:
        dst_rep = retest_root / name
        dst_rep.mkdir(parents=True, exist_ok=True)
        file_count = 0

        with tempfile.TemporaryDirectory() as tmp:
            if src.is_dir():
                source_dir = src
            else:
                tmp_dir = Path(tmp)
                with tarfile.open(src, "r:*") as tf:
                    tf.extractall(tmp_dir, filter="data")
                source_dir = tmp_dir

            for step in GATE_REPLICA_STEPS:
                src_step = source_dir / step
                if not src_step.is_dir():
                    continue
                for item in sorted(src_step.rglob("*")):
                    if not item.is_file():
                        continue
                    if ".bak" in item.name or item.name == "logging.txt":
                        continue
                    rel = item.relative_to(source_dir)
                    dst = dst_rep / rel
                    if item.name == "run_metadata.json" or (
                        rel.parts[0:2] == ("judge_qa", "history") and item.suffix == ".json"
                    ):
                        meta = json.loads(item.read_text(encoding="utf-8"))
                        _write_json(dst, sanitize_run_metadata(meta))
                    else:
                        scrub_gdrive = step in GDRIVE_SCRUB_STEPS
                        _copy_and_transform_file(
                            item,
                            dst,
                            scrub_gdrive=scrub_gdrive,
                            replacements=replacements,
                        )
                    file_count += 1

        counts[name] = file_count

    return counts


def overlay_replica_answers(
    overlays: list[tuple[str, set[str] | None, Path]],
    stage_root: Path,
    replacements: list[tuple[str, str]],
) -> dict[str, int]:
    """Merge extra arm ``judge_answers`` outputs into already-staged judge-retest replicas.

    Updates ``judge_answers/outputs/<arm>/``, unions ``judge_answers_checkpoint.json``,
    and updates item totals plus sanitized ``overlay_runs`` provenance in
    ``judge_answers/run_metadata.json``.

    Args:
        overlays: List of ``(replica_name, arm_filter_or_none, src_judge_answers_dir)``.
        stage_root: Parent staging directory containing ``judge-retest/``.
        replacements: Ordered ``(old, new)`` participant replacement pairs.

    Returns:
        Mapping of ``"<replica_name> (<arms>)"`` to number of overlaid output files.
    """
    retest_root = stage_root / "judge-retest"
    counts: dict[str, int] = {}

    for rep_name, arm_filter, src_ja in overlays:
        dst_ja = retest_root / rep_name / "judge_answers"
        if not dst_ja.is_dir():
            raise ValueError(
                f"Cannot overlay answers onto replica '{rep_name}': "
                f"staged directory not found at {dst_ja}"
            )
        src_outputs = src_ja / "outputs"
        if not src_outputs.is_dir():
            raise ValueError(f"Overlay source missing outputs/ directory: {src_outputs}")

        available = {p.name for p in src_outputs.iterdir() if p.is_dir()}
        selected = available if arm_filter is None else arm_filter
        missing = selected - available
        if missing:
            raise ValueError(
                f"Overlay arms {sorted(missing)} not found in {src_outputs} "
                f"(available: {sorted(available)})"
            )

        added_files = 0
        for arm in sorted(selected):
            src_arm = src_outputs / arm
            dst_arm = dst_ja / "outputs" / arm
            for item in sorted(src_arm.rglob("*")):
                if not item.is_file() or ".bak" in item.name:
                    continue
                rel = item.relative_to(src_arm)
                _copy_and_transform_file(
                    item,
                    dst_arm / rel,
                    scrub_gdrive=False,
                    replacements=replacements,
                )
                added_files += 1

        # Merge judge_answers_checkpoint.json
        dst_ckpt_path = dst_ja / "judge_answers_checkpoint.json"
        src_ckpt_path = src_ja / "judge_answers_checkpoint.json"
        if dst_ckpt_path.is_file() and src_ckpt_path.is_file():
            dst_ckpt = json.loads(dst_ckpt_path.read_text(encoding="utf-8"))
            src_ckpt = filter_arm_checkpoint(
                json.loads(src_ckpt_path.read_text(encoding="utf-8")), selected
            )
            merged_completed = list(
                dict.fromkeys(
                    [
                        *dst_ckpt.get("completed_files", []),
                        *src_ckpt.get("completed_files", []),
                    ]
                )
            )
            merged_failed = {
                **dst_ckpt.get("failed_files", {}),
                **src_ckpt.get("failed_files", {}),
            }
            dst_ckpt["completed_files"] = merged_completed
            dst_ckpt["failed_files"] = merged_failed
            dst_ckpt["total_files"] = len(merged_completed) + len(merged_failed)
            _write_json(dst_ckpt_path, dst_ckpt)

        # Update run_metadata.json totals and attach sanitized overlay provenance
        dst_meta_path = dst_ja / "run_metadata.json"
        src_meta_path = src_ja / "run_metadata.json"
        if dst_meta_path.is_file():
            dst_meta = json.loads(dst_meta_path.read_text(encoding="utf-8"))
            if isinstance(dst_meta.get("total_items"), int):
                dst_meta["total_items"] += added_files
            if isinstance(dst_meta.get("completed_items"), int):
                dst_meta["completed_items"] += added_files
            if src_meta_path.is_file():
                src_meta_clean = sanitize_run_metadata(
                    json.loads(src_meta_path.read_text(encoding="utf-8"))
                )
                overlay_runs = list(dst_meta.get("overlay_runs") or [])
                overlay_runs.append({"arms": sorted(selected), "run_metadata": src_meta_clean})
                dst_meta["overlay_runs"] = overlay_runs
            _write_json(dst_meta_path, dst_meta)

        counts[f"{rep_name} ({','.join(sorted(selected))})"] = added_files

    return counts


def create_tarball(source_dir: Path, tarball_path: Path) -> None:
    """Create a deterministic gzipped tarball rooted at ``source_dir.name``.

    Args:
        source_dir: Directory to archive.
        tarball_path: Destination ``.tar.gz`` path.
    """
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tf:
        tf.add(source_dir, arcname=source_dir.name)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        path: File path to hash.

    Returns:
        Lowercase 64-character SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sanitization_audit(
    staged_run_dir: Path,
    *,
    id_map_applied: bool,
    replacement_count: int,
) -> dict[str, Any]:
    """Audit the staged run for remaining personal metadata or host paths.

    Args:
        staged_run_dir: Staged ``<run_id>/`` directory.
        id_map_applied: Whether a participant identification map was applied.
        replacement_count: Number of replacement entries applied.

    Returns:
        Audit report dictionary intended for private author verification.
    """
    participants: Counter[str] = Counter()
    researchers: Counter[str] = Counter()
    event_contexts: Counter[str] = Counter()
    gdrive_paths_remaining = 0
    host_paths_remaining: list[str] = []

    cep_dir = staged_run_dir / "cep" / "outputs"
    if cep_dir.is_dir():
        for cep_file in sorted(cep_dir.glob("*_cep_qa.json")):
            data = json.loads(cep_file.read_text(encoding="utf-8"))
            sm = data.get("source_metadata") or {}
            if sm.get("participant_name"):
                participants[str(sm["participant_name"])] += 1
            if sm.get("researcher_name"):
                researchers[str(sm["researcher_name"])] += 1
            if sm.get("event_context"):
                event_contexts[str(sm["event_context"])] += 1
            if sm.get("source_gdrive_path"):
                gdrive_paths_remaining += 1

    for meta_file in sorted(staged_run_dir.rglob("run_metadata.json")):
        raw = meta_file.read_text(encoding="utf-8")
        if "/home/" in raw:
            host_paths_remaining.append(str(meta_file.relative_to(staged_run_dir)))

    return {
        "warning": (
            "DO NOT UPLOAD THIS FILE TO ZENODO. Review participant names, event contexts, "
            "and TCLE choices before publishing."
        ),
        "id_map_applied": id_map_applied,
        "replacement_count": replacement_count,
        "distinct_participant_names": len(participants),
        "participant_names": dict(participants.most_common()),
        "distinct_researcher_names": len(researchers),
        "researcher_names": dict(researchers.most_common()),
        "distinct_event_contexts": len(event_contexts),
        "event_contexts": dict(event_contexts.most_common()),
        "gdrive_paths_remaining": gdrive_paths_remaining,
        "run_metadata_with_host_paths": host_paths_remaining,
    }


@app.command()
def main(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help="Canonical pipeline run directory (e.g. results/thesis-run-02).",
        ),
    ] = Path("results/thesis-run-02"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Destination directory for Zenodo-ready files."),
    ] = Path("results/zenodo-release"),
    docs_dir: Annotated[
        Path | None,
        typer.Option(
            "--docs-dir",
            help="Directory containing README.md, DATA_USE_AGREEMENT.md, and LICENSE.",
        ),
    ] = None,
    arms: Annotated[
        str,
        typer.Option(
            "--arms",
            help="Comma-separated retrieval arms to include, or 'all'.",
        ),
    ] = ",".join(DEFAULT_ARMS),
    replica: Annotated[
        list[str] | None,
        typer.Option(
            "--replica",
            help="Repeatable 'name=path' judge-retest replica (directory or .tar.gz).",
        ),
    ] = None,
    gate_replica: Annotated[
        list[str] | None,
        typer.Option(
            "--gate-replica",
            help="Repeatable 'name=path' gate-only replica (stages only cep/ and judge_qa/).",
        ),
    ] = None,
    replica_answers_overlay: Annotated[
        list[str] | None,
        typer.Option(
            "--replica-answers-overlay",
            help=(
                "Repeatable 'name[:arm1,arm2]=path' judge_answers dir to merge into a staged "
                "replica (updates outputs/, checkpoint, and run_metadata)."
            ),
        ),
    ] = None,
    id_map: Annotated[
        Path | None,
        typer.Option(
            "--id-map",
            help="Optional JSON mapping of real names to TCLE pseudonyms/codes.",
        ),
    ] = None,
    keep_staging: Annotated[
        bool,
        typer.Option("--keep-staging", help="Keep unpacked staging directory under out-dir."),
    ] = False,
) -> None:
    """Build clean, Arandu-compatible Zenodo release archives and checksums."""
    run_path = run_dir.expanduser().resolve()
    out_path = out_dir.expanduser().resolve()

    if not run_path.is_dir():
        print_error(f"Run directory not found: {run_path}")
        raise typer.Exit(code=1)

    retrieve_outputs = run_path / "retrieve" / "outputs"
    available_arms = (
        {p.name for p in retrieve_outputs.iterdir() if p.is_dir()}
        if retrieve_outputs.is_dir()
        else set()
    )
    try:
        selected_arms = parse_arms_option(arms, available_arms)
        replica_specs = parse_replica_specs(replica or [])
        gate_replica_specs = parse_replica_specs(gate_replica or [])
        overlay_specs = parse_overlay_specs(replica_answers_overlay or [])
        replacements = load_id_map(id_map.expanduser().resolve() if id_map else None)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if not replacements:
        print_warning(
            "No --id-map provided: participant names from transcripts/metadata are NOT replaced. "
            "Do NOT publish to Zenodo before applying TCLE pseudonymization/anonymization."
        )

    out_path.mkdir(parents=True, exist_ok=True)
    staging_root = out_path / "_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    print_info(f"Staging canonical run {run_path.name} (arms: {sorted(selected_arms)})...")
    summary = stage_canonical_run(run_path, staging_root, selected_arms, replacements)
    print_info(
        f"Staged {summary['valid_transcripts']} valid transcripts across "
        f"{len(summary['included_steps'])} steps."
    )

    uploadable_files: list[Path] = []
    if docs_dir is not None:
        resolved_docs = docs_dir.expanduser().resolve()
        for doc_name in DOC_FILENAMES:
            src_doc = resolved_docs / doc_name
            if src_doc.is_file():
                dst_doc = out_path / doc_name
                shutil.copy2(src_doc, dst_doc)
                uploadable_files.append(dst_doc)
            else:
                print_warning(f"Documentation file not found in {resolved_docs}: {doc_name}")

    canonical_tar = out_path / f"{run_path.name}.tar.gz"
    print_info(f"Archiving {canonical_tar.name}...")
    create_tarball(staging_root / run_path.name, canonical_tar)
    uploadable_files.append(canonical_tar)

    if replica_specs or gate_replica_specs:
        if replica_specs:
            print_info(f"Staging {len(replica_specs)} judge-retest replica(s)...")
            rep_counts = stage_replicas(replica_specs, staging_root, replacements)
            for rep_name, rep_count in rep_counts.items():
                print_info(f"  replica '{rep_name}': {rep_count} files")
        if gate_replica_specs:
            print_info(f"Staging {len(gate_replica_specs)} gate-only replica(s)...")
            gate_counts = stage_gate_replicas(gate_replica_specs, staging_root, replacements)
            for rep_name, rep_count in gate_counts.items():
                print_info(f"  gate replica '{rep_name}': {rep_count} files")
        if overlay_specs:
            print_info(f"Applying {len(overlay_specs)} judge_answers overlay(s)...")
            ov_counts = overlay_replica_answers(overlay_specs, staging_root, replacements)
            for label, ov_count in ov_counts.items():
                print_info(f"  overlay '{label}': {ov_count} files")
        retest_tar = out_path / "judge-retest.tar.gz"
        print_info(f"Archiving {retest_tar.name}...")
        create_tarball(staging_root / "judge-retest", retest_tar)
        uploadable_files.append(retest_tar)

    audit = build_sanitization_audit(
        staging_root / run_path.name,
        id_map_applied=bool(replacements),
        replacement_count=len(replacements),
    )
    audit["summary"] = summary
    audit_path = out_path / "sanitization_audit.json"
    _write_json(audit_path, audit)

    checksum_lines = [f"{sha256_file(p)}  {p.name}" for p in sorted(uploadable_files)]
    checksum_path = out_path / "CHECKSUMS.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if not keep_staging:
        shutil.rmtree(staging_root)

    print_success(f"Zenodo release bundle written to {out_path}")
    print_info(
        f"Audit report (DO NOT UPLOAD): {audit_path} "
        f"(distinct participant_name values: {audit['distinct_participant_names']}, "
        f"remaining gdrive paths: {audit['gdrive_paths_remaining']}, "
        f"metadata host paths: {len(audit['run_metadata_with_host_paths'])})"
    )


if __name__ == "__main__":
    app()
