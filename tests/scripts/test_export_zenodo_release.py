"""Unit tests for ``scripts/export_zenodo_release.py``."""

from __future__ import annotations

import json
import tarfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.export_zenodo_release import (
    apply_replacements_to_json,
    build_sanitization_audit,
    create_tarball,
    filter_arm_checkpoint,
    filter_transcription_checkpoint,
    load_id_map,
    overlay_replica_answers,
    parse_arms_option,
    parse_overlay_specs,
    sanitize_host_path,
    sanitize_run_metadata,
    scrub_gdrive_paths,
    stage_canonical_run,
    stage_gate_replicas,
    stage_replicas,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sanitize_host_path_and_run_metadata() -> None:
    assert (
        sanitize_host_path("/home/users/fdsreckziegel/etno-kgc-preprocessing/results/thesis-run-02")
        == "results/thesis-run-02"
    )
    assert (
        sanitize_host_path("/home/users/fdsreckziegel/etno-kgc-preprocessing/cache/huggingface")
        == "cache/huggingface"
    )

    meta = {
        "execution": {"hostname": "tupi1", "username": "fdsreckziegel"},
        "output_directory": "/home/users/fdsreckziegel/arandu/results/thesis-run-02/answers",
        "checkpoint_file": "/home/users/fdsreckziegel/arandu/results/thesis-run-02/answers/c.json",
        "config": {
            "environment_variables": {
                "ARANDU_RESULTS_DIR": "/home/users/fdsreckziegel/arandu/results",
                "ARANDU_WORKERS": "3",
            }
        },
    }
    cleaned = sanitize_run_metadata(meta)
    assert cleaned["execution"]["hostname"] == "redacted"
    assert cleaned["execution"]["username"] == "redacted"
    assert cleaned["output_directory"] == "results/thesis-run-02/answers"
    assert cleaned["config"]["environment_variables"]["ARANDU_RESULTS_DIR"] == "results/" or (
        cleaned["config"]["environment_variables"]["ARANDU_RESULTS_DIR"].endswith("results")
    )
    assert "/home/" not in json.dumps(cleaned)


def test_scrub_gdrive_paths_and_id_map(tmp_path: Path) -> None:
    nested = {
        "source_metadata": {
            "participant_name": "Maria Silva",
            "source_gdrive_path": "/drive/Maria Silva/entrevista.mov",
        },
        "qa_pairs": [{"question": "O que Maria Silva disse?"}],
    }
    scrubbed = scrub_gdrive_paths(nested)
    assert scrubbed["source_metadata"]["source_gdrive_path"] is None

    map_file = tmp_path / "id_map.json"
    _write_json(map_file, {"Maria": "P01", "Maria Silva": "Participante 01"})
    replacements = load_id_map(map_file)
    assert replacements[0] == ("Maria Silva", "Participante 01")

    replaced = apply_replacements_to_json(scrubbed, replacements)
    assert replaced["source_metadata"]["participant_name"] == "Participante 01"
    assert replaced["qa_pairs"][0]["question"] == "O que Participante 01 disse?"


def test_filter_checkpoints() -> None:
    arm_ckpt = {
        "completed_files": [
            "bm25::cep::item1",
            "dense_bgem3_256::cep::item1",
            "oracle::cep::item2",
        ],
        "failed_files": {"dense_bgem3_256::cep::bad": "err", "bm25::cep::bad": "err"},
        "total_files": 5,
    }
    trimmed = filter_arm_checkpoint(arm_ckpt, {"bm25", "oracle"})
    assert trimmed["completed_files"] == ["bm25::cep::item1", "oracle::cep::item2"]
    assert trimmed["failed_files"] == {"bm25::cep::bad": "err"}
    assert trimmed["total_files"] == 3

    tr_ckpt = {
        "completed_files": ["id1", "id2", "id3"],
        "failed_files": {"id4": "timeout"},
        "total_files": 4,
    }
    trimmed_tr = filter_transcription_checkpoint(tr_ckpt, {"id1", "id3"})
    assert trimmed_tr["completed_files"] == ["id1", "id3"]
    assert trimmed_tr["failed_files"] == {}
    assert trimmed_tr["total_files"] == 2


def test_parse_arms_option_errors() -> None:
    available = {"bm25", "oracle", "null"}
    assert parse_arms_option("all", available) == available
    assert parse_arms_option("bm25, oracle", available) == {"bm25", "oracle"}
    with pytest.raises(ValueError, match="not present"):
        parse_arms_option("bm25,nonexistent", available)


def test_stage_canonical_run_and_replicas(tmp_path: Path) -> None:
    run_dir = tmp_path / "thesis-run-02"
    _write_json(
        run_dir / "pipeline.json",
        {
            "pipeline_id": "thesis-run-02",
            "steps_run": ["transcription", "cep", "kg", "retrieve", "annotation", "human_eval"],
        },
    )
    # 1 valid transcript and 1 rejected transcript
    _write_json(
        run_dir / "transcription/outputs/ok1_transcription.json",
        {
            "file_id": "ok1",
            "is_valid": True,
            "text": "Fala de Joao",
            "source_metadata": {"source_gdrive_path": "/drive/Joao.mov"},
        },
    )
    _write_json(
        run_dir / "transcription/outputs/bad1_transcription.json",
        {"file_id": "bad1", "is_valid": False, "text": "Thanks for watching"},
    )
    _write_json(
        run_dir / "transcription/checkpoint.json",
        {"completed_files": ["ok1", "bad1"], "failed_files": {}, "total_files": 2},
    )
    _write_json(
        run_dir / "cep/outputs/ok1_cep_qa.json",
        {
            "file_id": "ok1",
            "source_metadata": {
                "participant_name": "Joao",
                "researcher_name": "Pesq",
                "event_context": "Entrevista Joao",
                "source_gdrive_path": "/drive/Joao.mov",
            },
            "qa_pairs": [],
        },
    )
    # KG output + logging.txt and binary precompute indexes that must be dropped
    (run_dir / "kg/outputs/concepts").mkdir(parents=True, exist_ok=True)
    (run_dir / "kg/outputs/concepts/logging.txt").write_text("debug log", encoding="utf-8")
    (run_dir / "kg/outputs/graph.graphml").write_text("<node>Joao</node>", encoding="utf-8")
    (run_dir / "kg/outputs/atlas_output/precompute").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "kg/outputs/atlas_output/precompute/manifest.json", {"ok": True})
    (run_dir / "kg/outputs/atlas_output/precompute/node_embeddings.pkl").write_bytes(b"pickle")
    (run_dir / "kg/outputs/atlas_output/precompute/edge_faiss.index").write_bytes(b"faiss")
    # Retrieve with 2 arms, indexes dir, and .bak file
    _write_json(run_dir / "retrieve/outputs/bm25/cep/item1.json", {"text": "Joao"})
    _write_json(run_dir / "retrieve/outputs/dense_tmp/cep/item1.json", {"text": "skip"})
    (run_dir / "retrieve/indexes/bm25_idx").mkdir(parents=True, exist_ok=True)
    (run_dir / "retrieve/indexes/bm25_idx/idx.bin").write_bytes(b"binary")
    _write_json(
        run_dir / "retrieve/retrieve_checkpoint.json",
        {"completed_files": ["bm25::item1", "dense_tmp::item1"], "failed_files": {}},
    )
    (run_dir / "retrieve/retrieve_checkpoint.json.bak-preatlas").write_text("{}", encoding="utf-8")
    _write_json(
        run_dir / "retrieve/run_metadata.json",
        {"execution": {"hostname": "pcad01", "username": "user1"}},
    )
    # Non-publication directories that must be dropped
    _write_json(run_dir / "annotation/outputs/a.json", {})
    _write_json(run_dir / "human_eval/outputs/h.json", {})

    stage_root = tmp_path / "staging"
    summary = stage_canonical_run(
        run_dir,
        stage_root,
        selected_arms={"bm25"},
        replacements=[("Joao", "P01")],
    )
    staged = stage_root / "thesis-run-02"

    assert summary["valid_transcripts"] == 1
    assert not (staged / "transcription/outputs/bad1_transcription.json").exists()
    assert (staged / "transcription/outputs/ok1_transcription.json").is_file()
    assert not (staged / "annotation").exists()
    assert not (staged / "human_eval").exists()
    assert not (staged / "retrieve/indexes").exists()
    assert not (staged / "retrieve/retrieve_checkpoint.json.bak-preatlas").exists()
    assert not (staged / "retrieve/outputs/dense_tmp").exists()
    assert not (staged / "kg/outputs/concepts/logging.txt").exists()
    assert (staged / "kg/outputs/atlas_output/precompute/manifest.json").is_file()
    assert not (staged / "kg/outputs/atlas_output/precompute/node_embeddings.pkl").exists()
    assert not (staged / "kg/outputs/atlas_output/precompute/edge_faiss.index").exists()
    assert "<node>P01</node>" in (staged / "kg/outputs/graph.graphml").read_text(encoding="utf-8")

    audit = build_sanitization_audit(staged, id_map_applied=True, replacement_count=1)
    assert audit["gdrive_paths_remaining"] == 0
    assert audit["run_metadata_with_host_paths"] == []
    assert audit["participant_names"] == {"P01": 1}

    # Test replica staging from a tarball
    rep_src = tmp_path / "rep_src"
    _write_json(rep_src / "emic_judge/outputs/ok1.json", {"text": "Joao"})
    _write_json(
        rep_src / "emic_judge/run_metadata.json",
        {"execution": {"hostname": "node2", "username": "u2"}},
    )
    rep_tar = tmp_path / "r2.tar.gz"
    with tarfile.open(rep_tar, "w:gz") as tf:
        for child in rep_src.iterdir():
            tf.add(child, arcname=child.name)

    rep_counts = stage_replicas([("r2", rep_tar)], stage_root, replacements=[("Joao", "P01")])
    assert rep_counts["r2"] == 2
    assert (
        json.loads((stage_root / "judge-retest/r2/emic_judge/outputs/ok1.json").read_text())["text"]
        == "P01"
    )

    out_tar = tmp_path / "thesis-run-02.tar.gz"
    create_tarball(staged, out_tar)
    assert out_tar.is_file()


def test_gate_replica_and_answers_overlay(tmp_path: Path) -> None:
    stage_root = tmp_path / "staging"
    # Base replica r2 with bm25 only
    r2_src = tmp_path / "r2_src"
    _write_json(r2_src / "judge_answers/outputs/bm25/cep/item1.json", {"answer": "Joao"})
    _write_json(
        r2_src / "judge_answers/judge_answers_checkpoint.json",
        {"completed_files": ["bm25::cep::item1"], "failed_files": {}, "total_files": 1},
    )
    _write_json(
        r2_src / "judge_answers/run_metadata.json",
        {"status": "completed", "total_items": 1, "completed_items": 1},
    )
    stage_replicas([("r2", r2_src)], stage_root, replacements=[("Joao", "P01")])

    # Overlay source with dense_bgem3_256 and oracle (only overlay dense_bgem3_256)
    ov_src = tmp_path / "ov_ja"
    _write_json(ov_src / "outputs/dense_bgem3_256/cep/item1.json", {"answer": "Joao dense"})
    _write_json(ov_src / "outputs/oracle/cep/item1.json", {"answer": "Oracle"})
    _write_json(
        ov_src / "judge_answers_checkpoint.json",
        {
            "completed_files": ["dense_bgem3_256::cep::item1", "oracle::cep::item1"],
            "failed_files": {},
            "total_files": 2,
        },
    )
    _write_json(
        ov_src / "run_metadata.json",
        {
            "status": "completed",
            "execution": {"hostname": "tupi2", "username": "u1"},
            "output_directory": "/home/users/u1/arandu/results/dense-r2/judge_answers",
        },
    )

    specs = parse_overlay_specs([f"r2:dense_bgem3_256={ov_src}"])
    counts = overlay_replica_answers(specs, stage_root, replacements=[("Joao", "P01")])
    assert counts == {"r2 (dense_bgem3_256)": 1}

    dst_ja = stage_root / "judge-retest/r2/judge_answers"
    assert not (dst_ja / "outputs/oracle").exists()
    assert json.loads((dst_ja / "outputs/dense_bgem3_256/cep/item1.json").read_text()) == {
        "answer": "P01 dense"
    }
    merged_ckpt = json.loads((dst_ja / "judge_answers_checkpoint.json").read_text())
    assert merged_ckpt["completed_files"] == ["bm25::cep::item1", "dense_bgem3_256::cep::item1"]
    assert merged_ckpt["total_files"] == 2
    merged_meta = json.loads((dst_ja / "run_metadata.json").read_text())
    assert merged_meta["total_items"] == 2
    assert merged_meta["completed_items"] == 2
    assert merged_meta["overlay_runs"][0]["run_metadata"]["execution"]["hostname"] == "redacted"
    assert "/home/" not in json.dumps(merged_meta)

    # Gate-only replica r3-gate
    r3_src = tmp_path / "r3_src"
    _write_json(
        r3_src / "cep/outputs/ok1_cep_qa.json",
        {"source_metadata": {"participant_name": "Joao", "source_gdrive_path": "/g/a.mp4"}},
    )
    _write_json(
        r3_src / "judge_qa/run_metadata.json",
        {"execution": {"hostname": "tupi3", "username": "u1"}},
    )
    _write_json(r3_src / "answers/outputs/bm25/cep/skip.json", {"ignore": True})

    gate_counts = stage_gate_replicas(
        [("r3-gate", r3_src)], stage_root, replacements=[("Joao", "P01")]
    )
    assert gate_counts == {"r3-gate": 2}
    r3g = stage_root / "judge-retest/r3-gate"
    assert not (r3g / "answers").exists()
    cep_data = json.loads((r3g / "cep/outputs/ok1_cep_qa.json").read_text())
    assert cep_data["source_metadata"]["participant_name"] == "P01"
    assert cep_data["source_metadata"]["source_gdrive_path"] is None
