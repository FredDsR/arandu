"""Tests for ``cep_qa_pair_id`` and ``iter_cep_pairs`` in ``arandu.qa.schemas``.

Every consumer of ``results/<id>/cep/outputs/`` (retrieve, gold lookup,
analysis cross-cuts, non-answerable seeding, oracle) walks the directory
through ``iter_cep_pairs``, so its ordering, id scheme and skip rule are the
join contract between stages.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from arandu.qa.schemas import cep_qa_pair_id, iter_cep_pairs

if TYPE_CHECKING:
    from pathlib import Path


def _pair(question: str, chunk_id: str | None, passed: bool | None = None) -> dict:
    pair = {
        "question": question,
        "answer": "Resposta.",
        "context": "Contexto.",
        "question_type": "factual",
        "confidence": 0.9,
        "bloom_level": "remember",
        "chunk_id": chunk_id,
    }
    if passed is not None:
        pair["validation"] = {"passed": passed, "stage_results": {}}
    return pair


def _write_record(cep_dir: Path, source_file_id: str, pairs: list[dict]) -> None:
    record = {
        "source_file_id": source_file_id,
        "source_filename": f"{source_file_id}.mp4",
        "transcription_text": "Texto.",
        "chunker_id": "cep_4k",
        "qa_pairs": pairs,
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": len(pairs),
        "bloom_distribution": {},
    }
    (cep_dir / f"{source_file_id}_cep_qa.json").write_text(json.dumps(record))


class TestCepQaPairId:
    def test_composite_format(self) -> None:
        assert cep_qa_pair_id("file", "chk", 3) == "file:chk:3"

    def test_missing_chunk_renders_none(self) -> None:
        assert cep_qa_pair_id("file", None, 0) == "file:none:0"


class TestIterCepPairs:
    def test_yields_pairs_in_file_then_record_order(self, tmp_path: Path) -> None:
        _write_record(tmp_path, "b", [_pair("B0?", "c0")])
        _write_record(tmp_path, "a", [_pair("A0?", "c0"), _pair("A1?", "c1")])

        refs = list(iter_cep_pairs(tmp_path))

        assert [r.qa_pair_id for r in refs] == ["a:c0:0", "a:c1:1", "b:c0:0"]
        assert [r.idx for r in refs] == [0, 1, 0]
        assert refs[1].pair.question == "A1?"
        assert refs[2].record.source_file_id == "b"

    def test_rejected_pairs_are_yielded_and_keep_their_index(self, tmp_path: Path) -> None:
        # Filtering is the caller's job; the index runs over ALL pairs so a
        # rejected pair never shifts the ids after it.
        _write_record(tmp_path, "a", [_pair("A0?", "c0", passed=False), _pair("A1?", "c0")])

        refs = list(iter_cep_pairs(tmp_path))

        assert [r.qa_pair_id for r in refs] == ["a:c0:0", "a:c0:1"]
        assert refs[0].pair.is_judge_rejected

    def test_unreadable_file_is_logged_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_record(tmp_path, "a", [_pair("A0?", "c0")])
        (tmp_path / "broken.json").write_text("{not json")

        with caplog.at_level(logging.WARNING, logger="arandu.qa.schemas"):
            refs = list(iter_cep_pairs(tmp_path))

        assert [r.qa_pair_id for r in refs] == ["a:c0:0"]
        assert any("broken.json" in r.message for r in caplog.records)

    def test_undecodable_file_is_skipped(self, tmp_path: Path) -> None:
        _write_record(tmp_path, "a", [_pair("A0?", "c0")])
        (tmp_path / "latin1.json").write_bytes(b"\xff\xfe{")

        assert [r.qa_pair_id for r in iter_cep_pairs(tmp_path)] == ["a:c0:0"]

    def test_strict_raises_on_unreadable_file(self, tmp_path: Path) -> None:
        _write_record(tmp_path, "a", [_pair("A0?", "c0")])
        (tmp_path / "broken.json").write_text("{not json")

        with pytest.raises(ValueError):
            list(iter_cep_pairs(tmp_path, strict=True))

    @pytest.mark.parametrize("strict", [False, True])
    def test_repeated_ids_across_files_raise(self, tmp_path: Path, strict: bool) -> None:
        # Two files for the same source collide on every id; no copy is "right".
        _write_record(tmp_path, "a", [_pair("A0?", "c0")])
        (tmp_path / "zz_copy_cep_qa.json").write_text((tmp_path / "a_cep_qa.json").read_text())

        with pytest.raises(ValueError, match="repeats 1 qa_pair_ids"):
            list(iter_cep_pairs(tmp_path, strict=strict))

    def test_missing_dir_yields_nothing(self, tmp_path: Path) -> None:
        assert list(iter_cep_pairs(tmp_path / "absent")) == []
