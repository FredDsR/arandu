"""Tests for ``shared/rag/retrieve/loader.py`` — question stream iteration.

Locks the ``qa_pair_id`` composite format (the spec's
``"<file_id>:<chunk_id>:<idx>"``) and the optional-non-answerable
behaviour (silently skipped when the stage hasn't populated).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.shared.rag.retrieve.loader import load_questions

if TYPE_CHECKING:
    from pathlib import Path


def _write_cep_record(
    cep_dir: Path,
    source_file_id: str,
    qa_pairs: list[dict],
    chunker_id: str = "cep_4k",
) -> None:
    """Write a minimal ``QARecordCEP`` JSON file."""
    record = {
        "source_file_id": source_file_id,
        "source_filename": f"{source_file_id}.mp4",
        "transcription_text": "Texto de teste.",
        "chunker_id": chunker_id,
        "qa_pairs": qa_pairs,
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": len(qa_pairs),
        "bloom_distribution": {},
    }
    (cep_dir / f"{source_file_id}_cep_qa.json").write_text(json.dumps(record))


def _qa_pair(
    question: str,
    answer: str = "Resposta.",
    chunk_id: str | None = None,
    bloom_level: str = "remember",
    passed: bool | None = None,
) -> dict:
    pair = {
        "question": question,
        "answer": answer,
        "context": "Contexto.",
        "question_type": "factual",
        "confidence": 0.9,
        "bloom_level": bloom_level,
        "chunk_id": chunk_id,
    }
    if passed is not None:
        # Minimal JudgePipelineResult: is_valid derives from validation.passed,
        # is_judge_rejected is (is_valid is False). Empty stage_results is valid.
        pair["validation"] = {"stage_results": {}, "passed": passed}
    return pair


class TestLoadQuestionsCepOnly:
    def test_missing_cep_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            load_questions(tmp_path / "absent")

    def test_qa_pair_id_format_with_chunk_id(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(
            cep_dir,
            source_file_id="src_a",
            qa_pairs=[
                _qa_pair("Pergunta 1?", chunk_id="chk_aa"),
                _qa_pair("Pergunta 2?", chunk_id="chk_bb"),
            ],
        )

        questions = load_questions(cep_dir)

        # Composite id follows the schema docstring: <file_id>:<chunk_id>:<idx>.
        assert [q.qa_pair_id for q in questions] == [
            "src_a:chk_aa:0",
            "src_a:chk_bb:1",
        ]
        assert all(q.is_answerable for q in questions)
        assert all(q.source == "cep" for q in questions)
        assert all(q.chunker_id == "cep_4k" for q in questions)

    def test_qa_pair_id_with_none_chunk_id_uses_placeholder(self, tmp_path: Path) -> None:
        # The schema allows QAPairCEP.chunk_id to be None (CEP generator
        # doesn't always pin a chunk). The id must still be uniquely
        # identifiable; we substitute "none" — stable across reruns.
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(
            cep_dir,
            source_file_id="src_b",
            qa_pairs=[_qa_pair("Sem chunk?", chunk_id=None)],
        )

        questions = load_questions(cep_dir)
        assert questions[0].qa_pair_id == "src_b:none:0"

    def test_files_iterated_in_sorted_order(self, tmp_path: Path) -> None:
        # Stable ordering matters for checkpoint resume + cross-run diffs.
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(cep_dir, "src_b", [_qa_pair("B?")])
        _write_cep_record(cep_dir, "src_a", [_qa_pair("A?")])

        questions = load_questions(cep_dir)
        assert [q.source_file_id for q in questions] == ["src_a", "src_b"]

    def test_unreadable_cep_file_skipped(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(cep_dir, "src_ok", [_qa_pair("Ok?")])
        (cep_dir / "src_bad_cep_qa.json").write_text("not json {")

        questions = load_questions(cep_dir)
        assert [q.source_file_id for q in questions] == ["src_ok"]

    def test_judge_rejected_pairs_excluded_idx_preserved(self, tmp_path: Path) -> None:
        # The RAG benchmark must evaluate only judge-valid pairs. A rejected
        # pair (validation.passed=False) is dropped, but idx is taken over ALL
        # pairs so the ids of surviving pairs after it do not shift.
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(
            cep_dir,
            source_file_id="src_a",
            qa_pairs=[
                _qa_pair("Approved 0?", chunk_id="chk", passed=True),
                _qa_pair("Rejected 1?", chunk_id="chk", passed=False),
                _qa_pair("Approved 2?", chunk_id="chk", passed=True),
            ],
        )

        questions = load_questions(cep_dir)

        # Rejected pair dropped; the third pair keeps idx 2 (not renumbered to 1).
        assert [q.qa_pair_id for q in questions] == ["src_a:chk:0", "src_a:chk:2"]
        assert [q.question for q in questions] == ["Approved 0?", "Approved 2?"]

    def test_unjudged_pairs_kept(self, tmp_path: Path) -> None:
        # Unjudged pairs (validation absent, is_valid is None) are NOT rejected,
        # so a run where judge-qa has not run still retrieves every pair.
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(cep_dir, "src_a", [_qa_pair("Unjudged?", chunk_id="chk")])

        questions = load_questions(cep_dir)
        assert [q.qa_pair_id for q in questions] == ["src_a:chk:0"]


class TestLoadQuestionsNonAnswerable:
    def test_absent_nonanswerable_dir_silently_skipped(self, tmp_path: Path) -> None:
        # The non-answerable stage doesn't exist on disk yet for most runs;
        # absence is not an error.
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(cep_dir, "src_a", [_qa_pair("Q?")])

        questions = load_questions(cep_dir, tmp_path / "nonanswerable" / "outputs")
        assert len(questions) == 1
        assert questions[0].source == "cep"

    def test_nonanswerable_items_loaded_when_present(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        _write_cep_record(cep_dir, "src_a", [_qa_pair("CEP question?")])

        na_dir = tmp_path / "nonanswerable"
        na_dir.mkdir()
        (na_dir / "items.json").write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "qa_pair_id": "swap:0001",
                            "question": "Non-answerable question?",
                            "source_file_id": "src_a",
                            "chunker_id": "cep_4k",
                        }
                    ]
                }
            )
        )

        questions = load_questions(cep_dir, na_dir)
        sources = [q.source for q in questions]
        assert sources.count("cep") == 1
        assert sources.count("nonanswerable") == 1
        na_item = next(q for q in questions if q.source == "nonanswerable")
        assert na_item.qa_pair_id == "swap:0001"
        assert na_item.is_answerable is False
