"""Tests for chunking-related fields on QAPairCEP / QARecordCEP."""

from __future__ import annotations

from arandu.qa.schemas import QAPairCEP, QARecordCEP


def _pair(**overrides: object) -> QAPairCEP:
    defaults: dict[str, object] = {
        "question": "Em que ano?",
        "answer": "2024",
        "context": "A enchente de 2024 alagou a vila.",
        "question_type": "factual",
        "confidence": 0.9,
        "bloom_level": "remember",
    }
    defaults.update(overrides)
    return QAPairCEP(**defaults)  # type: ignore[arg-type]


class TestQAPairCEPChunkId:
    def test_chunk_id_optional_defaults_none(self) -> None:
        pair = _pair()
        assert pair.chunk_id is None

    def test_chunk_id_carries_when_provided(self) -> None:
        pair = _pair(chunk_id="a1b2c3d4e5f60718")
        assert pair.chunk_id == "a1b2c3d4e5f60718"


class TestQARecordCEPChunkerId:
    def _record(self, pairs: list[QAPairCEP], **overrides: object) -> QARecordCEP:
        defaults: dict[str, object] = {
            "source_file_id": "src1",
            "source_filename": "entrevista.mp4",
            "transcription_text": "A enchente de 2024 alagou a vila.",
            "qa_pairs": pairs,
            "model_id": "qwen3:14b",
            "provider": "ollama",
            "total_pairs": len(pairs),
        }
        defaults.update(overrides)
        return QARecordCEP(**defaults)  # type: ignore[arg-type]

    def test_chunker_id_defaults_to_cep_4k(self) -> None:
        rec = self._record([_pair()])
        assert rec.chunker_id == "cep_4k"

    def test_chunker_id_carries_when_provided(self) -> None:
        rec = self._record([_pair()], chunker_id="cep_4k")
        assert rec.chunker_id == "cep_4k"

    def test_loading_legacy_json_without_chunker_id_uses_default(self) -> None:
        # Records that pre-date the chunking refactor lack `chunker_id`.
        legacy = {
            "source_gdrive_id": "src1",
            "source_filename": "entrevista.mp4",
            "transcription_text": "A enchente de 2024 alagou a vila.",
            "qa_pairs": [_pair().model_dump()],
            "model_id": "qwen3:14b",
            "provider": "ollama",
            "total_pairs": 1,
        }
        import json

        rec = QARecordCEP.model_validate_json(json.dumps(legacy))
        assert rec.chunker_id == "cep_4k"
