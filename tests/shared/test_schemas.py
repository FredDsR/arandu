"""Tests for cross-domain shared schemas."""

from __future__ import annotations

import json
from typing import Any

from arandu.shared.schemas import EnrichedRecord


def _record_payload(text: str) -> dict[str, Any]:
    """Return a minimal-but-valid EnrichedRecord payload carrying ``text``."""
    return {
        "file_id": "file-1",
        "name": "entrevista.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 10.0,
        "transcription_status": "completed",
    }


class TestTranscriptionTextIsCanonical:
    """transcription_text is the coordinate space every char offset refers to.

    The chunk stage, CEP generation, the answer resolver and the atlas passage
    offsets all stamp offsets against this string. When one of them stripped and
    another did not, they produced disjoint chunk_id namespaces (issue #166).
    """

    def test_strips_leading_whitespace_on_construction(self) -> None:
        record = EnrichedRecord(**_record_payload(" O pescador guardou o barco."))

        assert record.transcription_text == "O pescador guardou o barco."

    def test_strips_trailing_whitespace_on_construction(self) -> None:
        record = EnrichedRecord(**_record_payload("O pescador guardou o barco.\n\n"))

        assert record.transcription_text == "O pescador guardou o barco."

    def test_strips_on_json_round_trip(self) -> None:
        """Already-persisted records load canonical, so no file needs rewriting.

        Whisper prefixes its output with a single space, so every transcription
        on disk carries one leading character that must not reach an offset.
        """
        payload = json.dumps(_record_payload(" O pescador guardou o barco."))

        record = EnrichedRecord.model_validate_json(payload)

        assert record.transcription_text == "O pescador guardou o barco."

    def test_leaves_already_canonical_text_untouched(self) -> None:
        text = "O pescador guardou o barco."

        record = EnrichedRecord(**_record_payload(text))

        assert record.transcription_text == text

    def test_preserves_interior_whitespace(self) -> None:
        """Only the edges are normalized; interior offsets must not move."""
        text = "Primeira fala.\n\nSegunda fala."

        record = EnrichedRecord(**_record_payload(text))

        assert record.transcription_text == text
