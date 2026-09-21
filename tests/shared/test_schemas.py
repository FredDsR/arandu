"""Tests for cross-domain shared schemas."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from tests.conftest import TranscriptionRecordPayloadBuilder


class TestTranscriptionTextIsCanonical:
    """transcription_text is the coordinate space every char offset refers to.

    The chunk stage, CEP generation, the answer resolver and the atlas passage
    offsets all stamp offsets against this string. When one of them stripped and
    another did not, they produced disjoint chunk_id namespaces (issue #166).
    """

    def test_strips_leading_whitespace_on_construction(
        self, transcription_record_payload: TranscriptionRecordPayloadBuilder
    ) -> None:
        record = EnrichedRecord(**transcription_record_payload(text=" O pescador guardou o barco."))

        assert record.transcription_text == "O pescador guardou o barco."

    def test_strips_trailing_whitespace_on_construction(
        self, transcription_record_payload: TranscriptionRecordPayloadBuilder
    ) -> None:
        record = EnrichedRecord(
            **transcription_record_payload(text="O pescador guardou o barco.\n\n")
        )

        assert record.transcription_text == "O pescador guardou o barco."

    def test_strips_on_json_round_trip(
        self, transcription_record_payload: TranscriptionRecordPayloadBuilder
    ) -> None:
        """Already-persisted records load canonical, so no file needs rewriting.

        Whisper prefixes its output with a single space, so every transcription
        on disk carries one leading character that must not reach an offset.
        """
        payload = json.dumps(transcription_record_payload(text=" O pescador guardou o barco."))

        record = EnrichedRecord.model_validate_json(payload)

        assert record.transcription_text == "O pescador guardou o barco."

    def test_leaves_already_canonical_text_untouched(
        self, transcription_record_payload: TranscriptionRecordPayloadBuilder
    ) -> None:
        text = "O pescador guardou o barco."

        record = EnrichedRecord(**transcription_record_payload(text=text))

        assert record.transcription_text == text

    def test_preserves_interior_whitespace(
        self, transcription_record_payload: TranscriptionRecordPayloadBuilder
    ) -> None:
        """Only the edges are normalized; interior offsets must not move."""
        text = "Primeira fala.\n\nSegunda fala."

        record = EnrichedRecord(**transcription_record_payload(text=text))

        assert record.transcription_text == text
