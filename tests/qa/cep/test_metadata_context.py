"""Tests for shared CEP source-metadata context helpers.

These helpers guarantee that QA generation and the QA judge format and inject
source metadata identically, so the judge cannot penalize a generator for
grounding an answer in metadata the judge never saw.
"""

from __future__ import annotations

from arandu.qa.cep.metadata_context import build_judge_context, format_metadata_section
from arandu.shared.schemas import SourceMetadata


class TestFormatMetadataSection:
    """Tests for format_metadata_section()."""

    def test_renders_all_fields_pt(self) -> None:
        """Portuguese labels render for every populated field."""
        metadata = SourceMetadata(
            participant_name="Aida",
            researcher_name="Julia",
            location="DOQUINHAS",
            recording_date="23-04-2026",
            event_context="Aida- 23-04-2026",
        )

        section = format_metadata_section(metadata, "pt")

        assert "Metadados da Entrevista:" in section
        assert "- Participante: Aida" in section
        assert "- Pesquisador(a): Julia" in section
        assert "- Local: DOQUINHAS" in section
        assert "- Data: 23-04-2026" in section
        assert "- Contexto: Aida- 23-04-2026" in section

    def test_renders_english_labels(self) -> None:
        """English labels render when language is 'en'."""
        metadata = SourceMetadata(location="Barra de Pelotas")

        section = format_metadata_section(metadata, "en")

        assert "Interview Metadata:" in section
        assert "- Location: Barra de Pelotas" in section

    def test_skips_none_fields(self) -> None:
        """Fields left as None are omitted entirely."""
        metadata = SourceMetadata(location="DOQUINHAS")

        section = format_metadata_section(metadata, "pt")

        assert "- Local: DOQUINHAS" in section
        assert "Participante" not in section
        assert "Pesquisador" not in section

    def test_empty_when_no_renderable_fields(self) -> None:
        """An all-None metadata renders to an empty string."""
        metadata = SourceMetadata()

        assert format_metadata_section(metadata, "pt") == ""

    def test_never_leaks_source_gdrive_path(self) -> None:
        """The Drive path (PII-ish folder names) must never be rendered."""
        metadata = SourceMetadata(
            location="DOQUINHAS",
            source_gdrive_path="/MyDrive/Secret Folder/aida.mp4",
        )

        section = format_metadata_section(metadata, "pt")

        assert "Secret Folder" not in section
        assert "aida.mp4" not in section


class TestBuildJudgeContext:
    """Tests for build_judge_context()."""

    def test_prepends_metadata_when_enabled(self) -> None:
        """The metadata block is prepended so the judge sees the same grounding."""
        metadata = SourceMetadata(location="DOQUINHAS")
        transcript = "O pescador guardou o barco antes da enchente."

        context = build_judge_context(
            transcript,
            metadata,
            enable_metadata=True,
            language="pt",
        )

        # Location lives only in metadata, never in the transcript.
        assert "DOQUINHAS" not in transcript
        assert "- Local: DOQUINHAS" in context
        assert transcript in context
        # Metadata precedes the transcript.
        assert context.index("DOQUINHAS") < context.index(transcript)

    def test_returns_plain_transcript_when_disabled(self) -> None:
        """With the flag off, generation injected nothing, so neither does the judge."""
        metadata = SourceMetadata(location="DOQUINHAS")
        transcript = "O pescador guardou o barco."

        context = build_judge_context(
            transcript,
            metadata,
            enable_metadata=False,
            language="pt",
        )

        assert context == transcript

    def test_returns_plain_transcript_when_no_metadata(self) -> None:
        """A record without source_metadata yields the bare transcript."""
        transcript = "O pescador guardou o barco."

        context = build_judge_context(
            transcript,
            None,
            enable_metadata=True,
            language="pt",
        )

        assert context == transcript

    def test_returns_plain_transcript_when_metadata_empty(self) -> None:
        """All-None metadata adds nothing to the context."""
        transcript = "O pescador guardou o barco."

        context = build_judge_context(
            transcript,
            SourceMetadata(),
            enable_metadata=True,
            language="pt",
        )

        assert context == transcript
