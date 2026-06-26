"""Shared source-metadata context helpers for CEP generation and judging.

QA *generation* may inject a source-metadata block (participant, researcher,
location, date, event context) into the prompt. If the *judge* does not see the
same block, answers and questions legitimately grounded in that metadata are
scored as fabricated or context-dependent (false rejections). These helpers are
the single, shared rendering path so generation and the judge cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.shared.schemas import SourceMetadata


def format_metadata_section(metadata: SourceMetadata, language: str) -> str:
    """Format source metadata as a prompt section.

    Only non-None fields are included, with language-aware labels. The
    Drive path is deliberately never rendered: it carries PII-ish folder
    names and must not reach the prompt.

    Args:
        metadata: Source metadata to format.
        language: Prompt language (ISO 639-1); ``"pt"`` selects Portuguese
            labels, anything else falls back to English.

    Returns:
        Formatted metadata section beginning with a leading newline, or an
        empty string when no fields are populated.
    """
    is_pt = language == "pt"

    fields: list[tuple[str, str]] = []
    if metadata.participant_name:
        fields.append(("Participante" if is_pt else "Participant", metadata.participant_name))
    if metadata.researcher_name:
        fields.append(("Pesquisador(a)" if is_pt else "Researcher", metadata.researcher_name))
    if metadata.location:
        fields.append(("Local" if is_pt else "Location", metadata.location))
    if metadata.recording_date:
        fields.append(("Data" if is_pt else "Date", metadata.recording_date))
    if metadata.event_context:
        fields.append(("Contexto" if is_pt else "Context", metadata.event_context))

    if not fields:
        return ""

    header = "Metadados da Entrevista:" if is_pt else "Interview Metadata:"
    lines = [f"- {label}: {value}" for label, value in fields]
    return f"\n{header}\n" + "\n".join(lines)


def render_metadata_context(
    source_metadata: SourceMetadata | None,
    *,
    enable_metadata: bool,
    language: str,
) -> str:
    """Render the metadata section iff it should be injected, else ``""``.

    Single source of truth for the injection gate (flag on AND metadata
    present) shared by generation and judging, so the two cannot drift on the
    decision of whether to include metadata.

    Args:
        source_metadata: Source metadata to render, if any.
        enable_metadata: Whether source-metadata context is enabled.
        language: Prompt language (ISO 639-1).

    Returns:
        The formatted metadata section, or ``""`` when metadata must not be
        injected or has no renderable fields.
    """
    if not enable_metadata or source_metadata is None:
        return ""
    return format_metadata_section(source_metadata, language)


def build_judge_context(
    transcription_text: str,
    source_metadata: SourceMetadata | None,
    *,
    enable_metadata: bool,
    language: str,
) -> str:
    """Build the grounding context the judge evaluates against.

    Appends the same metadata section generation used so faithfulness and
    self-containedness are judged against identical grounding. The block is
    placed *after* the transcript to match the generation prompt order
    (``$context`` then ``$metadata_section``). Gated on ``enable_metadata`` to
    stay symmetric with generation: when generation injected no metadata,
    neither does the judge.

    Args:
        transcription_text: Full transcription text of the record.
        source_metadata: Source metadata carried on the record, if any.
        enable_metadata: Whether source-metadata context was injected at
            generation time (``QARecordCEP.source_metadata_context_enabled``).
        language: Prompt language (ISO 639-1); pass the record's generation
            language so labels match what generation rendered.

    Returns:
        The transcription text, optionally followed by the metadata block.
    """
    section = render_metadata_context(
        source_metadata, enable_metadata=enable_metadata, language=language
    )
    if not section:
        return transcription_text

    return f"{transcription_text}\n\n{section.strip()}"
