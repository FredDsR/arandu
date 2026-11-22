"""Shared source-metadata rendering for generation, judging and annotation.

QA *generation* may inject a source-metadata block (participant, researcher,
location, date, event context) into the prompt. If the *judge* does not see the
same block, answers and questions legitimately grounded in that metadata are
scored as fabricated or context-dependent (false rejections). These helpers are
the single, shared rendering path so generation and the judge cannot drift.

The same asymmetry exists on the emic path, and is closed the same way: the
emic judge (``shared/emic/batch.py``) and the human annotation instrument
(``shared/human_eval``, ``shared/annotation``) both render from here, under the
same gate, so a pair whose answer names the participant is not read as
fabricated by either. Judge and annotator have to be blinded to the *same*
things, or the agreement study measures the difference between the two
instruments instead of the construct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.shared.schemas import SourceMetadata

#: Stands in for an empty metadata block, on every surface that renders one.
#:
#: Both measurement surfaces state unconditionally that their reader sees the
#: interview metadata, and the ruler licenses a name that appears in it as not
#: an addition. For a record carrying none -- or whose generation ran with the
#: gate closed -- an empty block would make that licence uncheckable: a name in
#: the answer could be waved through against a block that was never rendered.
#: Naming the absence keeps the provision verifiable, and keeps the judge and
#: the annotator saying the same thing about the same record.
NO_METADATA_TEXT = "(sem metadados registrados para esta entrevista)"


def format_metadata_lines(metadata: SourceMetadata, language: str) -> str:
    """Format source metadata as bare ``- Label: value`` lines, with no header.

    The single place where the fields, their order and their labels are
    decided; :func:`format_metadata_section` adds the prompt header on top of
    this. Surfaces that carry their own heading (the Label Studio canvas, whose
    sections are titled by the labeling config) consume these lines directly,
    so an annotator and the judge read the same fields in the same order.

    Only non-None fields are included. The Drive path is deliberately never
    rendered: it carries PII-ish folder names and must not reach a prompt or an
    annotator.

    Args:
        metadata: Source metadata to format.
        language: Prompt language (ISO 639-1); ``"pt"`` selects Portuguese
            labels, anything else falls back to English.

    Returns:
        One ``- Label: value`` line per populated field, newline-joined, with
        no leading or trailing newline; ``""`` when no field is populated.
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

    return "\n".join(f"- {label}: {value}" for label, value in fields)


def format_metadata_section(metadata: SourceMetadata, language: str) -> str:
    """Format source metadata as a prompt section.

    The header plus :func:`format_metadata_lines`.

    Args:
        metadata: Source metadata to format.
        language: Prompt language (ISO 639-1); ``"pt"`` selects Portuguese
            labels, anything else falls back to English.

    Returns:
        Formatted metadata section beginning with a leading newline, or an
        empty string when no fields are populated.
    """
    lines = format_metadata_lines(metadata, language)
    if not lines:
        return ""

    header = "Metadados da Entrevista:" if language == "pt" else "Interview Metadata:"
    return f"\n{header}\n{lines}"


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


def render_metadata_lines(
    source_metadata: SourceMetadata | None,
    *,
    enable_metadata: bool,
    language: str,
) -> str:
    """Render the header-less metadata lines iff they should be shown, else ``""``.

    Same gate as :func:`render_metadata_context`, different presentation: this
    is what a surface with its own section heading consumes (the human
    annotation instrument). Sharing the gate is what keeps the annotator and
    the judge blinded to exactly the same thing -- a record whose generation
    ran without metadata shows none on either side.

    Args:
        source_metadata: Source metadata to render, if any.
        enable_metadata: Whether source-metadata context was injected at
            generation time (``QARecordCEP.source_metadata_context_enabled``).
        language: Prompt language (ISO 639-1).

    Returns:
        The ``- Label: value`` lines, or ``""`` when metadata must not be
        shown or has no renderable fields.
    """
    if not enable_metadata or source_metadata is None:
        return ""
    return format_metadata_lines(source_metadata, language)


def build_judge_context(
    source_text: str,
    source_metadata: SourceMetadata | None,
    *,
    enable_metadata: bool,
    language: str,
) -> str:
    """Build the grounding context the judge evaluates against.

    Appends the same metadata section generation used so faithfulness and
    self-containedness are judged against identical grounding. The block is
    placed *after* the source text to match the generation prompt order
    (``$context`` then ``$metadata_section``). Gated on ``enable_metadata`` to
    stay symmetric with generation: when generation injected no metadata,
    neither does the judge.

    Args:
        source_text: Grounding text the pair was generated from. CEP
            generation runs per chunk, so this is the chunk text, not the
            whole transcription; see :func:`build_pair_judge_context`.
        source_metadata: Source metadata carried on the record, if any.
        enable_metadata: Whether source-metadata context was injected at
            generation time (``QARecordCEP.source_metadata_context_enabled``).
        language: Prompt language (ISO 639-1); pass the record's generation
            language so labels match what generation rendered.

    Returns:
        The source text, optionally followed by the metadata block.
    """
    section = render_metadata_context(
        source_metadata, enable_metadata=enable_metadata, language=language
    )
    if not section:
        return source_text

    return f"{source_text}\n\n{section.strip()}"


def build_pair_judge_context(
    pair_context: str,
    transcription_text: str,
    source_metadata: SourceMetadata | None,
    *,
    enable_metadata: bool,
    language: str,
) -> str:
    """Build the grounding context for a single QA pair.

    CEP generation slices the transcription into chunks and runs the Bloom
    ladder on one chunk at a time, persisting that slice on
    ``QAPairCEP.context``. Judging the pair against the whole transcription
    would let ``faithfulness`` pass on evidence the generator never saw, so
    the judge is grounded on the originating chunk instead.

    Args:
        pair_context: ``QAPairCEP.context``, the chunk text generation used.
        transcription_text: Full transcription text of the record, used as a
            fallback for legacy records whose pairs carry no context.
        source_metadata: Source metadata carried on the record, if any.
        enable_metadata: Whether source-metadata context was injected at
            generation time (``QARecordCEP.source_metadata_context_enabled``).
        language: Prompt language (ISO 639-1).

    Returns:
        The chunk text (or the full transcription when the pair carries no
        context), optionally followed by the metadata block.
    """
    return build_judge_context(
        pair_context.strip() or transcription_text,
        source_metadata,
        enable_metadata=enable_metadata,
        language=language,
    )
