"""CEP QA Generator - Main orchestrator for Cognitive Elicitation Pipeline.

Coordinates the CEP pipeline modules:
- Module I: Bloom Scaffolding (question generation by cognitive level)
- Module II: Reasoning & Grounding (reasoning traces and multi-hop detection)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from arandu.qa.cep.bloom_scaffolding import BloomScaffoldingGenerator
from arandu.qa.cep.reasoning import ReasoningEnricher
from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.chunking.registry import get_chunker

if TYPE_CHECKING:
    from arandu.qa.config import CEPConfig, QAConfig
    from arandu.shared.chunking.schemas import Chunk
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.schemas import EnrichedRecord

logger = logging.getLogger(__name__)

# Default chunker view used for CEP generation; matches `cep_4k` in the shared registry.
CEP_CHUNKER_ID = "cep_4k"


class CEPQAGenerator:
    """Orchestrate CEP pipeline for cognitive knowledge elicitation.

    Combines Bloom scaffolding and reasoning enrichment to generate
    high-quality, cognitively calibrated QA pairs.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Initialize CEP QA generator.

        Args:
            llm_client: Main LLM client for generation.
            qa_config: QA configuration.
            cep_config: CEP configuration.
        """
        self.llm_client = llm_client
        self.qa_config = qa_config
        self.cep_config = cep_config

        # Initialize Module I: Bloom Scaffolding
        self._bloom_generator = BloomScaffoldingGenerator(llm_client, qa_config, cep_config)

        # Initialize Module II: Reasoning Enrichment
        self._reasoning_enricher = ReasoningEnricher(llm_client, cep_config)

        logger.info(
            f"CEPQAGenerator initialized - "
            f"ScaffoldingContext={cep_config.enable_scaffolding_context}, "
            f"Reasoning={cep_config.enable_reasoning_traces}"
        )

    def generate_qa_pairs(self, transcription: EnrichedRecord) -> QARecordCEP:
        """Generate CEP-enhanced QA pairs from a transcription.

        The caller is responsible for filtering out records that are too
        short to carry extractable content; the transcription judge's
        ``content_length_floor`` heuristic owns that gate upstream.

        Args:
            transcription: EnrichedRecord containing transcription text.

        Returns:
            QARecordCEP with cognitive-level QA pairs.
        """
        text = transcription.transcription_text.strip()

        logger.info(f"Generating CEP QA pairs for {transcription.file_id} ({len(text)} chars)")

        chunks = self._chunk_with_offsets(text, transcription.file_id)
        logger.debug(f"Split into {len(chunks)} context chunks")

        if not chunks:
            return QARecordCEP(
                source_file_id=transcription.file_id,
                source_filename=transcription.name,
                source_metadata=transcription.source_metadata,
                source_metadata_context_enabled=self.cep_config.enable_source_metadata_context,
                transcription_text=text,
                qa_pairs=[],
                chunker_id=CEP_CHUNKER_ID,
                model_id=self.llm_client.model_id,
                provider=self.llm_client.provider.value,  # type: ignore[arg-type]
                language=self.cep_config.language,
                total_pairs=0,
                bloom_distribution={},
            )

        # The full Bloom ladder runs independently on each chunk: every chunk
        # produces ``sum(bloom_distribution.values())`` pairs scaffolded across
        # the Bloom hierarchy (remember -> understand -> analyze -> evaluate), so
        # the earlier-level pairs ground the later-level pairs within that same
        # chunk. The counts are NOT divided across chunks, so each chunk covers
        # every configured level and the document total scales with the chunk
        # count.
        all_pairs: list[QAPairCEP] = []

        for i, chunk in enumerate(chunks):
            context = text[chunk.start_char : chunk.end_char]

            # Module I: Bloom Scaffolding Generation
            pairs = self._bloom_generator.generate(
                context,
                source_metadata=transcription.source_metadata,
            )
            logger.debug(f"Chunk {i + 1}: Generated {len(pairs)} pairs")

            # Module II: Reasoning Enrichment
            if self.cep_config.enable_reasoning_traces:
                pairs = self._reasoning_enricher.enrich_batch(pairs, context)
                logger.debug(f"Chunk {i + 1}: Enriched with reasoning traces")

            # Stamp provenance: every pair generated from this chunk inherits its chunk_id.
            for pair in pairs:
                pair.chunk_id = chunk.chunk_id

            all_pairs.extend(pairs)

        # Calculate statistics
        bloom_dist = self._calculate_bloom_distribution(all_pairs)

        logger.info(f"Generated {len(all_pairs)} CEP QA pairs for {transcription.file_id}")

        return QARecordCEP(
            source_file_id=transcription.file_id,
            source_filename=transcription.name,
            source_metadata=transcription.source_metadata,
            source_metadata_context_enabled=self.cep_config.enable_source_metadata_context,
            transcription_text=text,
            qa_pairs=all_pairs,
            chunker_id=CEP_CHUNKER_ID,
            model_id=self.llm_client.model_id,
            provider=self.llm_client.provider.value,  # type: ignore[arg-type]
            language=self.cep_config.language,
            total_pairs=len(all_pairs),
            bloom_distribution=bloom_dist,
        )

    def _chunk_with_offsets(self, text: str, source_file_id: str) -> list[Chunk]:
        """Slice ``text`` via the shared ``cep_4k`` chunker.

        Args:
            text: Full transcription text.
            source_file_id: ID of the source ``EnrichedRecord`` (stamped on every Chunk).

        Returns:
            Offsets-only chunks covering the input text.
        """
        chunker = get_chunker(CEP_CHUNKER_ID)
        return chunker.chunk(text, source_file_id=source_file_id)

    def _chunk_text(self, text: str) -> list[str]:
        """Return the resolved text of each chunk produced by the ``cep_4k`` chunker.

        Kept as a thin compatibility wrapper around :meth:`_chunk_with_offsets`. New
        callers should prefer the offsets-aware variant so they can stamp
        ``chunk_id`` on generated artefacts.

        Args:
            text: Full transcription text.

        Returns:
            List of text chunks (substrings of ``text``).
        """
        return [text[c.start_char : c.end_char] for c in self._chunk_with_offsets(text, "inline")]

    def _calculate_bloom_distribution(
        self,
        pairs: list[QAPairCEP],
    ) -> dict[str, int]:
        """Calculate distribution of QA pairs across Bloom levels.

        Args:
            pairs: List of QA pairs.

        Returns:
            Dictionary mapping Bloom level to count.
        """
        counter = Counter(pair.bloom_level for pair in pairs)
        return dict(counter)
