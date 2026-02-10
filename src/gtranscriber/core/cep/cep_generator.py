"""CEP QA Generator - Main orchestrator for Cognitive Elicitation Pipeline.

Coordinates all three modules of the CEP pipeline:
- Module I: Bloom Scaffolding (question generation by cognitive level)
- Module II: Reasoning & Grounding (reasoning traces and multi-hop detection)
- Module III: LLM-as-a-Judge Validation (quality evaluation)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

from gtranscriber.core.cep.bloom_scaffolding import BloomScaffoldingGenerator
from gtranscriber.core.cep.reasoning import ReasoningEnricher
from gtranscriber.core.cep.validator import QAValidator
from gtranscriber.schemas import EnrichedRecord, QAPairCEP, QAPairValidated, QARecordCEP

if TYPE_CHECKING:
    from gtranscriber.config import CEPConfig, QAConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Context window size for chunking
MAX_CONTEXT_LENGTH = 4000
MIN_CONTEXT_LENGTH = 100


class CEPQAGenerator:
    """Orchestrate CEP pipeline for cognitive knowledge elicitation.

    Combines Bloom scaffolding, reasoning enrichment, and LLM-as-a-Judge
    validation to generate high-quality, cognitively calibrated QA pairs.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        validator_client: LLMClient | None = None,
    ) -> None:
        """Initialize CEP QA generator.

        Args:
            llm_client: Main LLM client for generation.
            qa_config: QA configuration.
            cep_config: CEP configuration.
            validator_client: Optional separate client for validation.
        """
        self.llm_client = llm_client
        self.qa_config = qa_config
        self.cep_config = cep_config
        self.validator_client = validator_client

        # Initialize Module I: Bloom Scaffolding
        self._bloom_generator = BloomScaffoldingGenerator(llm_client, qa_config, cep_config)

        # Initialize Module II: Reasoning Enrichment
        self._reasoning_enricher = ReasoningEnricher(llm_client, cep_config)

        # Initialize Module III: Validation (if enabled)
        self._validator: QAValidator | None = None
        if cep_config.enable_validation and validator_client:
            self._validator = QAValidator(validator_client, cep_config)

        logger.info(
            f"CEPQAGenerator initialized - "
            f"Bloom={cep_config.enable_bloom_scaffolding}, "
            f"ScaffoldingContext={cep_config.enable_scaffolding_context}, "
            f"Reasoning={cep_config.enable_reasoning_traces}, "
            f"Validation={cep_config.enable_validation}"
        )

    def generate_qa_pairs(self, transcription: EnrichedRecord) -> QARecordCEP:
        """Generate CEP-enhanced QA pairs from a transcription.

        Args:
            transcription: EnrichedRecord containing transcription text.

        Returns:
            QARecordCEP with cognitive-level QA pairs.

        Raises:
            ValueError: If transcription text is too short.
        """
        text = transcription.transcription_text.strip()

        if len(text) < MIN_CONTEXT_LENGTH:
            raise ValueError(
                f"Transcription too short for QA generation "
                f"({len(text)} chars < {MIN_CONTEXT_LENGTH})"
            )

        logger.info(f"Generating CEP QA pairs for {transcription.gdrive_id} ({len(text)} chars)")

        # Chunk text if too long
        contexts = self._chunk_text(text)
        logger.debug(f"Split into {len(contexts)} context chunks")

        # Calculate questions per chunk
        questions_per_chunk = max(1, self.qa_config.questions_per_document // len(contexts))

        all_pairs: list[QAPairCEP | QAPairValidated] = []

        for i, context in enumerate(contexts):
            num_questions = questions_per_chunk
            if i == 0:
                # First chunk gets remainder
                num_questions += self.qa_config.questions_per_document % len(contexts)

            # Module I: Bloom Scaffolding Generation
            pairs = self._bloom_generator.generate(context, num_questions)
            logger.debug(f"Chunk {i + 1}: Generated {len(pairs)} pairs")

            # Module II: Reasoning Enrichment
            if self.cep_config.enable_reasoning_traces:
                pairs = self._reasoning_enricher.enrich_batch(pairs, context)
                logger.debug(f"Chunk {i + 1}: Enriched with reasoning traces")

            # Module III: Validation (if enabled)
            if self._validator:
                validated_pairs = self._validator.validate_batch(pairs, context)
                all_pairs.extend(validated_pairs)
                logger.debug(f"Chunk {i + 1}: Validated {len(validated_pairs)} pairs")
            else:
                all_pairs.extend(pairs)

        # Trim to exact count
        all_pairs = all_pairs[: self.qa_config.questions_per_document]

        # Calculate statistics
        bloom_dist = self._calculate_bloom_distribution(all_pairs)
        validation_summary = self._calculate_validation_summary(all_pairs)
        validated_count = self._count_validated_pairs(all_pairs)

        logger.info(
            f"Generated {len(all_pairs)} CEP QA pairs for {transcription.gdrive_id} "
            f"(validated: {validated_count})"
        )

        return QARecordCEP(
            source_gdrive_id=transcription.gdrive_id,
            source_filename=transcription.name,
            transcription_text=text,
            qa_pairs=all_pairs,
            model_id=self.llm_client.model_id,
            validator_model_id=(self.validator_client.model_id if self.validator_client else None),
            provider=self.llm_client.provider.value,  # type: ignore[arg-type]
            language=self.cep_config.language,
            total_pairs=len(all_pairs),
            validated_pairs=validated_count,
            bloom_distribution=bloom_dist,
            validation_summary=validation_summary,
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text into manageable contexts.

        Args:
            text: Full transcription text.

        Returns:
            List of text chunks.
        """
        if len(text) <= MAX_CONTEXT_LENGTH:
            return [text]

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > MAX_CONTEXT_LENGTH:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                if sentence_length > MAX_CONTEXT_LENGTH:
                    chunks.append(sentence[:MAX_CONTEXT_LENGTH])
                    continue

            current_chunk.append(sentence)
            current_length += sentence_length + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _calculate_bloom_distribution(
        self,
        pairs: list[QAPairCEP | QAPairValidated],
    ) -> dict[str, int]:
        """Calculate distribution of QA pairs across Bloom levels.

        Args:
            pairs: List of QA pairs.

        Returns:
            Dictionary mapping Bloom level to count.
        """
        counter = Counter(pair.bloom_level for pair in pairs)
        return dict(counter)

    def _calculate_validation_summary(
        self,
        pairs: list[QAPairCEP | QAPairValidated],
    ) -> dict[str, float] | None:
        """Calculate aggregated validation metrics.

        Args:
            pairs: List of QA pairs.

        Returns:
            Dictionary with average scores, or None if no validation.
        """
        validated_pairs = [
            p for p in pairs if isinstance(p, QAPairValidated) and p.validation is not None
        ]

        if not validated_pairs:
            return None

        n = len(validated_pairs)
        return {
            "avg_faithfulness": sum(p.validation.faithfulness for p in validated_pairs) / n,
            "avg_bloom_calibration": sum(p.validation.bloom_calibration for p in validated_pairs)
            / n,
            "avg_informativeness": sum(p.validation.informativeness for p in validated_pairs) / n,
            "avg_overall_score": sum(p.validation.overall_score for p in validated_pairs) / n,
            "validation_pass_rate": sum(1 for p in validated_pairs if p.is_valid) / n,
        }

    def _count_validated_pairs(
        self,
        pairs: list[QAPairCEP | QAPairValidated],
    ) -> int:
        """Count pairs that passed validation.

        Args:
            pairs: List of QA pairs.

        Returns:
            Number of pairs passing validation threshold.
        """
        return sum(1 for p in pairs if isinstance(p, QAPairValidated) and p.is_valid)
