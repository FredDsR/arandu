"""QA pair generation from transcriptions using LLM-based strategies.

This module implements synthetic question-answer pair generation from
transcription text using various question types (factual, conceptual,
temporal, entity-focused) to create evaluation datasets.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gtranscriber.schemas import EnrichedRecord, QAPair, QARecord

if TYPE_CHECKING:
    from typing import Literal

    from gtranscriber.config import QAConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Context window size for chunking long transcriptions (in characters)
MAX_CONTEXT_LENGTH = 4000

# Minimum context length to attempt QA generation (in characters)
MIN_CONTEXT_LENGTH = 100

# Default prompts directory (relative to package root)
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "qa"


class QAGenerator:
    """Generate question-answer pairs from transcriptions using LLM.

    Supports multiple question generation strategies:
    - factual: Questions about specific facts and details
    - conceptual: Questions about concepts, themes, and ideas
    - temporal: Questions about time, sequence, and chronology
    - entity: Questions focused on entities (people, places, organizations)

    All answers are validated to be extractive from the source context.
    """

    def __init__(self, llm_client: LLMClient, config: QAConfig) -> None:
        """Initialize QA generator.

        Args:
            llm_client: LLM client for generating questions and answers.
            config: QA generation configuration.
        """
        self.llm_client = llm_client
        self.config = config
        self._prompts = self._load_prompts()
        logger.info(
            f"QAGenerator initialized with {llm_client.provider.value}/{llm_client.model_id} "
            f"(language={config.language})"
        )

    def _load_prompts(self) -> dict[str, Any]:
        """Load prompt templates based on language configuration.

        Returns:
            Dictionary containing prompt templates.

        Raises:
            FileNotFoundError: If prompt file not found.
            ValueError: If prompt file is invalid JSON.
        """
        if self.config.prompt_path:
            prompt_file = Path(self.config.prompt_path)
        else:
            prompt_file = DEFAULT_PROMPTS_DIR / f"{self.config.language}.json"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)

        logger.debug(f"Loaded prompts from {prompt_file}")
        return prompts

    def generate_qa_pairs(self, transcription: EnrichedRecord) -> QARecord:
        """Generate QA pairs from a transcription.

        Args:
            transcription: EnrichedRecord containing transcription text.

        Returns:
            QARecord with generated QA pairs.

        Raises:
            ValueError: If transcription text is too short.
        """
        text = transcription.transcription_text.strip()

        if len(text) < MIN_CONTEXT_LENGTH:
            raise ValueError(
                f"Transcription too short for QA generation "
                f"({len(text)} chars < {MIN_CONTEXT_LENGTH})"
            )

        logger.info(
            f"Generating QA pairs for {transcription.gdrive_id} "
            f"({len(text)} chars, {len(self.config.strategies)} strategies)"
        )

        # Chunk text if too long
        contexts = self._chunk_text(text)
        logger.debug(f"Split into {len(contexts)} context chunks")

        # Distribute questions across strategies and contexts
        all_pairs: list[QAPair] = []
        questions_per_strategy = max(
            1, self.config.questions_per_document // len(self.config.strategies)
        )

        for strategy in self.config.strategies:
            strategy_typed: Literal["factual", "conceptual", "temporal", "entity"] = strategy  # type: ignore[assignment]

            # Generate questions for this strategy across contexts
            for i, context in enumerate(contexts):
                # Distribute questions across chunks
                num_questions = questions_per_strategy // len(contexts)
                if i == 0:
                    # First chunk gets remainder
                    num_questions += questions_per_strategy % len(contexts)

                if num_questions == 0:
                    continue

                pairs = self._generate_for_context(
                    context=context,
                    strategy=strategy_typed,
                    num_questions=num_questions,
                )
                all_pairs.extend(pairs)

                # Stop if we've reached target
                if len(all_pairs) >= self.config.questions_per_document:
                    break

            if len(all_pairs) >= self.config.questions_per_document:
                break

        # Trim to exact count
        all_pairs = all_pairs[: self.config.questions_per_document]

        logger.info(f"Generated {len(all_pairs)} QA pairs for {transcription.gdrive_id}")

        return QARecord(
            source_gdrive_id=transcription.gdrive_id,
            source_filename=transcription.name,
            transcription_text=text,
            qa_pairs=all_pairs,
            model_id=self.llm_client.model_id,
            provider=self.llm_client.provider.value,  # type: ignore[arg-type]
            language=self.config.language,
            total_pairs=len(all_pairs),
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text into manageable contexts.

        Args:
            text: Full transcription text.

        Returns:
            List of text chunks, each <= MAX_CONTEXT_LENGTH.
        """
        if len(text) <= MAX_CONTEXT_LENGTH:
            return [text]

        # Split on sentence boundaries (period + space)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > MAX_CONTEXT_LENGTH:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Handle very long sentences
                if sentence_length > MAX_CONTEXT_LENGTH:
                    # Split by character limit
                    chunks.append(sentence[:MAX_CONTEXT_LENGTH])
                    continue

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate_for_context(
        self,
        context: str,
        strategy: Literal["factual", "conceptual", "temporal", "entity"],
        num_questions: int,
    ) -> list[QAPair]:
        """Generate QA pairs for a specific context and strategy.

        Args:
            context: Text context to generate questions from.
            strategy: Question generation strategy.
            num_questions: Number of questions to generate.

        Returns:
            List of QAPair objects.
        """
        prompt = self._build_prompt(context, strategy, num_questions)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            pairs = self._parse_response(response, context, strategy)
            logger.debug(
                f"Generated {len(pairs)} pairs for {strategy} strategy (requested {num_questions})"
            )
            return pairs

        except Exception as e:
            logger.error(f"Failed to generate QA pairs for {strategy} strategy: {e}")
            return []

    def _build_prompt(
        self,
        context: str,
        strategy: Literal["factual", "conceptual", "temporal", "entity"],
        num_questions: int,
    ) -> str:
        """Build prompt for QA generation based on strategy.

        Args:
            context: Source text context.
            strategy: Question generation strategy.
            num_questions: Number of questions to generate.

        Returns:
            Formatted prompt string.
        """
        # Get localized instructions from loaded prompts
        system_instruction = self._prompts["system_instruction"]
        strategy_instruction = self._prompts["strategy_instructions"][strategy]
        output_rules = "\n".join(
            f"{i + 1}. {rule}" for i, rule in enumerate(self._prompts["output_rules"])
        )
        output_format = self._prompts["output_format_instruction"]

        prompt = f"""{system_instruction}

Context:
{context}

Task:
{strategy_instruction}

Generate exactly {num_questions} question-answer pair(s) in JSON format. \
Each pair must follow these rules:
{output_rules}

Output format (JSON array):
[
  {{
    "question": "What is X?",
    "answer": "exact text from context",
    "confidence": 0.95
  }}
]

{output_format}"""

        return prompt

    def _parse_response(
        self,
        response: str,
        context: str,
        strategy: Literal["factual", "conceptual", "temporal", "entity"],
    ) -> list[QAPair]:
        """Parse LLM response into QAPair objects.

        Args:
            response: Raw LLM response text.
            context: Source context for validation.
            strategy: Question generation strategy used.

        Returns:
            List of validated QAPair objects.
        """
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code blocks
            response = re.sub(r"```(?:json)?\n?", "", response)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []

        if not isinstance(data, list):
            logger.warning("Response is not a JSON array")
            return []

        pairs: list[QAPair] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            confidence = item.get("confidence", 0.5)

            # Validate required fields
            if not question or not answer:
                logger.debug("Skipping pair with missing question or answer")
                continue

            # Validate answer is extractive
            if not self._is_extractive(answer, context):
                logger.debug(f"Skipping non-extractive answer: {answer[:50]}...")
                continue

            # Validate confidence range
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    confidence = 0.5
            except (ValueError, TypeError):
                confidence = 0.5

            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    context=context,
                    question_type=strategy,
                    confidence=confidence,
                    start_time=None,
                    end_time=None,
                )
            )

        return pairs

    def _is_extractive(self, answer: str, context: str) -> bool:
        """Check if answer is extractive from context.

        Args:
            answer: Proposed answer text.
            context: Source context.

        Returns:
            True if answer appears in context (case-insensitive).
        """
        # Normalize for comparison
        answer_normalized = answer.lower().strip()
        context_normalized = context.lower()

        # Check for exact substring match
        if answer_normalized in context_normalized:
            return True

        # Check for approximate match (allowing minor differences)
        # Split into words and check for presence
        answer_words = answer_normalized.split()
        if len(answer_words) <= 2:
            # For short answers, require exact match
            return False

        # For longer answers, allow if 80% of words are present
        words_found = sum(1 for word in answer_words if word in context_normalized.split())
        match_ratio = words_found / len(answer_words)

        return match_ratio >= 0.8
