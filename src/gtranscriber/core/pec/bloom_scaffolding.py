"""Module I: Bloom Scaffolding QA Generation.

Generates cognitively calibrated questions using Bloom's taxonomy levels
for structured cognitive scaffolding.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gtranscriber.schemas import QAPairPEC

if TYPE_CHECKING:
    from gtranscriber.config import PECConfig, QAConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default prompts directory (repo_root/prompts/qa/pec)
DEFAULT_PEC_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "pec"
)


class BloomScaffoldingGenerator:
    """Generate QA pairs calibrated to Bloom's taxonomy levels.

    Implements Module I of the PEC pipeline for cognitive scaffolding-based
    QA generation. Questions are distributed across Bloom levels according
    to the configured distribution.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        qa_config: QAConfig,
        pec_config: PECConfig,
    ) -> None:
        """Initialize the Bloom scaffolding generator.

        Args:
            llm_client: LLM client for generation.
            qa_config: QA configuration.
            pec_config: PEC configuration.
        """
        self.llm_client = llm_client
        self.qa_config = qa_config
        self.pec_config = pec_config
        self._prompts = self._load_prompts()
        logger.info(f"BloomScaffoldingGenerator initialized with levels: {pec_config.bloom_levels}")

    def _load_prompts(self) -> dict[str, Any]:
        """Load PEC prompt templates based on language configuration.

        Returns:
            Dictionary containing prompt templates.

        Raises:
            FileNotFoundError: If prompt file not found.
        """
        prompt_file = DEFAULT_PEC_PROMPTS_DIR / f"{self.pec_config.language}.json"

        if not prompt_file.exists():
            raise FileNotFoundError(f"PEC prompt file not found: {prompt_file}")

        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)

        logger.debug(f"Loaded PEC prompts from {prompt_file}")
        return prompts

    def generate(
        self,
        context: str,
        num_questions: int,
    ) -> list[QAPairPEC]:
        """Generate Bloom-calibrated QA pairs from context.

        Args:
            context: Source text context.
            num_questions: Total number of questions to generate.

        Returns:
            List of QAPairPEC with Bloom levels assigned.
        """
        pairs: list[QAPairPEC] = []

        # Calculate questions per Bloom level based on distribution
        level_counts = self._calculate_level_distribution(num_questions)

        for level, count in level_counts.items():
            if count == 0:
                continue

            level_pairs = self._generate_for_level(context, level, count)
            pairs.extend(level_pairs)

            logger.debug(f"Generated {len(level_pairs)} pairs at {level} level")

        return pairs

    def _calculate_level_distribution(self, total: int) -> dict[str, int]:
        """Calculate how many questions to generate per Bloom level.

        Args:
            total: Total number of questions.

        Returns:
            Dictionary mapping Bloom level to question count.
        """
        distribution: dict[str, int] = {}
        remaining = total

        # Get levels from config (only those enabled)
        levels = [
            level
            for level in self.pec_config.bloom_levels
            if level in self.pec_config.bloom_distribution
        ]

        # Calculate counts based on distribution weights
        for i, level in enumerate(levels):
            weight = self.pec_config.bloom_distribution.get(level, 0)

            if i == len(levels) - 1:
                # Last level gets remaining
                distribution[level] = remaining
            else:
                count = int(total * weight)
                distribution[level] = count
                remaining -= count

        return distribution

    def _generate_for_level(
        self,
        context: str,
        bloom_level: str,
        num_questions: int,
    ) -> list[QAPairPEC]:
        """Generate QA pairs for a specific Bloom level.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            num_questions: Number of questions to generate.

        Returns:
            List of QAPairPEC objects.
        """
        prompt = self._build_prompt(context, bloom_level, num_questions)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.qa_config.temperature,
                max_tokens=self.qa_config.max_tokens,
            )

            pairs = self._parse_response(response, context, bloom_level)
            return pairs

        except Exception as e:
            logger.error(f"Failed to generate QA pairs for {bloom_level} level: {e}")
            return []

    def _build_prompt(
        self,
        context: str,
        bloom_level: str,
        num_questions: int,
    ) -> str:
        """Build prompt for Bloom-calibrated QA generation.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            num_questions: Number of questions to generate.

        Returns:
            Formatted prompt string.
        """
        system_instruction = self._prompts["system_instruction"]
        level_info = self._prompts["bloom_levels"].get(bloom_level, {})

        level_description = level_info.get("description", bloom_level)
        level_instruction = level_info.get("instruction", "Generate questions.")
        question_starters = level_info.get("question_starters", [])
        examples = level_info.get("examples", [])

        # Build output rules
        output_rules = "\n".join(
            f"{i + 1}. {rule}" for i, rule in enumerate(self._prompts["output_rules"])
        )

        # Build examples section
        examples_text = ""
        if examples:
            examples_text = "\nExemplos de perguntas:\n" + "\n".join(f"- {ex}" for ex in examples)

        starters_text = ""
        if question_starters:
            starters_text = f"\nIniciadores sugeridos: {', '.join(question_starters)}"

        prompt = f"""{system_instruction}

Nível Cognitivo: {bloom_level.upper()} ({level_description})

Contexto:
{context}

Tarefa:
{level_instruction}
{starters_text}
{examples_text}

Gere exatamente {num_questions} par(es) pergunta-resposta no formato JSON.
Cada par deve seguir estas regras:
{output_rules}

Formato de saída (array JSON):
[
  {{
    "question": "A pergunta gerada",
    "answer": "A resposta baseada no contexto",
    "bloom_level": "{bloom_level}",
    "confidence": 0.85,
    "reasoning_trace": "Conexões lógicas (para analyze/evaluate)",
    "is_multi_hop": false,
    "hop_count": null,
    "tacit_inference": "Conhecimento implícito (opcional)"
  }}
]

{self._prompts["output_format_instruction"]}"""

        return prompt

    def _parse_response(
        self,
        response: str,
        context: str,
        bloom_level: str,
    ) -> list[QAPairPEC]:
        """Parse LLM response into QAPairPEC objects.

        Args:
            response: Raw LLM response text.
            context: Source context for validation.
            bloom_level: Bloom level used for generation.

        Returns:
            List of QAPairPEC objects.
        """
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []

        if not isinstance(data, list):
            logger.warning("Response is not a JSON array")
            return []

        pairs: list[QAPairPEC] = []

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

            # Validate confidence range
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    confidence = 0.5
            except (ValueError, TypeError):
                confidence = 0.5

            # Get optional PEC fields
            reasoning_trace = item.get("reasoning_trace")
            is_multi_hop = item.get("is_multi_hop", False)
            hop_count = item.get("hop_count")
            tacit_inference = item.get("tacit_inference")

            # Validate hop_count
            if hop_count is not None:
                try:
                    hop_count = int(hop_count)
                    if not 1 <= hop_count <= 5:
                        hop_count = None
                except (ValueError, TypeError):
                    hop_count = None

            # Determine question_type based on bloom_level for backward compatibility
            question_type = self._bloom_to_question_type(bloom_level)

            try:
                pair = QAPairPEC(
                    question=question,
                    answer=answer,
                    context=context,
                    question_type=question_type,
                    confidence=confidence,
                    bloom_level=bloom_level,  # type: ignore[arg-type]
                    reasoning_trace=reasoning_trace,
                    is_multi_hop=is_multi_hop,
                    hop_count=hop_count,
                    tacit_inference=tacit_inference,
                )
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Failed to create QAPairPEC: {e}")
                continue

        return pairs

    def _bloom_to_question_type(self, bloom_level: str) -> str:
        """Map Bloom level to legacy question_type for compatibility.

        Args:
            bloom_level: Bloom taxonomy level.

        Returns:
            Legacy question_type string.
        """
        mapping = {
            "remember": "factual",
            "understand": "conceptual",
            "apply": "conceptual",
            "analyze": "conceptual",
            "evaluate": "conceptual",
            "create": "conceptual",
        }
        return mapping.get(bloom_level, "factual")
