"""Module I: Bloom Scaffolding QA Generation.

Generates cognitively calibrated questions using Bloom's taxonomy levels
for structured cognitive scaffolding.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from gtranscriber.schemas import QAPairCEP

if TYPE_CHECKING:
    from gtranscriber.config import CEPConfig, QAConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_CEP_PROMPTS_DIR = Path(__file__).parents[4] / "prompts" / "qa" / "cep"

BLOOM_HIERARCHY: list[str] = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]


class BloomScaffoldingGenerator:
    """Generate QA pairs calibrated to Bloom's taxonomy levels.

    Implements Module I of the CEP pipeline for cognitive scaffolding-based
    QA generation. Questions are distributed across Bloom levels according
    to the configured distribution.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Initialize the Bloom scaffolding generator.

        Args:
            llm_client: LLM client for generation.
            qa_config: QA configuration.
            cep_config: CEP configuration.
        """
        self.llm_client = llm_client
        self.qa_config = qa_config
        self.cep_config = cep_config
        self._prompts = self._load_prompts()
        logger.info(f"BloomScaffoldingGenerator initialized with levels: {cep_config.bloom_levels}")

    def _load_prompts(self) -> dict[str, Any]:
        """Load CEP prompt templates based on language configuration.

        Returns:
            Dictionary containing prompt templates.

        Raises:
            FileNotFoundError: If prompt file not found.
        """
        lang_dir = DEFAULT_CEP_PROMPTS_DIR / self.cep_config.language

        data_file = lang_dir / "data.json"
        template_file = lang_dir / "bloom_scaffolding.md"

        if not data_file.exists():
            raise FileNotFoundError(f"CEP data file not found: {data_file}")
        if not template_file.exists():
            raise FileNotFoundError(f"CEP template file not found: {template_file}")

        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        data["_template"] = template_file.read_text(encoding="utf-8")

        logger.debug(f"Loaded CEP prompts from {lang_dir}")
        return data

    def _format_prior_pairs(self, pairs: list[QAPairCEP]) -> str:
        """Format prior QA pairs as scaffolding context for the next level.

        Args:
            pairs: Previously generated QA pairs.

        Returns:
            Formatted string of prior pairs, or empty string if none.
        """
        if not pairs:
            return ""

        # Take the last N pairs, biasing toward the immediately prior level
        max_pairs = self.cep_config.max_scaffolding_pairs
        selected = pairs[-max_pairs:]

        lines = [
            f"{i + 1}. [{pair.bloom_level.upper()}] Q: {pair.question} A: {pair.answer}"
            for i, pair in enumerate(selected)
        ]
        return "\n".join(lines)

    def generate(
        self,
        context: str,
        num_questions: int,
    ) -> list[QAPairCEP]:
        """Generate Bloom-calibrated QA pairs from context.

        Args:
            context: Source text context.
            num_questions: Total number of questions to generate.

        Returns:
            List of QAPairCEP with Bloom levels assigned.
        """
        pairs: list[QAPairCEP] = []

        # Calculate questions per Bloom level based on distribution
        level_counts = self._calculate_level_distribution(num_questions)

        # When scaffolding is enabled, sort levels by Bloom hierarchy
        if self.cep_config.enable_scaffolding_context:
            sorted_levels = [
                (level, level_counts[level]) for level in BLOOM_HIERARCHY if level in level_counts
            ]
        else:
            sorted_levels = list(level_counts.items())

        for level, count in sorted_levels:
            if count == 0:
                continue

            prior = pairs if self.cep_config.enable_scaffolding_context else []
            level_pairs = self._generate_for_level(
                context,
                level,
                count,
                prior_pairs=prior,
            )
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
            for level in self.cep_config.bloom_levels
            if level in self.cep_config.bloom_distribution
        ]

        # Calculate counts based on distribution weights
        for i, level in enumerate(levels):
            weight = self.cep_config.bloom_distribution.get(level, 0)

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
        *,
        prior_pairs: list[QAPairCEP] | None = None,
    ) -> list[QAPairCEP]:
        """Generate QA pairs for a specific Bloom level.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            num_questions: Number of questions to generate.
            prior_pairs: Previously generated QA pairs for scaffolding context.

        Returns:
            List of QAPairCEP objects.
        """
        prompt = self._build_prompt(
            context,
            bloom_level,
            num_questions,
            prior_pairs=prior_pairs,
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.qa_config.temperature,
                max_tokens=self.qa_config.max_tokens,
            )

            pairs = self._parse_response(response, context, bloom_level, generation_prompt=prompt)
            return pairs

        except Exception as e:
            logger.error(f"Failed to generate QA pairs for {bloom_level} level: {e}")
            return []

    def _build_prompt(
        self,
        context: str,
        bloom_level: str,
        num_questions: int,
        *,
        prior_pairs: list[QAPairCEP] | None = None,
    ) -> str:
        """Build prompt for Bloom-calibrated QA generation.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            num_questions: Number of questions to generate.
            prior_pairs: Previously generated QA pairs for scaffolding context.

        Returns:
            Formatted prompt string.
        """
        level_info = self._prompts["bloom_levels"].get(bloom_level, {})

        level_description = level_info.get("description", bloom_level)
        level_instruction = level_info.get("instruction", "Generate questions.")
        question_starters = level_info.get("question_starters", [])
        examples = level_info.get("examples", [])

        # Build output rules
        output_rules = "\n".join(
            f"{i + 1}. {rule}" for i, rule in enumerate(self._prompts["output_rules"])
        )

        # Pre-build conditional sections with localized labels
        starters_section = ""
        if question_starters:
            starters_label = self._prompts.get("starters_label", "Suggested starters")
            starters_section = f"\n{starters_label}: {', '.join(question_starters)}"

        examples_section = ""
        if examples:
            examples_label = self._prompts.get("examples_label", "Question examples")
            examples_section = f"\n{examples_label}:\n" + "\n".join(f"- {ex}" for ex in examples)

        # Build scaffolding section from prior pairs
        scaffolding_section = ""
        if prior_pairs:
            formatted = self._format_prior_pairs(prior_pairs)
            if formatted:
                header = self._prompts.get("scaffolding_header", "")
                scaffolding_section = f"\n{header}\n{formatted}"

        template = Template(self._prompts["_template"])
        return template.safe_substitute(
            bloom_level_upper=bloom_level.upper(),
            bloom_level=bloom_level,
            level_description=level_description,
            context=context,
            level_instruction=level_instruction,
            starters_section=starters_section,
            examples_section=examples_section,
            scaffolding_section=scaffolding_section,
            num_questions=num_questions,
            output_rules=output_rules,
        )

    def _parse_response(
        self,
        response: str,
        context: str,
        bloom_level: str,
        *,
        generation_prompt: str | None = None,
    ) -> list[QAPairCEP]:
        """Parse LLM response into QAPairCEP objects.

        Args:
            response: Raw LLM response text.
            context: Source context for validation.
            bloom_level: Bloom level used for generation.
            generation_prompt: LLM prompt used to generate the response.

        Returns:
            List of QAPairCEP objects.
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

        pairs: list[QAPairCEP] = []

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

            # Get optional CEP fields
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
                pair = QAPairCEP(
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
                    generation_prompt=generation_prompt,
                )
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Failed to create QAPairCEP: {e}")
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
