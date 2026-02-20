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

from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gtranscriber.schemas import QAPairCEP, SourceMetadata

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


class LLMResponseError(Exception):
    """Recoverable LLM response error (JSON parse, format, or validation failure)."""


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
        *,
        source_metadata: SourceMetadata | None = None,
    ) -> list[QAPairCEP]:
        """Generate Bloom-calibrated QA pairs from context.

        Args:
            context: Source text context.
            num_questions: Total number of questions to generate.
            source_metadata: Optional source metadata for prompt enrichment.

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
                source_metadata=source_metadata,
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
        source_metadata: SourceMetadata | None = None,
    ) -> list[QAPairCEP]:
        """Generate QA pairs for a specific Bloom level.

        Generates one QA pair per LLM call to ensure consistent quality and
        clean attribution of model thinking traces to individual pairs.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            num_questions: Number of questions to generate.
            prior_pairs: Previously generated QA pairs for scaffolding context.
            source_metadata: Optional source metadata for prompt enrichment.

        Returns:
            List of QAPairCEP objects.
        """
        pairs: list[QAPairCEP] = []
        all_prior_pairs = prior_pairs if prior_pairs else []

        for pair_index in range(num_questions):
            # Build scaffolding context from prior pairs (including current level);
            # downstream formatting may truncate to max_scaffolding_pairs.
            current_prior = all_prior_pairs + pairs

            prompt = self._build_prompt(
                context,
                bloom_level,
                prior_pairs=current_prior,
                source_metadata=source_metadata,
            )

            try:
                pair = self._generate_single_pair(
                    prompt=prompt,
                    context=context,
                    bloom_level=bloom_level,
                )
                pairs.append(pair)
                logger.debug(
                    "Generated pair %d/%d at %s level",
                    pair_index + 1,
                    num_questions,
                    bloom_level,
                )
            except LLMResponseError:
                logger.warning(
                    "Retries exhausted for pair %d/%d at %s level; skipping",
                    pair_index + 1,
                    num_questions,
                    bloom_level,
                    exc_info=True,
                )
                continue
            except Exception as e:
                logger.error(
                    "Failed to generate pair %d/%d for %s level: %s",
                    pair_index + 1,
                    num_questions,
                    bloom_level,
                    e,
                )
                continue

        if len(pairs) < num_questions:
            logger.warning(
                "Generated %d/%d pairs for %s level (some failed)",
                len(pairs),
                num_questions,
                bloom_level,
            )

        return pairs

    @retry(
        retry=retry_if_exception_type(LLMResponseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_single_pair(
        self,
        prompt: str,
        context: str,
        bloom_level: str,
    ) -> QAPairCEP:
        """Generate and validate a single QA pair via LLM call.

        Retries up to 3 times on recoverable LLM response errors (JSON parse
        failures, unexpected formats, validation errors).

        Args:
            prompt: Formatted prompt string.
            context: Source text context.
            bloom_level: Bloom taxonomy level.

        Returns:
            Validated QAPairCEP object.

        Raises:
            LLMResponseError: If response cannot be parsed/validated after retries.
        """
        result = self.llm_client.generate(
            prompt=prompt,
            temperature=self.qa_config.temperature,
            max_tokens=self.qa_config.max_tokens,
        )

        if result.thinking:
            logger.debug(
                "Thinking captured for %s level (%d chars)",
                bloom_level,
                len(result.thinking),
            )

        return self._parse_response(
            result.content,
            context,
            bloom_level,
            generation_prompt=prompt,
            generation_thinking=result.thinking,
        )

    def _build_prompt(
        self,
        context: str,
        bloom_level: str,
        *,
        prior_pairs: list[QAPairCEP] | None = None,
        source_metadata: SourceMetadata | None = None,
    ) -> str:
        """Build prompt for Bloom-calibrated QA generation.

        Args:
            context: Source text context.
            bloom_level: Bloom taxonomy level.
            prior_pairs: Previously generated QA pairs for scaffolding context.
            source_metadata: Optional source metadata for prompt enrichment.

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

        # Build metadata section (if enabled and available)
        metadata_section = ""
        if self.cep_config.enable_source_metadata_context and source_metadata is not None:
            metadata_section = self._format_metadata_section(source_metadata)

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
            metadata_section=metadata_section,
            output_rules=output_rules,
        )

    def _parse_response(
        self,
        response: str,
        context: str,
        bloom_level: str,
        *,
        generation_prompt: str | None = None,
        generation_thinking: str | None = None,
    ) -> QAPairCEP:
        """Parse LLM response into a single QAPairCEP.

        Expects a flat JSON object. Pydantic validates all fields.

        Args:
            response: Raw LLM response text.
            context: Source context for enrichment.
            bloom_level: Bloom level used for generation.
            generation_prompt: LLM prompt used to generate the response.
            generation_thinking: Model thinking trace for this pair.

        Returns:
            Validated QAPairCEP object.

        Raises:
            LLMResponseError: If JSON parsing or model validation fails.
        """
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)

        logger.debug(
            "LLM response for %s level (%d chars): %.500s",
            bloom_level,
            len(response),
            response,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise LLMResponseError(f"Expected JSON object, got {type(data).__name__}")

        question_type = self._bloom_to_question_type(bloom_level)

        try:
            # Merge: explicit values override any LLM-echoed duplicates
            return QAPairCEP(
                **{
                    **data,
                    "context": context,
                    "question_type": question_type,
                    "bloom_level": bloom_level,
                    "generation_prompt": generation_prompt,
                    "generation_thinking": generation_thinking,
                },
            )
        except (ValidationError, TypeError) as e:
            raise LLMResponseError(f"Validation failed: {e}") from e

    def _format_metadata_section(self, metadata: SourceMetadata) -> str:
        """Format source metadata as a prompt section.

        Only includes non-None fields. Language-aware labels based on
        the configured CEP language.

        Args:
            metadata: Source metadata to format.

        Returns:
            Formatted metadata section string, or empty string if no fields.
        """
        is_pt = self.cep_config.language == "pt"

        fields: list[tuple[str, str]] = []
        if metadata.participant_name:
            label = "Participante" if is_pt else "Participant"
            fields.append((label, metadata.participant_name))
        if metadata.researcher_name:
            label = "Pesquisador(a)" if is_pt else "Researcher"
            fields.append((label, metadata.researcher_name))
        if metadata.location:
            label = "Local" if is_pt else "Location"
            fields.append((label, metadata.location))
        if metadata.recording_date:
            label = "Data" if is_pt else "Date"
            fields.append((label, metadata.recording_date))
        if metadata.event_context:
            label = "Contexto" if is_pt else "Context"
            fields.append((label, metadata.event_context))

        if not fields:
            return ""

        header = "Metadados da Entrevista:" if is_pt else "Interview Metadata:"
        lines = [f"- {label}: {value}" for label, value in fields]
        return f"\n{header}\n" + "\n".join(lines)

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
