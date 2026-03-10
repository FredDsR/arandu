"""Module II: Reasoning and Grounding Enrichment.

Enriches QA pairs with reasoning traces, multi-hop detection, and
tacit knowledge inference explanations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from arandu.qa.schemas import QAPairCEP

if TYPE_CHECKING:
    from arandu.qa.config import CEPConfig
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default prompts directory (repo_root/prompts/qa/cep)
DEFAULT_CEP_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "cep"
)

# Bloom levels that benefit from reasoning traces
REASONING_LEVELS = {"analyze", "evaluate", "create"}


def _parse_bool(value: Any) -> bool:
    """Parse a boolean value from various representations.

    Handles actual booleans, strings ("true", "false", "1", "0"),
    and numeric values (1, 0).

    Args:
        value: Value to parse as boolean.

    Returns:
        Parsed boolean value, defaults to False for unrecognized values.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return False


class ReasoningEnricher:
    """Enrich QA pairs with reasoning traces and multi-hop detection.

    Implements Module II of the CEP pipeline. Adds logical reasoning
    explanations and identifies questions that require connecting
    distant parts of the text.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cep_config: CEPConfig,
    ) -> None:
        """Initialize reasoning enricher.

        Args:
            llm_client: LLM client for reasoning generation.
            cep_config: CEP configuration.
        """
        self.llm_client = llm_client
        self.cep_config = cep_config
        self._prompts = self._load_prompts()
        logger.info("ReasoningEnricher initialized")

    def _load_prompts(self) -> dict[str, Any]:
        """Load CEP prompt templates.

        Returns:
            Dictionary containing prompt templates.
        """
        lang_dir = DEFAULT_CEP_PROMPTS_DIR / self.cep_config.language

        data_file = lang_dir / "data.json"
        template_file = lang_dir / "reasoning.md"

        if not data_file.exists():
            raise FileNotFoundError(f"CEP data file not found: {data_file}")
        if not template_file.exists():
            raise FileNotFoundError(f"CEP template file not found: {template_file}")

        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        data["_template"] = template_file.read_text(encoding="utf-8")
        return data

    def enrich(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> QAPairCEP:
        """Enrich a QA pair with reasoning information.

        Only enriches pairs at analyze/evaluate/create levels if they
        don't already have reasoning traces.

        Args:
            qa_pair: QA pair to enrich.
            context: Full source context.

        Returns:
            Enriched QAPairCEP with reasoning trace and multi-hop info.
        """
        if not self.cep_config.enable_reasoning_traces:
            return qa_pair

        # Only enrich higher-level questions that lack reasoning
        if qa_pair.bloom_level not in REASONING_LEVELS:
            return qa_pair

        if qa_pair.reasoning_trace:
            # Already has reasoning trace
            return qa_pair

        try:
            enriched_data = self._generate_reasoning(qa_pair, context)

            # Create new pair with enriched data
            return QAPairCEP(
                question=qa_pair.question,
                answer=qa_pair.answer,
                context=qa_pair.context,
                question_type=qa_pair.question_type,
                confidence=qa_pair.confidence,
                start_time=qa_pair.start_time,
                end_time=qa_pair.end_time,
                bloom_level=qa_pair.bloom_level,
                reasoning_trace=enriched_data.get("reasoning_trace") or qa_pair.reasoning_trace,
                is_multi_hop=enriched_data.get("is_multi_hop", qa_pair.is_multi_hop),
                hop_count=enriched_data.get("hop_count") or qa_pair.hop_count,
                tacit_inference=enriched_data.get("tacit_inference") or qa_pair.tacit_inference,
                generation_prompt=qa_pair.generation_prompt,
            )

        except Exception as e:
            logger.warning(f"Failed to enrich QA pair: {e}", exc_info=True)
            return qa_pair

    def enrich_batch(
        self,
        qa_pairs: list[QAPairCEP],
        context: str,
    ) -> list[QAPairCEP]:
        """Enrich multiple QA pairs.

        Args:
            qa_pairs: List of QA pairs to enrich.
            context: Full source context.

        Returns:
            List of enriched QAPairCEP objects.
        """
        return [self.enrich(pair, context) for pair in qa_pairs]

    def _generate_reasoning(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> dict[str, Any]:
        """Generate reasoning trace for a QA pair.

        Args:
            qa_pair: QA pair to generate reasoning for.
            context: Full source context.

        Returns:
            Dictionary with reasoning_trace, is_multi_hop, hop_count, tacit_inference.
        """
        prompt = self._build_reasoning_prompt(qa_pair, context)

        result = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,  # Low for consistent reasoning
            max_tokens=self.cep_config.reasoning_max_tokens,
        )

        if result.thinking:
            logger.debug(
                "Thinking captured for reasoning enrichment (%d chars)", len(result.thinking)
            )

        return self._parse_reasoning_response(result.content)

    def _build_reasoning_prompt(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> str:
        """Build prompt for reasoning generation.

        Args:
            qa_pair: QA pair to analyze.
            context: Source context.

        Returns:
            Formatted prompt string.
        """
        template = Template(self._prompts["_template"])
        return template.safe_substitute(
            context=context,
            question=qa_pair.question,
            answer=qa_pair.answer,
            bloom_level=qa_pair.bloom_level,
        )

    def _parse_reasoning_response(self, response: str) -> dict[str, Any]:
        """Parse reasoning response from LLM.

        Args:
            response: Raw LLM response.

        Returns:
            Dictionary with reasoning data.
        """
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)

        logger.debug(
            "Reasoning LLM response (%d chars): %.500s",
            len(response),
            response,
        )

        try:
            data = json.loads(response)

            # Validate and clean data
            result: dict[str, Any] = {
                "reasoning_trace": data.get("reasoning_trace"),
                "is_multi_hop": _parse_bool(data.get("is_multi_hop", False)),
                "hop_count": None,
                "tacit_inference": data.get("tacit_inference"),
            }

            # Validate hop_count
            hop_count = data.get("hop_count")
            if hop_count is not None and result["is_multi_hop"]:
                try:
                    hop_count = int(hop_count)
                    if 1 <= hop_count <= self.cep_config.max_hop_count:
                        result["hop_count"] = hop_count
                except (ValueError, TypeError) as exc:
                    logger.debug(
                        "Ignoring non-integer hop_count in reasoning response: %r (error: %s)",
                        hop_count,
                        exc,
                    )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            return {}
