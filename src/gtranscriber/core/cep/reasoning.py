"""Module II: Reasoning and Grounding Enrichment.

Enriches QA pairs with reasoning traces, multi-hop detection, and
tacit knowledge inference explanations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gtranscriber.schemas import QAPairCEP

if TYPE_CHECKING:
    from gtranscriber.config import CEPConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default prompts directory (repo_root/prompts/qa/cep)
DEFAULT_CEP_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "cep"
)

# Bloom levels that benefit from reasoning traces
REASONING_LEVELS = {"analyze", "evaluate", "create"}


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
        prompt_file = DEFAULT_CEP_PROMPTS_DIR / f"{self.cep_config.language}.json"

        if not prompt_file.exists():
            raise FileNotFoundError(f"CEP prompt file not found: {prompt_file}")

        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)

        return prompts

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
            )

        except Exception as e:
            logger.warning(f"Failed to enrich QA pair: {e}")
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

        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,  # Low for consistent reasoning
            max_tokens=512,
        )

        return self._parse_reasoning_response(response)

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
        reasoning_instruction = self._prompts.get(
            "reasoning_instruction",
            "Explain the logical connections between facts that lead to the answer.",
        )
        tacit_instruction = self._prompts.get(
            "tacit_inference_instruction",
            "Identify implicit knowledge used in the answer.",
        )

        prompt = f"""Analise o seguinte par pergunta-resposta e forneça informações de raciocínio.

Contexto:
{context}

Pergunta: {qa_pair.question}
Resposta: {qa_pair.answer}
Nível Bloom: {qa_pair.bloom_level}

Tarefas:
1. {reasoning_instruction}
2. Determine se a resposta requer conectar informações de partes distantes do texto (multi-hop).
3. Se for multi-hop, indique quantos "saltos" de raciocínio são necessários (1-5).
4. {tacit_instruction}

Retorne APENAS um objeto JSON no seguinte formato:
{{
  "reasoning_trace": "Fato A + Fato B → Conclusão",
  "is_multi_hop": true/false,
  "hop_count": 2,
  "tacit_inference": "Conhecimento implícito identificado"
}}"""

        return prompt

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

        try:
            data = json.loads(response)

            # Validate and clean data
            result: dict[str, Any] = {
                "reasoning_trace": data.get("reasoning_trace"),
                "is_multi_hop": bool(data.get("is_multi_hop", False)),
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
