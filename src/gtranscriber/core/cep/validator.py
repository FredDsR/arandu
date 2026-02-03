"""Module III: LLM-as-a-Judge Validation.

Validates QA pairs for faithfulness, Bloom calibration, and informativeness
using an independent LLM judge.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gtranscriber.schemas import QAPairCEP, QAPairValidated, ValidationScore

if TYPE_CHECKING:
    from gtranscriber.config import CEPConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default validation prompts directory (repo_root/prompts/qa/cep/validation)
DEFAULT_VALIDATION_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "cep" / "validation"
)


class QAValidator:
    """Validate QA pairs using LLM-as-a-Judge approach.

    Implements Module III of the CEP pipeline. Evaluates each QA pair
    on three criteria:
    - Faithfulness: Is the answer grounded in the context?
    - Bloom Calibration: Does the question match the proposed cognitive level?
    - Informativeness: Does the answer reveal non-obvious knowledge?
    """

    def __init__(
        self,
        validator_client: LLMClient,
        cep_config: CEPConfig,
    ) -> None:
        """Initialize validator with separate LLM client.

        Args:
            validator_client: LLM client for validation (can differ from generator).
            cep_config: CEP configuration.
        """
        self.validator_client = validator_client
        self.cep_config = cep_config
        self._prompts = self._load_prompts()
        logger.info(
            f"QAValidator initialized with {validator_client.provider.value}/"
            f"{validator_client.model_id}"
        )

    def _load_prompts(self) -> dict[str, Any]:
        """Load validation prompt templates.

        Returns:
            Dictionary containing validation prompts.
        """
        prompt_file = DEFAULT_VALIDATION_PROMPTS_DIR / f"{self.cep_config.language}.json"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Validation prompt file not found: {prompt_file}")

        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)

        return prompts

    def validate(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> QAPairValidated:
        """Validate a single QA pair.

        Args:
            qa_pair: QA pair to validate.
            context: Source context for grounding check.

        Returns:
            QAPairValidated with validation scores.
        """
        try:
            prompt = self._build_validation_prompt(qa_pair, context)

            response = self.validator_client.generate(
                prompt=prompt,
                temperature=self.cep_config.validator_temperature,
                max_tokens=512,
            )

            scores = self._parse_validation_response(response)

            # Calculate overall score
            overall = self._calculate_overall_score(scores)
            scores.overall_score = overall

            # Determine if valid based on threshold
            is_valid = overall >= self.cep_config.validation_threshold

            return QAPairValidated(
                question=qa_pair.question,
                answer=qa_pair.answer,
                context=qa_pair.context,
                question_type=qa_pair.question_type,
                confidence=qa_pair.confidence,
                start_time=qa_pair.start_time,
                end_time=qa_pair.end_time,
                bloom_level=qa_pair.bloom_level,
                reasoning_trace=qa_pair.reasoning_trace,
                is_multi_hop=qa_pair.is_multi_hop,
                hop_count=qa_pair.hop_count,
                tacit_inference=qa_pair.tacit_inference,
                validation=scores,
                is_valid=is_valid,
            )

        except Exception as e:
            logger.warning(f"Validation failed for QA pair: {e}")
            # Return unvalidated pair
            return QAPairValidated(
                question=qa_pair.question,
                answer=qa_pair.answer,
                context=qa_pair.context,
                question_type=qa_pair.question_type,
                confidence=qa_pair.confidence,
                start_time=qa_pair.start_time,
                end_time=qa_pair.end_time,
                bloom_level=qa_pair.bloom_level,
                reasoning_trace=qa_pair.reasoning_trace,
                is_multi_hop=qa_pair.is_multi_hop,
                hop_count=qa_pair.hop_count,
                tacit_inference=qa_pair.tacit_inference,
                validation=None,
                is_valid=True,  # Default to valid if validation fails
            )

    def validate_batch(
        self,
        qa_pairs: list[QAPairCEP],
        context: str,
    ) -> list[QAPairValidated]:
        """Validate multiple QA pairs.

        Args:
            qa_pairs: List of QA pairs to validate.
            context: Source context.

        Returns:
            List of validated QA pairs.
        """
        return [self.validate(pair, context) for pair in qa_pairs]

    def _build_validation_prompt(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> str:
        """Build prompt for LLM-as-a-Judge validation.

        Args:
            qa_pair: QA pair to validate.
            context: Source context.

        Returns:
            Formatted validation prompt.
        """
        system_instruction = self._prompts["system_instruction"]
        criteria = self._prompts["evaluation_criteria"]
        validation_instruction = self._prompts["validation_instruction"]
        output_format = self._prompts["output_format_instruction"]

        # Build criteria descriptions
        faithfulness_desc = criteria["faithfulness"]["instruction"]
        bloom_desc = criteria["bloom_calibration"]["instruction"]
        informativeness_desc = criteria["informativeness"]["instruction"]

        # Get Bloom level description
        bloom_levels = criteria["bloom_calibration"]["levels"]
        bloom_level_desc = bloom_levels.get(qa_pair.bloom_level, f"Nível {qa_pair.bloom_level}")

        prompt = f"""{system_instruction}

Contexto Original:
{context}

Par Pergunta-Resposta a Avaliar:
- Pergunta: {qa_pair.question}
- Resposta: {qa_pair.answer}
- Nível Bloom Declarado: {qa_pair.bloom_level} ({bloom_level_desc})

{validation_instruction}

Critérios de Avaliação:

1. FAITHFULNESS (Fidelidade): {faithfulness_desc}
   Rubrica:
   - 1.0: Resposta completamente fundamentada no texto
   - 0.7: Resposta principalmente fundamentada com pequenas inferências
   - 0.4: Resposta parcialmente fundamentada com inferências significativas
   - 0.0: Resposta não fundamentada ou alucinada

2. BLOOM_CALIBRATION (Calibração de Bloom): {bloom_desc}
   O nível declarado é "{qa_pair.bloom_level}": {bloom_level_desc}
   Rubrica:
   - 1.0: Pergunta perfeitamente calibrada ao nível declarado
   - 0.7: Pergunta bem calibrada com pequena sobreposição
   - 0.4: Pergunta subcalibrada (exige nível menor)
   - 0.0: Pergunta totalmente descalibrada

3. INFORMATIVENESS (Informatividade): {informativeness_desc}
   Rubrica:
   - 1.0: Revela conhecimento tácito significativo
   - 0.7: Revela conhecimento útil e não-óbvio
   - 0.4: Informação comum mas bem articulada
   - 0.0: Informação trivial ou óbvia

{output_format}"""

        return prompt

    def _parse_validation_response(self, response: str) -> ValidationScore:
        """Parse validation response from LLM judge.

        Args:
            response: Raw LLM response.

        Returns:
            ValidationScore with parsed scores.
        """
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)

        try:
            data = json.loads(response)

            # Extract and validate scores
            faithfulness = self._validate_score(data.get("faithfulness", 0.5))
            bloom_calibration = self._validate_score(data.get("bloom_calibration", 0.5))
            informativeness = self._validate_score(data.get("informativeness", 0.5))
            rationale = data.get("judge_rationale") or data.get("rationale")

            return ValidationScore(
                faithfulness=faithfulness,
                bloom_calibration=bloom_calibration,
                informativeness=informativeness,
                overall_score=0.0,  # Will be calculated
                judge_rationale=rationale,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse validation response: {e}")
            # Return default scores
            return ValidationScore(
                faithfulness=0.5,
                bloom_calibration=0.5,
                informativeness=0.5,
                overall_score=0.5,
                judge_rationale="Falha ao processar resposta do validador",
            )

    def _validate_score(self, value: Any) -> float:
        """Validate and clamp score to [0.0, 1.0].

        Args:
            value: Raw score value.

        Returns:
            Validated float score.
        """
        try:
            score = float(value)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5

    def _calculate_overall_score(self, scores: ValidationScore) -> float:
        """Calculate weighted overall score.

        Args:
            scores: Individual validation scores.

        Returns:
            Weighted overall score.
        """
        return (
            scores.faithfulness * self.cep_config.faithfulness_weight
            + scores.bloom_calibration * self.cep_config.bloom_calibration_weight
            + scores.informativeness * self.cep_config.informativeness_weight
        )
