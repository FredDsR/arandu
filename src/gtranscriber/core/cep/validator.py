"""Module III: LLM-as-a-Judge Validation.

Validates QA pairs for faithfulness, Bloom calibration, and informativeness
using an independent LLM judge.

Supports both legacy single-call validation and new composable G-Eval-style
judge pipeline (one criterion per LLM call).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from gtranscriber.config import get_judge_config
from gtranscriber.core.judge import JudgePipeline, JudgeRegistry
from gtranscriber.schemas import QAPairCEP, QAPairValidated, ValidationScore

if TYPE_CHECKING:
    from gtranscriber.config import CEPConfig, JudgeConfig
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default validation prompts directory (repo_root/prompts/qa/cep/validation)
DEFAULT_VALIDATION_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "cep" / "validation"
)


class QAValidator:
    """Validate QA pairs using LLM-as-a-Judge approach.

    Implements Module III of the CEP pipeline. Evaluates each QA pair
    on four criteria:
    - Faithfulness: Is the answer grounded in the context?
    - Bloom Calibration: Does the question match the proposed cognitive level?
    - Informativeness: Does the answer reveal non-obvious knowledge?
    - Self-Containedness: Is the question understandable without context?

    Supports two modes:
    1. Composable pipeline (default): One LLM call per criterion (G-Eval approach)
    2. Legacy mode: Single LLM call for all criteria (backward compatibility)
    """

    def __init__(
        self,
        validator_client: LLMClient,
        cep_config: CEPConfig,
        judge_config: JudgeConfig | None = None,
    ) -> None:
        """Initialize validator with separate LLM client.

        Args:
            validator_client: LLM client for validation (can differ from generator).
            cep_config: CEP configuration.
            judge_config: Judge pipeline configuration. If None, loads from env.
        """
        self.validator_client = validator_client
        self.cep_config = cep_config
        self.judge_config = judge_config or get_judge_config()

        # Initialize composable judge pipeline if enabled
        if self.judge_config.use_composable_pipeline:
            self._init_composable_pipeline()
        else:
            # Legacy mode: load single-call prompts
            self._prompts = self._load_prompts()

        logger.info(
            f"QAValidator initialized with {validator_client.provider.value}/"
            f"{validator_client.model_id} "
            f"(composable_pipeline={self.judge_config.use_composable_pipeline})"
        )

    def _init_composable_pipeline(self) -> None:
        """Initialize composable G-Eval-style judge pipeline."""
        # Create registry with judge configuration
        registry = JudgeRegistry(
            llm_client=self.validator_client,
            language=self.judge_config.language,
            temperature=self.judge_config.temperature,
            max_tokens=self.judge_config.max_tokens,
        )

        # Get criteria for CEP validation
        criteria = registry.get_criteria("cep_validation")

        # Build weights dict from CEP config
        weights = {
            "faithfulness": self.cep_config.faithfulness_weight,
            "bloom_calibration": self.cep_config.bloom_calibration_weight,
            "informativeness": self.cep_config.informativeness_weight,
            "self_containedness": self.cep_config.self_containedness_weight,
        }

        # Create pipeline
        self.judge_pipeline = JudgePipeline(criteria=criteria, weights=weights)
        logger.info("Composable judge pipeline initialized")

    def _load_prompts(self) -> dict[str, Any]:
        """Load validation prompt templates.

        Returns:
            Dictionary containing validation prompts.
        """
        lang_dir = DEFAULT_VALIDATION_PROMPTS_DIR / self.cep_config.language

        data_file = lang_dir / "data.json"
        template_file = lang_dir / "prompt.md"

        if not data_file.exists():
            raise FileNotFoundError(f"Validation data file not found: {data_file}")
        if not template_file.exists():
            raise FileNotFoundError(f"Validation template not found: {template_file}")

        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        data["_template"] = template_file.read_text(encoding="utf-8")
        return data

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
            # Use composable pipeline or legacy validation
            if self.judge_config.use_composable_pipeline:
                scores = self._validate_composable(qa_pair, context)
            else:
                scores = self._validate_legacy(qa_pair, context)

            # Safety net: force self_containedness=1.0 for remember level
            if qa_pair.bloom_level == "remember":
                scores.self_containedness = 1.0

            # Calculate overall score (may be already calculated by pipeline)
            if scores.overall_score == 0.0 or not self.judge_config.use_composable_pipeline:
                overall = self._calculate_overall_score(scores)
                scores.overall_score = overall

            # Determine if valid based on threshold
            is_valid = scores.overall_score >= self.cep_config.validation_threshold

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
                generation_prompt=qa_pair.generation_prompt,
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
                generation_prompt=qa_pair.generation_prompt,
                validation=None,
                is_valid=True,  # Default to valid if validation fails
            )

    def _validate_composable(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> ValidationScore:
        """Validate using composable G-Eval-style pipeline.

        Args:
            qa_pair: QA pair to validate.
            context: Source context.

        Returns:
            ValidationScore with individual and overall scores.
        """
        # Load Bloom level descriptions for criterion-specific params
        bloom_level_desc = self._get_bloom_level_desc(qa_pair.bloom_level)

        # Prepare extra params for bloom_calibration criterion
        extra_params = {
            "bloom_calibration": {
                "bloom_level": qa_pair.bloom_level,
                "bloom_level_desc": bloom_level_desc,
            },
            "self_containedness": {
                "bloom_level": qa_pair.bloom_level,
            },
        }

        # Evaluate using pipeline
        return self.judge_pipeline.evaluate(
            context=context,
            question=qa_pair.question,
            answer=qa_pair.answer,
            **extra_params,
        )

    def _validate_legacy(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> ValidationScore:
        """Validate using legacy single-call approach.

        Args:
            qa_pair: QA pair to validate.
            context: Source context.

        Returns:
            ValidationScore with parsed scores.
        """
        prompt = self._build_validation_prompt(qa_pair, context)

        result = self.validator_client.generate(
            prompt=prompt,
            temperature=self.cep_config.validator_temperature,
            max_tokens=self.cep_config.validator_max_tokens,
        )

        return self._parse_validation_response(result.content, thinking=result.thinking)

    def _get_bloom_level_desc(self, bloom_level: str) -> str:
        """Get Bloom level description.

        Args:
            bloom_level: Bloom level name.

        Returns:
            Human-readable description.
        """
        # Try to get from loaded prompts if in legacy mode
        if hasattr(self, "_prompts"):
            bloom_levels = self._prompts.get("bloom_levels", {})
            return bloom_levels.get(bloom_level, bloom_level)

        # Fallback descriptions for composable mode
        descriptions = {
            "remember": "Requer apenas recordar fatos explícitos do texto",
            "understand": "Requer explicar, interpretar ou resumir conceitos",
            "apply": "Requer usar conhecimento em situação nova ou diferente",
            "analyze": "Requer identificar relações, padrões ou conexões entre ideias",
            "evaluate": "Requer fazer julgamentos, avaliar decisões ou justificar escolhas",
            "create": "Requer propor soluções, elaborar ideias ou criar algo novo",
        }
        return descriptions.get(bloom_level, bloom_level)

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
        bloom_levels = self._prompts["bloom_levels"]
        bloom_level_desc = bloom_levels.get(qa_pair.bloom_level, qa_pair.bloom_level)

        template = Template(self._prompts["_template"])
        return template.safe_substitute(
            context=context,
            question=qa_pair.question,
            answer=qa_pair.answer,
            bloom_level=qa_pair.bloom_level,
            bloom_level_desc=bloom_level_desc,
        )

    def _parse_validation_response(
        self,
        response: str,
        *,
        thinking: str | None = None,
    ) -> ValidationScore:
        """Parse validation response from LLM judge.

        Args:
            response: Raw LLM response.
            thinking: Optional thinking trace from the judge model.

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
            self_containedness = self._validate_score(data.get("self_containedness", 1.0))
            rationale = data.get("judge_rationale") or data.get("rationale")

            return ValidationScore(
                faithfulness=faithfulness,
                bloom_calibration=bloom_calibration,
                informativeness=informativeness,
                self_containedness=self_containedness,
                overall_score=0.0,  # Will be calculated
                judge_rationale=rationale,
                judge_thinking=thinking,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse validation response: {e}")
            # Return default scores
            return ValidationScore(
                faithfulness=0.5,
                bloom_calibration=0.5,
                informativeness=0.5,
                self_containedness=1.0,
                overall_score=0.0,  # Recalculated by _calculate_overall_score in validate()
                judge_rationale=self._prompts.get(
                    "parse_error_message", "Falha ao processar resposta do validador"
                ),
                judge_thinking=thinking,
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
            + scores.self_containedness * self.cep_config.self_containedness_weight
        )
