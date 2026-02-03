"""CEP - Cognitive Elicitation Pipeline.

Cognitive scaffolding QA generation based on Bloom's Taxonomy with
LLM-as-a-Judge validation.

Modules:
- bloom_scaffolding: Generate Bloom-calibrated QA pairs
- reasoning: Enrich QA pairs with reasoning traces
- validator: LLM-as-a-Judge validation
- cep_generator: Main orchestrator
"""

from gtranscriber.core.cep.bloom_scaffolding import BloomScaffoldingGenerator
from gtranscriber.core.cep.cep_generator import CEPQAGenerator
from gtranscriber.core.cep.reasoning import ReasoningEnricher
from gtranscriber.core.cep.validator import QAValidator

__all__ = [
    "BloomScaffoldingGenerator",
    "CEPQAGenerator",
    "QAValidator",
    "ReasoningEnricher",
]
