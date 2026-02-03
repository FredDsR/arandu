"""PEC - Pipeline de Elicitação Cognitiva.

Cognitive scaffolding QA generation based on Bloom's Taxonomy with
LLM-as-a-Judge validation.

Modules:
- bloom_scaffolding: Generate Bloom-calibrated QA pairs
- reasoning: Enrich QA pairs with reasoning traces
- validator: LLM-as-a-Judge validation
- pec_generator: Main orchestrator
"""

from gtranscriber.core.pec.bloom_scaffolding import BloomScaffoldingGenerator
from gtranscriber.core.pec.pec_generator import PECQAGenerator
from gtranscriber.core.pec.reasoning import ReasoningEnricher
from gtranscriber.core.pec.validator import QAValidator

__all__ = [
    "BloomScaffoldingGenerator",
    "PECQAGenerator",
    "QAValidator",
    "ReasoningEnricher",
]
