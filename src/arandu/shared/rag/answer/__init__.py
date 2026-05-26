"""Answerer machinery — LLM with structured output, held constant across retrieval arms.

Spec §5. Public entry points:

- :func:`run_answer_batch` — CLI driver, iterates ``RetrievalRecord``
  artifacts under a run and produces ``AnswerRecord`` artifacts.
- :class:`AnswererClient` — wraps :class:`LLMClient.generate_structured`
  with the spec's malformed-output retry contract (3 attempts with
  ``temperature += 0.1`` each, fallback to an abstained record on
  exhaustion).
- :class:`AnswererSettings` — Pydantic settings (env prefix
  ``ARANDU_ANSWERER_``).
- :func:`pack_passages` — token-budget passage packing.
"""

from arandu.shared.rag.answer.answerer import AnswererClient
from arandu.shared.rag.answer.batch import run_answer_batch
from arandu.shared.rag.answer.packer import pack_passages
from arandu.shared.rag.answer.schemas import AnswererOutput
from arandu.shared.rag.answer.settings import AnswererSettings

__all__ = [
    "AnswererClient",
    "AnswererOutput",
    "AnswererSettings",
    "pack_passages",
    "run_answer_batch",
]
