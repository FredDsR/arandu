"""``arandu retrieve`` machinery — settings, factory, loader, batch runner.

This module is the bridge between a populated ``results/<id>/`` run and
the per-arm :class:`Retriever` implementations. Public entry points:

- :func:`run_retrieve_batch` — top-level driver invoked by the CLI.
- :func:`build_retriever` — construct an arm-specific retriever from a
  pipeline id + arm-specific settings.
- :func:`load_questions` — iterate the run's QA + non-answerable items
  (when present) as ``(source, qa_pair_id, question_text, is_answerable)``.

Per-arm settings classes live in :mod:`arandu.shared.rag.retrieve.settings`
and read ``ARANDU_*`` env vars via Pydantic Settings.
"""

from arandu.shared.rag.retrieve.batch import run_retrieve_batch
from arandu.shared.rag.retrieve.factory import build_retriever
from arandu.shared.rag.retrieve.loader import QuestionRecord, load_questions
from arandu.shared.rag.retrieve.settings import (
    ArmName,
    AtlasRagRetrieveSettings,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)

__all__ = [
    "ArmName",
    "AtlasRagRetrieveSettings",
    "Bm25RetrieveSettings",
    "KHopRetrieveSettings",
    "QuestionRecord",
    "build_retriever",
    "load_questions",
    "run_retrieve_batch",
]
