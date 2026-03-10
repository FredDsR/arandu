"""Knowledge Graph Construction module.

Provides a framework-agnostic interface for building knowledge graphs
from transcription records.  The default backend is atlas-rag (AutoSchemaKG).

Public API:
    - KGConstructor: Protocol that all backends implement.
    - KGConstructionResult: Framework-agnostic output model.
    - create_kg_constructor: Factory that returns the configured backend.
    - run_batch_kg_construction: Batch orchestrator with results tracking.
"""

from arandu.kg.batch import run_batch_kg_construction
from arandu.kg.factory import create_kg_constructor
from arandu.kg.protocol import KGConstructor
from arandu.kg.schemas import KGConstructionResult

__all__ = [
    "KGConstructionResult",
    "KGConstructor",
    "create_kg_constructor",
    "run_batch_kg_construction",
]
