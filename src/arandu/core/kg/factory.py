"""Factory for creating KG constructor instances."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.core.kg.protocol import KGConstructor
    from arandu.kg.config import KGConfig


def create_kg_constructor(kg_config: KGConfig) -> KGConstructor:
    """Create a KG constructor for the configured backend.

    Uses deferred imports so that backend dependencies (e.g. ``atlas-rag``)
    are only required when the corresponding backend is selected.

    Args:
        kg_config: Knowledge graph configuration with backend selection.

    Returns:
        A ``KGConstructor`` implementation for the selected backend.

    Raises:
        ValueError: If the backend name is not recognized.
        ImportError: If the backend's dependencies are not installed.
    """
    backend = kg_config.backend

    if backend == "atlas":
        if importlib.util.find_spec("atlas_rag") is None:
            raise ImportError(
                "atlas-rag is required for the 'atlas' backend. "
                "Install it with: uv pip install atlas-rag"
            )
        from arandu.core.kg.atlas_backend import AtlasRagConstructor

        return AtlasRagConstructor(kg_config)

    valid_backends = ["atlas"]
    raise ValueError(f"Unknown KG backend: {backend!r}. Must be one of {valid_backends}")
