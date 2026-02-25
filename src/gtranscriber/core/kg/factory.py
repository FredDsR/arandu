"""Factory for creating KG constructor instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gtranscriber.config import KGConfig
    from gtranscriber.core.kg.protocol import KGConstructor


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
        try:
            from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor
        except ImportError as e:
            raise ImportError(
                "atlas-rag is required for the 'atlas' backend. "
                "Install it with: uv pip install atlas-rag"
            ) from e
        return AtlasRagConstructor(kg_config)

    valid_backends = ["atlas"]
    raise ValueError(f"Unknown KG backend: {backend!r}. Must be one of {valid_backends}")
