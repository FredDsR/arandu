"""Tests for KG constructor factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gtranscriber.config import KGConfig
from gtranscriber.core.kg.factory import create_kg_constructor

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestCreateKGConstructor:
    """Tests for create_kg_constructor factory function."""

    def test_atlas_backend_returns_constructor(self, mocker: MockerFixture) -> None:
        """Test factory returns an AtlasRagConstructor for atlas backend."""
        # Mock AtlasRagConstructor so we don't need atlas-rag installed
        from unittest.mock import MagicMock

        mock_cls = MagicMock()
        mock_instance = MagicMock(spec=["build_graph"])
        mock_cls.return_value = mock_instance

        mocker.patch(
            "gtranscriber.core.kg.atlas_backend.AtlasRagConstructor",
            mock_cls,
        )

        config = KGConfig(backend="atlas")
        result = create_kg_constructor(config)
        assert result is mock_instance
        mock_cls.assert_called_once_with(config)

    def test_unknown_backend_raises_value_error(self) -> None:
        """Test factory raises ValueError for unknown backend."""
        config = KGConfig.__new__(KGConfig)
        object.__setattr__(config, "backend", "nonexistent")

        with pytest.raises(ValueError, match="Unknown KG backend"):
            create_kg_constructor(config)

    def test_missing_dependency_raises_import_error(self, mocker: MockerFixture) -> None:
        """Test factory raises ImportError when atlas_backend import fails."""
        mocker.patch(
            "gtranscriber.core.kg.factory.AtlasRagConstructor",
            side_effect=ImportError("No module named 'atlas_rag'"),
            create=True,
        )
        # Simulate the import failing by patching the import itself
        original = __import__

        def fail_import(name: str, *args: object, **kwargs: object) -> object:
            if "atlas_backend" in name:
                raise ImportError("No module named 'atlas_rag'")
            return original(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=fail_import)

        config = KGConfig(backend="atlas")
        with pytest.raises(ImportError, match="atlas-rag is required"):
            create_kg_constructor(config)
