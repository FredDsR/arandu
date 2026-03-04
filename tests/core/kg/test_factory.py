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
        from unittest.mock import MagicMock

        mock_cls = MagicMock()
        mock_instance = MagicMock(spec=["build_graph"])
        mock_cls.return_value = mock_instance

        # Ensure find_spec reports atlas_rag as available
        mocker.patch("importlib.util.find_spec", return_value=MagicMock())
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
        """Test factory raises ImportError when atlas_rag is not installed."""
        mocker.patch("importlib.util.find_spec", return_value=None)

        config = KGConfig(backend="atlas")
        with pytest.raises(ImportError, match="atlas-rag is required"):
            create_kg_constructor(config)
