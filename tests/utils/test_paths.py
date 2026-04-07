"""Tests for arandu.utils.paths module."""

from __future__ import annotations

from pathlib import Path

from arandu.utils.paths import get_project_root


class TestGetProjectRoot:
    """Tests for get_project_root utility."""

    def test_returns_path_containing_pyproject_toml(self) -> None:
        """Verify that the returned path contains pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_path_object(self) -> None:
        """Verify that the return type is a Path."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_returns_absolute_path(self) -> None:
        """Verify that the returned path is absolute."""
        root = get_project_root()
        assert root.is_absolute()

    def test_contains_src_directory(self) -> None:
        """Verify that the root contains the src directory."""
        root = get_project_root()
        assert (root / "src").is_dir()
