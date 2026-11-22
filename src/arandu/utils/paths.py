"""Path utilities for the Arandu project."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.

    Walks up from this file to find the directory containing pyproject.toml.

    Returns:
        Absolute path to the project root.

    Raises:
        RuntimeError: If pyproject.toml is not found in any ancestor directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find project root (no pyproject.toml in ancestor directories)"
    raise RuntimeError(msg)
