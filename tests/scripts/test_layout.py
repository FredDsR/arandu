"""Layout rules for ``scripts/`` that the agent guide relies on."""

from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_no_pytest_prefix_in_scripts() -> None:
    offenders = sorted(str(p.relative_to(SCRIPTS)) for p in SCRIPTS.rglob("test_*.py"))
    assert offenders == []


def test_subpackages_are_importable_packages() -> None:
    for sub in ("smoke", "oneoff"):
        assert (SCRIPTS / sub / "__init__.py").is_file()
