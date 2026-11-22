"""Every analysis module imports cleanly from the ``analysis`` package."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
MODULES = sorted(p.stem for p in ANALYSIS_DIR.glob("*.py") if p.stem != "__init__")


def test_analysis_package_has_modules() -> None:
    assert "_judge_common" in MODULES
    assert "analyze_judge_retest" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    try:
        importlib.import_module(f"analysis.{name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"analysis", "scripts", "arandu"}:
            raise
        pytest.skip(f"optional dependency missing: {exc.name}")
