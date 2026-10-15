"""Repository-wide checks on tracked Markdown files."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def tracked_markdown() -> list[Path]:
    """Return every Markdown file tracked by Git."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    out = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True)
    return [ROOT / line for line in out.splitlines()]


def test_relative_markdown_links_resolve() -> None:
    broken = []
    for md in tracked_markdown():
        text = FENCE.sub("", md.read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert broken == []
