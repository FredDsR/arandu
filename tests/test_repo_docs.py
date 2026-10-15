"""Repository-wide checks on tracked Markdown files."""

from __future__ import annotations

import os
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


def tracked_paths() -> set[str]:
    """Return every tracked file path plus all of its parent directories."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    paths: set[str] = set()
    for line in out.splitlines():
        paths.add(line)
        paths.update(str(parent) for parent in Path(line).parents if str(parent) != ".")
    return paths


def broken_links(root: Path, tracked: set[str]) -> list[str]:
    """List relative Markdown links, per file in ``root``, whose target is not tracked."""
    broken = []
    for md in tracked_markdown_under(root, tracked):
        text = FENCE.sub("", (root / md).read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and os.path.normpath(Path(md).parent / path) not in tracked:
                broken.append(f"{md} -> {target}")
    return broken


def tracked_markdown_under(root: Path, tracked: set[str]) -> list[str]:
    """Return the tracked Markdown paths (relative to ``root``) that exist on disk."""
    return sorted(p for p in tracked if p.endswith(".md") and (root / p).is_file())


def test_relative_markdown_links_resolve() -> None:
    assert broken_links(ROOT, tracked_paths()) == []


def test_broken_links_flags_untracked_target(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text(
        "[ok](other.md) [dir](../docs) [local](scratch.md) [gone](nope.md#x)",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "scratch.md").write_text("on disk, untracked", encoding="utf-8")
    tracked = {"docs", "docs/page.md", "docs/other.md"}
    assert broken_links(tmp_path, tracked) == [
        "docs/page.md -> scratch.md",
        "docs/page.md -> nope.md#x",
    ]


TREE = re.compile(r"^[ \t]*(?:│[ \t]*)*[├└]── ", re.MULTILINE)
GUIDE_NAMES = {"AGENTS.md", "README.md"}


def test_guides_do_not_draw_file_trees() -> None:
    offenders = [
        str(md.relative_to(ROOT))
        for md in tracked_markdown()
        if md.name in GUIDE_NAMES and TREE.search(md.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_tree_pattern_matches_tree_lines_only() -> None:
    assert TREE.search("src/\n├── arandu/\n│   └── utils/\n")
    assert TREE.search("    │   └── file.py")
    assert not TREE.search("┌──────┐\n│ box  │\n└──────┘\n")
