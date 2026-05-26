"""Tests for the CEP cross-cut metadata loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.rag.analysis.loader import build_cross_cut_map

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildCrossCutMap:
    def test_indexes_by_composite_qa_pair_id(self, cep_outputs_dir: Path) -> None:
        mapping = build_cross_cut_map(cep_outputs_dir)
        assert set(mapping.keys()) == {"src:chk_00:0", "src:chk_00:1"}

    def test_carries_bloom_and_question_type(self, cep_outputs_dir: Path) -> None:
        mapping = build_cross_cut_map(cep_outputs_dir)
        first = mapping["src:chk_00:0"]
        assert first.bloom_level == "remember"
        assert first.question_type == "factual"
        second = mapping["src:chk_00:1"]
        assert second.bloom_level == "analyze"
        assert second.question_type == "conceptual"

    def test_absent_dir_returns_empty(self, tmp_path: Path) -> None:
        assert build_cross_cut_map(tmp_path / "does-not-exist") == {}

    def test_unreadable_files_skipped(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "cep" / "outputs"
        bad_dir.mkdir(parents=True)
        (bad_dir / "garbage.json").write_text("not json", encoding="utf-8")
        assert build_cross_cut_map(bad_dir) == {}
