"""Tests for the absence-check structures (spec §7.5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from arandu.qa.non_answerable import corpus_index as ci
from arandu.qa.non_answerable.corpus_index import SourceCorpusIndex, load_kg_node_set

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TranscriptionRecordWriter


class TestLoadKgNodeSet:
    """Tests for loading the normalized KG node label set from GraphML."""

    def test_reads_labels_lowercased(self, tmp_path: Path) -> None:
        graph = nx.DiGraph()
        graph.add_node("n0", label="Maria")
        graph.add_node("n1", label="Itaqui")
        path = tmp_path / "corpus_graph.graphml"
        nx.write_graphml(graph, str(path))
        nodes = load_kg_node_set(path)
        assert nodes == {"maria", "itaqui"}

    def test_falls_back_to_node_id_without_label(self, tmp_path: Path) -> None:
        graph = nx.DiGraph()
        graph.add_node("Cheia2024")
        path = tmp_path / "g.graphml"
        nx.write_graphml(graph, str(path))
        assert "cheia2024" in load_kg_node_set(path)

    def test_absent_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_kg_node_set(tmp_path / "missing.graphml") == set()


class TestSourceCorpusIndex:
    """Tests for the source-corpus membership gate (NER + token fallback)."""

    def test_token_fallback_indexes_alpha_tokens(
        self,
        tmp_path: Path,
        monkeypatch: object,
        write_transcription_record: TranscriptionRecordWriter,
    ) -> None:
        # Force the no-spaCy fallback so the test is deterministic + fast.
        monkeypatch.setattr(ci, "_portuguese_nlp", lambda: None)  # type: ignore[attr-defined]
        tdir = tmp_path / "transcription" / "outputs"
        write_transcription_record(tdir, "src1", "Maria viu a enchente em Itaqui", suffix="")
        index = SourceCorpusIndex(tdir)
        assert "maria" in index
        assert "itaqui" in index
        assert "enchente" in index
        assert "joana" not in index

    def test_backstop_catches_terms_the_span_set_misses(
        self,
        tmp_path: Path,
        monkeypatch: object,
        write_transcription_record: TranscriptionRecordWriter,
    ) -> None:
        # The fallback span set only holds alpha tokens >= 4 chars; the full-text
        # word-boundary backstop catches present terms it misses: multi-word
        # phrases, bare years (non-alpha), and short content words. Absent terms
        # still read as absent.
        monkeypatch.setattr(ci, "_portuguese_nlp", lambda: None)  # type: ignore[attr-defined]
        tdir = tmp_path / "transcription" / "outputs"
        write_transcription_record(
            tdir, "src1", "Em 2012 a avó morava na Ponta da Areia", suffix=""
        )
        index = SourceCorpusIndex(tdir)
        assert "Ponta da Areia" in index  # multi-word, not a single span
        assert "2012" in index  # bare year, not an alpha token
        assert "avó" in index  # 3-char content word, below the alpha floor
        assert "Ponta do Sol" not in index  # absent multi-word stays absent
        assert "2019" not in index  # absent year stays absent

    def test_absent_dir_is_empty(self, tmp_path: Path, monkeypatch: object) -> None:
        monkeypatch.setattr(ci, "_portuguese_nlp", lambda: None)  # type: ignore[attr-defined]
        index = SourceCorpusIndex(tmp_path / "does-not-exist")
        assert len(index) == 0
        assert "anything" not in index
