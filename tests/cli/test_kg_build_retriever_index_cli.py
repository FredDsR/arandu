"""Tests for ``arandu kg-build-retriever-index`` CLI command.

Locks the failure-mode surface (missing KG outputs, missing graphml,
missing API key, skip-on-existing-manifest) without paying the
embedding compute cost. The actual atlas-rag build is exercised by
``scripts/test_atlas_rag_retriever.py`` against the real test-kg-04 KG.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "results"
    monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(base))
    return base


def _seed_atlas_kg(
    base: Path, pipeline_id: str = "run_x", keyword: str = "transcriptions.json"
) -> Path:
    """Lay out a minimal ``results/<id>/kg/outputs/atlas_output/`` tree.

    The graphml is a stub — atlas-rag's ``build_index`` is mocked away in
    these tests, so the bytes never matter. Returns the path so tests can
    optionally write more under it.
    """
    graphml_dir = base / pipeline_id / "kg" / "outputs" / "atlas_output" / "kg_graphml"
    graphml_dir.mkdir(parents=True)
    (graphml_dir / f"{keyword}_graph.graphml").write_bytes(b'<?xml version="1.0"?><graphml/>')
    return base / pipeline_id / "kg" / "outputs" / "atlas_output"


class TestKgBuildRetrieverIndexCli:
    def test_missing_kg_outputs_fails_with_clear_error(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        # No `results/<id>/kg/outputs/` for the pipeline_id at all.
        result = runner.invoke(app, ["kg-build-retriever-index", "--id", "nonexistent"])

        assert result.exit_code == 1
        assert "kg outputs not found" in result.output

    def test_missing_graphml_fails_with_clear_error(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        # kg/outputs/ exists but the atlas-rag graphml does not — KG was
        # built with a different backend or interrupted before emission.
        (results_base / "run_x" / "kg" / "outputs").mkdir(parents=True)
        result = runner.invoke(app, ["kg-build-retriever-index", "--id", "run_x"])

        assert result.exit_code == 1
        assert "GraphML not found" in result.output

    def test_missing_gemini_api_key_fails_fast(
        self,
        runner: CliRunner,
        results_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Gemini is the default provider; without GEMINI_API_KEY the build
        # must refuse before invoking the (expensive) atlas-rag path.
        _seed_atlas_kg(results_base)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        result = runner.invoke(app, ["kg-build-retriever-index", "--id", "run_x"])

        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output

    def test_existing_manifest_skipped_without_rebuild(
        self,
        runner: CliRunner,
        results_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If precompute/manifest.json already exists AND matches the
        # requested params, the helper returns early without calling
        # atlas-rag's expensive build.
        atlas_dir = _seed_atlas_kg(results_base)
        precompute = atlas_dir / "precompute"
        precompute.mkdir()
        compatible_manifest = {
            "keyword": "transcriptions.json",
            "include_events": True,
            "include_concept": True,
            "sentence_encoder_model": "gemini-embedding-001",
        }
        (precompute / "manifest.json").write_text(json.dumps(compatible_manifest))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("arandu.kg.retriever_index.AtlasRagRetriever.build_index") as mock_build:
            result = runner.invoke(app, ["kg-build-retriever-index", "--id", "run_x"])

        assert result.exit_code == 0, result.output
        mock_build.assert_not_called()
        # Existing manifest is preserved untouched.
        assert json.loads((precompute / "manifest.json").read_text()) == compatible_manifest

    def test_existing_manifest_mismatch_raises_with_rebuild_hint(
        self,
        runner: CliRunner,
        results_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Manifest was built with include_events=True but the user invokes
        # with --no-include-events. Silent reuse would defer the error to
        # retrieve-time; surface it here with a --rebuild hint.
        atlas_dir = _seed_atlas_kg(results_base)
        precompute = atlas_dir / "precompute"
        precompute.mkdir()
        stale_manifest = {
            "keyword": "transcriptions.json",
            "include_events": True,
            "include_concept": True,
            "sentence_encoder_model": "gemini-embedding-001",
        }
        (precompute / "manifest.json").write_text(json.dumps(stale_manifest))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("arandu.kg.retriever_index.AtlasRagRetriever.build_index") as mock_build:
            result = runner.invoke(
                app,
                ["kg-build-retriever-index", "--id", "run_x", "--no-include-events"],
            )

        assert result.exit_code == 1
        assert "include_events" in result.output
        assert "--rebuild" in result.output
        mock_build.assert_not_called()

    def test_rebuild_flag_forces_call_even_with_existing_manifest(
        self,
        runner: CliRunner,
        results_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        atlas_dir = _seed_atlas_kg(results_base)
        precompute = atlas_dir / "precompute"
        precompute.mkdir()
        (precompute / "manifest.json").write_text(json.dumps({"sentinel": True}))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with (
            patch("arandu.kg.retriever_index.build_embedder") as mock_build_embedder,
            patch("arandu.kg.retriever_index.AtlasRagRetriever.build_index") as mock_build,
        ):
            mock_build_embedder.return_value = object()
            result = runner.invoke(app, ["kg-build-retriever-index", "--id", "run_x", "--rebuild"])

        assert result.exit_code == 0, result.output
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs["kg_outputs_dir"] == atlas_dir
        assert kwargs["keyword"] == "transcriptions.json"
        assert kwargs["include_events"] is True
        assert kwargs["include_concept"] is True

    def test_first_build_calls_atlas_rag(
        self,
        runner: CliRunner,
        results_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No precompute/manifest yet → must invoke the build path.
        atlas_dir = _seed_atlas_kg(results_base)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with (
            patch("arandu.kg.retriever_index.build_embedder") as mock_build_embedder,
            patch("arandu.kg.retriever_index.AtlasRagRetriever.build_index") as mock_build,
        ):
            mock_build_embedder.return_value = object()
            result = runner.invoke(app, ["kg-build-retriever-index", "--id", "run_x"])

        assert result.exit_code == 0, result.output
        mock_build.assert_called_once()
        assert atlas_dir == mock_build.call_args.kwargs["kg_outputs_dir"]
