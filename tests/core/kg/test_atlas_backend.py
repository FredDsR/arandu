"""Tests for AtlasRagConstructor backend."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime for tmp_path fixtures
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from gtranscriber.config import KGConfig
from gtranscriber.schemas import EnrichedRecord

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_record(gdrive_id: str = "test123", text: str = "Test transcription.") -> EnrichedRecord:
    """Create a minimal EnrichedRecord for testing."""
    return EnrichedRecord.model_validate(
        {
            "gdrive_id": gdrive_id,
            "name": "test.mp3",
            "mimeType": "audio/mpeg",
            "parents": ["folder"],
            "webContentLink": "https://drive.google.com/test",
            "size_bytes": 1024,
            "duration_milliseconds": 60000,
            "transcription_text": text,
            "detected_language": "pt",
            "language_probability": 0.95,
            "model_id": "whisper-large-v3",
            "compute_device": "cpu",
            "processing_duration_sec": 10.0,
            "transcription_status": "completed",
        }
    )


@pytest.fixture
def _mock_atlas_rag() -> dict[str, MagicMock]:
    """Mock all atlas_rag modules so AtlasRagConstructor can be imported."""
    mocks: dict[str, MagicMock] = {}

    # Create mock modules
    atlas_rag = MagicMock()
    atlas_rag.__version__ = "0.0.5"
    mocks["atlas_rag"] = atlas_rag

    triple_config = MagicMock()
    mocks["atlas_rag.kg_construction"] = MagicMock()
    mocks["atlas_rag.kg_construction.triple_config"] = triple_config
    mocks["atlas_rag.kg_construction.triple_extraction"] = MagicMock()
    mocks["atlas_rag.llm_generator"] = MagicMock()

    # Mock the prompt module with CONCEPT_INSTRUCTIONS dict
    prompt_module = MagicMock()
    prompt_module.CONCEPT_INSTRUCTIONS = {
        "en": {"event": "...", "entity": "...", "relation": "..."},
    }
    mocks["atlas_rag.llm_generator.prompt"] = MagicMock()
    mocks["atlas_rag.llm_generator.prompt.triple_extraction_prompt"] = prompt_module

    with patch.dict("sys.modules", mocks):
        yield mocks


class TestAtlasRagConstructorInit:
    """Tests for AtlasRagConstructor initialization."""

    def test_default_options(self, _mock_atlas_rag: dict) -> None:
        """Test constructor applies default atlas options."""
        from gtranscriber.core.kg.atlas_backend import ATLAS_DEFAULTS, AtlasRagConstructor

        config = KGConfig(backend="atlas")
        constructor = AtlasRagConstructor(config)

        for key, default_val in ATLAS_DEFAULTS.items():
            assert constructor._opts[key] == default_val

    def test_backend_options_override(self, _mock_atlas_rag: dict) -> None:
        """Test backend_options override atlas defaults."""
        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(
            backend="atlas",
            backend_options={"chunk_size": 4096, "batch_size_triple": 5},
        )
        constructor = AtlasRagConstructor(config)

        assert constructor._opts["chunk_size"] == 4096
        assert constructor._opts["batch_size_triple"] == 5
        # Non-overridden defaults stay
        assert constructor._opts["max_new_tokens"] == 2048


class TestPrepareInputData:
    """Tests for _prepare_input_data."""

    def test_creates_json_file(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test input data is written as expected JSON."""
        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend="atlas", language="pt")
        constructor = AtlasRagConstructor(config)
        records = [_make_record("id1", "Texto um."), _make_record("id2", "Texto dois.")]

        input_dir = tmp_path / "atlas_input"
        constructor._prepare_input_data(records, input_dir)

        output_file = input_dir / "transcriptions.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "id1"
        assert data[0]["text"] == "Texto um."
        assert data[0]["metadata"]["lang"] == "pt"
        assert data[1]["id"] == "id2"


class TestCreateOpenAIClient:
    """Tests for _create_openai_client."""

    def test_ollama_provider(self, _mock_atlas_rag: dict, mocker: MockerFixture) -> None:
        """Test OpenAI client construction for Ollama provider."""
        mock_openai_cls = mocker.patch("openai.OpenAI")

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="ollama", ollama_url="http://localhost:11434/v1")
        constructor = AtlasRagConstructor(config)
        constructor._create_openai_client()

        mock_openai_cls.assert_called_once_with(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

    def test_custom_provider_requires_base_url(self, _mock_atlas_rag: dict) -> None:
        """Test custom provider raises without base_url."""
        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="custom", base_url=None)
        constructor = AtlasRagConstructor(config)

        with pytest.raises(ValueError, match="base_url is required"):
            constructor._create_openai_client()

    def test_custom_provider_with_base_url(
        self, _mock_atlas_rag: dict, mocker: MockerFixture
    ) -> None:
        """Test custom provider uses provided base_url."""
        mock_openai_cls = mocker.patch("openai.OpenAI")

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="custom", base_url="http://my-api:8000/v1")
        constructor = AtlasRagConstructor(config)
        constructor._create_openai_client()

        mock_openai_cls.assert_called_once_with(
            api_key=None,
            base_url="http://my-api:8000/v1",
        )


class TestInjectConceptPrompts:
    """Tests for _inject_concept_prompts."""

    def test_injects_pt_prompts(self, _mock_atlas_rag: dict) -> None:
        """Test Portuguese concept prompts are injected into CONCEPT_INSTRUCTIONS."""
        from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (
            CONCEPT_INSTRUCTIONS,
        )

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        constructor._inject_concept_prompts()

        assert "pt" in CONCEPT_INSTRUCTIONS

    def test_skips_if_language_already_present(self, _mock_atlas_rag: dict) -> None:
        """Test injection is skipped when language already in dict."""
        from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (
            CONCEPT_INSTRUCTIONS,
        )

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(language="en")
        constructor = AtlasRagConstructor(config)
        original_en = CONCEPT_INSTRUCTIONS["en"]

        constructor._inject_concept_prompts()

        # English entry should not be overwritten
        assert CONCEPT_INSTRUCTIONS["en"] is original_en


class TestBuildProcessingConfig:
    """Tests for _build_processing_config."""

    def test_config_fields(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test ProcessingConfig receives correct field values."""
        from atlas_rag.kg_construction.triple_config import ProcessingConfig

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(
            model_id="qwen3:14b",
            backend_options={"chunk_size": 4096},
        )
        constructor = AtlasRagConstructor(config)

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        constructor._build_processing_config(input_dir, output_dir)

        ProcessingConfig.assert_called_once()
        call_kwargs = ProcessingConfig.call_args[1]
        assert call_kwargs["model_path"] == "qwen3:14b"
        assert call_kwargs["data_directory"] == str(input_dir)
        assert call_kwargs["chunk_size"] == 4096
        assert call_kwargs["batch_size_triple"] == 3  # default


class TestRunPipeline:
    """Tests for _run_pipeline."""

    def test_all_five_steps_called(self, _mock_atlas_rag: dict) -> None:
        """Test all 5 atlas-rag pipeline steps are called."""
        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend_options={"include_concept": True})
        constructor = AtlasRagConstructor(config)

        mock_extractor = MagicMock()
        constructor._run_pipeline(mock_extractor)

        mock_extractor.run_extraction.assert_called_once()
        mock_extractor.convert_json_to_csv.assert_called_once()
        mock_extractor.generate_concept_csv_temp.assert_called_once()
        mock_extractor.create_concept_csv.assert_called_once()
        mock_extractor.convert_to_graphml.assert_called_once()

    def test_skip_concept_steps(self, _mock_atlas_rag: dict) -> None:
        """Test concept steps are skipped when include_concept=False."""
        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend_options={"include_concept": False})
        constructor = AtlasRagConstructor(config)

        mock_extractor = MagicMock()
        constructor._run_pipeline(mock_extractor)

        mock_extractor.run_extraction.assert_called_once()
        mock_extractor.convert_json_to_csv.assert_called_once()
        mock_extractor.generate_concept_csv_temp.assert_not_called()
        mock_extractor.create_concept_csv.assert_not_called()
        mock_extractor.convert_to_graphml.assert_called_once()


class TestBuildResult:
    """Tests for _build_result."""

    def test_loads_graphml_and_builds_result(
        self, tmp_path: Path, _mock_atlas_rag: dict, mocker: MockerFixture
    ) -> None:
        """Test result is built correctly from GraphML output."""
        # Create a mock GraphML file
        atlas_output = tmp_path / "atlas_output"
        atlas_output.mkdir(parents=True)
        graphml_file = atlas_output / "graph.graphml"
        graphml_file.write_text("<graphml></graphml>")

        # Mock networkx (imported inside method as `import networkx as nx`)
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 42
        mock_graph.number_of_edges.return_value = 17
        mocker.patch("networkx.read_graphml", return_value=mock_graph)

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(model_id="test-model", provider="ollama", language="pt")
        constructor = AtlasRagConstructor(config)

        records = [_make_record("id1"), _make_record("id2")]
        result = constructor._build_result(records, tmp_path)

        assert result.node_count == 42
        assert result.edge_count == 17
        assert result.source_record_ids == ["id1", "id2"]
        assert result.metadata.model_id == "test-model"
        assert result.metadata.provider == "ollama"
        assert result.metadata.total_documents == 2
        assert result.metadata.total_nodes == 42
        assert result.metadata.total_edges == 17
        assert result.metadata.backend_version == "atlas-rag==0.0.5"

    def test_raises_when_no_graphml_found(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test FileNotFoundError when no GraphML output exists."""
        atlas_output = tmp_path / "atlas_output"
        atlas_output.mkdir(parents=True)

        from gtranscriber.core.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig()
        constructor = AtlasRagConstructor(config)

        with pytest.raises(FileNotFoundError, match="No GraphML file found"):
            constructor._build_result([], tmp_path)
