"""Tests for AtlasRagConstructor backend."""

from __future__ import annotations

import csv
import json
from pathlib import Path  # noqa: TC003 — used at runtime for tmp_path fixtures
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from arandu.kg.config import KGConfig
from arandu.shared.schemas import EnrichedRecord, SourceMetadata

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_record(file_id: str = "test123", text: str = "Test transcription.") -> EnrichedRecord:
    """Create a minimal EnrichedRecord for testing."""
    return EnrichedRecord.model_validate(
        {
            "file_id": file_id,
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


class _StubDatasetProcessor:
    """Minimal stand-in for ``atlas_rag.kg_construction.triple_extraction.DatasetProcessor``.

    Provides a real base class so that the enriched processor subclass can
    use standard inheritance (``super()``) during tests.
    """

    def __init__(self, config: Any, prompts: Any, schema: Any) -> None:
        self.config = config
        self.prompts = prompts
        self.schema = schema

    def create_sample_chunks(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the full document text as a single chunk by default."""
        return [{"text": sample["text"], "metadata": dict(sample.get("metadata", {}))}]


@pytest.fixture
def _mock_atlas_rag() -> dict[str, MagicMock]:
    """Mock all atlas_rag modules so AtlasRagConstructor can be imported."""
    import arandu.kg.atlas_backend as atlas_backend

    # Reset the cached enriched processor class so it is rebuilt with the mock
    atlas_backend._MetadataEnrichedProcessorCls = None

    mocks: dict[str, MagicMock] = {}

    # Create mock modules
    atlas_rag = MagicMock()
    atlas_rag.__version__ = "0.0.5"
    mocks["atlas_rag"] = atlas_rag

    triple_config = MagicMock()
    mocks["atlas_rag.kg_construction"] = MagicMock()
    mocks["atlas_rag.kg_construction.triple_config"] = triple_config

    # Use a real class for DatasetProcessor so subclassing works properly
    triple_extraction = MagicMock()
    triple_extraction.DatasetProcessor = _StubDatasetProcessor
    mocks["atlas_rag.kg_construction.triple_extraction"] = triple_extraction
    mocks["atlas_rag.llm_generator"] = MagicMock()

    # Mock the prompt module with CONCEPT_INSTRUCTIONS dict
    prompt_module = MagicMock()
    prompt_module.CONCEPT_INSTRUCTIONS = {
        "en": {"event": "...", "entity": "...", "relation": "..."},
    }
    mocks["atlas_rag.llm_generator.prompt"] = MagicMock()
    mocks["atlas_rag.llm_generator.prompt.triple_extraction_prompt"] = prompt_module

    # Mock csv_processing for the orphan-node patch
    csv_to_graphml_module = MagicMock()
    mocks["atlas_rag.kg_construction.utils"] = MagicMock()
    mocks["atlas_rag.kg_construction.utils.csv_processing"] = MagicMock()
    mocks["atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml"] = csv_to_graphml_module

    with patch.dict("sys.modules", mocks):
        yield mocks

    # Reset cache again after test so mocked class is not retained
    atlas_backend._MetadataEnrichedProcessorCls = None


class TestAtlasRagConstructorInit:
    """Tests for AtlasRagConstructor initialization."""

    def test_default_options(self, _mock_atlas_rag: dict) -> None:
        """Test constructor applies default atlas options."""
        from arandu.kg.atlas_backend import ATLAS_DEFAULTS, AtlasRagConstructor

        config = KGConfig(backend="atlas")
        constructor = AtlasRagConstructor(config)

        for key, default_val in ATLAS_DEFAULTS.items():
            assert constructor._opts[key] == default_val

    def test_backend_options_override(self, _mock_atlas_rag: dict) -> None:
        """Test backend_options override atlas defaults."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

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
        from arandu.kg.atlas_backend import AtlasRagConstructor

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


class TestBuildMetadataHeader:
    """Tests for _build_metadata_header."""

    @pytest.fixture
    def _pt_labels(self) -> dict[str, str]:
        """Portuguese labels matching metadata_labels.json."""
        return {
            "header": "[Contexto da Entrevista]",
            "transcription": "[Transcrição]",
            "participant": "Participante",
            "location": "Local",
            "date": "Data",
            "context": "Contexto",
            "researcher": "Pesquisador(a)",
            "sequence": "Sequência",
        }

    @pytest.fixture
    def _en_labels(self) -> dict[str, str]:
        """English labels matching metadata_labels.json."""
        return {
            "header": "[Interview Context]",
            "transcription": "[Transcription]",
            "participant": "Participant",
            "location": "Location",
            "date": "Date",
            "context": "Event Context",
            "researcher": "Researcher",
            "sequence": "Sequence",
        }

    def test_no_metadata_returns_empty(self, _mock_atlas_rag: dict) -> None:
        """When source_metadata is None, return empty string."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Texto original.")
        assert record.source_metadata is None

        result = AtlasRagConstructor._build_metadata_header(
            record,
            {"header": "H", "transcription": "T", "participant": "P"},
        )
        assert result == ""

    def test_pt_header(
        self,
        _mock_atlas_rag: dict,
        _pt_labels: dict[str, str],
    ) -> None:
        """Portuguese metadata header uses Portuguese labels."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Texto da entrevista.")
        record.source_metadata = SourceMetadata(
            participant_name="João da Silva",
            location="Barra de Pelotas",
            recording_date="2024-03-15",
            event_context="Audiência Câmara de Vereadores",
            researcher_name="Maria Santos",
            sequence_label="Parte I",
        )

        result = AtlasRagConstructor._build_metadata_header(record, _pt_labels)

        assert result.startswith("[Contexto da Entrevista]")
        assert "Participante: João da Silva" in result
        assert "Local: Barra de Pelotas" in result
        assert "Data: 2024-03-15" in result
        assert "Contexto: Audiência Câmara de Vereadores" in result
        assert "Pesquisador(a): Maria Santos" in result
        assert "Sequência: Parte I" in result
        assert result.endswith("[Transcrição]")

    def test_en_header(
        self,
        _mock_atlas_rag: dict,
        _en_labels: dict[str, str],
    ) -> None:
        """English metadata header uses English labels."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Interview text.")
        record.source_metadata = SourceMetadata(
            participant_name="John Smith",
            location="Porto Alegre",
        )

        result = AtlasRagConstructor._build_metadata_header(record, _en_labels)

        assert result.startswith("[Interview Context]")
        assert "Participant: John Smith" in result
        assert "Location: Porto Alegre" in result
        assert result.endswith("[Transcription]")

    def test_partial_metadata_omits_none_fields(
        self,
        _mock_atlas_rag: dict,
        _pt_labels: dict[str, str],
    ) -> None:
        """Only non-None fields appear in the header."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Texto.")
        record.source_metadata = SourceMetadata(participant_name="Ana")

        result = AtlasRagConstructor._build_metadata_header(record, _pt_labels)

        assert "Participante: Ana" in result
        assert "Local:" not in result
        assert "Data:" not in result

    def test_empty_metadata_returns_empty(
        self,
        _mock_atlas_rag: dict,
        _pt_labels: dict[str, str],
    ) -> None:
        """When all SourceMetadata fields are None, return empty string."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Texto.")
        record.source_metadata = SourceMetadata()

        result = AtlasRagConstructor._build_metadata_header(record, _pt_labels)
        assert result == ""

    def test_empty_labels_returns_empty(self, _mock_atlas_rag: dict) -> None:
        """When labels dict is empty (missing file), return empty string."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        record = _make_record(text="Texto.")
        record.source_metadata = SourceMetadata(participant_name="Ana")

        result = AtlasRagConstructor._build_metadata_header(record, {})
        assert result == ""


class TestMetadataEnrichedProcessor:
    """Tests for the DatasetProcessor subclass that prepends headers to chunks."""

    def test_prepends_header_to_all_chunks(
        self,
        _mock_atlas_rag: dict,
        mocker: MockerFixture,
    ) -> None:
        """Each chunk should have the metadata header prepended."""
        # Patch the stub so the base class returns multiple chunks
        mocker.patch.object(
            _StubDatasetProcessor,
            "create_sample_chunks",
            return_value=[
                {"text": "Chunk one text.", "metadata": {"lang": "pt"}},
                {"text": "Chunk two text.", "metadata": {"lang": "pt"}},
                {"text": "Chunk three text.", "metadata": {"lang": "pt"}},
            ],
        )

        from arandu.kg.atlas_backend import _get_enriched_processor_cls

        ProcessorCls = _get_enriched_processor_cls()

        # Simulate a document with _metadata_header in metadata
        sample = {
            "id": "doc1",
            "text": "Word " * 5000,
            "metadata": {
                "lang": "pt",
                "_metadata_header": "[Contexto]\nParticipante: Ana\n\n[Transcrição]",
            },
        }

        from atlas_rag.kg_construction.triple_config import ProcessingConfig

        config = ProcessingConfig(
            model_path="test",
            data_directory=".",
            filename_pattern="test",
            chunk_size=100,
        )
        processor = ProcessorCls(config, {}, {})
        chunks = processor.create_sample_chunks(sample)

        assert len(chunks) > 1, "Should produce multiple chunks"
        for chunk in chunks:
            assert chunk["text"].startswith("[Contexto]\nParticipante: Ana")

    def test_no_header_key_passes_through(self, _mock_atlas_rag: dict) -> None:
        """Documents without _metadata_header should pass through unchanged."""
        from arandu.kg.atlas_backend import _get_enriched_processor_cls

        ProcessorCls = _get_enriched_processor_cls()

        sample = {
            "id": "doc1",
            "text": "Some short text.",
            "metadata": {"lang": "pt"},
        }

        from atlas_rag.kg_construction.triple_config import ProcessingConfig

        config = ProcessingConfig(
            model_path="test",
            data_directory=".",
            filename_pattern="test",
        )
        processor = ProcessorCls(config, {}, {})
        chunks = processor.create_sample_chunks(sample)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Some short text."


class TestLoadMetadataLabels:
    """Tests for _load_metadata_labels."""

    def test_loads_configured_language(self, _mock_atlas_rag: dict) -> None:
        """Labels for the configured language are returned."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        labels = constructor._load_metadata_labels()

        assert labels["header"] == "[Contexto da Entrevista]"
        assert labels["participant"] == "Participante"

    def test_falls_back_to_english(self, _mock_atlas_rag: dict, tmp_path: Path) -> None:
        """Unknown language falls back to English labels."""
        from arandu.kg import atlas_backend

        labels_file = tmp_path / "metadata_labels.json"
        labels_file.write_text(
            json.dumps(
                {
                    "en": {"header": "[Context]", "participant": "Participant"},
                }
            )
        )
        original = atlas_backend._PROMPTS_DIR
        atlas_backend._PROMPTS_DIR = tmp_path

        try:
            config = KGConfig(language="pt")
            constructor = atlas_backend.AtlasRagConstructor(config)
            labels = constructor._load_metadata_labels()
            assert labels["header"] == "[Context]"
        finally:
            atlas_backend._PROMPTS_DIR = original

    def test_missing_file_returns_empty(
        self,
        _mock_atlas_rag: dict,
        tmp_path: Path,
    ) -> None:
        """Missing labels file returns empty dict (graceful degradation)."""
        from arandu.kg import atlas_backend

        original = atlas_backend._PROMPTS_DIR
        atlas_backend._PROMPTS_DIR = tmp_path / "nonexistent"

        try:
            config = KGConfig(language="pt")
            constructor = atlas_backend.AtlasRagConstructor(config)
            labels = constructor._load_metadata_labels()
            assert labels == {}
        finally:
            atlas_backend._PROMPTS_DIR = original


class TestParseExtractionRecords:
    """Tests for _parse_extraction_records."""

    def test_parses_jsonl_file(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Standard JSONL file (one JSON object per line) is parsed correctly."""
        from arandu.kg.atlas_backend import _parse_extraction_records

        f = tmp_path / "output.json"
        records = [{"id": f"chunk_{i}", "original_text": f"text {i}"} for i in range(5)]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        parsed, invalid = _parse_extraction_records(f)

        assert len(parsed) == 5
        assert invalid == 0
        assert parsed[0]["id"] == "chunk_0"
        assert parsed[4]["id"] == "chunk_4"

    def test_parses_pretty_printed_file(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Pretty-printed JSON objects (multi-line, concatenated) are parsed."""
        from arandu.kg.atlas_backend import _parse_extraction_records

        f = tmp_path / "output.json"
        records = [{"id": f"chunk_{i}", "original_text": f"text {i}"} for i in range(3)]
        # Write as concatenated pretty-printed objects (what the bug produces)
        f.write_text("\n".join(json.dumps(r, indent=4) for r in records) + "\n")

        parsed, invalid = _parse_extraction_records(f)

        assert len(parsed) == 3
        assert invalid == 0
        assert parsed[0]["id"] == "chunk_0"
        assert parsed[2]["id"] == "chunk_2"

    def test_skips_invalid_records(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Invalid JSON lines are skipped and counted."""
        from arandu.kg.atlas_backend import _parse_extraction_records

        f = tmp_path / "output.json"
        f.write_text(
            json.dumps({"id": "good_0"})
            + "\nNOT VALID JSON\n"
            + json.dumps({"id": "good_1"})
            + "\n"
        )

        parsed, invalid = _parse_extraction_records(f)

        assert len(parsed) == 2
        assert invalid == 1

    def test_empty_file_returns_empty(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Empty file returns empty list."""
        from arandu.kg.atlas_backend import _parse_extraction_records

        f = tmp_path / "output.json"
        f.write_text("")

        parsed, invalid = _parse_extraction_records(f)

        assert len(parsed) == 0
        assert invalid == 0

    def test_pretty_printed_with_nested_structures(
        self, tmp_path: Path, _mock_atlas_rag: dict
    ) -> None:
        """Pretty-printed JSON with nested arrays/dicts is parsed correctly."""
        from arandu.kg.atlas_backend import _parse_extraction_records

        f = tmp_path / "output.json"
        record = {
            "id": "chunk_0",
            "original_text": "Some text.",
            "metadata": {"lang": "pt"},
            "entity_relation_dict": [
                {"Head": "A", "Relation": "rel", "Tail": "B"},
                {"Head": "C", "Relation": "rel2", "Tail": "D"},
            ],
            "event_entity_dict": [],
        }
        f.write_text(json.dumps(record, indent=4) + "\n")

        parsed, _invalid = _parse_extraction_records(f)

        assert len(parsed) == 1
        assert parsed[0]["id"] == "chunk_0"
        assert len(parsed[0]["entity_relation_dict"]) == 2


class TestDetectResumeOffset:
    """Tests for _detect_resume_offset."""

    def test_no_output_dir_returns_zero(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """When atlas_output/kg_extraction/ does not exist, return 0."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        assert constructor._detect_resume_offset(tmp_path) == 0

    def test_empty_dir_returns_zero(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """When kg_extraction/ exists but has no files, return 0."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        (tmp_path / "atlas_output" / "kg_extraction").mkdir(parents=True)
        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        assert constructor._detect_resume_offset(tmp_path) == 0

    def test_complete_batches(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Count complete batches from JSONL lines."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        # 9 lines with batch_size=3 -> 3 complete batches
        output_file = kg_dir / "model_transcriptions.json_output_20260226_1_in_1.json"
        output_file.write_text(
            "\n".join([json.dumps({"id": f"chunk_{i}"}) for i in range(9)]) + "\n"
        )

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        assert constructor._detect_resume_offset(tmp_path) == 3

    def test_trims_partial_batch(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Partial last batch is trimmed from the output file."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        # 11 records with batch_size=3 -> 3 complete batches, 2 partial records trimmed
        output_file = kg_dir / "model_transcriptions.json_output_20260226_1_in_1.json"
        lines = [json.dumps({"id": f"chunk_{i}"}) for i in range(11)]
        output_file.write_text("\n".join(lines) + "\n")

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        result = constructor._detect_resume_offset(tmp_path)
        assert result == 3

        # File should now have exactly 9 JSONL lines
        remaining = [ln for ln in output_file.read_text().strip().split("\n") if ln.strip()]
        assert len(remaining) == 9

    def test_pretty_printed_file_counts_records_not_lines(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Pretty-printed JSON file counts actual records, not raw lines."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        # 9 records written as pretty-printed JSON (many lines per record)
        output_file = kg_dir / "model_transcriptions.json_output_20260226_1_in_1.json"
        records = [{"id": f"chunk_{i}", "text": f"text {i}"} for i in range(9)]
        output_file.write_text("\n".join(json.dumps(r, indent=4) for r in records) + "\n")

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        # Should count 9 records -> 3 batches, NOT thousands of lines
        assert constructor._detect_resume_offset(tmp_path) == 3

    def test_pretty_printed_file_is_normalized_to_jsonl(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Pretty-printed files are rewritten as JSONL after resume detection."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        output_file = kg_dir / "model_transcriptions.json_output_20260226_1_in_1.json"
        records = [{"id": f"chunk_{i}", "text": f"text {i}"} for i in range(6)]
        output_file.write_text("\n".join(json.dumps(r, indent=4) for r in records) + "\n")

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)
        constructor._detect_resume_offset(tmp_path)

        # File should now be proper JSONL (one object per line)
        lines = [ln for ln in output_file.read_text().strip().split("\n") if ln.strip()]
        assert len(lines) == 6
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj

    def test_multiple_output_files(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Records from multiple output files are summed."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        # First file: 6 records, second file: 3 records -> 9 total, 3 batches
        f1 = kg_dir / "model_transcriptions.json_output_20260226160000_1_in_1.json"
        f2 = kg_dir / "model_transcriptions.json_output_20260226180000_1_in_1.json"
        f1.write_text("\n".join([json.dumps({"id": f"chunk_{i}"}) for i in range(6)]) + "\n")
        f2.write_text("\n".join([json.dumps({"id": f"chunk_{i}"}) for i in range(6, 9)]) + "\n")

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        assert constructor._detect_resume_offset(tmp_path) == 3

    def test_removes_empty_output_files(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Empty extraction output files are removed during resume detection."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        f1 = kg_dir / "model_transcriptions.json_output_20260226160000_1_in_1.json"
        f2 = kg_dir / "model_transcriptions.json_output_20260226220000_1_in_1.json"
        f1.write_text("\n".join([json.dumps({"id": f"chunk_{i}"}) for i in range(9)]) + "\n")
        f2.write_text("")  # Empty file from aborted run

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        assert constructor._detect_resume_offset(tmp_path) == 3
        assert not f2.exists()

    def test_invalid_records_are_stripped(
        self,
        tmp_path: Path,
        _mock_atlas_rag: dict,
    ) -> None:
        """Invalid JSON records are removed, count based on valid records only."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        kg_dir = tmp_path / "atlas_output" / "kg_extraction"
        kg_dir.mkdir(parents=True)

        output_file = kg_dir / "model_transcriptions.json_output_20260226_1_in_1.json"
        lines = [json.dumps({"id": f"chunk_{i}"}) for i in range(9)]
        lines.insert(3, "INVALID JSON LINE")
        lines.insert(7, "ANOTHER BAD LINE")
        output_file.write_text("\n".join(lines) + "\n")

        config = KGConfig(backend_options={"batch_size_triple": 3})
        constructor = AtlasRagConstructor(config)

        # 9 valid records -> 3 batches (2 invalid lines stripped)
        assert constructor._detect_resume_offset(tmp_path) == 3

        # File should have exactly 9 valid JSONL lines
        remaining = [ln for ln in output_file.read_text().strip().split("\n") if ln.strip()]
        assert len(remaining) == 9
        for line in remaining:
            json.loads(line)  # Should not raise


class TestBuildLLMClient:
    """Tests for _build_llm_client (delegates to unified LLMClient)."""

    def test_ollama_provider(self, _mock_atlas_rag: dict, mocker: MockerFixture) -> None:
        """Test LLM client construction for Ollama provider."""
        mock_create = mocker.patch("arandu.kg.atlas_backend.create_llm_client")

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="ollama", ollama_url="http://localhost:11434/v1")
        constructor = AtlasRagConstructor(config)
        constructor._build_llm_client()

        mock_create.assert_called_once_with(
            provider="ollama",
            model_id=config.model_id,
            base_url="http://localhost:11434/v1",
        )

    def test_custom_provider_with_base_url(
        self, _mock_atlas_rag: dict, mocker: MockerFixture
    ) -> None:
        """Test custom provider passes base_url to create_llm_client."""
        mock_create = mocker.patch("arandu.kg.atlas_backend.create_llm_client")

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="custom", base_url="http://my-api:8000/v1")
        constructor = AtlasRagConstructor(config)
        constructor._build_llm_client()

        mock_create.assert_called_once_with(
            provider="custom",
            model_id=config.model_id,
            base_url="http://my-api:8000/v1",
        )

    def test_custom_provider_without_base_url(
        self, _mock_atlas_rag: dict, mocker: MockerFixture
    ) -> None:
        """Test custom provider passes None base_url to create_llm_client."""
        mock_create = mocker.patch("arandu.kg.atlas_backend.create_llm_client")

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(provider="custom", base_url=None)
        constructor = AtlasRagConstructor(config)
        constructor._build_llm_client()

        mock_create.assert_called_once_with(
            provider="custom",
            model_id=config.model_id,
            base_url=None,
        )


class TestInjectConceptPrompts:
    """Tests for _inject_concept_prompts."""

    def test_injects_pt_prompts(self, _mock_atlas_rag: dict) -> None:
        """Test Portuguese concept prompts are injected into CONCEPT_INSTRUCTIONS."""
        from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (
            CONCEPT_INSTRUCTIONS,
        )

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        constructor._inject_concept_prompts()

        assert "pt" in CONCEPT_INSTRUCTIONS

    def test_skips_if_language_already_present(self, _mock_atlas_rag: dict) -> None:
        """Test injection is skipped when language already in dict."""
        from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (
            CONCEPT_INSTRUCTIONS,
        )

        from arandu.kg.atlas_backend import AtlasRagConstructor

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

        from arandu.kg.atlas_backend import AtlasRagConstructor

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

    def test_passes_resume_from(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test resume_from is forwarded to ProcessingConfig."""
        from atlas_rag.kg_construction.triple_config import ProcessingConfig

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig()
        constructor = AtlasRagConstructor(config)

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        constructor._build_processing_config(input_dir, output_dir, resume_from=7)

        call_kwargs = ProcessingConfig.call_args[1]
        assert call_kwargs["resume_from"] == 7


class TestRunPipeline:
    """Tests for _run_pipeline."""

    @staticmethod
    def _setup_missing_csv(output_dir: Path) -> Path:
        """Create a minimal missing_concepts CSV so resume wrapper works."""
        triples_dir = output_dir / "atlas_output" / "triples_csv"
        triples_dir.mkdir(parents=True, exist_ok=True)
        csv_file = triples_dir / "missing_concepts_test_from_json.csv"
        csv_file.write_text("node,description,node_type\nfoo,bar,entity\n")
        return csv_file

    def test_all_five_steps_called(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test all 5 atlas-rag pipeline steps are called."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend_options={"include_concept": True})
        constructor = AtlasRagConstructor(config)

        self._setup_missing_csv(tmp_path)
        mock_extractor = MagicMock()
        constructor._run_pipeline(mock_extractor, tmp_path)

        mock_extractor.run_extraction.assert_called_once()
        mock_extractor.convert_json_to_csv.assert_called_once()
        mock_extractor.generate_concept_csv_temp.assert_called_once()
        mock_extractor.create_concept_csv.assert_called_once()
        mock_extractor.convert_to_graphml.assert_called_once()

    def test_skip_concept_steps(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Test concept steps are skipped when include_concept=False."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig(backend_options={"include_concept": False})
        constructor = AtlasRagConstructor(config)

        mock_extractor = MagicMock()
        constructor._run_pipeline(mock_extractor, tmp_path)

        mock_extractor.run_extraction.assert_called_once()
        mock_extractor.convert_json_to_csv.assert_called_once()
        mock_extractor.generate_concept_csv_temp.assert_not_called()
        mock_extractor.create_concept_csv.assert_not_called()
        mock_extractor.convert_to_graphml.assert_called_once()


class TestResumableConceptGeneration:
    """Tests for _run_concept_generation_with_resume."""

    @staticmethod
    def _setup_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
        """Create atlas_output directory structure and a missing_concepts CSV.

        Returns:
            Tuple of (missing_csv, concepts_dir, triples_dir).
        """
        triples_dir = output_dir / "atlas_output" / "triples_csv"
        concepts_dir = output_dir / "atlas_output" / "concepts"
        triples_dir.mkdir(parents=True, exist_ok=True)
        concepts_dir.mkdir(parents=True, exist_ok=True)

        missing_csv = triples_dir / "missing_concepts_test_from_json.csv"
        missing_csv.write_text(
            "node,description,node_type\n"
            "Rio Guaíba,grande rio do sul,entity\n"
            "enchente,evento climático,event\n"
            "afeta,relação causal,relation\n"
        )
        return missing_csv, concepts_dir, triples_dir

    def test_fresh_run(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Fresh run with no prior data — generation runs normally."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)
        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)

        mock_extractor = MagicMock()
        # Simulate atlas-rag writing a shard
        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river in southern Brazil,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_called_once_with(
            batch_size=16,
            language="pt",
        )
        # Final output should be concept_shard_0.csv
        assert shard.exists()
        # Backup should be cleaned up
        assert not missing_csv.with_suffix(".csv.bak").exists()

    def test_full_resume_all_completed(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """All nodes already in accumulator — generation is skipped."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        # Pre-populate accumulator with all nodes
        accumulator = concepts_dir / "concept_completed.csv"
        accumulator.write_text(
            "Rio Guaíba,large river in southern Brazil,entity\n"
            "enchente,climatic event,event\n"
            "afeta,causal relation,relation\n"
        )

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_not_called()
        # Accumulator should be renamed to shard
        shard = concepts_dir / "concept_shard_0.csv"
        assert shard.exists()
        assert not accumulator.exists()

    def test_partial_resume(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Leftover shard absorbed, generation runs with trimmed input."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        # Simulate interrupted shard with 1 completed node
        leftover_shard = concepts_dir / "concept_shard_0.csv"
        leftover_shard.write_text("Rio Guaíba,large river in southern Brazil,entity\n")

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text("enchente,climatic event,event\nafeta,causal relation,relation\n")

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_called_once()
        assert shard.exists()
        # Should contain all 3 nodes (1 from leftover + 2 from new run)
        import csv

        with shard.open(newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3

    def test_corrupted_rows_dropped(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Malformed rows in leftover shard are dropped, valid ones kept."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        # Leftover shard with corrupted rows
        leftover_shard = concepts_dir / "concept_shard_0.csv"
        leftover_shard.write_text(
            "Rio Guaíba,large river in southern Brazil,entity\n"
            "bad_row_only_two_cols,missing\n"
            ",,\n"
            "enchente,climatic event,INVALID_TYPE\n"
            "afeta,causal relation,relation\n"
        )

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text("enchente,climatic event,event\n")

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        assert shard.exists()
        import csv

        with shard.open(newline="") as f:
            rows = list(csv.reader(f))
        node_names = {row[0] for row in rows}
        # Valid rows: Rio Guaíba (entity) and afeta (relation) from leftover
        # Plus enchente (event) from new run
        assert "Rio Guaíba" in node_names
        assert "afeta" in node_names
        assert "enchente" in node_names

    def test_backup_restore_on_dirty_state(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Backup file from previous crash is restored before proceeding."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        # Simulate dirty state: input was trimmed, backup exists
        original_content = missing_csv.read_text()
        backup = missing_csv.with_suffix(".csv.bak")
        backup.write_text(original_content)
        # Overwrite input with trimmed version
        missing_csv.write_text("node,description,node_type\nenchente,evento climático,event\n")

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        # Input should be restored to original (all 3 nodes)
        restored = missing_csv.read_text()
        assert "Rio Guaíba" in restored
        assert "enchente" in restored
        assert "afeta" in restored

    def test_multiple_successive_failures(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Accumulator grows correctly across 2 interruptions."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)

        shard = concepts_dir / "concept_shard_0.csv"
        accumulator = concepts_dir / "concept_completed.csv"

        # --- Failure 1: only 1 node completed, then "crash" ---
        mock_ext1 = MagicMock()

        def write_shard_1(**kwargs: Any) -> None:
            shard.write_text("Rio Guaíba,large river,entity\n")

        mock_ext1.generate_concept_csv_temp.side_effect = write_shard_1
        constructor._run_concept_generation_with_resume(mock_ext1, tmp_path)

        # After first run, shard should have 1 node
        assert shard.exists()

        # Simulate crash: rename shard back to look like interrupted state
        # (In real life, the shard would be a leftover from atlas-rag being killed)
        # For the next resume, we need a leftover shard_0 and an accumulator
        # The finalize step already renamed accumulator -> shard_0
        # So simulate the atlas-rag crash by putting a new partial shard
        # while the previous result is now in concept_shard_0.csv
        # We need to set up for a second resume: put shard content into accumulator
        # and create a new leftover shard
        import csv

        with shard.open(newline="") as f:
            rows_after_first = list(csv.reader(f))
        accumulator.write_text("")
        with accumulator.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_after_first)
        shard.unlink()

        # Simulate a second partial shard from interrupted run
        shard.write_text("enchente,climatic event,event\n")

        # --- Failure 2: second node completed ---
        mock_ext2 = MagicMock()

        def write_shard_2(**kwargs: Any) -> None:
            shard.write_text("afeta,causal relation,relation\n")

        mock_ext2.generate_concept_csv_temp.side_effect = write_shard_2
        constructor._run_concept_generation_with_resume(mock_ext2, tmp_path)

        # Final shard should have all 3 nodes
        assert shard.exists()
        with shard.open(newline="") as f:
            final_rows = list(csv.reader(f))
        assert len(final_rows) == 3
        node_names = {row[0] for row in final_rows}
        assert node_names == {"Rio Guaíba", "enchente", "afeta"}

    def test_empty_input_csv(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Header-only input CSV — no nodes to process, completes immediately."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _, concepts_dir, triples_dir = self._setup_dirs(tmp_path)
        # Overwrite with header-only
        missing_csv = sorted(triples_dir.glob("missing_concepts*_from_json.csv"))[0]
        missing_csv.write_text("node,description,node_type\n")

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_not_called()
        # concept_shard_0.csv must exist so downstream create_concept_csv() has valid input
        assert (concepts_dir / "concept_shard_0.csv").exists()

    def test_merge_guard(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """After resume completes, only concept_shard_0.csv exists in concepts/."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        csv_files = list(concepts_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].name == "concept_shard_0.csv"

    def test_empty_shard_leftover(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Empty/header-only shard leftover — no rows absorbed, generation proceeds."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        # Empty shard from crash right after atlas-rag opened the file
        leftover_shard = concepts_dir / "concept_shard_0.csv"
        leftover_shard.write_text("")

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_called_once()
        assert shard.exists()

    def test_missing_input_csv(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Missing missing_concepts CSV raises FileNotFoundError."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        # Create output structure but no missing_concepts CSV
        triples_dir = tmp_path / "atlas_output" / "triples_csv"
        triples_dir.mkdir(parents=True, exist_ok=True)

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        with pytest.raises(FileNotFoundError, match="No missing_concepts CSV found"):
            constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

    def test_language_kwarg_passed(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Verify language=self._config.language is passed to generate_concept_csv_temp."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _ = self._setup_dirs(tmp_path)

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text("Rio Guaíba,large river,entity\n")

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        call_kwargs = mock_extractor.generate_concept_csv_temp.call_args[1]
        assert call_kwargs["language"] == "pt"

    def test_phantom_missing_concept_dropped(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Rows in missing_concepts.csv whose node ID isn't in triple_nodes/edges are dropped.

        Regression for the missing-node bug observed in cluster job 779433:
        atlas-rag's ``generate_concept`` calls ``temp_kg.predecessors(node_id)``
        for each missing-concepts row and raises ``NetworkXError`` when the
        node isn't in the graph. The prefilter drops phantom rows before
        atlas-rag sees them.
        """
        from arandu.kg.atlas_backend import AtlasRagConstructor

        missing_csv, concepts_dir, triples_dir = self._setup_dirs(tmp_path)

        # Re-write missing_concepts with a phantom row that isn't in any
        # triple_*.csv source; the patch should drop it.
        missing_csv.write_text(
            "node,description,node_type\n"
            "Rio Guaíba,grande rio do sul,entity\n"
            "phantom_node,not in graph anywhere,entity\n"
            "enchente,evento climático,event\n"
        )

        # Companion source CSVs the prefilter consults.
        nodes_csv = triples_dir / "triple_nodes_test_from_json.csv"
        nodes_csv.write_text("name:ID,type,concepts,synsets,:LABEL\nRio Guaíba,entity,,,Node\n")
        edges_csv = triples_dir / "triple_edges_test_from_json.csv"
        edges_csv.write_text(
            ":START_ID,:END_ID,relation,concepts,synsets,:TYPE\n"
            "Rio Guaíba,enchente,causou,,,Relation\n"
        )

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river in southern Brazil,entity\nenchente,climatic event,event\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        # Verify the phantom is dropped from the input atlas-rag ultimately sees.
        # Capture the missing_csv content right before generate_concept_csv_temp is called.
        captured: dict[str, list[str]] = {}

        def capture_then_write(**kwargs: Any) -> None:
            with missing_csv.open() as f:
                captured["names"] = [row[0] for row in csv.reader(f) if row and row[0] != "node"]
            write_shard(**kwargs)

        mock_extractor.generate_concept_csv_temp.side_effect = capture_then_write

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        # The phantom row must NOT have been visible to atlas-rag.
        assert "phantom_node" not in captured["names"], (
            f"phantom row leaked through to atlas-rag: {captured['names']}"
        )
        assert "Rio Guaíba" in captured["names"]
        assert "enchente" in captured["names"]

    def test_no_phantom_drop_when_all_present(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """Prefilter is a no-op when every missing_concepts row is in the graph."""
        from arandu.kg.atlas_backend import AtlasRagConstructor

        missing_csv, concepts_dir, triples_dir = self._setup_dirs(tmp_path)

        # _setup_dirs writes 3 nodes; build matching source CSVs.
        nodes_csv = triples_dir / "triple_nodes_test_from_json.csv"
        nodes_csv.write_text(
            "name:ID,type,concepts,synsets,:LABEL\n"
            "Rio Guaíba,entity,,,Node\n"
            "enchente,event,,,Node\n"
            "afeta,relation,,,Node\n"
        )
        edges_csv = triples_dir / "triple_edges_test_from_json.csv"
        edges_csv.write_text(":START_ID,:END_ID,relation,concepts,synsets,:TYPE\n")

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"
        captured: dict[str, list[str]] = {}

        def write_shard(**kwargs: Any) -> None:
            with missing_csv.open() as f:
                captured["names"] = [row[0] for row in csv.reader(f) if row and row[0] != "node"]
            shard.write_text(
                "Rio Guaíba,large river,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        # All 3 rows survive — no false drops.
        assert set(captured["names"]) == {"Rio Guaíba", "enchente", "afeta"}

    def test_phantom_drop_skipped_when_source_csvs_missing(
        self, tmp_path: Path, _mock_atlas_rag: dict
    ) -> None:
        """Defensive: if triple_nodes/edges CSVs are absent, prefilter is a no-op.

        Triple extraction must run before concept generation, so the source
        CSVs *should* exist. But if a test or odd resume state leaves them
        missing, the prefilter should not crash — atlas-rag will surface
        any actual problem itself.
        """
        from arandu.kg.atlas_backend import AtlasRagConstructor

        _missing_csv, concepts_dir, _triples_dir = self._setup_dirs(tmp_path)
        # Intentionally do NOT create triple_nodes / triple_edges CSVs.

        config = KGConfig(language="pt")
        constructor = AtlasRagConstructor(config)
        mock_extractor = MagicMock()

        shard = concepts_dir / "concept_shard_0.csv"

        def write_shard(**kwargs: Any) -> None:
            shard.write_text(
                "Rio Guaíba,large river,entity\n"
                "enchente,climatic event,event\n"
                "afeta,causal relation,relation\n"
            )

        mock_extractor.generate_concept_csv_temp.side_effect = write_shard

        # Should NOT raise.
        constructor._run_concept_generation_with_resume(mock_extractor, tmp_path)

        mock_extractor.generate_concept_csv_temp.assert_called_once()


class TestPatchedCsvsToTempGraphml:
    """Tests for _patched_csvs_to_temp_graphml orphan node handling."""

    _NODE_FIELDS: ClassVar[list[str]] = ["name:ID", "type", "concepts", "synsets", ":LABEL"]

    @staticmethod
    def _write_nodes_csv(path: Path, rows: list[dict[str, str]]) -> None:
        """Write a triple nodes CSV file."""
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TestPatchedCsvsToTempGraphml._NODE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_edges_csv(path: Path, rows: list[dict[str, str]]) -> None:
        """Write a triple edges CSV file."""
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_orphan_nodes_get_default_attributes(
        self, tmp_path: Path, _mock_atlas_rag: dict
    ) -> None:
        """Edge endpoints missing from nodes CSV get id and type attributes."""
        import pickle

        # Make get_node_id return input as-is (no hashing)
        csv_mod = _mock_atlas_rag["atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml"]
        csv_mod.get_node_id = lambda name, cache=None: name

        from arandu.kg.atlas_backend import _patched_csvs_to_temp_graphml

        nodes_file = tmp_path / "nodes.csv"
        edges_file = tmp_path / "edges.csv"
        self._write_nodes_csv(
            nodes_file,
            [
                {
                    "name:ID": "rio",
                    "type": "entity",
                    "concepts": "",
                    "synsets": "",
                    ":LABEL": "Node",
                },
            ],
        )
        self._write_edges_csv(
            edges_file,
            [
                {
                    ":START_ID": "rio",
                    ":END_ID": "enchente",
                    "relation": "causou",
                    "concepts": "",
                    "synsets": "",
                    ":TYPE": "Relation",
                },
            ],
        )

        config = MagicMock()
        config.output_directory = str(tmp_path / "output")
        config.filename_pattern = "test"

        _patched_csvs_to_temp_graphml(str(nodes_file), str(edges_file), config)

        pkl_path = tmp_path / "output" / "kg_graphml" / "test_without_concept.pkl"
        assert pkl_path.exists()

        with open(pkl_path, "rb") as f:
            g = pickle.load(f)

        # "enchente" was only in edges, should have been created with attributes
        orphan_nodes = [n for n in g.nodes if g.nodes[n].get("id") == "enchente"]
        assert len(orphan_nodes) == 1
        assert g.nodes[orphan_nodes[0]]["type"] == "entity"
        assert g.nodes[orphan_nodes[0]]["id"] == "enchente"

    def test_no_orphans_when_all_nodes_present(self, tmp_path: Path, _mock_atlas_rag: dict) -> None:
        """No orphan warning when all edge endpoints exist in nodes CSV."""
        csv_mod = _mock_atlas_rag["atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml"]
        csv_mod.get_node_id = lambda name, cache=None: name

        from arandu.kg.atlas_backend import _patched_csvs_to_temp_graphml

        nodes_file = tmp_path / "nodes.csv"
        edges_file = tmp_path / "edges.csv"
        self._write_nodes_csv(
            nodes_file,
            [
                {
                    "name:ID": "rio",
                    "type": "entity",
                    "concepts": "",
                    "synsets": "",
                    ":LABEL": "Node",
                },
                {
                    "name:ID": "enchente",
                    "type": "event",
                    "concepts": "",
                    "synsets": "",
                    ":LABEL": "Node",
                },
            ],
        )
        self._write_edges_csv(
            edges_file,
            [
                {
                    ":START_ID": "rio",
                    ":END_ID": "enchente",
                    "relation": "causou",
                    "concepts": "",
                    "synsets": "",
                    ":TYPE": "Relation",
                },
            ],
        )

        config = MagicMock()
        config.output_directory = str(tmp_path / "output")
        config.filename_pattern = "test"

        _patched_csvs_to_temp_graphml(str(nodes_file), str(edges_file), config)

        import pickle

        pkl_path = tmp_path / "output" / "kg_graphml" / "test_without_concept.pkl"
        with open(pkl_path, "rb") as f:
            g = pickle.load(f)

        # "enchente" should keep its original type from nodes CSV
        enchente_nodes = [n for n in g.nodes if g.nodes[n].get("id") == "enchente"]
        assert len(enchente_nodes) == 1
        assert g.nodes[enchente_nodes[0]]["type"] == "event"  # original, not "entity"


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

        from arandu.kg.atlas_backend import AtlasRagConstructor

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

        from arandu.kg.atlas_backend import AtlasRagConstructor

        config = KGConfig()
        constructor = AtlasRagConstructor(config)

        with pytest.raises(FileNotFoundError, match="No GraphML file found"):
            constructor._build_result([], tmp_path)
