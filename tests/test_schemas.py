"""Tests for Pydantic schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from gtranscriber.schemas import (
    EntityCoverageResult,
    EvaluationReport,
    GraphConnectivity,
    InputRecord,
    KGMetadata,
    QAPair,
    QAPairCEP,
    QAPairValidated,
    RelationMetricsResult,
    SemanticQualityResult,
    TranscriptionSegment,
)


class TestInputRecord:
    """Tests for InputRecord schema."""

    def test_valid_initialization(self) -> None:
        """Test initialization with valid data."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder456"],
            webContentLink="https://drive.google.com/file/123/download",
            size_bytes=1024,
        )

        assert record.gdrive_id == "file123"
        assert record.name == "test.mp3"
        assert record.mimeType == "audio/mpeg"
        assert record.parents == ["folder456"]
        assert record.size_bytes == 1024

    def test_parse_parents_from_string(self) -> None:
        """Test parsing parents field from JSON string."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents='["folder1", "folder2"]',
            webContentLink="https://drive.google.com/file/123/download",
        )

        assert record.parents == ["folder1", "folder2"]

    def test_parse_parents_from_single_quoted_string(self) -> None:
        """Test parsing parents field from single-quoted JSON string."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents="['folder1', 'folder2']",
            webContentLink="https://drive.google.com/file/123/download",
        )

        assert record.parents == ["folder1", "folder2"]

    def test_parse_parents_invalid_json(self) -> None:
        """Test parsing parents with invalid JSON returns empty list."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents="invalid_json",
            webContentLink="https://drive.google.com/file/123/download",
        )

        assert record.parents == []

    def test_parse_size_bytes_from_string(self) -> None:
        """Test parsing size_bytes from string."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder"],
            webContentLink="https://drive.google.com/file/123/download",
            size_bytes="2048",
        )

        assert record.size_bytes == 2048

    def test_parse_size_bytes_invalid_string(self) -> None:
        """Test parsing size_bytes with invalid string returns None."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder"],
            webContentLink="https://drive.google.com/file/123/download",
            size_bytes="invalid",
        )

        assert record.size_bytes is None

    def test_size_bytes_none(self) -> None:
        """Test size_bytes can be None."""
        record = InputRecord(
            gdrive_id="file123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder"],
            webContentLink="https://drive.google.com/file/123/download",
            size_bytes=None,
        )

        assert record.size_bytes is None

    def test_web_content_link_alias(self) -> None:
        """Test webContentLink field alias."""
        data = {
            "gdrive_id": "file123",
            "name": "test.mp3",
            "mimeType": "audio/mpeg",
            "parents": ["folder"],
            "webContentLink": "https://drive.google.com/file/123/download",
        }
        record = InputRecord(**data)

        assert record.web_content_link == "https://drive.google.com/file/123/download"


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment schema."""

    def test_valid_segment(self) -> None:
        """Test valid transcription segment."""
        segment = TranscriptionSegment(text="Hello world", start=0.0, end=1.5)

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 1.5


class TestQAPair:
    """Tests for QAPair schema."""

    def test_valid_qa_pair(self) -> None:
        """Test valid QA pair initialization."""
        qa_pair = QAPair(
            question="What is the capital of France?",
            answer="Paris",
            context="The capital of France is Paris.",
            question_type="factual",
            confidence=0.95,
        )

        assert qa_pair.question == "What is the capital of France?"
        assert qa_pair.answer == "Paris"
        assert qa_pair.confidence == 0.95
        assert qa_pair.question_type == "factual"

    def test_confidence_boundary_min(self) -> None:
        """Test minimum confidence boundary."""
        qa_pair = QAPair(
            question="Test?",
            answer="Answer",
            context="Context",
            question_type="factual",
            confidence=0.0,
        )

        assert qa_pair.confidence == 0.0

    def test_confidence_boundary_max(self) -> None:
        """Test maximum confidence boundary."""
        qa_pair = QAPair(
            question="Test?",
            answer="Answer",
            context="Context",
            question_type="factual",
            confidence=1.0,
        )

        assert qa_pair.confidence == 1.0

    def test_confidence_below_min(self) -> None:
        """Test validation error when confidence below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="factual",
                confidence=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_confidence_above_max(self) -> None:
        """Test validation error when confidence above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="factual",
                confidence=1.1,
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_valid_time_range(self) -> None:
        """Test valid time range for temporal QA pairs."""
        qa_pair = QAPair(
            question="Test?",
            answer="Answer",
            context="Context",
            question_type="temporal",
            confidence=0.9,
            start_time=0.0,
            end_time=5.0,
        )

        assert qa_pair.start_time == 0.0
        assert qa_pair.end_time == 5.0

    def test_start_time_without_end_time(self) -> None:
        """Test validation error when start_time without end_time."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="temporal",
                confidence=0.9,
                start_time=0.0,
                end_time=None,
            )
        assert "end_time required when start_time is provided" in str(exc_info.value)

    def test_end_time_without_start_time(self) -> None:
        """Test validation error when end_time without start_time."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="temporal",
                confidence=0.9,
                start_time=None,
                end_time=5.0,
            )
        assert "start_time required when end_time is provided" in str(exc_info.value)

    def test_start_time_greater_than_end_time(self) -> None:
        """Test validation error when start_time >= end_time."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="temporal",
                confidence=0.9,
                start_time=5.0,
                end_time=3.0,
            )
        assert "start_time must be less than end_time" in str(exc_info.value)

    def test_start_time_equal_to_end_time(self) -> None:
        """Test validation error when start_time equals end_time."""
        with pytest.raises(ValidationError) as exc_info:
            QAPair(
                question="Test?",
                answer="Answer",
                context="Context",
                question_type="temporal",
                confidence=0.9,
                start_time=5.0,
                end_time=5.0,
            )
        assert "start_time must be less than end_time" in str(exc_info.value)


class TestQAPairCEPGenerationPrompt:
    """Tests for QAPairCEP.generation_prompt field."""

    def test_defaults_to_none(self) -> None:
        """Test that generation_prompt defaults to None."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
        )
        assert pair.generation_prompt is None

    def test_accepts_string(self) -> None:
        """Test that generation_prompt accepts a string value."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            generation_prompt="Generate a question about...",
        )
        assert pair.generation_prompt == "Generate a question about..."

    def test_inherited_by_qa_pair_validated(self) -> None:
        """Test that QAPairValidated inherits generation_prompt."""
        pair = QAPairValidated(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            generation_prompt="The prompt",
            validation=None,
            is_valid=True,
        )
        assert pair.generation_prompt == "The prompt"

    def test_included_in_serialization(self) -> None:
        """Test that generation_prompt is included in JSON serialization."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            generation_prompt="My prompt",
        )
        data = pair.model_dump()
        assert "generation_prompt" in data
        assert data["generation_prompt"] == "My prompt"

    def test_none_included_in_serialization(self) -> None:
        """Test that None generation_prompt is included in JSON serialization."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
        )
        data = pair.model_dump()
        assert "generation_prompt" in data
        assert data["generation_prompt"] is None


class TestQAPairCEPReasoningTraceCoercion:
    """Tests for QAPairCEP.reasoning_trace list-to-string coercion."""

    def test_string_passthrough(self) -> None:
        """Test that a string reasoning_trace passes through unchanged."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace="rio sobe -> barco em risco",
        )
        assert pair.reasoning_trace == "rio sobe -> barco em risco"

    def test_list_coerced_to_string(self) -> None:
        """Test that a list reasoning_trace is joined with ' -> '."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace=["rio sobe", "barco em risco"],
        )
        assert pair.reasoning_trace == "rio sobe -> barco em risco"

    def test_none_accepted(self) -> None:
        """Test that None reasoning_trace is accepted."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace=None,
        )
        assert pair.reasoning_trace is None

    def test_empty_list_coerced_to_empty_string(self) -> None:
        """Test that an empty list reasoning_trace becomes empty string."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace=[],
        )
        assert pair.reasoning_trace == ""

    def test_single_item_list(self) -> None:
        """Test that a single-item list is joined correctly."""
        pair = QAPairCEP(
            question="Q?",
            answer="A",
            context="C",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace=["single step"],
        )
        assert pair.reasoning_trace == "single step"


class TestKGMetadata:
    """Tests for KGMetadata schema."""

    def test_valid_kg_metadata(self) -> None:
        """Test valid KG metadata initialization."""
        metadata = KGMetadata(
            graph_id="graph123",
            source_documents=["doc1", "doc2"],
            model_id="llama3.1:8b",
            provider="ollama",
            language="pt",
        )

        assert metadata.graph_id == "graph123"
        assert len(metadata.source_documents) == 2
        assert metadata.language == "pt"

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading KG metadata."""
        metadata = KGMetadata(
            graph_id="graph123",
            source_documents=["doc1"],
            model_id="llama3.1:8b",
            provider="ollama",
        )

        path = tmp_path / "kg_metadata.json"
        metadata.save(path)

        loaded = KGMetadata.load(path)
        assert loaded.graph_id == metadata.graph_id
        assert loaded.source_documents == metadata.source_documents


class TestEvaluationSchemas:
    """Tests for evaluation-related schemas."""

    def test_graph_connectivity(self) -> None:
        """Test GraphConnectivity schema."""
        connectivity = GraphConnectivity(
            average_degree=2.5,
            connected_components=1,
            largest_component_size=100,
            density=0.05,
        )

        assert connectivity.average_degree == 2.5
        assert connectivity.density == 0.05

    def test_entity_coverage_result(self) -> None:
        """Test EntityCoverageResult schema and computed field."""
        coverage = EntityCoverageResult(
            total_entities=100,
            unique_entities=80,
            entity_density=5.5,
            entity_type_distribution={"PERSON": 30, "LOCATION": 25, "ORG": 25},
        )

        assert coverage.total_entities == 100
        assert coverage.unique_entities == 80
        assert coverage.entity_diversity == 0.8

    def test_entity_coverage_diversity_zero_total(self) -> None:
        """Test entity_diversity when total_entities is zero."""
        coverage = EntityCoverageResult(
            total_entities=0,
            unique_entities=0,
            entity_density=0.0,
            entity_type_distribution={},
        )

        assert coverage.entity_diversity == 0.0

    def test_relation_metrics_result(self) -> None:
        """Test RelationMetricsResult schema and computed field."""
        connectivity = GraphConnectivity(
            average_degree=2.5,
            connected_components=1,
            largest_component_size=100,
            density=0.05,
        )

        metrics = RelationMetricsResult(
            total_relations=50,
            unique_relations=30,
            relation_density=0.5,
            graph_connectivity=connectivity,
        )

        assert metrics.total_relations == 50
        assert metrics.unique_relations == 30
        assert metrics.relation_diversity == 0.6

    def test_relation_metrics_diversity_zero_total(self) -> None:
        """Test relation_diversity when total_relations is zero."""
        connectivity = GraphConnectivity(
            average_degree=0.0,
            connected_components=1,
            largest_component_size=0,
            density=0.0,
        )

        metrics = RelationMetricsResult(
            total_relations=0,
            unique_relations=0,
            relation_density=0.0,
            graph_connectivity=connectivity,
        )

        assert metrics.relation_diversity == 0.0

    def test_semantic_quality_result(self) -> None:
        """Test SemanticQualityResult schema."""
        quality = SemanticQualityResult(
            coherence_score=0.85,
            information_density=2.5,
            knowledge_coverage=0.75,
        )

        assert quality.coherence_score == 0.85
        assert quality.information_density == 2.5
        assert quality.knowledge_coverage == 0.75


class TestEvaluationReport:
    """Tests for EvaluationReport schema."""

    def test_minimal_evaluation_report(self) -> None:
        """Test evaluation report with minimal data."""
        report = EvaluationReport(
            dataset_name="test_dataset",
            total_documents=10,
            total_qa_pairs=100,
        )

        assert report.dataset_name == "test_dataset"
        assert report.total_documents == 10
        assert report.total_qa_pairs == 100
        assert report.overall_score == 0.0

    def test_overall_score_with_qa_only(self) -> None:
        """Test overall score computation with only QA metrics."""
        report = EvaluationReport(
            dataset_name="test_dataset",
            total_documents=10,
            total_qa_pairs=100,
            qa_f1_score=0.8,
        )

        # With only QA F1, overall score should equal F1 score
        assert report.overall_score == 0.8

    def test_overall_score_all_components(self) -> None:
        """Test overall score computation with all components."""
        entity_coverage = EntityCoverageResult(
            total_entities=100,
            unique_entities=80,
            entity_density=5.0,
            entity_type_distribution={"PERSON": 50, "LOCATION": 30},
        )

        connectivity = GraphConnectivity(
            average_degree=2.0,
            connected_components=1,
            largest_component_size=100,
            density=0.05,
        )

        relation_metrics = RelationMetricsResult(
            total_relations=60,
            unique_relations=40,
            relation_density=0.6,
            graph_connectivity=connectivity,
        )

        semantic_quality = SemanticQualityResult(
            coherence_score=0.9,
            information_density=3.0,
            knowledge_coverage=0.85,
        )

        report = EvaluationReport(
            dataset_name="test_dataset",
            total_documents=10,
            total_qa_pairs=100,
            qa_f1_score=0.85,
            entity_coverage=entity_coverage,
            relation_metrics=relation_metrics,
            semantic_quality=semantic_quality,
        )

        # Weighted average:
        # QA F1 (0.85) * 0.3 = 0.255
        # Entity diversity (0.8) * 0.2 = 0.16
        # Relation density normalized (0.6/3.0) * 0.2 = 0.04
        # Semantic coherence (0.9) * 0.3 = 0.27
        # Total = 0.725
        expected_score = 0.725
        assert abs(report.overall_score - expected_score) < 0.01

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading evaluation report."""
        report = EvaluationReport(
            dataset_name="test_dataset",
            total_documents=10,
            total_qa_pairs=100,
            qa_f1_score=0.8,
            recommendations=["Improve entity coverage", "Add more temporal questions"],
        )

        path = tmp_path / "eval_report.json"
        report.save(path)

        loaded = EvaluationReport.load(path)
        assert loaded.dataset_name == report.dataset_name
        assert loaded.total_documents == report.total_documents
        assert loaded.qa_f1_score == report.qa_f1_score
        assert len(loaded.recommendations) == 2
