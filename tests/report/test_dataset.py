"""Tests for report dataset building."""

from __future__ import annotations

from arandu.qa.schemas import QAPairValidated, QARecordCEP
from arandu.report.collector import RunReport
from arandu.report.dataset import (
    QAPairRow,
    ReportDataset,
    RunSummaryRow,
    TranscriptionRow,
    build_dataset,
)
from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult, JudgeStepResult
from arandu.shared.schemas import (
    EnrichedRecord,
    PipelineMetadata,
    PipelineType,
    SourceMetadata,
)
from tests.report.helpers import make_run_metadata


def _make_enriched_record(
    name: str = "test.mp3",
    participant: str | None = "Maria",
    location: str | None = "Barra de Pelotas",
    is_valid: bool | None = True,
    overall_score: float = 0.85,
) -> EnrichedRecord:
    """Create a sample EnrichedRecord for testing."""
    source = None
    if participant or location:
        source = SourceMetadata(
            participant_name=participant,
            location=location,
            recording_date="2024-05-15",
        )
    # Build a JudgePipelineResult whose flattened mean lines up with the
    # requested overall_score — the test fixtures expose a single knob and
    # the report reader averages the criterion scores.
    quality = JudgePipelineResult(
        stage_results={
            "heuristic_filter": JudgeStepResult(
                criterion_scores={
                    "script_match": CriterionScore(
                        score=overall_score, threshold=0.6, rationale="ok"
                    ),
                    "repetition": CriterionScore(
                        score=overall_score, threshold=0.5, rationale="ok"
                    ),
                    "segment_quality": CriterionScore(
                        score=overall_score, threshold=0.4, rationale="ok"
                    ),
                    "content_density": CriterionScore(
                        score=overall_score, threshold=0.4, rationale="ok"
                    ),
                }
            )
        },
        passed=bool(is_valid),
    )
    return EnrichedRecord(
        file_id="gdrive_123",
        name=name,
        mimeType="audio/mpeg",
        parents=["parent_folder"],
        webContentLink="https://drive.google.com/test",
        transcription_text="Test transcription text.",
        detected_language="pt",
        language_probability=0.95,
        model_id="openai/whisper-large-v3",
        compute_device="cuda",
        processing_duration_sec=15.0,
        transcription_status="completed",
        is_valid=is_valid,
        source_metadata=source,
        transcription_quality=quality,
    )


def _make_judge_result(
    faithfulness: float = 0.9,
    bloom_calibration: float = 0.85,
    informativeness: float = 0.75,
    self_containedness: float = 0.95,
    threshold: float = 0.6,
) -> JudgePipelineResult:
    """Build a JudgePipelineResult with given criterion scores."""
    scores = {
        "faithfulness": CriterionScore(score=faithfulness, threshold=threshold, rationale="ok"),
        "bloom_calibration": CriterionScore(
            score=bloom_calibration, threshold=threshold, rationale="ok"
        ),
        "informativeness": CriterionScore(
            score=informativeness, threshold=threshold, rationale="ok"
        ),
        "self_containedness": CriterionScore(
            score=self_containedness, threshold=threshold, rationale="ok"
        ),
    }
    step = JudgeStepResult(criterion_scores=scores)
    passed = all(cs.passed for cs in scores.values())
    return JudgePipelineResult(
        stage_results={"cep_validation": step},
        passed=passed,
    )


def _make_cep_record(
    source_filename: str = "test.mp3",
    participant: str | None = "Maria",
    location: str | None = "Barra de Pelotas",
    num_pairs: int = 2,
) -> QARecordCEP:
    """Create a sample QARecordCEP for testing."""
    source = None
    if participant or location:
        source = SourceMetadata(
            participant_name=participant,
            location=location,
            recording_date="2024-05-15",
        )

    qa_pairs = []
    for i in range(num_pairs):
        validation = _make_judge_result(
            faithfulness=0.9 - i * 0.1,
            bloom_calibration=0.85,
            informativeness=0.75,
            self_containedness=0.95,
        )
        pair = QAPairValidated(
            question=f"Question {i}?",
            answer=f"Answer {i}.",
            context="Some context text.",
            question_type="factual",
            confidence=0.9,
            bloom_level="analyze",
            is_multi_hop=i == 0,
            hop_count=2 if i == 0 else None,
            validation=validation,
            is_valid=True,
        )
        qa_pairs.append(pair)

    return QARecordCEP(
        source_file_id="gdrive_123",
        source_filename=source_filename,
        source_metadata=source,
        transcription_text="Test transcription text.",
        qa_pairs=qa_pairs,
        model_id="gpt-4o",
        validator_model_id="gpt-4o-mini",
        provider="openai",
        total_pairs=num_pairs,
        validated_pairs=num_pairs,
        bloom_distribution={"analyze": num_pairs},
    )


def _make_run_report(
    pipeline_id: str = "run_001",
    with_transcriptions: bool = True,
    with_cep: bool = True,
) -> RunReport:
    """Create a sample RunReport for testing."""
    pipeline = PipelineMetadata(
        pipeline_id=pipeline_id,
        steps_run=["transcription", "cep"],
    )

    records = []
    if with_transcriptions:
        records = [_make_enriched_record()]

    cep_records = []
    if with_cep:
        cep_records = [_make_cep_record()]

    return RunReport(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        transcription_records=records,
        cep_records=cep_records,
    )


class TestBuildDataset:
    """Tests for build_dataset function."""

    def test_build_with_single_run(self) -> None:
        """Test building dataset from a single run."""
        report = _make_run_report()
        dataset = build_dataset([report])

        assert len(dataset.runs) == 1
        assert len(dataset.transcriptions) == 1
        assert len(dataset.qa_pairs) == 2
        assert dataset.runs[0].pipeline_id == "run_001"

    def test_build_with_multiple_runs(self) -> None:
        """Test building dataset from multiple runs."""
        reports = [_make_run_report("run_001"), _make_run_report("run_002")]
        dataset = build_dataset(reports)

        assert len(dataset.runs) == 2
        assert len(dataset.transcriptions) == 2
        assert len(dataset.qa_pairs) == 4

    def test_build_with_empty_reports(self) -> None:
        """Test building dataset with no reports."""
        dataset = build_dataset([])

        assert len(dataset.runs) == 0
        assert len(dataset.transcriptions) == 0
        assert len(dataset.qa_pairs) == 0
        assert dataset.generated_at != ""

    def test_qa_pair_flattening(self) -> None:
        """Test that QA pair fields are correctly flattened."""
        report = _make_run_report()
        dataset = build_dataset([report])

        qa = dataset.qa_pairs[0]
        assert qa.pipeline_id == "run_001"
        assert qa.source_filename == "test.mp3"
        assert qa.participant_name == "Maria"
        assert qa.location == "Barra de Pelotas"
        assert qa.bloom_level == "analyze"
        assert qa.is_multi_hop is True
        assert qa.hop_count == 2
        assert qa.confidence == 0.9
        assert qa.faithfulness == 0.9
        assert qa.bloom_calibration == 0.85
        assert qa.informativeness == 0.75
        assert qa.self_containedness == 0.95
        assert qa.overall_score == 0.8625
        assert qa.model_id == "gpt-4o"
        assert qa.validator_model_id == "gpt-4o-mini"
        assert qa.provider == "openai"
        assert qa.is_valid is True

    def test_transcription_flattening(self) -> None:
        """Test that transcription fields are correctly flattened."""
        report = _make_run_report()
        dataset = build_dataset([report])

        trans = dataset.transcriptions[0]
        assert trans.pipeline_id == "run_001"
        assert trans.source_filename == "test.mp3"
        assert trans.participant_name == "Maria"
        assert trans.location == "Barra de Pelotas"
        assert trans.is_valid is True
        assert trans.overall_quality == 0.85
        assert trans.script_match == 0.9
        assert trans.repetition == 0.8
        assert trans.segment_quality == 0.85
        assert trans.content_density == 0.7
        assert trans.processing_duration_sec == 15.0
        assert trans.model_id == "openai/whisper-large-v3"
        assert trans.detected_language == "pt"

    def test_run_summary_computation(self) -> None:
        """Test run summary row creation."""
        report = _make_run_report()
        dataset = build_dataset([report])

        run = dataset.runs[0]
        assert run.pipeline_id == "run_001"
        assert run.steps_run == ["transcription", "cep"]
        assert run.status == "unknown"  # no metadata in this fixture

    def test_missing_source_metadata(self) -> None:
        """Test flattening with missing source metadata."""
        record = _make_enriched_record(participant=None, location=None)
        report = RunReport(
            pipeline_id="run_no_meta",
            transcription_records=[record],
        )
        dataset = build_dataset([report])

        trans = dataset.transcriptions[0]
        assert trans.participant_name is None
        assert trans.location is None

    def test_missing_validation_scores(self) -> None:
        """Test flattening QA pairs without validation."""
        from arandu.qa.schemas import QAPairCEP

        qa_pair = QAPairCEP(
            question="Test?",
            answer="Answer.",
            context="Context.",
            question_type="factual",
            confidence=0.8,
            bloom_level="remember",
        )
        cep_record = QARecordCEP(
            source_file_id="gdrive_123",
            source_filename="test.mp3",
            transcription_text="Test.",
            qa_pairs=[qa_pair],
            model_id="gpt-4o",
            provider="openai",
            total_pairs=1,
            bloom_distribution={"remember": 1},
        )
        report = RunReport(
            pipeline_id="run_no_val",
            cep_records=[cep_record],
        )
        dataset = build_dataset([report])

        qa = dataset.qa_pairs[0]
        assert qa.faithfulness is None
        assert qa.bloom_calibration is None
        assert qa.overall_score is None


class TestReportDataset:
    """Tests for ReportDataset computed fields."""

    def test_pipeline_ids(self) -> None:
        """Test unique pipeline ID extraction."""
        dataset = ReportDataset(
            runs=[
                RunSummaryRow(pipeline_id="run_b"),
                RunSummaryRow(pipeline_id="run_a"),
            ]
        )
        assert dataset.pipeline_ids == ["run_a", "run_b"]

    def test_locations(self) -> None:
        """Test unique location extraction."""
        dataset = ReportDataset(
            transcriptions=[
                TranscriptionRow(pipeline_id="r1", source_filename="a.mp3", location="Pelotas"),
                TranscriptionRow(pipeline_id="r1", source_filename="b.mp3", location="Pelotas"),
                TranscriptionRow(pipeline_id="r1", source_filename="c.mp3", location="Cangucu"),
            ]
        )
        assert dataset.locations == ["Cangucu", "Pelotas"]

    def test_participants(self) -> None:
        """Test unique participant extraction."""
        dataset = ReportDataset(
            qa_pairs=[
                QAPairRow(pipeline_id="r1", source_filename="a.mp3", participant_name="Maria"),
                QAPairRow(pipeline_id="r1", source_filename="b.mp3", participant_name="Joao"),
                QAPairRow(pipeline_id="r1", source_filename="c.mp3", participant_name="Maria"),
            ]
        )
        assert dataset.participants == ["Joao", "Maria"]

    def test_bloom_levels(self) -> None:
        """Test unique Bloom level extraction."""
        dataset = ReportDataset(
            qa_pairs=[
                QAPairRow(pipeline_id="r1", source_filename="a.mp3", bloom_level="analyze"),
                QAPairRow(pipeline_id="r1", source_filename="b.mp3", bloom_level="remember"),
                QAPairRow(pipeline_id="r1", source_filename="c.mp3", bloom_level="analyze"),
            ]
        )
        assert dataset.bloom_levels == ["analyze", "remember"]

    def test_empty_dataset(self) -> None:
        """Test computed fields on empty dataset."""
        dataset = ReportDataset()
        assert dataset.pipeline_ids == []
        assert dataset.locations == []
        assert dataset.participants == []
        assert dataset.bloom_levels == []


class TestRunSummaryNewFields:
    """Tests for new RunSummaryRow fields added for dashboard support."""

    def test_run_summary_includes_model_ids(self) -> None:
        """Verify model IDs are extracted from config."""
        transcription_meta = make_run_metadata(
            pipeline_type=PipelineType.TRANSCRIPTION,
            config_values={"model_id": "openai/whisper-large-v3", "quality_threshold": 0.7},
        )
        cep_meta = make_run_metadata(
            pipeline_type=PipelineType.CEP,
            config_values={
                "model_id": "gpt-4o",
                "validator_model_id": "gpt-4o-mini",
                "provider": "openai",
            },
        )
        report = RunReport(
            pipeline_id="run_model_ids",
            transcription_metadata=transcription_meta,
            cep_metadata=cep_meta,
        )
        dataset = build_dataset([report])
        run = dataset.runs[0]

        assert run.model_id == "openai/whisper-large-v3"
        assert run.cep_model_id == "gpt-4o"
        assert run.validator_model_id == "gpt-4o-mini"
        assert run.provider == "openai"

    def test_run_summary_includes_thresholds(self) -> None:
        """Verify threshold values are extracted from config."""
        transcription_meta = make_run_metadata(
            pipeline_type=PipelineType.TRANSCRIPTION,
            config_values={"quality_threshold": 0.65},
        )
        cep_meta = make_run_metadata(
            pipeline_type=PipelineType.CEP,
            config_values={"validation_threshold": 0.75},
        )
        report = RunReport(
            pipeline_id="run_thresholds",
            transcription_metadata=transcription_meta,
            cep_metadata=cep_meta,
        )
        dataset = build_dataset([report])
        run = dataset.runs[0]

        assert run.quality_threshold == 0.65
        assert run.validation_threshold == 0.75

    def test_run_summary_validity_counts(self) -> None:
        """Verify valid/invalid counts are computed from records."""
        valid_record = _make_enriched_record(name="valid.mp3", is_valid=True)
        invalid_record = _make_enriched_record(name="invalid.mp3", is_valid=False)

        # Build a CEP record with 2 valid + 1 invalid QA pair
        invalid_qa = QAPairValidated(
            question="Q?",
            answer="A.",
            context="ctx",
            question_type="factual",
            confidence=0.5,
            bloom_level="remember",
            validation=_make_judge_result(
                faithfulness=0.3,
                bloom_calibration=0.3,
                informativeness=0.3,
                self_containedness=0.3,
                threshold=0.6,
            ),
            is_valid=False,
        )
        # _make_cep_record produces 2 valid pairs; add 1 invalid by building manually
        cep_record = QARecordCEP(
            source_file_id="gdrive_123",
            source_filename="test.mp3",
            source_metadata=SourceMetadata(
                participant_name="Maria", location="Pelotas", recording_date="2024-05-15"
            ),
            transcription_text="Test.",
            qa_pairs=[
                QAPairValidated(
                    question="Q1?",
                    answer="A1.",
                    context="ctx",
                    question_type="factual",
                    confidence=0.9,
                    bloom_level="analyze",
                    validation=_make_judge_result(
                        faithfulness=0.9,
                        bloom_calibration=0.85,
                        informativeness=0.75,
                        self_containedness=0.95,
                    ),
                    is_valid=True,
                ),
                QAPairValidated(
                    question="Q2?",
                    answer="A2.",
                    context="ctx",
                    question_type="factual",
                    confidence=0.85,
                    bloom_level="analyze",
                    validation=_make_judge_result(
                        faithfulness=0.8,
                        bloom_calibration=0.85,
                        informativeness=0.75,
                        self_containedness=0.95,
                    ),
                    is_valid=True,
                ),
                invalid_qa,
            ],
            model_id="gpt-4o",
            provider="openai",
            total_pairs=3,
            bloom_distribution={"analyze": 2, "remember": 1},
        )

        report = RunReport(
            pipeline_id="run_counts",
            transcription_records=[valid_record, invalid_record],
            cep_records=[cep_record],
        )
        dataset = build_dataset([report])
        run = dataset.runs[0]

        assert run.valid_transcriptions == 1
        assert run.invalid_transcriptions == 1
        assert run.valid_qa_pairs == 2
        assert run.invalid_qa_pairs == 1

    def test_run_summary_missing_config(self) -> None:
        """Verify graceful handling when metadata or config keys are absent."""
        # No metadata at all
        report = RunReport(pipeline_id="run_no_meta")
        dataset = build_dataset([report])
        run = dataset.runs[0]

        assert run.model_id is None
        assert run.cep_model_id is None
        assert run.validator_model_id is None
        assert run.provider is None
        assert run.quality_threshold is None
        assert run.validation_threshold is None
        assert run.valid_transcriptions == 0
        assert run.invalid_transcriptions == 0

    def test_run_summary_partial_config_keys(self) -> None:
        """Verify missing config keys default to None gracefully."""
        transcription_meta = make_run_metadata(
            pipeline_type=PipelineType.TRANSCRIPTION,
            config_values={"model_id": "whisper-small"},  # no quality_threshold key
        )
        report = RunReport(
            pipeline_id="run_partial",
            transcription_metadata=transcription_meta,
        )
        dataset = build_dataset([report])
        run = dataset.runs[0]

        assert run.model_id == "whisper-small"
        assert run.quality_threshold is None
