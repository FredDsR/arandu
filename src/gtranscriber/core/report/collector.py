"""Results collection and aggregation for report generation.

Discovers and loads pipeline results by scanning the filesystem directly,
without relying on index.json which can become stale.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from gtranscriber.schemas import (
    ConfigSnapshot,
    EnrichedRecord,
    PipelineMetadata,
    QARecordCEP,
    RunMetadata,
)

logger = logging.getLogger(__name__)


class RunReport(BaseModel):
    """Aggregated data for a single pipeline run.

    Contains metadata and outputs from all pipeline steps for a given run.
    """

    pipeline_id: str = Field(..., description="Pipeline ID for this run")
    pipeline: PipelineMetadata | None = Field(default=None, description="Pipeline-level metadata")
    transcription_metadata: RunMetadata | None = Field(
        default=None, description="Transcription step metadata"
    )
    cep_metadata: RunMetadata | None = Field(default=None, description="CEP step metadata")
    transcription_records: list[EnrichedRecord] = Field(
        default_factory=list, description="Transcription output records"
    )
    cep_records: list[QARecordCEP] = Field(default_factory=list, description="CEP QA records")


class ResultsCollector:
    """Aggregate pipeline results by scanning the filesystem directly.

    Does NOT use results/index.json — the PCAD sync process can leave
    it stale. Instead, discovers runs by walking the results directory.
    """

    def __init__(self, results_dir: str | Path) -> None:
        """Initialize the results collector.

        Args:
            results_dir: Path to the results directory containing pipeline runs.
        """
        self.results_dir = Path(results_dir)

    def discover_runs(self) -> list[str]:
        """List run IDs by scanning subdirectories of results_dir.

        Returns:
            List of pipeline IDs (subdirectory names).
        """
        if not self.results_dir.exists():
            return []

        return [
            d.name for d in self.results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    def load_run(self, pipeline_id: str) -> RunReport:
        """Load a single run by reading its pipeline.json and step outputs.

        Args:
            pipeline_id: The pipeline ID to load.

        Returns:
            RunReport containing all available data for this run.

        Raises:
            FileNotFoundError: If the pipeline directory doesn't exist.
        """
        pipeline_dir = self.results_dir / pipeline_id
        if not pipeline_dir.exists():
            raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")

        report = RunReport(pipeline_id=pipeline_id)

        # Load pipeline metadata
        pipeline_json = pipeline_dir / "pipeline.json"
        if pipeline_json.exists():
            report.pipeline = PipelineMetadata.load(pipeline_json)

        # Load transcription step
        transcription_dir = pipeline_dir / "transcription"
        if transcription_dir.exists():
            # Load run metadata
            run_metadata_file = transcription_dir / "run_metadata.json"
            if run_metadata_file.exists():
                report.transcription_metadata = RunMetadata.load(run_metadata_file)

            # Load output records
            outputs_dir = transcription_dir / "outputs"
            if outputs_dir.exists():
                for output_file in outputs_dir.glob("*.json"):
                    try:
                        record = EnrichedRecord.model_validate_json(output_file.read_text())
                        report.transcription_records.append(record)
                    except Exception:
                        logger.debug(
                            "Skipping invalid transcription file: %s",
                            output_file,
                            exc_info=True,
                        )

        # Load CEP step
        cep_dir = pipeline_dir / "cep"
        if cep_dir.exists():
            # Load run metadata
            run_metadata_file = cep_dir / "run_metadata.json"
            if run_metadata_file.exists():
                report.cep_metadata = RunMetadata.load(run_metadata_file)

            # Load output records
            outputs_dir = cep_dir / "outputs"
            if outputs_dir.exists():
                for output_file in outputs_dir.glob("*_cep_qa.json"):
                    try:
                        record = QARecordCEP.load(output_file)
                        report.cep_records.append(record)
                    except Exception:
                        logger.debug(
                            "Skipping invalid CEP file: %s",
                            output_file,
                            exc_info=True,
                        )

        return report

    def load_all_runs(self) -> list[RunReport]:
        """Discover and load all runs from disk.

        Returns:
            List of RunReport objects for all discovered pipeline runs.
        """
        run_ids = self.discover_runs()
        reports = []

        for run_id in run_ids:
            try:
                report = self.load_run(run_id)
                reports.append(report)
            except Exception:
                logger.warning("Failed to load run: %s", run_id, exc_info=True)

        return reports

    def load_run_config(self, pipeline_id: str, step: str) -> ConfigSnapshot | None:
        """Load configuration snapshot for a specific pipeline step.

        Args:
            pipeline_id: The pipeline ID to load config for.
            step: The pipeline step name (e.g. "transcription", "cep").

        Returns:
            ConfigSnapshot if available, None otherwise.
        """
        run_metadata_file = self.results_dir / pipeline_id / step / "run_metadata.json"
        if run_metadata_file.exists():
            try:
                metadata = RunMetadata.load(run_metadata_file)
                return metadata.config
            except Exception:
                logger.debug(
                    "Failed to load run_metadata for %s/%s", pipeline_id, step, exc_info=True
                )
        return None

    def load_qa_record(self, pipeline_id: str, source_filename: str) -> QARecordCEP | None:
        """Load a single CEP QA record by source filename.

        Scans the CEP outputs directory for a file matching the source filename
        pattern: `{source_filename_stem}_cep_qa.json`.

        Args:
            pipeline_id: The pipeline ID containing the record.
            source_filename: Original source filename to match.

        Returns:
            QARecordCEP if found, None otherwise.
        """
        stem = Path(source_filename).stem
        outputs_dir = self.results_dir / pipeline_id / "cep" / "outputs"
        target = outputs_dir / f"{stem}_cep_qa.json"
        if target.exists():
            try:
                return QARecordCEP.load(target)
            except Exception:
                logger.debug("Failed to load QA record: %s", target, exc_info=True)
        return None

    def load_transcription_record(
        self, pipeline_id: str, source_filename: str
    ) -> EnrichedRecord | None:
        """Load a single transcription record by source filename.

        Scans the transcription outputs directory for a JSON file matching
        the source filename.

        Args:
            pipeline_id: The pipeline ID containing the record.
            source_filename: Original source filename to match.

        Returns:
            EnrichedRecord if found, None otherwise.
        """
        outputs_dir = self.results_dir / pipeline_id / "transcription" / "outputs"
        if not outputs_dir.exists():
            return None
        for output_file in outputs_dir.glob("*.json"):
            try:
                record = EnrichedRecord.model_validate_json(output_file.read_text())
                if record.name == source_filename:
                    return record
            except Exception:
                logger.debug("Failed to load transcription record: %s", output_file, exc_info=True)
        return None

    def load_all_run_configs(self, pipeline_id: str) -> dict[str, ConfigSnapshot]:
        """Load all configuration snapshots for a pipeline run.

        Returns configs from both transcription and CEP steps.

        Args:
            pipeline_id: The pipeline ID to load configs for.

        Returns:
            Dictionary mapping step name to ConfigSnapshot.
        """
        configs: dict[str, ConfigSnapshot] = {}
        pipeline_dir = self.results_dir / pipeline_id
        for step in ("transcription", "cep"):
            run_metadata_file = pipeline_dir / step / "run_metadata.json"
            if run_metadata_file.exists():
                try:
                    metadata = RunMetadata.load(run_metadata_file)
                    configs[step] = metadata.config
                except Exception:
                    logger.debug(
                        "Failed to load run_metadata for %s/%s", pipeline_id, step, exc_info=True
                    )
        return configs
