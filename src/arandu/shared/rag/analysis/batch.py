"""Batch driver for ``arandu rag-analysis`` (spec §8).

Loads every judged :class:`AnswerRecord` under
``results/<id>/judge_answers/outputs/``, groups by retriever arm,
joins with CEP cross-cut metadata, and writes ``report.json`` +
``tables.md`` under ``results/<id>/analysis/outputs/`` via
:class:`ResultsManager`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.shared.config import ResultsConfig
from arandu.shared.rag.analysis.loader import CrossCutMeta, build_cross_cut_map
from arandu.shared.rag.analysis.metrics import ArmMetrics, aggregate_arm
from arandu.shared.rag.analysis.report import AnalysisReport, write_report
from arandu.shared.rag.schemas import AnswerRecord
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisBatchConfig(BaseModel):
    """Run-metadata snapshot for the analysis stage."""

    pipeline_id: str


class AnalysisBatchResult(BaseModel):
    """Summary of a completed analysis run."""

    pipeline_id: str
    run_dir: str
    report_json: str
    tables_md: str
    arms: list[str]
    records_loaded: int
    records_unreadable: int


def run_rag_analysis_batch(
    pipeline_id: str,
    *,
    base_dir: Path | None = None,
) -> AnalysisBatchResult:
    """Aggregate judged answers and write the analysis report.

    Args:
        pipeline_id: Run identifier. The ``judge_answers`` stage must
            be populated under this id; the ``cep`` stage outputs are
            consulted for Bloom + question-type cross-cuts and are
            silently skipped (with a logged warning) when absent.
        base_dir: Override the project ``results/`` root.

    Returns:
        :class:`AnalysisBatchResult` summary.

    Raises:
        FileNotFoundError: ``judge_answers`` outputs are absent.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    judged_dir = base / pipeline_id / "judge_answers" / "outputs"
    if not judged_dir.exists():
        raise FileNotFoundError(
            f"judge_answers outputs not found for pipeline_id {pipeline_id!r}: "
            f"{judged_dir}. Run `arandu judge-answers --id {pipeline_id}` first."
        )

    cross_cut = build_cross_cut_map(cep_dir=base / pipeline_id / "cep" / "outputs")

    results_mgr = ResultsManager(base, PipelineType.ANALYSIS, pipeline_id=pipeline_id)
    results_mgr.create_run(
        AnalysisBatchConfig(pipeline_id=pipeline_id),
        input_source=str(judged_dir),
    )

    by_arm, records_loaded, unreadable = _group_by_arm(judged_dir)
    joint = {
        arm: aggregate_arm(arm, records, slice_name="joint") for arm, records in by_arm.items()
    }
    by_bloom = _build_cross_cut_metrics(by_arm, cross_cut, attr="bloom_level")
    by_question_type = _build_cross_cut_metrics(by_arm, cross_cut, attr="question_type")

    report = AnalysisReport(
        pipeline_id=pipeline_id,
        joint=joint,
        by_bloom=by_bloom,
        by_question_type=by_question_type,
    )
    json_path, tables_path = write_report(results_mgr.outputs_dir, report)

    results_mgr.update_progress(records_loaded, unreadable, records_loaded + unreadable)
    results_mgr.complete_run(success=True)

    return AnalysisBatchResult(
        pipeline_id=pipeline_id,
        run_dir=str(results_mgr.run_dir),
        report_json=str(json_path),
        tables_md=str(tables_path),
        arms=sorted(by_arm),
        records_loaded=records_loaded,
        records_unreadable=unreadable,
    )


def _group_by_arm(judged_dir: Path) -> tuple[dict[str, list[AnswerRecord]], int, int]:
    """Walk ``<judged_dir>/<arm>/<source>/<file>.json`` → ``{arm: [records]}``.

    Returns the groupings plus ``(records_loaded, unreadable)`` counts so the
    batch summary can flag corrupted artifacts without failing the whole
    analysis.
    """
    by_arm: dict[str, list[AnswerRecord]] = defaultdict(list)
    loaded = 0
    unreadable = 0
    for path in sorted(judged_dir.glob("*/*/*.json")):
        try:
            record = AnswerRecord.load(path)
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable judged AnswerRecord %s: %s", path, exc)
            unreadable += 1
            continue
        by_arm[record.retriever_id].append(record)
        loaded += 1
    return dict(by_arm), loaded, unreadable


def _build_cross_cut_metrics(
    by_arm: dict[str, list[AnswerRecord]],
    cross_cut: dict[str, CrossCutMeta],
    *,
    attr: str,
) -> dict[str, dict[str, ArmMetrics]]:
    """Aggregate per-arm metrics keyed by ``attr`` (bloom_level | question_type).

    Records without a cross-cut entry are skipped from the stratified
    tables — they still appear in the joint aggregator because the
    joint slice has no strata dependency.
    """
    nested: dict[str, dict[str, list[AnswerRecord]]] = defaultdict(lambda: defaultdict(list))
    for arm, records in by_arm.items():
        for record in records:
            meta = cross_cut.get(record.qa_pair_id)
            if meta is None:
                continue
            stratum = getattr(meta, attr)
            nested[arm][stratum].append(record)
    return {
        arm: {
            stratum: aggregate_arm(arm, records, slice_name=f"{attr}={stratum}")
            for stratum, records in strata.items()
        }
        for arm, strata in nested.items()
    }
