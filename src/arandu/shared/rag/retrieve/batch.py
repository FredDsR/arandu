"""Batch runner for ``arandu retrieve`` — driver invoked by the CLI.

Iterates ``(arm, question)`` tuples, runs each retrieval, and persists
one :class:`RetrievalRecord` per pair. Checkpoint keys are composite
(``"<arm>::<qa_pair_id>"``) so resume works across arm sets.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.rag.retrieve.factory import build_retriever
from arandu.shared.rag.retrieve.loader import load_questions
from arandu.shared.rag.retrieve.settings import (
    ALL_ARMS,
    ArmName,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)
from arandu.shared.rag.schemas import RetrievalRecord
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.rag.protocol import Retriever
    from arandu.shared.rag.retrieve.loader import QuestionRecord

logger = logging.getLogger(__name__)


CHECKPOINT_FILENAME = "retrieve_checkpoint.json"


class RetrieveBatchConfig(BaseModel):
    """Persisted run-metadata snapshot for the retrieve stage.

    Attributes:
        pipeline_id: The run identifier.
        arms: Ordered list of arm names this run targeted.
        top_k: Passages per question per arm.
        rebuild_index: Whether arm-side indexes were forcibly rebuilt.
    """

    pipeline_id: str
    arms: list[str]
    top_k: int = Field(..., gt=0)
    rebuild_index: bool = False


class RetrieveBatchResult(BaseModel):
    """Summary of a completed retrieve batch run.

    Attributes:
        pipeline_id: Resolved run identifier.
        run_dir: Absolute path of ``results/<pipeline_id>/retrieve/``.
        retrievals_written: Count of ``RetrievalRecord`` JSON files emitted.
        retrievals_resumed: Count of (arm, qa_pair) tuples skipped because
            the checkpoint already marked them completed.
        retrievals_failed: Count of (arm, qa_pair) tuples that raised
            during retrieval. Failures are logged with full context;
            the run continues so a single bad arm doesn't kill the whole
            batch.
    """

    pipeline_id: str
    run_dir: str
    retrievals_written: int = 0
    retrievals_resumed: int = 0
    retrievals_failed: int = 0


def run_retrieve_batch(
    pipeline_id: str,
    arms: list[ArmName],
    top_k: int,
    *,
    bm25_settings: Bm25RetrieveSettings | None = None,
    khop_settings: KHopRetrieveSettings | None = None,
    rebuild_index: bool = False,
    base_dir: Path | None = None,
) -> RetrieveBatchResult:
    """Drive retrieval across one or more arms over a run's CEP + non-answerable items.

    Args:
        pipeline_id: Run identifier. The qa/cep stages must already be
            populated; each arm has its own prerequisites (chunk for
            bm25, kg for k-hop arms).
        arms: Ordered list of arm names to run. Empty list raises
            ``ValueError``.
        top_k: Passages per question per arm. Must be ``>= 1``; arms
            internally handle ``top_k=0`` defensively but the batch
            runner rejects that input upfront because a zero-result
            benchmark is almost certainly a misconfiguration.
        bm25_settings: BM25 arm settings (chunker view). Defaults
            instantiated from env when omitted.
        khop_settings: K-hop arm settings (k_hop, max_postings, keyword).
            Defaults instantiated from env when omitted.
        rebuild_index: Force arm-side index rebuilds where supported
            (BM25 only; k-hop arms build state in-memory).
        base_dir: Override the project ``results/`` root.

    Returns:
        Summary counts via :class:`RetrieveBatchResult`.

    Raises:
        FileNotFoundError: If the CEP outputs (or any arm-specific
            prerequisite) are missing for ``pipeline_id``.
        ValueError: If ``arms`` is empty or contains an unknown name,
            or ``top_k < 1``.
    """
    if not arms:
        raise ValueError("Must request at least one arm; got empty list.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise ValueError(f"Unknown arm(s) {unknown!r}. Valid arms: {list(ALL_ARMS)}.")

    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    cep_dir = base / pipeline_id / "cep" / "outputs"
    nonanswerable_dir = base / pipeline_id / "nonanswerable" / "outputs"

    questions = load_questions(cep_dir, nonanswerable_dir)
    if not questions:
        logger.warning(
            "No questions found for pipeline %s (CEP dir empty). Returning empty result.",
            pipeline_id,
        )

    config = RetrieveBatchConfig(
        pipeline_id=pipeline_id,
        arms=list(arms),
        top_k=top_k,
        rebuild_index=rebuild_index,
    )
    results_mgr = ResultsManager(
        base,
        PipelineType.RETRIEVE,
        pipeline_id=pipeline_id,
    )
    results_mgr.create_run(
        config,
        input_source=str(cep_dir),
        checkpoint_filename=CHECKPOINT_FILENAME,
    )
    checkpoint = CheckpointManager(results_mgr.run_dir / CHECKPOINT_FILENAME)

    total_units = len(questions) * len(arms)
    checkpoint.set_total_files(total_units)

    written = 0
    resumed = 0
    failed = 0
    for arm in arms:
        retriever = _build_with_logging(
            arm=arm,
            pipeline_id=pipeline_id,
            bm25_settings=bm25_settings,
            khop_settings=khop_settings,
            base_dir=base,
            rebuild_index=rebuild_index,
        )
        if retriever is None:
            # Arm-build failure already logged; skip every question for it.
            failed += len(questions)
            continue

        for question in questions:
            ckpt_key = _checkpoint_key(arm, question.qa_pair_id)
            if checkpoint.is_completed(ckpt_key):
                resumed += 1
                continue

            try:
                record = _retrieve_one(
                    retriever=retriever,
                    arm=arm,
                    question=question,
                    top_k=top_k,
                )
            except Exception as exc:
                logger.exception(
                    "Retrieval failed for arm=%s qa_pair_id=%s: %s",
                    arm,
                    question.qa_pair_id,
                    exc,
                )
                checkpoint.mark_failed(ckpt_key, str(exc))
                failed += 1
                continue

            output_path = _output_path(
                outputs_dir=results_mgr.outputs_dir,
                arm=arm,
                source=question.source,
                qa_pair_id=question.qa_pair_id,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            record.save(output_path)
            checkpoint.mark_completed(ckpt_key)
            written += 1

    results_mgr.update_progress(written + resumed, failed, total_units)
    results_mgr.complete_run(success=(failed == 0))

    return RetrieveBatchResult(
        pipeline_id=results_mgr.metadata.pipeline_id,
        run_dir=str(results_mgr.run_dir),
        retrievals_written=written,
        retrievals_resumed=resumed,
        retrievals_failed=failed,
    )


def _build_with_logging(
    *,
    arm: ArmName,
    pipeline_id: str,
    bm25_settings: Bm25RetrieveSettings | None,
    khop_settings: KHopRetrieveSettings | None,
    base_dir: Path,
    rebuild_index: bool,
) -> Retriever | None:
    """Construct a retriever for ``arm``; log + return ``None`` on failure.

    Per-arm build failures (missing index, missing KG outputs, ...) are
    logged with full context. Returning ``None`` lets the batch runner
    continue with the next arm rather than aborting the whole run.
    """
    settings: Bm25RetrieveSettings | KHopRetrieveSettings | None = None
    if arm == "bm25":
        settings = bm25_settings
    elif arm in ("khop_passage", "khop_triple"):
        settings = khop_settings

    try:
        return build_retriever(
            arm,
            pipeline_id=pipeline_id,
            settings=settings,
            base_dir=base_dir,
            rebuild_index=rebuild_index,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to build %s arm for pipeline %s: %s", arm, pipeline_id, exc)
        return None


def _retrieve_one(
    *,
    retriever: Retriever,
    arm: ArmName,
    question: QuestionRecord,
    top_k: int,
) -> RetrievalRecord:
    """Run one retrieval call and wrap the result as a :class:`RetrievalRecord`."""
    t0 = time.perf_counter()
    passages = retriever.retrieve(question.question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return RetrievalRecord(
        qa_pair_id=question.qa_pair_id,
        question=question.question,
        retriever_id=retriever.retriever_id,
        chunker_id=question.chunker_id,
        top_k=top_k,
        passages=passages,
        elapsed_ms=elapsed_ms,
        is_answerable=question.is_answerable,
    )


def _checkpoint_key(arm: ArmName, qa_pair_id: str) -> str:
    """Composite checkpoint key for the (arm, qa_pair) work unit."""
    return f"{arm}::{qa_pair_id}"


def _output_path(*, outputs_dir: Path, arm: ArmName, source: str, qa_pair_id: str) -> Path:
    """Resolve the on-disk path for one retrieval record.

    Layout: ``<outputs_dir>/<arm>/<source>/<sanitized_qa_pair_id>.json``.
    ``qa_pair_id`` carries ``:`` and possibly other path-unfriendly chars
    (``"<file_id>:<chunk_id>:<idx>"``); sanitize for cross-platform safety.
    """
    safe_id = qa_pair_id.replace(":", "__").replace("/", "_")
    return outputs_dir / arm / source / f"{safe_id}.json"
