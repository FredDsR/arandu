"""Batch runner for ``arandu answer`` — drives the Answerer over a populated retrieve stage.

Iterates ``RetrievalRecord`` artifacts produced by ``arandu retrieve``
under ``results/<id>/retrieve/outputs/<arm>/<source>/`` and writes
``AnswerRecord`` artifacts under
``results/<id>/answers/outputs/<arm>/<source>/``.

Same retriever-arm-as-shard layout, same per-arm failure isolation,
same composite checkpoint keys as the retrieve stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.llm_client import build_llm_client_from_settings, is_rate_limit_error
from arandu.shared.rag.answer.answerer import AnswererClient
from arandu.shared.rag.answer.packer import pack_passages
from arandu.shared.rag.answer.resolver import build_passage_text_map
from arandu.shared.rag.answer.settings import AnswererSettings
from arandu.shared.rag.schemas import AnswerRecord, RetrievalRecord
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType
from arandu.utils.concurrency import map_concurrent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Batched checkpoint persistence: full-file rewrites per record are
# O(n^2) total I/O over a large batch; an interval keeps I/O flat at
# the cost of re-processing up to N-1 records after a crash (idempotent).
_CHECKPOINT_SAVE_INTERVAL = 20

CHECKPOINT_FILENAME = "answer_checkpoint.json"


class AnswerBatchConfig(BaseModel):
    """Persisted run-metadata snapshot for the answers stage.

    Attributes:
        pipeline_id: The run identifier.
        answerer_provider: LLM provider used (e.g. ``"ollama"``).
        answerer_model_id: LLM model identifier.
        answerer_temperature: Starting temperature.
        language: Prompt language (``"pt"`` or ``"en"``).
    """

    pipeline_id: str
    answerer_provider: str
    answerer_model_id: str
    answerer_temperature: float = Field(..., ge=0.0, le=2.0)
    language: str


class AnswerBatchResult(BaseModel):
    """Summary of a completed answer batch run."""

    pipeline_id: str
    run_dir: str
    answers_written: int = 0
    answers_resumed: int = 0
    answers_failed: int = 0


def run_answer_batch(
    pipeline_id: str,
    *,
    settings: AnswererSettings | None = None,
    base_dir: Path | None = None,
) -> AnswerBatchResult:
    """Drive the answerer over every ``RetrievalRecord`` for ``pipeline_id``.

    Args:
        pipeline_id: Run identifier. The retrieve stage must be populated.
        settings: Answerer configuration. Defaults to
            :class:`AnswererSettings()` (reads ``ARANDU_ANSWERER_*``
            env vars).
        base_dir: Override the project ``results/`` root.

    Returns:
        Summary counts via :class:`AnswerBatchResult`.

    Raises:
        FileNotFoundError: If the retrieve stage's outputs aren't
            present for ``pipeline_id``.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved_settings = settings if settings is not None else AnswererSettings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    retrieve_outputs = base / pipeline_id / "retrieve" / "outputs"
    if not retrieve_outputs.exists():
        raise FileNotFoundError(
            f"Retrieve outputs not found for pipeline_id {pipeline_id!r}: "
            f"{retrieve_outputs}. Run `arandu retrieve --id {pipeline_id}` first."
        )

    llm_client = build_llm_client_from_settings(resolved_settings)
    answerer = AnswererClient(llm_client=llm_client, settings=resolved_settings)
    passage_text = _build_passage_text_map(base=base, pipeline_id=pipeline_id)

    config = AnswerBatchConfig(
        pipeline_id=pipeline_id,
        answerer_provider=resolved_settings.provider,
        answerer_model_id=resolved_settings.model_id,
        answerer_temperature=resolved_settings.temperature,
        language=resolved_settings.language,
    )
    results_mgr = ResultsManager(
        base,
        PipelineType.ANSWERS,
        pipeline_id=pipeline_id,
    )
    results_mgr.create_run(
        config,
        input_source=str(retrieve_outputs),
        checkpoint_filename=CHECKPOINT_FILENAME,
    )
    checkpoint = CheckpointManager(
        results_mgr.run_dir / CHECKPOINT_FILENAME, save_interval=_CHECKPOINT_SAVE_INTERVAL
    )

    retrieval_paths = sorted(retrieve_outputs.glob("*/*/*.json"))
    checkpoint.set_total_files(len(retrieval_paths))

    written = 0
    failed = 0

    # Path shape: <outputs>/<arm>/<source>/<safe_qa_pair_id>.json.
    # parts[-3:] = (arm, source, file). Used as the checkpoint key so
    # resume is cross-arm safe. Resume filtering happens up front so the
    # worker pool only ever sees real work.
    pending = [p for p in retrieval_paths if not checkpoint.is_completed(_checkpoint_key(p))]
    resumed = len(retrieval_paths) - len(pending)

    def _process(record_path: Path) -> AnswerRecord:
        """Load + answer one record on a worker thread (no shared state)."""
        retrieval = RetrievalRecord.load(record_path)
        return _answer_one(
            answerer=answerer,
            retrieval=retrieval,
            passage_text=passage_text,
            resolved_settings=resolved_settings,
        )

    # Workers run only `_process`; checkpoint writes and file saves stay
    # on this (main) thread, so no locking is needed. A rate-limited
    # record (exhausted client-side 429 retries) is requeued by the
    # helper after an adaptive slowdown instead of being failed.
    for record_path, answer_record, error in map_concurrent(
        _process,
        pending,
        workers=resolved_settings.workers,
        rate_limit_of=is_rate_limit_error,
    ):
        ckpt_key = _checkpoint_key(record_path)
        if error is not None:
            if isinstance(error, (OSError, ValidationError)):
                logger.warning("Skipping unreadable RetrievalRecord %s: %s", record_path, error)
                checkpoint.mark_failed(ckpt_key, f"load failed: {error}")
            else:
                # Per-record isolation: log + continue. Mirrors the
                # retrieve batch runner's rationale (Protocol abstraction
                # over the LLMClient internals).
                logger.error(
                    "Answer failed for arm=%s source=%s record=%s: %s",
                    _arm_from_path(record_path),
                    record_path.parts[-2],
                    record_path.stem,
                    error,
                    exc_info=error,
                )
                checkpoint.mark_failed(ckpt_key, str(error))
            failed += 1
            continue

        output_path = _output_path(
            outputs_dir=results_mgr.outputs_dir,
            record_path=record_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        answer_record.save(output_path)
        checkpoint.mark_completed(ckpt_key)
        written += 1

    checkpoint.flush()
    results_mgr.update_progress(written + resumed, failed, len(retrieval_paths))
    results_mgr.complete_run(success=(failed == 0))

    return AnswerBatchResult(
        pipeline_id=results_mgr.metadata.pipeline_id,
        run_dir=str(results_mgr.run_dir),
        answers_written=written,
        answers_resumed=resumed,
        answers_failed=failed,
    )


def _build_passage_text_map(*, base: Path, pipeline_id: str) -> dict[str, str]:
    """Aggregate every resolvable ``chunk_id → text`` for the run.

    Walks each chunker view dir under ``chunk/outputs/`` (BM25 arms)
    plus the ``passage_offsets.json`` sidecar (atlas-rag / khop_passage
    arms). triple-arm payloads bypass this map entirely.
    """
    chunk_outputs_root = base / pipeline_id / "chunk" / "outputs"
    chunk_dirs: list[Path] = []
    if chunk_outputs_root.exists():
        chunk_dirs = [p for p in chunk_outputs_root.iterdir() if p.is_dir()]

    passage_offsets_path = base / pipeline_id / "kg" / "outputs" / "passage_offsets.json"
    transcription_dir = base / pipeline_id / "transcription" / "outputs"

    return build_passage_text_map(
        chunk_dirs=chunk_dirs,
        passage_offsets_path=passage_offsets_path,
        transcription_dir=transcription_dir,
    )


def _answer_one(
    *,
    answerer: AnswererClient,
    retrieval: RetrievalRecord,
    passage_text: dict[str, str],
    resolved_settings: AnswererSettings,
) -> AnswerRecord:
    """Pack passages, run the answerer, and assemble the persistable record.

    Caps ``retrieval.passages`` at ``settings.top_k`` BEFORE the token-budget
    pack so the answerer is constrained to the same per-question fan-out
    regardless of how many passages the retriever returned (the methodology
    constant from spec §5.7's ``ARANDU_ANSWERER_TOP_K`` knob). The budget
    pack then trims further if even ``top_k`` passages exceed the context.
    """
    capped_passages = retrieval.passages[: resolved_settings.top_k]
    packed = pack_passages(
        capped_passages,
        passage_text=passage_text,
        max_context_tokens=resolved_settings.max_context_tokens,
        prompt_overhead_tokens=resolved_settings.prompt_overhead_tokens,
        max_answer_tokens=resolved_settings.max_tokens,
    )
    output, meta = answerer.answer(
        question=retrieval.question,
        passage_texts=[text for _, text in packed],
    )

    answer_record = AnswerRecord(
        # RetrievalRecord fields (re-emit so the joined record is self-contained):
        qa_pair_id=retrieval.qa_pair_id,
        question=retrieval.question,
        retriever_id=retrieval.retriever_id,
        chunker_id=retrieval.chunker_id,
        top_k=retrieval.top_k,
        passages=retrieval.passages,
        elapsed_ms=retrieval.elapsed_ms,
        is_answerable=retrieval.is_answerable,
        retriever_meta=retrieval.retriever_meta,
        # AnswerRecord-specific fields:
        answer_text=output.answer,
        abstained=output.abstained,
        rationale=output.rationale,
        answerer_model=resolved_settings.model_id,
        answerer_temperature=resolved_settings.temperature,
        answerer_meta={
            # LLM retry audit (attempts, final_temperature, fallback_reason if any).
            **meta,
            # Pack diagnostics — how many passages survived the top_k + token budget.
            "passages_after_top_k": len(capped_passages),
            "packed_passages": len(packed),
            # Settings snapshot — persisted on each record so reruns can
            # attribute outputs back to configuration even if run_metadata.json
            # is moved, lost, or carries an updated snapshot from a later run.
            "language": resolved_settings.language,
            "provider": resolved_settings.provider,
            "top_k": resolved_settings.top_k,
            "max_tokens": resolved_settings.max_tokens,
            "max_context_tokens": resolved_settings.max_context_tokens,
            "prompt_overhead_tokens": resolved_settings.prompt_overhead_tokens,
        },
    )
    return answer_record


def _checkpoint_key(record_path: Path) -> str:
    """Composite key: ``<arm>::<source>::<file_stem>``."""
    parts = record_path.parts
    return f"{parts[-3]}::{parts[-2]}::{record_path.stem}"


def _arm_from_path(record_path: Path) -> str:
    """Extract the arm name from a retrieve-stage record path."""
    return record_path.parts[-3]


def _output_path(*, outputs_dir: Path, record_path: Path) -> Path:
    """Mirror the input layout: ``<outputs>/<arm>/<source>/<file>.json``."""
    return outputs_dir / record_path.parts[-3] / record_path.parts[-2] / record_path.name
