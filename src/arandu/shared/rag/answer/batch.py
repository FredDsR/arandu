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
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.rag.answer.answerer import AnswererClient
from arandu.shared.rag.answer.packer import pack_passages
from arandu.shared.rag.answer.resolver import build_passage_text_map
from arandu.shared.rag.answer.settings import AnswererSettings
from arandu.shared.rag.schemas import AnswerRecord, RetrievalRecord
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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

    llm_client = _build_llm_client(resolved_settings)
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
    checkpoint = CheckpointManager(results_mgr.run_dir / CHECKPOINT_FILENAME)

    retrieval_paths = sorted(retrieve_outputs.glob("*/*/*.json"))
    checkpoint.set_total_files(len(retrieval_paths))

    written = 0
    resumed = 0
    failed = 0
    for record_path in retrieval_paths:
        # Path shape: <outputs>/<arm>/<source>/<safe_qa_pair_id>.json.
        # parts[-3:] = (arm, source, file). Use them as the checkpoint
        # key so resume is cross-arm safe.
        ckpt_key = _checkpoint_key(record_path)
        if checkpoint.is_completed(ckpt_key):
            resumed += 1
            continue

        try:
            retrieval = RetrievalRecord.load(record_path)
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable RetrievalRecord %s: %s", record_path, exc)
            checkpoint.mark_failed(ckpt_key, f"load failed: {exc}")
            failed += 1
            continue

        try:
            answer_record = _answer_one(
                answerer=answerer,
                retrieval=retrieval,
                passage_text=passage_text,
                resolved_settings=resolved_settings,
            )
        except Exception as exc:
            # Per-record isolation: log + continue. Mirrors the
            # retrieve batch runner's rationale (Protocol abstraction
            # over the LLMClient internals).
            logger.exception(
                "Answer failed for arm=%s qa_pair_id=%s: %s",
                _arm_from_path(record_path),
                retrieval.qa_pair_id,
                exc,
            )
            checkpoint.mark_failed(ckpt_key, str(exc))
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

    results_mgr.update_progress(written + resumed, failed, len(retrieval_paths))
    results_mgr.complete_run(success=(failed == 0))

    return AnswerBatchResult(
        pipeline_id=results_mgr.metadata.pipeline_id,
        run_dir=str(results_mgr.run_dir),
        answers_written=written,
        answers_resumed=resumed,
        answers_failed=failed,
    )


def _build_llm_client(settings: AnswererSettings) -> LLMClient:
    """Construct the unified LLMClient from answerer settings.

    Surfaces a clear ``RuntimeError`` if a cloud provider is requested
    without its API key env var set; ollama gets a free pass per
    :class:`LLMClient` convention.
    """
    try:
        provider_enum = LLMProvider(settings.provider)
    except ValueError as exc:
        raise ValueError(
            f"Unknown answerer provider {settings.provider!r}. "
            f"Valid: {[p.value for p in LLMProvider]}."
        ) from exc

    api_key = os.environ.get(settings.api_key_env)
    if not api_key and provider_enum is not LLMProvider.OLLAMA:
        raise RuntimeError(
            f"Answerer provider {settings.provider!r} requires {settings.api_key_env} "
            f"to be set. Either set it, or switch ARANDU_ANSWERER_PROVIDER to 'ollama'."
        )

    return LLMClient(
        provider=provider_enum,
        model_id=settings.model_id,
        api_key=api_key,
        base_url=settings.base_url,
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
