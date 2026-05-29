"""Batch runner for ``arandu judge-answers`` — driver invoked by the CLI.

Iterates ``AnswerRecord`` artifacts produced by ``arandu answer`` under
``results/<id>/answers/outputs/<arm>/<source>/`` and writes judged
copies under ``results/<id>/judge_answers/outputs/<arm>/<source>/`` —
each carrying the answer record's full content plus a populated
``validation`` field with per-criterion scores.

Emits an ``abstention_audit.jsonl`` file alongside the outputs when
the answerer's structured ``abstained`` flag disagrees with the
abstention judge's verdict (spec §6.4).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.rag.answer.resolver import build_passage_text_map
from arandu.shared.rag.judge_answers.audit import (
    AbstentionDisagreement,
    detect_disagreement,
    write_audit_log,
)
from arandu.shared.rag.judge_answers.gold_lookup import GoldRecord, build_gold_lookup
from arandu.shared.rag.judge_answers.judge import AnswerJudge
from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings
from arandu.shared.rag.schemas import AnswerRecord
from arandu.shared.results_manager import ResultsManager
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


CHECKPOINT_FILENAME = "judge_answers_checkpoint.json"


class JudgeAnswersBatchConfig(BaseModel):
    """Run-metadata snapshot for the judge_answers stage."""

    pipeline_id: str
    judge_provider: str
    judge_model_id: str
    judge_language: str
    judge_temperature: float


class JudgeAnswersBatchResult(BaseModel):
    """Summary of a completed judge_answers batch run."""

    pipeline_id: str
    run_dir: str
    judgments_written: int = 0
    judgments_resumed: int = 0
    judgments_failed: int = 0
    abstention_disagreements: int = 0


def run_judge_answers_batch(
    pipeline_id: str,
    *,
    settings: JudgeAnswersSettings | None = None,
    base_dir: Path | None = None,
    rejudge: bool = False,
) -> JudgeAnswersBatchResult:
    """Run the 4-criterion LLM judge over every ``AnswerRecord`` for ``pipeline_id``.

    Args:
        pipeline_id: Run identifier. The answers stage must be populated.
        settings: Judge configuration. Defaults to
            :class:`JudgeAnswersSettings()` (reads
            ``ARANDU_JUDGE_ANSWERS_*`` env vars).
        base_dir: Override the project ``results/`` root.
        rejudge: If True, clear the checkpoint before running so every
            record is re-judged. If False (default), records already
            marked completed in the checkpoint are skipped.

    Returns:
        :class:`JudgeAnswersBatchResult` summary.

    Raises:
        FileNotFoundError: If the answers stage outputs aren't present.
        RuntimeError: If a cloud-provider API key env var is unset.
    """
    resolved_settings = settings if settings is not None else JudgeAnswersSettings()
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    answers_outputs = base / pipeline_id / "answers" / "outputs"
    if not answers_outputs.exists():
        raise FileNotFoundError(
            f"Answers outputs not found for pipeline_id {pipeline_id!r}: "
            f"{answers_outputs}. Run `arandu answer --id {pipeline_id}` first."
        )

    llm_client = _build_llm_client(resolved_settings)
    judge = AnswerJudge(llm_client=llm_client, settings=resolved_settings)

    gold_lookup = build_gold_lookup(cep_dir=base / pipeline_id / "cep" / "outputs")
    passage_text = build_passage_text_map(
        chunk_dirs=_chunk_view_dirs(base / pipeline_id / "chunk" / "outputs"),
        passage_offsets_path=base / pipeline_id / "kg" / "outputs" / "passage_offsets.json",
        transcription_dir=base / pipeline_id / "transcription" / "outputs",
    )

    config = JudgeAnswersBatchConfig(
        pipeline_id=pipeline_id,
        judge_provider=resolved_settings.provider,
        judge_model_id=resolved_settings.model_id,
        judge_language=resolved_settings.language,
        judge_temperature=resolved_settings.temperature,
    )
    results_mgr = ResultsManager(
        base,
        PipelineType.JUDGE_ANSWERS,
        pipeline_id=pipeline_id,
    )
    results_mgr.create_run(
        config,
        input_source=str(answers_outputs),
        checkpoint_filename=CHECKPOINT_FILENAME,
    )
    checkpoint_path = results_mgr.run_dir / CHECKPOINT_FILENAME
    if rejudge and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = CheckpointManager(checkpoint_path)

    answer_paths = sorted(answers_outputs.glob("*/*/*.json"))
    checkpoint.set_total_files(len(answer_paths))

    written = 0
    resumed = 0
    failed = 0
    for record_path in answer_paths:
        ckpt_key = _checkpoint_key(record_path)
        if checkpoint.is_completed(ckpt_key):
            resumed += 1
            continue

        try:
            answer = AnswerRecord.load(record_path)
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable AnswerRecord %s: %s", record_path, exc)
            checkpoint.mark_failed(ckpt_key, f"load failed: {exc}")
            failed += 1
            continue

        # Answerable items need their CEP gold for the gold-scoring stage;
        # an orphan answerable qa_pair_id can't be judged. Non-answerable
        # items (from `arandu generate-non-answerable`) have no gold and
        # are judged on abstention alone — the commitment gate skips the
        # gold criteria for them, so a missing gold is expected.
        gold = gold_lookup.get(answer.qa_pair_id)
        if answer.is_answerable and gold is None:
            logger.warning(
                "No gold lookup for answerable qa_pair_id=%s — skipping judge for %s.",
                answer.qa_pair_id,
                record_path,
            )
            checkpoint.mark_failed(ckpt_key, "no gold lookup")
            failed += 1
            continue

        try:
            judged = _judge_one(
                judge=judge,
                answer=answer,
                gold=gold,
                passage_text=passage_text,
            )
        except Exception as exc:
            # Per-record isolation: log + continue per spec §6.6's
            # "all in score mode" contract (a bad record doesn't kill
            # the batch).
            logger.exception(
                "Judge failed for qa_pair_id=%s arm=%s: %s",
                answer.qa_pair_id,
                answer.retriever_id,
                exc,
            )
            checkpoint.mark_failed(ckpt_key, str(exc))
            failed += 1
            continue

        output_path = _output_path(outputs_dir=results_mgr.outputs_dir, record_path=record_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        judged.save(output_path)
        checkpoint.mark_completed(ckpt_key)
        written += 1

    # Build the audit from EVERY judged record on disk, not just this
    # invocation's newly-judged ones. On a resume run, the loop above
    # skips already-completed records but their disagreements still
    # belong in the audit; deriving from disk guarantees the audit is
    # always cumulative + consistent with what's actually persisted.
    # `detect_disagreement` reads τ_abstention from each criterion's
    # own CriterionScore.threshold — single source of truth.
    disagreements_count = 0
    if resolved_settings.abstention_disagreement_audit:
        disagreements = _collect_all_disagreements(results_mgr.outputs_dir)
        write_audit_log(results_mgr.outputs_dir, disagreements)
        disagreements_count = len(disagreements)

    results_mgr.update_progress(written + resumed, failed, len(answer_paths))
    results_mgr.complete_run(success=(failed == 0))

    return JudgeAnswersBatchResult(
        pipeline_id=results_mgr.metadata.pipeline_id,
        run_dir=str(results_mgr.run_dir),
        judgments_written=written,
        judgments_resumed=resumed,
        judgments_failed=failed,
        abstention_disagreements=disagreements_count,
    )


def _collect_all_disagreements(outputs_dir: Path) -> list[AbstentionDisagreement]:
    """Walk every judged record under ``outputs_dir`` and collect disagreements.

    Layout: ``<outputs_dir>/<arm>/<source>/<file>.json``. Unreadable
    records are skipped (logged at warning level); their absence from
    the audit list is acceptable because the failure was already
    recorded on the run's checkpoint when they failed to judge.
    """
    out: list[AbstentionDisagreement] = []
    for path in sorted(outputs_dir.glob("*/*/*.json")):
        try:
            judged = AnswerRecord.load(path)
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable judged AnswerRecord %s: %s", path, exc)
            continue
        disagreement = detect_disagreement(judged)
        if disagreement is not None:
            out.append(disagreement)
    return out


def _build_llm_client(settings: JudgeAnswersSettings) -> LLMClient:
    """Construct the unified LLMClient from judge settings.

    Mirrors the cloud-API-key + custom-base-url guards used in the
    answerer's :func:`_build_llm_client` so the failure surfaces stay
    consistent across stages.
    """
    try:
        provider_enum = LLMProvider(settings.provider)
    except ValueError as exc:
        raise ValueError(
            f"Unknown judge_answers provider {settings.provider!r}. "
            f"Valid: {[p.value for p in LLMProvider]}."
        ) from exc

    api_key = os.environ.get(settings.api_key_env)
    if not api_key and provider_enum is not LLMProvider.OLLAMA:
        raise RuntimeError(
            f"judge_answers provider {settings.provider!r} requires "
            f"{settings.api_key_env} to be set. Either set it, or switch "
            f"ARANDU_JUDGE_ANSWERS_PROVIDER to 'ollama'."
        )
    if provider_enum is LLMProvider.CUSTOM and not settings.base_url:
        raise ValueError(
            "provider='custom' requires a base URL. Set ARANDU_JUDGE_ANSWERS_BASE_URL "
            "or pass base_url=... explicitly."
        )

    return LLMClient(
        provider=provider_enum,
        model_id=settings.model_id,
        api_key=api_key,
        base_url=settings.base_url,
    )


def _judge_one(
    *,
    judge: AnswerJudge,
    answer: AnswerRecord,
    gold: GoldRecord | None,
    passage_text: dict[str, str],
) -> AnswerRecord:
    """Run the gated judge pipeline; attach the verdict to the record.

    Every kwarg below is consumed by at least one stage. The gates read
    ``is_answerable`` + ``abstained``; the abstention criterion reads
    ``abstained`` / ``answer_text`` / ``rationale``; the gold-scoring
    criteria read ``question`` / ``gold_answer`` / ``passages_text``.
    Criteria that don't reference a given kwarg silently ignore it (LLM
    prompts via ``string.Template.safe_substitute``).

    ``gold`` is ``None`` for non-answerable items (no CEP pair); the
    answerability gate rejects those before the gold criteria run, so the
    empty gold fields are never actually consumed.
    """
    passages_text = _format_passages(answer, passage_text)
    pipeline_result = judge.evaluate(
        is_answerable=answer.is_answerable,
        abstained=str(answer.abstained).lower(),
        answer_text=answer.answer_text or "",
        system_answer=answer.answer_text or "",
        rationale=answer.rationale,
        passages_text=passages_text,
        question=gold.question if gold is not None else answer.question,
        gold_answer=gold.gold_answer if gold is not None else "",
    )
    return answer.model_copy(update={"validation": pipeline_result})


def _format_passages(answer: AnswerRecord, passage_text: dict[str, str]) -> str:
    """Format retrieved passages as enumerated text for the LLM prompts.

    Honours :attr:`RetrievedPassage.payload` for triple-arm passages
    (those skip the chunk_id lookup). Unresolvable passages are
    silently dropped — the judge sees only the passages that have
    text grounding.
    """
    parts: list[str] = []
    for idx, passage in enumerate(answer.passages, start=1):
        if passage.payload is not None:
            text = passage.payload
        else:
            text = passage_text.get(passage.chunk_id, "")
        if not text:
            continue
        parts.append(f"[Passage {idx}]:\n{text}\n")
    return "\n".join(parts) if parts else "(no passages available)"


def _chunk_view_dirs(chunk_outputs_root: Path) -> list[Path]:
    """List per-chunker-view subdirectories (or empty when chunks/ is absent)."""
    if not chunk_outputs_root.exists():
        return []
    return [p for p in chunk_outputs_root.iterdir() if p.is_dir()]


def _checkpoint_key(record_path: Path) -> str:
    """Composite key: ``<arm>::<source>::<file_stem>``."""
    parts = record_path.parts
    return f"{parts[-3]}::{parts[-2]}::{record_path.stem}"


def _output_path(*, outputs_dir: Path, record_path: Path) -> Path:
    """Mirror the input layout: ``<outputs>/<arm>/<source>/<file>.json``."""
    return outputs_dir / record_path.parts[-3] / record_path.parts[-2] / record_path.name
