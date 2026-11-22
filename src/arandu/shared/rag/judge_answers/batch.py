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
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.config import ResultsConfig
from arandu.shared.llm_client import build_llm_client_from_settings, is_rate_limit_error
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
from arandu.utils.concurrency import map_concurrent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class _MissingGoldError(Exception):
    """Worker-side sentinel: an answerable record has no CEP gold to judge against."""


# Batched checkpoint persistence: full-file rewrites per record are
# O(n^2) total I/O over a large batch; an interval keeps I/O flat at
# the cost of re-processing up to N-1 records after a crash (idempotent).
_CHECKPOINT_SAVE_INTERVAL = 20

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
    """Run the gated answer-judge pipeline over every ``AnswerRecord`` for ``pipeline_id``.

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

    llm_client = build_llm_client_from_settings(resolved_settings)
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
    checkpoint = CheckpointManager(checkpoint_path, save_interval=_CHECKPOINT_SAVE_INTERVAL)

    answer_paths = sorted(answers_outputs.glob("*/*/*.json"))
    checkpoint.set_total_files(len(answer_paths))

    written = 0
    failed = 0

    # Resume filtering happens up front so the worker pool only ever
    # sees real work.
    pending = [p for p in answer_paths if not checkpoint.is_completed(_checkpoint_key(p))]
    resumed = len(answer_paths) - len(pending)

    def _process(record_path: Path) -> AnswerRecord:
        """Load, gold-check, and judge one record on a worker thread."""
        answer = AnswerRecord.load(record_path)
        # Answerable items need their CEP gold for the gold-scoring stage;
        # an orphan answerable qa_pair_id can't be judged. Non-answerable
        # items (from `arandu generate-non-answerable`) have no gold and
        # are judged on abstention alone — the commitment gate skips the
        # gold criteria for them, so a missing gold is expected.
        gold = gold_lookup.get(answer.qa_pair_id)
        if answer.is_answerable and gold is None:
            raise _MissingGoldError(answer.qa_pair_id)
        return _judge_one(judge=judge, answer=answer, gold=gold, passage_text=passage_text)

    # Workers run only `_process`; checkpoint writes and file saves stay
    # on this (main) thread, so no locking is needed. A rate-limited
    # record (exhausted client-side 429 retries) is requeued by the
    # helper after an adaptive slowdown instead of being failed.
    for record_path, judged, error in map_concurrent(
        _process,
        pending,
        workers=resolved_settings.workers,
        rate_limit_of=is_rate_limit_error,
    ):
        ckpt_key = _checkpoint_key(record_path)
        if error is not None:
            if isinstance(error, _MissingGoldError):
                logger.warning(
                    "No gold lookup for answerable qa_pair_id=%s — skipping judge for %s.",
                    error,
                    record_path,
                )
                checkpoint.mark_failed(ckpt_key, "no gold lookup")
            elif isinstance(error, (OSError, ValidationError)):
                logger.warning("Skipping unreadable AnswerRecord %s: %s", record_path, error)
                checkpoint.mark_failed(ckpt_key, f"load failed: {error}")
            else:
                # Per-record isolation: log + continue per spec §6.6's
                # "all in score mode" contract (a bad record doesn't kill
                # the batch).
                logger.error(
                    "Judge failed for arm=%s source=%s record=%s: %s",
                    record_path.parts[-3],
                    record_path.parts[-2],
                    record_path.stem,
                    error,
                    exc_info=error,
                )
                checkpoint.mark_failed(ckpt_key, str(error))
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

    checkpoint.flush()
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


def _judge_one(
    *,
    judge: AnswerJudge,
    answer: AnswerRecord,
    gold: GoldRecord | None,
    passage_text: dict[str, str],
) -> AnswerRecord:
    """Run the gated judge pipeline; attach the verdict to the record.

    Every kwarg below is consumed by at least one stage. Criteria that
    don't reference a given kwarg silently ignore it (LLM prompts via
    ``string.Template.safe_substitute``; heuristics via ``kwargs.get``):

    - gates: ``is_answerable`` (answerability) + ``abstained`` (commitment);
    - ``abstention``: ``abstained`` / ``answer_text`` / ``rationale``;
    - ``passage_coverage`` / ``answer_correctness`` / ``answer_faithfulness``:
      ``question`` / ``gold_answer`` / ``system_answer`` / ``passages_text``;
    - ``source_recovery`` (heuristic): ``retrieved_text`` (raw joined
      passage text, no ``[Passage N]`` markers) / ``context`` /
      ``passages_are_non_prose``.

    ``gold`` is ``None`` for non-answerable items (no CEP pair); the
    answerability gate rejects those before the gold criteria run, so the
    empty gold fields are never actually consumed.
    """
    resolved_passages = _resolve_passage_texts(answer, passage_text)
    pipeline_result = judge.evaluate(
        is_answerable=answer.is_answerable,
        abstained=str(answer.abstained).lower(),
        answer_text=answer.answer_text or "",
        system_answer=answer.answer_text or "",
        rationale=answer.rationale,
        passages_text=_format_passages(resolved_passages),
        # Raw joined passage text + non-prose flag for the deterministic
        # source_recovery criterion (separate from the LLM-facing
        # passages_text, which carries "[Passage N]" markers). Non-prose =
        # payload that isn't verbatim source text (triples), where token
        # containment is meaningless; inline-prose payloads (khop_passage,
        # payload_is_prose=True) and offset-resolved passages still score.
        retrieved_text="\n".join(resolved_passages),
        passages_are_non_prose=any(
            p.payload is not None and not p.payload_is_prose for p in answer.passages
        ),
        question=gold.question if gold is not None else answer.question,
        gold_answer=gold.gold_answer if gold is not None else "",
        context=gold.context if gold is not None else "",
    )
    return answer.model_copy(update={"validation": pipeline_result})


def _resolve_passage_texts(answer: AnswerRecord, passage_text: dict[str, str]) -> list[str]:
    """Resolve each passage to its raw text (payload or chunk lookup).

    Honours :attr:`RetrievedPassage.payload` for triple-arm passages
    (those skip the chunk_id lookup). Unresolvable / empty passages are
    dropped, so the caller sees only passages that have text grounding.
    """
    texts: list[str] = []
    for passage in answer.passages:
        text = (
            passage.payload
            if passage.payload is not None
            else passage_text.get(passage.chunk_id, "")
        )
        if text:
            texts.append(text)
    return texts


def _format_passages(resolved_passages: list[str]) -> str:
    """Format already-resolved passage texts as enumerated LLM-prompt text.

    Takes the output of :func:`_resolve_passage_texts` so the caller
    resolves once and reuses it (the raw join also feeds ``source_recovery``).
    """
    parts = [f"[Passage {idx}]:\n{text}\n" for idx, text in enumerate(resolved_passages, start=1)]
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
