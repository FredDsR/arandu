"""Run-metadata snapshot for ``arandu judge-qa``.

``judge-qa`` persists its verdicts *inside* the CEP records
(``QAPairCEP.validation``), so unlike the other stages it had no step
directory and no ``run_metadata.json``: nothing recorded the temperature,
thresholds or prompts a verdict was produced with. This module supplies the
snapshot, written to ``results/<id>/judge_qa/run_metadata.json`` whenever the
input directory follows the ``results/<id>/cep/outputs`` layout.

The snapshot is what the intra-judge reliability study pins its
pre-registration to (docs/planning/2026-09-28-intrajudge-reliability-design.md):
two gate runs are replicas only if their snapshots agree on model, temperature,
thresholds and prompt digests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from arandu.qa.cep.judge import GATE_CRITERIA
from arandu.qa.config import JudgeConfig  # noqa: TC001 (pydantic needs runtime access)
from arandu.shared.judge.factory import DEFAULT_JUDGE_PROMPTS_DIR
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.judge.schemas import JudgePipelineResult

# Bloom level descriptions fed to bloom_calibration (mirrors qa.cep.judge).
BLOOM_DESCRIPTIONS_DIR = get_project_root() / "prompts" / "qa" / "cep" / "validation"


class CriterionPromptSnapshot(BaseModel):
    """Threshold and prompt digest of one gate criterion.

    Attributes:
        threshold: Pass threshold read from the criterion's ``config.json``
            (``None`` when absent).
        temperature: Effective sampling temperature of the criterion: the
            ``temperature`` override in its ``config.json`` when present,
            otherwise the judge-wide temperature.
        sha256: Digest over the criterion's ``config.json`` and its prompt
            for the run language, so a prompt edit changes the digest.
            ``None`` when either file is missing, so two runs without prompts
            never compare as identical.
    """

    threshold: float | None
    temperature: float
    sha256: str | None


class JudgeQARunConfig(BaseModel):
    """Config snapshotted into ``run_metadata.json`` for a judge-qa run.

    Attributes:
        mode: ``rejudge`` re-scores every sampled pair; ``resume`` skips pairs
            that already carry a verdict.
        files: ``--files`` sampling cap (``None`` = all files).
        pairs: ``--pairs`` per-file sampling cap (``None`` = all pairs).
        provider: Resolved validator provider.
        model_id: Resolved validator model.
        base_url: Resolved validator endpoint.
        language: Language of the criterion prompts.
        judge: Resolved judge settings (temperature, max_tokens, ...).
        criteria: Threshold and prompt digest per gate criterion.
        bloom_descriptions_sha256: Digest of the Bloom level descriptions the
            ``bloom_calibration`` criterion receives.
    """

    mode: Literal["rejudge", "resume"]
    files: int | None = None
    pairs: int | None = None
    provider: str | None = None
    model_id: str
    base_url: str | None = None
    language: str
    judge: JudgeConfig
    criteria: dict[str, CriterionPromptSnapshot] = Field(default_factory=dict)
    bloom_descriptions_sha256: str | None = None


def resolve_pipeline_layout(input_dir: Path) -> tuple[Path, str] | None:
    """Return ``(results_base, pipeline_id)`` for a ``<base>/<id>/cep/outputs`` dir.

    Args:
        input_dir: Directory passed to ``judge-qa``.

    Returns:
        The results base directory and pipeline id, or ``None`` when the
        directory does not follow the results layout (an ad hoc dataset), in
        which case there is no step directory to write metadata into. The
        layout is recognised by the pipeline's ``pipeline.json``, not by the
        path shape alone: ``~/datasets/cep/outputs`` must not create
        ``~/datasets/judge_qa/`` and a stray ``~/index.json``.
    """
    resolved = input_dir.resolve()
    if resolved.name != "outputs" or resolved.parent.name != "cep":
        return None
    pipeline_dir = resolved.parent.parent
    if not (pipeline_dir / "pipeline.json").is_file():
        return None
    return pipeline_dir.parent, pipeline_dir.name


def archive_previous_snapshot(step_dir: Path) -> Path | None:
    """Move an existing ``run_metadata.json`` into ``history/`` before a new run.

    ``create_run`` rewrites the snapshot from scratch, so a resume would erase
    the record of the run that produced the verdicts it skips. Archiving keeps
    an append-only history: every run that wrote verdicts leaves its snapshot.

    Args:
        step_dir: ``results/<id>/judge_qa``.

    Returns:
        Path of the archived snapshot, or ``None`` when there was none.
    """
    current = step_dir / "run_metadata.json"
    if not current.is_file():
        return None
    try:
        started = json.loads(current.read_text(encoding="utf-8")).get("started_at")
    except (OSError, ValueError):
        started = None
    stamp = started or datetime.fromtimestamp(current.stat().st_mtime).isoformat()
    stamp = stamp.replace(":", "").replace("+", "_")
    history = step_dir / "history"
    history.mkdir(exist_ok=True)
    target = history / f"run_metadata.{stamp}.json"
    suffix = 1
    while target.exists():
        target = history / f"run_metadata.{stamp}.{suffix}.json"
        suffix += 1
    current.rename(target)
    return target


def _digest(paths: list[Path]) -> str | None:
    """SHA-256 over the (name, content) of each path, or ``None`` if any is missing."""
    if not all(path.is_file() for path in paths):
        return None
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_criteria(
    criteria_dir: Path, language: str, judge_temperature: float
) -> dict[str, CriterionPromptSnapshot]:
    """Read threshold, effective temperature and prompt digest for every gate criterion.

    Args:
        criteria_dir: ``prompts/judge/criteria``.
        language: Prompt language (``pt`` or ``en``).
        judge_temperature: Judge-wide temperature, which a criterion's
            ``config.json`` may override (see ``LLMCriterion`` loading).

    Returns:
        Mapping from criterion name to its snapshot.
    """
    out: dict[str, CriterionPromptSnapshot] = {}
    for name in GATE_CRITERIA:
        config_path = criteria_dir / name / "config.json"
        config: dict = {}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        override = config.get("temperature")
        out[name] = CriterionPromptSnapshot(
            threshold=config.get("threshold"),
            temperature=override if override is not None else judge_temperature,
            sha256=_digest([config_path, criteria_dir / name / language / "prompt.md"]),
        )
    return out


def snapshot_bloom_descriptions(validation_dir: Path, language: str) -> str | None:
    """Digest of the Bloom level descriptions file, or ``None`` when absent."""
    path = validation_dir / language / "data.json"
    return _digest([path]) if path.is_file() else None


def build_judge_qa_run_config(
    *,
    mode: Literal["rejudge", "resume"],
    files: int | None,
    pairs: int | None,
    provider: str | None,
    model_id: str,
    base_url: str | None,
    judge: JudgeConfig,
    criteria_dir: Path = DEFAULT_JUDGE_PROMPTS_DIR,
    bloom_descriptions_dir: Path = BLOOM_DESCRIPTIONS_DIR,
) -> JudgeQARunConfig:
    """Assemble the snapshot for a judge-qa run, digesting the prompts on disk.

    The criterion prompts are resolved in ``judge.language``, the language the
    criteria actually load (``judge-qa --language`` overrides it before this call).
    """
    return JudgeQARunConfig(
        mode=mode,
        files=files,
        pairs=pairs,
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        language=judge.language,
        judge=judge,
        criteria=snapshot_criteria(criteria_dir, judge.language, judge.temperature),
        bloom_descriptions_sha256=snapshot_bloom_descriptions(
            bloom_descriptions_dir, judge.language
        ),
    )


def has_judge_error(validation: JudgePipelineResult) -> bool:
    """Whether any criterion of a verdict failed to run (LLM or parse error).

    Such a verdict reads as a rejection (``CriterionScore.passed`` is False on
    error) but records an infrastructure failure, not a judgment, so
    ``judge-qa`` counts it as failed and re-judges it on resume.
    """
    return any(
        score.error is not None
        for stage in validation.stage_results.values()
        for score in stage.criterion_scores.values()
    )
