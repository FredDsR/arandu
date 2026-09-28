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
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from arandu.qa.config import JudgeConfig  # noqa: TC001 (pydantic needs runtime access)
from arandu.shared.judge.factory import DEFAULT_JUDGE_PROMPTS_DIR
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

# Criteria the gate can evaluate; remember pairs use only the first two
# (see QAJudge._build_pipeline).
GATE_CRITERIA: tuple[str, ...] = (
    "faithfulness",
    "bloom_calibration",
    "informativeness",
    "self_containedness",
)

# Bloom level descriptions fed to bloom_calibration (mirrors qa.cep.judge).
BLOOM_DESCRIPTIONS_DIR = get_project_root() / "prompts" / "qa" / "cep" / "validation"


class CriterionPromptSnapshot(BaseModel):
    """Threshold and prompt digest of one gate criterion.

    Attributes:
        threshold: Pass threshold read from the criterion's ``config.json``
            (``None`` when absent).
        sha256: Digest over the criterion's ``config.json`` and its prompt
            for the run language, so a prompt edit changes the digest.
    """

    threshold: float | None
    sha256: str


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
        which case there is no step directory to write metadata into.
    """
    resolved = input_dir.resolve()
    if resolved.name != "outputs" or resolved.parent.name != "cep":
        return None
    pipeline_dir = resolved.parent.parent
    return pipeline_dir.parent, pipeline_dir.name


def _digest(paths: list[Path]) -> str:
    """SHA-256 over the (name, content) of each existing path, in order."""
    h = hashlib.sha256()
    for path in paths:
        if path.is_file():
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_criteria(criteria_dir: Path, language: str) -> dict[str, CriterionPromptSnapshot]:
    """Read threshold and prompt digest for every gate criterion.

    Args:
        criteria_dir: ``prompts/judge/criteria``.
        language: Prompt language (``pt`` or ``en``).

    Returns:
        Mapping from criterion name to its snapshot.
    """
    out: dict[str, CriterionPromptSnapshot] = {}
    for name in GATE_CRITERIA:
        config_path = criteria_dir / name / "config.json"
        threshold: float | None = None
        if config_path.is_file():
            threshold = json.loads(config_path.read_text(encoding="utf-8")).get("threshold")
        out[name] = CriterionPromptSnapshot(
            threshold=threshold,
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
    criteria actually load (the CLI ``--language`` only reaches ``CEPConfig``).
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
        criteria=snapshot_criteria(criteria_dir, judge.language),
        bloom_descriptions_sha256=snapshot_bloom_descriptions(
            bloom_descriptions_dir, judge.language
        ),
    )
