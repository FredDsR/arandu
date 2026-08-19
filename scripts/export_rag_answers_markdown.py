#!/usr/bin/env python3
"""Export every judged RAG answer to a readable markdown for manual review.

Walks ``results/<pipeline_id>/judge_answers/outputs/<arm>/{cep,nonanswerable}/
*.json`` and renders one markdown document grouped by **retriever arm ->
answerability bucket -> answer record**. For each record the question is shown
with the system answer (or the abstention), the retrieved passages it was given,
and the judge's per-criterion scores (passage coverage, source recovery, answer
correctness, faithfulness, abstention) with their rationales.

This is the answer-stage sibling of ``export_cep_pairs_markdown.py``: where that
script dumps the generated CEP benchmark, this one dumps how each retriever arm
*answered* that benchmark and how the LLM-as-a-judge scored it, so the full
per-arm behaviour (over-caution, hallucination, coverage) can be read at once
alongside the aggregate ``analysis/outputs/tables.md``.

Reads raw JSON rather than the pydantic record so legacy on-disk validation
layouts still render (matches the sibling scripts' approach).

Usage:
    python3 scripts/export_rag_answers_markdown.py [pipeline_id] [--output PATH]
        [--arm NAME] [--committed-only]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from arandu.utils.logger import print_error, print_info, print_success

# Buckets produced by the answer/judge stages, in display order.
BUCKETS: tuple[tuple[str, str], ...] = (
    ("cep", "Respondiveis (CEP)"),
    ("nonanswerable", "Nao-respondiveis"),
)

# Passage payload preview length (chars) inside the collapsible passages block.
_PASSAGE_PREVIEW = 280
# Max retrieved passages listed per record (ranked); the rest are summarised.
_MAX_PASSAGES_SHOWN = 5


def _fmt_score(value: Any) -> str:
    """Format a numeric score to 3 decimals, or ``-`` when missing."""
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def _iter_criteria(validation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten the judge ``stage_results`` into an ordered list of criteria.

    Each stage holds a ``criterion_scores`` map of ``name -> {score, threshold,
    rationale, ...}``. This walks every stage in insertion order and returns one
    entry per criterion, tagged with its owning stage.

    Args:
        validation: The raw ``validation`` payload of an answer record, or None.

    Returns:
        List of dicts with keys ``stage``, ``name``, ``score``, ``threshold``,
        ``rationale``.
    """
    out: list[dict[str, Any]] = []
    if not validation:
        return out
    stages = validation.get("stage_results") or {}
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        scores = stage.get("criterion_scores") or {}
        for crit_name, crit in scores.items():
            if not isinstance(crit, dict):
                continue
            out.append(
                {
                    "stage": stage_name,
                    "name": crit_name,
                    "score": crit.get("score"),
                    "threshold": crit.get("threshold"),
                    "rationale": crit.get("rationale"),
                }
            )
    return out


def _render_passages(passages: list[dict[str, Any]]) -> list[str]:
    """Render the retrieved passages as a collapsible markdown block.

    Args:
        passages: The ``passages`` list of an answer record.

    Returns:
        List of markdown lines (possibly empty when no passages).
    """
    if not passages:
        return ["_Passagens recuperadas:_ nenhuma", ""]

    ranked = sorted(passages, key=lambda p: p.get("rank", 0))
    lines = [
        f"<details><summary>Passagens recuperadas ({len(passages)})</summary>",
        "",
    ]
    for p in ranked[:_MAX_PASSAGES_SHOWN]:
        rank = p.get("rank", "?")
        score = _fmt_score(p.get("score"))
        chunk_id = p.get("chunk_id", "?")
        payload = str(p.get("payload", "")).strip().replace("\n", " ")
        if len(payload) > _PASSAGE_PREVIEW:
            payload = payload[:_PASSAGE_PREVIEW].rstrip() + "..."
        lines.append(f"- **#{rank}** (score {score}, `{chunk_id}`): {payload}")
    if len(passages) > _MAX_PASSAGES_SHOWN:
        lines.append(f"- _... mais {len(passages) - _MAX_PASSAGES_SHOWN} passagem(ns)_")
    lines += ["", "</details>", ""]
    return lines


def _render_record(record: dict[str, Any], index: int) -> list[str]:
    """Render a single judged answer record as markdown lines.

    Args:
        record: A raw judged-answer dict.
        index: 1-based position of the record within its bucket.

    Returns:
        List of markdown lines.
    """
    abstained = bool(record.get("abstained"))
    is_valid = record.get("is_valid")
    valid_mark = {True: "valido", False: "rejeitado", None: "nao-julgado"}.get(
        is_valid, "nao-julgado"
    )
    validation = record.get("validation") or {}
    rejected_at = validation.get("rejected_at")
    flow = "abstido" if abstained else "comprometido"
    gate = f" | parou em: `{rejected_at}`" if rejected_at else ""

    question = str(record.get("question", "")).strip()
    lines = [
        f"#### Q{index} - {flow} | {valid_mark}{gate}",
        "",
        f"**Pergunta:** {question}",
        "",
    ]

    if abstained:
        lines += ["**Resposta:** _(abstido)_", ""]
    else:
        lines += [f"**Resposta:** {str(record.get('answer_text', '')).strip()}", ""]

    # Per-criterion judge scores (one compact line each), rationales collapsed.
    criteria = _iter_criteria(validation)
    if criteria:
        score_bits = [
            f"{c['name']} {_fmt_score(c['score'])}"
            + (f"/thr {_fmt_score(c['threshold'])}" if c["threshold"] is not None else "")
            for c in criteria
        ]
        lines += ["**Avaliacao:** " + " | ".join(score_bits), ""]
        rationale_lines = [
            f"- **{c['name']}** ({_fmt_score(c['score'])}): {str(c['rationale']).strip()}"
            for c in criteria
            if c["rationale"]
        ]
        if rationale_lines:
            lines += [
                "<details><summary>Rationale do juiz por criterio</summary>",
                "",
                *rationale_lines,
                "",
                "</details>",
                "",
            ]

    lines += _render_passages(record.get("passages") or [])
    return lines


def _render_arm(arm: str, arm_dir: Path, committed_only: bool) -> tuple[list[str], dict[str, int]]:
    """Render one retriever arm, grouped by answerability bucket.

    Args:
        arm: Arm directory name (e.g. ``khop_passage``).
        arm_dir: Path to ``judge_answers/outputs/<arm>``.
        committed_only: When True, skip records where the system abstained.

    Returns:
        Tuple of (markdown lines, stats dict with ``records``/``abstained``/
        ``valid``).
    """
    body: list[str] = []
    stats = Counter()
    bucket_blocks: list[str] = []

    for sub, label in BUCKETS:
        files = sorted((arm_dir / sub).glob("*.json"))
        records: list[dict[str, Any]] = []
        for f in files:
            try:
                records.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                print_error(f"  skip {f.name[:28]}: {exc}")
        rendered = [r for r in records if not (committed_only and r.get("abstained"))]
        stats["records"] += len(rendered)
        stats["abstained"] += sum(1 for r in rendered if r.get("abstained"))
        stats["valid"] += sum(1 for r in rendered if r.get("is_valid") is True)
        if not rendered:
            continue
        bucket_blocks += [f"### {label} ({len(rendered)})", ""]
        for i, r in enumerate(rendered, start=1):
            bucket_blocks += _render_record(r, i)

    if not bucket_blocks:
        return [], dict(stats)

    body += [
        f"## Arm: `{arm}`",
        "",
        f"_Registros:_ {stats['records']} | _Abstencoes:_ {stats['abstained']} "
        f"| _Validos:_ {stats['valid']}",
        "",
        *bucket_blocks,
        "---",
        "",
    ]
    return body, dict(stats)


def main() -> None:
    """CLI entry point: render all judged RAG answers of a run to markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline_id", nargs="?", default="mini-dry-run-qwen", help="Run id under results/"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output markdown path (default: under run dir)"
    )
    parser.add_argument(
        "--arm", default=None, help="Render only this retriever arm (e.g. khop_passage)"
    )
    parser.add_argument(
        "--committed-only", action="store_true", help="Skip records where the system abstained"
    )
    args = parser.parse_args()

    out_dir = Path("results") / args.pipeline_id / "judge_answers" / "outputs"
    if not out_dir.is_dir():
        print_error(f"No judge_answers outputs found: {out_dir}")
        raise SystemExit(1)

    arm_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    if args.arm:
        arm_dirs = [d for d in arm_dirs if d.name == args.arm]
    if not arm_dirs:
        print_error(f"No arm directories under {out_dir}")
        raise SystemExit(1)

    output = args.output or (
        Path("results") / args.pipeline_id / "judge_answers" / "answers_review.md"
    )

    body: list[str] = []
    totals: Counter[str] = Counter()
    per_arm: list[str] = []
    for arm_dir in arm_dirs:
        arm_lines, stats = _render_arm(arm_dir.name, arm_dir, args.committed_only)
        body += arm_lines
        for key, val in stats.items():
            totals[key] += val
        per_arm.append(
            f"{arm_dir.name}: {stats.get('records', 0)} reg "
            f"({stats.get('abstained', 0)} absten., {stats.get('valid', 0)} val.)"
        )

    scope = "apenas respostas comprometidas" if args.committed_only else "todas as respostas"
    head = [
        f"# Respostas RAG julgadas - `{args.pipeline_id}`",
        "",
        f"Arms: {len(arm_dirs)} | Registros: {totals.get('records', 0)} "
        f"| Abstencoes: {totals.get('abstained', 0)} | Validos: {totals.get('valid', 0)} "
        f"| Escopo: {scope}",
        "",
        "Por arm: " + " | ".join(per_arm),
        "",
        "Cada arm -> bucket de respondibilidade -> registro (pergunta, resposta ou "
        "abstencao, avaliacao do juiz por criterio, e passagens recuperadas).",
        "",
        "Metricas agregadas por arm em `analysis/outputs/tables.md`.",
        "",
        "---",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(head + body), encoding="utf-8")
    print_info(
        f"Arms: {len(arm_dirs)} | registros: {totals.get('records', 0)} "
        f"| abstencoes: {totals.get('abstained', 0)} | escopo: {scope}"
    )
    print_success(f"Markdown escrito em {output}")


if __name__ == "__main__":
    main()
