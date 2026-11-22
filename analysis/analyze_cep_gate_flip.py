"""Refilter the RAG benchmark under two judge-qa gates (read-only).

Re-judging the CEP stage only rewrites ``validation`` / ``is_valid``: the
question, answer, context, Bloom level and chunk_id of every pair stay
byte-identical, and the retrieve/answer/judge stages already cover ALL pairs
(approved or not). Comparing two runs of the same corpus is therefore a pure
refilter, with no stage to re-execute.

This tool partitions the pairs by the pair ``(old gate, new gate)`` and reports,
per cohort: size by Bloom level, which criterion crossed the threshold, the RAG
outcome per arm, and where the retrieved passages came from (source chunk, other
chunk of the same document, other document). The last table is only meaningful
for arms whose ``chunk_id`` shares the CEP namespace (``bm25_cep_4k``); the KG
arms emit synthetic ids and are reported as ``n/a``.

Only pairs present in both runs are compared; pairs that exist in one run only
(different file sets, a migrated ``chunk_id`` namespace) are counted and left
out of every table.

Run from the repo root:

    uv run python -m analysis.analyze_cep_gate_flip --old thesis-run-01 --new thesis-run-02
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from analysis._judge_common import (
    CRITERIA,
    DEFAULT_THRESHOLD,
    PairScores,
    approved_ids,
    load_pair_scores,
    outcome_table,
)
from arandu.shared.agreement.coefficients import cohen_kappa_weighted
from arandu.shared.chunking.schemas import ChunkSet
from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.shared.rag.analysis.loader import build_cross_cut_map
from arandu.utils.console import console
from arandu.utils.logger import print_error

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import AnswerRecord

BLOOM_ORDER = ("remember", "understand", "analyze", "evaluate")

# Arms whose passage chunk_id lives in the same namespace as the CEP pairs.
PASSAGE_NAMESPACE_ARMS = ("bm25_cep_4k",)


def split_pair_id(pair_id: str) -> tuple[str, str]:
    """Return ``(file_id, chunk_id)`` from a ``<file_id>:<chunk_id>:<idx>`` id.

    Args:
        pair_id: A ``qa_pair_id`` as the retrieve loader builds it.

    Returns:
        The source file id and the chunk id segment (``"none"`` for unchunked
        pairs).
    """
    file_id, chunk_id, _ = pair_id.rsplit(":", 2)
    return file_id, chunk_id


def load_chunk_owner(chunk_dir: Path) -> dict[str, str]:
    """Map every chunk of the run's chunk stage to its source file.

    The map must cover every chunk a retriever can return, not only the chunks
    that produced CEP pairs; otherwise a passage from the gold document whose
    chunk yielded no pair would count as "other doc".

    Args:
        chunk_dir: ``results/<id>/chunk/outputs/`` (one sub-directory per view).

    Returns:
        Mapping ``chunk_id -> source_file_id`` over every view. Unreadable
        ``ChunkSet`` files are skipped.
    """
    owner: dict[str, str] = {}
    for path in sorted(chunk_dir.rglob("*.json")):
        try:
            chunk_set = ChunkSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            continue
        for chunks in chunk_set.views.values():
            for chunk in chunks:
                owner[chunk.chunk_id] = chunk.source_file_id
    return owner


def build_cohorts(old_ok: set[str], new_ok: set[str], shared: set[str]) -> dict[str, set[str]]:
    """Partition the shared pairs by their ``(old gate, new gate)`` verdicts.

    Args:
        old_ok: Pairs approved under the old gate.
        new_ok: Pairs approved under the new gate.
        shared: Pairs present in both runs; the four cohorts partition it.

    Returns:
        ``stable approved``, ``newly approved``, ``newly rejected`` and
        ``stable rejected``, each a subset of ``shared``.
    """
    old_ok, new_ok = old_ok & shared, new_ok & shared
    return {
        "stable approved": old_ok & new_ok,
        "newly approved": new_ok - old_ok,
        "newly rejected": old_ok - new_ok,
        "stable rejected": shared - old_ok - new_ok,
    }


def verdict_kappa(old_ok: set[str], new_ok: set[str], shared: set[str]) -> float | None:
    """Cohen's kappa between the two binary gate verdicts over the shared pairs.

    Args:
        old_ok: Pairs approved under the old gate.
        new_ok: Pairs approved under the new gate.
        shared: Pairs present in both runs.

    Returns:
        The kappa, or ``None`` when it is undefined (no pairs, or both gates
        give every pair the same verdict).
    """
    ids = sorted(shared)
    old_bin: list[int | None] = [int(p in old_ok) for p in ids]
    new_bin: list[int | None] = [int(p in new_ok) for p in ids]
    return cohen_kappa_weighted(old_bin, new_bin, weights="linear", scale=(0, 1)).coefficient


def flip_criteria(old: dict[str, float | None], new: dict[str, float | None], t: float) -> str:
    """Name the criteria that crossed the threshold between two judgings.

    Args:
        old: Per-criterion scores under the old gate.
        new: Per-criterion scores under the new gate.
        t: Approval threshold.

    Returns:
        A ``+``-joined label, or a marker when no criterion explains the flip.
    """
    moved: list[str] = []
    for criterion in CRITERIA:
        before, after = old.get(criterion), new.get(criterion)
        if before is None or after is None:
            if (before is None) != (after is None):
                moved.append(f"{criterion} (judge error)")
            continue
        if (before >= t) != (after >= t):
            moved.append(criterion)
    return "+".join(moved) if moved else "(no criterion: error/absent)"


def _stability_lines(old_scores: PairScores, new_scores: PairScores, shared: set[str]) -> list[str]:
    """Per-criterion score stability between the two judgings."""
    lines = [
        "## Per-criterion stability\n",
        f"{'criterion':<20} {'n':>6} {'same anchor':>12} {'mean |delta|':>13} {'mean delta':>11}",
    ]
    for criterion in CRITERIA:
        same = total = 0
        abs_sum = signed_sum = 0.0
        for pid in sorted(shared):
            before = old_scores.get(pid, {}).get(criterion)
            after = new_scores.get(pid, {}).get(criterion)
            if before is None or after is None:
                continue
            total += 1
            abs_sum += abs(after - before)
            signed_sum += after - before
            same += int(abs(after - before) < 1e-9)
        if total:
            lines.append(
                f"{criterion:<20} {total:>6} {same / total:>11.1%} "
                f"{abs_sum / total:>13.3f} {signed_sum / total:>+11.3f}"
            )
    return lines


def _provenance_lines(
    name: str, ids: set[str], by_arm: dict[str, list[AnswerRecord]], owner: dict[str, str]
) -> list[str]:
    """Where the retrieved passages came from, for one cohort."""
    lines: list[str] = []
    for arm in PASSAGE_NAMESPACE_ARMS:
        records = [r for r in by_arm.get(arm, []) if r.is_answerable and r.qa_pair_id in ids]
        if not records:
            continue
        top1 = topk = 0
        origin: Counter[str] = Counter()
        for record in records:
            gold_file, gold_chunk = split_pair_id(record.qa_pair_id)
            chunk_ids = [p.chunk_id for p in record.passages]
            if chunk_ids and chunk_ids[0] == gold_chunk:
                top1 += 1
            if gold_chunk in chunk_ids:
                topk += 1
            for chunk_id in chunk_ids:
                if chunk_id == gold_chunk:
                    origin["source"] += 1
                elif owner.get(chunk_id) == gold_file:
                    origin["same doc"] += 1
                else:
                    origin["other doc"] += 1
        total = sum(origin.values()) or 1
        lines.append(
            f"{name:<20} {arm:<14} {100 * top1 / len(records):>9.1f}%"
            f" {100 * topk / len(records):>9.1f}%"
            f" {100 * origin['same doc'] / total:>10.1f}%"
            f" {100 * origin['other doc'] / total:>10.1f}%"
        )
    return lines


def render_report(
    old_id: str,
    new_id: str,
    root: Path,
    threshold: float,
) -> str:
    """Build the full flip-cohort report for two runs.

    Args:
        old_id: Run id judged by the old gate.
        new_id: Run id judged by the new gate (its RAG answers are used).
        root: Results root directory.
        threshold: Approval threshold.

    Returns:
        The report as plain text.

    Raises:
        ValueError: If the two runs share no CEP pair.
    """
    old_scores = load_pair_scores(root / old_id / "cep" / "outputs")
    new_scores = load_pair_scores(root / new_id / "cep" / "outputs")
    old_meta = build_cross_cut_map(root / old_id / "cep" / "outputs")
    new_meta = build_cross_cut_map(root / new_id / "cep" / "outputs")
    shared = set(old_meta) & set(new_meta)
    if not shared:
        raise ValueError(
            f"{old_id} and {new_id} share no CEP pair under {root}; "
            "check the run ids and --results-root."
        )

    old_ok = approved_ids(old_scores, threshold) & shared
    new_ok = approved_ids(new_scores, threshold) & shared
    cohorts = build_cohorts(old_ok, new_ok, shared)

    lines = [
        f"# CEP gate: {old_id} (old) vs {new_id} (new), threshold {threshold}",
        f"\npairs: {len(shared)} | approved old: {len(old_ok)} | approved new: {len(new_ok)}",
    ]
    only_old, only_new = len(set(old_meta) - shared), len(set(new_meta) - shared)
    if only_old or only_new:
        lines.append(f"left out (one run only): {only_old} in old, {only_new} in new")

    agree = len(cohorts["stable approved"]) + len(cohorts["stable rejected"])
    kappa = verdict_kappa(old_ok, new_ok, shared)
    kappa_text = "n/a" if kappa is None else f"{kappa:.3f}"
    lines.append(f"raw verdict agreement: {agree / len(shared):.1%} | Cohen's kappa: {kappa_text}")

    lines += [
        "\n## Cohort size by Bloom level\n",
        f"{'cohort':<20} {'total':>6} " + " ".join(f"{b:>11s}" for b in BLOOM_ORDER),
    ]
    for name, ids in cohorts.items():
        by_level = Counter(new_meta[p].bloom_level for p in ids)
        lines.append(
            f"{name:<20} {len(ids):>6} "
            + " ".join(f"{by_level.get(b, 0):>11d}" for b in BLOOM_ORDER)
        )

    lines.append("\n## Criterion that decided the flip\n")
    for name in ("newly approved", "newly rejected"):
        counts: Counter[str] = Counter()
        for pid in cohorts[name]:
            counts[flip_criteria(old_scores.get(pid, {}), new_scores.get(pid, {}), threshold)] += 1
        lines.append(f"### {name} (n={len(cohorts[name])})")
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        lines += [f"   {n:>4d}  {label}" for label, n in ranked]
        lines.append("")

    lines += _stability_lines(old_scores, new_scores, shared)

    by_arm, _, _ = _group_by_arm(root / new_id / "judge_answers" / "outputs")
    lines.append("\n## RAG outcome per cohort")
    for name, ids in cohorts.items():
        lines += outcome_table(name, ids, by_arm)

    owner = load_chunk_owner(root / new_id / "chunk" / "outputs")
    lines.append("\n## Passage provenance (CEP-namespace arms only)\n")
    if not owner:
        lines.append(f"n/a: no chunk stage under {root / new_id / 'chunk' / 'outputs'}")
        return "\n".join(lines)
    lines.append(
        f"{'cohort':<20} {'arm':<14} {'source top-1':>10} {'source top-k':>10}"
        f" {'same doc':>11} {'other doc':>11}"
    )
    for name, ids in cohorts.items():
        lines += _provenance_lines(name, ids, by_arm, owner)
    return "\n".join(lines)


def main() -> None:
    """Entry point for the flip-cohort report."""
    parser = argparse.ArgumentParser(description="Refilter the RAG benchmark under two gates.")
    parser.add_argument("--old", default="thesis-run-01", help="Run id of the old gate.")
    parser.add_argument("--new", default="thesis-run-02", help="Run id of the new gate.")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Approval threshold."
    )
    args = parser.parse_args()
    try:
        report = render_report(args.old, args.new, Path(args.results_root), args.threshold)
    except ValueError as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc
    console.print(report, markup=False, highlight=False, soft_wrap=True)


if __name__ == "__main__":
    main()
