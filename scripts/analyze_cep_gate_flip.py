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

Run from the repo root:

    uv run python -m scripts.analyze_cep_gate_flip --old thesis-run-01 --new thesis-run-02
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.shared.rag.analysis.metrics import aggregate_arm
from scripts._judge_analysis_common import (
    CRITERIA,
    DEFAULT_THRESHOLD,
    PairScores,
    load_pair_scores,
)

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import AnswerRecord

BLOOM_ORDER = ("remember", "understand", "analyze", "evaluate")

# Arms whose passage chunk_id lives in the same namespace as the CEP pairs.
PASSAGE_NAMESPACE_ARMS = ("bm25_cep_4k",)


def load_pair_meta(cep_dir: Path) -> dict[str, dict[str, str]]:
    """Map every ``qa_pair_id`` to the metadata the cohort tables need.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``.

    Returns:
        Mapping ``qa_pair_id -> {bloom_level, file_id, chunk_id}``, with the id
        built exactly as the retrieve loader builds it.
    """
    meta: dict[str, dict[str, str]] = {}
    for path in sorted(cep_dir.glob("*_cep_qa.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        file_id = record["source_gdrive_id"]
        for idx, pair in enumerate(record["qa_pairs"]):
            segment = pair.get("chunk_id") or "none"
            meta[f"{file_id}:{segment}:{idx}"] = {
                "bloom_level": pair["bloom_level"],
                "file_id": file_id,
                "chunk_id": segment,
            }
    return meta


def approved_ids(scores: PairScores, threshold: float) -> set[str]:
    """Return the pairs whose every evaluated criterion scores >= ``threshold``.

    Args:
        scores: Per-criterion scores from :func:`load_pair_scores`.
        threshold: Approval threshold.

    Returns:
        The approved ``qa_pair_id`` set. A pair with an errored criterion never
        passes, mirroring ``CriterionScore.passed``.
    """
    return {
        pid
        for pid, crit in scores.items()
        if crit and all(s is not None and s >= threshold for s in crit.values())
    }


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


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _report_outcome(name: str, ids: set[str], by_arm: dict[str, list[AnswerRecord]]) -> None:
    """Print the per-arm RAG outcome restricted to one cohort."""
    print(f"\n### {name}  (|answerable| = {len(ids)})")
    print(f"{'arm':<20} {'n':>6} {'KC':>7} {'Correct':>8} {'PassCov':>8} {'answered%':>10}")
    for arm in sorted(by_arm):
        records = [r for r in by_arm[arm] if r.is_answerable and r.qa_pair_id in ids]
        if not records:
            continue
        metrics = aggregate_arm(arm, records, slice_name=name)
        confusion = metrics.confusion
        answered = confusion.get("TC", 0) + confusion.get("TA", 0)
        denominator = sum(confusion.get(k, 0) for k in ("TC", "TA", "FC", "FA"))
        pct = f"{100 * answered / denominator:.0f}%" if denominator else "n/a"
        print(
            f"{arm:<20} {len(records):>6} {_fmt(metrics.knowledge_coverage.mean):>7} "
            f"{_fmt(metrics.answer_correctness.mean):>8} "
            f"{_fmt(metrics.passage_coverage.mean):>8} {pct:>10}"
        )


def _report_provenance(
    name: str,
    ids: set[str],
    by_arm: dict[str, list[AnswerRecord]],
    meta: dict[str, dict[str, str]],
    owner: dict[str, str],
) -> None:
    """Print where the retrieved passages came from, for one cohort."""
    for arm in PASSAGE_NAMESPACE_ARMS:
        records = [r for r in by_arm.get(arm, []) if r.is_answerable and r.qa_pair_id in ids]
        if not records:
            continue
        top1 = topk = 0
        origin: Counter[str] = Counter()
        for record in records:
            gold_chunk = meta[record.qa_pair_id]["chunk_id"]
            gold_file = meta[record.qa_pair_id]["file_id"]
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
        print(
            f"{name:<20} {arm:<14} {100 * top1 / len(records):>9.1f}%"
            f" {100 * topk / len(records):>9.1f}%"
            f" {100 * origin['same doc'] / total:>10.1f}%"
            f" {100 * origin['other doc'] / total:>10.1f}%"
        )


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

    root = Path(args.results_root)
    old_scores = load_pair_scores(root / args.old / "cep" / "outputs")
    new_scores = load_pair_scores(root / args.new / "cep" / "outputs")
    meta = load_pair_meta(root / args.new / "cep" / "outputs")
    owner = {m["chunk_id"]: m["file_id"] for m in meta.values()}

    old_ok = approved_ids(old_scores, args.threshold)
    new_ok = approved_ids(new_scores, args.threshold)
    every = set(meta)
    cohorts = {
        "stable approved": old_ok & new_ok,
        "newly approved": new_ok - old_ok,
        "newly rejected": old_ok - new_ok,
        "stable rejected": every - old_ok - new_ok,
    }

    print(f"# CEP gate: {args.old} (old) vs {args.new} (new), threshold {args.threshold}")
    print(f"\npairs: {len(every)} | approved old: {len(old_ok)} | approved new: {len(new_ok)}")

    agree = len(cohorts["stable approved"]) + len(cohorts["stable rejected"])
    p_observed = agree / len(every)
    p_old, p_new = len(old_ok) / len(every), len(new_ok) / len(every)
    p_expected = p_old * p_new + (1 - p_old) * (1 - p_new)
    kappa = (p_observed - p_expected) / (1 - p_expected)
    print(f"raw verdict agreement: {p_observed:.1%} | Cohen's kappa: {kappa:.3f}")

    print("\n## Cohort size by Bloom level\n")
    print(f"{'cohort':<20} {'total':>6} " + " ".join(f"{b:>11s}" for b in BLOOM_ORDER))
    for name, ids in cohorts.items():
        by_level = Counter(meta[p]["bloom_level"] for p in ids)
        print(
            f"{name:<20} {len(ids):>6} "
            + " ".join(f"{by_level.get(b, 0):>11d}" for b in BLOOM_ORDER)
        )

    print("\n## Criterion that decided the flip\n")
    for name in ("newly approved", "newly rejected"):
        counts: Counter[str] = Counter()
        for pid in cohorts[name]:
            label = flip_criteria(old_scores.get(pid, {}), new_scores.get(pid, {}), args.threshold)
            counts[label] += 1
        print(f"### {name} (n={len(cohorts[name])})")
        for label, n in counts.most_common():
            print(f"   {n:>4d}  {label}")
        print()

    print("## Per-criterion stability\n")
    print(f"{'criterion':<20} {'n':>6} {'same anchor':>12} {'mean |delta|':>13} {'mean delta':>11}")
    for criterion in CRITERIA:
        same = total = 0
        abs_sum = signed_sum = 0.0
        for pid in old_scores:
            before, after = old_scores[pid].get(criterion), new_scores.get(pid, {}).get(criterion)
            if before is None or after is None:
                continue
            total += 1
            abs_sum += abs(after - before)
            signed_sum += after - before
            same += int(abs(after - before) < 1e-9)
        if total:
            print(
                f"{criterion:<20} {total:>6} {same / total:>11.1%} "
                f"{abs_sum / total:>13.3f} {signed_sum / total:>+11.3f}"
            )

    by_arm, _, _ = _group_by_arm(root / args.new / "judge_answers" / "outputs")

    print("\n## RAG outcome per cohort")
    for name, ids in cohorts.items():
        _report_outcome(name, ids, by_arm)

    print("\n## Passage provenance (CEP-namespace arms only)\n")
    print(
        f"{'cohort':<20} {'arm':<14} {'source top-1':>10} {'source top-k':>10}"
        f" {'same doc':>11} {'other doc':>11}"
    )
    for name, ids in cohorts.items():
        _report_provenance(name, ids, by_arm, meta, owner)


if __name__ == "__main__":
    main()
