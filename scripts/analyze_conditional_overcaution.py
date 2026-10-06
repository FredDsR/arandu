"""Over-caution split by whether the retrieved evidence supports the answer (read-only).

Over-caution, ``FA / (FA + TC)``, counts every abstention on an answerable probe
as an error. Answerability is defined against the corpus, not the graph, so an
FA joins two failures: evidence that does not contain the answer (abstaining is
then correct for that index) and an answerer that declines evidence that does.
``passage_coverage`` (judge score in [0, 1] of whether the retrieved passages
support the reference answer, computed on every answerable record) separates
them. This script reports, per arm, the over-caution rate among records whose
passage coverage is at or above the threshold ("supported") and below it
("unsupported"), restricted to the useful pairs (approved by the judge-qa gate,
Bloom level above remember).

Backs the conditional over-caution sentence and the evidence-support columns
(mean and pass rate) of the COLING 2027 paper (Section 6, retrieval benchmark);
the unconditioned OC and the passage-coverage mean it prints reproduce Table 3 of
that paper for thesis-run-02.

Run from the repo root:

    uv run python -m scripts.analyze_conditional_overcaution --id thesis-run-02
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from arandu.shared.rag.analysis.batch import _group_by_arm
from arandu.shared.rag.analysis.classifier import classify_record
from arandu.shared.rag.analysis.loader import build_cross_cut_map
from arandu.shared.rag.analysis.wilson import wilson_ci
from scripts._judge_analysis_common import DEFAULT_THRESHOLD, approved_ids, load_pair_scores

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import AnswerRecord

RETRIEVAL_STAGE = "retrieval_scoring"
PASSAGE_COVERAGE = "passage_coverage"


def _passage_coverage(record: AnswerRecord) -> float | None:
    """Return the passage-coverage score, or ``None`` if absent or errored."""
    if record.validation is None:
        return None
    step = record.validation.stage_results.get(RETRIEVAL_STAGE)
    if step is None:
        return None
    cs = step.criterion_scores.get(PASSAGE_COVERAGE)
    if cs is None or cs.error is not None:
        return None
    return cs.score


def _rate(abstained: list[bool]) -> str:
    if not abstained:
        return "n/a"
    k = sum(abstained)
    return f"{k / len(abstained):.3f} ({k}/{len(abstained)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Over-caution split by passage coverage.")
    parser.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-02).")
    parser.add_argument("--results-root", default="results", help="Results root dir.")
    parser.add_argument(
        "--gate-threshold", type=float, default=DEFAULT_THRESHOLD, help="judge-qa approval."
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Passage coverage at or above which the evidence counts as supporting.",
    )
    args = parser.parse_args()

    base = Path(args.results_root) / args.id
    cep_dir = base / "cep" / "outputs"
    bloom = {pid: meta.bloom_level for pid, meta in build_cross_cut_map(cep_dir).items()}
    useful = {
        pid
        for pid in approved_ids(load_pair_scores(cep_dir), args.gate_threshold)
        if bloom.get(pid) != "remember"
    }
    by_arm, _, _ = _group_by_arm(base / "judge_answers" / "outputs")

    print(f"useful pairs: {len(useful)}; coverage threshold: {args.coverage_threshold}")
    header = (
        f"{'Arm':<22} {'n':>4} {'OC':>7} {'PassCov':>8}  {'OC|supported':<18} {'OC|unsupported'}"
    )
    print(header)
    for arm in sorted(by_arm):
        rows: list[tuple[bool, float | None]] = []
        for r in by_arm[arm]:
            if not r.is_answerable or r.qa_pair_id not in useful:
                continue
            cell = classify_record(r)
            if cell == "unknown":
                continue
            rows.append((cell == "FA", _passage_coverage(r)))
        if not rows:
            continue
        covs = [c for _, c in rows if c is not None]
        sup = [a for a, c in rows if c is not None and c >= args.coverage_threshold]
        uns = [a for a, c in rows if c is not None and c < args.coverage_threshold]
        oc = sum(a for a, _ in rows) / len(rows)
        cov = f"{sum(covs) / len(covs):.3f}" if covs else "n/a"
        print(f"{arm:<22} {len(rows):>4} {oc:>7.3f} {cov:>8}  {_rate(sup):<18} {_rate(uns)}")

    # Evidence support (``passage_coverage``) reported two ways: the mean of the
    # graded scores and the share of probes at or above the threshold, with a
    # 95% Wilson interval, overall and per Bloom level.
    print()
    print(f"evidence support (passage_coverage): mean and pass rate (>= {args.coverage_threshold})")
    print(f"{'Arm':<22} {'level':<11} {'n':>4} {'mean':>7}  {'pass':<16} {'95% Wilson'}")
    levels = ["all", "understand", "analyze", "evaluate"]
    for arm in sorted(by_arm):
        for level in levels:
            covs = [
                c
                for r in by_arm[arm]
                if r.is_answerable
                and r.qa_pair_id in useful
                and (level == "all" or bloom.get(r.qa_pair_id) == level)
                and classify_record(r) != "unknown"
                and (c := _passage_coverage(r)) is not None
            ]
            if not covs:
                continue
            k = sum(c >= args.coverage_threshold for c in covs)
            lo, hi = wilson_ci(k, len(covs))
            print(
                f"{arm:<22} {level:<11} {len(covs):>4} {sum(covs) / len(covs):>7.3f}  "
                f"{k / len(covs):.3f} ({k}/{len(covs)}){'':<2} [{lo:.3f}, {hi:.3f}]"
            )


if __name__ == "__main__":
    main()
