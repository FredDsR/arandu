"""Near-threshold QA-judge audit protocol.

Surfaces CEP QA pairs that the LLM-as-a-Judge REJECTED but only barely: the
pair would have passed if a single criterion had scored one anchor step higher.
With the anchored 5-point scale {1.0, 0.75, 0.5, 0.25, 0.0} and a 0.625
threshold, the closest failing anchor is 0.5 (margin 0.125). Pairs whose ONLY
sub-threshold criterion scored 0.5 are the prime candidates for "good example
excluded by too-strict filtering" and are the focus of this audit.

Protocol (what this script encodes):
  1. Load every ``*_cep_qa.json`` under the input dir; read each pair's
     ``validation.stage_results.*.criterion_scores`` (continuous criteria only;
     ordinal criteria run in score mode and never gate, so they are ignored).
  2. A pair FAILED iff ``validation.passed is False``. For each failed pair,
     collect its failing criteria (``score < threshold``) and the margin
     (``threshold - score``).
  3. Classify each failed pair:
       - ``sole_near_miss``  : exactly ONE failing criterion, scored 0.5
                               (margin == 0.125). Strongest recovery candidate.
       - ``all_near_miss``   : >1 failing criteria but ALL scored 0.5.
       - ``has_severe``      : at least one failing criterion <= 0.25 (a real
                               miss, not a borderline call).
  4. Rank ``sole_near_miss`` pairs by the strength of their PASSING criteria
     (higher mean = "strong pair, one borderline criterion"), so the best
     wrongly-excluded examples surface first.
  5. Report: counts by class, the gate-keeping criterion distribution, the
     per-Bloom-level breakdown, and the top-N candidate pairs verbatim (question,
     answer, the single failing criterion + its rationale, the passing scores)
     for human review.

The output is a human-review worklist, NOT an automated re-label: a person
decides whether each surfaced pair is genuinely good. Recurring patterns (e.g.
one criterion gate-keeping most near-misses) are evidence the threshold or that
criterion's rubric may be mis-calibrated.

Usage:
    python scripts/audit_near_threshold_qa.py <cep_outputs_dir> [--top N] \
        [--out report.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NEAR_MARGIN = 0.125  # threshold(0.625) - closest failing anchor(0.5)


def _iter_criteria(validation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Flatten all continuous criterion scores across stages.

    Returns (name, criterion_dict) pairs, skipping ordinal criteria (which run
    in score mode and do not gate) and criteria that errored.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for _stage, step in (validation.get("stage_results") or {}).items():
        for name, cs in (step.get("criterion_scores") or {}).items():
            if cs.get("score") is None:  # ordinal or errored -> not a continuous gate
                continue
            out.append((name, cs))
    return out


def classify_pair(validation: dict[str, Any]) -> dict[str, Any] | None:
    """Classify a FAILED pair by how close it was to passing.

    Returns None if the pair passed or carries no usable continuous criteria.
    """
    if validation.get("passed") is not False:
        return None
    crits = _iter_criteria(validation)
    if not crits:
        return None

    failing = [(n, c) for n, c in crits if c["score"] < c["threshold"]]
    passing = [(n, c) for n, c in crits if c["score"] >= c["threshold"]]
    if not failing:
        return None  # passed flag false but no continuous criterion failed; skip

    worst = min(c["score"] for _n, c in failing)
    margins = {n: round(c["threshold"] - c["score"], 4) for n, c in failing}
    has_severe = worst <= 0.25 + 1e-9
    all_near = all(m <= NEAR_MARGIN + 1e-9 for m in margins.values())

    if len(failing) == 1 and all_near:
        cls = "sole_near_miss"
    elif all_near:
        cls = "all_near_miss"
    elif has_severe:
        cls = "has_severe"
    else:
        cls = "other"

    pass_mean = sum(c["score"] for _n, c in passing) / len(passing) if passing else 0.0
    return {
        "cls": cls,
        "failing": [(n, c["score"], c["threshold"], c.get("rationale", "")) for n, c in failing],
        "passing": [(n, c["score"]) for n, c in passing],
        "pass_mean": round(pass_mean, 4),
        "gate": failing[0][0] if len(failing) == 1 else "+".join(sorted(n for n, _ in failing)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--top", type=int, default=20, help="Top-N sole-near-miss examples to dump.")
    ap.add_argument("--out", type=Path, default=None, help="Write markdown report here.")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_cep_qa.json"))
    total_pairs = judged = passed = 0
    by_cls: Counter[str] = Counter()
    sole_gate: Counter[str] = Counter()
    sole_by_bloom: Counter[str] = Counter()
    gate_by_bloom: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: list[dict[str, Any]] = []

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for p in data.get("qa_pairs", []):
            total_pairs += 1
            v = p.get("validation")
            if not v:
                continue
            judged += 1
            if v.get("passed") is True:
                passed += 1
                continue
            info = classify_pair(v)
            if info is None:
                continue
            by_cls[info["cls"]] += 1
            bloom = p.get("bloom_level", "?")
            if info["cls"] == "sole_near_miss":
                sole_gate[info["gate"]] += 1
                sole_by_bloom[bloom] += 1
                gate_by_bloom[info["gate"]][bloom] += 1
                candidates.append(
                    {
                        "file": f.name,
                        "bloom": bloom,
                        "question": p.get("question", ""),
                        "answer": p.get("answer", ""),
                        "gate": info["gate"],
                        "gate_score": info["failing"][0][1],
                        "gate_rationale": info["failing"][0][3],
                        "passing": info["passing"],
                        "pass_mean": info["pass_mean"],
                    }
                )

    candidates.sort(key=lambda c: c["pass_mean"], reverse=True)
    failed = judged - passed

    lines: list[str] = []
    w = lines.append
    w("# Near-threshold QA-judge audit — thesis-run-01\n")
    w(f"- Files: **{len(files)}** | pairs: **{total_pairs}** | judged: **{judged}** "
      f"| passed: **{passed}** ({passed / judged * 100:.1f}%) | failed: **{failed}**\n")
    w("## Failed-pair classification\n")
    w("| Class | Count | % of failed |")
    w("|---|---|---|")
    for cls in ("sole_near_miss", "all_near_miss", "has_severe", "other"):
        n = by_cls.get(cls, 0)
        w(f"| {cls} | {n} | {n / failed * 100:.1f}% |" if failed else f"| {cls} | {n} | - |")
    recoverable = by_cls.get("sole_near_miss", 0)
    w(f"\n**{recoverable} pairs ({recoverable / failed * 100:.1f}% of failures)** failed on a "
      "SINGLE criterion that scored 0.5 (one anchor below the 0.625 gate) — the prime "
      "over-strict-exclusion candidates.\n")

    w("## Gate-keeping criterion (sole near-miss fails)\n")
    w("| Criterion | Pairs gated |")
    w("|---|---|")
    for name, n in sole_gate.most_common():
        w(f"| {name} | {n} |")

    w("\n## Sole near-miss by Bloom level\n")
    w("| Bloom | Pairs |")
    w("|---|---|")
    for bloom, n in sole_by_bloom.most_common():
        w(f"| {bloom} | {n} |")

    w(f"\n## Recovery candidates by gate criterion (top {args.top} each, strongest first)\n")
    w("Review each criterion's near-misses as a cohort: a criterion whose 0.5 "
      "rationales repeatedly penalise *correct, faithful, well-formed* answers is "
      "over-strict; one that catches real defects (fabrication, context-dependence) "
      "is working.\n")
    for gate in [g for g, _ in sole_gate.most_common()]:
        group = [c for c in candidates if c["gate"] == gate]
        w(f"### Gate: `{gate}` — {len(group)} sole near-miss pairs\n")
        for i, c in enumerate(group[: args.top], 1):
            passing_str = ", ".join(f"{n}={s}" for n, s in c["passing"])
            w(f"**{i}. [{c['bloom']}]** (others: {passing_str})")
            w(f"- **Q:** {c['question']}")
            w(f"- **A:** {c['answer']}")
            w(f"- **{gate}=0.5 rationale:** {c['gate_rationale']}")
            w(f"- _source: {c['file']}_\n")

    report = "\n".join(lines)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    print(report[:4000])


if __name__ == "__main__":
    main()
