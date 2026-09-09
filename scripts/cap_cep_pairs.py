#!/usr/bin/env python3
"""Cap each cep `*_cep_qa.json` file's qa_pairs to a Bloom-stratified sample.

One-off dry-run helper: the per-chunk Bloom ladder generated 670 QA pairs across
4 transcripts (Barra-Celia I alone = 370), which would make judge-qa + the whole
downstream chain run for days. Trim to a small, Bloom-balanced sample per file so
the dry-run stays a fast plumbing shakeout. The full 670-pair set is backed up to
`cep/outputs_full670/` so it can be restored for the real test-kg-04 run.

Usage:
    python3 scripts/cap_cep_pairs.py <pipeline_id> [cap_per_file]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42


def main() -> None:
    pipeline_id = sys.argv[1] if len(sys.argv) > 1 else "mini-dry-run-qwen"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    out_dir = Path("results") / pipeline_id / "cep" / "outputs"
    backup_dir = Path("results") / pipeline_id / "cep" / "outputs_full670"
    backup_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    total_before = total_after = 0

    for f in sorted(out_dir.glob("*_cep_qa.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        pairs = record.get("qa_pairs", [])
        total_before += len(pairs)

        backup = backup_dir / f.name
        if not backup.exists():
            backup.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        if len(pairs) <= cap:
            total_after += len(pairs)
            print(f"  {f.name[:22]}: {len(pairs)} (kept all)")
            continue

        # Bloom-stratified round-robin so the cross-cut stays representative.
        groups: dict[str, list] = defaultdict(list)
        for p in pairs:
            groups[p.get("bloom_level", "?")].append(p)
        for g in groups.values():
            rng.shuffle(g)

        levels = sorted(groups)
        capped: list = []
        i = 0
        while len(capped) < cap and any(groups[lv] for lv in levels):
            lv = levels[i % len(levels)]
            if groups[lv]:
                capped.append(groups[lv].pop())
            i += 1

        record["qa_pairs"] = capped
        # Keep doc-level counters consistent with the trimmed pair list.
        # QARecordCEP enforces total_pairs == len(qa_pairs) on load; a stale
        # total_pairs makes the judge CLI silently skip the whole file.
        record["total_pairs"] = len(capped)
        record["validated_pairs"] = sum(1 for p in capped if p.get("is_valid"))
        record["validation_rate"] = (
            record["validated_pairs"] / len(capped) if capped else 0.0
        )
        f.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        total_after += len(capped)
        dist = defaultdict(int)
        for p in capped:
            dist[p.get("bloom_level", "?")] += 1
        print(f"  {f.name[:22]}: {len(pairs)} -> {len(capped)}  {dict(dist)}")

    print(f"TOTAL: {total_before} -> {total_after}  (backup: {backup_dir})")


if __name__ == "__main__":
    main()
