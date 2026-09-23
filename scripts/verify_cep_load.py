#!/usr/bin/env python3
"""Verify every cep outputs/*_cep_qa.json loads through QARecordCEP.

One-off dry-run helper, companion to cap_cep_pairs.py: a capped file with
stale doc-level counters fails the judge CLI's schema load silently, so this
check runs after any cap/restore to guarantee the judge will see all files.
Exits non-zero when any file fails.

Usage:
    python3 scripts/verify_cep_load.py <pipeline_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

from arandu.qa.schemas import QARecordCEP


def main() -> None:
    pipeline_id = sys.argv[1] if len(sys.argv) > 1 else "mini-dry-run-qwen"
    out_dir = Path("results") / pipeline_id / "cep" / "outputs"
    failures = 0
    for f in sorted(out_dir.glob("*_cep_qa.json")):
        try:
            record = QARecordCEP.model_validate_json(f.read_text(encoding="utf-8"))
            print(f"  {f.name[:22]}: OK ({len(record.qa_pairs)} pairs)")
        except Exception as exc:  # noqa: BLE001 — report-and-continue check
            failures += 1
            print(f"  {f.name[:22]}: FAIL {str(exc)[:160]}")
    if failures:
        sys.exit(1)
    print("all files load OK")


if __name__ == "__main__":
    main()
