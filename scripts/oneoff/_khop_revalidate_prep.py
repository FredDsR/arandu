#!/usr/bin/env python3
"""One-off khop re-validation prep (runs INSIDE a root container on the cluster).

Backs up then clears ONLY the khop_passage / khop_triple state of the
mini-dry-run-qwen run so a resume re-runs khop with the new linker while the
non-khop arms (atlas_rag, bm25, null) stay as an untouched control. Backup at
results/mini-dry-run-qwen/_khop_v1_backup/ makes it reversible. Paths are the
in-container /app/results mount.
"""

from __future__ import annotations

import glob
import json
import os
import shutil

R = "/app/results/mini-dry-run-qwen"
BK = os.path.join(R, "_khop_v1_backup")
ARMS = ("khop_passage", "khop_triple")
STAGES = ("retrieve/outputs", "answers/outputs", "judge_answers/outputs")
CKPTS = (
    "retrieve/**/retrieve_checkpoint.json",
    "answers/**/answer_checkpoint.json",
    "judge_answers/**/judge_answers_checkpoint.json",
)


def _keep(key: str) -> bool:
    return not any(key.startswith(f"{arm}::") for arm in ARMS)


def main() -> None:
    os.makedirs(BK, exist_ok=True)

    for stage in STAGES:
        for arm in ARMS:
            src = os.path.join(R, stage, arm)
            if os.path.isdir(src):
                dst = os.path.join(BK, stage, arm)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                shutil.rmtree(src)
                print(f"backed up + removed  {src}")

    for pattern in CKPTS:
        for cp in glob.glob(os.path.join(R, pattern), recursive=True):
            shutil.copy2(cp, os.path.join(BK, os.path.basename(cp) + ".bak"))
            data = json.load(open(cp))
            before = len(data.get("completed_files", []))
            if isinstance(data.get("completed_files"), list):
                data["completed_files"] = [k for k in data["completed_files"] if _keep(k)]
            if isinstance(data.get("failed_files"), dict):
                data["failed_files"] = {k: v for k, v in data["failed_files"].items() if _keep(k)}
            json.dump(data, open(cp, "w"), indent=2)
            print(f"stripped {cp}: {before} -> {len(data['completed_files'])} completed")

    print("PREP_DONE")


if __name__ == "__main__":
    main()
