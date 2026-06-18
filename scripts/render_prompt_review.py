#!/usr/bin/env python3
"""Render all PT prompts + criteria configs of the thesis run into one markdown.

Read-only snapshot for prompt review. Regenerate after editing any ``prompts/``
file. Output: ``docs/reports/thesis-run-prompts-review-pt.md``.

Example:
    uv run python scripts/render_prompt_review.py
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

OUT = Path("docs/reports/thesis-run-prompts-review-pt.md")

# (stage title, arandu command, [(path, kind)]) in pipeline order. kind: md|j2|json
STAGES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "1. judge-transcription",
        "arandu judge-transcription",
        [
            ("prompts/judge/criteria/language_drift/pt/prompt.md", "md"),
            ("prompts/judge/criteria/language_drift/config.json", "json"),
            ("prompts/judge/criteria/hallucination_loop/pt/prompt.md", "md"),
            ("prompts/judge/criteria/hallucination_loop/config.json", "json"),
        ],
    ),
    (
        "2. generate-cep-qa (CEP/QA generation)",
        "arandu generate-cep-qa",
        [
            ("prompts/qa/cep/pt/bloom_scaffolding.md", "md"),
            ("prompts/qa/cep/pt/data.json", "json"),
        ],
    ),
    (
        "3. judge-qa (CEP-pair validation)",
        "arandu judge-qa",
        [
            ("prompts/qa/cep/validation/pt/data.json", "json"),
            ("prompts/judge/criteria/faithfulness/pt/prompt.md", "md"),
            ("prompts/judge/criteria/faithfulness/config.json", "json"),
            ("prompts/judge/criteria/bloom_calibration/pt/prompt.md", "md"),
            ("prompts/judge/criteria/bloom_calibration/config.json", "json"),
            ("prompts/judge/criteria/informativeness/pt/prompt.md", "md"),
            ("prompts/judge/criteria/informativeness/config.json", "json"),
            ("prompts/judge/criteria/self_containedness/pt/prompt.md", "md"),
            ("prompts/judge/criteria/self_containedness/config.json", "json"),
        ],
    ),
    (
        "4. build-kg (triple extraction + concept generation)",
        "arandu build-kg",
        [
            ("prompts/kg/atlas/prompts.json", "json"),
            ("prompts/kg/atlas/concept_prompts.json", "json"),
            ("prompts/kg/atlas/metadata_labels.json", "json"),
            ("prompts/kg/atlas/schema.json", "json"),
        ],
    ),
    (
        "5. generate-non-answerable",
        "arandu generate-non-answerable",
        [("prompts/qa/non_answerable/pt.j2", "j2")],
    ),
    (
        "6. answer",
        "arandu answer",
        [("src/arandu/shared/rag/answer/prompts/answerer_pt.j2", "j2")],
    ),
    (
        "7. judge-answers",
        "arandu judge-answers",
        [
            ("prompts/judge/criteria/passage_coverage/pt/prompt.md", "md"),
            ("prompts/judge/criteria/passage_coverage/config.json", "json"),
            ("prompts/judge/criteria/abstention/pt/prompt.md", "md"),
            ("prompts/judge/criteria/abstention/config.json", "json"),
            ("prompts/judge/criteria/answer_correctness/pt/prompt.md", "md"),
            ("prompts/judge/criteria/answer_correctness/config.json", "json"),
            ("prompts/judge/criteria/answer_faithfulness/pt/prompt.md", "md"),
            ("prompts/judge/criteria/answer_faithfulness/config.json", "json"),
        ],
    ),
]


def main() -> int:
    """Assemble the review markdown from the stage manifest."""
    lines: list[str] = [
        "# Thesis run (`thesis-run-01`) — PT prompt review",
        "",
        "All Portuguese prompts + criteria configs the pipeline will use, in stage order.",
        "Prompts are baked into the cluster images at build time, so edit here -> "
        "redeploy `prompts/` -> rebuild -> run.",
        "",
        "---",
        "",
    ]
    missing: list[str] = []
    for title, cmd, files in STAGES:
        lines += [f"## {title}", "", f"`{cmd}`", ""]
        for path, kind in files:
            p = Path(path)
            lines += [f"### `{path}`", ""]
            if not p.exists():
                lines += ["> **MISSING**", ""]
                missing.append(path)
                continue
            text = p.read_text()
            if kind == "json":
                with contextlib.suppress(json.JSONDecodeError):
                    text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            fence = {"json": "json", "j2": "jinja"}.get(kind, "markdown")
            lines += [f"```{fence}", text.rstrip(), "```", ""]
        lines += ["---", ""]

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    if missing:
        print("MISSING:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
