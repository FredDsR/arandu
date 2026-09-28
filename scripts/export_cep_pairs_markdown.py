#!/usr/bin/env python3
"""Export every cep QA pair to a readable markdown for emic-validity review.

Walks ``results/<pipeline_id>/cep/outputs/*_cep_qa.json`` and renders one
markdown document grouped by **source interview -> chunk -> QA pair**. For each
chunk the source segment (``context``) is printed once, followed by every pair
generated from it: question, answer, reasoning trace and tacit inference (when
present), plus the judge evaluation scores and rationale.

Purpose: support the anthropological validation of emic readings (the
`anthropologist-validation-of-readings` gate). Unlike the blinded annotation
instrument, this dump exposes everything (bloom level, scores, reasoning) so
the reviewer can analyse the full corpus at once.

Reads raw JSON rather than ``QARecordCEP`` so legacy on-disk ``validation``
formats still render (matches the sibling ``cap_cep_pairs.py`` approach).

Usage:
    python3 scripts/export_cep_pairs_markdown.py [pipeline_id] [--output PATH] [--valid-only]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from arandu.utils.logger import print_error, print_info, print_success

CRITERIA = ("faithfulness", "bloom_calibration", "informativeness", "self_containedness")


def _short_id(filename: str) -> str:
    """Return the gdrive-id prefix of a ``*_cep_qa.json`` filename."""
    return filename.removesuffix("_cep_qa.json")


def _fmt_score(value: Any) -> str:
    """Format a numeric score to 2 decimals, or ``-`` when missing."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def _extract_validation(validation: dict[str, Any] | None) -> dict[str, Any]:
    """Pull overall/criterion scores and rationale from a ``validation`` dict.

    Tolerates the flat legacy layout (criteria as top-level keys) and the
    nested ``criterion_scores`` layout. Returns a normalised dict with the four
    named criteria, ``overall`` and ``rationale``.

    Args:
        validation: The raw ``validation`` payload from a QA pair, or None.

    Returns:
        Normalised score dict; values are None when unavailable.
    """
    out: dict[str, Any] = dict.fromkeys(CRITERIA)
    out["overall"] = None
    out["rationale"] = None
    if not validation:
        return out

    nested = validation.get("criterion_scores") or {}
    for crit in CRITERIA:
        if crit in validation and isinstance(validation[crit], (int, float)):
            out[crit] = validation[crit]
        elif crit in nested and isinstance(nested[crit], dict):
            out[crit] = nested[crit].get("score")
    out["overall"] = validation.get("overall_score")
    out["rationale"] = validation.get("judge_rationale")
    return out


def _render_pair(pair: dict[str, Any], index: int) -> list[str]:
    """Render a single QA pair as markdown lines.

    Args:
        pair: A raw QA-pair dict from ``qa_pairs``.
        index: 1-based position of the pair within its chunk.

    Returns:
        List of markdown lines (no trailing newline per line).
    """
    bloom = pair.get("bloom_level", "?")
    qtype = pair.get("question_type", "?")
    is_valid = pair.get("is_valid")
    valid_mark = {True: "valido", False: "rejeitado", None: "nao-julgado"}[is_valid]
    multi = ""
    if pair.get("is_multi_hop"):
        multi = f" | multi-hop ({pair.get('hop_count', '?')} hops)"

    v = _extract_validation(pair.get("validation"))
    lines = [
        f"#### Par {index} - bloom: `{bloom}` | tipo: `{qtype}` | {valid_mark}{multi}",
        "",
        f"**Pergunta:** {pair.get('question', '').strip()}",
        "",
        f"**Resposta:** {pair.get('answer', '').strip()}",
        "",
    ]

    tacit = pair.get("tacit_inference")
    if tacit:
        lines += [f"**Inferencia tacita:** {str(tacit).strip()}", ""]

    reasoning = pair.get("reasoning_trace")
    if reasoning:
        lines += [f"**Raciocinio (geracao):** {str(reasoning).strip()}", ""]

    score_line = (
        f"**Avaliacao:** overall {_fmt_score(v['overall'])} | "
        f"faithfulness {_fmt_score(v['faithfulness'])} | "
        f"bloom_calibration {_fmt_score(v['bloom_calibration'])} | "
        f"informativeness {_fmt_score(v['informativeness'])} | "
        f"self_containedness {_fmt_score(v['self_containedness'])}"
    )
    lines += [score_line, ""]

    if v["rationale"]:
        lines += [
            "<details><summary>Rationale do juiz</summary>",
            "",
            str(v["rationale"]).strip(),
            "",
            "</details>",
            "",
        ]
    return lines


def _render_source(record: dict[str, Any], valid_only: bool) -> tuple[list[str], int]:
    """Render one source file (interview) as markdown lines, grouped by chunk.

    Args:
        record: The parsed ``*_cep_qa.json`` document.
        valid_only: When True, skip pairs whose ``is_valid`` is not True.

    Returns:
        Tuple of (markdown lines, number of pairs rendered).
    """
    meta = record.get("source_metadata") or {}
    participant = meta.get("participant_name") or "(sem nome)"
    location = meta.get("location") or "(sem local)"
    rec_date = meta.get("recording_date") or "?"
    seq = meta.get("sequence_label") or ""
    event = meta.get("event_context") or ""
    short = _short_id(record.get("source_filename", "")) or record.get("source_file_id", "?")

    pairs = record.get("qa_pairs", [])
    if valid_only:
        pairs = [p for p in pairs if p.get("is_valid") is True]
    if not pairs:
        return [], 0

    header = f"## {participant} - {location}"
    if seq:
        header += f" ({seq})"
    lines = [
        header,
        "",
        f"_Fonte:_ `{short}` | _Data:_ {rec_date} | _Pares:_ {len(pairs)}",
        "",
    ]
    if event:
        lines += [f"_Contexto do evento:_ {event}", ""]

    # Group pairs by their source chunk (``context``), preserving order.
    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for p in pairs:
        ctx = p.get("context", "")
        if ctx not in chunks:
            order.append(ctx)
        chunks[ctx].append(p)

    for chunk_idx, ctx in enumerate(order, start=1):
        chunk_pairs = chunks[ctx]
        lines += [f"### Chunk {chunk_idx} ({len(chunk_pairs)} pares)", ""]
        quoted = "\n".join(f"> {ln}" for ln in ctx.strip().splitlines()) or "> (vazio)"
        lines += [quoted, ""]
        for i, p in enumerate(chunk_pairs, start=1):
            lines += _render_pair(p, i)

    lines += ["---", ""]
    return lines, len(pairs)


def main() -> None:
    """CLI entry point: render all cep pairs of a run to a markdown file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline_id", nargs="?", default="test-kg-04", help="Run id under results/"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: under the run dir)",
    )
    parser.add_argument(
        "--valid-only", action="store_true", help="Only include pairs with is_valid == True"
    )
    args = parser.parse_args()

    out_dir = Path("results") / args.pipeline_id / "cep" / "outputs"
    if not out_dir.is_dir():
        print_error(f"No cep outputs found: {out_dir}")
        raise SystemExit(1)

    files = sorted(out_dir.glob("*_cep_qa.json"))
    if not files:
        print_error(f"No *_cep_qa.json files in {out_dir}")
        raise SystemExit(1)

    output = args.output or (Path("results") / args.pipeline_id / "cep" / "emic_pairs_review.md")

    bloom_total: Counter[str] = Counter()
    body: list[str] = []
    total_pairs = 0
    rendered_pairs = 0
    for f in files:
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print_error(f"  skip {f.name[:22]}: {exc}")
            continue
        total_pairs += len(record.get("qa_pairs", []))
        for p in record.get("qa_pairs", []):
            if args.valid_only and p.get("is_valid") is not True:
                continue
            bloom_total[p.get("bloom_level", "?")] += 1
        src_lines, n = _render_source(record, args.valid_only)
        body += src_lines
        rendered_pairs += n

    scope = "apenas pares validos" if args.valid_only else "todos os pares"
    bloom_line = " | ".join(f"{lvl}: {bloom_total[lvl]}" for lvl in sorted(bloom_total))
    head = [
        f"# Pares CEP para revisao de validade emica - `{args.pipeline_id}`",
        "",
        f"Fontes: {len(files)} | Pares renderizados: {rendered_pairs} "
        f"(de {total_pairs} gerados) | Escopo: {scope}",
        "",
        f"Distribuicao Bloom: {bloom_line}",
        "",
        "Cada fonte -> chunk (segmento de origem, citado uma vez) -> pares gerados, "
        "com raciocinio e avaliacao do juiz.",
        "",
        "---",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(head + body), encoding="utf-8")
    print_info(f"Fontes: {len(files)} | pares: {rendered_pairs}/{total_pairs} | escopo: {scope}")
    print_success(f"Markdown escrito em {output}")


if __name__ == "__main__":
    main()
