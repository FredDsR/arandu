"""Retrieval-only comparison of BM25 against dense encoder x window grids (read-only).

For each dense configuration in ``DENSE_ARMS``, records the rank of the source
chunk in the top-10 retrieved chunks across three item sets:

- ``useful``: gate-approved CEP pairs above ``remember`` (excluding residual chunks).
- ``probes``: unanswerable probes seeded by ``useful`` pairs (excluding KG collisions).
- ``approved``: all gate-approved CEP pairs outside residual chunks.

Then splits ``useful`` pairs by lexical circularity (whether the question shares
a rare lemma with ``df <= RARE_DF`` with its source chunk) and computes paired
bootstrap CIs for ``hit@k(dense) - hit@k(bm25)``.

Two steps, both run from the repo root:

1. Export per-item ranks (requires sentence-transformer encoders; CPU suffices):

    uv run python -m analysis.analyze_dense_retrieval \\
        --id thesis-run-02 \\
        --replica results/judge-retest/r1-oracle \\
        --ranks \\
        --index-root results \\
        --ranks-file results/thesis-run-02/analysis/dense_retrieval_ranks.json

2. Compute summary tables and bootstrap CIs from the exported ranks:

    uv run python -m analysis.analyze_dense_retrieval \\
        --id thesis-run-02 \\
        --replica results/judge-retest/r1-oracle \\
        --ranks-file results/thesis-run-02/analysis/dense_retrieval_ranks.json \\
        --out-json results/thesis-run-02/analysis/dense_retrieval.json \\
        --out-md results/thesis-run-02/analysis/dense_retrieval.md
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

RARE_DF = 3
DENSE_ARMS = [
    "dense_mpnet128",
    "dense_bgem3_128",
    "dense_bgem3_256",
    "dense_bgem3_512",
    "dense_bgem3_whole",
    "dense_e5_128",
    "dense_e5_256",
    "dense_e5_512",
]
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20261005


def load_item_sets(run_dir: Path, replica_dir: Path) -> dict[str, list[Any]]:
    """Build the ``useful``, ``probes``, and ``approved`` retrieval item sets.

    Args:
        run_dir: Pipeline run directory (``results/<run-id>``).
        replica_dir: Canonical judge replica directory.

    Returns:
        Mapping from set name to list of ``RetrievalQuery`` items.
    """
    from analysis.analyze_judge_retest import (
        kg_collision_probes,
        load_cep,
        residual_pairs,
        useful_set,
    )
    from arandu.shared.rag.retrieve.loader import load_questions

    cep = load_cep(replica_dir / "cep" / "outputs")
    resid = residual_pairs(cep)
    approved = {p for p, m in cep.items() if m["approved"] and p not in resid}
    useful = useful_set(cep, approved)

    nonans_dataset = run_dir / "non_answerable" / "outputs" / "dataset.json"
    kg_graphml = run_dir / "kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml"
    excluded = kg_collision_probes(nonans_dataset, kg_graphml)

    questions = load_questions(run_dir / "cep" / "outputs", run_dir / "non_answerable" / "outputs")
    return {
        "useful": [q for q in questions if q.source == "cep" and q.qa_pair_id in useful],
        "probes": [
            q
            for q in questions
            if q.source != "cep" and q.qa_pair_id not in excluded and q.parent_qa_pair_id in useful
        ],
        "approved": [q for q in questions if q.source == "cep" and q.qa_pair_id in approved],
    }


def export_ranks(
    run_dir: Path,
    replica_dir: Path,
    index_root: Path,
    out_path: Path,
    arms: list[str] | None = None,
) -> None:
    """Compute top-10 source-chunk ranks for BM25 and dense arms and save to JSON.

    Args:
        run_dir: Pipeline run directory (``results/<run-id>``).
        replica_dir: Canonical judge replica directory.
        index_root: Root directory containing ``<run-id>/retrieve/indexes/``.
        out_path: Destination JSON file path.
        arms: Optional subset of dense arm presets (defaults to ``DENSE_ARMS``).
    """
    from arandu.shared.rag.retrieve.factory import (
        _build_chunk_resolver,
        _load_nonempty_chunks,
        dense_index_dir,
    )
    from arandu.shared.rag.retrieve.settings import DenseRetrieveSettings
    from arandu.shared.rag.retrievers.dense import DenseRetriever, SentenceTransformerEncoder

    selected_arms = arms or DENSE_ARMS
    groups = load_item_sets(run_dir, replica_dir)
    items = {q.qa_pair_id: q for group in groups.values() for q in group}
    gold = {pid: q.source_qa_pair_id.split(":")[1] for pid, q in items.items()}
    ids = list(items)
    res: dict[str, Any] = {
        "groups": {k: [q.qa_pair_id for q in g] for k, g in groups.items()},
        "gold": gold,
        "ranks": {},
        "models": {},
    }

    def _rank(chunk_ids: list[str], target: str) -> int | None:
        return chunk_ids.index(target) + 1 if target in chunk_ids else None

    bm25_ranks: dict[str, int | None] = {}
    for pid, q in items.items():
        path = run_dir / "retrieve/outputs/bm25" / q.source / f"{pid.replace(':', '__')}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        ranked = [p["chunk_id"] for p in sorted(doc["passages"], key=lambda p: p["rank"])]
        bm25_ranks[pid] = _rank(ranked, gold[pid])
    res["ranks"]["bm25"] = bm25_ranks

    base = DenseRetrieveSettings(_env_file=None)
    chunks = _load_nonempty_chunks(run_dir / "chunk" / "outputs" / "cep_4k", "cep_4k")
    resolver = _build_chunk_resolver(pipeline_id=run_dir.name, base_dir=run_dir.parent)
    enc: SentenceTransformerEncoder | None = None
    enc_model: str | None = None
    q_emb: np.ndarray | None = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for arm in selected_arms:
        cfg = base.for_arm(arm)
        if enc_model != cfg.model:
            enc = SentenceTransformerEncoder(
                cfg.model,
                query_prompt=cfg.query_prompt,
                passage_prompt=cfg.passage_prompt,
                batch_size=cfg.batch_size,
                device="cpu",
            )
            enc_model = cfg.model
            q_emb = enc.encode([items[pid].question for pid in ids], query=True)
            print(f"{cfg.model}: encoded {len(ids)} questions", flush=True)

        assert enc is not None and q_emb is not None
        index_dir = dense_index_dir(index_root, run_dir.name, cfg)
        if not (index_dir / "manifest.json").exists():
            DenseRetriever.build_index(
                chunks,
                resolver,
                index_dir,
                "cep_4k",
                enc,
                window_tokens=cfg.window_tokens,
                whole_chunk=cfg.whole_chunk,
                overlap_ratio=cfg.overlap_ratio,
            )
            print(f"{arm}: index built", flush=True)

        retriever = DenseRetriever(
            index_dir,
            enc,
            "cep_4k",
            window_tokens=cfg.window_tokens,
            whole_chunk=cfg.whole_chunk,
            overlap_ratio=cfg.overlap_ratio,
        )
        res["ranks"][arm] = {
            pid: _rank(
                [p.chunk_id for p in retriever.retrieve_by_embedding(q_emb[j], 10)], gold[pid]
            )
            for j, pid in enumerate(ids)
        }
        res["models"][arm] = {
            "model": cfg.model,
            "window_tokens": cfg.window_tokens,
            "whole_chunk": cfg.whole_chunk,
        }
        print(f"{arm}: done", flush=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")


def rare_lemma_flags(run_dir: Path, pids: list[str], rare_df: int = RARE_DF) -> dict[str, bool]:
    """Flag questions sharing at least one rare lemma (``df <= rare_df``) with their source chunk.

    Args:
        run_dir: Pipeline run directory.
        pids: QA pair IDs to check.
        rare_df: Document-frequency ceiling defining a rare lemma.

    Returns:
        Mapping ``qa_pair_id -> bool``.
    """
    from arandu.qa.schemas import iter_cep_pairs
    from arandu.shared.rag.retrievers._bm25_tokenize import portuguese_tokenizer

    question: dict[str, str] = {}
    chunk_text: dict[str, str] = {}
    for ref in iter_cep_pairs(run_dir / "cep" / "outputs", strict=True):
        question[ref.qa_pair_id] = ref.pair.question
        chunk_text.setdefault(ref.pair.chunk_id, ref.pair.context or "")

    tok = portuguese_tokenizer()
    lemmas = {cid: set(tok(txt)) for cid, txt in chunk_text.items()}
    doc_freq = collections.Counter(lem for s in lemmas.values() for lem in s)
    return {
        pid: any(
            doc_freq[lem] <= rare_df
            for lem in set(tok(question[pid])) & lemmas.get(pid.split(":")[1], set())
        )
        for pid in pids
    }


def compute_hits(ranks: np.ndarray) -> dict[str, float | int]:
    """Compute Hit@1, Hit@3, Hit@10, and MRR@10 from a rank array (0 = missed).

    Args:
        ranks: Integer array of ranks in ``1..10`` (or ``0`` when absent from top-10).

    Returns:
        Metrics dictionary including item count ``n``.
    """
    if len(ranks) == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "n": 0}
    metrics: dict[str, float | int] = {
        f"hit@{k}": float(((ranks >= 1) & (ranks <= k)).mean()) for k in (1, 3, 10)
    }
    metrics["mrr@10"] = float(np.where(ranks >= 1, 1.0 / np.maximum(ranks, 1), 0.0).mean())
    metrics["n"] = len(ranks)
    return metrics


def paired_hit_diff_ci(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    k: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> list[float]:
    """Paired bootstrap 95% CI for ``hit@k(ranks_a) - hit@k(ranks_b)``.

    Args:
        ranks_a: Rank array for arm A (0 = missed).
        ranks_b: Rank array for arm B (0 = missed).
        k: Cutoff rank.
        n_boot: Number of bootstrap draws.
        seed: Random generator seed.

    Returns:
        Two-element list ``[ci_low, ci_high]``.
    """
    ha = ((ranks_a >= 1) & (ranks_a <= k)).astype(float)
    hb = ((ranks_b >= 1) & (ranks_b <= k)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ranks_a), (n_boot, len(ranks_a)))
    diffs = (ha[idx] - hb[idx]).mean(axis=1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def analyze_ranks(
    ranks_path: Path,
    run_dir: Path,
    replica_dir: Path,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarize source-chunk retrieval metrics, rare-lemma splits, and CIs vs. BM25.

    Args:
        ranks_path: JSON file exported via ``--ranks``.
        run_dir: Pipeline run directory.
        replica_dir: Canonical judge replica directory.
        n_boot: Number of bootstrap draws.
        seed: Random generator seed.

    Returns:
        JSON-serializable analysis dictionary.
    """
    from analysis.analyze_judge_retest import load_cep, residual_pairs

    data = json.loads(ranks_path.read_text(encoding="utf-8"))
    resid = residual_pairs(load_cep(replica_dir / "cep" / "outputs"))
    data["groups"] = {
        group: [pid for pid in pids if pid not in resid and pid.split(":nonans")[0] not in resid]
        for group, pids in data["groups"].items()
    }
    arms = ["bm25", *[a for a in DENSE_ARMS if a in data["ranks"]]]

    def _arr(arm: str, pids: list[str]) -> np.ndarray:
        return np.array([data["ranks"][arm][pid] or 0 for pid in pids])

    out: dict[str, Any] = {
        "run_id": run_dir.name,
        "rare_df": RARE_DF,
        "n_boot": n_boot,
        "seed": seed,
        "models": data.get("models", {}),
        "groups": {},
        "rare": {},
        "vs_bm25": {},
    }
    for group, pids in data["groups"].items():
        out["groups"][group] = {arm: compute_hits(_arr(arm, pids)) for arm in arms}

    useful = data["groups"]["useful"]
    flags = rare_lemma_flags(run_dir, useful)
    with_rare = [p for p in useful if flags[p]]
    without_rare = [p for p in useful if not flags[p]]
    out["rare"]["share"] = len(with_rare) / max(1, len(useful))
    for arm in arms:
        out["rare"][arm] = {
            "with": compute_hits(_arr(arm, with_rare)),
            "without": compute_hits(_arr(arm, without_rare)),
        }

    for arm in arms[1:]:
        out["vs_bm25"][arm] = {}
        for name, pids in (
            ("useful", useful),
            ("with_rare", with_rare),
            ("without_rare", without_rare),
            ("probes", data["groups"]["probes"]),
        ):
            x, y = _arr(arm, pids), _arr("bm25", pids)
            out["vs_bm25"][arm][name] = {
                f"hit@{k}": {
                    "diff": float(((x >= 1) & (x <= k)).mean() - ((y >= 1) & (y <= k)).mean()),
                    "ci": paired_hit_diff_ci(x, y, k, n_boot=n_boot, seed=seed),
                }
                for k in (1, 10)
            }
    return out


def render_markdown(res: dict[str, Any]) -> str:
    """Render the dense retrieval comparison as Markdown tables.

    Args:
        res: Analysis dictionary from :func:`analyze_ranks`.

    Returns:
        Markdown-formatted report.
    """
    lines = [
        f"# Retrieval-Only Comparison: BM25 vs. Dense Configurations ({res['run_id']})",
        "",
        "Source-chunk rank within top-10 retrieved chunks.",
        "",
    ]
    for group, rows in res["groups"].items():
        n = next(iter(rows.values()))["n"]
        lines += [
            f"## {group} (n={n})",
            "",
            "| Arm | Hit@1 | Hit@3 | Hit@10 | MRR@10 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, h in rows.items():
            lines.append(
                f"| {arm} | {h['hit@1']:.3f} | {h['hit@3']:.3f} "
                f"| {h['hit@10']:.3f} | {h['mrr@10']:.3f} |"
            )
        lines.append("")

    lines += [
        f"## Term Circularity (useful pairs; rare lemma df <= {res['rare_df']}; "
        f"{res['rare']['share']:.1%} with rare lemma)",
        "",
        "| Arm | Hit@1 (with) | Hit@10 (with) | Hit@1 (without) | Hit@10 (without) |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, v in res["rare"].items():
        if arm == "share":
            continue
        lines.append(
            f"| {arm} | {v['with']['hit@1']:.3f} | {v['with']['hit@10']:.3f} "
            f"| {v['without']['hit@1']:.3f} | {v['without']['hit@10']:.3f} |"
        )

    lines += [
        "",
        f"## Difference vs. BM25 (paired item bootstrap, B={res['n_boot']})",
        "",
        "| Arm | Subset | Hit@1 diff | 95% CI | Hit@10 diff | 95% CI |",
        "|---|---|---:|---|---:|---|",
    ]
    for arm, sets in res["vs_bm25"].items():
        for subset, v in sets.items():
            h1, h10 = v["hit@1"], v["hit@10"]
            lines.append(
                f"| {arm} | {subset} | {h1['diff']:+.3f} "
                f"| [{h1['ci'][0]:+.3f}, {h1['ci'][1]:+.3f}] "
                f"| {h10['diff']:+.3f} "
                f"| [{h10['ci'][0]:+.3f}, {h10['ci'][1]:+.3f}] |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point for dense vs. BM25 retrieval-only analysis."""
    parser = argparse.ArgumentParser(
        description="Compare BM25 against dense encoder x window retrieval configurations."
    )
    parser.add_argument("--id", required=True, help="Pipeline run ID (e.g. thesis-run-02).")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Results root.")
    parser.add_argument(
        "--replica",
        type=Path,
        required=True,
        help="Canonical judge-retest replica directory (e.g. results/judge-retest/r1-oracle).",
    )
    parser.add_argument(
        "--ranks",
        action="store_true",
        help="Export per-item top-10 source-chunk ranks instead of analyzing an existing file.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Root holding <run-id>/retrieve/indexes/ (defaults to --results-root).",
    )
    parser.add_argument(
        "--ranks-file",
        type=Path,
        required=True,
        help="Path to write (--ranks) or read (analysis mode) the per-item ranks JSON.",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap draws.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap RNG seed.")
    args = parser.parse_args()

    run_dir = args.results_root / args.id
    if args.ranks:
        export_ranks(run_dir, args.replica, args.index_root or args.results_root, args.ranks_file)
        return

    res = analyze_ranks(args.ranks_file, run_dir, args.replica, n_boot=args.n_boot, seed=args.seed)
    md = render_markdown(res)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
