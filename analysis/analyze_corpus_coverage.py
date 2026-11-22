"""Corpus coverage, graph representation, and stratified retrieval analysis (read-only).

Evaluates how uneven oral-interview corpora propagate through benchmark
generation, knowledge graph induction, and retrieval evaluation:

1. Interviews: useful-pair yield by interview length, chunk count, length
   quartiles, Gini inequality, and non-contributing interviews.
2. Chunks: useful-pair yield across chunk length bands and sole rejection
   causes in zero-yield vs. contributing chunks.
3. Communities x Bloom levels: approval rates, rejection causes, and emic
   validity (EV >= 4) across communities.
4. Graph: passage matching, entity/event node counts, relation edges, and
   per-community graph density.
5. Retrieval across the corpus: retrieval metrics stratified by community,
   interview-length quartile, and emic validity gate.
6. Bootstraps: paired item- and interview-cluster bootstraps for KC and EKC
   differences, pooled length-band bootstraps across judge retest replicas,
   evidence-support abstention decomposition, BM25 source-chunk circularity,
   and community EV >= 4 stability.

Run from the repo root:

    uv run python -m analysis.analyze_corpus_coverage \\
        --id thesis-run-02 \\
        --replica results/judge-retest/r1-oracle \\
        --emic-replica results/judge-retest/r2 \\
        --emic-replica results/judge-retest/r3 \\
        --retest-replica results/judge-retest/r2 \\
        --retest-replica results/judge-retest/r3=results/judge-retest/r3-gate \\
        --out-json results/thesis-run-02/analysis/corpus_coverage.json \\
        --out-md results/thesis-run-02/analysis/corpus_coverage.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import unicodedata
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from analysis.analyze_judge_retest import (
    GATE_CRITERIA,
    HIGHER,
    TAU,
    TAU_EMIC,
    arm_aggregates,
    kg_collision_probes,
    load_emic,
    load_replica,
    residual_pairs,
    useful_set,
)

ARMS = [
    "bm25",
    "dense_bgem3_256",
    "dense_mpnet128",
    "khop_passage",
    "khop_triple",
    "atlas_rag",
    "oracle",
]
COMMUNITY_NAMES = {
    "BARRA DE PELOTAS": "Barra de Pelotas",
    "DOQUINHAS": "Doquinhas",
    "BALSA": "Balsa",
}
UNSPECIFIED_COMMUNITY = "Unspecified"
COMMUNITIES = ["Barra de Pelotas", "Doquinhas", "Balsa", UNSPECIFIED_COMMUNITY]
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20261005


def fold_text(text: str) -> str:
    """Strip diacritics and uppercase a string for robust location matching.

    Args:
        text: Input string.

    Returns:
        Normalized uppercase ASCII-folded string.
    """
    return (
        "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
        .upper()
        .strip()
    )


def map_community(location: str | None) -> str:
    """Map a raw location string to one of the canonical community labels.

    Args:
        location: Raw location field from ``source_metadata``.

    Returns:
        Canonical community name or ``"Unspecified"``.
    """
    if not location:
        return UNSPECIFIED_COMMUNITY
    folded = fold_text(location)
    for key, label in COMMUNITY_NAMES.items():
        if key in folded:
            return label
    return UNSPECIFIED_COMMUNITY


def spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation with average-rank tie handling.

    Args:
        x: First numeric sequence.
        y: Second numeric sequence of equal length.

    Returns:
        Spearman rank correlation coefficient.
    """

    def _ranks(values: list[float]) -> np.ndarray:
        arr = np.array(values, dtype=float)
        order = arr.argsort(kind="stable")
        ranked = np.empty(len(arr))
        ranked[order] = np.arange(len(arr), dtype=float)
        for unique_val in np.unique(arr):
            mask = arr == unique_val
            ranked[mask] = ranked[mask].mean()
        return ranked

    if len(x) < 2:
        return math.nan
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def gini(values: list[float]) -> float:
    """Compute the Gini coefficient of a non-negative sequence.

    Args:
        values: Non-negative numeric sequence.

    Returns:
        Gini coefficient in ``[0, 1]``, or ``NaN`` when the sum is zero.
    """
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return math.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * arr.sum()))


def load_interviews(cep_dir: Path) -> dict[str, dict[str, Any]]:
    """Load interview lengths, texts, and communities from CEP output JSONs.

    Args:
        cep_dir: Directory ``results/<run-id>/cep/outputs/``.

    Returns:
        Mapping from source file ID to ``chars``, ``text``, and ``community``.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(cep_dir.glob("*_cep_qa.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        source_id = doc.get("source_file_id") or doc["source_gdrive_id"]
        text = doc.get("transcription_text") or ""
        out[source_id] = {
            "chars": len(text),
            "text": text,
            "community": map_community((doc.get("source_metadata") or {}).get("location")),
        }
    return out


def sole_cause(scores: dict[str, float | None]) -> str:
    """Identify the sole criterion rejecting a pair, ``'2+'``, or ``'pass'``.

    Args:
        scores: Mapping from criterion name to score (or ``None`` on judge error).

    Returns:
        Criterion name, ``"2+"`` for multiple failures, or ``"pass"``.
    """
    failed = [c for c in GATE_CRITERIA if scores.get(c) is not None and scores[c] < TAU]
    missing = [c for c in GATE_CRITERIA if c in scores and scores[c] is None]
    failed.extend(missing)
    if not failed:
        return "pass"
    return failed[0] if len(failed) == 1 else "2+"


def interview_section(
    cep: dict[str, dict[str, Any]],
    approved: set[str],
    interviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute interview-level benchmark yield, quartiles, and concentration metrics.

    Args:
        cep: CEP pair metadata mapping loaded by ``load_replica``.
        approved: Set of gate-approved QA pair IDs.
        interviews: Mapping from :func:`load_interviews`.

    Returns:
        Interview coverage summary dictionary.
    """
    resid = residual_pairs(cep)
    base = {p: m for p, m in cep.items() if p not in resid}
    useful = useful_set(base, approved & set(base))
    chunks: dict[str, set[str]] = collections.defaultdict(set)
    useful_by_file: collections.Counter[str] = collections.Counter()
    for meta in base.values():
        chunks[meta["file"]].add(meta["chunk"])
    for pid in useful:
        useful_by_file[base[pid]["file"]] += 1

    rows = [
        {
            "file": fid,
            "community": iv["community"],
            "chars": iv["chars"],
            "chunks": len(chunks.get(fid, ())),
            "useful": useful_by_file.get(fid, 0),
        }
        for fid, iv in interviews.items()
    ]
    contrib = [r for r in rows if r["useful"] > 0]
    zero = [r for r in rows if r["useful"] == 0]
    single = [r for r in rows if r["chunks"] == 1]
    multi = [r for r in rows if r["chunks"] > 1]

    def _yield_per_chunk(subset: list[dict[str, Any]]) -> float:
        total_chunks = sum(r["chunks"] for r in subset)
        return sum(r["useful"] for r in subset) / total_chunks if total_chunks else math.nan

    lengths = sorted(r["chars"] for r in rows)
    cuts = [lengths[int(len(lengths) * k / 4)] for k in (1, 2, 3)]

    def _band(chars: int) -> int:
        return 1 + sum(chars >= threshold for threshold in cuts)

    total_chars = max(1, sum(lengths))
    total_useful = max(1, len(useful))
    bands: dict[str, dict[str, Any]] = {}
    for band_idx in (1, 2, 3, 4):
        subset = [r for r in rows if _band(r["chars"]) == band_idx]
        bands[f"Q{band_idx}"] = {
            "n": len(subset),
            "chars_min": min((r["chars"] for r in subset), default=0),
            "chars_max": max((r["chars"] for r in subset), default=0),
            "chars_share": sum(r["chars"] for r in subset) / total_chars,
            "chunks": sum(r["chunks"] for r in subset),
            "useful": sum(r["useful"] for r in subset),
            "useful_share": sum(r["useful"] for r in subset) / total_useful,
            "contributing": sum(r["useful"] > 0 for r in subset),
            "yield_per_chunk": _yield_per_chunk(subset),
        }

    return {
        "rows": rows,
        "n_interviews": len(rows),
        "n_useful": len(useful),
        "contributing": len(contrib),
        "zero": len(zero),
        "zero_median_chars": statistics.median(r["chars"] for r in zero) if zero else math.nan,
        "contrib_median_chars": (
            statistics.median(r["chars"] for r in contrib) if contrib else math.nan
        ),
        "zero_single_chunk": sum(r["chunks"] == 1 for r in zero),
        "spearman_chars_useful": spearman(
            [float(r["chars"]) for r in rows], [float(r["useful"]) for r in rows]
        ),
        "spearman_chars_useful_per_chunk": (
            spearman(
                [float(r["chars"]) for r in rows],
                [r["useful"] / r["chunks"] for r in rows],
            )
            if all(r["chunks"] for r in rows)
            else math.nan
        ),
        "single_chunk": {
            "n": len(single),
            "yield_per_chunk": _yield_per_chunk(single),
            "contributing": sum(r["useful"] > 0 for r in single),
        },
        "multi_chunk": {
            "n": len(multi),
            "yield_per_chunk": _yield_per_chunk(multi),
            "contributing": sum(r["useful"] > 0 for r in multi),
        },
        "gini_chars": gini([float(r["chars"]) for r in rows]),
        "gini_useful": gini([float(r["useful"]) for r in rows]),
        "length_quartile_cuts": cuts,
        "length_bands": bands,
        "useful": sorted(useful),
        "band_of_file": {r["file"]: f"Q{_band(r['chars'])}" for r in rows},
    }


def chunk_section(cep: dict[str, dict[str, Any]], approved: set[str]) -> dict[str, Any]:
    """Summarize useful-pair counts and rejection causes across chunks.

    Args:
        cep: CEP pair metadata mapping.
        approved: Set of gate-approved QA pair IDs.

    Returns:
        Chunk yield and rejection breakdown dictionary.
    """
    resid = residual_pairs(cep)
    base = {p: m for p, m in cep.items() if p not in resid}
    by_chunk: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for pid, meta in base.items():
        by_chunk[(meta["file"], meta["chunk"])].append(pid)

    dist: collections.Counter[int] = collections.Counter()
    len_bins: dict[str, list[int]] = {"<1k": [], "1-2k": [], "2-3k": [], "3-4k": []}
    causes_zero: collections.Counter[str] = collections.Counter()
    causes_contrib: collections.Counter[str] = collections.Counter()

    for pids in by_chunk.values():
        cands = [p for p in pids if base[p]["bloom"] in HIGHER]
        n_useful = sum(p in approved for p in cands)
        dist[n_useful] += 1
        clen = base[pids[0]]["ctx_len"]
        if clen < 1000:
            band = "<1k"
        elif clen < 2000:
            band = "1-2k"
        elif clen < 3000:
            band = "2-3k"
        else:
            band = "3-4k"
        len_bins[band].append(n_useful)
        target = causes_zero if n_useful == 0 else causes_contrib
        for pid in cands:
            if pid not in approved:
                target[sole_cause(base[pid]["scores"])] += 1

    return {
        "n_chunks": len(by_chunk),
        "useful_per_chunk_dist": dict(sorted(dist.items())),
        "by_length": {
            band: {
                "n": len(vals),
                "mean_useful": statistics.mean(vals) if vals else math.nan,
                "share_contributing": sum(x > 0 for x in vals) / len(vals) if vals else math.nan,
            }
            for band, vals in len_bins.items()
        },
        "rejections_in_zero_chunks": dict(causes_zero.most_common()),
        "rejections_in_contributing_chunks": dict(causes_contrib.most_common()),
    }


def community_section(
    cep: dict[str, dict[str, Any]],
    approved: set[str],
    interviews: dict[str, dict[str, Any]],
    emic: dict[str, float | None],
) -> dict[str, Any]:
    """Break down approval, Bloom levels, rejection causes, and EV >= 4 by community.

    Args:
        cep: CEP pair metadata mapping.
        approved: Set of gate-approved QA pair IDs.
        interviews: Mapping from :func:`load_interviews`.
        emic: Mapping from pair ID to emic validity score.

    Returns:
        Per-community summary dictionary.
    """
    resid = residual_pairs(cep)
    base = {p: m for p, m in cep.items() if p not in resid}
    out: dict[str, Any] = {}
    for comm in COMMUNITIES:
        pids = [p for p, m in base.items() if interviews[m["file"]]["community"] == comm]
        cands = [p for p in pids if base[p]["bloom"] in HIGHER]
        useful = [p for p in cands if p in approved]
        rem = [p for p in pids if base[p]["bloom"] == "remember"]
        by_level: dict[str, dict[str, Any]] = {}
        for lvl in HIGHER:
            cand_lvl = [p for p in cands if base[p]["bloom"] == lvl]
            useful_lvl = [p for p in cand_lvl if p in approved]
            by_level[lvl] = {
                "candidates": len(cand_lvl),
                "useful": len(useful_lvl),
                "rate": len(useful_lvl) / len(cand_lvl) if cand_lvl else math.nan,
            }

        def _ev_share(pair_ids: list[str]) -> tuple[float, int]:
            vals = [emic[p] for p in pair_ids if emic.get(p) is not None]
            return ((sum(e >= TAU_EMIC for e in vals) / len(vals)) if vals else math.nan, len(vals))

        causes = collections.Counter(
            sole_cause(base[p]["scores"]) for p in cands if p not in approved
        )
        out[comm] = {
            "pairs": len(pids),
            "remember_pass": sum(p in approved for p in rem) / len(rem) if rem else math.nan,
            "candidates": len(cands),
            "useful": len(useful),
            "useful_rate": len(useful) / len(cands) if cands else math.nan,
            "by_level": by_level,
            "ev_ge4_remember": _ev_share(rem),
            "ev_ge4_candidates": _ev_share(cands),
            "ev_ge4_useful": _ev_share(useful),
            "rejection_causes": {
                k: v / len(cands) for k, v in causes.most_common() if len(cands) > 0
            },
        }
    return out


def graph_section(run_dir: Path, interviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Measure how each interview and community is represented in the induced KG.

    Args:
        run_dir: Pipeline run directory (``results/<run-id>``).
        interviews: Mapping from :func:`load_interviews`.

    Returns:
        Graph coverage metrics per interview and per community.
    """
    graphml = run_dir / "kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml"
    graph = nx.read_graphml(graphml)
    node_type = {n: d.get("type") for n, d in graph.nodes(data=True)}

    file_of_passage: dict[str, str] = {}
    unmatched = 0
    norm_text = {fid: " ".join(iv["text"].split()) for fid, iv in interviews.items()}
    for node, data in graph.nodes(data=True):
        if node_type[node] != "passage":
            continue
        body = data.get("id", "")
        # AutoSchemaKG passage nodes prepend metadata before "[Transcrição]".
        body = " ".join(body.split("[Transcrição]", 1)[-1].split())
        probe = body[:80]
        hits = [fid for fid, norm in norm_text.items() if probe and probe in norm]
        if len(hits) == 1:
            file_of_passage[node] = hits[0]
        else:
            unmatched += 1

    files_of_node: dict[str, set[str]] = collections.defaultdict(set)
    for u, v, data in graph.edges(data=True):
        if data.get("type") == "Source" and v in file_of_passage:
            files_of_node[u].add(file_of_passage[v])

    ent_ev = [n for n in graph if node_type[n] in ("entity", "event")]
    per_file: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"nodes": 0, "rel_edges": 0, "passages": 0}
    )
    for fid in file_of_passage.values():
        per_file[fid]["passages"] += 1
    for node in ent_ev:
        for fid in files_of_node.get(node, ()):
            per_file[fid]["nodes"] += 1

    cross = 0
    rel_total = 0
    for u, v, data in graph.edges(data=True):
        if data.get("type") != "Relation":
            continue
        rel_total += 1
        fu, fv = files_of_node.get(u, set()), files_of_node.get(v, set())
        shared = fu & fv
        if shared:
            for fid in shared:
                per_file[fid]["rel_edges"] += 1
        elif fu and fv:
            cross += 1

    rows = [
        {
            "file": fid,
            "community": iv["community"],
            "chars": iv["chars"],
            **per_file.get(fid, {"nodes": 0, "rel_edges": 0, "passages": 0}),
        }
        for fid, iv in interviews.items()
    ]
    total_nodes = max(1, sum(r["nodes"] for r in rows))
    total_chars = max(1, sum(r["chars"] for r in rows))
    by_comm: dict[str, dict[str, Any]] = {}
    for comm in COMMUNITIES:
        subset = [r for r in rows if r["community"] == comm]
        kchars = sum(r["chars"] for r in subset) / 1000.0
        by_comm[comm] = {
            "interviews": len(subset),
            "nodes_per_kchar": sum(r["nodes"] for r in subset) / kchars if kchars else math.nan,
            "rel_edges_per_kchar": (
                sum(r["rel_edges"] for r in subset) / kchars if kchars else math.nan
            ),
            "nodes_share": sum(r["nodes"] for r in subset) / total_nodes,
            "chars_share": sum(r["chars"] for r in subset) / total_chars,
        }

    return {
        "passages": sum(1 for n in graph if node_type[n] == "passage"),
        "passages_matched": len(file_of_passage),
        "passages_unmatched": unmatched,
        "interviews_with_passage": sum(r["passages"] > 0 for r in rows),
        "interviews_with_nodes": sum(r["nodes"] > 0 for r in rows),
        "interviews_without_relation_edges": sum(r["rel_edges"] == 0 for r in rows),
        "entity_event_nodes": len(ent_ev),
        "entity_event_nodes_in_2plus_interviews": sum(
            len(files_of_node.get(n, ())) > 1 for n in ent_ev
        ),
        "relation_edges": rel_total,
        "relation_edges_within_interview_share": (
            sum(r["rel_edges"] for r in rows) / rel_total if rel_total else math.nan
        ),
        "relation_edges_cross_interview": cross,
        "spearman_chars_nodes": spearman(
            [float(r["chars"]) for r in rows], [float(r["nodes"]) for r in rows]
        ),
        "nodes_per_kchar_median": statistics.median(
            r["nodes"] / (r["chars"] / 1000.0) for r in rows if r["chars"]
        ),
        "gini_nodes": gini([float(r["nodes"]) for r in rows]),
        "by_community": by_comm,
        "rows": rows,
    }


def retrieval_section(
    rep: dict[str, Any], useful: set[str], groups: dict[str, set[str]]
) -> dict[str, Any]:
    """Compute per-arm retrieval metrics restricted to each item subset.

    Args:
        rep: Loaded replica dictionary.
        useful: Useful pair IDs.
        groups: Named subsets of pair IDs.

    Returns:
        Nested dictionary ``group -> arm -> metrics``.
    """
    recs = rep["answers"]["records"]
    cep = rep["cep"]
    arms = [a for a in ARMS if any(k[0] == a for k in recs)]
    metric_keys = (
        "n_answerable",
        "n_nonans",
        "kc",
        "kc_n",
        "oc",
        "hall",
        "f1_abs",
        "pc_pass",
        "cond_oc",
        "cond_oc_n",
    )
    out: dict[str, Any] = {}
    for name, ids in groups.items():
        agg = arm_aggregates(recs, cep, useful & ids, arms=arms)
        out[name] = {arm: {k: agg[arm][k] for k in metric_keys} for arm in arms}
    return out


def emic_stability_section(
    cep: dict[str, dict[str, Any]],
    approved: set[str],
    interviews: dict[str, dict[str, Any]],
    emic_reps: list[dict[str, float | None]],
) -> dict[str, Any]:
    """Compute EV >= 4 share per community across emic-judge retest replicas.

    Args:
        cep: CEP pair metadata mapping.
        approved: Set of gate-approved QA pair IDs.
        interviews: Mapping from :func:`load_interviews`.
        emic_reps: List of emic score mappings, one per replica.

    Returns:
        Per-community EV >= 4 shares across replicas for useful and candidate sets.
    """
    resid = residual_pairs(cep)
    base = {p: m for p, m in cep.items() if p not in resid}
    useful = useful_set(base, approved & set(base))
    cands = {p for p in base if base[p]["bloom"] in HIGHER}
    out: dict[str, Any] = {}
    for comm in COMMUNITIES:
        row: dict[str, list[float]] = {}
        for name, ids in (("useful", useful), ("candidates", cands)):
            ps = [p for p in ids if interviews[base[p]["file"]]["community"] == comm]
            vals: list[float] = []
            for em in emic_reps:
                scored = [em[p] for p in ps if em.get(p) is not None]
                share = sum(e >= TAU_EMIC for e in scored) / len(scored) if scored else math.nan
                vals.append(share)
            row[name] = vals
        out[comm] = row
    return out


def _kc_mean(recs_kc: np.ndarray) -> float:
    return float(np.nanmean(recs_kc)) if np.isfinite(recs_kc).any() else math.nan


def _kc_array(rep: dict[str, Any], arm: str, pids: list[str]) -> np.ndarray:
    facts = rep["answers"]["facts"]
    out = np.full(len(pids), np.nan)
    for j, pid in enumerate(pids):
        fact = facts.get((arm, pid))
        if (
            fact
            and fact["label"] == "TC"
            and fact["answer_correctness"] is not None
            and fact["answer_faithfulness"] is not None
        ):
            out[j] = fact["answer_correctness"] * fact["answer_faithfulness"]
    return out


def kc_diff_bootstrap(
    rep: dict[str, Any],
    pids: list[str],
    arm_a: str,
    arm_b: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    cluster: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Paired bootstrap CI for ``KC(arm_a) - KC(arm_b)`` over items or clusters.

    Args:
        rep: Loaded replica dictionary.
        pids: Pair IDs to evaluate.
        arm_a: First retrieval arm.
        arm_b: Second retrieval arm.
        n_boot: Number of bootstrap replicates.
        seed: Random generator seed.
        cluster: Optional mapping ``pid -> cluster_id`` for cluster resampling.

    Returns:
        Point estimate, 95% percentile CI, item count, and TC counts.
    """
    arr_a, arr_b = _kc_array(rep, arm_a, pids), _kc_array(rep, arm_b, pids)
    est = _kc_mean(arr_a) - _kc_mean(arr_b)
    rng = np.random.default_rng(seed)
    if cluster is None:
        groups = [np.array([j]) for j in range(len(pids))]
    else:
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for j, pid in enumerate(pids):
            grouped[cluster[pid]].append(j)
        groups = [np.array(idxs) for idxs in grouped.values()]

    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        boots[i] = _kc_mean(arr_a[idx]) - _kc_mean(arr_b[idx])
    boots = boots[np.isfinite(boots)]
    return {
        "est": est,
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": len(pids),
        "tc_a": int(np.isfinite(arr_a).sum()),
        "tc_b": int(np.isfinite(arr_b).sum()),
    }


def evsup_strata(rep: dict[str, Any], pids: list[str]) -> dict[str, Any]:
    """Decompose over-caution into lacking evidence vs. declining sufficient evidence.

    Args:
        rep: Loaded replica dictionary.
        pids: Answerable pair IDs.

    Returns:
        Per-arm breakdown satisfying ``oc_lacking + oc_declining == oc`` and
        ``ekc == kc * (1 - oc)``.
    """
    facts = rep["answers"]["facts"]
    out: dict[str, Any] = {}
    for arm in ARMS:
        rows: list[dict[str, Any]] = []
        for pid in pids:
            fact = facts.get((arm, pid))
            if not fact or fact["label"] not in ("TC", "FA"):
                continue
            ev = fact["passage_coverage"]
            kc = (
                fact["answer_correctness"] * fact["answer_faithfulness"]
                if fact["label"] == "TC"
                and fact["answer_correctness"] is not None
                and fact["answer_faithfulness"] is not None
                else None
            )
            rows.append(
                {
                    "suff": ev is not None and ev >= TAU,
                    "ev_missing": ev is None,
                    "tc": fact["label"] == "TC",
                    "kc": kc,
                }
            )
        if not rows:
            continue
        n = len(rows)

        def _stratum(subset: list[dict[str, Any]]) -> dict[str, Any]:
            tc_rows = [r for r in subset if r["tc"]]
            kcs = [r["kc"] for r in tc_rows if r["kc"] is not None]
            return {
                "n": len(subset),
                "tc": len(tc_rows),
                "fa": len(subset) - len(tc_rows),
                "oc": (len(subset) - len(tc_rows)) / len(subset) if subset else math.nan,
                "kc": statistics.mean(kcs) if kcs else math.nan,
                "kc_n": len(kcs),
            }

        suff = _stratum([r for r in rows if r["suff"]])
        insuff = _stratum([r for r in rows if not r["suff"]])
        all_rows = _stratum(rows)
        out[arm] = {
            "n": n,
            "ev_missing": sum(r["ev_missing"] for r in rows),
            "share_sufficient": suff["n"] / n,
            "sufficient": suff,
            "insufficient": insuff,
            "kc": all_rows["kc"],
            "oc": all_rows["oc"],
            "oc_lacking": insuff["fa"] / n,
            "oc_declining": suff["fa"] / n,
            "ekc": all_rows["kc"] * (1.0 - all_rows["oc"]),
        }
    return out


def bm25_source_section(
    rep: dict[str, Any], run_dir: Path, pids: list[str], probes: list[str]
) -> dict[str, Any]:
    """Measure BM25 source-chunk rank and lexical circularity on useful pairs and probes.

    Args:
        rep: Loaded replica dictionary.
        run_dir: Pipeline run directory.
        pids: Useful pair IDs.
        probes: Unanswerable probe IDs.

    Returns:
        Dictionary of source-chunk hit rates, rare-lemma stratification, and
        rank-stratified BM25 vs. oracle metrics.
    """
    from arandu.qa.schemas import iter_cep_pairs
    from arandu.shared.rag.retrievers._bm25_tokenize import portuguese_tokenizer

    rare_df = 3
    facts = rep["answers"]["facts"]
    out_dir = run_dir / "retrieve" / "outputs" / "bm25"
    question: dict[str, str] = {}
    chunk_text: dict[str, str] = {}
    for ref in iter_cep_pairs(run_dir / "cep" / "outputs", strict=True):
        question[ref.qa_pair_id] = ref.pair.question
        chunk_text.setdefault(ref.pair.chunk_id, ref.pair.context or "")

    tok = portuguese_tokenizer()
    chunk_lemmas = {cid: set(tok(txt)) for cid, txt in chunk_text.items()}
    doc_freq = collections.Counter(lem for lemmas in chunk_lemmas.values() for lem in lemmas)

    def _ranking(pid: str) -> list[str]:
        sub = "nonanswerable" if pid.endswith(":nonans") else "cep"
        path = out_dir / sub / f"{pid.replace(':', '__')}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        return [p["chunk_id"] for p in sorted(doc["passages"], key=lambda x: x["rank"])]

    rows: list[dict[str, Any]] = []
    for pid in pids:
        ids = _ranking(pid)
        src = pid.split(":")[1]
        rank = ids.index(src) + 1 if src in ids else None
        q_lemmas = set(tok(question[pid]))
        other = next((cid for cid in ids if cid != src), None)
        src_overlap = q_lemmas & chunk_lemmas.get(src, set())
        rows.append(
            {
                "pid": pid,
                "rank": rank,
                "q_lemmas": len(q_lemmas),
                "share_src": len(src_overlap) / len(q_lemmas) if q_lemmas else math.nan,
                "share_other": (
                    len(q_lemmas & chunk_lemmas.get(other, set())) / len(q_lemmas)
                    if q_lemmas and other
                    else math.nan
                ),
                "rare_src": any(doc_freq[lem] <= rare_df for lem in src_overlap),
            }
        )

    n = len(rows)

    def _hit(k: int) -> float:
        return sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n if n else math.nan

    def _arm_stats(arm: str, pair_ids: list[str]) -> dict[str, Any]:
        fs = [facts[(arm, p)] for p in pair_ids if (arm, p) in facts]
        tc = [f for f in fs if f["label"] == "TC"]
        kcs = [
            f["answer_correctness"] * f["answer_faithfulness"]
            for f in tc
            if f["answer_correctness"] is not None and f["answer_faithfulness"] is not None
        ]
        cnt = sum(f["label"] in ("TC", "FA") for f in fs)
        ev = [f["passage_coverage_pass"] for f in fs if f["passage_coverage_pass"] is not None]
        kc = statistics.mean(kcs) if kcs else math.nan
        oc = 1.0 - len(tc) / cnt if cnt else math.nan
        return {
            "n": len(pair_ids),
            "tc": len(tc),
            "kc": kc,
            "oc": oc,
            "ekc": kc * (1.0 - oc),
            "evsup": statistics.mean(ev) if ev else math.nan,
        }

    strata = {
        "rank 1": lambda r: r == 1,
        "rank 2-10": lambda r: r is not None and r > 1,
        "not retrieved": lambda r: r is None,
    }
    by_rank: dict[str, dict[str, Any]] = {}
    for name, predicate in strata.items():
        ps = [r["pid"] for r in rows if predicate(r["rank"])]
        by_rank[name] = {arm: _arm_stats(arm, ps) for arm in ("bm25", "oracle")}
        by_rank[name]["pids"] = ps

    rare = [r for r in rows if r["rare_src"]]
    no_rare = [r for r in rows if not r["rare_src"]]

    def _hit1(rs: list[dict[str, Any]]) -> float:
        return sum(r["rank"] == 1 for r in rs) / len(rs) if rs else math.nan

    def _hit10(rs: list[dict[str, Any]]) -> float:
        return sum(r["rank"] is not None for r in rs) / len(rs) if rs else math.nan

    probe_ranks: list[int | None] = []
    for pid in probes:
        ids = _ranking(pid)
        src = pid.split(":")[1]
        probe_ranks.append(ids.index(src) + 1 if src in ids else None)

    return {
        "n": n,
        "chunks_with_text": len(chunk_text),
        "rare_df": rare_df,
        "hit_at": {k: _hit(k) for k in (1, 3, 5, 10)},
        "rank_counts": dict(
            collections.Counter("none" if r["rank"] is None else r["rank"] for r in rows)
        ),
        "share_src_mean": statistics.mean(r["share_src"] for r in rows),
        "share_other_mean": statistics.mean(
            r["share_other"] for r in rows if not math.isnan(r["share_other"])
        ),
        "share_src_full": sum(r["share_src"] >= 0.999 for r in rows) / n if n else math.nan,
        "q_lemmas_median": statistics.median(r["q_lemmas"] for r in rows),
        "rare": {
            "share": len(rare) / n if n else math.nan,
            "hit1": _hit1(rare),
            "hit10": _hit10(rare),
            "hit1_without": _hit1(no_rare),
            "hit10_without": _hit10(no_rare),
        },
        "by_rank": by_rank,
        "probes": {
            "n": len(probes),
            "hit1": sum(r == 1 for r in probe_ranks) / len(probes) if probes else math.nan,
            "hit10": sum(r is not None for r in probe_ranks) / len(probes) if probes else math.nan,
        },
    }


def _ekc_arrays(
    rep: dict[str, Any], arm: str, pids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    facts = rep["answers"]["facts"]
    kc = np.full(len(pids), np.nan)
    tc = np.zeros(len(pids))
    cnt = np.zeros(len(pids))
    for j, pid in enumerate(pids):
        fact = facts.get((arm, pid))
        if not fact or fact["label"] not in ("TC", "FA"):
            continue
        cnt[j] = 1.0
        if fact["label"] == "TC":
            tc[j] = 1.0
            if fact["answer_correctness"] is not None and fact["answer_faithfulness"] is not None:
                kc[j] = fact["answer_correctness"] * fact["answer_faithfulness"]
    return kc, tc, cnt


def _ekc(kc: np.ndarray, tc: np.ndarray, cnt: np.ndarray) -> float:
    return _kc_mean(kc) * float(tc.sum()) / float(cnt.sum()) if cnt.sum() else math.nan


def ekc_diff_bootstrap(
    reps: list[dict[str, Any]],
    pids: list[str],
    arm_a: str,
    arm_b: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    cluster: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Paired bootstrap CI for ``EKC(arm_a) - EKC(arm_b)`` pooled across judge runs.

    Args:
        reps: Loaded judge retest replicas.
        pids: Pair IDs to evaluate.
        arm_a: First retrieval arm.
        arm_b: Second retrieval arm.
        n_boot: Number of bootstrap replicates.
        seed: Random generator seed.
        cluster: Optional mapping ``pid -> cluster_id`` for interview-level resampling.

    Returns:
        Dictionary with pooled point estimate, per-run deltas, and 95% CI.
    """
    valid_reps = [
        r
        for r in reps
        if any(k[0] == arm_a for k in r["answers"]["facts"])
        and any(k[0] == arm_b for k in r["answers"]["facts"])
    ]
    n_runs = len(valid_reps)
    a_arrs = [_ekc_arrays(r, arm_a, pids) for r in valid_reps]
    b_arrs = [_ekc_arrays(r, arm_b, pids) for r in valid_reps]

    ak = np.stack([a[0] for a in a_arrs])
    at = np.stack([a[1] for a in a_arrs])
    ac = np.stack([a[2] for a in a_arrs])
    bk = np.stack([b[0] for b in b_arrs])
    bt = np.stack([b[1] for b in b_arrs])
    bc = np.stack([b[2] for b in b_arrs])

    per_run = [_ekc(ak[k], at[k], ac[k]) - _ekc(bk[k], bt[k], bc[k]) for k in range(n_runs)]
    if cluster is None:
        groups = [np.array([j]) for j in range(len(pids))]
    else:
        grouped: dict[str, list[int]] = collections.defaultdict(list)
        for j, pid in enumerate(pids):
            grouped[cluster[pid]].append(j)
        groups = [np.array(idxs) for idxs in grouped.values()]

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        run_idx = rng.integers(0, n_runs, len(idx))
        boots[i] = _ekc(ak[run_idx, idx], at[run_idx, idx], ac[run_idx, idx]) - _ekc(
            bk[run_idx, idx], bt[run_idx, idx], bc[run_idx, idx]
        )
    boots = boots[np.isfinite(boots)]
    return {
        "est": float(np.mean(per_run)),
        "per_run": [float(x) for x in per_run],
        "n_runs": n_runs,
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": len(pids),
        "ekc_a": [float(_ekc(ak[k], at[k], ac[k])) for k in range(n_runs)],
        "ekc_b": [float(_ekc(bk[k], bt[k], bc[k])) for k in range(n_runs)],
    }


def band_bootstrap_pooled(
    reps: list[dict[str, Any]],
    long_ids: list[str],
    short_ids: list[str],
    arm_b: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap ``KC(bm25) - KC(arm_b)`` in Q4, Q1-Q3, and their gap across replicas.

    Args:
        reps: Loaded judge retest replicas.
        long_ids: Useful pair IDs from longest-quartile interviews (Q4).
        short_ids: Useful pair IDs from remaining interviews (Q1-Q3).
        arm_b: Comparison retrieval arm.
        n_boot: Number of bootstrap replicates.
        seed: Random generator seed.

    Returns:
        Dictionary with pooled point estimates, 95% CIs, and per-replica values.
    """
    data = [
        {
            regime: (_kc_array(r, "bm25", ids), _kc_array(r, arm_b, ids))
            for regime, ids in (("long", long_ids), ("short", short_ids))
        }
        for r in reps
    ]

    def _stat(il: np.ndarray, is_: np.ndarray, ks: list[int]) -> np.ndarray:
        vals = []
        for k in ks:
            (al, bl), (as_, bs) = data[k]["long"], data[k]["short"]
            dl = _kc_mean(al[il]) - _kc_mean(bl[il])
            ds = _kc_mean(as_[is_]) - _kc_mean(bs[is_])
            vals.append((dl, ds, dl - ds))
        return np.nanmean(np.array(vals), axis=0)

    point = _stat(np.arange(len(long_ids)), np.arange(len(short_ids)), list(range(len(reps))))
    rng = np.random.default_rng(seed)
    boots = np.array(
        [
            _stat(
                rng.integers(0, len(long_ids), len(long_ids)),
                rng.integers(0, len(short_ids), len(short_ids)),
                list(rng.integers(0, len(reps), len(reps))),
            )
            for _ in range(n_boot)
        ]
    )
    ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    per_rep = [
        list(_stat(np.arange(len(long_ids)), np.arange(len(short_ids)), [k]))
        for k in range(len(reps))
    ]
    names = ("long", "short", "gap")
    return {
        "n_long": len(long_ids),
        "n_short": len(short_ids),
        "n_replicas": len(reps),
        **{
            name: {
                "est": float(point[i]),
                "ci": [float(ci[0, i]), float(ci[1, i])],
                "per_replica": [float(r[i]) for r in per_rep],
            }
            for i, name in enumerate(names)
        },
    }


def community_ev_bootstrap(
    cep: dict[str, dict[str, Any]],
    emic: dict[str, float | None],
    useful: list[str],
    interviews: dict[str, dict[str, Any]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    drop_top: tuple[int, ...] = (0, 2, 10),
) -> dict[str, Any]:
    """Cluster-bootstrap EV >= 4 differences between communities over interviews.

    Args:
        cep: CEP pair metadata mapping.
        emic: Mapping from pair ID to emic validity score.
        useful: Useful pair IDs.
        interviews: Mapping from :func:`load_interviews`.
        n_boot: Number of bootstrap replicates.
        seed: Random generator seed.
        drop_top: Numbers of top-producing interviews to drop in sensitivity checks.

    Returns:
        Dictionary of pairwise community differences and top-interview drop checks.
    """
    by_file: dict[str, list[str]] = collections.defaultdict(list)
    for pid in useful:
        if emic.get(pid) is not None:
            by_file[cep[pid]["file"]].append(pid)
    comm_files: dict[str, list[str]] = collections.defaultdict(list)
    for fid in by_file:
        comm_files[interviews[fid]["community"]].append(fid)

    def _share(files: list[str] | np.ndarray) -> float:
        vals = [emic[p] >= TAU_EMIC for fid in files for p in by_file[fid]]
        return float(np.mean(vals)) if vals else math.nan

    rng = np.random.default_rng(seed)
    diffs: dict[str, dict[str, Any]] = {}
    base = "Barra de Pelotas"
    for other in ("Doquinhas", "Balsa"):
        fa, fb = comm_files[base], comm_files[other]
        est = _share(fa) - _share(fb)
        bs = [
            _share(rng.choice(fa, len(fa))) - _share(rng.choice(fb, len(fb))) for _ in range(n_boot)
        ]
        diffs[f"{base} - {other}"] = {
            "est": est,
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "interviews": [len(fa), len(fb)],
        }

    top = sorted(by_file, key=lambda fid: -len(by_file[fid]))
    dropped: dict[str, dict[str, Any]] = {}
    for k in drop_top:
        keep = set(top[k:])
        dropped[f"drop_top_{k}"] = {
            comm: {
                "share": _share([fid for fid in fs if fid in keep]),
                "pairs": sum(len(by_file[fid]) for fid in fs if fid in keep),
            }
            for comm, fs in comm_files.items()
        }
    return {
        "differences": diffs,
        "after_dropping": dropped,
        "top_interviews": [
            {"community": interviews[fid]["community"], "pairs": len(by_file[fid])}
            for fid in top[:10]
        ],
    }


def _fmt(val: Any, nd: int = 3) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    if isinstance(val, float):
        return f"{val:.{nd}f}"
    return str(val)


def render_markdown(res: dict[str, Any]) -> str:
    """Render the full corpus coverage report as Markdown.

    Args:
        res: Result dictionary built by :func:`main`.

    Returns:
        Markdown-formatted report.
    """
    lines = [f"# Corpus Coverage Analysis ({res['run_id']})", ""]
    iv = res["interviews"]
    lines += [
        "## Interviews",
        "",
        f"- Interviews: {iv['n_interviews']}; useful pairs: {iv['n_useful']}; "
        f"contributing: {iv['contributing']}; zero useful: {iv['zero']}",
        f"- Median chars: zero-yield {iv['zero_median_chars']}, "
        f"contributing {iv['contrib_median_chars']}",
        f"- Zero-yield single-chunk interviews: {iv['zero_single_chunk']} of {iv['zero']}",
        f"- Spearman(length, useful) = {_fmt(iv['spearman_chars_useful'])}; "
        f"Spearman(length, useful/chunk) = {_fmt(iv['spearman_chars_useful_per_chunk'])}",
        f"- Gini(chars) = {_fmt(iv['gini_chars'])}, Gini(useful) = {_fmt(iv['gini_useful'])}",
        f"- Quartile cutoffs (chars): {iv['length_quartile_cuts']}",
        "",
        "| Quartile | n | Chars | % text | Chunks | Useful | % useful "
        "| Contributing | Useful/chunk |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for band, v in iv["length_bands"].items():
        chars_pct = 100 * v["chars_share"]
        useful_pct = 100 * v["useful_share"]
        lines.append(
            f"| {band} | {v['n']} | {v['chars_min']}-{v['chars_max']} | {chars_pct:.1f} "
            f"| {v['chunks']} | {v['useful']} | {useful_pct:.1f} "
            f"| {v['contributing']} | {_fmt(v['yield_per_chunk'])} |"
        )

    ch = res["chunks"]
    lines += [
        "",
        "## Chunks",
        "",
        f"- Chunks: {ch['n_chunks']}; useful-per-chunk distribution: {ch['useful_per_chunk_dist']}",
        "",
        "| Length band | n | Mean useful | % contributing |",
        "|---|---:|---:|---:|",
    ]
    for band, v in ch["by_length"].items():
        lines.append(
            f"| {band} | {v['n']} | {_fmt(v['mean_useful'])} "
            f"| {_fmt(100 * v['share_contributing'], 1)} |"
        )

    lines += [
        "",
        "## Communities",
        "",
        "| Community | Pairs | Remember pass | Candidates | Useful | Rate | "
        "EV>=4 Remember | EV>=4 Cand. | EV>=4 Useful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comm, v in res["communities"].items():
        lines.append(
            f"| {comm} | {v['pairs']} | {_fmt(v['remember_pass'])} | {v['candidates']} "
            f"| {v['useful']} | {_fmt(v['useful_rate'])} | {_fmt(v['ev_ge4_remember'][0])} "
            f"| {_fmt(v['ev_ge4_candidates'][0])} | {_fmt(v['ev_ge4_useful'][0])} "
            f"(n={v['ev_ge4_useful'][1]}) |"
        )

    gr = res["graph"]
    lines += [
        "",
        "## Induced Knowledge Graph",
        "",
        f"- Passages: {gr['passages']} (matched: {gr['passages_matched']}, "
        f"unmatched: {gr['passages_unmatched']})",
        f"- Entity/event nodes: {gr['entity_event_nodes']} "
        f"(in 2+ interviews: {gr['entity_event_nodes_in_2plus_interviews']})",
        f"- Relation edges: {gr['relation_edges']} "
        f"(within-interview share: {_fmt(gr['relation_edges_within_interview_share'])}, "
        f"cross-interview: {gr['relation_edges_cross_interview']})",
        f"- Spearman(chars, nodes) = {_fmt(gr['spearman_chars_nodes'])}, "
        f"Gini(nodes) = {_fmt(gr['gini_nodes'])}",
        "",
        "| Community | Interviews | % text | % nodes | Nodes / 1k chars | Relations / 1k chars |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comm, v in gr["by_community"].items():
        lines.append(
            f"| {comm} | {v['interviews']} | {100 * v['chars_share']:.1f} "
            f"| {100 * v['nodes_share']:.1f} | {_fmt(v['nodes_per_kchar'], 2)} "
            f"| {_fmt(v['rel_edges_per_kchar'], 2)} |"
        )

    for title, key in (
        ("Retrieval by Community", "retrieval_by_community"),
        ("Retrieval by Interview Length Quartile", "retrieval_by_length"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Group | Arm | Useful | Probes | KC | TC | OC | Hall | F1_abs | EvSup | Cond. OC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for grp, arms in res[key].items():
            for arm, v in arms.items():
                lines.append(
                    f"| {grp} | {arm} | {v['n_answerable']} | {v['n_nonans']} | {_fmt(v['kc'])} "
                    f"| {v['kc_n']} | {_fmt(v['oc'])} | {_fmt(v['hall'])} | {_fmt(v['f1_abs'])} "
                    f"| {_fmt(v['pc_pass'])} | {_fmt(v['cond_oc'])} |"
                )

    emic_gate = res["retrieval_emic_gate"]
    lines += [
        "",
        f"## Retrieval Under Emic QA Gate (EV >= {TAU_EMIC}; {emic_gate['n_useful']} useful pairs)",
        "",
        "| Arm | KC | OC | EKC | Hall | Probes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, v in emic_gate["arms"].items():
        lines.append(
            f"| {arm} | {_fmt(v['kc'])} | {_fmt(v['oc'])} | {_fmt(v['ekc'])} "
            f"| {_fmt(v['hall'])} | {v['n_nonans']} |"
        )

    bs = res["bm25_source"]
    lines += [
        "",
        "## BM25 Source-Chunk Circularity",
        "",
        f"- Useful pairs: {bs['n']}; hit@1={_fmt(bs['hit_at'][1])}, "
        f"hit@3={_fmt(bs['hit_at'][3])}, hit@10={_fmt(bs['hit_at'][10])}",
        f"- Rare-lemma share (df <= {bs['rare_df']}): {_fmt(bs['rare']['share'])} "
        f"(with: @1={_fmt(bs['rare']['hit1'])}, @10={_fmt(bs['rare']['hit10'])}; "
        f"without: @1={_fmt(bs['rare']['hit1_without'])}, @10={_fmt(bs['rare']['hit10_without'])})",
        f"- Probes ({bs['probes']['n']}): seed chunk @1={_fmt(bs['probes']['hit1'])}, "
        f"@10={_fmt(bs['probes']['hit10'])}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point for corpus coverage analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze corpus coverage, graph representation, and stratified retrieval."
    )
    parser.add_argument("--id", required=True, help="Pipeline run ID (e.g. thesis-run-02).")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Results root.")
    parser.add_argument(
        "--replica",
        type=Path,
        required=True,
        help="Canonical judge-retest replica directory (e.g. results/judge-retest/r1-oracle).",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument(
        "--retest-replica",
        action="append",
        default=[],
        help="Additional judge re-run directory (optionally PATH=GATE_FROM).",
    )
    parser.add_argument(
        "--emic-replica",
        type=Path,
        action="append",
        default=[],
        help="Additional replica directories whose emic_judge outputs are compared.",
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap draws.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap RNG seed.")
    args = parser.parse_args()

    run_dir = args.results_root / args.id
    nonans_dataset = run_dir / "non_answerable" / "outputs" / "dataset.json"
    kg_graphml = run_dir / "kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml"

    exclude = kg_collision_probes(nonans_dataset, kg_graphml)
    rep = load_replica(args.replica, None, None, exclude)
    cep = rep["cep"]
    approved = {p for p, m in cep.items() if m["approved"]}
    interviews = load_interviews(run_dir / "cep" / "outputs")
    emic = rep["emic"]

    iv = interview_section(cep, approved, interviews)
    useful = set(iv["useful"])
    by_comm = {
        comm: {p for p, m in cep.items() if interviews[m["file"]]["community"] == comm}
        for comm in COMMUNITIES
    }
    by_len = {
        band: {p for p, m in cep.items() if iv["band_of_file"][m["file"]] == band}
        for band in ("Q1", "Q2", "Q3", "Q4")
    }

    res: dict[str, Any] = {
        "run_id": args.id,
        "replica": str(args.replica),
        "excluded_probes": len(exclude),
        "interviews": {k: v for k, v in iv.items() if k not in ("useful", "band_of_file")},
        "chunks": chunk_section(cep, approved),
        "communities": community_section(cep, approved, interviews, emic),
        "graph": graph_section(run_dir, interviews),
        "retrieval_by_community": retrieval_section(rep, useful, by_comm),
        "retrieval_by_length": retrieval_section(rep, useful, by_len),
        "retrieval_all": retrieval_section(rep, useful, {"all": set(cep)}),
    }

    reps_emic = [
        emic,
        *(load_emic(p / "emic_judge" / "outputs", cep)[0] for p in args.emic_replica),
    ]
    res["emic_by_community_replicas"] = emic_stability_section(cep, approved, interviews, reps_emic)

    useful_list = sorted(useful)
    long_ids = [p for p in useful_list if iv["band_of_file"][cep[p]["file"]] == "Q4"]
    short_ids = [p for p in useful_list if iv["band_of_file"][cep[p]["file"]] != "Q4"]
    file_of = {p: cep[p]["file"] for p in useful_list}

    kc_boot: dict[str, Any] = {}
    for arm in ("khop_passage", "atlas_rag"):
        for name, ids in (("Q4", long_ids), ("Q1-Q3", short_ids), ("all", useful_list)):
            kc_boot[f"bm25-{arm}|{name}|item"] = kc_diff_bootstrap(
                rep, ids, "bm25", arm, args.n_boot, args.seed
            )
            kc_boot[f"bm25-{arm}|{name}|interview"] = kc_diff_bootstrap(
                rep, ids, "bm25", arm, args.n_boot, args.seed, file_of
            )
        for comm in ("Barra de Pelotas", "Doquinhas", "Balsa"):
            ids = [p for p in useful_list if interviews[cep[p]["file"]]["community"] == comm]
            kc_boot[f"bm25-{arm}|{comm}|interview"] = kc_diff_bootstrap(
                rep, ids, "bm25", arm, args.n_boot, args.seed, file_of
            )
    res["kc_diff_bootstrap"] = kc_boot

    retest = [rep]
    for spec in args.retest_replica:
        path_str, _, gate_str = spec.partition("=")
        retest.append(
            load_replica(Path(path_str), None, Path(gate_str) if gate_str else None, exclude)
        )
    res["band_bootstrap_pooled"] = {
        arm: band_bootstrap_pooled(retest, long_ids, short_ids, arm, args.n_boot, args.seed)
        for arm in ("khop_passage", "atlas_rag")
    }
    res["community_ev_bootstrap"] = community_ev_bootstrap(
        cep, emic, useful_list, interviews, args.n_boot, args.seed
    )
    res["evsup_strata"] = [evsup_strata(r, useful_list) for r in retest]

    useful_emic = {p for p in useful if emic.get(p) is not None and emic[p] >= TAU_EMIC}
    emic_tab = retrieval_section(rep, useful_emic, {"emic_gate": set(cep)})["emic_gate"]
    for v in emic_tab.values():
        v["ekc"] = v["kc"] * (1.0 - v["oc"]) if v["kc"] is not None else None
    res["retrieval_emic_gate"] = {"n_useful": len(useful_emic), "arms": emic_tab}

    ekc_boot: dict[str, Any] = {}
    arm_pairs = [("bm25", g) for g in ("khop_passage", "khop_triple", "atlas_rag")] + [
        ("oracle", g) for g in ("bm25", "khop_passage", "khop_triple", "atlas_rag")
    ]
    for arm_a, arm_b in arm_pairs:
        ekc_boot[f"{arm_a}-{arm_b}|item"] = ekc_diff_bootstrap(
            retest, useful_list, arm_a, arm_b, args.n_boot, args.seed
        )
        ekc_boot[f"{arm_a}-{arm_b}|interview"] = ekc_diff_bootstrap(
            retest, useful_list, arm_a, arm_b, args.n_boot, args.seed, file_of
        )
    res["ekc_diff_bootstrap"] = ekc_boot

    probe_ids = sorted(
        k[1]
        for k in rep["answers"]["facts"]
        if k[0] == "bm25" and k[1].endswith(":nonans") and k[1].removesuffix(":nonans") in useful
    )
    res["bm25_source"] = bm25_source_section(rep, run_dir, useful_list, probe_ids)
    for v in res["bm25_source"]["by_rank"].values():
        v["ekc_bm25_minus_oracle"] = ekc_diff_bootstrap(
            [rep], v.pop("pids"), "bm25", "oracle", args.n_boot, args.seed
        )

    md = render_markdown(res)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
