#!/usr/bin/env python3
"""Old-vs-new k-hop seeding/retrieval distribution on the test-kg-04 KG (local).

Isolates the linker change: this PR altered ONLY the entity link, so we hold the
retriever's ego-graph + scoring fixed and swap just the seeding to compare OLD
(bare-whitespace tokens + hard max_postings=200 cap) vs NEW (lemmatized +
IDF-weighted top-K) on the test-kg-04 CEP questions. Pure lexical + graph — runs
locally, no GPU/LLM. Usage:  uv run python scripts/test_khop_seeding.py
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from arandu.shared.rag.retrievers import _khop_common as kc
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever

KG_DIR = Path("results/test-kg-04/kg/outputs/atlas_output")
KG_PATH = KG_DIR / "kg_graphml" / "transcriptions.json_graph.graphml"
CEP = Path("results/test-kg-04/cep/outputs")
PROBLEM_TERMS = ["enchente", "pesca", "trapiche", "Valverde", "rio", "bagre"]
RETRIEVE_SAMPLE = 40   # full-retrieve is ego-graph-bound; sample for speed
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_OLD_CAP = 200


# ---- OLD linker (reimplemented inline: whitespace tokens + hard cap) ---------
def _old_tokens(text: str, *, filter_stopwords: bool) -> list[str]:
    toks = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    if filter_stopwords:
        toks = [t for t in toks if t not in kc._STOPWORDS and len(t) >= kc._MIN_TOKEN_LEN]
    return toks


def _old_index(g: nx.DiGraph) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for node_id, attrs in g.nodes(data=True):
        if attrs.get("type") in kc._LINKABLE_TYPES:
            for t in _old_tokens(attrs.get("id", ""), filter_stopwords=False):
                idx[t].add(node_id)
    return idx


def _old_link(question: str, idx: dict[str, set[str]]) -> set[str]:
    seeds: set[str] = set()
    for t in set(_old_tokens(question, filter_stopwords=True)):
        postings = idx.get(t, ())
        if len(postings) > _OLD_CAP:
            continue
        seeds.update(postings)
    return seeds


# ---- helpers -----------------------------------------------------------------
def _questions() -> list[tuple[str, str]]:
    """Return (question, bloom_level) over all test-kg-04 CEP pairs."""
    out: list[tuple[str, str]] = []
    for f in sorted(CEP.glob("*_cep_qa.json")):
        for p in json.loads(f.read_text()).get("qa_pairs", []):
            out.append((p["question"], p.get("bloom_level", "?")))
    return out


def _dist(counts: list[int]) -> str:
    if not counts:
        return "n=0"
    s = sorted(counts)
    pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return (
        f"min={s[0]} p50={statistics.median(s):.0f} mean={statistics.mean(s):.1f} "
        f"p90={pct(0.9)} max={s[-1]}"
    )


def main() -> None:
    g = nx.read_graphml(str(KG_PATH))
    old_idx = _old_index(g)
    r = KHopSubgraphRetriever(KG_DIR, top_k_seeds=50)  # new index built inside
    qs = _questions()
    print(f"KG: {g.number_of_nodes()} nodes | CEP questions: {len(qs)}\n")

    # ---- SEED LEVEL (all questions) ----
    old_seed_counts, new_seed_counts = [], []
    old_empty = new_empty = 0
    by_bloom_old: dict[str, list[int]] = defaultdict(list)
    by_bloom_new: dict[str, list[int]] = defaultdict(list)
    for q, bloom in qs:
        o = len(_old_link(q, old_idx))
        n = len(list(r._entity_link(q)))
        old_seed_counts.append(o)
        new_seed_counts.append(n)
        old_empty += o == 0
        new_empty += n == 0
        by_bloom_old[bloom].append(o == 0)
        by_bloom_new[bloom].append(n == 0)

    print("=== SEED LEVEL (all questions) ===")
    print(f"OLD empty-seed: {old_empty:4d} ({old_empty / len(qs):.1%})  dist {_dist(old_seed_counts)}")
    print(f"NEW empty-seed: {new_empty:4d} ({new_empty / len(qs):.1%})  dist {_dist(new_seed_counts)}")
    print("\nempty-seed fraction by Bloom level (OLD -> NEW):")
    for bloom in sorted(by_bloom_new):
        ob = by_bloom_old[bloom]
        nb = by_bloom_new[bloom]
        print(
            f"  {bloom:11s} n={len(nb):4d}  {sum(ob) / len(ob):5.1%} -> {sum(nb) / len(nb):5.1%}"
        )

    # ---- RETRIEVAL LEVEL (NEW retriever, sample) ----
    # NEW only: top_k_seeds=50 bounds the ego graph so this is fast. The OLD
    # linker is unbounded (a common token seeds thousands of nodes -> ego-graph
    # explosion), which is the very pathology being fixed; its retrieval-level
    # cost is captured by the OLD empty-seed fraction above (empty seeds ->
    # empty retrieval -> answerer abstains).
    sample = qs[:: max(1, len(qs) // RETRIEVE_SAMPLE)][:RETRIEVE_SAMPLE]
    new_results = [len(r.retrieve(q, top_k=10)) for q, _ in sample]
    print(f"\n=== RETRIEVAL LEVEL (NEW retriever, sample of {len(sample)}) ===")
    print(f"NEW empty-result: {new_results.count(0):3d} ({new_results.count(0)/len(sample):.1%})  passages/query {_dist(new_results)}")

    # ---- PROBLEM TERMS ----
    print("\n=== problem terms (seed counts, OLD -> NEW) ===")
    for t in PROBLEM_TERMS:
        print(f"  {t:12s} old={len(_old_link(t, old_idx)):5d}  new={len(list(r._entity_link(t))):4d}")


if __name__ == "__main__":
    main()
