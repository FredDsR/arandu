#!/usr/bin/env python3
"""Structural profile of a run's knowledge graph, for the paper's Setup and Discussion.

Complements ``kg_structural_metrics.py`` (whole-graph NetworkX summary) with the
questions the paper actually asks of the graph:

1. What does the relation-only subgraph over entities and events look like
   (degree tail, clustering, components), and which nodes are its hubs?
2. How concentrated is the induced concept layer on a few generic concepts?
3. How far does the k-hop retriever's 2-hop ego graph reach from a typical
   entity, and how much of that reach comes through concept nodes?
4. Do relation edges and Louvain communities follow the interview locale?
5. How many entity labels differ only in case or spacing (entity resolution)?

Every number printed is also written to
``results/<id>/analysis/kg_structure_profile.json``. Run from the repo root:

    uv run python -m scripts.kg_structure_profile --id thesis-run-02

Notes:
    - Locale comes from the ``Local:`` line of the passage metadata block; a
      node's locale is the locale of the passage(s) it has a Source edge to.
      atlas-rag emits one Source edge per node for almost every node, so the
      locale and passage attributions are first-occurrence approximations.
    - The ego-graph sample and Louvain are seeded; change ``--seed`` to check
      sensitivity.
    - "Relation" edges include the participation edges atlas-rag synthesizes
      between events and their participants (``envolve``/``involves``); their
      share is reported separately.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

from arandu.kg.atlas_backend import (
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG,
    SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN,
)

SYNTHESIZED_PREDICATES: frozenset[str] = frozenset(
    {SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_EN}
    | set(SYNTHESIZED_EVENT_PARTICIPATION_PREDICATE_BY_LANG.values())
)
GRAPHML_REL = Path("kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml")
QUANTILES = (25, 50, 75, 90, 95, 99, 99.9)
TOP_N = 20


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def ccdf_slope(degrees: list[int], min_degree: int = 3) -> float | None:
    """Least-squares slope of log CCDF vs log degree for degree >= min_degree."""
    counts = collections.Counter(degrees)
    n = len(degrees)
    cum, pts = 0, []
    for d in sorted(counts, reverse=True):
        cum += counts[d]
        if d >= min_degree:
            pts.append((math.log(d), math.log(cum / n)))
    if len(pts) < 5:
        return None
    mx = statistics.mean(x for x, _ in pts)
    my = statistics.mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    return sxy / sxx


def load(run_dir: Path) -> nx.DiGraph:
    path = run_dir / GRAPHML_REL
    if not path.exists():
        raise SystemExit(f"GraphML not found: {path}")
    return nx.read_graphml(path)


def profile(g: nx.DiGraph, *, seed: int, ego_sample: int) -> dict:
    typ = {n: d.get("type") for n, d in g.nodes(data=True)}
    label = {n: d.get("id", "") for n, d in g.nodes(data=True)}
    out: dict = {}

    # 1. Composition
    out["nodes_by_type"] = dict(collections.Counter(typ.values()))
    out["edges_by_type"] = dict(collections.Counter(d["type"] for _, _, d in g.edges(data=True)))
    synth = sum(
        1 for _, _, d in g.edges(data=True)
        if d["type"] == "Relation" and d.get("relation") in SYNTHESIZED_PREDICATES
    )
    out["relation_edges_synthesized_participation"] = synth
    out["relation_edges_synthesized_share"] = synth / out["edges_by_type"]["Relation"]

    # Provenance and locale
    src: dict[str, set[str]] = collections.defaultdict(set)
    for u, v, d in g.edges(data=True):
        if d["type"] == "Source":
            src[u].add(v)
    locale_of_passage = {}
    for n in g:
        if typ[n] == "passage":
            m = re.search(r"Local:\s*([^\n]+)", label[n])
            locale_of_passage[n] = m.group(1).strip() if m else None
    out["passages_by_locale"] = dict(collections.Counter(
        loc or "(no locale)" for loc in locale_of_passage.values()
    ))
    locale_of_node = {
        n: {locale_of_passage[p] for p in ps if locale_of_passage.get(p)} for n, ps in src.items()
    }
    out["entity_event_nodes_with_source_edge"] = sum(
        1 for n in src if typ[n] in ("entity", "event")
    )
    out["entity_event_nodes_in_more_than_one_passage"] = sum(
        1 for n, ps in src.items() if typ[n] in ("entity", "event") and len(ps) > 1
    )

    # 2. Relation-only subgraph (undirected, simple; self-loops kept so that the
    # node and component counts match the Setup paragraph of the paper)
    r = nx.Graph()
    for u, v, d in g.edges(data=True):
        if d["type"] == "Relation":
            r.add_edge(u, v)
    comps = sorted((len(c) for c in nx.connected_components(r)), reverse=True)
    deg = dict(r.degree())
    degs = list(deg.values())
    rel = out["relation_subgraph"] = {
        "nodes": r.number_of_nodes(),
        "edges": r.number_of_edges(),
        "components": len(comps),
        "largest_component": comps[0],
        "largest_component_fraction": comps[0] / r.number_of_nodes(),
        "components_of_size_2": sum(1 for c in comps if c == 2),
        "degree_mean": statistics.mean(degs),
        "degree_quantiles": {str(q): percentile(degs, q) for q in QUANTILES},
        "degree_max": max(degs),
        "share_degree_1": sum(1 for d in degs if d == 1) / len(degs),
        "ccdf_loglog_slope_deg_ge_3": ccdf_slope(degs),
        "average_clustering": nx.average_clustering(r),
        "transitivity": nx.transitivity(r),
    }
    hubs = sorted(deg.items(), key=lambda kv: -kv[1])[:TOP_N]
    hub_set = {n for n, _ in hubs}
    rel["hubs"] = [{"label": label[n], "type": typ[n], "degree": d} for n, d in hubs]
    rel["edges_touching_a_hub"] = sum(1 for u, v in r.edges() if u in hub_set or v in hub_set)
    rel["edges_touching_a_hub_share"] = rel["edges_touching_a_hub"] / r.number_of_edges()

    # Edges within / across passage and locale
    same_p = cross_p = same_l = cross_l = unknown = 0
    for u, v in r.edges():
        pu, pv = src.get(u), src.get(v)
        if not pu or not pv:
            unknown += 1
            continue
        if pu & pv:
            same_p += 1
        else:
            cross_p += 1
        lu, lv = locale_of_node.get(u, set()), locale_of_node.get(v, set())
        if lu and lv:
            if lu & lv:
                same_l += 1
            else:
                cross_l += 1
    rel["edges_same_passage"] = same_p
    rel["edges_cross_passage"] = cross_p
    rel["edges_same_locale"] = same_l
    rel["edges_cross_locale"] = cross_l
    rel["edges_cross_locale_share"] = cross_l / (same_l + cross_l)
    rel["edges_without_provenance"] = unknown

    # Louvain on the giant component, with locale purity
    giant = r.subgraph(max(nx.connected_components(r), key=len))
    comm = louvain_communities(giant, seed=seed)
    purities = []
    rows = []
    for c in sorted(comm, key=len, reverse=True):
        cnt: collections.Counter = collections.Counter()
        passages: set[str] = set()
        for n in c:
            passages |= src.get(n, set())
            for loc in locale_of_node.get(n, set()):
                cnt[loc] += 1
        total = sum(cnt.values())
        if total:
            top_loc, top_n = cnt.most_common(1)[0]
            purities.append(top_n / total)
            rows.append({
                "size": len(c), "passages": len(passages),
                "dominant_locale": top_loc, "purity": top_n / total,
            })
    rel["louvain"] = {
        "seed": seed,
        "communities": len(comm),
        "modularity": modularity(giant, comm),
        "size_median": statistics.median(len(c) for c in comm),
        "purity_median": statistics.median(purities),
        "communities_with_purity_ge_0.8": sum(1 for p in purities if p >= 0.8),
        "largest": rows[:10],
    }

    # 3. Concept layer concentration
    cdeg = sorted(
        ((n, d) for n, d in g.degree() if typ[n] == "concept"), key=lambda kv: -kv[1]
    )
    ctotal = sum(d for _, d in cdeg)
    ctop = sum(d for _, d in cdeg[:TOP_N])
    out["concept_layer"] = {
        "concepts": len(cdeg),
        "concept_degree_median": statistics.median(d for _, d in cdeg),
        "concept_degree_max": cdeg[0][1],
        "top_concepts": [{"label": label[n], "degree": d} for n, d in cdeg[:TOP_N]],
        "top_concepts_share_of_concept_edge_endpoints": ctop / ctotal,
    }

    # 4. Two-hop ego reach from a random entity (as the k-hop arms expand it)
    und = g.to_undirected(as_view=True)
    no_concept = und.subgraph([n for n in g if typ[n] != "concept"])
    rng = random.Random(seed)
    entities = sorted(n for n in g if typ[n] == "entity")
    sample = rng.sample(entities, min(ego_sample, len(entities)))
    sizes, via_concept, passages = [], [], []
    for s in sample:
        ego = nx.ego_graph(und, s, radius=2)
        ego_nc = nx.ego_graph(no_concept, s, radius=2)
        sizes.append(ego.number_of_nodes())
        via_concept.append(ego.number_of_nodes() - ego_nc.number_of_nodes())
        passages.append(sum(1 for n in ego if typ[n] == "passage"))
    out["two_hop_ego_from_random_entity"] = {
        "seed": seed,
        "sample": len(sample),
        "nodes_median": statistics.median(sizes),
        "nodes_p90": percentile(sizes, 90),
        "nodes_max": max(sizes),
        "nodes_reached_only_via_concepts_median": statistics.median(via_concept),
        "nodes_reached_only_via_concepts_p90": percentile(via_concept, 90),
        "passages_median": statistics.median(passages),
        "passages_p90": percentile(passages, 90),
        "passages_total": out["nodes_by_type"].get("passage", 0),
    }

    # 5. Entity label collisions
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for n in g:
        if typ[n] == "entity":
            groups[" ".join(label[n].split()).casefold()].append(n)
    coll = {k: v for k, v in groups.items() if len(v) > 1}
    out["entity_label_collisions"] = {
        "groups": len(coll),
        "nodes_involved": sum(len(v) for v in coll.values()),
        "examples": [
            {"label": k, "nodes": len(v)}
            for k, v in sorted(coll.items(), key=lambda kv: -len(kv[1]))[:10]
        ],
    }
    return out


def summarize(p: dict) -> str:
    rel, lv, cl, ego, col = (
        p["relation_subgraph"], p["relation_subgraph"]["louvain"],
        p["concept_layer"], p["two_hop_ego_from_random_entity"], p["entity_label_collisions"],
    )
    hubs = ", ".join(f"{h['label']} ({h['degree']})" for h in rel["hubs"][:10])
    concepts = ", ".join(c["label"] for c in cl["top_concepts"][:10])
    return "\n".join([
        f"Relation-only subgraph: {rel['nodes']} nodes, {rel['edges']} edges, "
        f"{rel['components']} components, giant component {rel['largest_component_fraction']:.1%}",
        f"  degree: median {rel['degree_quantiles']['50']:.0f}, p90 {rel['degree_quantiles']['90']:.0f}, "
        f"p99 {rel['degree_quantiles']['99']:.0f}, max {rel['degree_max']}; "
        f"degree-1 share {rel['share_degree_1']:.1%}; CCDF slope {rel['ccdf_loglog_slope_deg_ge_3']:.2f}",
        f"  clustering {rel['average_clustering']:.3f}, transitivity {rel['transitivity']:.3f}",
        f"  top-{TOP_N} hubs touch {rel['edges_touching_a_hub_share']:.1%} of edges: {hubs}",
        f"  edges cross-passage {rel['edges_cross_passage']} / same-passage {rel['edges_same_passage']}; "
        f"cross-locale share {rel['edges_cross_locale_share']:.1%}",
        f"  Louvain: {lv['communities']} communities, modularity {lv['modularity']:.3f}, "
        f"median locale purity {lv['purity_median']:.2f}, "
        f"{lv['communities_with_purity_ge_0.8']} with purity >= 0.8",
        f"Concept layer: top-{TOP_N} concepts carry "
        f"{cl['top_concepts_share_of_concept_edge_endpoints']:.1%} of concept-edge endpoints: {concepts}",
        f"2-hop ego from a random entity (n={ego['sample']}): median {ego['nodes_median']:.0f} nodes, "
        f"p90 {ego['nodes_p90']:.0f}, max {ego['nodes_max']}; via concepts only: median "
        f"{ego['nodes_reached_only_via_concepts_median']:.0f}, p90 {ego['nodes_reached_only_via_concepts_p90']:.0f}; "
        f"passages median {ego['passages_median']:.0f}, p90 {ego['passages_p90']:.0f} of {ego['passages_total']}",
        f"Entity label collisions (case/space-insensitive): {col['groups']} groups, {col['nodes_involved']} nodes",
        f"Synthesized participation share of relation edges: {p['relation_edges_synthesized_share']:.1%}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="Pipeline run id (e.g. thesis-run-02).")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--seed", type=int, default=0, help="Seed for Louvain and the ego sample.")
    ap.add_argument("--ego-sample", type=int, default=300, help="Entities sampled for the ego-reach estimate.")
    ap.add_argument("--out", help="JSON output path (default: results/<id>/analysis/kg_structure_profile.json).")
    args = ap.parse_args()

    run_dir = Path(args.results_root) / args.id
    g = load(run_dir)
    p = profile(g, seed=args.seed, ego_sample=args.ego_sample)
    p["run_id"] = args.id
    p["graphml"] = str(run_dir / GRAPHML_REL)
    out = Path(args.out) if args.out else run_dir / "analysis" / "kg_structure_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summarize(p))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
