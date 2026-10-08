"""The emic judge as a gate criterion: what changes if useful pairs need EV >= 4 (read-only).

The paper's benchmark is the set of pairs above Remember that pass the
conjunctive judge-qa gate. Here the emic judge joins the gate with the
pre-registered threshold tau_e = 4, so a useful pair must also preserve the
speaker's frame. The script recomputes, for the published gate and for the
emic gate, the RQ1 yield (per level, chunks and documents reached) and the
retrieval table (KC, hallucination, over-caution, abstention F1, evidence
support mean and pass rate, conditional over-caution, KC per level), and the
paired bootstrap of BM25 minus each graph arm over items x replicas.

Two sources of variation are kept apart, as in ``analyze_judge_retest.py``:

* ``fixed``: the canonical emic-gated set (gate and EV of replica 1) for every
  replica, so only the answer judges vary;
* ``own``: each replica's own gate verdicts and own EV, so gate, emic and
  answer judges all vary.

Run from the repo root:

    R=results/judge-retest
    uv run python -m scripts.analyze_emic_gate \\
        --replica $R/r1 --replica $R/r2 --replica $R/r3 --gate-from 3=$R/r3-gate \\
        --cache-dir $R/cache --out-json $R/out/emic_gate.json \\
        --out-md $R/out/emic_gate.md --n-boot 2000 --seed 20261005
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_judge_retest import (
    ARMS,
    GRAPH_ARMS,
    HIGHER,
    TAU_EMIC,
    arm_aggregates,
    load_replica,
    paired_bootstrap,
    residual_pairs,
    useful_set,
)


def gate_sets(rep: dict[str, Any]) -> dict[str, set[str]]:
    """Published useful set and its emic-gated subset for one replica."""
    cep = rep["cep"]
    approved = {p for p, m in cep.items() if m["approved"]}
    useful = useful_set(cep, approved)
    ev = rep["emic"]
    return {
        "published": useful,
        "emic_gate": {p for p in useful if ev.get(p) is not None and ev[p] >= TAU_EMIC},
    }


def yield_block(cep: dict[str, dict[str, Any]], useful: set[str]) -> dict[str, Any]:
    """Useful pairs per level, rate over candidates, chunks and documents reached."""
    resid = residual_pairs(cep)
    cands = [p for p, m in cep.items() if m["bloom"] in HIGHER and p not in resid]
    chunks = {(m["file"], m["chunk"]) for p, m in cep.items() if p not in resid}
    docs = {m["file"] for p, m in cep.items() if p not in resid}
    u = [p for p in useful if p not in resid]
    return {
        "n_useful": len(u),
        "n_candidates": len(cands),
        "yield": len(u) / len(cands),
        "by_level": {
            lvl: {
                "n": sum(cep[p]["bloom"] == lvl for p in u),
                "of": sum(cep[p]["bloom"] == lvl for p in cands),
            }
            for lvl in HIGHER
        },
        "chunks_reached": len({(cep[p]["file"], cep[p]["chunk"]) for p in u}),
        "chunks_total": len(chunks),
        "docs_reached": len({cep[p]["file"] for p in u}),
        "docs_total": len(docs),
    }


def _arm_rows(aggs: dict[str, Any]) -> dict[str, Any]:
    keep = ["kc", "kc_n", "hall", "hall_ci", "oc", "f1_abs", "pc", "pc_pass", "cond_oc"]
    keep += ["cond_oc_n", "n_answerable", "n_nonans"]
    return {
        arm: {k: aggs[arm][k] for k in keep}
        | {
            "kc_by_level": {lvl: aggs[arm]["by_level"][lvl]["kc"] for lvl in HIGHER},
            "tc_by_level": {lvl: aggs[arm]["by_level"][lvl]["tc"] for lvl in HIGHER},
        }
        for arm in ARMS
        if arm in aggs
    }


def analyse(reps: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    sets = [gate_sets(r) for r in reps]
    c0 = reps[0]["cep"]
    res: dict[str, Any] = {"tau_emic": TAU_EMIC, "scenarios": {}}
    for name in ("published", "emic_gate"):
        canon = sets[0][name]
        per_rep = [
            _arm_rows(arm_aggregates(r["answers"]["records"], r["cep"], s[name]))
            for r, s in zip(reps, sets, strict=True)
        ]
        res["scenarios"][name] = {
            "yield_canonical": yield_block(c0, canon),
            "yield_by_replica": [
                yield_block(r["cep"], s[name]) for r, s in zip(reps, sets, strict=True)
            ],
            "overlap_all_replicas": len(set.intersection(*(s[name] for s in sets))),
            "arms_canonical": _arm_rows(arm_aggregates(reps[0]["answers"]["records"], c0, canon)),
            "arms_by_replica_own_set": per_rep,
            "boot_fixed": paired_bootstrap(reps, [canon] * len(reps), n_boot, seed)["bm25_minus"],
            "boot_own": paired_bootstrap(reps, [s[name] for s in sets], n_boot, seed)["bm25_minus"],
        }
    return res


def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(c: dict[str, Any]) -> str:
    lo, hi = c["ci"]
    star = " *" if c["excludes_zero"] else ""
    return f"{_f(c['point'])} [{_f(lo)}; {_f(hi)}]{star}"


def render_md(res: dict[str, Any]) -> str:
    out = [
        "# Juiz êmico como critério do portão (EV >= 4)",
        "",
        "`published`: portão do paper; `emic_gate`: portão e EV >= 4. "
        "IC 95% bootstrap pareado itens x réplicas; * exclui zero.",
        "",
    ]
    for name, sc in res["scenarios"].items():
        y = sc["yield_canonical"]
        out += [f"## `{name}`", ""]
        lv = ", ".join(f"{k} {v['n']}/{v['of']}" for k, v in y["by_level"].items())
        out.append(
            f"Úteis {y['n_useful']} de {y['n_candidates']} ({y['yield'] * 100:.1f}%); {lv}; "
            f"chunks {y['chunks_reached']}/{y['chunks_total']}; "
            f"documentos {y['docs_reached']}/{y['docs_total']}; "
            f"úteis em todas as réplicas: {sc['overlap_all_replicas']}; "
            "úteis por réplica: " + ", ".join(str(b["n_useful"]) for b in sc["yield_by_replica"])
        )
        out += [
            "",
            "| Braço | KC | Und | Ana | Eva | Hall [Wilson] | OC | F1abs | ES média | ES aprov. "
            "| OC cond. (n) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, m in sc["arms_canonical"].items():
            kl = m["kc_by_level"]
            out.append(
                f"| {arm} | {_f(m['kc'])} | {_f(kl['understand'])} | {_f(kl['analyze'])} | "
                f"{_f(kl['evaluate'])} | {_f(m['hall'])} [{_f(m['hall_ci'][0])}; "
                f"{_f(m['hall_ci'][1])}] | {_f(m['oc'])} | {_f(m['f1_abs'])} | {_f(m['pc'])} | "
                f"{_f(m['pc_pass'])} | {_f(m['cond_oc'])} ({m['cond_oc_n']}) |"
            )
        out += ["", "Itens TC por nível (Und/Ana/Eva): "]
        for arm, m in sc["arms_canonical"].items():
            t = m["tc_by_level"]
            out.append(f"- {arm}: {t['understand']} / {t['analyze']} / {t['evaluate']}")
        out += ["", "KC por réplica (conjunto próprio de cada réplica):", ""]
        for arm in sc["arms_canonical"]:
            vals = [_f(r[arm]["kc"]) for r in sc["arms_by_replica_own_set"]]
            out.append(f"- {arm}: {' / '.join(vals)}")
        for boot in ("boot_fixed", "boot_own"):
            out += ["", f"BM25 menos braço de grafo, `{boot}`:", ""]
            out += ["| Nível | Braço | KC | KC conj. | Hall | OC | ES média | ES aprov. |"]
            out += ["|---|---|---|---|---|---|---|---|"]
            for lvl, by in sc[boot].items():
                for g in GRAPH_ARMS:
                    d = by[g]
                    out.append(
                        f"| {lvl} | {g} | {_ci(d['kc'])} | {_ci(d['kc_joint'])} | "
                        f"{_ci(d['hall'])} | {_ci(d['oc'])} | {_ci(d['pc'])} | "
                        f"{_ci(d['pc_pass'])} |"
                    )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emic judge as a gate criterion.")
    parser.add_argument("--replica", action="append", required=True, type=Path)
    parser.add_argument("--gate-from", action="append", default=[])
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    gate_from = {int(k): Path(v) for k, v in (g.split("=", 1) for g in args.gate_from)}
    reps = [
        load_replica(p, args.cache_dir, gate_from.get(i + 1)) for i, p in enumerate(args.replica)
    ]
    res = analyse(reps, args.n_boot, args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    md = render_md(res)
    args.out_md.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
