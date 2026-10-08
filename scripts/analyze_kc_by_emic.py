"""Knowledge coverage stratified by the emic validity of the reference pair (read-only).

KC scores an answer against the pair's reference answer, so a reference that the
generator flattened (caveat dropped, mechanism rewritten in neutral vocabulary)
makes KC measure the retrieval of the flattened version. This script asks whether
the arms' KC, and BM25's lead over the graph arms, differ between pairs the emic
judge scores as preserving the speaker's frame (EV >= 4) and pairs it scores as
having lost it (EV <= 3).

The emic judge is only tentatively reliable (test-retest alpha 0.681 overall,
0.554 above Remember, ``scripts/analyze_judge_retest.py``), so the strata are
built three ways:

* ``canonical``: EV of replica 1 (the published scores);
* ``stable``: pairs whose EV >= 4 classification is the same in all replicas
  (pairs that flip are dropped and counted);
* ``median``: median EV over the replicas.

Answers and gate verdicts are the canonical ones (replica 1); the other
replicas only enter through the emic strata. Per arm and stratum it reports KC
(mean correctness x faithfulness over the TC records), the TC count, correctness,
over-caution and evidence support (mean, pass rate >= tau). The BM25 minus graph
arm KC difference in each stratum, and the difference of those differences
(high minus low), get paired item bootstraps (items resampled within the stratum
pair set, the same draw for both arms; KC on each side is averaged over the
items that arm committed to, as in Table 3). ``kc_joint`` is the strictly paired
version over items both arms committed to. Spearman's rho between EV and
correctness on the TC records closes each arm block.

Pair sets: ``useful`` (gate-approved above Remember, the benchmark) and
``candidates`` (every pair above Remember, without the residual chunks under 100
characters, the base of the paper's RQ1 counts).

Run from the repo root:

    R=results/judge-retest
    uv run python -m scripts.analyze_kc_by_emic \\
        --replica $R/r1 --replica $R/r2 --replica $R/r3 --gate-from 3=$R/r3-gate \\
        --out-json $R/out/kc_by_emic.json --out-md $R/out/kc_by_emic.md \\
        --n-boot 2000 --seed 20261005
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_judge_retest import (
    GRAPH_ARMS,
    HIGHER,
    TAU,
    TAU_EMIC,
    load_replica,
    residual_pairs,
    useful_set,
)

ARMS = ["bm25", *GRAPH_ARMS]


def _strata(reps: list[dict[str, Any]], pids: list[str]) -> dict[str, dict[str, list[str]]]:
    """High (EV >= tau_e) and low pair lists under the three stratification rules."""
    out: dict[str, dict[str, list[str]]] = {}
    ev0 = reps[0]["emic"]
    out["canonical"] = {
        "high": [p for p in pids if ev0.get(p) is not None and ev0[p] >= TAU_EMIC],
        "low": [p for p in pids if ev0.get(p) is not None and ev0[p] < TAU_EMIC],
    }
    stable_hi, stable_lo, flips = [], [], []
    med_hi, med_lo = [], []
    for p in pids:
        evs = [r["emic"].get(p) for r in reps]
        if any(e is None for e in evs):
            continue
        hi = [e >= TAU_EMIC for e in evs]
        if all(hi):
            stable_hi.append(p)
        elif not any(hi):
            stable_lo.append(p)
        else:
            flips.append(p)
        (med_hi if statistics.median(evs) >= TAU_EMIC else med_lo).append(p)
    out["stable"] = {"high": stable_hi, "low": stable_lo, "flipped": flips}
    out["median"] = {"high": med_hi, "low": med_lo}
    return out


def _arm_arrays(facts: dict[tuple[str, str], dict[str, Any]], arm: str, pids: list[str]) -> dict:
    lab = np.array([facts[(arm, p)]["label"] for p in pids])
    kc = np.full(len(pids), np.nan)
    corr = np.full(len(pids), np.nan)
    pc = np.full(len(pids), np.nan)
    for j, p in enumerate(pids):
        f = facts[(arm, p)]
        c, fa = f["answer_correctness"], f["answer_faithfulness"]
        if f["label"] == "TC" and c is not None:
            corr[j] = c
            if fa is not None:
                kc[j] = c * fa
        if f["passage_coverage"] is not None and f["label"] != "unknown":
            pc[j] = f["passage_coverage"]
    return {"lab": lab, "kc": kc, "corr": corr, "pc": pc}


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else math.nan


def _summary(a: dict[str, np.ndarray]) -> dict[str, Any]:
    tc = int((a["lab"] == "TC").sum())
    fa = int((a["lab"] == "FA").sum())
    fin = np.isfinite(a["pc"])
    return {
        "n": len(a["lab"]),
        "n_tc": tc,
        "kc": _nanmean(a["kc"]),
        "correctness": _nanmean(a["corr"]),
        "oc": fa / (fa + tc) if fa + tc else math.nan,
        "pc": _nanmean(a["pc"]),
        "pc_pass": float((a["pc"][fin] >= TAU).mean()) if fin.any() else math.nan,
    }


def _ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    if not len(x):
        return [math.nan, math.nan]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _diffs(arrs: dict[str, dict[str, np.ndarray]], idx: np.ndarray, g: str) -> tuple[float, float]:
    """(KC bm25 - g, kc_joint bm25 - g) on the items ``idx``."""
    b, o = arrs["bm25"], arrs[g]
    d = _nanmean(b["kc"][idx]) - _nanmean(o["kc"][idx])
    both = np.isfinite(b["kc"][idx]) & np.isfinite(o["kc"][idx])
    dj = float((b["kc"][idx][both] - o["kc"][idx][both]).mean()) if both.any() else math.nan
    return d, dj


def contrast(
    facts: dict[tuple[str, str], dict[str, Any]],
    high: list[str],
    low: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Point estimates and bootstrap CIs of bm25 - graph KC in each stratum and their gap."""
    hi = {arm: _arm_arrays(facts, arm, high) for arm in ARMS}
    lo = {arm: _arm_arrays(facts, arm, low) for arm in ARMS}
    out: dict[str, Any] = {}
    for g in GRAPH_ARMS:
        rng = np.random.default_rng(seed)
        boot = np.full((n_boot, 6), np.nan)
        for b in range(n_boot):
            ih = rng.integers(0, len(high), len(high))
            il = rng.integers(0, len(low), len(low))
            dh, djh = _diffs(hi, ih, g)
            dl, djl = _diffs(lo, il, g)
            boot[b] = [dh, dl, dh - dl, djh, djl, djh - djl]
        dh, djh = _diffs(hi, np.arange(len(high)), g)
        dl, djl = _diffs(lo, np.arange(len(low)), g)
        point = [dh, dl, dh - dl, djh, djl, djh - djl]
        names = ["high", "low", "gap", "joint_high", "joint_low", "joint_gap"]
        out[g] = {n: {"est": float(point[k]), "ci": _ci(boot[:, k])} for k, n in enumerate(names)}
    return out


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    # Average ranks for ties.
    for v, r in ((np.array(x), rx), (np.array(y), ry)):
        for u in np.unique(v):
            m = v == u
            r[m] = r[m].mean()
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(reps: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    r0 = reps[0]
    c0 = r0["cep"]
    facts = r0["answers"]["facts"]
    approved = {p for p, m in c0.items() if m["approved"]}
    sets = {
        "useful": sorted(useful_set(c0, approved)),
        "candidates": sorted(
            p for p, m in c0.items() if m["bloom"] in HIGHER and p not in residual_pairs(c0)
        ),
    }
    res: dict[str, Any] = {"n_replicas": len(reps), "tau_emic": TAU_EMIC, "sets": {}}
    for set_name, pids in sets.items():
        pids = [p for p in pids if all((arm, p) in facts for arm in ARMS)]
        block: dict[str, Any] = {"n": len(pids), "rules": {}}
        for rule, st in _strata(reps, pids).items():
            rb: dict[str, Any] = {k: len(v) for k, v in st.items()}
            rb["arms"] = {}
            for arm in ARMS:
                rb["arms"][arm] = {
                    s: _summary(_arm_arrays(facts, arm, st[s])) for s in ("high", "low")
                }
            rb["contrast"] = contrast(facts, st["high"], st["low"], n_boot, seed)
            rb["by_level"] = {}
            for lvl in HIGHER:
                lh = [p for p in st["high"] if c0[p]["bloom"] == lvl]
                ll = [p for p in st["low"] if c0[p]["bloom"] == lvl]
                rb["by_level"][lvl] = {
                    "n_high": len(lh),
                    "n_low": len(ll),
                    "kc": {
                        arm: {
                            "high": _summary(_arm_arrays(facts, arm, lh))["kc"],
                            "low": _summary(_arm_arrays(facts, arm, ll))["kc"],
                        }
                        for arm in ARMS
                    },
                }
            block["rules"][rule] = rb
        rho: dict[str, Any] = {}
        ev0 = r0["emic"]
        for arm in ARMS:
            xs, ys = [], []
            for p in pids:
                f = facts[(arm, p)]
                if f["label"] == "TC" and f["answer_correctness"] is not None and ev0.get(p):
                    xs.append(float(ev0[p]))
                    ys.append(f["answer_correctness"])
            rho[arm] = {"n": len(xs), "rho": _spearman(xs, ys)}
        block["spearman_ev_correctness_tc"] = rho
        res["sets"][set_name] = block
    return res


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _ci_s(c: dict[str, Any]) -> str:
    return f"{_f(c['est'])} [{_f(c['ci'][0])}; {_f(c['ci'][1])}]"


def render_md(res: dict[str, Any]) -> str:
    lines = [
        "# KC estratificada pela validade êmica do par",
        "",
        f"Corte EV >= {res['tau_emic']}; réplicas êmicas: {res['n_replicas']}. "
        "Respostas e portão canônicos (réplica 1).",
        "",
    ]
    for set_name, block in res["sets"].items():
        lines += [f"## Conjunto `{set_name}` (n = {block['n']})", ""]
        for rule, rb in block["rules"].items():
            counts = ", ".join(f"{k} {v}" for k, v in rb.items() if isinstance(v, int))
            lines += [f"### Estrato `{rule}` ({counts})", ""]
            lines += [
                "| Braço | estrato | n | TC | KC | correção | OC | ES média | ES aprov. |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for arm, by in rb["arms"].items():
                for s, m in by.items():
                    lines.append(
                        f"| {arm} | {s} | {m['n']} | {m['n_tc']} | {_f(m['kc'])} | "
                        f"{_f(m['correctness'])} | {_f(m['oc'])} | {_f(m['pc'])} | "
                        f"{_f(m['pc_pass'])} |"
                    )
            lines += [
                "",
                "KC bm25 menos braço de grafo (IC 95% bootstrap por item):",
                "",
                "| Braço | EV alta | EV baixa | alta menos baixa | conj. alta | conj. baixa "
                "| conj. alta menos baixa |",
                "|---|---|---|---|---|---|---|",
            ]
            for g, c in rb["contrast"].items():
                lines.append(
                    f"| {g} | {_ci_s(c['high'])} | {_ci_s(c['low'])} | {_ci_s(c['gap'])} | "
                    f"{_ci_s(c['joint_high'])} | {_ci_s(c['joint_low'])} | "
                    f"{_ci_s(c['joint_gap'])} |"
                )
            lines += ["", "KC por nível (alta / baixa):", ""]
            lines += ["| Nível | n alta | n baixa | " + " | ".join(a for a in ARMS) + " |"]
            lines += ["|---|---|---|" + "---|" * len(ARMS)]
            for lvl, lb in rb["by_level"].items():
                cells = " | ".join(
                    f"{_f(lb['kc'][a]['high'])} / {_f(lb['kc'][a]['low'])}" for a in ARMS
                )
                lines.append(f"| {lvl} | {lb['n_high']} | {lb['n_low']} | {cells} |")
            lines.append("")
        lines += ["Spearman EV x correção nos TC (EV canônica):", ""]
        for arm, r in block["spearman_ev_correctness_tc"].items():
            lines.append(f"- {arm}: rho = {_f(r['rho'])} (n = {r['n']})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="KC stratified by emic validity.")
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
    args.out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    args.out_md.write_text(render_md(res))
    print(render_md(res))


if __name__ == "__main__":
    main()
