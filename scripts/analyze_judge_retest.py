"""Test-retest (intra-judge) reliability of the three LLM judges (read-only).

Implements the analysis pre-registered in
``docs/planning/2026-09-28-intrajudge-reliability-design.md``: the same corpus is
re-judged R times under an identical configuration, and the replicas are treated
as coders. Replica 1 is the canonical ``thesis-run-02`` and supplies the
canonical verdicts; majority vote / median only enter as sensitivity analyses.
Thresholds are the pre-registered ones (gate tau = 0.625, emic tau_e = 4,
abstention tau read from the criterion) and are never tuned here.

Each ``--replica`` is the directory a round's export tar was extracted to, i.e.
it holds ``cep/outputs``, ``judge_qa``, ``emic_judge`` and ``judge_answers``.
The first ``--replica`` is the canonical one. ``--gate-from N=DIR`` reads
replica N's gate verdicts from another export of the same round (used when the
round's final tar was taken after its CEP verdicts were cleared for the next
round); the replica's own export is still checked against it.

What it computes (sections of the JSON / Markdown output):

1. ``config``: model, provider, temperature, thresholds and prompt hashes of
   every judge stage per replica (from each stage's ``run_metadata.json``), plus
   identity checks on the frozen inputs (CEP pairs, answers) and the coverage of
   ``judge_answers`` per replica. A ``judge_answers`` round that did not finish
   is detected from its checkpoint: only records listed as completed in that
   round's checkpoint count as re-judged; the rest are carried over from the
   previous round and are treated as missing.
1b. ``completeness``: item counts per judge and replica against replica 1,
   duplicates, and records whose judge output repeats an earlier replica's.
1c. ``provenance``: prompt digests at the analysed commit against the gate's
   recorded ones, last prompt change against each replica's start, and any
   recorded Ollama or image version.
2. ``sanity``: the published canonical numbers, recomputed on replica 1.
3. ``gate``: Krippendorff's alpha (ordinal, scale 0..4) and Gwet's AC2
   (quadratic) per criterion, nominal alpha / AC2 / pairwise Cohen kappa for the
   binary verdict, item instability D_i = 2 k_i (R - k_i) / (R (R - 1)), score
   amplitude, mean signed and absolute differences, margin dependence; over the
   total dataset, the useful candidates (every pair above Remember) and each
   Bloom level.
4. ``emic``: the same for the 1..5 emic score and the binary EV >= 4 filter.
5. ``answers``: the same per answer-judge criterion, for the evidence-support
   pass cut (``passage_coverage >= tau``, nominal) and for the TC/FC/FA/TA
   cell, in the total, candidates and useful record strata, by arm and level.
5b. ``rejection_emic``: emic score of gate-rejected pairs by failing criterion.
6. ``aggregates``: every published aggregate per replica in the total and
   candidates strata (answer-judge noise only) and in the useful stratum under
   three sources of variation (answer judge with the canonical useful set; gate
   with the canonical answers; both), the qualitative conclusions (a) to (e),
   and paired bootstraps over items x replicas for the BM25-vs-graph
   differences, overall and per level.

Krippendorff's alpha, AC2 and weighted kappa come from
``arandu.shared.agreement.coefficients`` (first-principles implementations with
a fixed scale). If the ``krippendorff`` package is importable its alpha is
reported alongside as a cross-check. Alpha CIs use a vectorised item bootstrap
whose point estimate is asserted equal to the arandu one.

Run from the repo root (``results/judge-retest`` holds the extracted tars):

    R=results/judge-retest
    uv run --with krippendorff python -m scripts.analyze_judge_retest \\
        --replica $R/r1 --replica $R/r2 --replica $R/r3 --gate-from 3=$R/r3-gate \\
        --out-json $R/out/retest.json --out-md $R/out/retest.md \\
        --n-boot 2000 --seed 20261005
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import pickle
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from arandu.qa.schemas import QARecordCEP
from arandu.shared.agreement.coefficients import (
    cohen_kappa_weighted,
    gwet_ac2,
    krippendorff_alpha,
)
from arandu.shared.emic.schemas import EmicSourceScores
from arandu.shared.rag.analysis.classifier import classify_record
from arandu.shared.rag.analysis.metrics import aggregate_arm
from arandu.shared.rag.schemas import AnswerRecord
from scripts._judge_analysis_common import DEFAULT_THRESHOLD, approved_ids, load_pair_scores

TAU = DEFAULT_THRESHOLD
TAU_EMIC = 4
ARMS = ["null", "bm25", "khop_passage", "khop_triple", "atlas_rag"]
GRAPH_ARMS = ["khop_passage", "khop_triple", "atlas_rag"]
BLOOM = ["remember", "understand", "analyze", "evaluate"]
HIGHER = ["understand", "analyze", "evaluate"]
GATE_CRITERIA = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
ANS_CRITERIA = ["abstention", "passage_coverage", "answer_correctness", "answer_faithfulness"]
LABELS = ["TA", "TC", "FA", "FC"]
RESIDUAL_CHUNK_CHARS = 100

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_cep(cep_dir: Path) -> dict[str, dict[str, Any]]:
    """``qa_pair_id -> pair facts`` with the official id and approval rule."""
    scores = load_pair_scores(cep_dir)
    approved = approved_ids(scores, TAU)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(cep_dir.glob("*.json")):
        rec = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        for idx, pair in enumerate(rec.qa_pairs):
            pid = f"{rec.source_file_id}:{pair.chunk_id or 'none'}:{idx}"
            v = pair.validation
            out[pid] = {
                "bloom": str(pair.bloom_level),
                "scores": scores.get(pid, {}),
                "approved": pid in approved,
                "passed": None if v is None else bool(v.passed),
                "file": rec.source_file_id,
                "idx": idx,
                "chunk": pair.chunk_id,
                "ctx_len": len(pair.context or ""),
                "content_sha": _sha([pair.question, pair.answer, pair.context, pair.bloom_level]),
                "validation_sha": None if v is None else _sha(v.model_dump()),
            }
    return out


def load_emic(
    emic_dir: Path, cep: dict[str, dict[str, Any]]
) -> tuple[dict[str, int | None], dict[str, int]]:
    """``qa_pair_id -> emic score`` joined through (source_file_id, pair_index).

    Also returns the raw entry count and how many entries repeat a pair, so the
    completeness check can flag duplicated or missing scores.
    """
    by_file_idx = {(m["file"], m["idx"]): pid for pid, m in cep.items()}
    out: dict[str, int | None] = {}
    shas: dict[str, str] = {}
    entries = dups = 0
    for path in sorted(emic_dir.glob("*.json")):
        rec = EmicSourceScores.model_validate_json(path.read_text(encoding="utf-8"))
        for s in rec.scores:
            pid = by_file_idx[(rec.source_file_id, s.pair_index)]
            entries += 1
            dups += pid in out
            out[pid] = s.emic_score
            shas[pid] = _sha(s.model_dump())
    counts = {"entries": entries, "duplicates": dups, "files": len(list(emic_dir.glob("*.json")))}
    return out, counts | {"entry_sha": shas}


def _tree_sha(root: Path) -> str | None:
    """Digest of every file under ``root`` (relative name + bytes), or None if absent."""
    if not root.exists():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _criterion(rec: AnswerRecord, name: str) -> float | None:
    if rec.validation is None:
        return None
    for step in rec.validation.stage_results.values():
        cs = step.criterion_scores.get(name)
        if cs is not None:
            return None if cs.error is not None or cs.score is None else cs.score
    return None


def load_answers(root: Path) -> dict[str, Any]:
    """Judged AnswerRecords keyed by (arm, qa_pair_id), plus checkpoint membership."""
    ckpt_path = root / "judge_answers_checkpoint.json"
    ckpt_raw = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else None
    ckpt = set(ckpt_raw["completed_files"]) if ckpt_raw is not None else None
    recs: dict[tuple[str, str], AnswerRecord] = {}
    facts: dict[tuple[str, str], dict[str, Any]] = {}
    files_by_arm_kind: Counter[str] = Counter()
    dup_keys = 0
    for path in sorted((root / "outputs").glob("*/*/*.json")):
        arm, kind = path.parts[-3], path.parts[-2]
        rec = AnswerRecord.load(path)
        key = (arm, rec.qa_pair_id)
        files_by_arm_kind[f"{arm}/{kind}"] += 1
        dup_keys += key in recs
        recs[key] = rec
        facts[key] = {
            "kind": kind,
            "label": classify_record(rec),
            "answerable": rec.is_answerable,
            "abstained_flag": rec.abstained,
            "in_checkpoint": None if ckpt is None else f"{arm}::{kind}::{path.stem}" in ckpt,
            "validation_sha": _sha(None if rec.validation is None else rec.validation.model_dump()),
            "answer_sha": _sha([rec.answer_text, rec.abstained, rec.is_answerable]),
            **{c: _criterion(rec, c) for c in ANS_CRITERIA},
        }
        pc = facts[key]["passage_coverage"]
        # Evidence support pass cut (the paper's pass rate): score >= tau.
        facts[key]["passage_coverage_pass"] = None if pc is None else int(pc >= TAU)
    return {
        "records": recs,
        "facts": facts,
        "checkpoint_size": None if ckpt is None else len(ckpt),
        "checkpoint_failed": None if ckpt_raw is None else len(ckpt_raw.get("failed_files") or {}),
        "checkpoint_total": None if ckpt_raw is None else ckpt_raw.get("total_files"),
        "checkpoint_started_at": None if ckpt_raw is None else ckpt_raw.get("started_at"),
        "files_by_arm_kind": dict(sorted(files_by_arm_kind.items())),
        "duplicate_keys": dup_keys,
    }


def load_replica(path: Path, cache_dir: Path | None, gate_from: Path | None) -> dict[str, Any]:
    """Load one replica directory (optionally pickled to ``cache_dir``).

    ``gate_from`` is an alternative export of the same round from which the gate
    verdicts (``cep/outputs``) and the ``judge_qa`` metadata are read, for a round
    whose final tar was taken after its CEP verdicts had already been cleared for
    the next round. The replica's own ``cep/outputs`` is still loaded, to verify
    that only the verdicts differ and that the emic outputs of both exports match.
    """
    gate_root = gate_from if gate_from is not None else path
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{path.resolve()}|{gate_root.resolve()}|v2"
        cache = cache_dir / f"{hashlib.sha1(tag.encode()).hexdigest()}.pkl"
        if cache.exists():
            return pickle.loads(cache.read_bytes())
    cep = load_cep(gate_root / "cep" / "outputs")
    emic, emic_counts = load_emic(path / "emic_judge" / "outputs", cep)
    meta = {
        stage: json.loads((root / stage / "run_metadata.json").read_text())
        for stage, root in (
            ("judge_qa", gate_root),
            ("emic_judge", path),
            ("judge_answers", path),
        )
        if (root / stage / "run_metadata.json").exists()
    }
    gate_source: dict[str, Any] = {"cep_dir": str(gate_root / "cep" / "outputs")}
    if gate_from is not None:
        own = load_cep(path / "cep" / "outputs")
        gate_source |= {
            "overridden": True,
            "own_cep_pairs": len(own),
            "own_cep_pairs_with_verdict": sum(m["passed"] is not None for m in own.values()),
            "same_ids_as_override": set(own) == set(cep),
            "content_mismatches_vs_override": sum(
                1 for p in own if p in cep and own[p]["content_sha"] != cep[p]["content_sha"]
            ),
            "emic_outputs_identical": _tree_sha(path / "emic_judge")
            == _tree_sha(gate_root / "emic_judge"),
            "judge_qa_dir_identical": _tree_sha(path / "judge_qa")
            == _tree_sha(gate_root / "judge_qa"),
        }
    else:
        gate_source["overridden"] = False
    rep = {
        "path": str(path),
        "cep": cep,
        "emic": emic,
        "emic_counts": emic_counts,
        "answers": load_answers(path / "judge_answers"),
        "meta": meta,
        "gate_source": gate_source,
        "history": sorted(p.name for p in (gate_root / "judge_qa" / "history").glob("*.json"))
        if (gate_root / "judge_qa" / "history").exists()
        else [],
    }
    if cache is not None:
        cache.write_bytes(pickle.dumps(rep))
    return rep


# --------------------------------------------------------------------------- #
# Coefficients
# --------------------------------------------------------------------------- #


def to_ord(x: float | None) -> int | None:
    """Map a 0..1 score on the 0.25 grid to 0..4."""
    if x is None:
        return None
    v = x / 0.25
    iv = round(v)
    if abs(v - iv) > 1e-9:
        raise ValueError(f"score {x} is not on the 0.25 grid")
    return int(iv)


def _unit_coincidences(units: list[list[int | None]], cats: list[int]) -> np.ndarray:
    """Per-unit Krippendorff coincidence contributions, shape (n_units, K*K)."""
    pos = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    rows = []
    for u in units:
        present = [r for r in u if r is not None]
        m = len(present)
        row = np.zeros(k * k)
        if m >= 2:
            for i in range(m):
                for j in range(m):
                    if i != j:
                        row[pos[present[i]] * k + pos[present[j]]] += 1.0 / (m - 1)
        rows.append(row)
    return np.array(rows) if rows else np.zeros((0, k * k))


def _alpha_from_o(o: np.ndarray, cats: list[int], level: str) -> float | None:
    k = len(cats)
    o = o.reshape(k, k)
    n = o.sum()
    if n <= 1:
        return None
    marg = o.sum(axis=1)
    delta = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if level == "nominal":
                delta[a, b] = 0.0 if a == b else 1.0
            else:
                lo, hi = min(a, b), max(a, b)
                term = marg[lo : hi + 1].sum() - (marg[a] + marg[b]) / 2.0
                delta[a, b] = term * term
    d_o = (o * delta).sum() / n
    d_e = (np.outer(marg, marg) * delta).sum() / (n * (n - 1))
    if d_e == 0:
        return None
    return float(1.0 - d_o / d_e)


def alpha_with_ci(
    units: list[list[int | None]],
    scale: tuple[int, int],
    level: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Arandu alpha point estimate + vectorised percentile bootstrap CI over units."""
    res = krippendorff_alpha(units, level=level, scale=scale)  # type: ignore[arg-type]
    cats = list(range(scale[0], scale[1] + 1))
    contrib = _unit_coincidences(units, cats)
    usable = contrib[contrib.sum(axis=1) > 0]
    point_np = _alpha_from_o(usable.sum(axis=0), cats, level) if len(usable) else None
    if res.coefficient is not None and point_np is not None:
        assert abs(res.coefficient - point_np) < 1e-9, (res.coefficient, point_np)
    lo = hi = None
    if n_boot > 0 and len(usable) > 1 and res.coefficient is not None:
        n = len(usable)
        boots = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n)
            a = _alpha_from_o(w @ usable, cats, level)
            if a is not None:
                boots.append(a)
        if boots:
            lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    ext = None
    if KRIPP is not None and res.coefficient is not None:
        data = np.array([[np.nan if r is None else r for r in u] for u in units], dtype=float).T
        ext = float(
            KRIPP.alpha(
                reliability_data=data,
                level_of_measurement=level,
                value_domain=cats,
            )
        )
    return {"alpha": res.coefficient, "ci": [lo, hi], "n_items": res.n_items, "alpha_pkg": ext}


def reliability_block(
    units: list[list[int | None]],
    scale: tuple[int, int],
    level: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Alpha (+CI), AC2, pairwise agreement / kappa, Delta and amplitude for one table cell."""
    r = len(units[0]) if units else 0
    out: dict[str, Any] = alpha_with_ci(units, scale, level, n_boot, rng)
    weights = "quadratic"
    # Weighted kappa / AC2 are meaningful for ordinal scales and, with two
    # categories, reduce to Cohen's kappa / Gwet's AC1; for a nominal scale with
    # more than two categories they are not defined here.
    weighted_ok = level != "nominal" or scale[1] - scale[0] == 1
    out["ac2"] = gwet_ac2(units, weights=weights, scale=scale).coefficient if weighted_ok else None
    pairs = {}
    for a in range(r):
        for b in range(a + 1, r):
            xa = [u[a] for u in units]
            xb = [u[b] for u in units]
            both = [(x, y) for x, y in zip(xa, xb, strict=True) if x is not None and y is not None]
            kap = (
                cohen_kappa_weighted(xa, xb, weights=weights, scale=scale).coefficient
                if weighted_ok
                else None
            )
            exact = sum(x == y for x, y in both) / len(both) if both else None
            diffs = [y - x for x, y in both]
            pairs[f"{a + 1}-{b + 1}"] = {
                "n": len(both),
                "exact_agreement": exact,
                "kappa": kap,
                "mean_delta": float(np.mean(diffs)) if diffs else None,
                "mean_abs_delta": float(np.mean(np.abs(diffs))) if diffs else None,
            }
    out["pairwise"] = pairs
    amps = Counter()
    n_use = 0
    for u in units:
        present = [x for x in u if x is not None]
        if len(present) >= 2:
            n_use += 1
            amps[max(present) - min(present)] += 1
    out["amplitude"] = {str(k): v for k, v in sorted(amps.items())}
    out["frac_amplitude_ge2"] = (
        sum(v for k, v in amps.items() if k >= 2) / n_use if n_use and level != "nominal" else None
    )
    out["frac_all_equal"] = amps.get(0, 0) / n_use if n_use else None
    return out


def instability(binary_units: list[list[int | None]]) -> dict[str, Any]:
    """k_i distribution and D_i = 2 k (R-k) / (R (R-1)) over complete items."""
    complete = [u for u in binary_units if all(x is not None for x in u)]
    if not complete:
        return {"n": 0}
    r = len(complete[0])
    ks = Counter(sum(u) for u in complete)
    d = [2 * k * (r - k) / (r * (r - 1)) for u in complete for k in [sum(u)]]
    return {
        "n": len(complete),
        "R": r,
        "k_counts": {str(k): ks.get(k, 0) for k in range(r + 1)},
        "stable_positive": ks.get(r, 0),
        "unstable": sum(ks.get(k, 0) for k in range(1, r)),
        "stable_negative": ks.get(0, 0),
        "frac_unstable": sum(ks.get(k, 0) for k in range(1, r)) / len(complete),
        "mean_D": float(np.mean(d)),
    }


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def useful_set(cep: dict[str, dict[str, Any]], approved: set[str]) -> set[str]:
    """Gate-approved pairs above Remember."""
    return {pid for pid in approved if cep[pid]["bloom"] != "remember"}


def residual_pairs(cep: dict[str, dict[str, Any]]) -> set[str]:
    """Pairs of the residual chunks (< 100 characters) excluded from the RQ1 counts."""
    return {pid for pid, m in cep.items() if m["ctx_len"] < RESIDUAL_CHUNK_CHARS}


def gate_aggregates(cep: dict[str, dict[str, Any]], approved: set[str]) -> dict[str, Any]:
    """Approval counts, per-level pass and useful yield (base 2,652 and 2,670)."""
    resid = residual_pairs(cep)
    base = set(cep) - resid
    by_level = {}
    for lvl in BLOOM:
        ids = {p for p in base if cep[p]["bloom"] == lvl}
        by_level[lvl] = {
            "n": len(ids),
            "pass": len(ids & approved),
            "rate": len(ids & approved) / len(ids),
        }
    cand = {p for p in base if cep[p]["bloom"] != "remember"}
    useful = cand & approved
    return {
        "approved_all": len(approved),
        "n_all": len(cep),
        "residual_chunks": len({cep[p]["chunk"] for p in resid}),
        "residual_pairs": len(resid),
        "approved_base": len(approved & base),
        "n_base": len(base),
        "by_level": by_level,
        "useful_all": len(useful_set(cep, approved)),
        "useful_base": len(useful),
        "candidates_base": len(cand),
        "useful_yield": len(useful) / len(cand),
    }


def emic_aggregates(
    emic: dict[str, int | None], cep: dict[str, dict[str, Any]], approved: set[str]
) -> dict[str, Any]:
    """Distribution of EV overall, by level, and by gate verdict."""

    def dist(vals: list[int]) -> dict[str, Any]:
        n = len(vals)
        c = Counter(vals)
        return {
            "n": n,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "pct": {str(k): 100 * c.get(k, 0) / n for k in range(1, 6)},
            "pct_ge4": 100 * sum(v >= TAU_EMIC for v in vals) / n,
            "pct_le2": 100 * sum(v <= 2 for v in vals) / n,
        }

    scored = {p: v for p, v in emic.items() if v is not None}
    out = {"all": dist(list(scored.values())), "n_missing": len(emic) - len(scored)}
    out["by_level"] = {
        lvl: dist([v for p, v in scored.items() if cep[p]["bloom"] == lvl]) for lvl in BLOOM
    }
    means = [out["by_level"][lvl]["mean"] for lvl in BLOOM]
    ge4 = [out["by_level"][lvl]["pct_ge4"] for lvl in BLOOM]
    out["monotone_mean"] = all(a > b for a, b in itertools.pairwise(means))
    out["monotone_ge4"] = all(a > b for a, b in itertools.pairwise(ge4))
    out["by_verdict"] = {
        "approved": dist([v for p, v in scored.items() if p in approved]),
        "rejected": dist([v for p, v in scored.items() if p not in approved]),
    }
    out["by_level_verdict"] = {
        lvl: {
            "approved": dist(
                [v for p, v in scored.items() if p in approved and cep[p]["bloom"] == lvl]
            ),
            "rejected": dist(
                [v for p, v in scored.items() if p not in approved and cep[p]["bloom"] == lvl]
            ),
        }
        for lvl in BLOOM
    }
    return out


def _pm(m: Any) -> float | None:
    return getattr(m, "value", None) if hasattr(m, "value") else getattr(m, "mean", None)


def _pass_rate(recs: list[AnswerRecord]) -> tuple[float | None, int]:
    """Evidence-support pass rate (score >= tau) over classified records with a score.

    Same base as ``analyze_conditional_overcaution.py`` (the paper's Pass column):
    answerable records whose cell is known and whose passage coverage was scored.
    """
    vals = [
        c
        for r in recs
        if classify_record(r) != "unknown" and (c := _criterion(r, "passage_coverage")) is not None
    ]
    return (sum(c >= TAU for c in vals) / len(vals) if vals else None), len(vals)


def arm_aggregates(
    records: dict[tuple[str, str], AnswerRecord],
    cep: dict[str, dict[str, Any]],
    useful: set[str],
    restrict: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Published retrieval table over ``useful`` (+ nonans probes with a seed in it).

    ``useful`` is any pair set: the gate-approved higher-order pairs (the
    paper's table), every candidate above Remember, or the whole dataset. Each
    Bloom level present in the set gets KC, evidence support (mean and pass
    rate), OC, conditional OC and, from the probes whose seed is at that level,
    hallucination.
    """
    out = {}
    levels = [lvl for lvl in BLOOM if any(cep[p]["bloom"] == lvl for p in useful)]
    for arm in ARMS:
        ans = []
        non = []
        for (a, pid), rec in records.items():
            if a != arm or (restrict is not None and (a, pid) not in restrict):
                continue
            if rec.is_answerable and pid in useful:
                ans.append(rec)
            elif not rec.is_answerable and pid.removesuffix(":nonans") in useful:
                non.append(rec)
        joint = aggregate_arm(arm, ans + non)
        a_only = aggregate_arm(arm, ans)
        cond = aggregate_arm(
            arm,
            [r for r in ans if (_criterion(r, "passage_coverage") or -1) >= TAU],
        )
        by_level = {}
        for lvl in levels:
            la = [r for r in ans if cep[r.qa_pair_id]["bloom"] == lvl]
            ln = [r for r in non if cep[r.qa_pair_id.removesuffix(":nonans")]["bloom"] == lvl]
            m = aggregate_arm(arm, la)
            mn = aggregate_arm(arm, ln)
            mc = aggregate_arm(
                arm, [r for r in la if (_criterion(r, "passage_coverage") or -1) >= TAU]
            )
            pp, ppn = _pass_rate(la)
            by_level[lvl] = {
                "n_answerable": len(la),
                "n_nonans": len(ln),
                "kc": m.knowledge_coverage.mean,
                "kc_n": m.knowledge_coverage.n,
                "tc": m.confusion["TC"],
                "pc": m.passage_coverage.mean,
                "pc_pass": pp,
                "pc_pass_n": ppn,
                "oc": m.over_cautiousness_rate.value,
                "cond_oc": mc.over_cautiousness_rate.value,
                "cond_oc_n": mc.over_cautiousness_rate.denominator,
                "hall": mn.hallucination_rate.value,
            }
        pp, ppn = _pass_rate(ans)
        out[arm] = {
            "n_answerable": len(ans),
            "n_nonans": len(non),
            "kc": a_only.knowledge_coverage.mean,
            "kc_n": a_only.knowledge_coverage.n,
            "oc": a_only.over_cautiousness_rate.value,
            "pc": a_only.passage_coverage.mean,
            "pc_pass": pp,
            "pc_pass_n": ppn,
            "hall": joint.hallucination_rate.value,
            "hall_ci": [joint.hallucination_rate.ci_lower, joint.hallucination_rate.ci_upper],
            "f1_abs": joint.abstention_f1,
            "cond_oc": cond.over_cautiousness_rate.value,
            "cond_oc_n": cond.over_cautiousness_rate.denominator,
            "confusion": joint.confusion,
            "by_level": by_level,
        }
    return out


def _gt(a: float | None, b: float | None) -> bool | None:
    """``a > b``, or None when either side is undefined (empty slice)."""
    return None if a is None or b is None else a > b


def _all(vals: list[bool | None]) -> bool | None:
    """All true; None when any comparison was undefined."""
    return None if any(v is None for v in vals) else all(vals)


def conclusions(
    arms: dict[str, Any] | None, emic: dict[str, Any] | None, gate: dict[str, Any] | None
) -> dict[str, Any]:
    """Qualitative conclusions (a)-(e) evaluated on one scenario (point estimates)."""
    out: dict[str, Any] = {}
    if arms is not None:
        b = arms["bm25"]
        lv = {a: arms[a]["by_level"] for a in ["bm25", *GRAPH_ARMS]}
        levels = list(lv["bm25"])
        out["a_bm25_highest_kc"] = _all([_gt(b["kc"], arms[g]["kc"]) for g in GRAPH_ARMS])
        beats_all = [
            _all([_gt(lv[g][lvl]["kc"], lv["bm25"][lvl]["kc"]) for lvl in HIGHER])
            for g in GRAPH_ARMS
        ]
        out["a_no_graph_arm_beats_bm25_at_all_levels"] = (
            None if any(x is None for x in beats_all) else not any(beats_all)
        )
        for metric in ("kc", "pc", "pc_pass"):
            out[f"a_level_leaders_{metric}"] = {
                lvl: max(
                    [a for a in lv if lv[a][lvl][metric] is not None],
                    key=lambda x, m=metric, lv_=lvl: lv[x][lv_][m],
                )
                for lvl in levels
            }
            out[f"a_graph_arms_above_bm25_{metric}"] = {
                lvl: [g for g in GRAPH_ARMS if _gt(lv[g][lvl][metric], lv["bm25"][lvl][metric])]
                for lvl in levels
            }
        out["a_bm25_highest_pc"] = _all([_gt(b["pc"], arms[g]["pc"]) for g in GRAPH_ARMS])
        out["a_bm25_highest_pc_pass"] = _all(
            [_gt(b["pc_pass"], arms[g]["pc_pass"]) for g in GRAPH_ARMS]
        )
        out["b_graph_lower_hall"] = _all([_gt(b["hall"], arms[g]["hall"]) for g in GRAPH_ARMS])
        out["c_graph_higher_oc"] = _all([_gt(arms[g]["oc"], b["oc"]) for g in GRAPH_ARMS])
        out["c_graph_higher_cond_oc"] = _all(
            [_gt(arms[g]["cond_oc"], b["cond_oc"]) for g in GRAPH_ARMS]
        )
    if emic is not None:
        out["d_emic_monotone_mean"] = emic["monotone_mean"]
        out["d_emic_monotone_ge4"] = emic["monotone_ge4"]
    if gate is not None:
        out["e_useful_yield"] = gate["useful_yield"]
        out["e_yield_in_0.20_0.30"] = 0.20 <= gate["useful_yield"] <= 0.30
    return out


# --------------------------------------------------------------------------- #
# Paired bootstrap over items x replicas (pre-registered metric 5)
# --------------------------------------------------------------------------- #

BOOT_METRICS = ["kc", "kc_joint", "pc", "pc_pass", "hall", "oc", "cond_oc"]


def paired_bootstrap(
    reps: list[dict[str, Any]],
    useful_by_rep: list[set[str]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """CI of bm25 - graph arm for every table metric, overall and per Bloom level.

    Items are resampled with replacement and, for each drawn item, one replica
    is drawn uniformly; the same replica serves every arm of that item (paired).
    ``useful_by_rep[r]`` is the pair set under replica ``r``: passing the same
    set for every replica isolates answer-judge noise, passing each replica's
    own gate-approved set adds gate noise (a drawn item only counts if replica
    ``r`` admits it). Answerable items and unanswerable probes (keyed by their
    seed pair) are resampled independently. Per-level values are computed on the
    same draws, restricted to the items of that level (probes by seed level).

    KC is a mean over the items an arm committed to (TC), so the two sides of
    ``kc`` are averaged over different item sets: only the item draw and the
    replica draw are shared. ``kc_joint`` is the strictly paired version, the
    mean of the per-item difference over the items both arms committed to under
    the same replica. Evidence support (``pc`` mean, ``pc_pass`` share >= tau) is
    defined on every classified answerable item and is paired in the strict
    sense too. Every resample starts from ``seed``.
    """
    rng = np.random.default_rng(seed)
    code = {lab: i for i, lab in enumerate(LABELS)}
    c0 = reps[0]["cep"]

    def arrays(pids: list[str], suffix: str) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for arm in ["bm25", *GRAPH_ARMS]:
            lab = np.full((len(reps), len(pids)), -1)
            kc = np.full((len(reps), len(pids)), np.nan)
            pc = np.full((len(reps), len(pids)), np.nan)
            for ri, rep in enumerate(reps):
                facts = rep["answers"]["facts"]
                for j, pid in enumerate(pids):
                    f = facts[(arm, pid + suffix)]
                    lab[ri, j] = code.get(f["label"], -1)
                    if (
                        f["label"] == "TC"
                        and f["answer_correctness"] is not None
                        and f["answer_faithfulness"] is not None
                    ):
                        kc[ri, j] = f["answer_correctness"] * f["answer_faithfulness"]
                    if f["passage_coverage"] is not None and f["label"] != "unknown":
                        pc[ri, j] = f["passage_coverage"]
            out[arm] = {"lab": lab, "kc": kc, "pc": pc}
        return out

    universe = set.union(*useful_by_rep)
    ans_ids = sorted(universe)
    non_ids = sorted(
        p.removesuffix(":nonans")
        for (a, p) in reps[0]["answers"]["facts"]
        if a == "bm25" and p.endswith(":nonans") and p.removesuffix(":nonans") in universe
    )
    A = arrays(ans_ids, "")
    N = arrays(non_ids, ":nonans")
    mask_a = np.array([[p in u for p in ans_ids] for u in useful_by_rep])
    mask_n = np.array([[p in u for p in non_ids] for u in useful_by_rep])
    lvl_a = np.array([c0[p]["bloom"] for p in ans_ids])
    lvl_n = np.array([c0[p]["bloom"] for p in non_ids])
    levels = ["all", *(lvl for lvl in BLOOM if (lvl_a == lvl).any())]

    def nanmean(x: np.ndarray) -> float:
        return float(np.nanmean(x)) if np.isfinite(x).any() else math.nan

    def metrics(
        arm: str, ia: np.ndarray, ra: np.ndarray, ineg: np.ndarray, rn: np.ndarray, lvl: str
    ) -> dict[str, float]:
        keep = mask_a[ra, ia] & ((lvl_a[ia] == lvl) if lvl != "all" else True)
        lab = A[arm]["lab"][ra, ia][keep]
        kc = A[arm]["kc"][ra, ia][keep]
        pc = A[arm]["pc"][ra, ia][keep]
        tc = (lab == code["TC"]).sum()
        fa = (lab == code["FA"]).sum()
        cond = pc >= TAU
        ctc = ((lab == code["TC"]) & cond).sum()
        cfa = ((lab == code["FA"]) & cond).sum()
        keep_n = mask_n[rn, ineg] & ((lvl_n[ineg] == lvl) if lvl != "all" else True)
        nl = N[arm]["lab"][rn, ineg][keep_n]
        fc = (nl == code["FC"]).sum()
        ta = (nl == code["TA"]).sum()
        fin = np.isfinite(pc)
        return {
            "kc": nanmean(kc),
            "pc": nanmean(pc),
            "pc_pass": float((pc[fin] >= TAU).mean()) if fin.any() else math.nan,
            "hall": fc / (fc + ta) if fc + ta else math.nan,
            "oc": fa / (fa + tc) if fa + tc else math.nan,
            "cond_oc": cfa / (cfa + ctc) if cfa + ctc else math.nan,
        }

    def joint_kc(g: str, ia: np.ndarray, ra: np.ndarray, lvl: str) -> tuple[float, int]:
        keep = mask_a[ra, ia] & ((lvl_a[ia] == lvl) if lvl != "all" else True)
        d = A["bm25"]["kc"][ra, ia][keep] - A[g]["kc"][ra, ia][keep]
        fin = np.isfinite(d)
        return (float(d[fin].mean()) if fin.any() else math.nan), int(fin.sum())

    na, nn, r = len(ans_ids), len(non_ids), len(reps)
    # Point estimate: pooled over all replicas (every item x every replica).
    ia_all = np.tile(np.arange(na), r)
    ra_all = np.repeat(np.arange(r), na)
    in_all = np.tile(np.arange(nn), r)
    rn_all = np.repeat(np.arange(r), nn)
    point = {
        lvl: {
            arm: metrics(arm, ia_all, ra_all, in_all, rn_all, lvl) for arm in ["bm25", *GRAPH_ARMS]
        }
        for lvl in levels
    }
    joint_point = {lvl: {g: joint_kc(g, ia_all, ra_all, lvl) for g in GRAPH_ARMS} for lvl in levels}
    diffs: dict[str, dict[str, dict[str, list[float]]]] = {
        lvl: {g: defaultdict(list) for g in GRAPH_ARMS} for lvl in levels
    }
    for _ in range(n_boot):
        ia = rng.integers(0, na, na)
        ra = rng.integers(0, r, na)
        ineg = rng.integers(0, nn, nn)
        rn = rng.integers(0, r, nn)
        for lvl in levels:
            mb = metrics("bm25", ia, ra, ineg, rn, lvl)
            for g in GRAPH_ARMS:
                mg = metrics(g, ia, ra, ineg, rn, lvl)
                for k in mb:
                    diffs[lvl][g][k].append(mb[k] - mg[k])
                diffs[lvl][g]["kc_joint"].append(joint_kc(g, ia, ra, lvl)[0])
    out: dict[str, Any] = {
        "n_answerable_universe": na,
        "n_nonans_universe": nn,
        "R": r,
        "seed": seed,
        "levels": levels,
        "n_answerable_by_level": {lvl: int((lvl_a == lvl).sum()) for lvl in levels[1:]},
        "n_nonans_by_level": {lvl: int((lvl_n == lvl).sum()) for lvl in levels[1:]},
        "pooled_point": point,
        "bm25_minus": {},
    }
    for lvl in levels:
        out["bm25_minus"][lvl] = {}
        for g in GRAPH_ARMS:
            out["bm25_minus"][lvl][g] = {}
            for k in BOOT_METRICS:
                v = np.array(diffs[lvl][g][k])
                v = v[np.isfinite(v)]
                if k == "kc_joint":
                    pt, n_joint = joint_point[lvl][g]
                else:
                    pt, n_joint = point[lvl]["bm25"][k] - point[lvl][g][k], None
                lo = hi = None
                if len(v) >= 0.95 * n_boot and len(v) > 0:
                    lo, hi = (float(x) for x in np.percentile(v, [2.5, 97.5]))
                out["bm25_minus"][lvl][g][k] = {
                    "point": pt,
                    "ci": [lo, hi],
                    "n_finite_boot": len(v),
                    "n_joint_tc_pooled": n_joint,
                    "excludes_zero": None if lo is None else bool(lo > 0 or hi < 0),
                    "graph_better": None
                    if lo is None
                    else bool((hi < 0) if k in ("kc", "kc_joint", "pc", "pc_pass") else (lo > 0)),
                }
    return out


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def config_section(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Configuration identity and frozen-input identity across replicas."""

    def pick(meta: dict[str, Any], stage: str) -> dict[str, Any]:
        cv = meta.get("config", {}).get("config_values", {})
        base = {
            "run_id": meta.get("run_id"),
            "status": meta.get("status"),
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at"),
            "total_items": meta.get("total_items"),
            "completed_items": meta.get("completed_items"),
            "failed_items": meta.get("failed_items"),
            "hostname": meta.get("execution", {}).get("hostname"),
            "device": meta.get("hardware", {}).get("device_type"),
            "gpu": meta.get("hardware", {}).get("gpu_name"),
            "cpu_count": meta.get("hardware", {}).get("cpu_count"),
            "python": meta.get("hardware", {}).get("python_version"),
            "input_source": meta.get("input_source"),
        }
        if stage == "judge_qa":
            base |= {
                "mode": cv.get("mode"),
                "provider": cv.get("provider"),
                "model": cv.get("model_id"),
                "language": cv.get("language"),
                "temperature": cv.get("judge", {}).get("temperature"),
                "max_tokens": cv.get("judge", {}).get("max_tokens"),
                "bloom_descriptions_sha256": cv.get("bloom_descriptions_sha256"),
            }
            for c, v in (cv.get("criteria") or {}).items():
                base[f"{c}.threshold"] = v.get("threshold")
                base[f"{c}.temperature"] = v.get("temperature")
                base[f"{c}.sha256"] = v.get("sha256")
        elif stage == "emic_judge":
            llm = cv.get("llm", {})
            base |= {
                "scope": cv.get("scope"),
                "provider": llm.get("provider"),
                "model": llm.get("model_id"),
                "temperature": llm.get("temperature"),
                "max_tokens": llm.get("max_tokens"),
                "language": llm.get("language"),
                "workers": llm.get("workers"),
                "prompt_hash": "not recorded",
            }
        else:
            base |= {
                "config.pipeline_id": cv.get("pipeline_id"),
                "provider": cv.get("judge_provider"),
                "model": cv.get("judge_model_id"),
                "language": cv.get("judge_language"),
                "temperature": cv.get("judge_temperature"),
                "prompt_hash": "not recorded",
            }
        return base

    stages = {}
    for stage in ("judge_qa", "emic_judge", "judge_answers"):
        rows = [pick(rep["meta"].get(stage, {}), stage) for rep in reps]
        keys = sorted({k for row in rows for k in row})
        stages[stage] = {
            k: {
                "values": [row.get(k) for row in rows],
                "identical": len({json.dumps(row.get(k)) for row in rows}) == 1,
            }
            for k in keys
        }
    # Frozen inputs.
    c0 = reps[0]["cep"]
    cep_ident = {
        f"r1-r{i + 1}": {
            "same_ids": set(c0) == set(rep["cep"]),
            "content_mismatches": sum(
                1
                for p in c0
                if p in rep["cep"] and rep["cep"][p]["content_sha"] != c0[p]["content_sha"]
            ),
        }
        for i, rep in enumerate(reps[1:], start=1)
    }
    f0 = reps[0]["answers"]["facts"]
    ans_ident = {
        f"r1-r{i + 1}": {
            "same_keys": set(f0) == set(rep["answers"]["facts"]),
            "answer_mismatches": sum(
                1
                for k in f0
                if k in rep["answers"]["facts"]
                and rep["answers"]["facts"][k]["answer_sha"] != f0[k]["answer_sha"]
            ),
        }
        for i, rep in enumerate(reps[1:], start=1)
    }
    coverage = []
    for i, rep in enumerate(reps):
        facts = rep["answers"]["facts"]
        prev = reps[i - 1]["answers"]["facts"] if i > 0 else None
        row = {
            "replica": i + 1,
            "records": len(facts),
            "checkpoint_completed": rep["answers"]["checkpoint_size"],
        }
        if prev is not None:
            row |= {
                "in_ckpt_identical_to_prev": sum(
                    1
                    for k, f in facts.items()
                    if f["in_checkpoint"] and f["validation_sha"] == prev[k]["validation_sha"]
                ),
                "in_ckpt_different_from_prev": sum(
                    1
                    for k, f in facts.items()
                    if f["in_checkpoint"] and f["validation_sha"] != prev[k]["validation_sha"]
                ),
                "not_in_ckpt_identical_to_prev": sum(
                    1
                    for k, f in facts.items()
                    if not f["in_checkpoint"] and f["validation_sha"] == prev[k]["validation_sha"]
                ),
                "not_in_ckpt_different_from_prev": sum(
                    1
                    for k, f in facts.items()
                    if not f["in_checkpoint"] and f["validation_sha"] != prev[k]["validation_sha"]
                ),
            }
        coverage.append(row)
    gate_errors = [
        {
            "replica": i + 1,
            "pairs_with_errored_or_missing_criterion": sorted(
                p
                for p, m in rep["cep"].items()
                if not m["scores"] or any(v is None for v in m["scores"].values())
            ),
            "approved_rule_vs_passed_flag_mismatches": sum(
                1
                for m in rep["cep"].values()
                if m["passed"] is not None and m["passed"] != m["approved"]
            ),
        }
        for i, rep in enumerate(reps)
    ]
    return {
        "stages": stages,
        "judge_qa_history": [rep["history"] for rep in reps],
        "cep_identity": cep_ident,
        "answer_identity": ans_ident,
        "answers_coverage": coverage,
        "gate_errors": gate_errors,
        "emic_missing": [sum(v is None for v in rep["emic"].values()) for rep in reps],
    }


def completeness_section(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Item counts per judge and replica against replica 1, duplicates and carried copies.

    A record whose full judge output (scores and rationales) is byte-identical to
    an earlier replica's was most likely carried over instead of re-judged; the
    counts below expose that per judge and per arm.
    """
    r1 = reps[0]
    pids1 = set(r1["cep"])
    keys1 = set(r1["answers"]["facts"])
    rows = []
    for i, rep in enumerate(reps):
        ans = rep["answers"]
        facts = ans["facts"]
        meta = rep["meta"]
        row: dict[str, Any] = {
            "replica": i + 1,
            "gate_source": rep["gate_source"],
            "gate_status": meta.get("judge_qa", {}).get("status"),
            "gate_completed_items": meta.get("judge_qa", {}).get("completed_items"),
            "gate_failed_items": meta.get("judge_qa", {}).get("failed_items"),
            "cep_pairs": len(rep["cep"]),
            "cep_pairs_with_verdict": sum(m["passed"] is not None for m in rep["cep"].values()),
            "cep_missing_vs_r1": len(pids1 - set(rep["cep"])),
            "cep_extra_vs_r1": len(set(rep["cep"]) - pids1),
            "emic_status": meta.get("emic_judge", {}).get("status"),
            "emic_files": rep["emic_counts"]["files"],
            "emic_entries": rep["emic_counts"]["entries"],
            "emic_duplicates": rep["emic_counts"]["duplicates"],
            "emic_scored": sum(v is not None for v in rep["emic"].values()),
            "emic_missing_vs_r1": len(pids1 - set(rep["emic"])),
            "answers_status": meta.get("judge_answers", {}).get("status"),
            "answers_completed_items": meta.get("judge_answers", {}).get("completed_items"),
            "answers_failed_items": meta.get("judge_answers", {}).get("failed_items"),
            "answers_records": len(facts),
            "answers_files_by_arm_kind": ans["files_by_arm_kind"],
            "answers_files_by_arm_kind_equal_r1": ans["files_by_arm_kind"]
            == r1["answers"]["files_by_arm_kind"],
            "answers_duplicate_keys": ans["duplicate_keys"],
            "answers_missing_vs_r1": len(keys1 - set(facts)),
            "answers_extra_vs_r1": len(set(facts) - keys1),
            "answers_unclassified": sum(f["label"] == "unknown" for f in facts.values()),
            "checkpoint_completed": ans["checkpoint_size"],
            "checkpoint_failed": ans["checkpoint_failed"],
            "checkpoint_total": ans["checkpoint_total"],
            "checkpoint_started_at": ans["checkpoint_started_at"],
            "answers_not_in_checkpoint": sum(f["in_checkpoint"] is False for f in facts.values()),
        }
        for j in range(i):
            prev = reps[j]
            pf = prev["answers"]["facts"]
            same = [
                k
                for k, f in facts.items()
                if k in pf and f["validation_sha"] == pf[k]["validation_sha"]
            ]
            row[f"answers_identical_to_r{j + 1}"] = len(same)
            row[f"answers_identical_to_r{j + 1}_by_arm"] = dict(
                sorted(Counter(k[0] for k in same).items())
            )
            row[f"gate_verdicts_identical_to_r{j + 1}"] = sum(
                1
                for p, m in rep["cep"].items()
                if m["validation_sha"] is not None
                and p in prev["cep"]
                and m["validation_sha"] == prev["cep"][p]["validation_sha"]
            )
            es, eprev = rep["emic_counts"]["entry_sha"], prev["emic_counts"]["entry_sha"]
            row[f"emic_entries_identical_to_r{j + 1}"] = sum(
                1 for p, h in es.items() if eprev.get(p) == h
            )
        rows.append(row)
    return {"rows": rows}


PROMPT_FILES = {
    "judge_qa": {
        c: [f"prompts/judge/criteria/{c}/config.json", f"prompts/judge/criteria/{c}/pt/prompt.md"]
        for c in GATE_CRITERIA
    },
    "emic_judge": {
        "emic_validity": [
            "prompts/judge/criteria/emic_validity/config.json",
            "prompts/judge/criteria/emic_validity/pt/prompt.md",
        ]
    },
    "judge_answers": {
        c: [f"prompts/judge/criteria/{c}/config.json", f"prompts/judge/criteria/{c}/pt/prompt.md"]
        for c in ANS_CRITERIA
    },
}


JUDGE_CODE_PATHS = {
    "judge_qa": ["src/arandu/qa/cep", "src/arandu/shared/judge"],
    "emic_judge": ["src/arandu/shared/emic", "src/arandu/shared/judge"],
    "judge_answers": ["src/arandu/shared/rag/judge_answers", "src/arandu/shared/judge"],
}


def _prompt_digest(paths: list[Path]) -> str | None:
    """Same digest the gate writes to its run_metadata (name + bytes of each file)."""
    if not all(p.is_file() for p in paths):
        return None
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.strip()


def provenance_section(reps: list[dict[str, Any]], repo: Path) -> dict[str, Any]:
    """Prompt identity and runtime versions that the exports record (or do not).

    The gate records a digest per criterion; the emic and answer judges record
    none. For every judge this recomputes the digest of the prompts at the
    analysed commit, checks it against the gate's recorded digests, and gives the
    date of the last commit touching each prompt so it can be compared with the
    start of each replica. Any metadata key naming an Ollama or image version is
    collected; an empty list means the exports do not record it.
    """

    def keys_matching(obj: Any, prefix: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}.{k}" if prefix else k
                low = k.lower()
                if ("ollama" in low and "version" in low) or low in ("image", "image_digest"):
                    found.append(name)
                found += keys_matching(v, name)
        return found

    out: dict[str, Any] = {
        "head": _git(repo, "rev-parse", "--short=8", "HEAD"),
        "stages": {},
        "runtime_version_keys": {
            f"r{i + 1}": {stage: keys_matching(m) for stage, m in rep["meta"].items()}
            for i, rep in enumerate(reps)
        },
    }
    for stage, crits in PROMPT_FILES.items():
        starts = [rep["meta"].get(stage, {}).get("started_at") for rep in reps]
        rows = {}
        for c, rel in crits.items():
            paths = [repo / r for r in rel]
            last = _git(repo, "log", "-1", "--format=%cI", "--", *rel)
            dirty = _git(repo, "status", "--porcelain", "--", *rel)
            recorded = (
                [
                    rep["meta"]["judge_qa"]["config"]["config_values"]["criteria"][c]["sha256"]
                    for rep in reps
                ]
                if stage == "judge_qa"
                else None
            )
            head = _prompt_digest(paths)
            last_dt = datetime.fromisoformat(last) if last else None
            rows[c] = {
                "head_sha256": head,
                "recorded_sha256": recorded,
                "head_equals_recorded": None
                if recorded is None
                else all(x == head for x in recorded),
                "last_commit": _git(repo, "log", "-1", "--format=%h", "--", *rel),
                "last_commit_date": last,
                "working_tree_clean": dirty == "",
                "unchanged_since_before_each_replica": [
                    None
                    if st is None or last_dt is None
                    else last_dt < datetime.fromisoformat(st.replace("Z", "+00:00"))
                    for st in starts
                ],
            }
        code = {}
        for rel in JUDGE_CODE_PATHS[stage]:
            last = _git(repo, "log", "-1", "--format=%cI", "--", rel)
            last_dt = datetime.fromisoformat(last) if last else None
            code[rel] = {
                "last_commit": _git(repo, "log", "-1", "--format=%h", "--", rel),
                "last_commit_date": last,
                "unchanged_since_before_each_replica": [
                    None
                    if st is None or last_dt is None
                    else last_dt < datetime.fromisoformat(st.replace("Z", "+00:00"))
                    for st in starts
                ],
            }
        out["stages"][stage] = {"replica_started_at": starts, "prompts": rows, "code": code}
    return out


def complete_answer_replicas(reps: list[dict[str, Any]]) -> list[int]:
    """Indices of replicas whose judge_answers round finished (all records re-judged)."""
    out = []
    for i, rep in enumerate(reps):
        meta = rep["meta"].get("judge_answers", {})
        if meta.get("status") == "completed":
            out.append(i)
    return out


def answer_value(
    rep: dict[str, Any], key: tuple[str, str], field: str, rejudged: set[int], i: int
) -> Any:
    """A judge-answers field of replica ``i``, or None when that record was not re-judged."""
    f = rep["answers"]["facts"][key]
    if i not in rejudged and not f["in_checkpoint"]:
        return None
    return f[field]


def pair_strata(c0: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Total dataset, useful candidates (every pair above Remember) and each level."""
    pids = sorted(c0)
    return {
        "total": pids,
        "candidates": [p for p in pids if c0[p]["bloom"] != "remember"],
        **{lvl: [p for p in pids if c0[p]["bloom"] == lvl] for lvl in BLOOM},
    }


def gate_section(
    reps: list[dict[str, Any]], n_boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Gate judge reliability per criterion and for the verdict, by Bloom level."""
    c0 = reps[0]["cep"]
    pids = sorted(c0)
    strata = pair_strata(c0)
    out: dict[str, Any] = {
        "criteria": {},
        "verdict": {},
        "instability": {},
        "margin": {},
        "means": {},
    }
    for crit in GATE_CRITERIA:
        out["criteria"][crit] = {}
        for name, ids in strata.items():
            units = [[to_ord(rep["cep"][p]["scores"].get(crit)) for rep in reps] for p in ids]
            if sum(1 for u in units if sum(x is not None for x in u) >= 2) < 2:
                continue
            out["criteria"][crit][name] = reliability_block(units, (0, 4), "ordinal", n_boot, rng)
        out["means"][crit] = [
            float(
                np.mean(
                    [v for v in (rep["cep"][p]["scores"].get(crit) for p in pids) if v is not None]
                )
            )
            for rep in reps
        ]
    approved = [{p for p, m in rep["cep"].items() if m["approved"]} for rep in reps]
    for name, ids in strata.items():
        units = [[int(p in a) for a in approved] for p in ids]
        out["verdict"][name] = reliability_block(units, (0, 1), "nominal", n_boot, rng)
        out["instability"][name] = instability(units)
    # Majority vote as sensitivity.
    maj = {p for p in pids if sum(p in a for a in approved) * 2 > len(reps)}
    out["majority_vote_approved"] = len(maj)
    # Margin dependence: binding score = min over evaluated criteria; mean across replicas.
    bins = defaultdict(lambda: [0, 0])
    for p in pids:
        mins = []
        for rep in reps:
            s = rep["cep"][p]["scores"]
            vals = [v if v is not None else 0.0 for v in s.values()]
            mins.append(min(vals) if vals else 0.0)
        dist = abs(float(np.mean(mins)) - TAU)
        b = "<=0.125" if dist <= 0.125 + 1e-9 else ("<=0.375" if dist <= 0.375 + 1e-9 else ">0.375")
        bins[b][0] += 1
        bins[b][1] += int(0 < sum(p in a for a in approved) < len(reps))
    out["margin"] = {
        k: {"n": v[0], "unstable": v[1], "frac_unstable": v[1] / v[0]}
        for k, v in sorted(bins.items())
    }
    # Useful-set stability (higher-order pairs).
    useful = [useful_set(rep["cep"], a) for rep, a in zip(reps, approved, strict=True)]
    u0 = useful[0]
    out["useful_sets"] = {
        "sizes": [len(u) for u in useful],
        "intersection_all": len(set.intersection(*useful)),
        "union_all": len(set.union(*useful)),
        "jaccard_vs_r1": [len(u0 & u) / len(u0 | u) for u in useful],
        "canonical_kept": [len(u0 & u) for u in useful],
        "majority_vote": len(useful_set(c0, maj)),
    }
    return out


def emic_section(
    reps: list[dict[str, Any]], n_boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Emic judge reliability (1..5) and EV >= 4 filter stability, by level."""
    c0 = reps[0]["cep"]
    pids = sorted(c0)
    strata = pair_strata(c0)
    out: dict[str, Any] = {
        "ordinal": {},
        "binary": {},
        "instability": {},
        "abs_diff": {},
        "margin": {},
    }
    for name, ids in strata.items():
        units = [[rep["emic"].get(p) for rep in reps] for p in ids]
        out["ordinal"][name] = reliability_block(units, (1, 5), "ordinal", n_boot, rng)
        bunits = [[None if x is None else int(x >= TAU_EMIC) for x in u] for u in units]
        out["binary"][name] = reliability_block(bunits, (0, 1), "nominal", n_boot, rng)
        out["instability"][name] = instability(bunits)
        diffs = Counter()
        for u in units:
            for a in range(len(u)):
                for b in range(a + 1, len(u)):
                    if u[a] is not None and u[b] is not None:
                        diffs[abs(u[a] - u[b])] += 1
        tot = sum(diffs.values())
        out["abs_diff"][name] = {
            str(k): {"n": diffs.get(k, 0), "pct": 100 * diffs.get(k, 0) / tot} for k in range(5)
        }
    bins = defaultdict(lambda: [0, 0])
    for p in pids:
        vals = [rep["emic"].get(p) for rep in reps]
        if any(v is None for v in vals):
            continue
        dist = abs(float(np.mean(vals)) - (TAU_EMIC - 0.5))
        b = "<=0.5" if dist <= 0.5 + 1e-9 else ("<=1.5" if dist <= 1.5 + 1e-9 else ">1.5")
        k = sum(v >= TAU_EMIC for v in vals)
        bins[b][0] += 1
        bins[b][1] += int(0 < k < len(vals))
    out["margin"] = {
        k: {"n": v[0], "unstable": v[1], "frac_unstable": v[1] / v[0]}
        for k, v in sorted(bins.items())
    }
    out["means"] = [
        float(np.mean([v for v in rep["emic"].values() if v is not None])) for rep in reps
    ]
    return out


def answers_section(
    reps: list[dict[str, Any]], n_boot: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Answer judge reliability per criterion, pass cut and cell, in three record strata.

    Strata over records: ``total`` (every probe: the answerable probe of each of
    the 2,670 pairs plus the 334 unanswerable probes), ``candidates`` (probes
    whose pair, or seed pair for an unanswerable probe, is above Remember,
    whatever the gate said) and ``useful`` (canonical useful set of replica 1,
    secondary). Each is pooled over the retrieval arms and split by arm; the
    total and candidates strata are also split by Bloom level. Only replicas
    whose round finished enter; a partial round adds a scope restricted to the
    records every replica re-judged.
    """
    done = set(complete_answer_replicas(reps))
    keys = sorted(reps[0]["answers"]["facts"])
    c0 = reps[0]["cep"]
    useful0 = useful_set(c0, {p for p, m in c0.items() if m["approved"]})

    def level(k: tuple[str, str]) -> str:
        return c0[k[1].removesuffix(":nonans")]["bloom"]

    preds = {
        "total": lambda k: True,
        "candidates": lambda k: level(k) != "remember",
        "useful": lambda k: k[1].removesuffix(":nonans") in useful0,
    }
    scopes: dict[str, tuple[list[int], list[tuple[str, str]]]] = {
        "complete_replicas": (sorted(done), keys)
    }
    every = list(range(len(reps)))
    if len(done) < len(reps):
        scopes["all_replicas_rejudged_records"] = (
            every,
            [
                k
                for k in keys
                if all(i in done or reps[i]["answers"]["facts"][k]["in_checkpoint"] for i in every)
            ],
        )
    out: dict[str, Any] = {"scopes": {}}
    for sname, (ridx, sk) in scopes.items():
        block: dict[str, Any] = {
            "replicas": [i + 1 for i in ridx],
            "n_records": len(sk),
            "criteria": {},
            "cell": {},
            "abstain_decision": {},
            "transitions": {},
        }
        # The null arm is constant (always abstains, coverage 0) and would inflate a
        # pooled coefficient, so the pooled strata cover the retrieval arms only.
        strata: dict[str, list[tuple[str, str]]] = {}
        for stratum, pred in preds.items():
            ks = [k for k in sk if pred(k)]
            strata[stratum] = [k for k in ks if k[0] != "null"]
            for arm in ARMS:
                strata[f"{stratum}.{arm}"] = [k for k in ks if k[0] == arm]
            if stratum != "useful":
                for lvl in BLOOM if stratum == "total" else HIGHER:
                    strata[f"{stratum}.{lvl}"] = [
                        k for k in ks if k[0] != "null" and level(k) == lvl
                    ]
        for crit in [*ANS_CRITERIA, "passage_coverage_pass"]:
            nominal = crit == "passage_coverage_pass"
            block["criteria"][crit] = {}
            for name, ks in strata.items():
                units = [
                    [
                        (lambda v: v if nominal else to_ord(v))(
                            answer_value(reps[i], k, crit, done, i)
                        )
                        for i in ridx
                    ]
                    for k in ks
                ]
                if sum(1 for u in units if sum(x is not None for x in u) >= 2) < 2:
                    continue
                block["criteria"][crit][name] = reliability_block(
                    units,
                    (0, 1) if nominal else (0, 4),
                    "nominal" if nominal else "ordinal",
                    n_boot,
                    rng,
                )
        for name, ks in strata.items():
            lab_units = []
            dec_units = []
            for k in ks:
                labs = [answer_value(reps[i], k, "label", done, i) for i in ridx]
                lab_units.append(
                    [None if lab in (None, "unknown") else LABELS.index(lab) for lab in labs]
                )
                ab = [answer_value(reps[i], k, "abstention", done, i) for i in ridx]
                dec_units.append([None if a is None else int(a >= TAU) for a in ab])
            block["cell"][name] = reliability_block(lab_units, (0, 3), "nominal", 0, rng)
            block["cell"][name].pop("ac2", None)
            block["cell"][name].pop("frac_amplitude_ge2", None)
            block["cell"][name].pop("amplitude", None)
            stable = sum(1 for u in lab_units if None not in u and len(set(u)) == 1)
            complete = sum(1 for u in lab_units if None not in u)
            block["cell"][name]["frac_cell_stable"] = stable / complete if complete else None
            block["cell"][name]["n_cell_unstable"] = complete - stable
            block["abstain_decision"][name] = reliability_block(
                dec_units, (0, 1), "nominal", n_boot, rng
            )
        # Cell transitions between every pair of replicas of this scope, per stratum.
        for stratum, pred in preds.items():
            block["transitions"][stratum] = {}
            for a, b in itertools.combinations(ridx, 2):
                trans: Counter[str] = Counter()
                for k in sk:
                    if not pred(k):
                        continue
                    x = answer_value(reps[a], k, "label", done, a)
                    y = answer_value(reps[b], k, "label", done, b)
                    if x != y:
                        trans[f"{x}->{y}"] += 1
                block["transitions"][stratum][f"r{a + 1}->r{b + 1}"] = dict(sorted(trans.items()))
        out["scopes"][sname] = block
    return out


def rejection_emic_section(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Emic score of gate-rejected pairs by the criterion that failed, per replica.

    For each replica, with its own gate verdicts and its own emic scores, a pair
    fails a criterion when that criterion's score is missing (errored) or below
    tau. ``any`` counts every rejected pair failing the criterion (a pair can
    fail several); ``only`` counts the pairs for which it is the sole failure.
    Approved pairs of the same stratum are the reference.
    """

    def dist(vals: list[int]) -> dict[str, Any]:
        if not vals:
            return {"n": 0, "mean": None, "pct_ge4": None}
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "pct_ge4": 100 * sum(v >= TAU_EMIC for v in vals) / len(vals),
        }

    out: dict[str, Any] = {}
    for i, rep in enumerate(reps):
        cep, emic = rep["cep"], rep["emic"]
        strata = pair_strata(reps[0]["cep"])
        res: dict[str, Any] = {}
        for name, ids in strata.items():
            ids = [p for p in ids if emic.get(p) is not None]
            fails = {
                p: {c for c, v in cep[p]["scores"].items() if v is None or v < TAU}
                for p in ids
                if not cep[p]["approved"]
            }
            row: dict[str, Any] = {
                "approved": dist([emic[p] for p in ids if cep[p]["approved"]]),
                "rejected": dist([emic[p] for p in fails]),
                "rejected_without_scores": sum(1 for p in fails if not cep[p]["scores"]),
                "by_criterion": {},
            }
            for c in GATE_CRITERIA:
                anyc = [emic[p] for p, f in fails.items() if c in f]
                only = [emic[p] for p, f in fails.items() if f == {c}]
                if not anyc and not only:
                    continue
                row["by_criterion"][c] = {"any": dist(anyc), "only": dist(only)}
            res[name] = row
        out[f"r{i + 1}"] = res
    return out


ARM_METRICS = ["kc", "pc", "pc_pass", "hall", "oc", "cond_oc", "f1_abs"]
LEVEL_METRICS = ["kc", "pc", "pc_pass", "hall", "oc", "cond_oc"]


def _range_row(values: dict[str, float | None]) -> dict[str, Any]:
    vals = [v for v in values.values() if v is not None]
    return {
        "values": values,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "range": (max(vals) - min(vals)) if vals else None,
    }


def arm_robustness(scen: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each retrieval-table aggregate across scenarios: values, min, max and range."""
    rows: dict[str, dict[str, Any]] = {}
    first = next(iter(scen.values()))
    for arm in ["bm25", *GRAPH_ARMS]:
        for metric in ARM_METRICS:
            rows[f"{arm}.{metric}"] = _range_row({k: v[arm][metric] for k, v in scen.items()})
        for lvl in first[arm]["by_level"]:
            for metric in LEVEL_METRICS:
                rows[f"{arm}.{metric}.{lvl}"] = _range_row(
                    {k: v[arm]["by_level"][lvl][metric] for k, v in scen.items()}
                )
    return rows


def judge_robustness(ag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Gate and emic aggregates across the replicas."""
    rows: dict[str, dict[str, Any]] = {}
    reps = {f"r{i + 1}": i for i in range(len(ag["gate"]))}

    def add(name: str, values: dict[str, float | None]) -> None:
        rows[name] = _range_row(values)

    add("gate.approved_all", {k: ag["gate"][i]["approved_all"] for k, i in reps.items()})
    add("gate.useful", {k: ag["gate"][i]["useful_all"] for k, i in reps.items()})
    add("gate.useful_yield", {k: ag["gate"][i]["useful_yield"] for k, i in reps.items()})
    for lvl in BLOOM:
        add(
            f"gate.pass.{lvl}",
            {k: ag["gate"][i]["by_level"][lvl]["rate"] for k, i in reps.items()},
        )
    add("emic.mean", {k: ag["emic"][i]["all"]["mean"] for k, i in reps.items()})
    for sc in ("5", "3", "4"):
        add(f"emic.pct{sc}", {k: ag["emic"][i]["all"]["pct"][sc] for k, i in reps.items()})
    for lvl in BLOOM:
        add(
            f"emic.mean.{lvl}", {k: ag["emic"][i]["by_level"][lvl]["mean"] for k, i in reps.items()}
        )
        add(
            f"emic.pct_ge4.{lvl}",
            {k: ag["emic"][i]["by_level"][lvl]["pct_ge4"] for k, i in reps.items()},
        )
    for verdict in ("approved", "rejected"):
        add(
            f"emic.mean.{verdict}",
            {k: ag["emic"][i]["by_verdict"][verdict]["mean"] for k, i in reps.items()},
        )
        add(
            f"emic.pct_ge4.{verdict}",
            {k: ag["emic"][i]["by_verdict"][verdict]["pct_ge4"] for k, i in reps.items()},
        )
        for lvl in BLOOM:
            add(
                f"emic.mean.{verdict}.{lvl}",
                {
                    k: ag["emic"][i]["by_level_verdict"][lvl][verdict]["mean"]
                    for k, i in reps.items()
                },
            )
    return rows


def aggregates_section(reps: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    """Published aggregates per replica, in the three strata, and the paired bootstraps.

    ``total`` (all 2,670 pairs and the 334 probes) and ``candidates`` (the 1,335
    pairs above Remember and the probes seeded by them) do not depend on the gate,
    so their only source of variation is the answer judge (scenario A). The
    ``useful`` stratum, the paper's table, has three: A (answer judge replicas on
    the canonical useful set U1), B (each replica's useful set with the canonical
    answers) and C (own useful set and own answers).
    """
    done = complete_answer_replicas(reps)
    approved = [{p for p, m in rep["cep"].items() if m["approved"]} for rep in reps]
    useful = [useful_set(rep["cep"], a) for rep, a in zip(reps, approved, strict=True)]
    c0 = reps[0]["cep"]
    strata = pair_strata(c0)
    fixed = {"total": set(strata["total"]), "candidates": set(strata["candidates"])}
    out: dict[str, Any] = {}
    out["gate"] = [gate_aggregates(rep["cep"], a) for rep, a in zip(reps, approved, strict=True)]
    maj = {p for p in c0 if sum(p in a for a in approved) * 2 > len(reps)}
    out["gate_majority_vote"] = gate_aggregates(c0, maj)
    out["emic"] = [
        emic_aggregates(rep["emic"], rep["cep"], a) for rep, a in zip(reps, approved, strict=True)
    ]
    out["emic_with_canonical_gate"] = [
        emic_aggregates(rep["emic"], c0, approved[0]) for rep in reps
    ]
    med = {}
    for p in c0:
        vals = [rep["emic"].get(p) for rep in reps]
        vals = [v for v in vals if v is not None]
        med[p] = int(np.median(vals)) if len(vals) % 2 == 1 else None
    out["emic_median"] = emic_aggregates(med, c0, approved[0])
    # Gate-free strata: answer judge noise only.
    out["fixed_strata"] = {
        name: {f"A r{i + 1}": arm_aggregates(reps[i]["answers"]["records"], c0, ids) for i in done}
        for name, ids in fixed.items()
    }
    # A: answer judge noise, canonical useful set.
    out["A_answers_fixed_useful"] = {
        f"r{i + 1}": arm_aggregates(reps[i]["answers"]["records"], c0, useful[0]) for i in done
    }
    # A (subset): records re-judged by every replica (only when a round is partial).
    partial = [i for i in range(len(reps)) if i not in done]
    if partial:
        sub = {
            k
            for k in reps[0]["answers"]["facts"]
            if all(reps[i]["answers"]["facts"][k]["in_checkpoint"] for i in partial)
        }
        out["A_subset_rejudged_by_all"] = {
            "n_records": len(sub),
            **{
                f"r{i + 1}": arm_aggregates(
                    reps[i]["answers"]["records"], c0, useful[0], restrict=sub
                )
                for i in range(len(reps))
            },
        }
    # B: gate noise, canonical answers.
    out["B_gate_useful_canonical_answers"] = {
        f"U{i + 1}": arm_aggregates(reps[0]["answers"]["records"], c0, useful[i])
        for i in range(len(reps))
    }
    out["B_majority_useful_canonical_answers"] = arm_aggregates(
        reps[0]["answers"]["records"], c0, useful_set(c0, maj)
    )
    # C: both (own useful set, own answers), complete replicas only.
    out["C_own_useful_own_answers"] = {
        f"r{i + 1}": arm_aggregates(reps[i]["answers"]["records"], c0, useful[i]) for i in done
    }
    useful_scen: dict[str, dict[str, Any]] = {}
    for i in done:
        useful_scen[f"A r{i + 1}"] = out["A_answers_fixed_useful"][f"r{i + 1}"]
    for key, arms in out["B_gate_useful_canonical_answers"].items():
        if key != "U1":
            useful_scen[f"B {key}"] = arms
    for i in done:
        if i != 0:
            useful_scen[f"C r{i + 1}"] = out["C_own_useful_own_answers"][f"r{i + 1}"]
    scen_by_stratum = {**out["fixed_strata"], "useful": useful_scen}
    out["robustness"] = {name: arm_robustness(sc) for name, sc in scen_by_stratum.items()}
    out["robustness"]["judges"] = judge_robustness(out)
    concl: dict[str, Any] = {}
    for name, sc in scen_by_stratum.items():
        for k, arms in sc.items():
            concl[f"{name} | {k}"] = conclusions(arms, None, None)
    for i in range(len(reps)):
        concl[f"gate/emic r{i + 1}"] = conclusions(None, out["emic"][i], out["gate"][i])
    if partial:
        for i in range(len(reps)):
            concl[f"A-subset r{i + 1}"] = conclusions(
                out["A_subset_rejudged_by_all"][f"r{i + 1}"], None, None
            )
    out["conclusions"] = concl
    done_reps = [reps[i] for i in done]
    out["paired_bootstrap"] = {
        **{
            f"{name}_answers_noise": paired_bootstrap(done_reps, [ids] * len(done), n_boot, seed)
            for name, ids in fixed.items()
        },
        "useful_answers_noise_fixed_U1": paired_bootstrap(
            done_reps, [useful[0]] * len(done), n_boot, seed
        ),
        "useful_gate_and_answers_noise_own_U": paired_bootstrap(
            done_reps, [useful[i] for i in done], n_boot, seed
        ),
    }
    return out


def sanity_section(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the published canonical numbers on replica 1."""
    r1 = reps[0]
    approved = {p for p, m in r1["cep"].items() if m["approved"]}
    useful = useful_set(r1["cep"], approved)
    return {
        "gate": gate_aggregates(r1["cep"], approved),
        "arms": arm_aggregates(r1["answers"]["records"], r1["cep"], useful),
        "emic": emic_aggregates(r1["emic"], r1["cep"], approved),
    }


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "sim" if x else "não"
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_f(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _rel_rows(blocks: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for name, b in blocks.items():
        pw = b["pairwise"]
        rows.append(
            [
                name,
                b["n_items"],
                b["alpha"],
                f"[{_f(b['ci'][0])}, {_f(b['ci'][1])}]",
                b.get("alpha_pkg"),
                b.get("ac2"),
                " / ".join(_f(v["exact_agreement"]) for v in pw.values()),
                " / ".join(_f(v["kappa"]) for v in pw.values()),
                " / ".join(_f(v["mean_delta"]) for v in pw.values()),
                b.get("frac_amplitude_ge2"),
            ]
        )
    return rows


REL_HEADER = [
    "estrato",
    "n",
    "alpha",
    "IC95",
    "alpha (pkg)",
    "AC2",
    "acordo exato (pares)",
    "kappa (pares)",
    "Delta medio (pares)",
    "frac amplitude>=2",
]


def _arm_table(arms: dict[str, Any]) -> str:
    """Retrieval table in the paper's layout plus the n behind each cell."""
    header = [
        "braço",
        "KC",
        "Und",
        "Ana",
        "Eva",
        "Eva n KC",
        "Eva TC",
        "Eva pc",
        "Hall",
        "Hall IC",
        "OC",
        "OC cond (n)",
        "F1abs",
        "ES média",
        "ES pass",
        "Eva ES pass",
        "n resp",
        "n sem resp",
    ]
    rows = []
    for a, m in arms.items():
        lv = m["by_level"]
        rows.append(
            [
                a,
                m["kc"],
                lv["understand"]["kc"],
                lv["analyze"]["kc"],
                lv["evaluate"]["kc"],
                lv["evaluate"]["kc_n"],
                lv["evaluate"]["tc"],
                lv["evaluate"]["pc"],
                m["hall"],
                f"[{_f(m['hall_ci'][0])}, {_f(m['hall_ci'][1])}]",
                m["oc"],
                f"{_f(m['cond_oc'])} ({m['cond_oc_n']})",
                m["f1_abs"],
                m["pc"],
                m["pc_pass"],
                lv["evaluate"]["pc_pass"],
                m["n_answerable"],
                m["n_nonans"],
            ]
        )
    return _table(header, rows)


def _level_table(arms: dict[str, Any]) -> str:
    """Per-level block of the retrieval table: every metric, one row per arm x level."""
    rows = []
    for a, m in arms.items():
        for lvl, d in m["by_level"].items():
            rows.append(
                [
                    a,
                    lvl,
                    d["n_answerable"],
                    d["n_nonans"],
                    d["kc"],
                    d["tc"],
                    d["pc"],
                    d["pc_pass"],
                    d["hall"],
                    d["oc"],
                    f"{_f(d['cond_oc'])} ({d['cond_oc_n']})",
                ]
            )
    return _table(
        [
            "braço",
            "nível",
            "n resp",
            "n sem resp",
            "KC",
            "TC",
            "ES média",
            "ES pass",
            "Hall",
            "OC",
            "OC cond (n)",
        ],
        rows,
    )


def render_md(res: dict[str, Any]) -> str:
    """Render every table used by the report."""
    md = [
        "# Teste-reteste dos juízes: saída do script",
        "",
        f"Ambiente: `{json.dumps(res['environment'])}`",
        "",
    ]
    md.append("## 1. Configuração")
    for stage, fields in res["config"]["stages"].items():
        md.append(f"\n### {stage}\n")
        header = ["campo", *[f"r{i + 1}" for i in range(len(res["replicas"]))], "idêntico"]
        md.append(_table(header, [[k, *v["values"], v["identical"]] for k, v in fields.items()]))
    md.append(
        "\n```json\n"
        + json.dumps(
            {
                k: res["config"][k]
                for k in (
                    "judge_qa_history",
                    "cep_identity",
                    "answer_identity",
                    "answers_coverage",
                    "emic_missing",
                )
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n```"
    )
    md.append("\n```json\n" + json.dumps(res["config"]["gate_errors"], indent=1) + "\n```")
    md.append("\n## 1b. Completude por juiz e réplica\n")
    rows = res["completeness"]["rows"]
    ckeys = list(dict.fromkeys(k for r in rows for k in r))
    md.append(
        _table(
            ["campo", *[f"r{r['replica']}" for r in rows]],
            [
                [
                    k,
                    *(
                        json.dumps(r.get(k), ensure_ascii=False)
                        if isinstance(r.get(k), dict)
                        else r.get(k)
                        for r in rows
                    ),
                ]
                for k in ckeys
            ],
        )
    )
    md.append("\n## 1c. Proveniência dos prompts e versões\n")
    md.append("```json\n" + json.dumps(res["provenance"], indent=1, ensure_ascii=False) + "\n```")
    s = res["sanity"]
    md.append("\n## 2. Sanidade (réplica 1)\n")
    md.append("```json\n" + json.dumps(s["gate"], indent=1) + "\n```")
    md.append(_arm_table(s["arms"]))
    e = s["emic"]
    md.append(
        "\n"
        + _table(
            ["nível", "n", "média", ">=4", "<=2", "3", "4", "5"],
            [
                [
                    k,
                    d["n"],
                    d["mean"],
                    d["pct_ge4"],
                    d["pct_le2"],
                    d["pct"]["3"],
                    d["pct"]["4"],
                    d["pct"]["5"],
                ]
                for k, d in [("all", e["all"]), *e["by_level"].items()]
            ],
        )
    )
    md.append(
        "\nEV aprovados vs rejeitados: "
        f"{_f(e['by_verdict']['approved']['mean'])} vs {_f(e['by_verdict']['rejected']['mean'])}; "
        f">=4: {_f(e['by_verdict']['approved']['pct_ge4'])} vs "
        f"{_f(e['by_verdict']['rejected']['pct_ge4'])}"
    )
    g = res["gate"]
    md.append("\n## 3. Portão\n")
    for crit, blocks in g["criteria"].items():
        md.append(
            f"\n### {crit} (médias por réplica: {', '.join(_f(x) for x in g['means'][crit])})\n"
        )
        md.append(_table(REL_HEADER, _rel_rows(blocks)))
    md.append("\n### veredito binário\n")
    md.append(_table(REL_HEADER, _rel_rows(g["verdict"])))
    md.append("\n### instabilidade do veredito\n")
    md.append(
        _table(
            ["estrato", "n", "k=0", "k=1", "k=2", "k=3", "instáveis", "frac instável", "D médio"],
            [
                [n, b["n"], *b["k_counts"].values(), b["unstable"], b["frac_unstable"], b["mean_D"]]
                for n, b in g["instability"].items()
            ],
        )
    )
    md.append(
        "\nMargem: "
        + json.dumps(g["margin"])
        + f"\n\nAprovados por voto majoritário: {g['majority_vote_approved']}\n\nConjuntos úteis: "
        + json.dumps(g["useful_sets"])
    )
    em = res["emic"]
    md.append(f"\n## 4. Êmico (médias por réplica: {', '.join(_f(x) for x in em['means'])})\n")
    md.append(_table(REL_HEADER, _rel_rows(em["ordinal"])))
    md.append("\n### filtro EV>=4\n")
    md.append(_table(REL_HEADER, _rel_rows(em["binary"])))
    md.append(
        _table(
            ["estrato", "n", "k=0", "k=1", "k=2", "k=3", "instáveis", "frac instável", "D médio"],
            [
                [n, b["n"], *b["k_counts"].values(), b["unstable"], b["frac_unstable"], b["mean_D"]]
                for n, b in em["instability"].items()
            ],
        )
    )
    md.append(
        "\n|diferença| (pares de réplicas): "
        + json.dumps(em["abs_diff"])
        + "\n\nMargem: "
        + json.dumps(em["margin"])
    )
    md.append("\n## 5. Juiz de respostas\n")
    for sname, blk in res["answers"]["scopes"].items():
        md.append(
            f"\n### escopo {sname} (réplicas {blk['replicas']}, {blk['n_records']} registros)\n"
        )
        for crit, blocks in blk["criteria"].items():
            md.append(f"\n#### {crit}\n")
            md.append(_table(REL_HEADER, _rel_rows(blocks)))
        md.append("\n#### decisão de abstenção (score >= tau)\n")
        md.append(_table(REL_HEADER, _rel_rows(blk["abstain_decision"])))
        md.append("\n#### célula TC/FC/FA/TA (alpha nominal)\n")
        md.append(
            _table(
                ["estrato", "n", "alpha", "acordo exato (pares)", "frac célula estável"],
                [
                    [
                        n,
                        b["n_items"],
                        b["alpha"],
                        " / ".join(_f(v["exact_agreement"]) for v in b["pairwise"].values()),
                        b["frac_cell_stable"],
                    ]
                    for n, b in blk["cell"].items()
                ],
            )
        )
        md.append("\nTransições: " + json.dumps(blk["transitions"]))
    md.append("\n## 5b. EV dos rejeitados pelo critério que reprovou (L4)\n")
    rej_rows = []
    for rk, res_r in res["rejection_emic"].items():
        for stratum, row in res_r.items():
            base = [rk, stratum]
            rej_rows.append(
                [
                    *base,
                    "aprovados",
                    row["approved"]["n"],
                    row["approved"]["mean"],
                    row["approved"]["pct_ge4"],
                ]
            )
            rej_rows.append(
                [
                    *base,
                    "rejeitados",
                    row["rejected"]["n"],
                    row["rejected"]["mean"],
                    row["rejected"]["pct_ge4"],
                ]
            )
            for c, d in row["by_criterion"].items():
                for kind in ("any", "only"):
                    rej_rows.append(
                        [*base, f"{c} ({kind})", d[kind]["n"], d[kind]["mean"], d[kind]["pct_ge4"]]
                    )
    md.append(_table(["réplica", "estrato", "grupo", "n", "EV média", "% EV>=4"], rej_rows))
    ag = res["aggregates"]
    md.append("\n## 6. Agregados publicados\n")
    md.append(
        _table(
            [
                "réplica",
                "aprov/2670",
                "aprov base",
                "Rem",
                "Und",
                "Ana",
                "Eva",
                "úteis",
                "úteis base",
                "rendimento",
            ],
            [
                [
                    f"r{i + 1}",
                    x["approved_all"],
                    x["approved_base"],
                    *(x["by_level"][lvl]["rate"] for lvl in BLOOM),
                    x["useful_all"],
                    x["useful_base"],
                    x["useful_yield"],
                ]
                for i, x in enumerate(ag["gate"])
            ]
            + [
                [
                    "maioria",
                    ag["gate_majority_vote"]["approved_all"],
                    ag["gate_majority_vote"]["approved_base"],
                    *(ag["gate_majority_vote"]["by_level"][lvl]["rate"] for lvl in BLOOM),
                    ag["gate_majority_vote"]["useful_all"],
                    ag["gate_majority_vote"]["useful_base"],
                    ag["gate_majority_vote"]["useful_yield"],
                ]
            ],
        )
    )
    for label, lst in (
        ("emic (própria réplica)", ag["emic"]),
        ("emic mediana", [ag["emic_median"]]),
    ):
        md.append(f"\n### {label}\n")
        md.append(
            _table(
                [
                    "réplica",
                    "média",
                    "%5",
                    "%3",
                    "%4",
                    "Rem média",
                    "Und",
                    "Ana",
                    "Eva",
                    "Rem >=4",
                    "Und",
                    "Ana",
                    "Eva",
                    "monótona média",
                    "monótona >=4",
                    "aprov média",
                    "rej média",
                ],
                [
                    [
                        i + 1,
                        x["all"]["mean"],
                        x["all"]["pct"]["5"],
                        x["all"]["pct"]["3"],
                        x["all"]["pct"]["4"],
                        *(x["by_level"][lvl]["mean"] for lvl in BLOOM),
                        *(x["by_level"][lvl]["pct_ge4"] for lvl in BLOOM),
                        x["monotone_mean"],
                        x["monotone_ge4"],
                        x["by_verdict"]["approved"]["mean"],
                        x["by_verdict"]["rejected"]["mean"],
                    ]
                    for i, x in enumerate(lst)
                ],
            )
        )
    md.append("\n### EV por nível e veredito do portão (própria réplica)\n")
    md.append(
        _table(
            ["réplica", "nível", "aprov n", "aprov média", "rej n", "rej média"],
            [
                [
                    i + 1,
                    lvl,
                    x["by_level_verdict"][lvl]["approved"]["n"],
                    x["by_level_verdict"][lvl]["approved"]["mean"],
                    x["by_level_verdict"][lvl]["rejected"]["n"],
                    x["by_level_verdict"][lvl]["rejected"]["mean"],
                ]
                for i, x in enumerate(ag["emic"])
                for lvl in BLOOM
            ],
        )
    )
    for name, scen in ag["fixed_strata"].items():
        md.append(f"\n### estrato {name} (sem portão; só ruído do juiz de respostas)\n")
        for k, arms in scen.items():
            md.append(f"\n{k}\n")
            md.append(_arm_table(arms))
            md.append("\n" + _level_table(arms))
    for key in (
        "A_answers_fixed_useful",
        "A_subset_rejudged_by_all",
        "B_gate_useful_canonical_answers",
        "C_own_useful_own_answers",
    ):
        if key not in ag:
            continue
        md.append(f"\n### {key}\n")
        for scen, arms in ag[key].items():
            if scen == "n_records":
                md.append(f"registros: {arms}\n")
                continue
            md.append(f"\n{scen}\n")
            md.append(_arm_table(arms))
            md.append("\n" + _level_table(arms))
    md.append("\n### B maioria\n")
    md.append(_arm_table(ag["B_majority_useful_canonical_answers"]))
    for name, rob in ag["robustness"].items():
        md.append(f"\n### robustez, {name} (valores por cenário, mínimo, máximo, amplitude)\n")
        md.append(
            _table(
                ["agregado", "valores", "mín", "máx", "amplitude"],
                [
                    [
                        k,
                        "; ".join(f"{s}={_f(x)}" for s, x in v["values"].items()),
                        v["min"],
                        v["max"],
                        v["range"],
                    ]
                    for k, v in rob.items()
                ],
            )
        )
    md.append("\n### conclusões\n")
    keys = sorted({k for v in ag["conclusions"].values() for k in v})
    md.append(
        _table(
            ["cenário", *keys],
            [
                [
                    s,
                    *(
                        json.dumps(v.get(k)) if isinstance(v.get(k), dict) else v.get(k)
                        for k in keys
                    ),
                ]
                for s, v in ag["conclusions"].items()
            ],
        )
    )
    for variant, pb in ag["paired_bootstrap"].items():
        md.append(
            f"\n### bootstrap pareado itens x réplicas: {variant} (R={pb['R']}, "
            f"universo {pb['n_answerable_universe']} pares, {pb['n_nonans_universe']} sondas, "
            f"por nível {json.dumps(pb['n_answerable_by_level'])} / "
            f"{json.dumps(pb['n_nonans_by_level'])}, B={res['n_boot']}, seed={pb['seed']})\n"
        )
        md.append(
            _table(
                [
                    "nível",
                    "contraste",
                    "métrica",
                    "bm25",
                    "braço",
                    "diferença",
                    "IC95",
                    "exclui zero",
                    "grafo melhor",
                    "n TC conjunto",
                ],
                [
                    [
                        lvl,
                        f"bm25 - {g}",
                        k,
                        pb["pooled_point"][lvl]["bm25"].get(k),
                        pb["pooled_point"][lvl][g].get(k),
                        v["point"],
                        f"[{_f(v['ci'][0])}, {_f(v['ci'][1])}]",
                        v["excludes_zero"],
                        v["graph_better"],
                        v["n_joint_tc_pooled"],
                    ]
                    for lvl, by_g in pb["bm25_minus"].items()
                    for g, d in by_g.items()
                    for k, v in d.items()
                ],
            )
        )
    return "\n".join(md) + "\n"


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Intra-judge test-retest analysis.")
    parser.add_argument(
        "--replica",
        action="append",
        required=True,
        type=Path,
        help="Replica dir (repeat; first = canonical).",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Full results JSON.")
    parser.add_argument("--out-md", type=Path, required=True, help="Markdown tables.")
    parser.add_argument(
        "--n-boot", type=int, default=1000, help="Bootstrap resamples (alpha CIs, paired diffs)."
    )
    parser.add_argument("--seed", type=int, default=20261005, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Optional pickle cache of loaded replicas."
    )
    parser.add_argument(
        "--gate-from",
        action="append",
        default=[],
        metavar="N=DIR",
        help="Read replica N's gate verdicts (cep/outputs, judge_qa) from DIR (repeatable).",
    )
    parser.add_argument(
        "--repo", type=Path, default=Path("."), help="arandu checkout whose prompts are digested."
    )
    args = parser.parse_args()
    if len(args.replica) < 2:
        parser.error("need at least two replicas")
    gate_from: dict[int, Path] = {}
    for item in args.gate_from:
        n, _, d = item.partition("=")
        gate_from[int(n) - 1] = Path(d)
    reps = [load_replica(p, args.cache_dir, gate_from.get(i)) for i, p in enumerate(args.replica)]
    rng = np.random.default_rng(args.seed)
    res: dict[str, Any] = {
        "replicas": [str(p) for p in args.replica],
        "gate_from": {f"r{i + 1}": str(d) for i, d in gate_from.items()},
        "n_boot": args.n_boot,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "krippendorff": KRIPP_VERSION,
            "alpha_impl": "arandu.shared.agreement.coefficients.krippendorff_alpha",
        },
    }
    res["config"] = config_section(reps)
    res["completeness"] = completeness_section(reps)
    res["provenance"] = provenance_section(reps, args.repo)
    res["sanity"] = sanity_section(reps)
    res["gate"] = gate_section(reps, args.n_boot, rng)
    res["emic"] = emic_section(reps, args.n_boot, rng)
    res["answers"] = answers_section(reps, args.n_boot, rng)
    res["rejection_emic"] = rejection_emic_section(reps)
    res["aggregates"] = aggregates_section(reps, args.n_boot, args.seed)
    for out_path in (args.out_json, args.out_md):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    args.out_md.write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")


try:
    import krippendorff as KRIPP

    KRIPP_VERSION: str | None = importlib.metadata.version("krippendorff")
except ImportError:
    KRIPP = None
    KRIPP_VERSION = None


if __name__ == "__main__":
    main()
