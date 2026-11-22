"""Bloom-stratified CEP pair yield, rejection decomposition, and concentration (read-only).

Replaces the exploratory notebook previously used to inspect the CEP QA judge
funnel. Reads ``results/<run-id>/cep/outputs/*_cep_qa.json``, drops residual
chunks below ``--min-chunk-chars`` (default 100), and computes:

1. Marginal approval rates per criterion x Bloom level (with 95% Wilson CIs)
   and joint approval rates per Bloom level.
2. Unique blockers per criterion, drop-one criterion sensitivity scenarios, and
   an additive per-pair rejection decomposition.
3. Useful-pair yield (gate-approved pairs above ``remember``) across chunk
   length bands, short vs. long chunk cutoff, and document length quintiles.
4. Ladder budget accounting, chunk utilization vs. binomial independence,
   Gini inequality from raw text to useful pairs, and top-document
   concentration shares.
5. Heuristic metadata-echo rate among generated, approved, and useful pairs.

Optionally writes JSON/Markdown summaries and publication figures (PDF + 300 dpi
PNG) to ``--figures-dir``.

Run from the repo root:

    uv run python -m analysis.analyze_bloom_yield --id thesis-run-02
    uv run python -m analysis.analyze_bloom_yield --id thesis-run-02 \\
        --out-json results/thesis-run-02/analysis/bloom_yield.json \\
        --out-md results/thesis-run-02/analysis/bloom_yield.md \\
        --figures-dir results/thesis-run-02/analysis/figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from arandu.shared.rag.analysis.wilson import wilson_ci

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MIN_CHUNK_CHARS = 100
CHUNK_WINDOW_CHARS = 4000
HEAVY_FIELDS = ("context", "generation_prompt", "generation_thinking")

CRITERIA = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
BLOOM_ORDER = ["remember", "understand", "analyze", "evaluate"]
TACIT_LEVELS = [level for level in BLOOM_ORDER if level != "remember"]

CRITERION_LABELS = {
    "faithfulness": "Faithfulness",
    "bloom_calibration": "Bloom calibration",
    "informativeness": "Informativeness",
    "self_containedness": "Self-containment",
}
BLOOM_LABELS = {
    "remember": "Remember",
    "understand": "Understand",
    "analyze": "Analyze",
    "evaluate": "Evaluate",
}

SEGMENT_ORDER = ["approved", *CRITERIA, "multi", "unjudged"]
SEGMENT_LABELS = {
    "approved": "Approved",
    "faithfulness": f"{CRITERION_LABELS['faithfulness']} only",
    "bloom_calibration": f"{CRITERION_LABELS['bloom_calibration']} only",
    "informativeness": f"{CRITERION_LABELS['informativeness']} only",
    "self_containedness": f"{CRITERION_LABELS['self_containedness']} only",
    "multi": "2+ criteria",
    "unjudged": "Unjudged (judge error)",
}

CHUNK_BINS = [0, 500, 1000, 2000, 3000, 3500, 3900, 4001]
CHUNK_LABELS = ["<0.5k", "0.5-1k", "1-2k", "2-3k", "3-3.5k", "3.5-3.9k", "3.9-4k"]

DOC_BINS = [0, 1, 2, 3, 4, 5, 6, 11, float("inf")]
DOC_LABELS = ["0", "1", "2", "3", "4", "5", "6-10", "11+"]

# Lexical patterns identifying Portuguese/English pairs that refer to interview
# metadata rather than the speaker's narrative.
META_IN_ANSWER = re.compile(r"metadad|metadata", re.IGNORECASE)
META_IN_QUESTION = re.compile(
    r"\b(?:(?:[dn]?a|nesta) (?:entrevista|conversa)|o evento ocorreu|entrevistad[ao] em"
    r"|in this interview|recorded on)\b",
    re.IGNORECASE,
)

# Visual palette shared across publication figures.
BLOOM_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRITERION_MARKERS = {
    "faithfulness": "o",
    "bloom_calibration": "s",
    "informativeness": "^",
    "self_containedness": "D",
}
SEGMENT_COLORS = {
    "approved": BLOOM_RAMP[2],
    "faithfulness": "#c08457",
    "bloom_calibration": "#c9a227",
    "informativeness": "#8a9a5b",
    "self_containedness": "#a2708f",
    "multi": "#6b6a63",
    "unjudged": GRID,
}


def _criterion_columns(pair: dict[str, Any]) -> dict[str, Any]:
    """Flatten CEP validator criteria into ``score_<name>`` and ``thr_<name>`` columns."""
    validation = pair.get("validation") or {}
    stages = validation.get("stage_results") or {}
    scores = (stages.get("cep_validation") or {}).get("criterion_scores") or {}

    columns: dict[str, Any] = {
        "passed": validation.get("passed"),
        "rejected_at": validation.get("rejected_at"),
    }
    for name, criterion in scores.items():
        columns[f"score_{name}"] = criterion.get("score")
        columns[f"thr_{name}"] = criterion.get("threshold")
    return columns


def _flatten_pair(doc: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """Build one flat DataFrame row from a CEP document and one QA pair."""
    meta = doc.get("source_metadata") or {}
    source_id = doc.get("source_file_id") or doc.get("source_gdrive_id")
    row: dict[str, Any] = {
        "source_id": source_id,
        "source_filename": doc.get("source_filename"),
        "location": meta.get("location"),
        "recording_date": meta.get("recording_date"),
        "model_id": doc.get("model_id"),
        "validator_model_id": doc.get("validator_model_id"),
        "language": doc.get("language"),
        "generation_timestamp": doc.get("generation_timestamp"),
    }
    row.update({k: v for k, v in pair.items() if k not in HEAVY_FIELDS and k != "validation"})
    row.update({f"{field}_chars": len(pair.get(field) or "") for field in HEAVY_FIELDS})
    row.update(_criterion_columns(pair))
    return row


def load_cep_pairs(cep_dir: Path) -> pd.DataFrame:
    """Load all ``*_cep_qa.json`` files from ``cep_dir`` into one row per QA pair.

    Args:
        cep_dir: Directory ``results/<run-id>/cep/outputs/``.

    Returns:
        DataFrame with flattened source, generation, and validation columns.

    Raises:
        FileNotFoundError: If no ``*_cep_qa.json`` files exist in ``cep_dir``.
    """
    files = sorted(cep_dir.glob("*_cep_qa.json"))
    if not files:
        raise FileNotFoundError(f"No *_cep_qa.json files found in {cep_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_flatten_pair(doc, pair) for pair in doc.get("qa_pairs", []))

    df = pd.DataFrame(rows)
    for col in ("bloom_level", "question_type", "location", "rejected_at"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    for criterion in CRITERIA:
        for prefix in ("score", "thr"):
            col = f"{prefix}_{criterion}"
            if col not in df.columns:
                df[col] = float("nan")
    return df


def drop_tiny_chunks(
    pairs: pd.DataFrame, min_chars: int = MIN_CHUNK_CHARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude pairs originating from residual chunks shorter than ``min_chars``.

    Filters at chunk granularity so every retained chunk keeps its full Bloom
    ladder intact.

    Args:
        pairs: QA pairs DataFrame from :func:`load_cep_pairs`.
        min_chars: Minimum chunk context length in characters.

    Returns:
        Tuple ``(kept_pairs, dropped_chunks_summary)``.
    """
    tiny = pairs["context_chars"] < min_chars
    dropped = (
        pairs[tiny]
        .groupby("chunk_id", observed=True)
        .agg(
            source_id=("source_id", "first"),
            context_chars=("context_chars", "max"),
            pairs=("context_chars", "size"),
        )
        .sort_values("context_chars")
    )
    return pairs[~tiny].reset_index(drop=True), dropped


def criterion_long(pairs: pd.DataFrame) -> pd.DataFrame:
    """Melt pairs into one row per evaluated ``(pair, criterion)``.

    Unevaluated criteria (e.g., ``informativeness`` and ``self_containedness``
    at the ``remember`` level) are omitted rather than treated as zeros.

    Args:
        pairs: QA pairs DataFrame.

    Returns:
        Long-format DataFrame with ``bloom_level``, ``criterion``, ``score``,
        ``threshold``, and boolean ``approved``.
    """
    frames = []
    for criterion in CRITERIA:
        score_col, thr_col = f"score_{criterion}", f"thr_{criterion}"
        sub = pairs.dropna(subset=[score_col, thr_col])
        frames.append(
            pd.DataFrame(
                {
                    "bloom_level": sub["bloom_level"].astype(str),
                    "criterion": criterion,
                    "score": sub[score_col].to_numpy(),
                    "threshold": sub[thr_col].to_numpy(),
                    "approved": (sub[score_col] >= sub[thr_col]).to_numpy(),
                }
            )
        )
    long_df = pd.concat(frames, ignore_index=True)
    long_df["bloom_level"] = pd.Categorical(long_df["bloom_level"], BLOOM_ORDER, ordered=True)
    long_df["criterion"] = pd.Categorical(long_df["criterion"], CRITERIA, ordered=True)
    return long_df


def approval_rates(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute marginal approval rates with 95% Wilson CIs per criterion and Bloom level.

    Args:
        long_df: Long-format DataFrame from :func:`criterion_long`.

    Returns:
        DataFrame indexed by ``(criterion, bloom_level)`` with ``evaluated``,
        ``approved``, ``rate``, ``ci_lo``, and ``ci_hi``.
    """
    grid = pd.MultiIndex.from_product([CRITERIA, BLOOM_ORDER], names=["criterion", "bloom_level"])
    rates = (
        long_df.groupby(["criterion", "bloom_level"], observed=True)["approved"]
        .agg(evaluated="size", approved="sum")
        .reindex(grid)
    )
    rates["rate"] = rates["approved"] / rates["evaluated"]
    rates[["ci_lo", "ci_hi"]] = [
        wilson_ci(int(k), int(n)) if pd.notna(n) and int(n) > 0 else (math.nan, math.nan)
        for k, n in zip(rates["approved"], rates["evaluated"], strict=True)
    ]
    return rates


def approved_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build a pair x criterion approval matrix (1.0 pass, 0.0 fail, NaN unevaluated).

    Args:
        pairs: QA pairs DataFrame.

    Returns:
        Float DataFrame indexed like ``pairs`` with one column per criterion.
    """
    cols: dict[str, pd.Series] = {}
    for criterion in CRITERIA:
        score = pairs[f"score_{criterion}"]
        thr = pairs[f"thr_{criterion}"]
        missing = score.isna() | thr.isna()
        cols[criterion] = (score >= thr).astype(float).mask(missing)
    return pd.DataFrame(cols, index=pairs.index)


def with_usefulness(pairs: pd.DataFrame) -> pd.DataFrame:
    """Annotate pairs with conjunctive approval and tacit-usefulness flags.

    Distinguishes:
    - ``approved``: evaluated on at least one criterion with zero failures.
    - ``candidate``: Bloom level above ``remember`` (``understand``, ``analyze``,
      ``evaluate``).
    - ``useful``: ``approved & candidate``.

    Args:
        pairs: QA pairs DataFrame.

    Returns:
        Copy of ``pairs`` with ``n_evaluated``, ``n_failed``, ``approved``,
        ``candidate``, and ``useful`` columns.
    """
    matrix = approved_matrix(pairs)
    out = pairs.copy()
    out["n_evaluated"] = matrix.notna().sum(axis=1)
    out["n_failed"] = matrix.eq(0.0).sum(axis=1)
    out["approved"] = (out["n_failed"] == 0) & (out["n_evaluated"] > 0)
    out["candidate"] = out["bloom_level"].isin(TACIT_LEVELS)
    out["useful"] = out["approved"] & out["candidate"]
    return out


def joint_rates(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute conjunctive approval rates per Bloom level with 95% Wilson CIs.

    Args:
        pairs: DataFrame annotated by :func:`with_usefulness`.

    Returns:
        DataFrame indexed by ``bloom_level`` with ``pairs``, ``approved``,
        ``rate``, ``ci_lo``, and ``ci_hi``.
    """
    rates = (
        pairs.groupby("bloom_level", observed=True)["approved"]
        .agg(pairs="size", approved="sum")
        .reindex(BLOOM_ORDER)
    )
    rates["rate"] = rates["approved"] / rates["pairs"]
    rates[["ci_lo", "ci_hi"]] = [
        wilson_ci(int(k), int(n)) if pd.notna(n) and int(n) > 0 else (math.nan, math.nan)
        for k, n in zip(rates["approved"], rates["pairs"], strict=True)
    ]
    return rates


def unique_blockers(pairs: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Count pairs rejected solely by one criterion per Bloom level.

    Args:
        pairs: Annotated pairs DataFrame.
        matrix: Approval matrix from :func:`approved_matrix`.

    Returns:
        DataFrame (Bloom level x criterion) plus total ``rejected`` column.
    """
    counts: dict[str, pd.Series] = {}
    for criterion in CRITERIA:
        others = [c for c in CRITERIA if c != criterion]
        only = matrix[criterion].eq(0.0) & matrix[others].fillna(1.0).eq(1.0).all(axis=1)
        counts[criterion] = only.groupby(pairs["bloom_level"], observed=True).sum()
    table = pd.DataFrame(counts).reindex(BLOOM_ORDER).fillna(0).astype(int)
    table["rejected"] = (
        (~pairs["approved"])
        .groupby(pairs["bloom_level"], observed=True)
        .sum()
        .reindex(BLOOM_ORDER)
        .fillna(0)
        .astype(int)
    )
    return table


def drop_one_scenarios(pairs: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute approved pair counts per Bloom level if each criterion were dropped.

    Args:
        pairs: Annotated pairs DataFrame.
        matrix: Approval matrix from :func:`approved_matrix`.

    Returns:
        DataFrame of Bloom level x scenario with a ``TOTAL`` row.
    """

    def yield_for(columns: list[str]) -> pd.Series:
        ok = matrix[columns].fillna(1.0).eq(1.0).all(axis=1)
        grouped = ok.groupby(pairs["bloom_level"], observed=True).sum()
        return grouped.reindex(BLOOM_ORDER).fillna(0).astype(int)

    scenarios: dict[str, list[str]] = {"current (4 criteria)": CRITERIA}
    for criterion in CRITERIA:
        scenarios[f"without {criterion}"] = [c for c in CRITERIA if c != criterion]

    table = pd.DataFrame({name: yield_for(cols) for name, cols in scenarios.items()})
    table.loc["TOTAL"] = table.sum()
    return table


def applicable_criteria(pairs: pd.DataFrame) -> dict[str, list[str]]:
    """Infer which criteria were evaluated at each Bloom level from observed scores.

    Args:
        pairs: QA pairs DataFrame.

    Returns:
        Mapping from Bloom level to list of criteria with at least one score.
    """
    mapping: dict[str, list[str]] = {}
    for level in BLOOM_ORDER:
        sub = pairs[pairs["bloom_level"] == level]
        mapping[level] = [c for c in CRITERIA if sub[f"score_{c}"].notna().any()]
    return mapping


def rejection_reason(
    pairs: pd.DataFrame,
    matrix: pd.DataFrame,
    applicable: dict[str, list[str]],
) -> pd.Series:
    """Assign each pair a mutually exclusive fate from ``SEGMENT_ORDER``.

    Args:
        pairs: QA pairs DataFrame.
        matrix: Approval matrix from :func:`approved_matrix`.
        applicable: Mapping from :func:`applicable_criteria`.

    Returns:
        Series of segment keys aligned to ``pairs``.
    """
    reason = pd.Series(index=pairs.index, dtype="object", name="reason")
    for level, expected in applicable.items():
        mask = pairs["bloom_level"] == level
        if not mask.any():
            continue
        block = matrix.loc[mask, expected]
        failures = block.eq(0.0)
        n_failures = failures.sum(axis=1)

        label = pd.Series("approved", index=block.index)
        label[(n_failures == 0) & block.isna().any(axis=1)] = "unjudged"
        one_fail = n_failures == 1
        label[one_fail] = failures.loc[one_fail].idxmax(axis=1)
        label[n_failures > 1] = "multi"
        reason[mask] = label
    return reason


def rejection_decomposition(pairs: pd.DataFrame) -> pd.DataFrame:
    """Tabulate mutually exclusive pair outcomes per Bloom level.

    Args:
        pairs: DataFrame with ``bloom_level`` and ``reason`` columns.

    Returns:
        DataFrame (Bloom level x segment key) summing to total pairs per level.
    """
    table = pd.crosstab(pairs["bloom_level"], pairs["reason"])
    return table.reindex(index=BLOOM_ORDER, columns=SEGMENT_ORDER, fill_value=0)


def chunk_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate QA pair counts and useful yield per chunk.

    Args:
        pairs: Annotated pairs DataFrame.

    Returns:
        DataFrame indexed by ``chunk_id`` with ``source_id``, ``chars``,
        ``generated``, ``candidates``, ``useful``, ``rate``, and ``band``.

    Raises:
        ValueError: If ``context_chars`` varies within a single ``chunk_id``.
    """
    if (pairs.groupby("chunk_id", observed=True)["context_chars"].nunique() > 1).any():
        raise ValueError("context_chars varies within a single chunk_id")

    table = pairs.groupby("chunk_id", observed=True).agg(
        source_id=("source_id", "first"),
        chars=("context_chars", "max"),
        generated=("useful", "size"),
        candidates=("candidate", "sum"),
        useful=("useful", "sum"),
    )
    table["rate"] = table["useful"] / table["candidates"].replace(0, float("nan"))
    table["band"] = pd.cut(table["chars"], CHUNK_BINS, labels=CHUNK_LABELS, right=False)
    return table


def band_yield(chunks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate useful-pair rate and density per chunk-length band.

    Args:
        chunks: DataFrame from :func:`chunk_table`.

    Returns:
        DataFrame indexed by length band with ``chunks``, ``chars``,
        ``candidates``, ``useful``, ``rate``, ``ci_lo``, ``ci_hi``, and
        ``useful_per_1k``.
    """
    table = chunks.groupby("band", observed=True).agg(
        chunks=("useful", "size"),
        chars=("chars", "sum"),
        candidates=("candidates", "sum"),
        useful=("useful", "sum"),
    )
    table["rate"] = table["useful"] / table["candidates"].replace(0, float("nan"))
    table[["ci_lo", "ci_hi"]] = [
        wilson_ci(int(k), int(n)) if int(n) > 0 else (math.nan, math.nan)
        for k, n in zip(table["useful"], table["candidates"], strict=True)
    ]
    table["useful_per_1k"] = table["useful"] / table["chars"] * 1000.0
    return table


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman rank correlation without requiring SciPy.

    Args:
        left: First numeric series.
        right: Second numeric series, aligned by index.

    Returns:
        Spearman correlation coefficient in ``[-1, 1]`` (or ``NaN``).
    """
    return float(left.rank().corr(right.rank()))


def short_vs_long(pairs: pd.DataFrame, cutoff: int = 1000) -> pd.DataFrame:
    """Compare conjunctive approval rates below vs. at/above ``cutoff`` chars.

    Args:
        pairs: Annotated pairs DataFrame.
        cutoff: Character length boundary between short and long chunks.

    Returns:
        DataFrame indexed by ``(bloom_level, length_group)`` with ``pairs``,
        ``approved``, ``rate``, ``ci_lo``, and ``ci_hi``.
    """
    short = pairs["context_chars"] < cutoff
    group = pd.Series(
        pd.Categorical(
            short.map({True: f"<{cutoff}", False: f">={cutoff}"}),
            [f"<{cutoff}", f">={cutoff}"],
            ordered=True,
        ),
        index=pairs.index,
        name="length_group",
    )
    table = pairs.groupby([pairs["bloom_level"], group], observed=True)["approved"].agg(
        pairs="size", approved="sum"
    )
    table["rate"] = table["approved"] / table["pairs"]
    table[["ci_lo", "ci_hi"]] = [
        wilson_ci(int(k), int(n)) if int(n) > 0 else (math.nan, math.nan)
        for k, n in zip(table["approved"], table["pairs"], strict=True)
    ]
    return table.reindex(BLOOM_ORDER, level="bloom_level")


def gini(values: pd.Series) -> float:
    """Compute the Gini coefficient of a non-negative distribution.

    Args:
        values: Series of non-negative values.

    Returns:
        Gini coefficient in ``[0, 1]``, or ``NaN`` if the sum is zero.
    """
    ordered = values.dropna().sort_values().to_numpy(dtype=float)
    n = len(ordered)
    total = float(ordered.sum())
    if n == 0 or total == 0.0:
        return math.nan
    pos = pd.RangeIndex(1, n + 1).to_numpy()
    return float((2 * (pos * ordered).sum()) / (n * total) - (n + 1) / n)


def document_profile(chunks: pd.DataFrame) -> pd.DataFrame:
    """Summarize chunks, chars, candidate budget, and useful pairs per document.

    Args:
        chunks: DataFrame from :func:`chunk_table`.

    Returns:
        DataFrame indexed by ``source_id``.
    """
    profile = chunks.groupby("source_id").agg(
        chunks=("chars", "size"),
        chars=("chars", "sum"),
        candidates=("candidates", "sum"),
        useful=("useful", "sum"),
    )
    profile["rate"] = profile["useful"] / profile["candidates"].replace(0, float("nan"))
    profile["useful_per_1k"] = profile["useful"] / profile["chars"] * 1000.0
    return profile


def ladder_accounting(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize the Bloom ladder budget and useful-pair yield per level.

    Args:
        pairs: Annotated pairs DataFrame.

    Returns:
        DataFrame indexed by Bloom level plus ``TOTAL``.
    """
    table = (
        pairs.groupby("bloom_level", observed=True)
        .agg(pairs=("useful", "size"), candidates=("candidate", "sum"), useful=("useful", "sum"))
        .reindex(BLOOM_ORDER)
        .fillna(0)
        .astype(int)
    )
    table.loc["TOTAL"] = table.sum()
    table["useful_rate"] = table["useful"] / table["candidates"].replace(0, float("nan"))
    return table


def coverage_distributions(
    chunks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-chunk, per-document, and cumulative concentration distributions.

    Args:
        chunks: DataFrame from :func:`chunk_table`.

    Returns:
        Tuple ``(per_chunk, per_document, concentration)``.
    """
    ceiling = int(chunks["candidates"].max()) if len(chunks) else 3
    per_chunk = (
        chunks["useful"]
        .value_counts()
        .reindex(range(ceiling + 1), fill_value=0)
        .sort_index()
        .to_frame("chunks")
    )
    per_chunk.index.name = "useful_in_chunk"
    per_chunk["share"] = per_chunk["chunks"] / max(1, len(chunks))

    per_doc_useful = chunks.groupby("source_id")["useful"].sum()
    doc_band = pd.cut(per_doc_useful, DOC_BINS, labels=DOC_LABELS, right=False)
    per_document = doc_band.value_counts().reindex(DOC_LABELS, fill_value=0).to_frame("documents")
    per_document.index.name = "useful_in_document"
    per_document["share"] = per_document["documents"] / max(1, len(per_doc_useful))

    by_doc = chunks.groupby("source_id")
    useful_sorted = per_doc_useful.sort_values(ascending=False)
    cand_sorted = by_doc["candidates"].sum().sort_values(ascending=False)
    chars_sorted = by_doc["chars"].sum().sort_values(ascending=False)
    n_docs = len(useful_sorted)
    concentration = pd.DataFrame(
        {
            "docs_cum": [i / n_docs for i in range(1, n_docs + 1)],
            "useful_cum": useful_sorted.cumsum().to_numpy() / max(1.0, float(useful_sorted.sum())),
            "candidates_cum": (
                cand_sorted.cumsum().to_numpy() / max(1.0, float(cand_sorted.sum()))
            ),
            "chars_cum": chars_sorted.cumsum().to_numpy() / max(1.0, float(chars_sorted.sum())),
        }
    )
    return per_chunk, per_document, concentration


def concentration_shares(
    profile: pd.DataFrame, cutoffs: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
) -> pd.DataFrame:
    """Compute cumulative shares of text, candidates, and useful pairs at top cutoffs.

    Args:
        profile: DataFrame from :func:`document_profile`.
        cutoffs: Fractions of top-ranked documents to report.

    Returns:
        DataFrame indexed by cutoff label.
    """
    rows: dict[str, dict[str, float]] = {}
    for cutoff in cutoffs:
        k = max(1, round(cutoff * len(profile)))
        rows[f"top_{int(cutoff * 100)}pct"] = {
            col: float(
                profile[col].sort_values(ascending=False).iloc[:k].sum()
                / max(1.0, float(profile[col].sum()))
            )
            for col in ("chars", "candidates", "useful")
        }
    return pd.DataFrame(rows).T


def chunk_binomial_clustering(
    pairs: pd.DataFrame, chunks: pd.DataFrame, per_chunk: pd.DataFrame
) -> dict[str, Any]:
    """Compare observed useful-pairs-per-chunk counts against binomial independence.

    Args:
        pairs: Annotated pairs DataFrame.
        chunks: DataFrame from :func:`chunk_table`.
        per_chunk: Chunk distribution from :func:`coverage_distributions`.

    Returns:
        Dictionary with global candidate pass rate, chi-square statistic,
        observed vs. expected chunk utilization, and per-k table.
    """
    ceiling = int(chunks["candidates"].max()) if len(chunks) else 3
    p_global = float(pairs.loc[pairs["candidate"], "useful"].mean())
    expected = {
        k: float(
            len(chunks) * comb(ceiling, k) * (p_global**k) * ((1.0 - p_global) ** (ceiling - k))
        )
        for k in range(ceiling + 1)
    }
    chi2 = 0.0
    table: dict[str, dict[str, float]] = {}
    for k in range(ceiling + 1):
        obs = float(per_chunk.loc[k, "chunks"])
        exp = expected[k]
        part = ((obs - exp) ** 2 / exp) if exp > 0 else 0.0
        chi2 += part
        table[str(k)] = {"observed": obs, "expected": exp, "chi2_part": part}

    utilized = int((chunks["useful"] > 0).sum())
    expected_util = 1.0 - (1.0 - p_global) ** ceiling
    return {
        "candidates_per_chunk": ceiling,
        "global_useful_rate": p_global,
        "observed_utilized_share": utilized / max(1, len(chunks)),
        "expected_utilized_share": expected_util,
        "chi2": chi2,
        "df": ceiling,
        "by_count": table,
    }


def flag_metadata_echo(pairs: pd.DataFrame) -> pd.Series:
    """Heuristically flag pairs whose question or answer echoes interview metadata.

    Args:
        pairs: DataFrame containing ``question`` and ``answer`` columns.

    Returns:
        Boolean Series aligned to ``pairs``.
    """
    answer_hit = pairs["answer"].fillna("").str.contains(META_IN_ANSWER)
    question_hit = pairs["question"].fillna("").str.contains(META_IN_QUESTION)
    return answer_hit | question_hit


def _save_figure(fig: Figure, path: Path) -> None:
    """Save a figure as both PDF and 300 dpi PNG."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    facecolor = fig.get_facecolor()
    fig.savefig(path, facecolor=facecolor, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor=facecolor, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_vs_joint(
    marginal: pd.DataFrame,
    joint: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.5, 3.4),
) -> tuple[Figure, Axes]:
    """Plot marginal per-criterion approval rates against conjunctive approval.

    Args:
        marginal: Output of :func:`approval_rates`.
        joint: Output of :func:`joint_rates`.
        figsize: Figure width and height in inches.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for index, level in enumerate(BLOOM_ORDER):
        y = len(BLOOM_ORDER) - 1 - index
        j_rate = joint.loc[level, "rate"]
        marginals = [
            marginal.loc[(c, level), "rate"]
            for c in CRITERIA
            if pd.notna(marginal.loc[(c, level), "rate"])
        ]
        extremes = [*marginals, j_rate]
        ax.hlines(y, min(extremes), max(extremes), color=GRID, linewidth=7, zorder=1)
        for c in CRITERIA:
            rate = marginal.loc[(c, level), "rate"]
            if pd.isna(rate):
                continue
            ax.scatter(
                rate,
                y,
                marker=CRITERION_MARKERS[c],
                s=32,
                facecolor=SURFACE,
                edgecolor=MUTED,
                linewidth=1.1,
                zorder=2,
            )
        ax.scatter(j_rate, y, s=78, color=BLOOM_RAMP[index], zorder=3)
        ax.annotate(
            f"{j_rate:.0%}",
            (j_rate, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.hlines(
            y,
            joint.loc[level, "ci_lo"],
            joint.loc[level, "ci_hi"],
            color=BLOOM_RAMP[index],
            linewidth=2.2,
            zorder=4,
        )

    ax.set_yticks(range(len(BLOOM_ORDER)))
    ax.set_yticklabels([BLOOM_LABELS[b] for b in reversed(BLOOM_ORDER)])
    ax.set_ylim(-0.6, len(BLOOM_ORDER) - 0.4)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Approval rate", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(AXIS)

    handles = [
        Line2D(
            [],
            [],
            marker=CRITERION_MARKERS[c],
            linestyle="none",
            markersize=5.5,
            markerfacecolor=SURFACE,
            markeredgecolor=MUTED,
            label=CRITERION_LABELS[c],
        )
        for c in CRITERIA
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7,
            color=BLOOM_RAMP[2],
            label="Joint (all criteria)",
        )
    )
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.0,
    )
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig, ax


def plot_rejection_decomposition(
    decomp: pd.DataFrame,
    *,
    normalize: bool = True,
    figsize: tuple[float, float] = (8.5, 3.2),
    min_share: float = 0.028,
) -> tuple[Figure, Axes]:
    """Plot stacked bars decomposing pair outcomes per Bloom level.

    Args:
        decomp: Output of :func:`rejection_decomposition`.
        normalize: When ``True``, normalize each bar to 100%.
        figsize: Figure width and height in inches.
        min_share: Minimum segment width fraction required for an inline count label.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    totals = decomp.sum(axis=1)
    data = decomp.div(totals, axis=0) if normalize else decomp.astype(float)
    limit = 1.0 if normalize else float(totals.max())

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    positions = [len(BLOOM_ORDER) - 1 - i for i in range(len(BLOOM_ORDER))]
    left = pd.Series(0.0, index=data.index)
    for key in SEGMENT_ORDER:
        values = data[key]
        ax.barh(
            positions,
            values,
            left=left,
            height=0.62,
            color=SEGMENT_COLORS[key],
            edgecolor=SURFACE,
            linewidth=1.0,
            label=SEGMENT_LABELS[key],
            zorder=2,
        )
        for y, base, val, count in zip(positions, left, values, decomp[key], strict=True):
            if limit > 0 and val / limit >= min_share:
                ax.text(
                    base + val / 2,
                    y,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=SURFACE if key == "approved" else INK,
                    zorder=3,
                )
        left = left + values

    for y, end, total in zip(positions, left, totals, strict=True):
        label = f"n = {int(total)}" if normalize else f"{int(total)} pairs"
        ax.text(end + limit * 0.015, y, label, va="center", fontsize=8.5, color=MUTED)

    ax.set_yticks(positions)
    ax.set_yticklabels([BLOOM_LABELS[b] for b in BLOOM_ORDER])
    ax.set_ylim(-0.6, len(BLOOM_ORDER) - 0.4)
    ax.set_xlim(0, limit * 1.13)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if normalize:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel(
        "Share of generated pairs" if normalize else "Generated pairs",
        color=MUTED,
        fontsize=9,
    )
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(AXIS)

    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.0,
    )
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig, ax


def plot_chunk_yield(
    bands: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.5, 3.2),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot useful-pair rate and density per 1k chars across chunk-length bands.

    Args:
        bands: Output of :func:`band_yield`.
        figsize: Figure width and height in inches.

    Returns:
        Matplotlib ``(figure, (ax_rate, ax_density))`` tuple.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    fig, (ax_rate, ax_dens) = plt.subplots(1, 2, figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    positions = list(range(len(bands)))

    for ax in (ax_rate, ax_dens):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(positions)
        ax.set_xticklabels(bands.index, rotation=45, ha="right")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(AXIS)
        ax.set_xlabel("Chunk length (characters)", color=MUTED, fontsize=8.5)

    ax_rate.vlines(positions, bands["ci_lo"], bands["ci_hi"], color=MUTED, linewidth=1.2)
    ax_rate.scatter(positions, bands["rate"], s=48, color=BLOOM_RAMP[2], zorder=3)
    ax_rate.set_ylim(0, min(1.0, max(0.4, float(bands["ci_hi"].max()) * 1.15)))
    ax_rate.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_rate.set_ylabel("Useful pairs / candidates", color=MUTED, fontsize=8.5)
    ax_rate.set_title("Useful yield rate", loc="left", fontsize=10, color=INK)

    ax_dens.vlines(positions, 0, bands["useful_per_1k"], color=GRID, linewidth=6)
    ax_dens.scatter(positions, bands["useful_per_1k"], s=48, color=BLOOM_RAMP[3], zorder=3)
    ax_dens.set_ylim(0, float(bands["useful_per_1k"].max()) * 1.22)
    ax_dens.set_ylabel("Useful pairs per 1,000 chars", color=MUTED, fontsize=8.5)
    ax_dens.set_title("Yield density", loc="left", fontsize=10, color=INK)
    for x, val in zip(positions, bands["useful_per_1k"], strict=True):
        ax_dens.annotate(
            f"{val:.2f}",
            (x, val),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color=MUTED,
        )

    sns.despine(fig=fig, left=True)
    fig.tight_layout()
    return fig, (ax_rate, ax_dens)


def analyze_run(
    cep_dir: Path,
    *,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    figures_dir: Path | None = None,
    run_id: str = "run",
) -> dict[str, Any]:
    """Run the complete Bloom/CEP yield analysis on a CEP outputs directory.

    Args:
        cep_dir: Directory containing ``*_cep_qa.json`` files.
        min_chunk_chars: Minimum chunk length floor in characters.
        figures_dir: Optional directory where PDF/PNG figures are written.
        run_id: Run identifier used in summary metadata and figure filenames.

    Returns:
        JSON-serializable dictionary with all summary tables and metrics.
    """
    raw = load_cep_pairs(cep_dir)
    kept, dropped = drop_tiny_chunks(raw, min_chars=min_chunk_chars)
    pairs = with_usefulness(kept)
    matrix = approved_matrix(pairs)

    long_df = criterion_long(pairs)
    marginal = approval_rates(long_df)
    joint = joint_rates(pairs)
    blockers = unique_blockers(pairs, matrix)
    drop_one = drop_one_scenarios(pairs, matrix)

    pairs["reason"] = rejection_reason(pairs, matrix, applicable_criteria(pairs))
    decomp = rejection_decomposition(pairs)

    chunks = chunk_table(pairs)
    bands = band_yield(chunks)
    short_long = short_vs_long(pairs)
    ladder = ladder_accounting(pairs)
    per_chunk, per_doc_dist, _ = coverage_distributions(chunks)
    profile = document_profile(chunks)
    conc = concentration_shares(profile)
    clustering = chunk_binomial_clustering(pairs, chunks, per_chunk)

    pairs["metadata_echo"] = flag_metadata_echo(pairs)
    approved_pairs = pairs[pairs["approved"]]

    if figures_dir is not None:
        fig_mj, _ = plot_marginal_vs_joint(marginal, joint)
        _save_figure(fig_mj, figures_dir / f"marginal-vs-joint-{run_id}.pdf")
        fig_dec, _ = plot_rejection_decomposition(decomp, normalize=True)
        _save_figure(fig_dec, figures_dir / f"rejection-decomposition-{run_id}.pdf")
        fig_cy, _ = plot_chunk_yield(bands)
        _save_figure(fig_cy, figures_dir / f"chunk-yield-{run_id}.pdf")

    return {
        "run_id": run_id,
        "min_chunk_chars": min_chunk_chars,
        "corpus": {
            "raw_documents": int(raw["source_id"].nunique()),
            "raw_chunks": int(raw["chunk_id"].nunique()),
            "raw_pairs": len(raw),
            "dropped_chunks": len(dropped),
            "dropped_pairs": int(dropped["pairs"].sum()) if len(dropped) else 0,
            "retained_documents": int(pairs["source_id"].nunique()),
            "retained_chunks": len(chunks),
            "retained_pairs": len(pairs),
            "approved_pairs": int(pairs["approved"].sum()),
            "candidate_pairs": int(pairs["candidate"].sum()),
            "useful_pairs": int(pairs["useful"].sum()),
        },
        "joint_rates": joint.reset_index().to_dict(orient="records"),
        "marginal_rates": marginal.reset_index().to_dict(orient="records"),
        "unique_blockers": blockers.reset_index().to_dict(orient="records"),
        "drop_one_scenarios": drop_one.reset_index().to_dict(orient="records"),
        "rejection_decomposition": decomp.reset_index().to_dict(orient="records"),
        "ladder_accounting": ladder.reset_index().to_dict(orient="records"),
        "chunk_band_yield": bands.reset_index().to_dict(orient="records"),
        "short_vs_long": short_long.reset_index().to_dict(orient="records"),
        "correlations": {
            "spearman_chunk_chars_vs_useful_rate": spearman(chunks["chars"], chunks["rate"]),
            "spearman_doc_chars_vs_useful_per_1k": spearman(
                profile["chars"], profile["useful_per_1k"]
            ),
            "spearman_doc_chars_vs_useful_rate": spearman(profile["chars"], profile["rate"]),
        },
        "gini": {
            "chars": gini(profile["chars"]),
            "candidates": gini(profile["candidates"]),
            "useful": gini(profile["useful"]),
            "useful_rate": gini(profile["rate"]),
        },
        "concentration_shares": conc.reset_index().to_dict(orient="records"),
        "per_document_distribution": per_doc_dist.reset_index().to_dict(orient="records"),
        "chunk_binomial_clustering": clustering,
        "metadata_echo": {
            "all_pairs": int(pairs["metadata_echo"].sum()),
            "all_pairs_rate": float(pairs["metadata_echo"].mean()),
            "approved_pairs": int(approved_pairs["metadata_echo"].sum()),
            "approved_pairs_rate": (
                float(approved_pairs["metadata_echo"].mean()) if len(approved_pairs) else math.nan
            ),
            "useful_pairs": int(pairs.loc[pairs["useful"], "metadata_echo"].sum()),
        },
    }


def _fmt(val: Any, nd: int = 3) -> str:
    """Format floats cleanly, rendering NaN/None as ``n/a``."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    if isinstance(val, float):
        return f"{val:.{nd}f}"
    return str(val)


def render_markdown(res: dict[str, Any]) -> str:
    """Render the analysis result dictionary as a Markdown report.

    Args:
        res: Result dictionary from :func:`analyze_run`.

    Returns:
        Markdown-formatted report string.
    """
    c = res["corpus"]
    lines = [
        f"# Bloom / CEP Yield Analysis ({res['run_id']})",
        "",
        "## Overview",
        "",
        f"- Documents: {c['retained_documents']} (raw: {c['raw_documents']})",
        f"- Chunks: {c['retained_chunks']} retained, {c['dropped_chunks']} dropped "
        f"(< {res['min_chunk_chars']} chars, {c['dropped_pairs']} pairs)",
        f"- Pairs: {c['retained_pairs']} retained, {c['approved_pairs']} approved, "
        f"{c['useful_pairs']} useful out of {c['candidate_pairs']} higher-order candidates",
        "",
        "## Joint Approval by Bloom Level",
        "",
        "| Bloom level | Pairs | Approved | Rate | 95% Wilson CI |",
        "|---|---:|---:|---:|---|",
    ]
    for row in res["joint_rates"]:
        lines.append(
            f"| {row['bloom_level']} | {int(row['pairs'])} | {int(row['approved'])} "
            f"| {_fmt(row['rate'])} | [{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] |"
        )

    lines += [
        "",
        "## Additive Rejection Decomposition",
        "",
        "| Bloom level | " + " | ".join(SEGMENT_ORDER) + " |",
        "|---|" + "---:|" * len(SEGMENT_ORDER),
    ]
    for row in res["rejection_decomposition"]:
        vals = " | ".join(str(int(row[seg])) for seg in SEGMENT_ORDER)
        lines.append(f"| {row['bloom_level']} | {vals} |")

    lines += [
        "",
        "## Chunk-Length Band Yield",
        "",
        "| Band | Chunks | Candidates | Useful | Rate | Useful / 1k chars |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in res["chunk_band_yield"]:
        lines.append(
            f"| {row['band']} | {int(row['chunks'])} | {int(row['candidates'])} "
            f"| {int(row['useful'])} | {_fmt(row['rate'])} | {_fmt(row['useful_per_1k'], 2)} |"
        )

    g = res["gini"]
    cl = res["chunk_binomial_clustering"]
    me = res["metadata_echo"]
    lines += [
        "",
        "## Concentration, Inequality, and Metadata Echo",
        "",
        f"- Gini (chars / candidates / useful): "
        f"{_fmt(g['chars'])} / {_fmt(g['candidates'])} / {_fmt(g['useful'])}",
        f"- Utilized chunks: {_fmt(cl['observed_utilized_share'])} "
        f"(under binomial independence: {_fmt(cl['expected_utilized_share'])}, "
        f"chi2 = {_fmt(cl['chi2'], 1)}, df = {cl['df']})",
        f"- Metadata echo: {me['all_pairs']} all ({_fmt(me['all_pairs_rate'])}), "
        f"{me['approved_pairs']} approved ({_fmt(me['approved_pairs_rate'])}), "
        f"{me['useful_pairs']} useful",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point for Bloom/CEP yield analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Bloom-stratified CEP pair yield, rejections, and concentration."
    )
    parser.add_argument("--id", required=True, help="Pipeline run ID (e.g. thesis-run-02).")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Results root.")
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=MIN_CHUNK_CHARS,
        help="Minimum chunk length in characters (default: 100).",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional directory to write PDF and 300 dpi PNG figures.",
    )
    args = parser.parse_args()

    cep_dir = args.results_root / args.id / "cep" / "outputs"
    res = analyze_run(
        cep_dir,
        min_chunk_chars=args.min_chunk_chars,
        figures_dir=args.figures_dir,
        run_id=args.id,
    )
    md = render_markdown(res)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
