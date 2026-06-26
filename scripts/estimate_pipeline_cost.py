#!/usr/bin/env python
"""Estimate full-pipeline cost (transcribe -> rag-analysis) to pre-check viability.

The pipeline's cost is NOT a single number: it spans three units that cannot
be summed, so they are reported separately.

  (A) GPU transcription time  -- whisper-large; audio_hours * real_time_factor.
  (B) LLM generation calls    -- judge-transcription, build-kg (triple extraction
                                 + concept generation), generate-cep-qa, judge-qa,
                                 generate-non-answerable, retrieve NER, answer,
                                 judge-answers.
  (C) Embedding calls         -- kg-build-retriever-index (nodes + edges + passages)
                                 and query encoding at retrieve.

Stages with no model cost (chunk, kg-link-passages, rag-analysis) are omitted.

LLM-call model (coefficients read from the code, 2026-06):

    judge-transcription   n_docs * 2                 language_drift + hallucination_loop
                                                      (gated behind no-LLM heuristics)
    build-kg extraction   kg_passages * passes       AutoSchemaKG passes per passage (~3)
    build-kg concept      kg_nodes / concept_batch    batch_size_concept = 16
    generate-cep-qa       cep_chunks * Q * (1 + 1)    generation + reasoning[if enabled]
    judge-qa              cep_pairs * criteria        4 (3 for 'remember'-level pairs)
    generate-non-answer.  n_nonanswerable * 1         entity-swap, happy path
    retrieve NER          N_questions * 1             only the atlas_rag arm hits an LLM
    answer                n_arms * N_questions
    judge-answers         n_arms * (answerable*4 + non_answerable*1)   gated cascade

  cep_chunks  = sum ceil(doc_chars / cep_chunk_size)     (CEP generation chunking)
  kg_passages = sum ceil(doc_chars / kg_passage_size)    (atlas-rag chunking, 8192)
  cep_pairs   = cep_chunks * Q
  answerable  = cep_pairs * judge_pass_rate
  N_questions = answerable + n_nonanswerable

KG entity density (nodes/edges per passage) is corpus-specific. Defaults are
calibrated to the test-kg-04 run (~14k nodes, ~60k edges over 422 passages);
override --kg-nodes-per-passage / --kg-edges-per-node for a different corpus.

Complexity (LLM calls): O(n_arms * (corpus_chars / cep_chunk_size) * Q) for the
QA/eval chain + O(corpus_chars / kg_passage_size) for KG. Linear in corpus size
and ladder size, inversely linear in chunk size, fixed xN for the arms.

Dollar cost (--providers): on top of the call counts, each stage carries a
rough token profile (input + output tokens per call); multiplying by a
provider's per-token price gives a USD estimate, split into the GENERATION
phase (corpus/KG build) and the EVALUATION phase (judge + RAG eval). The
built-in PRICING table holds representative public list prices and MUST be
verified against the provider's pricing page before being cited; override it
with --pricing-file.

Usage:
    python scripts/estimate_pipeline_cost.py --id thesis-run-01 --cep-chunk-size 8000 -Q 6
    python scripts/estimate_pipeline_cost.py --id thesis-run-01 -Q 6 --remember-fraction 0.5 \
        --judge-pass-rate 0.54 --nonanswerable 400 --providers gemini-2.5-flash,gpt-4o
    python scripts/estimate_pipeline_cost.py --synthetic 20:80000 --cep-chunk-size 8000 \
        -Q 10 --nonanswerable 400 --audio-hours 30 --providers all
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# --- LLM-generation coefficients (from the code) ---------------------------
JUDGE_TRANSCRIPTION_CRITERIA = 2  # language_drift + hallucination_loop (gated)
GEN_CALLS_PER_PAIR = 1  # bloom_scaffolding: one generate() per pair
REASONING_CALLS_PER_PAIR = 1  # reasoning.enrich: one generate() per pair
JUDGE_QA_CRITERIA = 4  # faithfulness, bloom_calibration, informativeness, self_containedness
JUDGE_QA_CRITERIA_REMEMBER = 3  # 'remember' level drops self_containedness
NONANSWERABLE_CALLS_PER_ITEM = 1  # perturbation: one generate_structured (happy path)
RETRIEVE_NER_CALLS_PER_Q = 1  # only atlas_rag runs an LLM (NER) at retrieve time
ANSWER_CALLS_PER_ARM_Q = 1  # answerer: one generate per (arm, question)
JUDGE_ANS_CALLS_ANSWERABLE = 4  # abstention + passage_coverage + correctness + faithfulness
JUDGE_ANS_CALLS_NONANSWERABLE = 1  # non-answerable fails answerability gate -> abstention only

# --- KG / embedding coefficients -------------------------------------------
KG_PASSAGE_SIZE = 8192  # atlas-rag ATLAS_DEFAULTS["chunk_size"]
KG_EXTRACTION_PASSES = 3  # AutoSchemaKG: entity-relation, event-entity, event-relation
KG_CONCEPT_BATCH = 16  # ATLAS_DEFAULTS["batch_size_concept"]
# Calibrated to test-kg-04: ~14000 nodes, ~60000 edges over 422 passages.
KG_NODES_PER_PASSAGE = 33.0
KG_EDGES_PER_NODE = 4.3

DEFAULT_ARMS = 5  # bm25, atlas_rag, khop_passage, khop_triple, null
# Default Bloom ladder is 3:1:1:1 (remember:understand:analyze:evaluate) -> remember = 3/6.
REMEMBER_FRACTION = 0.5
WHISPER_RTF = 0.30  # whisper-large-v3 GPU real-time factor (frac of audio duration)

# --- Token + pricing model -------------------------------------------------
CHARS_PER_TOKEN = 4.0  # rough English/Portuguese heuristic for char->token conversion
RETRIEVED_CONTEXT_TOKENS = 1500.0  # approx top-k passages fed to answerer / answer-judge

# Fixed per-call prompt/instruction overhead in tokens (system + rubric), by stage.
PROMPT_TOKENS: dict[str, float] = {
    "judge_transcription": 500,
    "build_kg_extraction": 600,
    "build_kg_concept": 400,
    "cep_generation": 700,
    "judge_qa": 800,
    "generate_non_answerable": 500,
    "retrieve_ner": 300,
    "answer": 600,
    "judge_answers": 800,
}

# Estimated output (completion) tokens per call, by stage.
OUTPUT_TOKENS: dict[str, float] = {
    "judge_transcription": 150,
    "build_kg_extraction": 1200,  # triple lists are verbose
    "build_kg_concept": 300,
    "cep_generation": 200,  # one QA pair; reasoning adds more (handled below)
    "judge_qa": 250,  # verdict + rationale, per criterion
    "generate_non_answerable": 200,
    "retrieve_ner": 150,
    "answer": 300,
    "judge_answers": 250,
}

# Phase classification for the generation-vs-evaluation decision.
GENERATION_STAGES = frozenset(
    {"build_kg_extraction", "build_kg_concept", "cep_generation", "generate_non_answerable"}
)
EVALUATION_STAGES = frozenset(
    {"judge_transcription", "judge_qa", "retrieve_ner", "answer", "judge_answers"}
)

# Representative public list prices in USD per 1,000,000 tokens (input, output).
# AS OF 2026-06 AND APPROXIMATE -- VERIFY against the provider's pricing page before
# citing, or override with --pricing-file (JSON: {"name": [in_per_1m, out_per_1m]}).
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "local-ollama": (0.0, 0.0),  # self-hosted: $0 in tokens, cost is GPU hours instead
}


@dataclass
class Config:
    """Estimation inputs."""

    cep_chunk_size: int
    questions_per_chunk: int
    reasoning_enabled: bool
    judge_pass_rate: float
    remember_fraction: float
    nonanswerable: int
    n_arms: int
    kg_passage_size: int
    kg_nodes_per_passage: float
    kg_edges_per_node: float
    audio_hours: float
    whisper_rtf: float
    cep_chunks: int | None = None  # real CEP chunk count; overrides the char-based model


def _chunks(doc_chars: int, size: int) -> int:
    """Ceil-division chunk count (lower bound; boundary-aware chunker emits ~10-20% more)."""
    return max(1, math.ceil(doc_chars / size))


def _token_profiles(cfg: Config, mean_doc_tokens: float) -> dict[str, tuple[float, float]]:
    """Build per-stage (input_tokens, output_tokens) per-call estimates.

    Context-dependent inputs derive from chunk/passage sizes; the rest are the
    fixed prompt overhead. These are deliberately rough: they exist to turn
    call counts into an order-of-magnitude dollar figure, not to be exact.

    Args:
        cfg: Estimation inputs.
        mean_doc_tokens: Mean transcription length in tokens (for judge-transcription).

    Returns:
        Mapping of stage name to ``(input_tokens, output_tokens)`` per call.
    """
    cep_chunk_tokens = cfg.cep_chunk_size / CHARS_PER_TOKEN
    kg_passage_tokens = cfg.kg_passage_size / CHARS_PER_TOKEN
    cep_out = OUTPUT_TOKENS["cep_generation"] + (
        OUTPUT_TOKENS["cep_generation"] if cfg.reasoning_enabled else 0
    )
    # input = fixed prompt overhead + the variable context the stage actually reads.
    context_in: dict[str, float] = {
        "judge_transcription": mean_doc_tokens,
        "build_kg_extraction": kg_passage_tokens,
        "build_kg_concept": KG_CONCEPT_BATCH * 20,  # ~16 node names per batch
        "cep_generation": cep_chunk_tokens,
        "judge_qa": cep_chunk_tokens + 100,  # chunk context + the QA pair
        "generate_non_answerable": 200,  # one QA pair
        "retrieve_ner": 50,  # just the question
        "answer": RETRIEVED_CONTEXT_TOKENS + 50,  # passages + question
        "judge_answers": RETRIEVED_CONTEXT_TOKENS + 350,  # passages + question + answer
    }
    profiles: dict[str, tuple[float, float]] = {}
    for stage, ctx in context_in.items():
        out = cep_out if stage == "cep_generation" else OUTPUT_TOKENS[stage]
        profiles[stage] = (PROMPT_TOKENS[stage] + ctx, out)
    return profiles


def estimate(doc_chars: list[int], cfg: Config) -> dict[str, object]:
    """Compute per-stage cost across the three unit groups, with token totals."""
    n_docs = len(doc_chars)
    cep_chunks = (
        cfg.cep_chunks
        if cfg.cep_chunks is not None
        else sum(_chunks(c, cfg.cep_chunk_size) for c in doc_chars)
    )
    kg_passages = sum(_chunks(c, cfg.kg_passage_size) for c in doc_chars)
    kg_nodes = kg_passages * cfg.kg_nodes_per_passage
    kg_edges = kg_nodes * cfg.kg_edges_per_node

    pairs = cep_chunks * cfg.questions_per_chunk
    answerable = pairs * cfg.judge_pass_rate
    n_questions = answerable + cfg.nonanswerable

    gen_per_pair = GEN_CALLS_PER_PAIR + (REASONING_CALLS_PER_PAIR if cfg.reasoning_enabled else 0)
    avg_judge_qa = (
        cfg.remember_fraction * JUDGE_QA_CRITERIA_REMEMBER
        + (1 - cfg.remember_fraction) * JUDGE_QA_CRITERIA
    )

    llm: dict[str, float] = {
        "judge_transcription": n_docs * JUDGE_TRANSCRIPTION_CRITERIA,
        "build_kg_extraction": kg_passages * KG_EXTRACTION_PASSES,
        "build_kg_concept": kg_nodes / KG_CONCEPT_BATCH,
        "cep_generation": pairs * gen_per_pair,
        "judge_qa": pairs * avg_judge_qa,
        "generate_non_answerable": cfg.nonanswerable * NONANSWERABLE_CALLS_PER_ITEM,
        "retrieve_ner": n_questions * RETRIEVE_NER_CALLS_PER_Q,
        "answer": cfg.n_arms * n_questions * ANSWER_CALLS_PER_ARM_Q,
        "judge_answers": cfg.n_arms
        * (
            answerable * JUDGE_ANS_CALLS_ANSWERABLE
            + cfg.nonanswerable * JUDGE_ANS_CALLS_NONANSWERABLE
        ),
    }

    mean_doc_tokens = (sum(doc_chars) / n_docs) / CHARS_PER_TOKEN if n_docs else 0.0
    profiles = _token_profiles(cfg, mean_doc_tokens)
    tokens_in = {stage: calls * profiles[stage][0] for stage, calls in llm.items()}
    tokens_out = {stage: calls * profiles[stage][1] for stage, calls in llm.items()}

    # Embeddings: index build (nodes + edges + passages) + one query encoding per question.
    embeddings = kg_nodes + kg_edges + kg_passages + n_questions
    gpu_transcription_hours = cfg.audio_hours * cfg.whisper_rtf

    return {
        "n_docs": n_docs,
        "cep_chunks": cep_chunks,
        "kg_passages": kg_passages,
        "kg_nodes": kg_nodes,
        "kg_edges": kg_edges,
        "cep_pairs": pairs,
        "answerable": answerable,
        "n_questions": n_questions,
        "llm": llm,
        "llm_total": sum(llm.values()),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_in_total": sum(tokens_in.values()),
        "tokens_out_total": sum(tokens_out.values()),
        "embedding_calls": embeddings,
        "gpu_transcription_hours": gpu_transcription_hours,
    }


def stage_cost_usd(
    tokens_in: dict[str, float], tokens_out: dict[str, float], price_in: float, price_out: float
) -> dict[str, float]:
    """Per-stage USD cost from token totals and a provider's per-1M-token prices."""
    return {
        stage: tokens_in[stage] / 1e6 * price_in + tokens_out[stage] / 1e6 * price_out
        for stage in tokens_in
    }


def phase_totals(stage_values: dict[str, float]) -> dict[str, float]:
    """Split a per-stage map into generation, evaluation, and total subtotals."""
    generation = sum(v for s, v in stage_values.items() if s in GENERATION_STAGES)
    evaluation = sum(v for s, v in stage_values.items() if s in EVALUATION_STAGES)
    return {"generation": generation, "evaluation": evaluation, "total": generation + evaluation}


def read_corpus(run_id: str, results_root: Path) -> tuple[list[int], float]:
    """Read (char counts, total audio hours) from a run's transcription outputs."""
    out = results_root / run_id / "transcription" / "outputs"
    chars: list[int] = []
    total_ms = 0
    for f in sorted(out.glob("*_transcription.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        chars.append(len(data.get("transcription_text") or ""))
        total_ms += int(data.get("duration_milliseconds") or 0)
    if not chars:
        raise SystemExit(f"No transcription outputs found under {out}")
    return chars, total_ms / 3_600_000


def parse_synthetic(spec: str) -> list[int]:
    """Parse ``N:CHARS`` synthetic corpus spec (N docs of CHARS chars each)."""
    n, chars = spec.split(":")
    return [int(chars)] * int(n)


def load_pricing(pricing_file: Path | None) -> dict[str, tuple[float, float]]:
    """Return the pricing table, overlaying a JSON override file if given."""
    if pricing_file is None:
        return dict(PRICING)
    raw = json.loads(pricing_file.read_text(encoding="utf-8"))
    merged = dict(PRICING)
    for name, pair in raw.items():
        merged[name] = (float(pair[0]), float(pair[1]))
    return merged


def resolve_providers(spec: str, pricing: dict[str, tuple[float, float]]) -> list[str]:
    """Resolve a comma-separated provider spec (or 'all') against the pricing table."""
    if spec == "all":
        return list(pricing)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in pricing]
    if unknown:
        raise SystemExit(f"Unknown provider(s): {', '.join(unknown)}. Known: {', '.join(pricing)}")
    return names


def _print_calls(r: dict[str, object]) -> None:
    """Print the call-count and resource sections (the provider-agnostic view)."""
    llm: dict[str, float] = r["llm"]  # type: ignore[assignment]
    print("=" * 64)
    print("  (B) LLM CALLS / TOKENS")
    print(f"      {'stage':<26}{'calls':>10}{'tok_in':>14}{'tok_out':>12}")
    tokens_in: dict[str, float] = r["tokens_in"]  # type: ignore[assignment]
    tokens_out: dict[str, float] = r["tokens_out"]  # type: ignore[assignment]
    for stage, n in llm.items():
        print(f"      {stage:<26}{n:>10,.0f}{tokens_in[stage]:>14,.0f}{tokens_out[stage]:>12,.0f}")
    print(
        f"      {'-- TOTAL':<26}{r['llm_total']:>10,.0f}"
        f"{r['tokens_in_total']:>14,.0f}{r['tokens_out_total']:>12,.0f}"
    )
    print("=" * 64)
    print(f"  (C) EMBEDDING CALLS         {r['embedding_calls']:>14,.0f}")
    print(f"  (A) GPU TRANSCRIPTION (h)   {r['gpu_transcription_hours']:>14.1f}")


def _print_costs(r: dict[str, object], providers: list[str], pricing: dict) -> None:
    """Print the per-provider USD cost table with generation/evaluation split."""
    tokens_in: dict[str, float] = r["tokens_in"]  # type: ignore[assignment]
    tokens_out: dict[str, float] = r["tokens_out"]  # type: ignore[assignment]
    print("=" * 64)
    print("  USD COST BY PROVIDER (rough; verify PRICING before citing)")
    print(f"      {'provider':<20}{'generation':>14}{'evaluation':>14}{'total':>12}")
    for name in providers:
        price_in, price_out = pricing[name]
        per_stage = stage_cost_usd(tokens_in, tokens_out, price_in, price_out)
        ph = phase_totals(per_stage)
        print(
            f"      {name:<20}${ph['generation']:>12,.2f}"
            f"${ph['evaluation']:>12,.2f}${ph['total']:>10,.2f}"
        )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--id", help="Pipeline run id; reads char counts + audio hours from results/<id>/."
    )
    src.add_argument("--synthetic", help="Synthetic corpus 'N:CHARS' (N docs of CHARS chars).")
    p.add_argument("--results-root", default="results", type=Path)
    p.add_argument(
        "--cep-chunk-size", type=int, default=8000, help="CEP generation chunk size (chars)."
    )
    p.add_argument("-Q", "--questions", type=int, default=10, help="Bloom-ladder size per chunk.")
    p.add_argument(
        "--cep-chunks",
        type=int,
        default=None,
        help="Override modeled CEP chunk count with the real value (cep_pairs = cep_chunks*Q).",
    )
    p.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Model the (deleted 2026-06-17) Module II reasoning-enrichment call per pair. "
        "Off by default; thinking/reasoning tokens are not modeled.",
    )
    p.add_argument(
        "--judge-pass-rate", type=float, default=0.8, help="Fraction of pairs surviving judge-qa."
    )
    p.add_argument(
        "--remember-fraction",
        type=float,
        default=REMEMBER_FRACTION,
        help="Fraction of pairs at the 'remember' Bloom level (3:1:1:1 ladder -> 0.5).",
    )
    p.add_argument(
        "--nonanswerable", type=int, default=0, help="Non-answerable items entering retrieve."
    )
    p.add_argument("--arms", type=int, default=DEFAULT_ARMS, help="Retrieval arms.")
    p.add_argument("--kg-passage-size", type=int, default=KG_PASSAGE_SIZE)
    p.add_argument("--kg-nodes-per-passage", type=float, default=KG_NODES_PER_PASSAGE)
    p.add_argument("--kg-edges-per-node", type=float, default=KG_EDGES_PER_NODE)
    p.add_argument(
        "--audio-hours", type=float, default=0.0, help="Total audio hours (for GPU estimate)."
    )
    p.add_argument(
        "--whisper-rtf", type=float, default=WHISPER_RTF, help="Whisper real-time factor."
    )
    p.add_argument(
        "--providers",
        default="",
        help="Comma-separated provider names or 'all' for a USD cost table. Omit to skip.",
    )
    p.add_argument(
        "--pricing-file",
        type=Path,
        default=None,
        help='JSON file overriding PRICING: {"name": [in_per_1m, out_per_1m]}.',
    )
    args = p.parse_args()

    audio_hours = args.audio_hours
    if args.synthetic:
        doc_chars = parse_synthetic(args.synthetic)
    else:
        doc_chars, measured_hours = read_corpus(args.id, args.results_root)
        if audio_hours == 0.0:
            audio_hours = measured_hours

    cfg = Config(
        cep_chunk_size=args.cep_chunk_size,
        questions_per_chunk=args.questions,
        reasoning_enabled=args.with_reasoning,
        judge_pass_rate=args.judge_pass_rate,
        remember_fraction=args.remember_fraction,
        nonanswerable=args.nonanswerable,
        n_arms=args.arms,
        kg_passage_size=args.kg_passage_size,
        kg_nodes_per_passage=args.kg_nodes_per_passage,
        kg_edges_per_node=args.kg_edges_per_node,
        audio_hours=audio_hours,
        whisper_rtf=args.whisper_rtf,
        cep_chunks=args.cep_chunks,
    )
    r = estimate(doc_chars, cfg)

    print(f"corpus: {r['n_docs']} docs, {sum(doc_chars):,} chars, {audio_hours:.1f} audio-hours")
    print(
        f"config: cep_chunk={cfg.cep_chunk_size}, Q={cfg.questions_per_chunk}, "
        f"reasoning={cfg.reasoning_enabled}, judge_pass={cfg.judge_pass_rate}, "
        f"remember={cfg.remember_fraction}, non_answerable={cfg.nonanswerable}, "
        f"arms={cfg.n_arms}, kg_passage={cfg.kg_passage_size}"
    )
    print("-" * 64)
    print(
        f"  cep_chunks={r['cep_chunks']:,}  kg_passages={r['kg_passages']:,}  "
        f"kg_nodes~{r['kg_nodes']:,.0f}  kg_edges~{r['kg_edges']:,.0f}"
    )
    print(
        f"  cep_pairs={r['cep_pairs']:,.0f}  answerable~{r['answerable']:,.0f}  "
        f"questions~{r['n_questions']:,.0f}"
    )
    _print_calls(r)
    if args.providers:
        pricing = load_pricing(args.pricing_file)
        providers = resolve_providers(args.providers, pricing)
        _print_costs(r, providers, pricing)
    print("=" * 64)


if __name__ == "__main__":
    main()
