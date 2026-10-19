#!/usr/bin/env python
"""Estimate full-pipeline cost (transcribe -> rag-analysis) to pre-check viability.

The pipeline's cost is NOT a single number: it spans three units that cannot
be summed, so they are reported separately.

  (A) GPU transcription time  -- whisper-large; audio_hours * real_time_factor.
  (B) LLM generation calls    -- judge-transcription, build-kg (triple extraction
                                 + concept generation), generate-cep-qa (with the
                                 scaffolding gate), judge-qa, emic-judge,
                                 generate-non-answerable, retrieve NER, answer,
                                 judge-answers.
  (C) Embedding calls         -- kg-build-retriever-index (nodes + edges + passages)
                                 and query encoding at retrieve.

Stages with no model cost (chunk, kg-link-passages, rag-analysis) are omitted.

LLM-call model (coefficients read from the code, 2026-10):

    judge-transcription   n_docs * 2                 language_drift + hallucination_loop
                                                      (gated behind no-LLM heuristics)
    build-kg extraction   kg_passages * passes       AutoSchemaKG passes per passage (~3)
    build-kg concept      kg_nodes / concept_batch    batch_size_concept = 16
    generate-cep-qa       cep_chunks * Q * 1          one generation call per pair
    judge-qa              cep_pairs * criteria        GATE_CRITERIA (4); REMEMBER_CRITERIA
                                                      (2) for 'remember' pairs
    emic-judge            cep_pairs * 1               emic_validity, scope 'all'
    generate-non-answer.  n_nonanswerable * 1         entity-swap, happy path
    retrieve NER          N_questions * 1             only the atlas_rag arm hits an LLM
    answer                n_arms * N_questions
    judge-answers         n_arms * (answerable*4 + non_answerable*1)   gated cascade

  passed_docs = transcriptions the judge admits (CEP and KG skip rejected ones)
  cep_chunks  = sum ceil(doc_chars / cep_chunk_size)  over passed_docs (cep_4k view)
  kg_passages = sum ceil(doc_chars / kg_passage_size) over passed_docs (atlas-rag, 8192)
  cep_pairs   = cep_chunks * Q
  answerable  = cep_pairs * judge_pass_rate
  N_questions = answerable + n_nonanswerable

With the scaffolding gate on (the default) every pair is judged inside
generate-cep-qa, so judge-qa's calls are booked to the GENERATION phase and
judge-qa itself only re-judges pairs without a clean verdict. Pass --no-gate to
book them to EVALUATION instead.

KG entity density (nodes/edges per passage) is corpus-specific. Defaults are
calibrated to the test-kg-04 run (~14k nodes, ~60k edges over 422 passages);
override --kg-nodes-per-passage / --kg-edges-per-node for a different corpus.

Complexity (LLM calls): O(n_arms * (corpus_chars / cep_chunk_size) * Q) for the
QA/eval chain + O(corpus_chars / kg_passage_size) for KG. Linear in corpus size
and ladder size, inversely linear in chunk size, fixed xN for the arms.

Dollar cost (--providers): on top of the call counts, each stage carries a
rough token profile (input + output tokens per call); multiplying by a
provider's per-token price gives a USD estimate, split into the GENERATION
phase (corpus/KG build) and the EVALUATION phase (judge + RAG eval). Thinking
models also bill their reasoning as output: --reasoning-tokens adds that many
output tokens to every call (0 assumes thinking is off, e.g.
ARANDU_<STAGE>_EXTRA_BODY='{"reasoning_effort": "none"}'). Context-cache
discounts are not modelled, so prices are cache-miss list prices. The built-in
PRICING table MUST be verified against the provider's pricing page before being
cited; override it with --pricing-file.

Usage:
    python scripts/estimate_pipeline_cost.py --id thesis-run-02 --nonanswerable 400 \
        --providers deepseek-flash,sabia-4,gemini-2.5-flash
    python scripts/estimate_pipeline_cost.py --id thesis-run-02 --nonanswerable 400 \
        --reasoning-tokens 1500 --providers all
    python scripts/estimate_pipeline_cost.py --synthetic 20:80000 --transcription-pass-rate 0.5 \
        --nonanswerable 400 --audio-hours 30 --providers all
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from rich.table import Table

from arandu.qa.cep.judge import GATE_CRITERIA, REMEMBER_CRITERIA
from arandu.qa.config import CEPConfig
from arandu.utils.console import console

# --- LLM-generation coefficients (from the code) ---------------------------
JUDGE_TRANSCRIPTION_CRITERIA = 2  # language_drift + hallucination_loop (gated)
GEN_CALLS_PER_PAIR = 1  # bloom_scaffolding: one generate() per pair
JUDGE_QA_CRITERIA = len(GATE_CRITERIA)
JUDGE_QA_CRITERIA_REMEMBER = len(REMEMBER_CRITERIA)
EMIC_CALLS_PER_PAIR = 1  # emic_validity ordinal criterion, scope 'all'
NONANSWERABLE_CALLS_PER_ITEM = 1  # perturbation: one generate_structured (happy path)
RETRIEVE_NER_CALLS_PER_Q = 1  # only atlas_rag runs an LLM (NER) at retrieve time
ANSWER_CALLS_PER_ARM_Q = 1  # answerer: one generate per (arm, question)
JUDGE_ANS_CALLS_ANSWERABLE = 4  # abstention + passage_coverage + correctness + faithfulness
JUDGE_ANS_CALLS_NONANSWERABLE = 1  # non-answerable fails answerability gate -> abstention only

# --- Corpus / KG / embedding coefficients ----------------------------------
CEP_CHUNK_SIZE = 4000  # chunk registry: cep_4k = RecursiveChunker(chunk_size=4000)
KG_PASSAGE_SIZE = 8192  # atlas-rag ATLAS_DEFAULTS["chunk_size"]
KG_EXTRACTION_PASSES = 3  # AutoSchemaKG: entity-relation, event-entity, event-relation
KG_CONCEPT_BATCH = 16  # ATLAS_DEFAULTS["batch_size_concept"]
# Calibrated to test-kg-04: ~14000 nodes, ~60000 edges over 422 passages.
KG_NODES_PER_PASSAGE = 33.0
KG_EDGES_PER_NODE = 4.3

DEFAULT_ARMS = 6  # bm25, atlas_rag, khop_passage, khop_triple, null, oracle
_LADDER = CEPConfig().bloom_distribution
DEFAULT_QUESTIONS = sum(_LADDER.values())  # 3:1:1:1 -> 6
REMEMBER_FRACTION = _LADDER.get("remember", 0) / DEFAULT_QUESTIONS
DEFAULT_JUDGE_PASS_RATE = 0.7
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
    "emic_judge": 900,
    "generate_non_answerable": 500,
    "retrieve_ner": 300,
    "answer": 600,
    "judge_answers": 800,
}

# Estimated visible output (completion) tokens per call, by stage, without reasoning.
OUTPUT_TOKENS: dict[str, float] = {
    "judge_transcription": 150,
    "build_kg_extraction": 1200,  # triple lists are verbose
    "build_kg_concept": 300,
    "cep_generation": 200,  # one QA pair
    "judge_qa": 250,  # verdict + rationale, per criterion
    "emic_judge": 250,
    "generate_non_answerable": 200,
    "retrieve_ner": 150,
    "answer": 300,
    "judge_answers": 250,
}

# Phase classification for the generation-vs-evaluation decision. judge_qa
# moves to GENERATION when the scaffolding gate is on (see phase_totals).
GENERATION_STAGES = frozenset(
    {"build_kg_extraction", "build_kg_concept", "cep_generation", "generate_non_answerable"}
)
EVALUATION_STAGES = frozenset(
    {
        "judge_transcription",
        "judge_qa",
        "emic_judge",
        "retrieve_ner",
        "answer",
        "judge_answers",
    }
)

# Maritaca bills in BRL. Conversion rate is a third-party mid-market quote
# (late July 2026); recompute before citing a USD figure.
BRL_PER_USD = 5.07

# Public list prices in USD per 1,000,000 tokens (input at cache miss, output),
# verified against the providers' pricing pages on 2026-10-19:
# developers.openai.com/api/docs/pricing, ai.google.dev/gemini-api/docs/pricing,
# api-docs.deepseek.com/quick_start/pricing, docs.maritaca.ai/pt/precos,
# alibabacloud.com/help/en/model-studio/model-pricing, docs.mistral.ai,
# platform.minimax.io/docs/guides/pricing, platform.kimi.ai/docs/pricing/chat,
# docs.x.ai/docs/models. Text tier, Standard inference, shortest-context tier.
# Prices drift -- re-verify before citing, or override with --pricing-file
# (JSON: {"name": [in, out]}).
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI: gpt-4o is legacy but still callable; GPT-5.x is the current family.
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    # Google Gemini (3-flash is a preview; 3.6-flash price holds through 2026-12-31).
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),  # <=200k tier
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),  # <=200k tier
    "gemini-3.6-flash": (0.75, 3.75),
    # DeepSeek: peak rate; off-peak (outside 01-04 and 06-10 UTC on weekdays) is half.
    "deepseek-flash": (0.30, 1.20),
    "deepseek-flash-offpeak": (0.15, 0.60),
    "deepseek-v4-pro": (1.32, 3.96),
    "deepseek-v4-pro-offpeak": (0.66, 1.98),
    # Maritaca (pt-BR native): BRL list price converted at BRL_PER_USD.
    "sabiazinho-4": (1.00 / BRL_PER_USD, 4.00 / BRL_PER_USD),
    "sabia-4": (5.00 / BRL_PER_USD, 20.00 / BRL_PER_USD),
    # Other OpenAI-compatible options.
    "qwen3.8-flash": (0.15, 0.47),
    "qwen3.7-plus": (0.40, 1.60),
    "mistral-small-2603": (0.15, 0.60),
    "minimax-m3": (0.30, 1.20),
    "kimi-k2.6": (0.95, 4.00),
    "grok-4.20-non-reasoning": (1.25, 2.50),
    "local-ollama": (0.0, 0.0),  # self-hosted: $0 in tokens, cost is GPU hours instead
}


@dataclass
class Config:
    """Estimation inputs."""

    cep_chunk_size: int
    questions_per_chunk: int
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
    reasoning_tokens: float = 0.0  # thinking tokens billed as output on every call
    emic: bool = True  # run emic-judge over every pair
    gate: bool = True  # scaffolding gate judges pairs inside generate-cep-qa


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
    # input = fixed prompt overhead + the variable context the stage actually reads.
    context_in: dict[str, float] = {
        "judge_transcription": mean_doc_tokens,
        "build_kg_extraction": kg_passage_tokens,
        "build_kg_concept": KG_CONCEPT_BATCH * 20,  # ~16 node names per batch
        "cep_generation": cep_chunk_tokens,
        "judge_qa": cep_chunk_tokens + 100,  # chunk context + the QA pair
        "emic_judge": cep_chunk_tokens + 100,  # chunk context + the QA pair
        "generate_non_answerable": 200,  # one QA pair
        "retrieve_ner": 50,  # just the question
        "answer": RETRIEVED_CONTEXT_TOKENS + 50,  # passages + question
        "judge_answers": RETRIEVED_CONTEXT_TOKENS + 350,  # passages + question + answer
    }
    return {
        stage: (PROMPT_TOKENS[stage] + ctx, OUTPUT_TOKENS[stage] + cfg.reasoning_tokens)
        for stage, ctx in context_in.items()
    }


def estimate(doc_chars: list[int], cfg: Config, n_judged: int | None = None) -> dict[str, object]:
    """Compute per-stage cost across the three unit groups, with token totals.

    Args:
        doc_chars: Character count of each transcription that reaches CEP and KG
            (the ones the transcription judge admits).
        cfg: Estimation inputs.
        n_judged: Transcriptions the judge evaluates (all of them, admitted or
            not). Defaults to ``len(doc_chars)``.

    Returns:
        Counts, per-stage LLM calls and token totals, embeddings and GPU hours.
    """
    n_docs = len(doc_chars)
    n_judged = n_docs if n_judged is None else n_judged
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

    avg_judge_qa = (
        cfg.remember_fraction * JUDGE_QA_CRITERIA_REMEMBER
        + (1 - cfg.remember_fraction) * JUDGE_QA_CRITERIA
    )

    llm: dict[str, float] = {
        "judge_transcription": n_judged * JUDGE_TRANSCRIPTION_CRITERIA,
        "build_kg_extraction": kg_passages * KG_EXTRACTION_PASSES,
        "build_kg_concept": kg_nodes / KG_CONCEPT_BATCH,
        "cep_generation": pairs * GEN_CALLS_PER_PAIR,
        "judge_qa": pairs * avg_judge_qa,
        "emic_judge": pairs * EMIC_CALLS_PER_PAIR if cfg.emic else 0.0,
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
        "n_judged": n_judged,
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


def phase_totals(stage_values: dict[str, float], *, gate: bool = True) -> dict[str, float]:
    """Split a per-stage map into generation, evaluation, and total subtotals.

    Args:
        stage_values: Per-stage values (calls, tokens or USD).
        gate: Whether the scaffolding gate judges pairs during generation; when
            true, judge_qa counts as generation.

    Returns:
        ``generation``, ``evaluation`` and ``total`` subtotals.
    """
    generation_stages = GENERATION_STAGES | {"judge_qa"} if gate else GENERATION_STAGES
    generation = sum(v for s, v in stage_values.items() if s in generation_stages)
    evaluation = sum(
        v for s, v in stage_values.items() if s in EVALUATION_STAGES and s not in generation_stages
    )
    return {"generation": generation, "evaluation": evaluation, "total": generation + evaluation}


def read_corpus(run_id: str, results_root: Path) -> tuple[list[int], int, float]:
    """Read a run's transcription outputs.

    Args:
        run_id: Pipeline run id.
        results_root: Root of the results tree.

    Returns:
        Char counts of the transcriptions the judge admitted (or did not judge),
        the number of transcriptions read, and the total audio hours.
    """
    out = results_root / run_id / "transcription" / "outputs"
    admitted: list[int] = []
    n_read = 0
    total_ms = 0
    for f in sorted(out.glob("*_transcription.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        n_read += 1
        total_ms += int(data.get("duration_milliseconds") or 0)
        validation = data.get("validation")
        # CEP and KG skip records the judge rejected; unjudged records go through.
        if validation is not None and not validation.get("passed", True):
            continue
        admitted.append(len(data.get("transcription_text") or ""))
    if not n_read:
        raise SystemExit(f"No transcription outputs found under {out}")
    return admitted, n_read, total_ms / 3_600_000


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


def _print_calls(r: dict[str, object], gate: bool) -> None:
    """Print the call-count and resource sections (the provider-agnostic view)."""
    llm: dict[str, float] = r["llm"]  # type: ignore[assignment]
    tokens_in: dict[str, float] = r["tokens_in"]  # type: ignore[assignment]
    tokens_out: dict[str, float] = r["tokens_out"]  # type: ignore[assignment]
    generation_stages = GENERATION_STAGES | {"judge_qa"} if gate else GENERATION_STAGES
    table = Table(title="(B) LLM calls / tokens")
    table.add_column("stage")
    table.add_column("phase")
    table.add_column("calls", justify="right")
    table.add_column("tok_in", justify="right")
    table.add_column("tok_out", justify="right")
    for stage, n in llm.items():
        phase = "generation" if stage in generation_stages else "evaluation"
        table.add_row(
            stage, phase, f"{n:,.0f}", f"{tokens_in[stage]:,.0f}", f"{tokens_out[stage]:,.0f}"
        )
    table.add_section()
    table.add_row(
        "TOTAL",
        "",
        f"{r['llm_total']:,.0f}",
        f"{r['tokens_in_total']:,.0f}",
        f"{r['tokens_out_total']:,.0f}",
    )
    console.print(table)
    console.print(f"(C) embedding calls: {r['embedding_calls']:,.0f}")
    console.print(f"(A) GPU transcription hours: {r['gpu_transcription_hours']:.1f}")


def _print_costs(r: dict[str, object], providers: list[str], pricing: dict, gate: bool) -> None:
    """Print the per-provider USD cost table with generation/evaluation split."""
    tokens_in: dict[str, float] = r["tokens_in"]  # type: ignore[assignment]
    tokens_out: dict[str, float] = r["tokens_out"]  # type: ignore[assignment]
    table = Table(title="USD cost by provider (rough; verify PRICING before citing)")
    for column in ("provider", "in $/1M", "out $/1M", "generation", "evaluation", "total"):
        table.add_column(column, justify="left" if column == "provider" else "right")
    for name in providers:
        price_in, price_out = pricing[name]
        per_stage = stage_cost_usd(tokens_in, tokens_out, price_in, price_out)
        ph = phase_totals(per_stage, gate=gate)
        table.add_row(
            name,
            f"{price_in:.3f}",
            f"{price_out:.3f}",
            f"${ph['generation']:,.2f}",
            f"${ph['evaluation']:,.2f}",
            f"${ph['total']:,.2f}",
        )
    console.print(table)


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
        "--transcription-pass-rate",
        type=float,
        default=1.0,
        help="Fraction of synthetic docs the transcription judge admits (--id reads verdicts).",
    )
    p.add_argument(
        "--cep-chunk-size",
        type=int,
        default=CEP_CHUNK_SIZE,
        help="CEP generation chunk size (chars).",
    )
    p.add_argument(
        "-Q",
        "--questions",
        type=int,
        default=DEFAULT_QUESTIONS,
        help="Bloom-ladder size per chunk (default: CEPConfig's bloom_distribution).",
    )
    p.add_argument(
        "--cep-chunks",
        type=int,
        default=None,
        help="Override modeled CEP chunk count with the real value (cep_pairs = cep_chunks*Q).",
    )
    p.add_argument(
        "--judge-pass-rate",
        type=float,
        default=DEFAULT_JUDGE_PASS_RATE,
        help="Fraction of pairs surviving judge-qa.",
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
        "--reasoning-tokens",
        type=float,
        default=0.0,
        help="Thinking tokens billed as output per LLM call (0 = thinking off).",
    )
    p.add_argument("--no-emic", action="store_true", help="Leave emic-judge out of the run.")
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="Scaffolding gate off: judge-qa runs as a separate evaluation stage.",
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
        all_chars = parse_synthetic(args.synthetic)
        n_judged = len(all_chars)
        doc_chars = all_chars[: round(n_judged * args.transcription_pass_rate)]
    else:
        doc_chars, n_judged, measured_hours = read_corpus(args.id, args.results_root)
        if audio_hours == 0.0:
            audio_hours = measured_hours

    cfg = Config(
        cep_chunk_size=args.cep_chunk_size,
        questions_per_chunk=args.questions,
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
        reasoning_tokens=args.reasoning_tokens,
        emic=not args.no_emic,
        gate=not args.no_gate,
    )
    r = estimate(doc_chars, cfg, n_judged=n_judged)

    console.print(
        f"corpus: {n_judged} docs judged, {r['n_docs']} admitted "
        f"({sum(doc_chars):,} chars), {audio_hours:.1f} audio-hours"
    )
    console.print(
        f"config: cep_chunk={cfg.cep_chunk_size}, Q={cfg.questions_per_chunk}, "
        f"judge_pass={cfg.judge_pass_rate}, remember={cfg.remember_fraction:.2f}, "
        f"non_answerable={cfg.nonanswerable}, arms={cfg.n_arms}, "
        f"kg_passage={cfg.kg_passage_size}, reasoning_tokens={cfg.reasoning_tokens:.0f}, "
        f"gate={cfg.gate}, emic={cfg.emic}"
    )
    console.print(
        f"cep_chunks={r['cep_chunks']:,}  kg_passages={r['kg_passages']:,}  "
        f"kg_nodes~{r['kg_nodes']:,.0f}  kg_edges~{r['kg_edges']:,.0f}  "
        f"cep_pairs={r['cep_pairs']:,.0f}  answerable~{r['answerable']:,.0f}  "
        f"questions~{r['n_questions']:,.0f}"
    )
    _print_calls(r, cfg.gate)
    if args.providers:
        pricing = load_pricing(args.pricing_file)
        providers = resolve_providers(args.providers, pricing)
        _print_costs(r, providers, pricing, cfg.gate)


if __name__ == "__main__":
    main()
