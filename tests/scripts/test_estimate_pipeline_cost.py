"""Tests for the estimate_pipeline_cost script's load-bearing logic.

Scoped deliberately small: the script is an order-of-magnitude estimator with
rough token coefficients, so only the logic that would change the conclusion if
it broke silently is guarded here -- the real-chunk override and the
generation-vs-evaluation split (and that evaluation dominates at 5 arms).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.qa.cep.judge import GATE_CRITERIA, REMEMBER_CRITERIA
from scripts.estimate_pipeline_cost import (
    Config,
    estimate,
    phase_totals,
    read_corpus,
    stage_cost_usd,
)

if TYPE_CHECKING:
    from pathlib import Path


def _cfg(**overrides: object) -> Config:
    """Build a Config with sensible defaults, overriding named fields."""
    base = {
        "cep_chunk_size": 8000,
        "questions_per_chunk": 6,
        "judge_pass_rate": 0.54,
        "remember_fraction": 0.5,
        "nonanswerable": 0,
        "n_arms": 5,
        "kg_passage_size": 8192,
        "kg_nodes_per_passage": 33.0,
        "kg_edges_per_node": 4.3,
        "audio_hours": 0.0,
        "whisper_rtf": 0.30,
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


def test_cep_chunks_override_drives_pairs() -> None:
    """An explicit cep_chunks overrides the char-based model; pairs = chunks * Q.

    This is what makes the estimate "real" (fed the run's true 445 chunks) rather
    than a ceil-division guess, so it must not silently revert to the model.
    """
    r = estimate([50_000, 50_000], _cfg(questions_per_chunk=6, cep_chunks=445))
    assert r["cep_chunks"] == 445
    assert r["cep_pairs"] == 445 * 6


def test_phase_totals_split_and_sum() -> None:
    """Generation + evaluation subtotals partition the per-stage cost exactly."""
    r = estimate([50_000, 50_000], _cfg(nonanswerable=100))
    per_stage = stage_cost_usd(r["tokens_in"], r["tokens_out"], 1.0, 1.0)
    ph = phase_totals(per_stage)
    assert ph["generation"] > 0
    assert ph["evaluation"] > 0
    assert ph["total"] == pytest.approx(ph["generation"] + ph["evaluation"])


def test_evaluation_dominates_with_many_arms() -> None:
    """The arms multiplier makes evaluation outweigh generation at 5 arms.

    This is the headline conclusion the cost analysis rests on; guard it.
    """
    r = estimate([50_000, 50_000, 50_000], _cfg(n_arms=5, nonanswerable=200))
    per_stage = stage_cost_usd(r["tokens_in"], r["tokens_out"], 1.0, 1.0)
    ph = phase_totals(per_stage)
    assert ph["evaluation"] > ph["generation"]


def test_judge_qa_calls_follow_the_gate_criteria() -> None:
    """judge-qa calls use the code's criteria counts, not a hard-coded copy."""
    r = estimate([4000], _cfg(cep_chunk_size=4000, questions_per_chunk=2, remember_fraction=0.5))
    expected = 2 * (0.5 * len(REMEMBER_CRITERIA) + 0.5 * len(GATE_CRITERIA))
    assert r["llm"]["judge_qa"] == pytest.approx(expected)


def test_gate_books_judge_qa_to_generation() -> None:
    """With the scaffolding gate, pairs are judged inside generate-cep-qa."""
    values = {"judge_qa": 10.0, "cep_generation": 1.0, "answer": 5.0}
    gated = phase_totals(values, gate=True)
    ungated = phase_totals(values, gate=False)
    assert gated == {"generation": 11.0, "evaluation": 5.0, "total": 16.0}
    assert ungated == {"generation": 1.0, "evaluation": 15.0, "total": 16.0}


def test_reasoning_tokens_add_output_to_every_call() -> None:
    """Thinking tokens are billed as output on each LLM call."""
    base = estimate([50_000], _cfg())
    thinking = estimate([50_000], _cfg(reasoning_tokens=1000))
    extra = thinking["tokens_out_total"] - base["tokens_out_total"]
    assert extra == pytest.approx(1000 * base["llm_total"])


def test_emic_judge_one_call_per_pair_and_optional() -> None:
    r = estimate([50_000], _cfg(questions_per_chunk=6))
    assert r["llm"]["emic_judge"] == r["cep_pairs"]
    assert estimate([50_000], _cfg(emic=False))["llm"]["emic_judge"] == 0


def test_read_corpus_drops_rejected_transcriptions(tmp_path: Path) -> None:
    """CEP and KG skip what the transcription judge rejected; the judge still sees all."""
    out = tmp_path / "run" / "transcription" / "outputs"
    out.mkdir(parents=True)
    records = {
        "a": {"transcription_text": "x" * 10, "validation": {"passed": True}},
        "b": {"transcription_text": "y" * 20, "validation": {"passed": False}},
        "c": {"transcription_text": "z" * 30, "duration_milliseconds": 3_600_000},
    }
    for name, data in records.items():
        (out / f"{name}_transcription.json").write_text(json.dumps(data))

    chars, n_read, hours = read_corpus("run", tmp_path)

    assert sorted(chars) == [10, 30]
    assert n_read == 3
    assert hours == pytest.approx(1.0)
