"""Tests for the estimate_pipeline_cost script's load-bearing logic.

Scoped deliberately small: the script is an order-of-magnitude estimator with
rough token coefficients, so only the logic that would change the conclusion if
it broke silently is guarded here -- the real-chunk override and the
generation-vs-evaluation split (and that evaluation dominates at 5 arms).
"""

from __future__ import annotations

import pytest

from scripts.estimate_pipeline_cost import (
    Config,
    estimate,
    phase_totals,
    stage_cost_usd,
)


def _cfg(**overrides: object) -> Config:
    """Build a Config with sensible defaults, overriding named fields."""
    base = {
        "cep_chunk_size": 8000,
        "questions_per_chunk": 6,
        "reasoning_enabled": False,
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
