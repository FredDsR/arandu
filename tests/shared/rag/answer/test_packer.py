"""Tests for ``pack_passages`` — token-budget passage packing."""

from __future__ import annotations

import pytest

from arandu.shared.rag.answer.packer import estimate_tokens, pack_passages
from arandu.shared.rag.schemas import RetrievedPassage


def _passage(chunk_id: str, rank: int = 0, payload: str | None = None) -> RetrievedPassage:
    return RetrievedPassage(chunk_id=chunk_id, rank=rank, score=1.0, payload=payload)


class TestEstimateTokens:
    def test_empty_string_zero_tokens(self) -> None:
        assert estimate_tokens("") == 0

    def test_non_empty_at_least_one(self) -> None:
        # Floor-division could yield 0 for tiny strings; the helper
        # guards against that so a long list of one-char passages
        # doesn't get a free pass through the budget.
        assert estimate_tokens("a") == 1

    def test_chars_per_token_heuristic(self) -> None:
        # 30 chars / 3 chars-per-token = 10 tokens.
        text = "x" * 30
        assert estimate_tokens(text) == 10


class TestPackPassages:
    def test_packs_within_budget(self) -> None:
        passages = [_passage("c1"), _passage("c2"), _passage("c3")]
        passage_text = {"c1": "x" * 30, "c2": "x" * 30, "c3": "x" * 30}
        # budget = 100 - 30 - 30 = 40 → fits one passage (10 tokens),
        # the second would push to 20 ≤ 40 so it fits too, the third
        # pushes to 30 ≤ 40 so all three fit.
        result = pack_passages(
            passages,
            passage_text=passage_text,
            max_context_tokens=100,
            prompt_overhead_tokens=30,
            max_answer_tokens=30,
        )
        assert [p.chunk_id for p, _ in result] == ["c1", "c2", "c3"]

    def test_stops_when_budget_exhausted(self) -> None:
        # Each passage = 30 chars = 10 tokens. Budget = 15. First fits
        # (used=10); second would push to 20 > 15 → break (NOT truncated).
        passages = [_passage("c1"), _passage("c2"), _passage("c3")]
        passage_text = {"c1": "x" * 30, "c2": "x" * 30, "c3": "x" * 30}
        result = pack_passages(
            passages,
            passage_text=passage_text,
            max_context_tokens=100,
            prompt_overhead_tokens=80,
            max_answer_tokens=5,
        )
        assert [p.chunk_id for p, _ in result] == ["c1"]

    def test_payload_overrides_chunk_id_lookup(self) -> None:
        # khop_triple-style passage: payload set, chunk_id is opaque.
        # The map doesn't carry the chunk_id; packer falls back to payload.
        p = _passage("triple:abc", payload="Maria --[vive_em]--> Itaqui")
        result = pack_passages(
            [p],
            passage_text={},  # no chunk_id resolution available
            max_context_tokens=100,
            prompt_overhead_tokens=10,
            max_answer_tokens=10,
        )
        assert len(result) == 1
        assert result[0][1] == "Maria --[vive_em]--> Itaqui"

    def test_unresolvable_chunk_id_silently_skipped(self) -> None:
        # Passage points at a chunk_id the resolver doesn't know about
        # (corpus drift, missing sidecar, etc.). Packer drops it without
        # raising — downstream count just reflects fewer packed passages.
        passages = [_passage("missing"), _passage("present")]
        passage_text = {"present": "x" * 30}
        result = pack_passages(
            passages,
            passage_text=passage_text,
            max_context_tokens=100,
            prompt_overhead_tokens=10,
            max_answer_tokens=10,
        )
        assert [p.chunk_id for p, _ in result] == ["present"]

    def test_non_positive_budget_raises(self) -> None:
        # Catches operator misconfig (e.g. prompt_overhead so large that
        # nothing fits) at the entry point with a clear error.
        with pytest.raises(ValueError, match="Token budget exhausted before packing"):
            pack_passages(
                [_passage("c1")],
                passage_text={"c1": "x" * 30},
                max_context_tokens=50,
                prompt_overhead_tokens=40,
                max_answer_tokens=20,
            )

    def test_empty_passages_returns_empty(self) -> None:
        # The "no passages retrieved" path (Null arm, empty entity-link)
        # must produce a clean empty list — the answerer's prompt
        # template handles that branch separately.
        result = pack_passages(
            [],
            passage_text={},
            max_context_tokens=100,
            prompt_overhead_tokens=10,
            max_answer_tokens=10,
        )
        assert result == []
