"""Token-budget passage packing for the answerer (spec §5.5).

Greedy by rank: include passages in their ranked order until the
remaining budget can't fit the next one. The first passage that exceeds
the budget is dropped (along with all lower-ranked passages) — we do
NOT truncate a passage mid-sentence, since partial passages tend to
confuse the answerer's grounding more than missing ones.

Token cost is estimated via a simple character-count heuristic
(:func:`estimate_tokens`). For thesis purposes this is accurate enough:
budget overruns of ~10% just mean the answerer's context window is
slightly under-utilized, not that it explodes. Swapping in a real
tokenizer (tiktoken / transformers AutoTokenizer) would be a follow-up
if the heuristic proves too loose in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import RetrievedPassage


# Portuguese text averages ~3 characters per token under GPT-style BPE
# tokenizers (vs ~4 for English). Use 3 as the conservative (pessimistic)
# estimate so the budget calculation overshoots on space rather than
# undershooting and blowing the context window.
_CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """Estimate token count via character-count heuristic.

    Returns at least 1 for any non-empty text so a stream of tiny
    passages doesn't compound floor-division errors into a free pass.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def pack_passages(
    passages: list[RetrievedPassage],
    *,
    passage_text: dict[str, str],
    max_context_tokens: int,
    prompt_overhead_tokens: int,
    max_answer_tokens: int,
) -> list[tuple[RetrievedPassage, str]]:
    """Pack passages by rank until the token budget is exhausted.

    Args:
        passages: Ranked retrieval results. Iterated in given order
            (caller should pre-sort by ``rank`` ascending if it isn't
            already).
        passage_text: ``chunk_id`` → resolved text. The caller resolves
            this externally (via :class:`ChunkResolver`, the
            ``passage_offsets`` sidecar, or the ``payload`` field) so
            the packer stays simple and unit-testable. A passage whose
            ``chunk_id`` has no entry in this map is silently skipped.
        max_context_tokens: Total context window of the answerer LLM.
        prompt_overhead_tokens: Reserved budget for the rendered prompt
            (everything except passage content).
        max_answer_tokens: Reserved budget for the answerer's response.

    Returns:
        A list of ``(passage, text)`` tuples, prefix of the input
        ranking, that fits within the budget. Always preserves rank
        order.

    Raises:
        ValueError: If the budget is non-positive (caller misconfiguration).
    """
    budget = max_context_tokens - prompt_overhead_tokens - max_answer_tokens
    if budget <= 0:
        raise ValueError(
            f"Token budget exhausted before packing: max_context_tokens="
            f"{max_context_tokens} - prompt_overhead_tokens={prompt_overhead_tokens} "
            f"- max_answer_tokens={max_answer_tokens} = {budget}. "
            f"Reduce prompt_overhead_tokens or max_answer_tokens, or raise max_context_tokens."
        )

    out: list[tuple[RetrievedPassage, str]] = []
    used = 0
    for p in passages:
        text = _resolve_passage_text(p, passage_text)
        if text is None:
            continue
        cost = estimate_tokens(text)
        if used + cost > budget:
            break
        out.append((p, text))
        used += cost
    return out


def _resolve_passage_text(passage: RetrievedPassage, passage_text: dict[str, str]) -> str | None:
    """Look up the text for ``passage`` — prefer payload, fall back to chunk_id map.

    Triple-emitting retrievers (e.g. :class:`KHopTripleRetriever`)
    populate ``RetrievedPassage.payload`` because their output isn't a
    sliceable chunk; we honour that override here so the packer
    doesn't need to know which retriever produced what.
    """
    if passage.payload is not None:
        return passage.payload
    return passage_text.get(passage.chunk_id)
