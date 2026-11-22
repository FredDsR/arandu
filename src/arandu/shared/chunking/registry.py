"""Default chunker registry — maps known ``chunker_id`` values to configured adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.chunking.chonkie_adapter import ChonkieAdapter

if TYPE_CHECKING:
    from arandu.shared.chunking.protocol import Chunker

KNOWN_CHUNKER_IDS: tuple[str, ...] = (
    "cep_4k",
    "bm25_512t",
    "bm25_1024t",
    "bm25_4k",
    "nx_2k",
)


def get_chunker(chunker_id: str, *, tokenizer: str | None = None) -> Chunker:
    """Resolve ``chunker_id`` to a configured ``ChonkieAdapter``.

    Char-mode views (``cep_4k``, ``bm25_4k``, ``nx_2k``) instantiate the chonkie
    ``RecursiveChunker`` with the matching ``chunk_size``.

    Token-mode views (``bm25_512t``, ``bm25_1024t``) instantiate ``TokenChunker``
    with the requested ``tokenizer`` (defaults to chonkie's char-based fallback
    when omitted; the BM25 baseline task overrides this with a real tokenizer
    such as ``qwen2``).

    Args:
        chunker_id: One of :data:`KNOWN_CHUNKER_IDS`.
        tokenizer: Optional tokenizer name passed to chonkie for token-mode chunkers.

    Raises:
        ValueError: If ``chunker_id`` is not recognized.
    """
    from chonkie import RecursiveChunker, TokenChunker

    match chunker_id:
        case "cep_4k":
            return ChonkieAdapter(chunker_id, RecursiveChunker(chunk_size=4000))
        case "bm25_4k":
            return ChonkieAdapter(chunker_id, RecursiveChunker(chunk_size=4000))
        case "nx_2k":
            return ChonkieAdapter(chunker_id, RecursiveChunker(chunk_size=2000))
        case "bm25_512t":
            kwargs: dict[str, object] = {"chunk_size": 512, "chunk_overlap": 64}
            if tokenizer is not None:
                kwargs["tokenizer"] = tokenizer
            return ChonkieAdapter(chunker_id, TokenChunker(**kwargs), tokenizer_id=tokenizer)
        case "bm25_1024t":
            kwargs_1024: dict[str, object] = {"chunk_size": 1024, "chunk_overlap": 128}
            if tokenizer is not None:
                kwargs_1024["tokenizer"] = tokenizer
            return ChonkieAdapter(chunker_id, TokenChunker(**kwargs_1024), tokenizer_id=tokenizer)
        case _:
            raise ValueError(f"Unknown chunker_id: {chunker_id!r}. Known: {KNOWN_CHUNKER_IDS}")
