"""Per-arm settings for ``arandu retrieve``.

Each retriever arm exposes a distinct set of knobs (chunker view, k-hop
radius, postings cap, LLM/embedder for atlas-rag's NER + edge filter,
…). Rather than overload one CLI command with a union of all knobs,
each arm has its own :class:`BaseSettings` subclass (env prefix
``ARANDU_<ARM>_``). The CLI instantiates only the settings classes for
arms it was asked to run.

Atlas-rag is the only arm that hits the network at retrieve time (NER
step in front of PPR). It needs an :class:`LLMClient` and a sentence
encoder; configuration lives in :class:`AtlasRagRetrieveSettings` plus
the existing :class:`arandu.shared.embeddings.EmbedderSettings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from arandu.shared.llm_settings import validate_atlas_extra_body

if TYPE_CHECKING:
    from typing import Self


_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

ArmName = Literal["bm25", "atlas_rag", "khop_passage", "khop_triple", "null", "oracle", "dense"]
# Arm names include the named dense configurations (DENSE_PRESETS), which a
# Literal cannot list; KNOWN_ARMS is the runtime catalog.

# Stable ordering — drives the CLI's default arm set + the iteration
# order in the batch runner. Keeps benchmark output reproducible.
#
# ``atlas_rag`` is recognized but not yet wired in this PR: the factory
# routes it to a clear "deferred to a follow-up PR" error. Including it
# in the catalog so ``--arm atlas_rag`` produces the helpful error
# message rather than an unknown-arm rejection at the batch-runner
# validation layer. The CLI's ``_DEFAULT_ARMS`` filter excludes it.
ALL_ARMS: tuple[ArmName, ...] = (
    "bm25",
    "atlas_rag",
    "khop_passage",
    "khop_triple",
    "null",
    "oracle",
    "dense",
)


MPNET_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BGE_M3_MODEL = "BAAI/bge-m3"
E5_INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# One task instruction for the instruction-tuned encoders; both model cards
# recommend a one-sentence task description in English.
DENSE_TASK = (
    "Given a question about an oral history interview, "
    "retrieve the transcript passages that answer the question"
)
# multilingual-e5-large-instruct: "Instruct: {task}\nQuery: {query}" on the
# query side only; passages go in without a prefix.
E5_QUERY_PROMPT = f"Instruct: {DENSE_TASK}\nQuery: "
# Qwen3-Embedding: "Instruct: {task}\nQuery:{query}" on the query side only
# (the model card's format has no space after the colon).
QWEN3_QUERY_PROMPT = f"Instruct: {DENSE_TASK}\nQuery:"


def _preset(
    model: str, window: int | None, query_prompt: str = "", *, whole: bool = False
) -> dict[str, Any]:
    return {
        "model": model,
        "window_tokens": window,
        "whole_chunk": whole,
        "query_prompt": query_prompt,
        "passage_prompt": "",
    }


# Named dense configurations, each its own arm (output directory), so two of
# them can coexist in one run. The suffix is the encoder input per window in
# tokens, or ``whole`` for one window per chunk.
DENSE_PRESETS: dict[str, dict[str, Any]] = {
    "dense_mpnet128": _preset(MPNET_MODEL, 128),
    "dense_bgem3_128": _preset(BGE_M3_MODEL, 128),
    "dense_bgem3_256": _preset(BGE_M3_MODEL, 256),
    "dense_bgem3_512": _preset(BGE_M3_MODEL, 512),
    "dense_bgem3_whole": _preset(BGE_M3_MODEL, None, whole=True),
    "dense_e5_128": _preset(E5_INSTRUCT_MODEL, 128, E5_QUERY_PROMPT),
    "dense_e5_256": _preset(E5_INSTRUCT_MODEL, 256, E5_QUERY_PROMPT),
    "dense_e5_512": _preset(E5_INSTRUCT_MODEL, 512, E5_QUERY_PROMPT),
    "dense_qwen3_128": _preset(QWEN3_EMBEDDING_MODEL, 128, QWEN3_QUERY_PROMPT),
    "dense_qwen3_256": _preset(QWEN3_EMBEDDING_MODEL, 256, QWEN3_QUERY_PROMPT),
    "dense_qwen3_512": _preset(QWEN3_EMBEDDING_MODEL, 512, QWEN3_QUERY_PROMPT),
    "dense_qwen3_whole": _preset(QWEN3_EMBEDDING_MODEL, None, QWEN3_QUERY_PROMPT, whole=True),
}

# Every arm name the retrieve stage accepts. The dense presets are opt-in
# (``--arm dense_bgem3_512``); they stay out of the default set.
KNOWN_ARMS: tuple[str, ...] = (*ALL_ARMS, *DENSE_PRESETS)


class Bm25RetrieveSettings(BaseSettings):
    """Knobs for the BM25 arm.

    Attributes:
        chunker_id: ChunkSet view to query against. Must match a chunker
            id present in ``results/<id>/chunk/outputs/<chunker_id>/``
            (the chunking stage emits one directory per view it ran).
            Defaults to ``"cep_4k"`` — the project's primary view, used
            by the CEP QA generator.

    """

    chunker_id: str = Field(
        default="cep_4k",
        description="Chunker view to retrieve over (must exist under chunk/outputs/).",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_BM25_", extra="ignore")


class DenseRetrieveSettings(BaseSettings):
    """Knobs for the dense (embedding) arms.

    The plain ``dense`` arm reads these from ``ARANDU_DENSE_*``. Its default
    encoder is the one atlas-rag uses, so it differs from atlas-rag only by
    the graph. The named arms of :data:`DENSE_PRESETS` override the encoder,
    window and prompts and keep the rest (chunker view, overlap, unit,
    device, batch size).

    Attributes:
        chunker_id: ChunkSet view to index, as for BM25 (``"cep_4k"``).
        model: ``sentence_transformers`` model name.
        window_tokens: Encoder input per sub-passage, special tokens and
            passage prompt included (``128`` is a full mpnet input).
            ``None`` (default) uses the encoder's ``max_seq_length``.
        whole_chunk: Encode each chunk whole instead of in windows (for
            long-input encoders such as bge-m3 and Qwen3-Embedding).
        overlap_ratio: Share of a window repeated by the next one.
        query_prompt: Text prepended to each question (instruction-tuned
            encoders such as e5-instruct and Qwen3-Embedding).
        passage_prompt: Text prepended to each sub-passage.
        unit: ``"parent"`` returns the chunks (same ``chunk_id`` as BM25),
            ``"subpassage"`` returns the windows inline.
        device: Torch device; ``None`` lets sentence-transformers pick. A
            CUDA out-of-memory error moves the encoder to the CPU.
        batch_size: Texts per forward pass.
    """

    chunker_id: str = Field(
        default="cep_4k",
        description="Chunker view to index (must exist under chunk/outputs/).",
    )
    model: str = Field(default=MPNET_MODEL, description="sentence-transformers model name.")
    window_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Encoder input tokens per sub-passage; unset = encoder max_seq_length.",
    )
    whole_chunk: bool = Field(default=False, description="Encode each chunk whole.")
    overlap_ratio: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="Share of each window repeated by the next window.",
    )
    query_prompt: str = Field(default="", description="Text prepended to each question.")
    passage_prompt: str = Field(default="", description="Text prepended to each sub-passage.")
    unit: Literal["parent", "subpassage"] = Field(
        default="parent",
        description="Returned unit: parent chunks (max over sub-passages) or sub-passages.",
    )
    device: str | None = Field(
        default=None,
        description="Torch device for the encoder (cpu, cuda); unset = auto.",
    )
    batch_size: int = Field(default=16, ge=1, description="Encoder batch size.")

    model_config = SettingsConfigDict(env_prefix="ARANDU_DENSE_", extra="ignore")

    def for_arm(self, arm: str) -> DenseRetrieveSettings:
        """Return the settings of dense arm ``arm``: a preset applied over these.

        Raises:
            ValueError: If ``arm`` is not a dense arm.
        """
        if arm == "dense":
            return self
        if arm not in DENSE_PRESETS:
            raise ValueError(f"{arm!r} is not a dense arm.")
        return self.model_copy(update=DENSE_PRESETS[arm])


def is_dense_arm(arm: str) -> bool:
    """Whether ``arm`` is the plain ``dense`` arm or one of :data:`DENSE_PRESETS`."""
    return arm == "dense" or arm in DENSE_PRESETS


class AtlasRagRetrieveSettings(BaseSettings):
    """Knobs for the atlas-rag (HippoRAG-style) arm.

    atlas-rag uses an LLM at retrieve time for the NER step that
    identifies entities in the query before personalized PageRank
    seeding. The provider + model + base_url are read from
    ``ARANDU_ATLAS_RAG_*`` env vars; the sentence encoder is read
    separately from ``ARANDU_EMBEDDER_*`` (the same vars the
    ``arandu kg-build-retriever-index`` command uses, so the encoder
    that built the precompute matches the one used to embed queries).

    Attributes:
        provider: LLM provider for the NER step. ``"openai"``,
            ``"ollama"``, or ``"custom"``. Defaults to ``"openai"`` —
            the Gemini-compatible endpoint is the project's cloud path
            (set ``base_url`` to the Gemini URL when using it).
        model_id: Model identifier passed to the LLMClient (e.g.
            ``"gemini-2.5-flash"`` on Gemini, ``"qwen3:14b"`` on Ollama).
        api_key: API key of the NER/filter client
            (``ARANDU_ATLAS_RAG_API_KEY``); required unless the provider is
            ollama.
        base_url: Optional explicit base URL. Required for ``custom``
            provider; ignored for ``openai``/``ollama`` defaults.
        extra_body: Extra request fields for the NER/filter LLM, as JSON.
            atlas-rag forwards only ``reasoning_effort`` (e.g. ``"none"`` to
            switch off thinking on DeepSeek); other keys are rejected.
        keyword: atlas-rag's filename pattern. Defaults to project
            convention ``"transcriptions.json"`` (must match the value
            used at index-build time).
        include_events: Whether event nodes participate in the embedded
            node set. Must match the value used at index-build time —
            mismatches are caught by atlas-rag's manifest validator.
        include_concept: Whether concept nodes participate. Same
            mismatch-detection contract as ``include_events``.
    """

    provider: str = Field(
        default="openai",
        description="LLM provider for the NER step: openai, ollama, custom.",
    )
    model_id: str = Field(
        default="gemini-2.5-flash",
        description="Model identifier (e.g. gemini-2.5-flash, qwen3:14b).",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key of the NER/filter client; required unless provider is ollama.",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for OpenAI-compatible endpoints. When unset, defaults "
            "are applied per-provider in the post-init validator: openai "
            "defaults to Gemini's compatibility URL (the project's primary "
            "cloud path); ollama leaves it None so LLMClient picks its own "
            "localhost default; custom requires an explicit value."
        ),
    )
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Extra request fields; atlas-rag forwards only reasoning_effort.",
    )
    keyword: str = Field(
        default="transcriptions.json",
        description="atlas-rag filename pattern (must match index build).",
    )
    include_events: bool = Field(
        default=True,
        description="Include event nodes in the embedded set.",
    )
    include_concept: bool = Field(
        default=True,
        description="Include concept nodes in the embedded set.",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_ATLAS_RAG_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Normalize provider to lowercase so env-var case doesn't break dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("extra_body")
    @classmethod
    def _validate_extra_body(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject extra_body keys atlas-rag cannot forward."""
        return validate_atlas_extra_body(v, consumer="ARANDU_ATLAS_RAG_")

    @model_validator(mode="after")
    def _default_base_url_per_provider(self) -> Self:
        """Apply per-provider base_url defaults when the user hasn't set one.

        Without this, the field-level default of ``None`` would leave Gemini
        users with no base URL (LLMClient's PROVIDER_URLS[openai] is also
        None — OpenAI proper). Setting the project's primary cloud path
        here keeps the "just works" UX without baking the URL into a default
        that would override the ollama path's localhost endpoint.
        """
        if self.base_url is not None:
            return self
        if self.provider == "openai":
            object.__setattr__(self, "base_url", _GEMINI_OPENAI_BASE_URL)
        # ollama: leave None — LLMClient will use http://localhost:11434/v1.
        # custom: leave None — the factory will surface a clear error if
        # LLMClient can't resolve a URL.
        return self


class KHopRetrieveSettings(BaseSettings):
    """Knobs shared by the two k-hop arms (passage + triple).

    Both arms operate on the same atlas-rag KG and share the same
    entity-link + k-hop ego graph machinery. They differ only in their
    output unit (passages vs linearized triples), so the configuration
    surface is identical.

    Attributes:
        k_hop: Ego-graph radius around entity-linked seeds. Higher →
            more recall, slower, more dilution. The 2026-05-23 calibration
            against ``test-kg-04`` runs at ``k_hop=2`` for seconds-per-query
            wall time.
        top_k_seeds: Max entity-link seeds kept per question, ranked by
            summed smoothed-IDF weight. Bounds ego-graph size; replaces
            the previous per-token hard cap on inverted-index expansion.
        keyword: atlas-rag's filename pattern for the graphml. Defaults
            to project convention ``"transcriptions.json"`` (see
            :mod:`arandu.kg.atlas_backend`).

    """

    k_hop: int = Field(
        default=2,
        ge=1,
        description="Ego-graph radius around entity-linked seeds.",
    )
    top_k_seeds: int = Field(
        default=50,
        ge=1,
        description=(
            "Max entity-link seeds kept per question, ranked by summed "
            "smoothed-IDF weight. Bounds ego-graph size (replaces the old "
            "previous per-token posting hard cap)."
        ),
    )
    keyword: str = Field(
        default="transcriptions.json",
        description="atlas-rag filename pattern (rarely changed).",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_KHOP_", extra="ignore")
