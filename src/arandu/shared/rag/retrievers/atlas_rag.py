"""atlas-rag (HippoRAG-style) retriever — Phase C spec §4.4.

Wraps ``atlas_rag.retriever.HippoRAGRetriever`` behind our :class:`Retriever`
Protocol. All atlas-rag imports are deferred to method bodies, matching the
convention enforced in :mod:`arandu.kg.atlas_backend`; this module is the
second (and only other) place outside that one that touches the atlas-rag
SDK directly.

Phase C decision (PR #100 follow-up): the precomputed retrieval artifacts
co-locate with the KG under ``results/<id>/kg/outputs/atlas_output/precompute/``
rather than under ``retrieval_indexes/`` (the BM25 home). atlas-rag's
index is tightly coupled to *this specific* KG — embeddings index THIS
graph's nodes and edges — so splitting the two would force duplication or
fragile symlinks.

The retriever exposes the same surface as :class:`BM25Retriever`:

- :meth:`build_index` (classmethod) runs atlas-rag's
  ``create_embeddings_and_index`` against an existing KG dir and writes a
  manifest documenting the inputs.
- ``__init__`` loads precomputed artifacts and assembles the internal
  ``HippoRAGRetriever``.
- :meth:`retrieve` runs the score-capturing PPR pipeline and adapts the
  result to a list of :class:`RetrievedPassage`. Returned ``chunk_id`` is
  atlas-rag's own ``passage_id``; ``arandu kg-link-passages`` (PR #100)
  is the bridge that maps those back to source character offsets when
  downstream judges need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


PRECOMPUTE_DIR_NAME = "precompute"
MANIFEST_FILENAME = "manifest.json"
GRAPHML_SUBDIR = "kg_graphml"


class AtlasRagRetriever:
    """HippoRAG-style retriever over an atlas-rag-built KG.

    Attributes:
        retriever_id: Stable identifier; defaults to ``atlas_rag_hipporag``.
    """

    RETRIEVER_FAMILY = "atlas_rag"
    DEFAULT_RETRIEVER_ID = "atlas_rag_hipporag"

    retriever_id: str

    def __init__(
        self,
        kg_outputs_dir: Path,
        llm_client: Any,
        sentence_encoder: Any,
        sentence_encoder_model: str,
        keyword: str = "transcriptions.json",
        include_events: bool = False,
        include_concept: bool = True,
        retriever_id: str | None = None,
    ) -> None:
        """Construct the retriever from a built atlas-rag index.

        Args:
            kg_outputs_dir: The atlas-rag output dir
                (e.g. ``results/<id>/kg/outputs/atlas_output/``). Must
                contain the graphml under ``kg_graphml/`` and a populated
                ``precompute/``.
            llm_client: An ``openai.OpenAI``-shaped client used to construct
                atlas-rag's ``LLMGenerator``. The unified ``LLMClient``'s
                ``.client`` is compatible (see
                :mod:`arandu.kg.atlas_backend`).
            sentence_encoder: An atlas-rag ``BaseEmbeddingModel`` instance.
            sentence_encoder_model: Model identifier; must match the model
                used at :meth:`build_index` time.
            keyword: atlas-rag's filename pattern; defaults to project
                convention ``"transcriptions.json"``.
            include_events / include_concept: Mirror of the index-build
                flags; controls which node types are referenced.
            retriever_id: Optional override of
                :attr:`DEFAULT_RETRIEVER_ID`.

        Raises:
            FileNotFoundError: If ``kg_outputs_dir``, the precompute dir,
                the manifest, or any precomputed artifact is missing.
            ValueError: If the manifest disagrees with constructor args.
        """
        if not kg_outputs_dir.exists():
            raise FileNotFoundError(f"atlas-rag kg outputs dir not found: {kg_outputs_dir}")
        precompute_dir = kg_outputs_dir / PRECOMPUTE_DIR_NAME
        if not precompute_dir.exists():
            raise FileNotFoundError(
                f"atlas-rag precompute dir not found at {precompute_dir}. "
                f"Run AtlasRagRetriever.build_index first."
            )
        manifest_path = precompute_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"atlas-rag retriever manifest not found at {manifest_path}. "
                f"Rebuild the precompute via AtlasRagRetriever.build_index."
            )
        manifest = json.loads(manifest_path.read_text())
        self._validate_manifest(
            manifest,
            expected_model=sentence_encoder_model,
            expected_keyword=keyword,
            expected_include_events=include_events,
            expected_include_concept=include_concept,
        )

        self.retriever_id = retriever_id or self.DEFAULT_RETRIEVER_ID
        self._kg_outputs_dir = kg_outputs_dir
        self._sentence_encoder_model = sentence_encoder_model
        self._keyword = keyword

        data = self._load_data_dict(
            precompute_dir=precompute_dir,
            graphml_path=kg_outputs_dir / GRAPHML_SUBDIR / f"{keyword}_graph.graphml",
            sentence_encoder_model=sentence_encoder_model,
            include_events=include_events,
            include_concept=include_concept,
        )
        self._inner = self._build_inner_retriever(
            llm_client=llm_client,
            sentence_encoder=sentence_encoder,
            data=data,
        )

    @classmethod
    def build_index(
        cls,
        kg_outputs_dir: Path,
        sentence_encoder: Any,
        sentence_encoder_model: str,
        keyword: str = "transcriptions.json",
        include_events: bool = False,
        include_concept: bool = True,
    ) -> None:
        """Compute embeddings + faiss indexes for the KG at ``kg_outputs_dir``.

        Thin wrapper around ``atlas_rag.vectorstore.create_embeddings_and_index``.
        Writes atlas-rag's own artifacts under
        ``<kg_outputs_dir>/precompute/`` plus a wrapper manifest at
        ``<kg_outputs_dir>/precompute/manifest.json``.

        Raises:
            FileNotFoundError: If the source graphml is missing.
        """
        from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index

        graphml_path = kg_outputs_dir / GRAPHML_SUBDIR / f"{keyword}_graph.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(
                f"atlas-rag source graphml not found at {graphml_path}. "
                f"Build the KG via `arandu build-kg` first."
            )

        precompute_dir = kg_outputs_dir / PRECOMPUTE_DIR_NAME
        precompute_dir.mkdir(parents=True, exist_ok=True)

        create_embeddings_and_index(
            sentence_encoder=sentence_encoder,
            model_name=sentence_encoder_model,
            working_directory=str(kg_outputs_dir),
            keyword=keyword,
            include_events=include_events,
            include_concept=include_concept,
        )

        graphml_sha = hashlib.sha256(graphml_path.read_bytes()).hexdigest()
        manifest = {
            "kg_outputs_dir": str(kg_outputs_dir),
            "keyword": keyword,
            "include_events": include_events,
            "include_concept": include_concept,
            "sentence_encoder_model": sentence_encoder_model,
            "graphml_sha256": graphml_sha,
            "built_at": datetime.now(UTC).isoformat(),
        }
        (precompute_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        *,
        expected_model: str,
        expected_keyword: str,
        expected_include_events: bool,
        expected_include_concept: bool,
    ) -> None:
        for field, expected in (
            ("sentence_encoder_model", expected_model),
            ("keyword", expected_keyword),
            ("include_events", expected_include_events),
            ("include_concept", expected_include_concept),
        ):
            if manifest.get(field) != expected:
                raise ValueError(
                    f"atlas-rag manifest field {field!r} mismatch: "
                    f"manifest has {manifest.get(field)!r}, constructor "
                    f"received {expected!r}. Rebuild the precompute or "
                    f"pass matching args."
                )

    @staticmethod
    def _load_data_dict(
        *,
        precompute_dir: Path,
        graphml_path: Path,
        sentence_encoder_model: str,
        include_events: bool,
        include_concept: bool,
    ) -> dict[str, Any]:
        """Load embeddings + graph into the dict ``HippoRAGRetriever`` expects.

        Filenames mirror ``atlas_rag.vectorstore.create_graph_index.create_embeddings_and_index``
        exactly — drift here is a load-time error.
        """
        import networkx as nx

        encoder_short = sentence_encoder_model.split("/")[-1]
        flags = f"event{include_events}_concept{include_concept}"
        keyword = graphml_path.stem.replace("_graph", "")
        node_emb_path = precompute_dir / f"{keyword}_{flags}_{encoder_short}_node_embeddings.pkl"
        edge_emb_path = precompute_dir / f"{keyword}_{flags}_{encoder_short}_edge_embeddings.pkl"
        node_list_path = precompute_dir / f"{keyword}_{flags}_node_list.pkl"
        edge_list_path = precompute_dir / f"{keyword}_{flags}_edge_list.pkl"
        text_emb_path = precompute_dir / f"{keyword}_{encoder_short}_text_embeddings.pkl"
        text_dict_path = precompute_dir / f"{keyword}_original_text_dict_with_node_id.pkl"

        for required in (
            node_emb_path,
            edge_emb_path,
            node_list_path,
            edge_list_path,
            text_emb_path,
            text_dict_path,
            graphml_path,
        ):
            if not required.exists():
                raise FileNotFoundError(
                    f"atlas-rag retriever artifact missing: {required}. "
                    f"Rebuild the precompute via AtlasRagRetriever.build_index."
                )

        with node_emb_path.open("rb") as f:
            node_embeddings = pickle.load(f)
        with edge_emb_path.open("rb") as f:
            edge_embeddings = pickle.load(f)
        with node_list_path.open("rb") as f:
            node_list = pickle.load(f)
        with edge_list_path.open("rb") as f:
            edge_list = pickle.load(f)
        with text_emb_path.open("rb") as f:
            text_embeddings = pickle.load(f)
        with text_dict_path.open("rb") as f:
            text_dict = pickle.load(f)
        with graphml_path.open("rb") as f:
            kg = nx.read_graphml(f)

        return {
            "node_embeddings": node_embeddings,
            "edge_embeddings": edge_embeddings,
            "node_list": node_list,
            "edge_list": edge_list,
            "text_embeddings": text_embeddings,
            "text_dict": text_dict,
            "KG": kg,
        }

    @staticmethod
    def _build_inner_retriever(
        *,
        llm_client: Any,
        sentence_encoder: Any,
        data: dict[str, Any],
    ) -> Any:
        """Construct the upstream HippoRAGRetriever; atlas-rag is imported here."""
        from atlas_rag.llm_generator import LLMGenerator
        from atlas_rag.retriever import HippoRAGRetriever

        llm_generator = LLMGenerator(
            client=llm_client,
            model_name=getattr(llm_client, "_arandu_model_id", "unknown"),
        )
        return HippoRAGRetriever(
            llm_generator=llm_generator,
            sentence_encoder=sentence_encoder,
            data=data,
        )

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Run HippoRAG retrieval and return ranked passages.

        Args:
            question: Natural-language query.
            top_k: Maximum number of passages to return.

        Returns:
            Ranked list of :class:`RetrievedPassage`. Each ``chunk_id``
            is atlas-rag's ``passage_id``; offsets are looked up via the
            ``passage_offsets.json`` sidecar by downstream judges.
        """
        scored = self._retrieve_with_scores(question, top_k=top_k)
        return _atlas_results_to_retrieved_passages(scored)

    def _retrieve_with_scores(self, question: str, *, top_k: int) -> list[tuple[str, float]]:
        """Score-capturing variant of upstream ``HippoRAGRetriever.retrieve``.

        Upstream's ``retrieve()`` computes PageRank scores but discards
        them at the final ``zip(*top_passages)`` step (cf.
        ``atlas_rag.retriever.hipporag`` in atlas-rag ``0.0.5.post1``).
        We replicate the upstream body and additionally surface the
        ``(passage_id, score)`` pairs for ``RetrievedPassage.score``.
        Revisit on the v006 upgrade tracked in the
        ``atlas-rag-v006-cleanup`` work session.
        """
        import networkx as nx

        inner = self._inner
        cfg = inner.inference_config

        if not cfg.is_filter_edges:
            personalization = inner.q2kg_fn(question, topN=cfg.topk_edges)
        else:
            personalization = inner.q2kg_fn(question, topN=cfg.topk_nodes)

        pr = nx.pagerank(
            inner.KG,
            personalization=personalization,
            alpha=cfg.ppr_alpha,
            max_iter=cfg.ppr_max_iter,
            tol=cfg.ppr_tol,
        )
        for node in pr:
            pr[node] = round(pr[node], 4)
            if pr[node] < 0.001:
                pr[node] = 0

        passage_probs: dict[str, float] = {}
        for node, weight in pr.items():
            file_ids = inner.KG.nodes[node]["file_id"].split(",")
            for file_id in set(file_ids):
                if file_id == "concept_file":
                    continue
                for node_id in inner.file_id_to_node_id.get(file_id, ()):
                    passage_probs[node_id] = passage_probs.get(node_id, 0.0) + weight

        ranked = sorted(passage_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (inner.text_id_to_node_name[passage_id], float(score)) for passage_id, score in ranked
        ]


def _atlas_results_to_retrieved_passages(
    results: Sequence[tuple[str, float]],
) -> list[RetrievedPassage]:
    """Convert atlas-rag's ``[(passage_id, score), ...]`` into our schema.

    Pure function — easy to unit-test independently of the heavy atlas-rag
    machinery. ``results`` is expected in descending-score order; ranks
    are assigned by position.
    """
    return [
        RetrievedPassage(
            chunk_id=passage_id,
            rank=rank,
            score=float(score),
            retriever_meta={"score_method": "hipporag"},
        )
        for rank, (passage_id, score) in enumerate(results)
    ]
