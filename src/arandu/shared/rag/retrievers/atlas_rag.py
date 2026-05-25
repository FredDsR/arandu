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

from arandu.kg.passage_offsets import build_passage_text_to_atlas_passage_id
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
        llm_model_id: str,
        sentence_encoder: Any,
        sentence_encoder_model: str,
        keyword: str = "transcriptions.json",
        include_events: bool = True,
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
            llm_model_id: Model identifier passed to atlas-rag's
                ``LLMGenerator`` (e.g. ``"qwen3:14b"``). The unified
                ``LLMClient`` exposes this as ``.model_id``; callers
                using ``LLMClient`` should forward that value here.
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
                the manifest, the ``kg_extraction`` subdir, or any
                precomputed artifact is missing.
            ValueError: If the manifest disagrees with constructor args,
                or if ``kg_extraction`` is present but contains no
                parseable records (empty passage_text→passage_id bridge).
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
        graphml_path = kg_outputs_dir / GRAPHML_SUBDIR / f"{keyword}_graph.graphml"
        kg_extraction_dir = kg_outputs_dir / "kg_extraction"
        if not kg_extraction_dir.exists():
            raise FileNotFoundError(
                f"atlas-rag kg_extraction dir not found at {kg_extraction_dir}. "
                f"Required to bridge KG passage nodes to atlas-rag's "
                f"synthesized passage_id namespace; without it every "
                f"RetrievedPassage would carry an opaque KG hash that can't "
                f"be joined with passage_offsets.json."
            )
        manifest = json.loads(manifest_path.read_text())
        self._validate_manifest(
            manifest,
            expected_model=sentence_encoder_model,
            expected_keyword=keyword,
            expected_include_events=include_events,
            expected_include_concept=include_concept,
            graphml_path=graphml_path,
        )

        self.retriever_id = retriever_id or self.DEFAULT_RETRIEVER_ID
        self._kg_outputs_dir = kg_outputs_dir
        self._sentence_encoder_model = sentence_encoder_model
        self._keyword = keyword

        # passage_text → "<source_file_id>:<chunk_index>" — bridges atlas-rag's
        # internal KG-passage-text identity to the synthesized passage_id
        # namespace used by `passage_offsets.json` (PR #100). Upstream's
        # HippoRAG returns passage TEXT for each match; we look it up here to
        # emit a stable, sidecar-joinable `RetrievedPassage.chunk_id`.
        self._text_to_passage_id = build_passage_text_to_atlas_passage_id(kg_extraction_dir)
        if not self._text_to_passage_id:
            raise ValueError(
                f"kg_extraction at {kg_extraction_dir} contained no parseable "
                f"records — passage_text→passage_id bridge is empty and every "
                f"retrieval would silently drop all results. Rebuild the KG "
                f"via `arandu build-kg`."
            )

        data = self._load_data_dict(
            precompute_dir=precompute_dir,
            graphml_path=graphml_path,
            sentence_encoder_model=sentence_encoder_model,
            include_events=include_events,
            include_concept=include_concept,
            artifact_sha256=manifest.get("artifact_sha256", {}),
        )
        self._inner = self._build_inner_retriever(
            llm_client=llm_client,
            llm_model_id=llm_model_id,
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
        include_events: bool = True,
        include_concept: bool = True,
    ) -> None:
        """Compute embeddings + faiss indexes for the KG at ``kg_outputs_dir``.

        Thin wrapper around ``atlas_rag.vectorstore.create_embeddings_and_index``.
        Writes atlas-rag's own artifacts under
        ``<kg_outputs_dir>/precompute/`` plus a wrapper manifest at
        ``<kg_outputs_dir>/precompute/manifest.json``.

        ``include_events`` and ``include_concept`` control which node types
        participate in the node embedding set. Atlas-rag accepts only three
        of the four combinations (``(False, True)`` raises upstream) — we
        defend that here with a clearer error. Defaults to ``(True, True)``,
        matching how the project's KG is constructed (events + concepts +
        entities) per :mod:`arandu.kg.atlas_backend`.

        Raises:
            FileNotFoundError: If the source graphml is missing.
            ValueError: If ``(include_events, include_concept) == (False, True)``,
                which atlas-rag rejects.
        """
        # Validate args BEFORE importing atlas-rag so the error fires cleanly
        # in environments where the `kg` extra is not installed (e.g. CI).
        if not include_events and include_concept:
            raise ValueError(
                "atlas-rag does not support include_events=False with "
                "include_concept=True. Valid combinations are: "
                "(False, False), (True, False), or (True, True)."
            )

        graphml_path = kg_outputs_dir / GRAPHML_SUBDIR / f"{keyword}_graph.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(
                f"atlas-rag source graphml not found at {graphml_path}. "
                f"Build the KG via `arandu build-kg` first."
            )

        from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index

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

        # Record sha256 of every artifact the retriever will pickle.load on
        # construction, mirroring the BM25 retriever's integrity model. Used
        # by `_load_data_dict` to fail before `pickle.load` if any artifact
        # has changed or been tampered with.
        artifact_paths = cls._artifact_paths(
            precompute_dir=precompute_dir,
            keyword=keyword,
            sentence_encoder_model=sentence_encoder_model,
            include_events=include_events,
            include_concept=include_concept,
        )
        artifact_sha256 = {
            relname: hashlib.sha256(path.read_bytes()).hexdigest()
            for relname, path in artifact_paths.items()
        }

        manifest = {
            "kg_outputs_dir": str(kg_outputs_dir),
            "keyword": keyword,
            "include_events": include_events,
            "include_concept": include_concept,
            "sentence_encoder_model": sentence_encoder_model,
            "graphml_sha256": hashlib.sha256(graphml_path.read_bytes()).hexdigest(),
            "artifact_sha256": artifact_sha256,
            "built_at": datetime.now(UTC).isoformat(),
        }
        (precompute_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _artifact_paths(
        *,
        precompute_dir: Path,
        keyword: str,
        sentence_encoder_model: str,
        include_events: bool,
        include_concept: bool,
    ) -> dict[str, Path]:
        """Map a stable artifact label to its on-disk path.

        Filenames mirror ``atlas_rag.vectorstore.create_graph_index.create_embeddings_and_index``
        exactly. The label keys are what end up in
        ``manifest["artifact_sha256"]`` so `_load_data_dict` can verify
        integrity without re-deriving filenames from arguments.
        """
        encoder_short = sentence_encoder_model.split("/")[-1]
        flags = f"event{include_events}_concept{include_concept}"
        return {
            "node_embeddings": precompute_dir
            / f"{keyword}_{flags}_{encoder_short}_node_embeddings.pkl",
            "edge_embeddings": precompute_dir
            / f"{keyword}_{flags}_{encoder_short}_edge_embeddings.pkl",
            "node_list": precompute_dir / f"{keyword}_{flags}_node_list.pkl",
            "edge_list": precompute_dir / f"{keyword}_{flags}_edge_list.pkl",
            "text_embeddings": precompute_dir / f"{keyword}_{encoder_short}_text_embeddings.pkl",
            "text_dict": precompute_dir / f"{keyword}_original_text_dict_with_node_id.pkl",
        }

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        *,
        expected_model: str,
        expected_keyword: str,
        expected_include_events: bool,
        expected_include_concept: bool,
        graphml_path: Path,
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

        # KG drift: if the graphml has been rebuilt without re-running
        # build_index, the embeddings index nodes that may no longer exist.
        # Detect this loudly — silent stale precompute would corrupt PPR.
        recorded_graphml_sha = manifest.get("graphml_sha256")
        if recorded_graphml_sha is None:
            raise ValueError(
                "atlas-rag manifest has no 'graphml_sha256' field — refusing "
                "to load an unverifiable precompute. Rebuild the index."
            )
        actual_graphml_sha = hashlib.sha256(graphml_path.read_bytes()).hexdigest()
        if actual_graphml_sha != recorded_graphml_sha:
            raise ValueError(
                f"graphml sha256 mismatch at {graphml_path}: manifest records "
                f"{recorded_graphml_sha[:12]}…, computed {actual_graphml_sha[:12]}… "
                f"— the KG was rebuilt without rebuilding the precompute. "
                f"Rerun AtlasRagRetriever.build_index."
            )

    @classmethod
    def _load_data_dict(
        cls,
        *,
        precompute_dir: Path,
        graphml_path: Path,
        sentence_encoder_model: str,
        include_events: bool,
        include_concept: bool,
        artifact_sha256: dict[str, str],
    ) -> dict[str, Any]:
        """Load embeddings + graph into the dict ``HippoRAGRetriever`` expects.

        Each pickle artifact is integrity-verified against the manifest's
        ``artifact_sha256`` map before ``pickle.load`` is called — mirrors
        BM25's tamper-detection model and prevents arbitrary-code-execution
        if the precompute dir is influenced by an untrusted writer.
        """
        import networkx as nx

        keyword = graphml_path.stem.replace("_graph", "")
        artifacts = cls._artifact_paths(
            precompute_dir=precompute_dir,
            keyword=keyword,
            sentence_encoder_model=sentence_encoder_model,
            include_events=include_events,
            include_concept=include_concept,
        )

        for path in (*artifacts.values(), graphml_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"atlas-rag retriever artifact missing: {path}. "
                    f"Rebuild the precompute via AtlasRagRetriever.build_index."
                )

        loaded: dict[str, Any] = {}
        for label, path in artifacts.items():
            recorded = artifact_sha256.get(label)
            if recorded is None:
                raise ValueError(
                    f"atlas-rag manifest has no sha256 for artifact "
                    f"{label!r} — refusing to load an unverifiable pickle. "
                    f"Rebuild the index."
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != recorded:
                raise ValueError(
                    f"sha256 mismatch on {path}: manifest records "
                    f"{recorded[:12]}…, computed {actual[:12]}… — artifact "
                    f"may be corrupted or tampered. Rebuild the index."
                )
            with path.open("rb") as f:
                loaded[label] = pickle.load(f)

        with graphml_path.open("rb") as f:
            kg = nx.read_graphml(f)
        cls._patch_orphan_file_ids(kg)

        return {
            "node_embeddings": loaded["node_embeddings"],
            "edge_embeddings": loaded["edge_embeddings"],
            "node_list": loaded["node_list"],
            "edge_list": loaded["edge_list"],
            "text_embeddings": loaded["text_embeddings"],
            "text_dict": loaded["text_dict"],
            "KG": kg,
        }

    @staticmethod
    def _patch_orphan_file_ids(kg: Any) -> int:
        """Inject the ``concept_file`` sentinel on KG nodes missing ``file_id``.

        Atlas-rag's :class:`HippoRAGRetriever.__init__` reads
        ``node["file_id"]`` unconditionally on every node, but our KG
        construction (`kg/atlas_backend.py` with qwen3:14b on PT prompts)
        emits **event nodes without a `file_id` attribute** — 4446 of 4451
        event nodes in ``test-kg-04``. The retriever's ``retrieve()`` already
        skips nodes whose ``file_id == "concept_file"`` (atlas-rag's
        own sentinel for orphan/concept nodes that shouldn't contribute
        to passage scoring), so injecting that value lets the orphan events
        survive into the graph + participate in PPR traversal without
        muddying passage probabilities.

        Returns the number of nodes patched. Callers may log it if useful.
        """
        patched = 0
        for _, attrs in kg.nodes(data=True):
            if "file_id" not in attrs:
                attrs["file_id"] = "concept_file"
                patched += 1
        if patched:
            logger.info(
                "Patched %d KG node(s) missing 'file_id' with the "
                "'concept_file' sentinel (atlas-rag compatibility shim).",
                patched,
            )
        return patched

    @staticmethod
    def _build_inner_retriever(
        *,
        llm_client: Any,
        llm_model_id: str,
        sentence_encoder: Any,
        data: dict[str, Any],
    ) -> Any:
        """Construct the upstream HippoRAGRetriever; atlas-rag is imported here."""
        from atlas_rag.llm_generator import LLMGenerator
        from atlas_rag.retriever import HippoRAGRetriever

        llm_generator = LLMGenerator(
            client=llm_client,
            model_name=llm_model_id,
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
        if top_k <= 0:
            # Contract: caller may pass top_k=0 to disable an arm without
            # branching elsewhere. Return [] cleanly rather than letting
            # the bridge loop's `len(out) >= 0` short-circuit yield a
            # stray record.
            return []
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

        # Pull more candidates than top_k because some may have no
        # `kg_extraction` bridge and need to be dropped. Walk the full ranked
        # list lazily, accumulating only successfully-bridged results.
        ranked_all = sorted(passage_probs.items(), key=lambda x: x[1], reverse=True)
        return self._bridge_to_atlas_passage_ids(ranked_all, top_k=top_k)

    def _bridge_to_atlas_passage_ids(
        self, ranked: list[tuple[str, float]], *, top_k: int
    ) -> list[tuple[str, float]]:
        """Convert ``(kg_passage_node, score)`` pairs to ``(<source_file_id>:<idx>, score)``.

        For each ranked KG passage node, look up the passage text via the
        inner retriever's ``text_id_to_node_name`` then bridge to atlas-rag's
        synthesized passage_id via ``self._text_to_passage_id``. Passages
        whose text has no matching ``kg_extraction`` JSONL record are
        dropped — surfacing them with an opaque KG hash would defeat the
        ``passage_offsets.json`` join the downstream judges depend on.

        Walks the full ranked list lazily and stops at ``top_k`` successful
        bridges, so a sparse bridge map doesn't shrink the result silently
        when un-bridgeable passages happen to rank highest.
        """
        out: list[tuple[str, float]] = []
        for kg_passage_id, score in ranked:
            text = self._inner.text_id_to_node_name[kg_passage_id]
            atlas_passage_id = self._text_to_passage_id.get(text)
            if atlas_passage_id is None:
                logger.debug(
                    "Skipping KG passage %s — no matching kg_extraction "
                    "record (text-equality miss).",
                    kg_passage_id,
                )
                continue
            out.append((atlas_passage_id, float(score)))
            if len(out) >= top_k:
                break
        return out


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
