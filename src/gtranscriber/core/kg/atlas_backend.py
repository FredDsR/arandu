"""Atlas-RAG (AutoSchemaKG) backend for knowledge graph construction.

This module is the ONLY place in the codebase that imports ``atlas_rag``.
All atlas-rag imports are deferred to method bodies so that the dependency
is only required when the ``atlas`` backend is actually selected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gtranscriber.config import KGConfig
    from gtranscriber.schemas import EnrichedRecord

from gtranscriber.core.kg.schemas import KGConstructionResult

logger = logging.getLogger(__name__)

# Default values for atlas-rag-specific options (overridden via backend_options)
ATLAS_DEFAULTS: dict[str, Any] = {
    "batch_size_triple": 3,
    "batch_size_concept": 16,
    "chunk_size": 8192,
    "max_new_tokens": 2048,
    "include_concept": True,
    "max_workers": 3,
}

# Path to Portuguese prompt files relative to the project root
_PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts" / "kg" / "atlas"


class AtlasRagConstructor:
    """KG constructor using atlas-rag (AutoSchemaKG) as the extraction backend.

    Implements the ``KGConstructor`` protocol. Receives framework-agnostic
    ``EnrichedRecord`` objects and orchestrates atlas-rag's 5-step pipeline
    behind a single ``build_graph()`` call.

    Args:
        config: KG pipeline configuration.
    """

    def __init__(self, config: KGConfig) -> None:
        self._config = config
        self._opts: dict[str, Any] = {**ATLAS_DEFAULTS, **config.backend_options}

    # ------------------------------------------------------------------
    # Public API (Protocol)
    # ------------------------------------------------------------------

    def build_graph(
        self,
        records: list[EnrichedRecord],
        output_dir: Path,
    ) -> KGConstructionResult:
        """Build a knowledge graph from transcription records.

        Args:
            records: Validated transcription records.
            output_dir: Directory for graph artifacts.

        Returns:
            Framework-agnostic construction result.
        """
        from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
        from atlas_rag.llm_generator import LLMGenerator

        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Prepare input data for atlas-rag
        input_dir = output_dir / "atlas_input"
        self._prepare_input_data(records, input_dir)

        # Step 2: Inject Portuguese concept prompts (monkey-patch)
        if self._opts["include_concept"]:
            self._inject_concept_prompts()

        # Step 3: Create OpenAI client and LLM generator
        client = self._create_openai_client()
        model = LLMGenerator(client, self._config.model_id)

        # Step 4: Build ProcessingConfig
        processing_config = self._build_processing_config(input_dir, output_dir)

        # Step 5: Run atlas-rag extraction pipeline
        extractor = KnowledgeGraphExtractor(model, processing_config)
        self._run_pipeline(extractor)

        # Step 6: Load graph and build result
        return self._build_result(records, output_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input_data(
        self,
        records: list[EnrichedRecord],
        input_dir: Path,
    ) -> None:
        """Convert EnrichedRecord list to atlas-rag JSON input format.

        Args:
            records: Transcription records to convert.
            input_dir: Directory to write the input JSON file.
        """
        input_dir.mkdir(parents=True, exist_ok=True)

        documents = []
        for record in records:
            documents.append(
                {
                    "id": record.gdrive_id,
                    "text": record.transcription_text,
                    "metadata": {"lang": self._config.language},
                }
            )

        input_file = input_dir / "transcriptions.json"
        input_file.write_text(json.dumps(documents, ensure_ascii=False, indent=2))
        logger.info("Prepared %d documents for atlas-rag in %s", len(documents), input_file)

    def _inject_concept_prompts(self) -> None:
        """Monkey-patch CONCEPT_INSTRUCTIONS with Portuguese entries.

        atlas-rag only ships en/zh-CN/zh-HK concept prompts and has no
        file-based custom concept prompt support.  We inject ``"pt"``
        entries at runtime before creating the extractor.
        """
        from atlas_rag.llm_generator.prompt.triple_extraction_prompt import (
            CONCEPT_INSTRUCTIONS,
        )

        if self._config.language in CONCEPT_INSTRUCTIONS:
            return

        concept_file = _PROMPTS_DIR / "concept_prompts.json"
        if not concept_file.exists():
            logger.warning("Concept prompt file not found: %s", concept_file)
            return

        all_prompts = json.loads(concept_file.read_text())
        lang = self._config.language
        if lang in all_prompts:
            CONCEPT_INSTRUCTIONS[lang] = all_prompts[lang]
            logger.info("Injected '%s' concept prompts into atlas-rag", lang)

    def _create_openai_client(self) -> Any:
        """Construct an OpenAI client from KGConfig settings.

        Mirrors the base_url resolution logic in ``LLMClient.__init__``.

        Returns:
            An ``openai.OpenAI`` client instance.
        """
        from openai import OpenAI

        base_url = self._config.base_url
        api_key: str | None = None

        if self._config.provider == "ollama":
            base_url = base_url or self._config.ollama_url
            api_key = "ollama"
        elif self._config.provider == "custom":
            if not base_url:
                raise ValueError("base_url is required for the 'custom' provider")

        return OpenAI(api_key=api_key, base_url=base_url)

    def _build_processing_config(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> Any:
        """Build atlas-rag ProcessingConfig from KGConfig and backend_options.

        Args:
            input_dir: Directory containing prepared input JSON.
            output_dir: Directory for atlas-rag outputs.

        Returns:
            A ``ProcessingConfig`` instance.
        """
        from atlas_rag.kg_construction.triple_config import ProcessingConfig

        prompt_path = str(_PROMPTS_DIR / "prompts.json")
        schema_path = str(_PROMPTS_DIR / "schema.json")

        return ProcessingConfig(
            model_path=self._config.model_id,
            data_directory=str(input_dir),
            filename_pattern="transcriptions.json",
            output_directory=str(output_dir / "atlas_output"),
            batch_size_triple=self._opts["batch_size_triple"],
            batch_size_concept=self._opts["batch_size_concept"],
            chunk_size=self._opts["chunk_size"],
            max_new_tokens=self._opts["max_new_tokens"],
            max_workers=self._opts["max_workers"],
            include_concept=self._opts["include_concept"],
            triple_extraction_prompt_path=prompt_path,
            triple_extraction_schema_path=schema_path,
        )

    def _run_pipeline(self, extractor: Any) -> None:
        """Run the atlas-rag extraction pipeline steps.

        Args:
            extractor: A ``KnowledgeGraphExtractor`` instance.
        """
        logger.info("Starting triple extraction...")
        extractor.run_extraction()

        logger.info("Converting JSON to CSV...")
        extractor.convert_json_to_csv()

        if self._opts["include_concept"]:
            logger.info("Generating concept CSV (temporary)...")
            extractor.generate_concept_csv_temp(
                batch_size=self._opts["batch_size_concept"],
            )
            logger.info("Creating final concept CSV...")
            extractor.create_concept_csv()

        logger.info("Converting to GraphML...")
        extractor.convert_to_graphml()
        logger.info("Atlas-rag pipeline completed")

    def _build_result(
        self,
        records: list[EnrichedRecord],
        output_dir: Path,
    ) -> KGConstructionResult:
        """Load the generated GraphML and construct the result.

        Args:
            records: Original transcription records.
            output_dir: Directory containing atlas-rag output.

        Returns:
            Framework-agnostic construction result.
        """
        import networkx as nx

        from gtranscriber.schemas import KGMetadata

        # Find the GraphML file produced by atlas-rag
        atlas_output = output_dir / "atlas_output"
        graphml_files = list(atlas_output.glob("**/*.graphml"))
        if not graphml_files:
            raise FileNotFoundError(f"No GraphML file found in atlas-rag output: {atlas_output}")

        graph_file = graphml_files[0]
        graph = nx.read_graphml(graph_file)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        logger.info("Graph loaded: %d nodes, %d edges from %s", node_count, edge_count, graph_file)

        # Build provenance metadata
        source_ids = [r.gdrive_id for r in records]

        try:
            import atlas_rag

            backend_ver = f"atlas-rag=={atlas_rag.__version__}"
        except AttributeError:
            backend_ver = "atlas-rag"

        metadata = KGMetadata(
            graph_id=output_dir.name,
            source_documents=source_ids,
            model_id=self._config.model_id,
            provider=self._config.provider,
            language=self._config.language,
            total_documents=len(records),
            total_nodes=node_count,
            total_edges=edge_count,
            backend_version=backend_ver,
        )

        # Save metadata sidecar
        metadata_path = graph_file.with_suffix(".metadata.json")
        metadata.save(metadata_path)
        logger.info("Metadata saved to %s", metadata_path)

        return KGConstructionResult(
            graph_file=graph_file,
            metadata=metadata,
            node_count=node_count,
            edge_count=edge_count,
            source_record_ids=source_ids,
        )
