"""Atlas-RAG (AutoSchemaKG) backend for knowledge graph construction.

This module is the ONLY place in the codebase that imports ``atlas_rag``.
All atlas-rag imports are deferred to method bodies so that the dependency
is only required when the ``atlas`` backend is actually selected.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.kg.config import KGConfig
    from arandu.shared.schemas import EnrichedRecord

from arandu.kg.schemas import KGConstructionResult
from arandu.shared.llm_client import create_llm_client
from arandu.utils.paths import get_project_root

logger = logging.getLogger(__name__)


def _parse_extraction_records(file_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Parse an extraction output file, handling both JSONL and pretty-printed JSON.

    Atlas-rag normally writes one JSON object per line (JSONL), but some runs
    (or older versions) produce pretty-printed JSON with one object spanning
    multiple lines.  This function transparently handles both formats.

    Args:
        file_path: Path to the extraction output file.

    Returns:
        Tuple of (valid_records, invalid_count).
    """
    content = file_path.read_text()
    if not content.strip():
        return [], 0

    # Fast path: try JSONL first (one object per line)
    lines = [line for line in content.split("\n") if line.strip()]
    records: list[dict[str, Any]] = []
    all_valid = True

    for line in lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                all_valid = False
        except json.JSONDecodeError:
            all_valid = False

    if all_valid and records:
        return records, 0

    # Slow path: content is not valid JSONL — try incremental JSON parsing
    # to handle pretty-printed or concatenated JSON objects
    records = []
    invalid_count = 0
    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)

    while idx < length:
        # Skip whitespace between objects
        while idx < length and content[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break

        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                invalid_count += 1
            idx = end_idx
        except json.JSONDecodeError:
            # Skip to the next line to try recovery
            next_newline = content.find("\n", idx)
            if next_newline == -1:
                invalid_count += 1
                break
            invalid_count += 1
            idx = next_newline + 1

    return records, invalid_count


# Default values for atlas-rag-specific options (overridden via backend_options)
ATLAS_DEFAULTS: dict[str, Any] = {
    "batch_size_triple": 3,
    "batch_size_concept": 16,
    "chunk_size": 8192,
    "max_new_tokens": 2048,
    "include_concept": True,
    "max_workers": 3,
}

# Path to atlas-rag prompt files relative to the project root
_PROMPTS_DIR = get_project_root() / "prompts" / "kg" / "atlas"

# Lazy reference — populated on first use by _get_enriched_processor_cls()
_MetadataEnrichedProcessorCls: type | None = None


def _create_enriched_processor_cls() -> type:
    """Create a DatasetProcessor subclass for metadata header injection.

    Returns:
        A subclass of ``atlas_rag.kg_construction.triple_extraction.DatasetProcessor``.
    """
    from atlas_rag.kg_construction.triple_extraction import DatasetProcessor

    class _MetadataEnrichedProcessor(DatasetProcessor):
        """DatasetProcessor that prepends metadata headers to every chunk.

        Reads a ``_metadata_header`` key from the document metadata dict,
        removes it before calling ``super()``, then prepends the header
        string to each chunk's text.
        """

        def create_sample_chunks(
            self,
            sample: dict[str, Any],
        ) -> list[dict[str, Any]]:
            """Override to inject metadata header into each chunk."""
            metadata = sample.get("metadata", {})
            header = metadata.get("_metadata_header", "")

            if header:
                sample = {
                    **sample,
                    "metadata": {k: v for k, v in metadata.items() if k != "_metadata_header"},
                }

            chunks = super().create_sample_chunks(sample)

            if header:
                for chunk in chunks:
                    chunk["text"] = f"{header}\n{chunk['text']}"

            return chunks

    return _MetadataEnrichedProcessor


def _get_enriched_processor_cls() -> type:
    """Return (and cache) the metadata-enriched DatasetProcessor subclass."""
    global _MetadataEnrichedProcessorCls
    if _MetadataEnrichedProcessorCls is None:
        _MetadataEnrichedProcessorCls = _create_enriched_processor_cls()
    return _MetadataEnrichedProcessorCls


def _patched_csvs_to_temp_graphml(
    triple_node_file: str,
    triple_edge_file: str,
    config: Any,
) -> None:
    """Patched ``csvs_to_temp_graphml`` that guards against orphan nodes.

    atlas-rag's ``add_edge()`` auto-creates nodes without attributes when
    edge endpoints are missing from the nodes CSV. This causes
    ``generate_concept()`` to crash with ``KeyError: 'id'``.

    This patch ensures both edge endpoints exist as fully-attributed
    nodes before adding edges. Can be removed once atlas-rag upstream
    fixes the issue (HKUST-KnowComp/AutoSchemaKG).

    Args:
        triple_node_file: Path to the triple nodes CSV.
        triple_edge_file: Path to the triple edges CSV.
        config: atlas-rag ``ProcessingConfig`` instance.
    """
    import os
    import pickle

    import networkx as nx
    from atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml import (
        get_node_id,
    )

    g = nx.DiGraph()
    entity_to_id: dict[str, str] = {}

    # Add triple nodes (same as original)
    with open(triple_node_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            mapped_id = get_node_id(node_id, entity_to_id)
            if mapped_id not in g.nodes:
                g.add_node(mapped_id, id=node_id, type=row["type"])

    # Add triple edges — ensure endpoints exist with attributes
    orphan_count = 0
    with open(triple_edge_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            if start_id not in g.nodes:
                g.add_node(start_id, id=row[":START_ID"], type="entity")
                orphan_count += 1
            if end_id not in g.nodes:
                g.add_node(end_id, id=row[":END_ID"], type="entity")
                orphan_count += 1
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

    if orphan_count:
        logger.warning("Patched %d orphan nodes with default attributes", orphan_count)

    output_name = (
        f"{config.output_directory}/kg_graphml/{config.filename_pattern}_without_concept.pkl"
    )
    output_dir = os.path.dirname(output_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_name, "wb") as output_file:
        pickle.dump(g, output_file)


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
        self._llm_client: Any | None = None

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

        # Step 3: Create LLM generator using unified LLMClient
        if self._llm_client is None:
            self._llm_client = self._build_llm_client()
        generation_config = self._build_generation_config()
        model = LLMGenerator(
            self._llm_client.client,
            self._config.model_id,
            default_config=generation_config,
        )

        # Step 4: Detect resume offset and build ProcessingConfig
        resume_from = self._detect_resume_offset(output_dir)
        processing_config = self._build_processing_config(
            input_dir,
            output_dir,
            resume_from=resume_from,
        )

        # Step 5: Run atlas-rag extraction pipeline
        extractor = KnowledgeGraphExtractor(model, processing_config)
        self._run_pipeline(extractor, output_dir)

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

        Builds a per-document metadata header and stores it in the metadata
        dict under ``_metadata_header``.  The enriched processor subclass
        reads this key after chunking and prepends it to every chunk.

        Args:
            records: Transcription records to convert.
            input_dir: Directory to write the input JSON file.
        """
        input_dir.mkdir(parents=True, exist_ok=True)
        labels = self._load_metadata_labels()

        documents = []
        for record in records:
            header = self._build_metadata_header(record, labels)
            metadata: dict[str, Any] = {"lang": self._config.language}
            if header:
                metadata["_metadata_header"] = header
            documents.append(
                {
                    "id": record.file_id,
                    "text": record.transcription_text,
                    "metadata": metadata,
                }
            )

        input_file = input_dir / "transcriptions.json"
        input_file.write_text(json.dumps(documents, ensure_ascii=False, indent=2))
        logger.info("Prepared %d documents for atlas-rag in %s", len(documents), input_file)

    @staticmethod
    def _build_metadata_header(
        record: EnrichedRecord,
        labels: dict[str, str],
    ) -> str:
        """Build a metadata header string for a transcription record.

        If the record has no source metadata, all fields are None, or no
        labels are provided, returns an empty string. Otherwise builds a
        header with translated labels ending with the transcription marker.

        Args:
            record: A single transcription record.
            labels: Translated label mapping loaded from metadata_labels.json.

        Returns:
            Header string to prepend to chunks, or empty string.
        """
        meta = record.source_metadata
        if meta is None or not labels:
            return ""

        field_map = (
            ("participant", meta.participant_name),
            ("location", meta.location),
            ("date", meta.recording_date),
            ("context", meta.event_context),
            ("researcher", meta.researcher_name),
            ("sequence", meta.sequence_label),
        )
        lines = [f"{labels.get(key, key.title())}: {value}" for key, value in field_map if value]

        if not lines:
            return ""

        header_title = labels.get("header", "[Context]")
        transcription_marker = labels.get("transcription", "[Transcription]")
        return f"{header_title}\n" + "\n".join(lines) + f"\n\n{transcription_marker}"

    def _load_metadata_labels(self) -> dict[str, str]:
        """Load translated metadata labels from the prompts directory.

        Reads ``metadata_labels.json`` (same directory as ``prompts.json``)
        and selects the entry matching ``self._config.language``. Falls back
        to English if the language key is missing or the file does not exist.

        Returns:
            Label mapping for the configured language.
        """
        labels_file = _PROMPTS_DIR / "metadata_labels.json"
        if not labels_file.exists():
            logger.warning("Metadata labels file not found: %s", labels_file)
            return {}

        all_labels = json.loads(labels_file.read_text())
        lang = self._config.language
        if lang in all_labels:
            return all_labels[lang]

        logger.warning(
            "Language '%s' not in metadata labels, falling back to 'en'",
            lang,
        )
        return all_labels.get("en", {})

    def _detect_resume_offset(self, output_dir: Path) -> int:
        """Detect completed batches from a previous extraction run.

        Scans ``atlas_output/kg_extraction/`` for existing output files, parses
        them (handling both JSONL and pretty-printed formats), counts valid JSON
        records, and normalizes all files to proper JSONL.  If a partial last
        batch is detected, trailing records are trimmed so the resumed run does
        not produce duplicates.

        Args:
            output_dir: The pipeline output directory (parent of ``atlas_output``).

        Returns:
            Number of complete batches to skip (``resume_from`` value).
        """
        kg_dir = output_dir / "atlas_output" / "kg_extraction"
        if not kg_dir.exists():
            return 0

        output_files = sorted(kg_dir.glob("*.json"))
        if not output_files:
            return 0

        batch_size = self._opts["batch_size_triple"]

        # Parse and normalize each file, counting valid records
        record_counts: list[int] = []
        for f in output_files:
            records, invalid = _parse_extraction_records(f)

            if invalid > 0:
                logger.warning("Stripped %d invalid records from %s", invalid, f)

            if not records:
                logger.info("Removing empty extraction file %s", f)
                f.unlink()
                continue

            # Rewrite as proper JSONL (normalizes pretty-printed files)
            f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
            record_counts.append(len(records))

        total_records = sum(record_counts)
        completed_batches = total_records // batch_size
        expected_records = completed_batches * batch_size

        # Trim the last file if it contains a partial batch
        if total_records > expected_records:
            # Re-read the last surviving file
            remaining_files = sorted(kg_dir.glob("*.json"))
            if remaining_files:
                last_file = remaining_files[-1]
                last_lines = [ln for ln in last_file.read_text().strip().split("\n") if ln.strip()]
                records_in_previous = total_records - len(last_lines)
                keep = expected_records - records_in_previous

                if keep <= 0:
                    logger.info("Removing fully-partial extraction file %s", last_file)
                    last_file.unlink()
                else:
                    last_file.write_text("\n".join(last_lines[:keep]) + "\n")
                    trimmed = len(last_lines) - keep
                    logger.info("Trimmed %d partial records from %s", trimmed, last_file)

        if completed_batches > 0:
            logger.info(
                "Resuming from batch %d (%d chunks already processed)",
                completed_batches,
                expected_records,
            )

        return completed_batches

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

    def _build_llm_client(self) -> Any:
        """Build a unified ``LLMClient`` from KGConfig settings.

        Delegates provider/base_url resolution to the shared ``LLMClient``
        so that API key handling and URL precedence stay consistent with
        other pipelines (QA, CEP).

        Returns:
            A configured ``LLMClient`` instance.
        """
        return create_llm_client(
            provider=self._config.provider,
            model_id=self._config.model_id,
            base_url=self._config.base_url
            or (self._config.ollama_url if self._config.provider == "ollama" else None),
        )

    def _build_generation_config(self) -> Any:
        """Build an atlas-rag ``GenerationConfig`` from KGConfig settings.

        Returns:
            A ``GenerationConfig`` instance with temperature from KGConfig.
        """
        from atlas_rag.llm_generator.generation_config import GenerationConfig

        return GenerationConfig(
            temperature=self._config.temperature,
            max_tokens=self._opts["max_new_tokens"],
        )

    def _build_processing_config(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        resume_from: int = 0,
    ) -> Any:
        """Build atlas-rag ProcessingConfig from KGConfig and backend_options.

        Args:
            input_dir: Directory containing prepared input JSON.
            output_dir: Directory for atlas-rag outputs.
            resume_from: Batch index to resume extraction from (0 = fresh start).

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
            resume_from=resume_from,
        )

    def _run_pipeline(self, extractor: Any, output_dir: Path) -> None:
        """Run the atlas-rag extraction pipeline steps.

        Temporarily replaces atlas-rag's ``DatasetProcessor`` with a subclass
        that prepends metadata headers to every chunk, then restores the
        original class after extraction completes.

        Args:
            extractor: A ``KnowledgeGraphExtractor`` instance.
            output_dir: Pipeline output directory (parent of ``atlas_output``).
        """
        from atlas_rag.kg_construction import triple_extraction

        enriched_cls = _get_enriched_processor_cls()
        original_cls = triple_extraction.DatasetProcessor

        logger.info("Starting triple extraction...")
        triple_extraction.DatasetProcessor = enriched_cls
        try:
            extractor.run_extraction()
        finally:
            triple_extraction.DatasetProcessor = original_cls

        logger.info("Converting JSON to CSV...")
        from atlas_rag.kg_construction.utils.csv_processing import csv_to_graphml

        original_csvs_to_temp = csv_to_graphml.csvs_to_temp_graphml
        csv_to_graphml.csvs_to_temp_graphml = _patched_csvs_to_temp_graphml
        try:
            extractor.convert_json_to_csv()
        finally:
            csv_to_graphml.csvs_to_temp_graphml = original_csvs_to_temp

        if self._opts["include_concept"]:
            self._run_concept_generation_with_resume(extractor, output_dir)
            extractor.create_concept_csv()

        logger.info("Converting to GraphML...")
        extractor.convert_to_graphml()
        logger.info("Atlas-rag pipeline completed")

    # ------------------------------------------------------------------
    # Resumable concept generation
    # ------------------------------------------------------------------

    _VALID_NODE_TYPES = frozenset({"event", "entity", "relation"})

    def _run_concept_generation_with_resume(
        self,
        extractor: Any,
        output_dir: Path,
    ) -> None:
        """Run concept generation with resume support.

        Wraps ``extractor.generate_concept_csv_temp()`` so that progress
        survives interruptions (e.g. SLURM timeouts).  A cumulative
        ``concept_completed.csv`` file grows across failures.  On each
        resume, already-conceptualized nodes are excluded from the input.

        Args:
            extractor: A ``KnowledgeGraphExtractor`` instance.
            output_dir: Pipeline output directory (parent of ``atlas_output``).

        Raises:
            FileNotFoundError: If the missing_concepts CSV does not exist.
        """
        atlas_output = output_dir / "atlas_output"
        concepts_dir = atlas_output / "concepts"
        triples_dir = atlas_output / "triples_csv"

        # Step 0: Locate missing_concepts CSV (input)
        missing_csvs = sorted(triples_dir.glob("missing_concepts*_from_json.csv"))
        if not missing_csvs:
            raise FileNotFoundError(
                f"No missing_concepts CSV found in {triples_dir}. "
                "Triple extraction + convert_json_to_csv must run first."
            )
        missing_csv = missing_csvs[0]

        concepts_dir.mkdir(parents=True, exist_ok=True)
        shard_file = concepts_dir / "concept_shard_0.csv"
        accumulator = concepts_dir / "concept_completed.csv"
        backup_file = missing_csv.with_suffix(missing_csv.suffix + ".bak")

        # Step 2: Restore backup if dirty state from previous crash
        if backup_file.exists():
            logger.info("Restoring input CSV from backup: %s", backup_file)
            backup_file.replace(missing_csv)

        # Step 3: Absorb leftover shard from interrupted run
        if shard_file.exists():
            valid_rows = self._read_valid_concept_rows(shard_file)
            if valid_rows:
                self._append_to_accumulator(accumulator, valid_rows)
                logger.info(
                    "Absorbed %d rows from interrupted shard into accumulator",
                    len(valid_rows),
                )
            shard_file.unlink()

        # Step 4: Read completed node names
        completed_nodes = self._read_completed_nodes(accumulator)

        # Step 5: Backup input CSV
        logger.info("Backing up input CSV: %s -> %s", missing_csv, backup_file)
        shutil.copy2(missing_csv, backup_file)

        # Step 6: Trim input CSV to exclude completed nodes
        remaining = self._trim_input_csv(missing_csv, completed_nodes)

        # Step 7: If all nodes are done, skip generation
        if remaining == 0:
            logger.info("All concept nodes already completed, skipping generation")
            self._restore_and_finalize(
                missing_csv, backup_file, accumulator, shard_file, concepts_dir
            )
            # Ensure concept_shard_0.csv exists so downstream create_concept_csv()
            # has a valid input even when there were zero remaining nodes.
            if not shard_file.exists():
                shard_file.write_text("")
            return

        # Step 8: Run concept generation
        logger.info(
            "Generating concepts for %d remaining nodes (%d already completed)",
            remaining,
            len(completed_nodes),
        )
        extractor.generate_concept_csv_temp(
            batch_size=self._opts["batch_size_concept"],
            language=self._config.language,
        )

        # Step 9: Absorb new shard output
        if shard_file.exists():
            valid_rows = self._read_valid_concept_rows(shard_file)
            if valid_rows:
                self._append_to_accumulator(accumulator, valid_rows)
                logger.info("Absorbed %d new concept rows", len(valid_rows))
            shard_file.unlink()

        # Steps 11-13: Restore input, finalize accumulator
        self._restore_and_finalize(missing_csv, backup_file, accumulator, shard_file, concepts_dir)

    def _read_valid_concept_rows(self, csv_path: Path) -> list[list[str]]:
        """Read and validate rows from a concept CSV file.

        Each valid row has exactly 3 non-empty columns with column 2
        being one of ``event``, ``entity``, ``relation`` (lowercase).

        Args:
            csv_path: Path to a concept CSV file.

        Returns:
            List of valid [node, description, node_type] rows.
        """
        valid: list[list[str]] = []
        try:
            with csv_path.open(newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 3 and all(cell.strip() for cell in row):
                        if row[2].strip().lower() in self._VALID_NODE_TYPES:
                            stripped = [cell.strip() for cell in row]
                            stripped[2] = stripped[2].lower()
                            valid.append(stripped)
                        else:
                            logger.debug("Dropping row with invalid node_type: %s", row)
                    else:
                        logger.debug("Dropping malformed concept row: %s", row)
        except (OSError, csv.Error) as exc:
            logger.warning("Failed to read concept CSV %s: %s", csv_path, exc)
        return valid

    def _append_to_accumulator(
        self,
        accumulator: Path,
        rows: list[list[str]],
    ) -> None:
        """Append rows to the accumulator CSV, deduplicating by node name.

        Deduplication is case-sensitive on column 0 (node name). On
        conflict, the latest entry wins.

        Args:
            accumulator: Path to ``concept_completed.csv``.
            rows: Validated concept rows to append.
        """
        existing: dict[str, list[str]] = {}
        if accumulator.exists():
            with accumulator.open(newline="") as f:
                for row in csv.reader(f):
                    if len(row) == 3:
                        existing[row[0]] = row

        for row in rows:
            existing[row[0]] = row

        with accumulator.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(existing.values())

    def _read_completed_nodes(self, accumulator: Path) -> set[str]:
        """Read completed node names from the accumulator.

        Args:
            accumulator: Path to ``concept_completed.csv``.

        Returns:
            Set of completed node names (column 0, case-sensitive).
        """
        if not accumulator.exists():
            return set()

        nodes: set[str] = set()
        with accumulator.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 1 and row[0].strip():
                    nodes.add(row[0].strip())
        return nodes

    def _trim_input_csv(self, csv_path: Path, completed: set[str]) -> int:
        """Rewrite input CSV excluding already-completed nodes.

        Reads all rows, filters out nodes in ``completed``, and rewrites
        the file.  The first row is treated as a header if present.

        Args:
            csv_path: Path to the missing_concepts CSV.
            completed: Node names to exclude.

        Returns:
            Number of remaining data rows (excluding header).
        """
        with csv_path.open(newline="") as f:
            all_rows = list(csv.reader(f))

        if not all_rows:
            return 0

        header = all_rows[0]
        data_rows = [row for row in all_rows[1:] if row and row[0] not in completed]

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)

        if completed:
            logger.info(
                "Trimmed input CSV: %d -> %d rows (%d already completed)",
                len(all_rows) - 1,
                len(data_rows),
                len(all_rows) - 1 - len(data_rows),
            )
        return len(data_rows)

    def _restore_and_finalize(
        self,
        missing_csv: Path,
        backup_file: Path,
        accumulator: Path,
        shard_file: Path,
        concepts_dir: Path,
    ) -> None:
        """Restore input CSV and rename accumulator to final shard.

        Args:
            missing_csv: Original input CSV path.
            backup_file: Backup of original input.
            accumulator: Cumulative concept CSV.
            shard_file: Expected final shard path.
            concepts_dir: Concepts output directory.
        """
        # Step 11: Restore original input from backup
        if backup_file.exists():
            backup_file.replace(missing_csv)
            logger.info("Restored original input CSV from backup")

        # Step 12: Rename accumulator to concept_shard_0.csv
        if accumulator.exists():
            accumulator.replace(shard_file)
            logger.info("Finalized concept output: %s", shard_file)

        # Step 13: Guard — only concept_shard_0.csv should exist
        csv_files = [f for f in concepts_dir.glob("*.csv") if f.name != "concept_shard_0.csv"]
        if csv_files:
            logger.warning(
                "Unexpected CSV files in concepts/: %s",
                [f.name for f in csv_files],
            )

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

        from arandu.kg.schemas import KGMetadata

        # Find the newest GraphML file produced by atlas-rag
        atlas_output = output_dir / "atlas_output"
        graphml_files = sorted(
            atlas_output.glob("**/*.graphml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not graphml_files:
            raise FileNotFoundError(f"No GraphML file found in atlas-rag output: {atlas_output}")

        graph_file = graphml_files[0]
        graph = nx.read_graphml(graph_file)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        logger.info("Graph loaded: %d nodes, %d edges from %s", node_count, edge_count, graph_file)

        # Build provenance metadata
        source_ids = [r.file_id for r in records]

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
