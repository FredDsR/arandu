# KG Module: Agent Guide

Knowledge-graph construction on top of atlas-rag (AutoSchemaKG). Configuration
comes from `KGConfig` (`config.py`, env prefix `ARANDU_KG_`). The CLI entry
point is `arandu kg` (`cli/kg.py` -> `batch.run_batch_kg_construction`), which
resolves transcriptions via `ResultsManager` and writes under
`results/<pipeline-id>/kg/outputs/`.

## Module map

| File | Role |
| ---- | ---- |
| `atlas_backend.py` | The atlas-rag backend: extraction, concept generation, GraphML conversion, all resume logic and upstream shims |
| `batch.py` | Orchestrator: loads `EnrichedRecord`s, checkpointing, ResultsManager wiring |
| `config.py` | `KGConfig` (pydantic-settings, `ARANDU_KG_` prefix) |
| `factory.py` | Backend dispatch (`atlas` only today) |
| `passage_offsets.py` | `arandu kg-link-passages`: maps atlas-rag passages to char offsets in transcriptions |
| `retriever_index.py` | `arandu kg-build-retriever-index`: atlas-rag embedding precompute |
| `protocol.py`, `schemas.py` | `KGConstructor` protocol + result schemas |

## Pipeline stages (`AtlasBackend.build_graph` -> `_run_pipeline`)

1. Prepare input: `EnrichedRecord`s -> atlas-rag JSON under `atlas_input/`,
   with a per-document metadata header prepended to every chunk (enriched
   `DatasetProcessor` subclass, swapped in via monkey-patch).
2. Triple extraction (`run_extraction`): one LLM call per batch, output JSONL
   under `atlas_output/kg_extraction/`.
3. `convert_json_to_csv` -> `triples_csv/` (with the patched
   `csvs_to_temp_graphml`).
4. Concept generation with resume (the slow stage; see below).
5. Post-processing shims: relabel the synthesized English "is participated by"
   predicate to Portuguese; backfill `triple_nodes` with endpoints only
   referenced by `triple_edges` (upstream ID-consistency bug, AutoSchemaKG#45).
6. `convert_to_graphml` -> `kg_graphml/`.

Output layout: `results/<id>/kg/outputs/atlas_output/{kg_extraction,triples_csv,concepts,kg_graphml}`.
It is `atlas_output/` under `kg/outputs/`, never directly under `results/<id>/`.

## Resume semantics (interruptions are safe)

Both LLM stages survive SLURM timeouts and crashes. Resubmitting the same job
is always the right recovery; nothing is lost.

**Extraction** (`_detect_resume_offset`): scans `kg_extraction/*.json`, strips
invalid records, normalizes pretty-printed files to JSONL, trims any partial
trailing batch, and returns the number of completed batches to skip.

**Concept generation** (`_run_concept_generation_with_resume`): wraps
atlas-rag's `generate_concept_csv_temp` with a shard/accumulator protocol in
`atlas_output/concepts/`:

- During a run, atlas-rag appends rows to `concept_shard_0.csv` (the live
  shard). `concept_completed.csv` (the accumulator) may not exist yet on a
  first run: that is normal, not a sign of zero progress.
- On startup, a leftover shard from an interrupted run is validated and
  absorbed into the accumulator, completed nodes are trimmed from the input
  CSV (a `.bak` backup protects the original), and phantom rows (node IDs
  absent from `triple_*.csv`) are dropped.
- On clean completion the shard is absorbed and the accumulator is renamed
  back to `concept_shard_0.csv`, which is the final artifact downstream
  `create_concept_csv()` consumes. So `concept_shard_0.csv` is BOTH the live
  scratch file and the finished output; distinguish by whether the job is
  still running.

## What to expect at runtime

- Concept generation logs nothing per batch: stdout is buffered inside the
  container and progress goes to the shard CSV. Judge progress by the mtime
  and row count of `concepts/concept_shard_0.csv`, not by the SLURM log.
- Rough throughput observed 2026-06-11 (qwen3:14b on an RTX 4090, 32k ctx,
  `OLLAMA_NUM_PARALLEL=3`): ~150 nodes/hour, calls taking 2.5-4 min each.
  A few-thousand-node corpus does not fit a 24h wall; plan for one resume.
- "Item N is a duplicate triple" and "missing required keys" lines in the
  extraction log are normal LLM-output noise, not failures.
- Several upstream atlas-rag bugs are shimmed locally in `atlas_backend.py`
  (orphan `file_id`s, attributeless temp-kg nodes, phantom concept rows,
  endpoint backfill). atlas-rag 0.0.6 fixes some upstream; do not file new
  upstream issues, carry the shims (see project memory).
