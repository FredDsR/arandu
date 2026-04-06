---
title: KG Construction
description: Build knowledge graphs from transcription results using AutoSchemaKG.
---

Build knowledge graphs from transcription results using AutoSchemaKG for entity and relation extraction.

## Overview

The KG construction pipeline extracts entities and relations from transcribed text to build knowledge graphs. Features include:

- **Entity Extraction**: People, locations, organizations, events, dates, concepts
- **Relation Extraction**: Semantic relationships between entities
- **Dynamic Schema**: AutoSchemaKG infers schema from data
- **Metadata Enrichment**: Automatic injection of source metadata into extraction context
- **GraphML Export**: NetworkX-compatible format for analysis

## Prerequisites

- Transcription results in `results/` directory
- Docker with Compose v2
- LLM provider (Ollama recommended)
- (Optional) GPU for faster Ollama inference

## Quick Start

### Using Docker Compose

```bash
# Start KG construction with Ollama sidecar
docker compose --profile kg up
```

### Using SLURM

```bash
sbatch scripts/slurm/run_kg_construction.slurm
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARANDU_KG_BACKEND` | `atlas` | KGC backend: `atlas` (AutoSchemaKG) |
| `ARANDU_KG_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `ARANDU_KG_MODEL_ID` | `llama3.1:8b` | Model for extraction |
| `ARANDU_KG_OLLAMA_URL` | `http://localhost:11434/v1` | Ollama API URL |
| `ARANDU_KG_BASE_URL` | *(none)* | Custom OpenAI-compatible endpoint |
| `ARANDU_KG_LANGUAGE` | `pt` | Language code (ISO 639-1): `pt`, `en` |
| `ARANDU_KG_TEMPERATURE` | `0.5` | LLM temperature (0.0-2.0, lower = more consistent) |
| `ARANDU_KG_OUTPUT_DIR` | `knowledge_graphs` | Output directory for graph artifacts |

### Language Support

The pipeline supports multilingual extraction via language-specific prompts:

| Code | Language | Prompts Directory |
|------|----------|-------------------|
| `pt` | Portuguese | `prompts/kg/atlas/` (language-keyed JSON) |
| `en` | English | `prompts/kg/atlas/` (language-keyed JSON) |

### Example .env Configuration

```bash
# KG Construction Settings
ARANDU_KG_BACKEND=atlas
ARANDU_KG_PROVIDER=ollama
ARANDU_KG_MODEL_ID=llama3.1:8b
ARANDU_KG_LANGUAGE=pt
ARANDU_KG_TEMPERATURE=0.5

# Directories
ARANDU_RESULTS_DIR=./results
ARANDU_KG_OUTPUT_DIR=./knowledge_graphs
```

## Usage Examples

### Basic Usage

```bash
# Default configuration
docker compose --profile kg up
```

### Custom Model

```bash
# Use larger model for better extraction
ARANDU_KG_MODEL_ID=llama3.1:70b docker compose --profile kg up
```

### Different Language

```bash
# Extract from English transcriptions
ARANDU_KG_LANGUAGE=en docker compose --profile kg up
```

### Using OpenAI

```bash
# Use OpenAI for extraction
export ARANDU_KG_PROVIDER=openai
export ARANDU_KG_MODEL_ID=gpt-4o
export OPENAI_API_KEY=sk-...
docker compose --profile kg up
```

### SLURM with Custom Settings

```bash
# Submit to specific partition
ARANDU_KG_MODEL_ID=qwen3:14b PIPELINE_ID=test-cep-01 \
  sbatch scripts/slurm/kg/tupi.slurm
```

## Output Format

Knowledge graphs are saved in the output directory:

```
<output_dir>/
├── atlas_input/
│   └── transcriptions.json        # Input prepared for atlas-rag
├── atlas_output/
│   ├── kg_extraction/             # Raw extraction results
│   ├── triples_csv/               # Extracted triples as CSV
│   └── <model>_<timestamp>.graphml  # Final knowledge graph
└── <model>_<timestamp>.metadata.json  # Provenance sidecar
```

### GraphML Structure

The GraphML files are NetworkX-compatible and contain:

**Nodes (Entities)**:
- `id`: Unique identifier
- `label`: Entity text
- `type`: Entity type (PERSON, LOCATION, ORGANIZATION, EVENT, DATE, CONCEPT)

**Edges (Relations)**:
- `source`: Source entity ID
- `target`: Target entity ID
- `relation`: Relation type (LOCATED_IN, CAUSED_BY, AFFECTED_BY, etc.)

### Metadata Schema

```json
{
  "graph_id": "test-cep-01",
  "source_documents": ["1abc123xyz", "2def456uvw"],
  "model_id": "qwen3:14b",
  "provider": "ollama",
  "language": "pt",
  "created_at": "2026-02-26T15:45:00Z",
  "total_documents": 2,
  "total_nodes": 342,
  "total_edges": 187,
  "backend_version": "atlas-rag==0.0.5"
}
```

## Working with Graphs

### Loading with NetworkX

```python
import networkx as nx

# Load the corpus graph
graph = nx.read_graphml("knowledge_graphs/corpus_graph.graphml")

# Basic statistics
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Density: {nx.density(graph):.4f}")

# List entity types
entity_types = {}
for node, attrs in graph.nodes(data=True):
    etype = attrs.get('type', 'UNKNOWN')
    entity_types[etype] = entity_types.get(etype, 0) + 1
print(f"Entity types: {entity_types}")
```

### Querying the Graph

```python
import networkx as nx

graph = nx.read_graphml("knowledge_graphs/corpus_graph.graphml")

# Find all PERSON entities
persons = [n for n, d in graph.nodes(data=True) if d.get('type') == 'PERSON']
print(f"Found {len(persons)} persons")

# Get neighbors of a node
node = persons[0]
neighbors = list(graph.neighbors(node))
print(f"Neighbors of {node}: {neighbors}")

# Find paths between entities
if len(persons) > 1:
    try:
        path = nx.shortest_path(graph, persons[0], persons[1])
        print(f"Path: {' -> '.join(path)}")
    except nx.NetworkXNoPath:
        print("No path found")
```

### Loading Metadata

```python
from arandu.schemas import KGMetadata

# Load metadata
metadata = KGMetadata.load("knowledge_graphs/corpus_graph_metadata.json")

print(f"Graph ID: {metadata.graph_id}")
print(f"Source documents: {len(metadata.source_documents)}")
print(f"Language: {metadata.language}")
print(f"Model: {metadata.model_id}")
```

## Entity Types

AutoSchemaKG extracts the following entity types:

| Type | Description | Examples |
|------|-------------|----------|
| `PERSON` | People, groups | "Maria Silva", "the community" |
| `LOCATION` | Geographic places | "Rio de Janeiro", "the riverbank" |
| `ORGANIZATION` | Institutions | "IBAMA", "the university" |
| `EVENT` | Occurrences | "the 2023 flood", "the meeting" |
| `DATE` | Temporal references | "January 2023", "last year" |
| `CONCEPT` | Abstract ideas | "environmental protection", "tradition" |

## Relation Types

Common relation types extracted:

| Relation | Description | Example |
|----------|-------------|---------|
| `LOCATED_IN` | Spatial containment | "community LOCATED_IN riverbank" |
| `CAUSED_BY` | Causal relationship | "flood CAUSED_BY heavy rain" |
| `AFFECTED_BY` | Impact relationship | "crops AFFECTED_BY drought" |
| `OCCURRED_IN` | Temporal/spatial | "meeting OCCURRED_IN January" |
| `BELONGS_TO` | Membership | "Maria BELONGS_TO community" |

## Monitoring Progress

### Docker Logs

```bash
# Watch KG construction logs
docker compose --profile kg logs -f arandu-kg

# Check Ollama status
docker compose --profile kg logs ollama
```

### SLURM Logs

```bash
# Monitor job output
tail -f logs/arandu-kg_<jobid>.out
```

## Atlas Backend Metadata Injection

When transcription records have source metadata (participant name, location, date, event context), the atlas backend automatically prepends a translated metadata header to **every chunk** sent to the LLM for triple extraction. This provides provenance context that improves entity and relation quality.

### How It Works

1. During input preparation, each document's `SourceMetadata` fields are formatted into a header using translated labels from `prompts/kg/atlas/metadata_labels.json`
2. The header is stored in the document's metadata dict as `_metadata_header`
3. A custom `DatasetProcessor` subclass intercepts atlas-rag's chunking step
4. After text is split into chunks, the header is prepended to **each chunk's text**
5. The LLM sees the full provenance context in every extraction prompt

### Example Chunk (Portuguese)

```
[Contexto da Entrevista]
Participante: João da Silva
Local: Barra de Pelotas
Data: 2024-03-15
Contexto: Audiência Câmara de Vereadores

[Transcrição]
...então a água subiu muito rápido, em menos de duas horas já estava...
```

### Example Chunk (English)

```
[Interview Context]
Participant: João da Silva
Location: Barra de Pelotas
Date: 2024-03-15
Event Context: Audiência Câmara de Vereadores

[Transcription]
...so the water rose very fast, in less than two hours it was already...
```

### Adding a New Language

Add a new key to `prompts/kg/atlas/metadata_labels.json`:

```json
{
  "es": {
    "header": "[Contexto de la Entrevista]",
    "transcription": "[Transcripción]",
    "participant": "Participante",
    "location": "Ubicación",
    "date": "Fecha",
    "context": "Contexto del Evento",
    "researcher": "Investigador(a)",
    "sequence": "Secuencia"
  }
}
```

Then add `"es"` to `KGConfig.validate_language` in `src/arandu/config.py`.

### Disabling Metadata Injection

If source metadata is not available on the transcription records (i.e., `source_metadata` is `None`), the header is simply omitted and chunks are passed through unmodified. No configuration flag is needed.

## Batch-Level Resume

When a SLURM job times out or is interrupted during triple extraction, the atlas backend automatically resumes from the last completed batch on the next run. No manual intervention is needed.

### How It Works

1. Atlas-rag writes extraction results as JSONL lines to `atlas_output/kg_extraction/`
2. On the next run, the backend counts existing JSONL lines and divides by `batch_size_triple` to determine completed batches
3. Any partial last batch (incomplete lines from a mid-batch interruption) is trimmed to avoid duplicates
4. `ProcessingConfig.resume_from` is set to skip already-processed batches
5. Atlas-rag creates a new timestamped output file for the remaining batches
6. Downstream steps (`json2csv`, concept generation, GraphML export) read all files in the directory and merge results automatically

### Requirements

- The `atlas_output/` directory from the previous run must be preserved (same `output_dir`)
- The input data must be identical (same records produce the same chunks and batch boundaries)
- `batch_size_triple` must be the same between runs

### Example

```bash
# First run — times out after processing 9 chunks (3 batches of 3)
PIPELINE_ID=test-cep-01 sbatch scripts/slurm/kg/tupi.slurm

# Resubmit — automatically detects 3 completed batches, resumes from batch 4
PIPELINE_ID=test-cep-01 sbatch scripts/slurm/kg/tupi.slurm
```

The logs will show:
```
Resuming from batch 3 (9 chunks already processed)
```

## Best Practices

1. **Model Selection**
   - Use `llama3.1:8b` for balanced speed/quality
   - Use `llama3.1:70b` for higher quality extraction
   - Lower temperature (0.3-0.5) for more consistent extraction

2. **Language Settings**
   - Match `KG_LANGUAGE` to your transcription language
   - Use appropriate prompt templates

3. **Backend Options**
   - Adjust `chunk_size` for memory constraints
   - Tune `batch_size_triple` for API rate limits
   - Set `max_workers` to control parallelism

## Troubleshooting

### Empty Graph

```bash
# Check if transcription results exist
ls -la results/*.json

# Verify transcription has content
cat results/<file_id>.json | jq '.transcription_text | length'
```

### Poor Entity Extraction

- Use larger model: `ARANDU_KG_MODEL_ID=llama3.1:70b`
- Lower temperature: `ARANDU_KG_TEMPERATURE=0.3`
- Verify language setting matches transcription language

### Memory Issues

```bash
# Use smaller model
export ARANDU_KG_MODEL_ID=llama3.2:3b

# Reduce chunk size via backend options
export ARANDU_KG_BACKEND_OPTIONS='{"chunk_size": 4096}'
```

### Ollama Timeout

```bash
# Increase keep-alive time
export OLLAMA_KEEP_ALIVE=30m

# Restart with more memory
docker compose --profile kg down
docker compose --profile kg up
```

---

**See also**: [QA Generation](qa-generation) | [Evaluation](evaluation) | [Configuration](../../configuration)
