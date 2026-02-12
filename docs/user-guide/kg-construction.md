# Knowledge Graph Construction Guide

> **⚠️ STATUS: PLANNED - NOT YET IMPLEMENTED**
>
> Knowledge graph construction using AutoSchemaKG is planned for a future release. The configuration schemas (`KGConfig`, `KGMetadata`) exist in `src/gtranscriber/config.py` and `src/gtranscriber/schemas.py`, but there are currently:
> - **No CLI commands** for KG construction (no `build-kg` command)
> - **No pipeline modules** implementing the KG construction logic
> - **No dependencies** added yet (`atlas-rag`, `networkx` are not in `pyproject.toml`)
>
> This documentation describes the **planned functionality** and serves as a specification for future implementation.

---

Build knowledge graphs from transcription results using AutoSchemaKG for entity and relation extraction.

## Overview

The KG construction pipeline extracts entities and relations from transcribed text to build knowledge graphs. Features include:

- **Entity Extraction**: People, locations, organizations, events, dates, concepts
- **Relation Extraction**: Semantic relationships between entities
- **Dynamic Schema**: AutoSchemaKG infers schema from data
- **Graph Merging**: Combine individual document graphs into corpus-level graph
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
| `GTRANSCRIBER_KG_PROVIDER` | `ollama` | LLM provider: `openai`, `ollama`, `custom` |
| `GTRANSCRIBER_KG_MODEL_ID` | `llama3.1:8b` | Model for extraction |
| `GTRANSCRIBER_KG_OLLAMA_URL` | `http://ollama:11434` | Ollama API URL |
| `GTRANSCRIBER_KG_MERGE_GRAPHS` | `true` | Merge into corpus graph |
| `GTRANSCRIBER_KG_OUTPUT_FORMAT` | `graphml` | Export format: `graphml`, `json` |
| `GTRANSCRIBER_KG_LANGUAGE` | `pt` | Language code (ISO 639-1) |
| `GTRANSCRIBER_KG_TEMPERATURE` | `0.5` | LLM temperature (lower = more consistent) |
| `GTRANSCRIBER_WORKERS` | `2` | Parallel workers |

### Language Support

The pipeline supports multilingual extraction via language-specific prompts:

| Code | Language | Prompt File |
|------|----------|-------------|
| `pt` | Portuguese | `prompts/pt_prompts.json` |
| `en` | English | `prompts/en_prompts.json` |
| `es` | Spanish | `prompts/es_prompts.json` |

### Example .env Configuration

```bash
# KG Construction Settings
GTRANSCRIBER_KG_PROVIDER=ollama
GTRANSCRIBER_KG_MODEL_ID=llama3.1:8b
GTRANSCRIBER_KG_MERGE_GRAPHS=true
GTRANSCRIBER_KG_OUTPUT_FORMAT=graphml
GTRANSCRIBER_KG_LANGUAGE=pt
GTRANSCRIBER_KG_TEMPERATURE=0.5
GTRANSCRIBER_WORKERS=4

# Directories
GTRANSCRIBER_RESULTS_DIR=./results
GTRANSCRIBER_KG_DIR=./knowledge_graphs
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
GTRANSCRIBER_KG_MODEL_ID=llama3.1:70b docker compose --profile kg up
```

### Different Language

```bash
# Extract from English transcriptions
GTRANSCRIBER_KG_LANGUAGE=en docker compose --profile kg up
```

### Skip Graph Merging

```bash
# Keep individual document graphs only
GTRANSCRIBER_KG_MERGE_GRAPHS=false docker compose --profile kg up
```

### Using OpenAI

```bash
# Use OpenAI for extraction
export GTRANSCRIBER_KG_PROVIDER=openai
export GTRANSCRIBER_KG_MODEL_ID=gpt-4o
export OPENAI_API_KEY=sk-...
docker compose --profile kg up
```

### SLURM with Custom Settings

```bash
# Submit with custom model and workers
KG_MODEL=llama3.1:70b WORKERS=8 KG_LANGUAGE=pt \
  sbatch scripts/slurm/run_kg_construction.slurm
```

## Output Format

Knowledge graphs are saved in `knowledge_graphs/`:

```
knowledge_graphs/
├── corpus_graph.graphml           # Merged corpus graph
├── corpus_graph_metadata.json     # Provenance metadata
├── individual/                    # Per-document graphs (optional)
│   ├── <gdrive_id_1>.graphml
│   └── <gdrive_id_2>.graphml
└── checkpoints/
    └── kg_checkpoint.json         # For resumption
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
  "graph_id": "corpus_merged_2026_01_26",
  "source_documents": ["1abc123xyz", "2def456uvw"],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "language": "pt",
  "prompt_path": "prompts/pt_prompts.json",
  "created_at": "2026-01-26T15:45:00Z"
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
from gtranscriber.schemas import KGMetadata

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
docker compose --profile kg logs -f gtranscriber-kg

# Check Ollama status
docker compose --profile kg logs ollama
```

### SLURM Logs

```bash
# Monitor job output
tail -f logs/gtranscriber-kg_<jobid>.out
```

## Best Practices

1. **Model Selection**
   - Use `llama3.1:8b` for balanced speed/quality
   - Use `llama3.1:70b` for higher quality extraction
   - Lower temperature (0.3-0.5) for more consistent extraction

2. **Language Settings**
   - Match `KG_LANGUAGE` to your transcription language
   - Use appropriate prompt templates

3. **Graph Merging**
   - Enable for corpus-level analysis
   - Disable if you need per-document graphs only

4. **Workers**
   - Match to available CPU cores / 2
   - KG construction is more memory-intensive than QA

## Troubleshooting

### Empty Graph

```bash
# Check if transcription results exist
ls -la results/*.json

# Verify transcription has content
cat results/<gdrive_id>.json | jq '.transcription_text | length'
```

### Poor Entity Extraction

- Use larger model: `GTRANSCRIBER_KG_MODEL_ID=llama3.1:70b`
- Lower temperature: `GTRANSCRIBER_KG_TEMPERATURE=0.3`
- Verify language setting matches transcription language

### Memory Issues

```bash
# Reduce workers
export GTRANSCRIBER_WORKERS=1

# Use smaller model
export GTRANSCRIBER_KG_MODEL_ID=llama3.2:3b
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

**See also**: [QA Generation](qa-generation.md) | [Evaluation](evaluation.md) | [Configuration](configuration.md)
