# Data Schemas Documentation

This document provides complete specifications for all data schemas used in the Knowledge Graph Construction Pipeline.

## Table of Contents

1. [Input Schemas](#input-schemas)
2. [QA Generation Schemas](#qa-generation-schemas)
3. [PEC QA Generation Schemas](#pec-qa-generation-schemas)
4. [Knowledge Graph Schemas](#knowledge-graph-schemas)
5. [Evaluation Schemas](#evaluation-schemas)
6. [Schema Relationships](#schema-relationships)

---

## Input Schemas

### EnrichedRecord

Represents a transcription record from the P1 pipeline. This is the primary input for QA generation and KG construction.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gdrive_id` | `str` | Yes | Google Drive ID of original media file |
| `filename` | `str` | Yes | Original filename |
| `transcription_text` | `str` | Yes | Full transcription text |
| `segments` | `list[Segment]` | Yes | Timestamped segments |
| `detected_language` | `str` | Yes | Detected language code (ISO 639-1, e.g., "pt") |
| `metadata` | `dict` | Yes | Additional metadata including `lang` field |
| `created_at` | `datetime` | Yes | When transcription was created |

**Language Metadata**:

The `metadata.lang` field is **critical** for multilingual KG construction. AutoSchemaKG uses this field to route extraction to the correct language-specific prompts.

**Example**:
```json
{
  "gdrive_id": "1abc123xyz",
  "filename": "entrevista_enchente_2023.mp3",
  "transcription_text": "A enchente foi causada por chuvas intensas...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "A enchente foi causada por chuvas intensas"
    }
  ],
  "detected_language": "pt",
  "metadata": {
    "lang": "pt",
    "duration_seconds": 3600,
    "sample_rate": 16000
  },
  "created_at": "2026-01-14T10:00:00Z"
}
```

**Important**: When processing Portuguese transcriptions:
- `detected_language` should be `"pt"`
- `metadata.lang` **must** be set to `"pt"` for AutoSchemaKG language routing
- The transcription pipeline should automatically populate these fields

**Python Implementation**:
```python
from datetime import datetime
from pydantic import BaseModel, Field

class Segment(BaseModel):
    """Timestamped transcription segment."""
    start: float
    end: float
    text: str

class EnrichedRecord(BaseModel):
    """Transcription record from P1 pipeline."""
    gdrive_id: str
    filename: str
    transcription_text: str
    segments: list[Segment]
    detected_language: str = Field(description="ISO 639-1 language code")
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def ensure_language_metadata(self) -> None:
        """Ensure metadata.lang is set for AutoSchemaKG compatibility."""
        if "lang" not in self.metadata:
            self.metadata["lang"] = self.detected_language
```

---

## QA Generation Schemas

### QAPair

Represents a single question-answer pair generated from a transcription.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | `str` | Yes | The generated question |
| `answer` | `str` | Yes | The ground truth answer (extractive from context) |
| `context` | `str` | Yes | Source text segment from which QA was generated |
| `question_type` | `Literal["factual", "conceptual", "temporal", "entity"]` | Yes | Strategy used for question generation |
| `confidence` | `float` | Yes | Generation confidence score (0.0-1.0) |
| `start_time` | `float \| None` | No | Segment start time in seconds (if available) |
| `end_time` | `float \| None` | No | Segment end time in seconds (if available) |

**Example**:
```json
{
  "question": "What caused the flooding in the region?",
  "answer": "Heavy rainfall combined with poor drainage infrastructure",
  "context": "The flooding was caused by heavy rainfall combined with poor drainage infrastructure. Many residents were evacuated.",
  "question_type": "factual",
  "confidence": 0.92,
  "start_time": 45.3,
  "end_time": 52.1
}
```

**Validation Rules**:
- `confidence` must be between 0.0 and 1.0
- `answer` must be a substring of or semantically contained in `context`
- `question_type` must be one of: "factual", "conceptual", "temporal", "entity"
- If `start_time` is provided, `end_time` must also be provided
- `start_time` < `end_time` when both are provided

**Python Implementation**:
```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class QAPair(BaseModel):
    """Represents a single question-answer pair generated from a transcription."""
    question: str
    answer: str
    context: str
    question_type: Literal["factual", "conceptual", "temporal", "entity"]
    confidence: float = Field(ge=0.0, le=1.0)
    start_time: float | None = None
    end_time: float | None = None

    @field_validator("answer")
    @classmethod
    def answer_must_be_extractive(cls, v: str, info) -> str:
        """Validate that answer is extractive from context."""
        context = info.data.get("context", "")
        if context and v.lower() not in context.lower():
            raise ValueError("Answer must be extractive from context")
        return v

    @model_validator(mode="after")
    def validate_time_range(self) -> "QAPair":
        """Validate temporal constraints."""
        if self.start_time is not None and self.end_time is None:
            raise ValueError("end_time required when start_time is provided")
        if self.end_time is not None and self.start_time is None:
            raise ValueError("start_time required when end_time is provided")
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("start_time must be less than end_time")
        return self
```

### QARecord

Represents the complete QA dataset for a single transcription document.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_gdrive_id` | `str` | Yes | Google Drive ID of original media file |
| `source_filename` | `str` | Yes | Original filename |
| `transcription_text` | `str` | Yes | Full transcription text |
| `qa_pairs` | `list[QAPair]` | Yes | List of generated QA pairs |
| `model_id` | `str` | Yes | LLM model used for generation (e.g., "llama3.1:8b") |
| `provider` | `Literal["openai", "ollama", "custom"]` | Yes | LLM provider used |
| `generation_timestamp` | `datetime` | Yes | When QA pairs were generated |
| `total_pairs` | `int` | Yes | Total number of QA pairs generated |

**Example**:
```json
{
  "source_gdrive_id": "1abc123xyz",
  "source_filename": "interview_2023_flood.mp3",
  "transcription_text": "The flooding was caused by heavy rainfall...",
  "qa_pairs": [
    {
      "question": "What caused the flooding?",
      "answer": "Heavy rainfall combined with poor drainage",
      "context": "The flooding was caused by heavy rainfall...",
      "question_type": "factual",
      "confidence": 0.92,
      "start_time": 45.3,
      "end_time": 52.1
    }
  ],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "generation_timestamp": "2026-01-14T10:30:00Z",
  "total_pairs": 12
}
```

**Validation Rules**:
- `total_pairs` must equal `len(qa_pairs)`
- `provider` must be one of: "openai", "ollama", "custom"
- All `qa_pairs` must pass QAPair validation

**Python Implementation**:
```python
from datetime import datetime
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class QARecord(BaseModel):
    """Represents the complete QA dataset for a single transcription document."""
    source_gdrive_id: str
    source_filename: str
    transcription_text: str
    qa_pairs: list[QAPair]
    model_id: str
    provider: Literal["openai", "ollama", "custom"]
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    total_pairs: int

    @model_validator(mode="after")
    def validate_total_pairs(self) -> "QARecord":
        """Validate that total_pairs matches actual count."""
        if self.total_pairs != len(self.qa_pairs):
            raise ValueError(
                f"total_pairs ({self.total_pairs}) must equal len(qa_pairs) ({len(self.qa_pairs)})"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save QA record to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QARecord":
        """Load QA record from JSON file."""
        return cls.model_validate_json(Path(path).read_text())
```

---

## PEC QA Generation Schemas

The PEC (Pipeline de Elicitação Cognitiva) extends the standard QA schemas with cognitive scaffolding based on Bloom's Taxonomy.

### QAPairPEC

Extends QAPair with Bloom taxonomy levels and reasoning traces for cognitively-calibrated question generation.

**Fields** (in addition to QAPair fields):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bloom_level` | `Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]` | Yes | Bloom's taxonomy cognitive level |
| `reasoning_trace` | `str \| None` | No | Logical connection chain for higher-level questions |
| `is_multi_hop` | `bool` | No | Whether question requires connecting distant information |
| `hop_count` | `int \| None` | No | Number of reasoning hops (1-5 when is_multi_hop=True) |
| `tacit_inference` | `str \| None` | No | Explanation of implicit domain knowledge surfaced |

**Example**:
```json
{
  "question": "Por que o pescador guarda o barco quando o rio sobe?",
  "answer": "Para evitar perda do equipamento devido ao risco de enchente",
  "context": "Se o rio sobe rápido, guardo o barco para evitar perda",
  "question_type": "conceptual",
  "confidence": 0.88,
  "bloom_level": "analyze",
  "reasoning_trace": "Fato: rio sobe → Ação: guardar barco → Razão: evitar perda",
  "is_multi_hop": false,
  "hop_count": null,
  "tacit_inference": "Subida rápida do rio indica risco iminente de enchente"
}
```

**Bloom Level Semantics**:
- `remember`: Recall explicit facts from text
- `understand`: Explain and interpret concepts
- `apply`: Use knowledge in new situations
- `analyze`: Identify relationships and patterns
- `evaluate`: Make judgments and justify decisions
- `create`: Propose solutions or create something new

**Python Implementation**:
```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class QAPairPEC(BaseModel):
    """QA pair with Bloom's Taxonomy cognitive scaffolding."""
    question: str
    answer: str
    context: str
    question_type: Literal["factual", "conceptual", "temporal", "entity"]
    confidence: float = Field(ge=0.0, le=1.0)
    bloom_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
    reasoning_trace: str | None = None
    is_multi_hop: bool = False
    hop_count: int | None = Field(default=None, ge=1, le=5)
    tacit_inference: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @field_validator("hop_count", mode="before")
    @classmethod
    def validate_hop_count(cls, v, info):
        """Validate hop_count is only set when is_multi_hop is True."""
        is_multi_hop = info.data.get("is_multi_hop", False)
        if v is not None and not is_multi_hop:
            return None
        if is_multi_hop and v is not None and not (1 <= v <= 5):
            return 2  # Default to 2 hops for multi-hop questions
        return v
```

### ValidationScore

Represents LLM-as-a-Judge validation scores for a QA pair.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `faithfulness` | `float` | Yes | Answer grounded in context (0.0-1.0) |
| `bloom_calibration` | `float` | Yes | Question matches declared cognitive level (0.0-1.0) |
| `informativeness` | `float` | Yes | Answer reveals non-obvious knowledge (0.0-1.0) |
| `overall_score` | `float` | Yes | Weighted average of criteria |
| `judge_rationale` | `str \| None` | No | LLM's explanation of the scores |

**Scoring Weights** (default):
- Faithfulness: 40%
- Bloom Calibration: 30%
- Informativeness: 30%

**Example**:
```json
{
  "faithfulness": 0.85,
  "bloom_calibration": 0.78,
  "informativeness": 0.72,
  "overall_score": 0.79,
  "judge_rationale": "Answer well grounded, question requires appropriate analysis level."
}
```

**Python Implementation**:
```python
from pydantic import BaseModel, Field

class ValidationScore(BaseModel):
    """LLM-as-a-Judge validation scores."""
    faithfulness: float = Field(ge=0.0, le=1.0)
    bloom_calibration: float = Field(ge=0.0, le=1.0)
    informativeness: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    judge_rationale: str | None = None
```

### QAPairValidated

Extends QAPairPEC with validation results.

**Fields** (in addition to QAPairPEC fields):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validation` | `ValidationScore \| None` | No | Validation scores from LLM-as-a-Judge |
| `is_valid` | `bool` | Yes | Whether pair passes validation threshold |

**Example**:
```json
{
  "question": "Por que o pescador guarda o barco?",
  "answer": "Para evitar perda durante enchentes.",
  "context": "Se o rio sobe rápido, guardo o barco para evitar perda.",
  "question_type": "conceptual",
  "confidence": 0.9,
  "bloom_level": "analyze",
  "validation": {
    "faithfulness": 0.9,
    "bloom_calibration": 0.8,
    "informativeness": 0.7,
    "overall_score": 0.81
  },
  "is_valid": true
}
```

### QARecordPEC

Complete PEC QA dataset for a single transcription, extending QARecord with cognitive scaffolding metadata.

**Fields** (in addition to QARecord fields):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `qa_pairs` | `list[QAPairPEC]` | Yes | PEC-enhanced QA pairs |
| `bloom_distribution` | `dict[str, int]` | Yes | Count of pairs per Bloom level |
| `validated_pairs` | `int \| None` | No | Number of pairs passing validation |
| `validation_summary` | `ValidationSummary \| None` | No | Aggregate validation statistics |
| `pec_version` | `str` | Yes | PEC pipeline version |

**Example**:
```json
{
  "source_gdrive_id": "1abc123xyz",
  "source_filename": "interview_2023.mp3",
  "transcription_text": "O pescador contou que quando o rio sobe...",
  "qa_pairs": [...],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "generation_timestamp": "2026-02-03T10:30:00Z",
  "total_pairs": 12,
  "bloom_distribution": {
    "remember": 3,
    "understand": 4,
    "analyze": 3,
    "evaluate": 2
  },
  "validated_pairs": 10,
  "validation_summary": {
    "avg_faithfulness": 0.85,
    "avg_bloom_calibration": 0.78,
    "avg_informativeness": 0.72,
    "avg_overall": 0.79
  },
  "pec_version": "1.0"
}
```

**JSONL Export**:

QARecordPEC supports JSONL export for KGQA training compatibility:

```python
record = QARecordPEC.load("pec_dataset/1abc123xyz_pec_qa.json")

# Export to file
record.to_jsonl("output.jsonl")

# Get as string
jsonl_content = record.to_jsonl()
```

Each line in the JSONL contains one QA pair with all PEC fields:
```json
{"question": "O que aconteceu?", "answer": "...", "context": "...", "bloom_level": "remember", "confidence": 0.92}
{"question": "Por que isso aconteceu?", "answer": "...", "context": "...", "bloom_level": "analyze", "reasoning_trace": "..."}
```

**Python Implementation**:
```python
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

class ValidationSummary(BaseModel):
    """Aggregate validation statistics."""
    avg_faithfulness: float = Field(ge=0.0, le=1.0)
    avg_bloom_calibration: float = Field(ge=0.0, le=1.0)
    avg_informativeness: float = Field(ge=0.0, le=1.0)
    avg_overall: float = Field(ge=0.0, le=1.0)

class QARecordPEC(BaseModel):
    """Complete PEC QA dataset for a single transcription."""
    source_gdrive_id: str
    source_filename: str
    transcription_text: str
    qa_pairs: list[QAPairPEC]
    model_id: str
    provider: str
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    total_pairs: int
    bloom_distribution: dict[str, int]
    validated_pairs: int | None = None
    validation_summary: ValidationSummary | None = None
    pec_version: str = "1.0"

    def to_jsonl(self, path: str | Path | None = None) -> str | None:
        """Export QA pairs to JSONL format."""
        lines = [pair.model_dump_json() for pair in self.qa_pairs]
        jsonl_content = "\n".join(lines)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(jsonl_content + "\n" if jsonl_content else "")
            return None

        return jsonl_content

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QARecordPEC":
        return cls.model_validate_json(Path(path).read_text())
```

---

## Knowledge Graph Schemas

### Design Decision: GraphML-First Approach

Instead of defining custom schema classes (KGNode, KGEdge, etc.), we use **AutoSchemaKG's native GraphML output** directly with NetworkX. This provides:

- **Simplicity**: No custom conversion code to maintain
- **Standard format**: GraphML is widely supported
- **Direct NetworkX compatibility**: Load with `nx.read_graphml()`
- **AutoSchemaKG validation**: Triples are validated during extraction

### AutoSchemaKG Pipeline

The KG construction uses AutoSchemaKG's extraction pipeline:

```python
from atlas_rag.kg_construction import KGExtractor

# Initialize extractor
kg_extractor = KGExtractor(config)

# 1. Extract triples from text
kg_extractor.run_extraction()

# 2. Convert to CSV format
kg_extractor.convert_json_to_csv()

# 3. Run schema induction (conceptualization)
kg_extractor.generate_concept_csv_temp()

# 4. Create concept CSV
kg_extractor.create_concept_csv()

# 5. Export to GraphML (NetworkX-compatible)
kg_extractor.convert_to_graphml()
```

### Loading into NetworkX

```python
import networkx as nx

# Load GraphML directly into NetworkX
graph = nx.read_graphml("knowledge_graphs/corpus_graph.graphml")

# Access nodes and edges with all attributes preserved
for node, attrs in graph.nodes(data=True):
    print(f"Node: {node}, Type: {attrs.get('type')}, Label: {attrs.get('label')}")

for u, v, attrs in graph.edges(data=True):
    print(f"Edge: {u} -> {v}, Relation: {attrs.get('relation')}")

# Compute statistics using NetworkX
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
print(f"Density: {nx.density(graph)}")
print(f"Connected components: {nx.number_connected_components(graph)}")
```

### KGMetadata (Optional)

For provenance tracking, a lightweight metadata sidecar file can be stored alongside the GraphML:

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `graph_id` | `str` | Yes | Unique graph identifier |
| `source_documents` | `list[str]` | Yes | List of source document IDs |
| `created_at` | `datetime` | Yes | When graph was created |
| `model_id` | `str` | Yes | LLM model used for extraction |
| `provider` | `str` | Yes | LLM provider |
| `language` | `str` | Yes | Language code used for extraction (ISO 639-1) |
| `prompt_path` | `str` | Yes | Path to prompt template file used |

**Example** (`corpus_graph_metadata.json`):
```json
{
  "graph_id": "corpus_merged_2026_01_14",
  "source_documents": ["1abc123xyz", "2def456uvw", "3ghi789rst"],
  "created_at": "2026-01-14T15:45:00Z",
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "language": "pt",
  "prompt_path": "prompts/pt_prompts.json"
}
```

**Python Implementation**:
```python
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

class KGMetadata(BaseModel):
    """Lightweight metadata for provenance tracking."""
    graph_id: str
    source_documents: list[str]
    model_id: str
    provider: str
    language: str = Field(default="pt", description="ISO 639-1 language code")
    prompt_path: str = Field(default="prompts/pt_prompts.json")
    created_at: datetime = Field(default_factory=datetime.now)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KGMetadata":
        return cls.model_validate_json(Path(path).read_text())
```

### Node Types (from AutoSchemaKG)

AutoSchemaKG extracts nodes with these semantic types (stored as node attributes in GraphML):

- `PERSON` - People, groups of people
- `LOCATION` - Geographic locations
- `ORGANIZATION` - Companies, institutions
- `EVENT` - Occurrences, incidents
- `DATE` - Temporal references
- `CONCEPT` - Abstract ideas
- `OBJECT` - Physical objects
- Custom types from dynamic schema induction

### Common Relations

Relations are stored as edge attributes in GraphML:

- `LOCATED_IN` - Spatial containment
- `CAUSED_BY` - Causal relationship
- `AFFECTED_BY` - Impact relationship
- `OCCURRED_IN` - Temporal/spatial occurrence
- `BELONGS_TO` - Membership
- Custom relations from dynamic schema induction

---

## Evaluation Schemas

### EntityCoverageResult

Represents entity coverage metrics.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_entities` | `int` | Yes | Total entities extracted |
| `unique_entities` | `int` | Yes | Number of unique entities |
| `entity_density` | `float` | Yes | Entities per 100 tokens |
| `entity_type_distribution` | `dict[str, int]` | Yes | Count by entity type |
| `entity_diversity` | `float` | Computed | unique_entities / total_entities (auto-calculated) |

**Example**:
```json
{
  "total_entities": 4823,
  "unique_entities": 1523,
  "entity_density": 12.4,
  "entity_diversity": 0.316,
  "entity_type_distribution": {
    "PERSON": 342,
    "LOCATION": 189,
    "EVENT": 423,
    "ORGANIZATION": 156
  }
}
```

**Interpretation**:
- **Entity Density**: Higher values indicate more entity mentions per text unit
- **Entity Diversity**: Values closer to 1.0 indicate less repetition
- Ideal diversity: 0.3-0.6 (some repetition for coherence, but not excessive)

**Python Implementation**:
```python
from pydantic import BaseModel, Field, computed_field

class EntityCoverageResult(BaseModel):
    """Entity coverage metrics."""
    total_entities: int = Field(ge=0)
    unique_entities: int = Field(ge=0)
    entity_density: float = Field(ge=0.0, description="Entities per 100 tokens")
    entity_type_distribution: dict[str, int]

    @computed_field
    @property
    def entity_diversity(self) -> float:
        """Compute entity diversity as unique/total ratio."""
        if self.total_entities == 0:
            return 0.0
        return self.unique_entities / self.total_entities
```

### RelationMetricsResult

Represents relation metrics.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_relations` | `int` | Yes | Total relations extracted |
| `unique_relations` | `int` | Yes | Number of unique relations |
| `relation_density` | `float` | Yes | Relations per entity |
| `graph_connectivity` | `GraphConnectivity` | Yes | Nested connectivity metrics model |
| `relation_diversity` | `float` | Computed | unique_relations / total_relations (auto-calculated) |

**Example**:
```json
{
  "total_relations": 3847,
  "unique_relations": 1204,
  "relation_density": 2.53,
  "relation_diversity": 0.313,
  "graph_connectivity": {
    "average_degree": 5.05,
    "connected_components": 3,
    "largest_component_size": 1421,
    "density": 0.0033
  }
}
```

**Interpretation**:
- **Relation Density**: Typical range 1.5-3.0 (depends on domain)
- **Connectivity**: Fewer components = more cohesive knowledge
- **Density**: Values close to 0 are normal for large graphs

**Python Implementation**:
```python
from pydantic import BaseModel, Field, computed_field

class GraphConnectivity(BaseModel):
    """Graph connectivity metrics."""
    average_degree: float = Field(ge=0.0)
    connected_components: int = Field(ge=1)
    largest_component_size: int = Field(ge=0)
    density: float = Field(ge=0.0, le=1.0)

class RelationMetricsResult(BaseModel):
    """Relation density and connectivity metrics."""
    total_relations: int = Field(ge=0)
    unique_relations: int = Field(ge=0)
    relation_density: float = Field(ge=0.0, description="Relations per entity")
    graph_connectivity: GraphConnectivity

    @computed_field
    @property
    def relation_diversity(self) -> float:
        """Compute relation diversity as unique/total ratio."""
        if self.total_relations == 0:
            return 0.0
        return self.unique_relations / self.total_relations
```

### SemanticQualityResult

Represents semantic quality metrics.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `coherence_score` | `float` | Yes | Semantic coherence (0.0-1.0) |
| `information_density` | `float` | Yes | (Entities + Relations) / text_length |
| `knowledge_coverage` | `float` | Yes | Entities covered by QA pairs (0.0-1.0) |

**Example**:
```json
{
  "coherence_score": 0.78,
  "information_density": 0.042,
  "knowledge_coverage": 0.64
}
```

**Interpretation**:
- **Coherence**: >0.7 is good, >0.8 is excellent
- **Information Density**: Higher values indicate more structured knowledge
- **Knowledge Coverage**: >0.5 means QA pairs cover majority of entities

**Python Implementation**:
```python
from pydantic import BaseModel, Field

class SemanticQualityResult(BaseModel):
    """Semantic quality metrics."""
    coherence_score: float = Field(ge=0.0, le=1.0)
    information_density: float = Field(ge=0.0)
    knowledge_coverage: float = Field(ge=0.0, le=1.0)
```

### EvaluationReport

Represents a comprehensive evaluation report.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_name` | `str` | Yes | Name/identifier of evaluated dataset |
| `evaluation_timestamp` | `datetime` | Yes | When evaluation was run |
| `total_documents` | `int` | Yes | Number of documents evaluated |
| `total_qa_pairs` | `int` | Yes | Total QA pairs in dataset |
| `qa_exact_match` | `float \| None` | No | Exact match score (0.0-1.0) |
| `qa_f1_score` | `float \| None` | No | F1 score (0.0-1.0) |
| `qa_bleu_score` | `float \| None` | No | BLEU score (0.0-100.0) |
| `entity_coverage` | `EntityCoverageResult \| None` | No | Entity coverage metrics |
| `relation_metrics` | `RelationMetricsResult \| None` | No | Relation metrics |
| `semantic_quality` | `SemanticQualityResult \| None` | No | Semantic quality metrics |
| `recommendations` | `list[str]` | Yes | Improvement suggestions |
| `overall_score` | `float` | Computed | Weighted average score (auto-calculated, 0.0-1.0) |

**Example**:
```json
{
  "dataset_name": "etno_kgc_evaluation_2026_01_14",
  "evaluation_timestamp": "2026-01-14T18:30:00Z",
  "total_documents": 187,
  "total_qa_pairs": 2244,
  "qa_exact_match": 0.68,
  "qa_f1_score": 0.79,
  "qa_bleu_score": 72.3,
  "entity_coverage": { ... },
  "relation_metrics": { ... },
  "semantic_quality": { ... },
  "overall_score": 0.73,
  "recommendations": [
    "Increase entity diversity by reducing repetitive mentions",
    "Improve relation extraction for temporal relations",
    "Consider using more specific question strategies for conceptual QA"
  ]
}
```

**Overall Score Calculation**:
```python
overall_score = (
    0.3 * qa_f1_score +
    0.2 * entity_coverage.entity_diversity +
    0.2 * relation_metrics.relation_density / 3.0 +  # normalized
    0.3 * semantic_quality.coherence_score
)
```

**Python Implementation**:
```python
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, computed_field

class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""
    dataset_name: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    total_documents: int = Field(ge=0)
    total_qa_pairs: int = Field(ge=0)

    # QA metrics
    qa_exact_match: float | None = Field(default=None, ge=0.0, le=1.0)
    qa_f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    qa_bleu_score: float | None = Field(default=None, ge=0.0, le=100.0)

    # Component metrics
    entity_coverage: EntityCoverageResult | None = None
    relation_metrics: RelationMetricsResult | None = None
    semantic_quality: SemanticQualityResult | None = None

    # Summary
    recommendations: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def overall_score(self) -> float:
        """Compute weighted overall score."""
        components = []
        weights = []

        if self.qa_f1_score is not None:
            components.append(self.qa_f1_score)
            weights.append(0.3)

        if self.entity_coverage is not None:
            components.append(self.entity_coverage.entity_diversity)
            weights.append(0.2)

        if self.relation_metrics is not None:
            # Normalize relation_density (assuming max ~3.0)
            components.append(min(self.relation_metrics.relation_density / 3.0, 1.0))
            weights.append(0.2)

        if self.semantic_quality is not None:
            components.append(self.semantic_quality.coherence_score)
            weights.append(0.3)

        if not components:
            return 0.0

        # Normalize weights and compute weighted average
        total_weight = sum(weights)
        return sum(c * w for c, w in zip(components, weights)) / total_weight

    def save(self, path: str | Path) -> None:
        """Save evaluation report to JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        """Load evaluation report from JSON file."""
        return cls.model_validate_json(Path(path).read_text())
```

---

## Schema Relationships

```mermaid
graph TD
    A[EnrichedRecord] --> B[QAGenerator]
    B --> C[QARecord]
    C --> D[QAPair]

    A --> P[PECQAGenerator]
    P --> Q[QARecordPEC]
    Q --> R[QAPairPEC]
    R --> S[ValidationScore]
    R --> T[QAPairValidated]

    A --> E[AutoSchemaKG]
    E --> F[GraphML File]
    E --> G[KGMetadata]

    F --> H[NetworkX Graph]

    C --> I[Evaluator]
    Q --> I
    H --> I
    A --> I
    I --> J[EvaluationReport]
    J --> K[EntityCoverageResult]
    J --> L[RelationMetricsResult]
    J --> M[SemanticQualityResult]
```

## File Format Examples

### QARecord JSON File

**Filename**: `qa_<gdrive_id>.json`

```json
{
  "source_gdrive_id": "1abc123xyz",
  "source_filename": "interview_2023_flood.mp3",
  "transcription_text": "Full transcription here...",
  "qa_pairs": [
    {
      "question": "What caused the flooding?",
      "answer": "Heavy rainfall",
      "context": "The flooding was caused by heavy rainfall...",
      "question_type": "factual",
      "confidence": 0.92,
      "start_time": 45.3,
      "end_time": 52.1
    }
  ],
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "generation_timestamp": "2026-01-14T10:30:00Z",
  "total_pairs": 12
}
```

### Knowledge Graph Files

Knowledge graphs are stored as **GraphML files** (AutoSchemaKG native output) with optional metadata sidecars.

**Graph file**: `<graph_id>.graphml`
**Metadata file**: `<graph_id>_metadata.json`

**Example Directory Structure**:
```
knowledge_graphs/
├── corpus_graph.graphml           # Main graph (NetworkX-compatible)
├── corpus_graph_metadata.json     # Provenance metadata
├── individual/                    # Per-document graphs (optional)
│   ├── 1abc123xyz.graphml
│   └── 2def456uvw.graphml
└── checkpoints/
    └── kg_checkpoint.json
```

**Metadata Example** (`corpus_graph_metadata.json`):
```json
{
  "graph_id": "corpus_merged",
  "source_documents": ["1abc123xyz", "2def456uvw"],
  "created_at": "2026-01-14T15:45:00Z",
  "model_id": "llama3.1:8b",
  "provider": "ollama",
  "language": "pt",
  "prompt_path": "prompts/pt_prompts.json"
}
```

**Loading and Computing Statistics**:
```python
import networkx as nx

# Load graph
graph = nx.read_graphml("knowledge_graphs/corpus_graph.graphml")

# Compute statistics on demand
stats = {
    "total_nodes": graph.number_of_nodes(),
    "total_edges": graph.number_of_edges(),
    "density": nx.density(graph),
    "connected_components": nx.number_connected_components(graph),
    "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
}
```

### EvaluationReport JSON File

**Filename**: `evaluation_report_<timestamp>.json`

```json
{
  "dataset_name": "etno_kgc_evaluation",
  "evaluation_timestamp": "2026-01-14T18:30:00Z",
  "total_documents": 187,
  "total_qa_pairs": 2244,
  "qa_exact_match": 0.68,
  "qa_f1_score": 0.79,
  "qa_bleu_score": 72.3,
  "entity_coverage": {
    "total_entities": 4823,
    "unique_entities": 1523,
    "entity_density": 12.4,
    "entity_diversity": 0.316,
    "entity_type_distribution": {"PERSON": 342}
  },
  "relation_metrics": {
    "total_relations": 3847,
    "unique_relations": 1204,
    "relation_density": 2.53,
    "relation_diversity": 0.313,
    "graph_connectivity": {
      "average_degree": 5.05,
      "connected_components": 3
    }
  },
  "semantic_quality": {
    "coherence_score": 0.78,
    "information_density": 0.042,
    "knowledge_coverage": 0.64
  },
  "overall_score": 0.73,
  "recommendations": [
    "Increase entity diversity",
    "Improve temporal relation extraction"
  ]
}
```

---

**Document Version**: 1.2
**Last Updated**: 2026-02-03
