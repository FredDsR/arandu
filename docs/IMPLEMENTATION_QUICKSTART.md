# Implementation Quick-Start Guide

**For starting implementation in a new thread or session**

This guide provides everything needed to start implementing the Knowledge Graph Construction Pipeline from scratch.

## 📋 Pre-Implementation Checklist

Before starting implementation, ensure you have:

- [ ] Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (at least sections 1-5)
- [ ] Reviewed [DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md) to understand data models
- [ ] Reviewed [FILE_STRUCTURE.md](implementation/FILE_STRUCTURE.md) to know what to build
- [ ] Have access to the existing codebase
- [ ] Have development environment set up (Python 3.13+, pip/uv)

## 🎯 Implementation Context

### What's Already Done (P1)

The G-Transcriber transcription pipeline is **complete and working**:
- Whisper ASR transcription
- Batch processing with checkpointing
- Google Drive integration
- Docker and SLURM support

**Key existing files**:
- `src/gtranscriber/main.py` - CLI with commands
- `src/gtranscriber/config.py` - Pydantic configuration
- `src/gtranscriber/schemas.py` - Data models (EnrichedRecord, etc.)
- `src/gtranscriber/core/batch.py` - Batch processing pattern
- `docker-compose.yml` - Docker services
- `scripts/slurm/` - SLURM job scripts

### What Needs to Be Built (P2)

Three new capabilities:

1. **QA Generation** - Generate synthetic question-answer pairs from transcriptions
2. **KG Construction** - Build knowledge graphs using AutoSchemaKG
3. **Evaluation** - Measure quality across four metric categories

## 🏗️ Implementation Phases

### Phase 1: Foundation (Week 1) ⬅️ **START HERE**

**Goal**: Infrastructure setup (LLM client, config, schemas, Docker, SLURM)

**Critical Files to Create**:
1. `src/gtranscriber/core/llm_client.py` (~300 lines)
2. Extend `src/gtranscriber/config.py` (add ~100 lines)
3. Extend `src/gtranscriber/schemas.py` (add ~300 lines)
4. Extend `docker-compose.yml` (add ~120 lines)
5. Create SLURM scripts in `scripts/slurm/` (3 files, ~80 lines each)
6. Update `pyproject.toml` dependencies

### Phase 2: QA Generation (Week 2)

**Goal**: Synthetic QA dataset generation

**Critical Files to Create**:
1. `src/gtranscriber/core/qa_generator.py` (~400 lines)
2. `src/gtranscriber/core/qa_batch.py` (~250 lines)
3. Extend `src/gtranscriber/main.py` (add `generate_qa` command)

### Phase 3: KG Construction (Week 3-4)

**Goal**: Knowledge graph construction with AutoSchemaKG

**Critical Files to Create**:
1. `src/gtranscriber/core/kg_builder.py` (~350 lines)
2. `src/gtranscriber/core/kg_batch.py` (~300 lines)
3. Extend `src/gtranscriber/main.py` (add `build_kg` command)

### Phase 4: Evaluation (Week 5)

**Goal**: Metrics and evaluation

**Critical Files to Create**:
1. `src/gtranscriber/core/metrics.py` (~500 lines)
2. `src/gtranscriber/core/evaluator.py` (~350 lines)
3. Extend `src/gtranscriber/main.py` (add `evaluate` command)

### Phase 5: Research (Week 6-8)

**Goal**: Research other frameworks and GraphRAG

**Deliverables**:
- `docs/KG_FRAMEWORKS_COMPARISON.md`
- `docs/GRAPHRAG_INTEGRATION_PLAN.md`

## 🚀 Phase 1 Step-by-Step Instructions

### Step 1.1: Create LLM Client

**File**: `src/gtranscriber/core/llm_client.py`

**What to implement**:

```python
from enum import Enum
from typing import Protocol
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMProvider(Enum):
    """LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class LLMClient(Protocol):
    """Unified LLM client interface"""

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        ...

    def is_available(self) -> bool:
        """Check if provider is available"""
        ...

class OpenAIClient:
    """OpenAI API client"""

    def __init__(self, model_id: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Implement with retry logic
        ...

class AnthropicClient:
    """Anthropic Claude API client"""

    def __init__(self, model_id: str, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model_id = model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Implement with retry logic
        ...

class OllamaClient:
    """Ollama local API client"""

    def __init__(self, model_id: str, base_url: str = "http://localhost:11434"):
        import httpx
        self.client = httpx.Client(base_url=base_url, timeout=60.0)
        self.model_id = model_id

    def is_available(self) -> bool:
        # Health check
        ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Implement with retry logic
        ...

def create_llm_client(provider: LLMProvider, model_id: str, **kwargs) -> LLMClient:
    """Factory function to create LLM client"""
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(model_id, kwargs.get("api_key"))
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(model_id, kwargs.get("api_key"))
    elif provider == LLMProvider.OLLAMA:
        return OllamaClient(model_id, kwargs.get("base_url", "http://localhost:11434"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

**Reference**: [DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md) for understanding what this client will generate

### Step 1.2: Extend Configuration

**File**: `src/gtranscriber/config.py`

**What to add** (after existing settings):

```python
# QA Generation Settings
qa_provider: str = Field(
    default="ollama",
    description="LLM provider for QA generation: openai, anthropic, ollama"
)
qa_model_id: str = Field(
    default="llama3.1:8b",
    description="Model ID for QA generation"
)
qa_ollama_url: str = Field(
    default="http://localhost:11434",
    description="Ollama API base URL"
)
openai_api_key: str | None = Field(
    default=None,
    description="OpenAI API key"
)
anthropic_api_key: str | None = Field(
    default=None,
    description="Anthropic API key"
)
questions_per_document: int = Field(
    default=10,
    ge=1,
    le=50,
    description="Number of QA pairs to generate per document"
)
qa_strategies: list[str] = Field(
    default=["factual", "conceptual"],
    description="Question generation strategies"
)
qa_temperature: float = Field(
    default=0.7,
    ge=0.0,
    le=2.0,
    description="Temperature for QA generation"
)
qa_output_dir: Path = Field(
    default=Path("qa_dataset"),
    description="Output directory for QA datasets"
)

# KG Construction Settings
kg_provider: str = Field(
    default="ollama",
    description="LLM provider for KG construction"
)
kg_model_id: str = Field(
    default="llama3.1:8b",
    description="Model ID for KG construction"
)
kg_merge_graphs: bool = Field(
    default=True,
    description="Merge individual graphs into corpus-level graph"
)
kg_output_format: str = Field(
    default="json",
    pattern="^(json|graphml)$",
    description="Graph export format: json or graphml"
)
kg_output_dir: Path = Field(
    default=Path("knowledge_graphs"),
    description="Output directory for knowledge graphs"
)

# Evaluation Settings
evaluation_metrics: list[str] = Field(
    default=["qa", "entity", "relation", "semantic"],
    description="Metrics to compute during evaluation"
)
embedding_model: str = Field(
    default="sentence-transformers/all-MiniLM-L6-v2",
    description="Sentence transformer model for embeddings"
)
evaluation_output_dir: Path = Field(
    default=Path("evaluation"),
    description="Output directory for evaluation reports"
)

@field_validator("qa_strategies")
def validate_strategies(cls, v: list[str]) -> list[str]:
    """Validate QA strategies"""
    valid = {"factual", "conceptual", "temporal", "entity"}
    for strategy in v:
        if strategy not in valid:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid}")
    return v
```

**Reference**: [CONFIGURATION.md](implementation/CONFIGURATION.md) for complete settings list

### Step 1.3: Extend Schemas

**File**: `src/gtranscriber/schemas.py`

**What to add** (after existing schemas):

```python
# QA Generation Schemas

class QAPair(BaseModel):
    """Single question-answer pair"""
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Ground truth answer")
    context: str = Field(..., description="Source text segment")
    question_type: str = Field(..., description="Question strategy type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Generation confidence")
    start_time: float | None = Field(None, description="Segment start time in seconds")
    end_time: float | None = Field(None, description="Segment end time in seconds")

class QARecord(BaseModel):
    """QA dataset record for one document"""
    source_gdrive_id: str = Field(..., description="Google Drive file ID")
    source_filename: str = Field(..., description="Original filename")
    transcription_text: str = Field(..., description="Full transcription")
    qa_pairs: list[QAPair] = Field(..., description="Generated QA pairs")
    model_id: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider")
    generation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Generation timestamp"
    )
    total_pairs: int = Field(..., description="Total QA pairs generated")

# Knowledge Graph Schemas

class KGNode(BaseModel):
    """Knowledge graph node (entity/event)"""
    id: str = Field(..., description="Unique node ID")
    label: str = Field(..., description="Entity/event text")
    type: str = Field(..., description="Semantic concept type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node properties")
    source_documents: list[str] = Field(default_factory=list, description="Source doc IDs")

class KGEdge(BaseModel):
    """Knowledge graph edge (relation)"""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: str = Field(..., description="Relation type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Edge properties")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")

class KGStatistics(BaseModel):
    """Knowledge graph statistics"""
    total_nodes: int = Field(..., description="Total number of nodes")
    total_edges: int = Field(..., description="Total number of edges")
    node_types: dict[str, int] = Field(..., description="Node type distribution")
    edge_types: dict[str, int] = Field(..., description="Edge type distribution")
    average_degree: float = Field(..., description="Average node degree")
    connected_components: int = Field(..., description="Number of connected components")
    density: float = Field(..., ge=0.0, le=1.0, description="Graph density")

class KGRecord(BaseModel):
    """Complete knowledge graph with metadata"""
    graph_id: str = Field(..., description="Unique graph ID")
    source_documents: list[str] = Field(..., description="Source document IDs")
    creation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    model_id: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider")
    statistics: KGStatistics = Field(..., description="Graph statistics")
    nodes: list[KGNode] = Field(..., description="Graph nodes")
    edges: list[KGEdge] = Field(..., description="Graph edges")

    def to_networkx(self) -> "nx.Graph":
        """Convert to NetworkX graph"""
        import networkx as nx
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.id, label=node.label, type=node.type, **node.properties)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, relation=edge.relation, **edge.properties)
        return G

# Evaluation Schemas

class EntityCoverageResult(BaseModel):
    """Entity coverage metrics"""
    total_entities: int
    unique_entities: int
    entity_density: float
    entity_diversity: float
    entity_type_distribution: dict[str, int]

class RelationMetricsResult(BaseModel):
    """Relation metrics"""
    total_relations: int
    unique_relations: int
    relation_density: float
    relation_diversity: float
    graph_connectivity: dict

class SemanticQualityResult(BaseModel):
    """Semantic quality metrics"""
    coherence_score: float
    information_density: float
    knowledge_coverage: float

class EvaluationReport(BaseModel):
    """Comprehensive evaluation report"""
    dataset_name: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    total_documents: int
    total_qa_pairs: int

    # Metric results
    qa_exact_match: float | None = None
    qa_f1_score: float | None = None
    qa_bleu_score: float | None = None
    entity_coverage: EntityCoverageResult | None = None
    relation_metrics: RelationMetricsResult | None = None
    semantic_quality: SemanticQualityResult | None = None

    # Summary
    overall_score: float
    recommendations: list[str]
```

**Reference**: [DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md) for complete schema specifications

### Step 1.4: Update Dependencies

**File**: `pyproject.toml`

**What to add** to dependencies array:

```toml
dependencies = [
    # ... existing dependencies ...

    # New - LLM Integration
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "httpx>=0.27.0",

    # New - KG Construction
    "atlas-rag>=0.0.5",
    "networkx>=3.1",

    # New - Evaluation
    "scikit-learn>=1.3.0",
    "sentence-transformers>=2.2.0",
    "nltk>=3.8.0",
    "sacrebleu>=2.3.0",
]
```

**Then install**:
```bash
pip install -e .
```

**Reference**: [DEPENDENCIES.md](implementation/DEPENDENCIES.md) for complete dependency list

### Step 1.5: Update Docker Compose

**File**: `docker-compose.yml`

**What to add** (at end of file):

```yaml
  # QA Generation Service
  gtranscriber-qa:
    extends:
      service: gtranscriber
    container_name: gtranscriber-qa-${SLURM_JOB_ID:-local}

    environment:
      - GTRANSCRIBER_QA_PROVIDER=${GTRANSCRIBER_QA_PROVIDER:-ollama}
      - GTRANSCRIBER_QA_MODEL_ID=${GTRANSCRIBER_QA_MODEL_ID:-llama3.1:8b}
      - GTRANSCRIBER_QA_OLLAMA_URL=${GTRANSCRIBER_QA_OLLAMA_URL:-http://host.docker.internal:11434}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - GTRANSCRIBER_WORKERS=${GTRANSCRIBER_WORKERS:-2}

    command: >
      generate-qa
      /app/results
      --output-dir /app/qa_dataset
      --workers ${GTRANSCRIBER_WORKERS:-2}

    volumes:
      - ${GTRANSCRIBER_RESULTS_DIR:-./results}:/app/results:ro
      - ${GTRANSCRIBER_QA_DIR:-./qa_dataset}:/app/qa_dataset:rw

    profiles:
      - qa

  # KG Construction Service
  gtranscriber-kg:
    extends:
      service: gtranscriber
    container_name: gtranscriber-kg-${SLURM_JOB_ID:-local}

    environment:
      - GTRANSCRIBER_KG_PROVIDER=${GTRANSCRIBER_KG_PROVIDER:-ollama}
      - GTRANSCRIBER_KG_MODEL_ID=${GTRANSCRIBER_KG_MODEL_ID:-llama3.1:8b}
      - GTRANSCRIBER_WORKERS=${GTRANSCRIBER_WORKERS:-2}

    command: >
      build-kg
      /app/results
      --output-dir /app/knowledge_graphs
      --workers ${GTRANSCRIBER_WORKERS:-2}

    volumes:
      - ${GTRANSCRIBER_RESULTS_DIR:-./results}:/app/results:ro
      - ${GTRANSCRIBER_KG_DIR:-./knowledge_graphs}:/app/knowledge_graphs:rw

    profiles:
      - kg

  # Evaluation Service
  gtranscriber-eval:
    extends:
      service: gtranscriber
    container_name: gtranscriber-eval-${SLURM_JOB_ID:-local}

    command: >
      evaluate
      /app/qa_dataset
      /app/results
      --kg-path /app/knowledge_graphs/merged_graph.json
      --output /app/evaluation/report.json

    volumes:
      - ${GTRANSCRIBER_QA_DIR:-./qa_dataset}:/app/qa_dataset:ro
      - ${GTRANSCRIBER_RESULTS_DIR:-./results}:/app/results:ro
      - ${GTRANSCRIBER_KG_DIR:-./knowledge_graphs}:/app/knowledge_graphs:ro
      - ${GTRANSCRIBER_EVAL_DIR:-./evaluation}:/app/evaluation:rw

    profiles:
      - evaluate
```

### Step 1.6: Create SLURM Scripts

Create three SLURM job scripts following the pattern of existing scripts.

**File 1**: `scripts/slurm/run_qa_generation.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-qa
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j-qa.out
#SBATCH --error=slurm-%j-qa.err

# Configuration
export GTRANSCRIBER_QA_PROVIDER="${QA_PROVIDER:-ollama}"
export GTRANSCRIBER_QA_MODEL_ID="${QA_MODEL:-llama3.1:8b}"
export WORKERS="${WORKERS:-4}"

# Source common SLURM logic
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"

# Run QA generation
docker compose -f "$COMPOSE_FILE" --profile qa up gtranscriber-qa --abort-on-container-exit
```

**File 2**: `scripts/slurm/run_kg_construction.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-kg
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-kg.out
#SBATCH --error=slurm-%j-kg.err

# Configuration
export GTRANSCRIBER_KG_PROVIDER="${KG_PROVIDER:-ollama}"
export GTRANSCRIBER_KG_MODEL_ID="${KG_MODEL:-llama3.1:8b}"
export WORKERS="${WORKERS:-8}"

# Source common SLURM logic
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"

# Run KG construction
docker compose -f "$COMPOSE_FILE" --profile kg up gtranscriber-kg --abort-on-container-exit
```

**File 3**: `scripts/slurm/run_evaluation.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=gtranscriber-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j-eval.out
#SBATCH --error=slurm-%j-eval.err

# Source common SLURM logic
source "${SLURM_SUBMIT_DIR}/scripts/slurm/job_common.sh"

# Run evaluation
docker compose -f "$COMPOSE_FILE" --profile evaluate up gtranscriber-eval --abort-on-container-exit
```

### Step 1.7: Verification

After completing Phase 1, verify everything works:

```bash
# Test imports
python -c "from gtranscriber.core.llm_client import LLMProvider, create_llm_client; print('LLM client OK')"
python -c "from gtranscriber.config import TranscriberConfig; c = TranscriberConfig(); print(f'Config OK: qa_provider={c.qa_provider}')"
python -c "from gtranscriber.schemas import QAPair, KGRecord, EvaluationReport; print('Schemas OK')"

# Test Docker builds
docker compose build gtranscriber-qa
docker compose build gtranscriber-kg

# Test configuration loading
python -c "from gtranscriber.config import TranscriberConfig; print(TranscriberConfig().model_dump_json(indent=2))"
```

## 📚 Key Documentation References

When implementing, refer to these docs:

1. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Complete plan with all phases
2. **[DATA_SCHEMAS.md](implementation/DATA_SCHEMAS.md)** - All data models in detail
3. **[CONFIGURATION.md](implementation/CONFIGURATION.md)** - Configuration settings
4. **[FILE_STRUCTURE.md](implementation/FILE_STRUCTURE.md)** - What files to create/modify
5. **[DEPENDENCIES.md](implementation/DEPENDENCIES.md)** - All dependencies
6. **[CLI_REFERENCE.md](implementation/CLI_REFERENCE.md)** - CLI commands (for later phases)

## 🔍 Finding Existing Patterns

When implementing new features, look at existing code for patterns:

**For batch processing**:
- Study `src/gtranscriber/core/batch.py` - ProcessPoolExecutor pattern

**For checkpointing**:
- Study `src/gtranscriber/core/checkpoint.py` - Checkpoint manager

**For CLI commands**:
- Study `src/gtranscriber/main.py` - Typer command patterns

**For configuration**:
- Study `src/gtranscriber/config.py` - Pydantic Settings

## ⚠️ Common Pitfalls

1. **Don't skip reading existing code** - Understand the patterns before implementing
2. **Follow existing patterns** - Don't invent new ways to do things
3. **Test incrementally** - Test each component as you build it
4. **Check schemas** - Ensure data models match documentation
5. **Environment variables** - Use GTRANSCRIBER_ prefix consistently

## ✅ Phase 1 Completion Checklist

- [ ] `llm_client.py` created and tested
- [ ] `config.py` extended with all new settings
- [ ] `schemas.py` extended with all new models
- [ ] `docker-compose.yml` updated with new services
- [ ] SLURM scripts created (3 files)
- [ ] `pyproject.toml` updated with dependencies
- [ ] All dependencies installed successfully
- [ ] All imports work without errors
- [ ] Docker services build successfully
- [ ] Configuration loads without errors

## 🚦 Starting a New Thread

If starting implementation in a new thread, say:

> "I'm implementing the Knowledge Graph Construction Pipeline for G-Transcriber. I've reviewed the documentation in docs/IMPLEMENTATION_PLAN.md and docs/IMPLEMENTATION_QUICKSTART.md. I'm starting with Phase 1: Foundation. Please help me implement [specific component, e.g., 'the LLM client in llm_client.py']."

Provide this context:
- Which phase you're working on
- Which specific component
- Reference the documentation files
- Mention any blockers or questions

---

**Document Version**: 1.0
**Last Updated**: 2026-01-14
**Ready for Implementation**: ✅ Yes
