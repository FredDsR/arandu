# Methodology: Tacit Knowledge Elicitation from Ethnographic Interviews Using LLM-Powered Pipelines

## 1. Overview

This document describes the methodological framework for **G-Transcriber**, a composable pipeline designed to elicit and structure tacit knowledge from ethnographic audio/video interviews with riverine communities affected by critical climate events in southern Brazil.

The methodology integrates four sequential stages: **(1)** automated speech recognition and transcription, **(2)** cognitively-scaffolded question-answer generation grounded in Bloom's Taxonomy, **(3)** knowledge graph construction via entity and relation extraction, and **(4)** multi-dimensional quality evaluation. Each stage is designed as an independent, resumable module that feeds structured output into the next, enabling reproducibility and iterative refinement.

---

## 2. Pipeline Architecture

The following diagram presents the end-to-end methodology, from raw ethnographic media to structured knowledge representations and evaluation metrics.

```mermaid
flowchart TB
    %% ── Styling ──────────────────────────────────────────────
    classDef input fill:#4a6fa5,stroke:#2d4a7a,color:#fff,rx:12,ry:12
    classDef phase1 fill:#48a9a6,stroke:#2d7a78,color:#fff,rx:8,ry:8
    classDef phase2 fill:#d4a373,stroke:#a67c52,color:#fff,rx:8,ry:8
    classDef phase3 fill:#7b68ae,stroke:#5a4d8a,color:#fff,rx:8,ry:8
    classDef phase4 fill:#c44569,stroke:#943349,color:#fff,rx:8,ry:8
    classDef output fill:#2d6a4f,stroke:#1b4332,color:#fff,rx:12,ry:12
    classDef decision fill:#e9c46a,stroke:#c9a227,color:#333,rx:4,ry:4
    classDef artifact fill:#f4f1de,stroke:#bbb,color:#333,rx:6,ry:6

    %% ── Input ────────────────────────────────────────────────
    IN["Ethnographic Audio/Video Archive<br/><i>Google Drive Catalog (CSV)</i>"]:::input

    %% ── Phase 1: Transcription ──────────────────────────────
    subgraph P1["<b>Phase 1 &mdash; Automated Transcription</b>"]
        direction TB
        DL["Download from<br/>Google Drive"]:::phase1
        FF["Extract Audio<br/><i>FFmpeg / FFprobe</i>"]:::phase1
        WH["Whisper ASR<br/><i>whisper-large-v3</i>"]:::phase1
        QV{"Quality<br/>Validation"}:::decision
        ER["EnrichedRecord<br/><i>JSON</i>"]:::artifact
        SK["Skipped<br/><i>is_valid = false</i>"]:::artifact

        DL --> FF --> WH --> QV
        QV -- "score >= 0.5" --> ER
        QV -- "score < 0.5" --> SK
    end

    %% ── Phase 2: CEP QA Generation ─────────────────────────
    subgraph P2["<b>Phase 2 &mdash; Cognitive Elicitation Pipeline (CEP)</b>"]
        direction TB
        subgraph M1["Module I &mdash; Bloom Scaffolding"]
            BL["Bloom Level<br/>Distribution"]:::phase2
            SG["Scaffolded QA<br/>Generation"]:::phase2
            BL --> SG
        end
        subgraph M2["Module II &mdash; Reasoning & Grounding"]
            RT["Reasoning Trace<br/>Generation"]:::phase2
            MH["Multi-hop<br/>Detection"]:::phase2
            TI["Tacit Inference<br/>Extraction"]:::phase2
            RT --> MH --> TI
        end
        subgraph M3["Module III &mdash; LLM-as-a-Judge"]
            FA["Faithfulness<br/><i>40%</i>"]:::phase2
            BC["Bloom Calibration<br/><i>30%</i>"]:::phase2
            IF["Informativeness<br/><i>30%</i>"]:::phase2
            VD{"Threshold<br/>score >= 0.6"}:::decision
            FA & BC & IF --> VD
        end
        QR["QARecordCEP<br/><i>JSON / JSONL</i>"]:::artifact

        M1 --> M2 --> M3
        VD -- "valid" --> QR
    end

    %% ── Phase 3: Knowledge Graph Construction ───────────────
    subgraph P3["<b>Phase 3 &mdash; Knowledge Graph Construction</b>"]
        direction TB
        TE["Triple Extraction<br/><i>AutoSchemaKG</i>"]:::phase3
        SI["Schema Induction<br/><i>Conceptualization</i>"]:::phase3
        GM["Graph Merging<br/><i>Corpus-level</i>"]:::phase3
        GR["GraphML Output<br/><i>NetworkX</i>"]:::artifact

        TE --> SI --> GM --> GR
    end

    %% ── Phase 4: Evaluation ─────────────────────────────────
    subgraph P4["<b>Phase 4 &mdash; Multi-Dimensional Evaluation</b>"]
        direction TB
        QM["QA Metrics<br/><i>EM, F1, BLEU</i>"]:::phase4
        EC["Entity Coverage<br/><i>Density, Diversity</i>"]:::phase4
        RM["Relation Metrics<br/><i>Connectivity, Degree</i>"]:::phase4
        SQ["Semantic Quality<br/><i>Coherence, Coverage</i>"]:::phase4
        RP["EvaluationReport<br/><i>JSON</i>"]:::artifact

        QM & EC & RM & SQ --> RP
    end

    %% ── Connections ─────────────────────────────────────────
    IN --> P1
    ER --> P2
    ER --> P3
    QR --> P4
    GR --> P4
    RP --> OUT["Structured Knowledge<br/>Representation &<br/>Quality Report"]:::output
```

---

## 3. Phase 1 -- Automated Transcription

### 3.1 Objective

Convert ethnographic audio and video recordings into structured, timestamped text transcriptions with quality guarantees sufficient for downstream LLM processing.

### 3.2 Data Source

Input files are organized in a **Google Drive catalog** (CSV) containing metadata such as `gdrive_id`, filename, MIME type, byte size, and duration. Only audio and video MIME types are selected for processing.

### 3.3 Speech Recognition

Transcription is performed using **OpenAI Whisper** (`whisper-large-v3` or `whisper-large-v3-turbo`) via the Hugging Face Transformers library. The engine supports:

- **Hardware-agnostic execution**: CUDA, Apple MPS, and CPU fallback
- **8-bit quantization**: Reduces VRAM requirements for GPU-constrained environments
- **Parallel batch processing**: `ProcessPoolExecutor` with one model instance per worker process
- **Checkpoint and resume**: Atomic progress tracking allows interrupted jobs to resume without reprocessing

### 3.4 Transcription Quality Validation

Each transcription undergoes a heuristic quality validation composed of four weighted dimensions:

| Dimension | Weight | What It Detects |
|-----------|--------|-----------------|
| **Script/Charset Match** | 35% | Wrong language output (e.g., CJK characters for Portuguese audio) |
| **Repetition Detection** | 30% | Repeated words, phrases, or hallucinated loops |
| **Segment Patterns** | 20% | Suspicious timestamps or degenerate segments |
| **Content Density** | 15% | Words per minute outside plausible range (30--300 wpm) |

Records scoring below **0.5** are flagged as `is_valid: false` and **automatically excluded** from downstream pipelines. This prevents low-quality transcriptions from degrading QA generation and knowledge graph construction.

### 3.5 Output

Each processed file produces an **EnrichedRecord** (JSON) containing: transcription text, timestamped segments, detected language, language probability, quality scores, hardware metadata, and processing duration.

---

## 4. Phase 2 -- Cognitive Elicitation Pipeline (CEP)

### 4.1 Objective

Generate cognitively-calibrated question-answer pairs that systematically elicit tacit knowledge from interview transcriptions, using Bloom's Taxonomy as a scaffolding framework to ensure cognitive diversity and depth.

### 4.2 Theoretical Foundation

The CEP is grounded in **Bloom's Revised Taxonomy** (Anderson & Krathwohl, 2001), which organizes cognitive processes into six progressive levels of complexity:

```mermaid
flowchart LR
    classDef low fill:#a8dadc,stroke:#457b9d,color:#1d3557,rx:8,ry:8
    classDef mid fill:#e9c46a,stroke:#e76f51,color:#264653,rx:8,ry:8
    classDef high fill:#e76f51,stroke:#c1121f,color:#fff,rx:8,ry:8

    R["<b>Remember</b><br/><i>Recall facts</i>"]:::low
    U["<b>Understand</b><br/><i>Explain ideas</i>"]:::low
    AP["<b>Apply</b><br/><i>Use in context</i>"]:::mid
    AN["<b>Analyze</b><br/><i>Draw connections</i>"]:::mid
    E["<b>Evaluate</b><br/><i>Justify judgments</i>"]:::high
    C["<b>Create</b><br/><i>Produce original ideas</i>"]:::high

    R --> U --> AP --> AN --> E --> C
```

Higher-order levels (Analyze, Evaluate, Create) are especially relevant for surfacing **tacit knowledge** -- the implicit, experience-based understanding that domain experts possess but rarely articulate explicitly. By forcing the LLM to generate questions at these levels, the pipeline extracts reasoning patterns, causal chains, and practical know-how that would remain invisible through simple factual recall.

### 4.3 Module I -- Bloom Scaffolding

Questions are generated according to a configurable **Bloom level distribution**:

| Cognitive Level | Default Allocation | Purpose |
|-----------------|-------------------|---------|
| **Remember** | 20% | Establish factual baseline from the transcript |
| **Understand** | 30% | Verify conceptual comprehension of described processes |
| **Analyze** | 30% | Identify causal relationships and hidden patterns |
| **Evaluate** | 20% | Surface expert judgment and decision-making rationale |

A key design feature is **scaffolding context**: when generating higher-level questions (Analyze, Evaluate), the prompt includes QA pairs already generated at lower levels. This mirrors how human cognition builds complex understanding on top of foundational knowledge.

### 4.4 Module II -- Reasoning & Grounding

For higher-order cognitive levels, this module enriches QA pairs with:

- **Reasoning traces**: Explicit logical chains connecting facts to conclusions (e.g., `"River rises -> Fisherman stores boat -> Reason: avoid equipment loss"`)
- **Multi-hop detection**: Identifies questions requiring synthesis of information from distant parts of the transcript (1--5 reasoning hops)
- **Tacit inference extraction**: Surfaces implicit domain knowledge that the interviewee assumed but did not verbalize (e.g., `"Rapid river rise indicates imminent flood risk"`)

### 4.5 Module III -- LLM-as-a-Judge Validation

Each generated QA pair undergoes automated quality validation using a separate LLM invocation acting as an evaluator. Three criteria are assessed with detailed rubrics:

```mermaid
flowchart LR
    classDef criterion fill:#264653,stroke:#1d3557,color:#fff,rx:10,ry:10
    classDef score fill:#f4f1de,stroke:#bbb,color:#333,rx:6,ry:6
    classDef gate fill:#e9c46a,stroke:#c9a227,color:#333,rx:4,ry:4

    QA["QA Pair<br/>Under Review"] --> F
    QA --> B
    QA --> I

    F["<b>Faithfulness</b><br/><i>Weight: 40%</i><br/>Is the answer grounded<br/>in the source text?"]:::criterion
    B["<b>Bloom Calibration</b><br/><i>Weight: 30%</i><br/>Does the question match<br/>the declared cognitive level?"]:::criterion
    I["<b>Informativeness</b><br/><i>Weight: 30%</i><br/>Does the answer reveal<br/>non-obvious knowledge?"]:::criterion

    F --> WS["Weighted Score<br/><i>0.4F + 0.3B + 0.3I</i>"]:::score
    B --> WS
    I --> WS
    WS --> G{"score >= 0.6"}:::gate
    G -- "Pass" --> V["Valid QA Pair"]
    G -- "Fail" --> R["Rejected"]
```

This approach ensures that the final QA dataset is **faithful** to the source material, **cognitively calibrated** to the intended Bloom level, and **informative** in terms of tacit knowledge content.

### 4.6 Output

Each transcription yields a **QARecordCEP** (JSON) containing: the complete set of QA pairs with Bloom annotations, reasoning traces, validation scores, Bloom distribution summary, and validation pass rates. An optional **JSONL** export provides a flat format suitable for downstream KGQA model training.

---

## 5. Phase 3 -- Knowledge Graph Construction

### 5.1 Objective

Extract structured entity-relation triples from transcription text and organize them into a corpus-level knowledge graph that represents the semantic structure of the interviewees' knowledge.

### 5.2 Extraction Pipeline

Knowledge graph construction uses **AutoSchemaKG**, which performs:

1. **Triple extraction**: Identifies `(subject, predicate, object)` triples from natural language text using an LLM
2. **Schema induction**: Automatically conceptualizes entity types and relation types from the extracted triples (dynamic typing rather than a fixed ontology)
3. **Graph merging**: Combines per-document graphs into a unified corpus-level graph with entity resolution

```mermaid
flowchart LR
    classDef process fill:#7b68ae,stroke:#5a4d8a,color:#fff,rx:8,ry:8
    classDef data fill:#f4f1de,stroke:#bbb,color:#333,rx:6,ry:6

    T["Transcription<br/>Text"]:::data
    TE["Triple<br/>Extraction"]:::process
    CSV["Triples<br/>(CSV)"]:::data
    SI["Schema<br/>Induction"]:::process
    CC["Concept<br/>CSV"]:::data
    GE["Graph<br/>Export"]:::process
    IG["Individual<br/>GraphML"]:::data
    MG["Corpus<br/>Merge"]:::process
    CG["Corpus<br/>GraphML"]:::data

    T --> TE --> CSV --> SI --> CC --> GE --> IG --> MG --> CG
```

### 5.3 Entity and Relation Types

AutoSchemaKG dynamically induces types from the data, though common types in this ethnographic domain include:

- **Entities**: `PERSON`, `LOCATION`, `EVENT`, `DATE`, `CONCEPT`, `OBJECT`, `ORGANIZATION`
- **Relations**: `LOCATED_IN`, `CAUSED_BY`, `AFFECTED_BY`, `OCCURRED_IN`, `BELONGS_TO`, and dynamically discovered domain-specific relations

### 5.4 Output

- **Per-document graphs**: Individual `.graphml` files for each transcription
- **Corpus-level graph**: A merged `corpus_graph.graphml` with entity resolution across all interviews
- **Provenance metadata**: A JSON sidecar tracking which documents contributed to each entity and relation

---

## 6. Phase 4 -- Multi-Dimensional Evaluation

### 6.1 Objective

Quantify the quality of knowledge elicitation across complementary dimensions, providing a holistic assessment of how well the pipeline captures and structures the tacit knowledge present in the ethnographic interviews.

### 6.2 Evaluation Dimensions

```mermaid
flowchart TB
    classDef dim fill:#c44569,stroke:#943349,color:#fff,rx:8,ry:8
    classDef metric fill:#f4f1de,stroke:#bbb,color:#333,rx:4,ry:4
    classDef report fill:#2d6a4f,stroke:#1b4332,color:#fff,rx:10,ry:10

    subgraph QA["QA Quality"]
        EM["Exact Match (EM)"]:::metric
        F1["Token-level F1"]:::metric
        BL["BLEU Score"]:::metric
    end

    subgraph ENT["Entity Coverage"]
        TD["Total / Unique Entities"]:::metric
        ED["Entity Density<br/><i>per 100 tokens</i>"]:::metric
        EV["Entity Diversity<br/><i>unique / total</i>"]:::metric
    end

    subgraph REL["Relation Metrics"]
        RD["Relation Density<br/><i>per entity</i>"]:::metric
        GC["Graph Connectivity<br/><i>degree, components</i>"]:::metric
    end

    subgraph SEM["Semantic Quality"]
        CO["Coherence Score"]:::metric
        ID["Information Density"]:::metric
        KC["Knowledge Coverage<br/><i>entities in QA pairs</i>"]:::metric
    end

    QA:::dim --> R
    ENT:::dim --> R
    REL:::dim --> R
    SEM:::dim --> R

    R["<b>Overall Score</b><br/><i>0.3 QA + 0.2 Entity +<br/>0.2 Relation + 0.3 Semantic</i>"]:::report
```

### 6.3 Overall Score Computation

The composite evaluation score is computed as a weighted average:

$$\text{Overall} = 0.3 \cdot F_1 + 0.2 \cdot \text{EntityDiversity} + 0.2 \cdot \min\!\left(\frac{\text{RelationDensity}}{3.0},\ 1.0\right) + 0.3 \cdot \text{Coherence}$$

This weighting prioritizes **answer quality** (F1) and **semantic coherence** while incorporating structural metrics from the knowledge graph to ensure the elicitation process captures rich, interconnected knowledge.

### 6.4 Output

An **EvaluationReport** (JSON) aggregating all metrics, dimensional scores, and the composite overall score.

---

## 7. Technical Infrastructure

### 7.1 LLM Integration

All LLM interactions use a **unified LLM client** built on the OpenAI SDK, supporting interchangeable backends:

| Provider | Use Case | Default Model |
|----------|----------|---------------|
| **Ollama** (local) | Development, HPC clusters | `qwen3:14b` |
| **OpenAI** | High-quality production runs | `gpt-4o` |
| **Custom** | Any OpenAI-compatible endpoint | Configurable |

The choice of `qwen3:14b` as default balances Portuguese language support, JSON output reliability, and inference speed on 24GB GPUs.

### 7.2 Batch Processing Architecture

All pipeline phases share a consistent batch processing pattern:

```mermaid
flowchart TB
    classDef infra fill:#264653,stroke:#1d3557,color:#fff,rx:8,ry:8
    classDef flow fill:#2a9d8f,stroke:#21867a,color:#fff,rx:6,ry:6

    CL["CLI Command<br/><i>Typer + Rich</i>"]:::infra
    CF["Configuration<br/><i>Pydantic Settings</i>"]:::infra
    TL["Task Loader<br/><i>Catalog / Directory</i>"]:::flow
    CP["Checkpoint<br/>Manager"]:::infra
    FI{"Filter<br/>completed?"}:::flow

    subgraph EX["Parallel Execution"]
        direction LR
        W1["Worker 1<br/><i>Model loaded once</i>"]:::flow
        W2["Worker 2<br/><i>Model loaded once</i>"]:::flow
        WN["Worker N<br/><i>Model loaded once</i>"]:::flow
    end

    RS["Result<br/><i>(id, success, msg)</i>"]:::flow
    SV["Save Output +<br/>Update Checkpoint"]:::infra

    CL --> CF --> TL --> FI
    CP --> FI
    FI -- "remaining tasks" --> EX
    EX --> RS --> SV --> CP
```

Key properties of this architecture:

- **Checkpoint-driven resumability**: Every completed or failed task is atomically recorded, allowing interrupted jobs to resume from the last successful checkpoint
- **Global worker pattern**: Heavy models (Whisper, LLM connections) are initialized once per worker process, avoiding repeated loading overhead
- **Batched future submission**: Tasks are submitted to the executor in controlled batches to prevent memory exhaustion on large corpora
- **Deterministic error handling**: Worker functions never raise exceptions; they return `(id, success, message)` triples that the orchestrator records

### 7.3 Deployment Options

| Environment | Description | Hardware |
|-------------|-------------|----------|
| **Local CLI** | Direct `gtranscriber` invocation | Developer workstation |
| **Docker Compose** | Multi-service profiles (`cep`, `kg`, `evaluate`) | Server with Docker |
| **SLURM** | HPC batch jobs with partition-specific scripts | NVIDIA L40S, RTX 4090, AMD |

### 7.4 Reproducibility

Every pipeline run records:

- **Execution environment**: SLURM detection, node hostname, partition
- **Hardware info**: GPU model, VRAM, CUDA version, CPU count
- **Configuration snapshot**: Complete settings used for the run
- **Timing**: Start/end timestamps, per-document processing duration
- **Model provenance**: Model ID, provider, temperature, and prompt templates

---

## 8. Data Flow Summary

The following diagram provides a compact view of the data artifacts flowing between pipeline phases:

```mermaid
flowchart LR
    classDef src fill:#4a6fa5,stroke:#2d4a7a,color:#fff,rx:10,ry:10
    classDef json fill:#f4f1de,stroke:#bbb,color:#333,rx:6,ry:6
    classDef graph fill:#d8b4fe,stroke:#a78bfa,color:#333,rx:6,ry:6
    classDef final fill:#2d6a4f,stroke:#1b4332,color:#fff,rx:10,ry:10

    A["Audio / Video<br/><i>Google Drive</i>"]:::src
    B["EnrichedRecord<br/><i>.json</i>"]:::json
    C["QARecordCEP<br/><i>.json / .jsonl</i>"]:::json
    D["Knowledge Graph<br/><i>.graphml</i>"]:::graph
    E["EvaluationReport<br/><i>.json</i>"]:::final

    A -- "Whisper ASR +<br/>Quality Filter" --> B
    B -- "CEP<br/>(Bloom + Reasoning +<br/>Validation)" --> C
    B -- "AutoSchemaKG<br/>(Triples + Schema<br/>Induction)" --> D
    C --> E
    D --> E
```

---

## 9. Summary

This methodology implements a **four-phase composable pipeline** for tacit knowledge elicitation from ethnographic interviews:

1. **Phase 1 (Transcription)** converts raw audio/video into quality-validated text using Whisper ASR with automatic failure detection
2. **Phase 2 (CEP)** generates cognitively-scaffolded QA pairs across Bloom's Taxonomy levels, enriched with reasoning traces and validated by an LLM-as-a-Judge
3. **Phase 3 (Knowledge Graph)** extracts entity-relation structures using AutoSchemaKG with dynamic schema induction
4. **Phase 4 (Evaluation)** quantifies elicitation quality across QA accuracy, entity coverage, graph connectivity, and semantic coherence

Each phase is independently deployable, checkpoint-resumable, and configurable through environment variables. The pipeline is designed for execution on HPC clusters via SLURM, enabling processing of large ethnographic corpora with GPU-accelerated inference. All phases share a consistent batch processing architecture that ensures fault tolerance and reproducibility.

---

### References

- Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives*. Longman.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *Proceedings of ICML*.
- AutoSchemaKG: LLM-driven knowledge graph construction with schema induction.
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*.
