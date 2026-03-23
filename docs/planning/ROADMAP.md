# Master's Roadmap

**Created**: 2026-03-23
**Target**: June 2026 (Phases A-C) | December 2026 (hard deadline)
**Scope**: Technical implementation + dissertation writing

---

## Timeline Overview

| Phase | Issue | When | Goal | Blocked By |
|-------|-------|------|------|------------|
| **A: Unblock KG** | [#78](https://github.com/FredDsR/etno-kgc-preprocessing/issues/78) | April (~2 weeks) | Get a complete GraphML from 309 transcriptions | Nothing |
| **B: Judge Refactor** | [#79](https://github.com/FredDsR/etno-kgc-preprocessing/issues/79) | April–May (~3 weeks) | Composable multi-stage judge as shared module | Nothing (parallel with A) |
| **C: RAG Evaluation** | [#80](https://github.com/FredDsR/etno-kgc-preprocessing/issues/80) | May–June (~4 weeks) | BM25 vs GraphRAG comparison using QA pairs | A + B |
| **D: Writing & Human Eval** | — | Parallel, intensifies June–Aug | Dissertation chapters, specialist evaluation, articles | B for human eval protocol, C for results chapter |
| **E: Polish** | — | July–Dec if needed | Bloom article, KG framework comparison, conferences | D for human eval data |

### Chronogram

```mermaid
gantt
    title Master's Roadmap — Arandu (2026)
    dateFormat YYYY-MM-DD
    axisFormat %b %Y

    section Implementation
    A · Unblock KG                        :task_a, 2026-04-01, 14d
    B · Judge refactor                    :task_b, 2026-04-01, 23d
    C · RAG evaluation + experiments      :task_c, after task_a task_b, 33d

    section Research
    Human evaluation with specialists     :task_h, after task_b, 55d
    Chapters 1-3                          :active, task_w, 2026-04-01, 75d
    Results chapter                       :task_r, after task_c task_h, 30d

    section Publication
    Articles + conferences                :task_p, after task_r, 60d

    section Milestones
    June target (A-C done)                :milestone, 2026-06-30, 0d
    December hard deadline                :milestone, 2026-12-15, 0d
```

### Dependency Graph

```mermaid
flowchart TD
    subgraph impl ["Technical Implementation"]
        A["A: Unblock KG · #78"]
        B["B: Judge Refactor · #79"]
        C["C: RAG Evaluation · #80"]
        A --> C
        B --> C
    end

    subgraph human ["Human Evaluation"]
        D2["Design annotation protocol"]
        D3["Recruit specialists + sessions"]
        D4["Compute kappa analysis"]
        B --> D2 --> D3 --> D4
    end

    subgraph writing ["Writing"]
        D1["Chapters 1-3 (start immediately)"]
        D5["Evaluation results chapter"]
        C --> D5
        D4 --> D5
    end

    subgraph pub ["Publication"]
        E1["Bloom taxonomy article"]
        E2["KG framework comparison"]
        E3["Conference submissions"]
        D4 --> E1
        D5 --> E1
        C --> E2
        E1 --> E3
    end
```

---

## Phase A: Unblock KG

**Goal**: Produce a complete Knowledge Graph from the 309 transcriptions.

**Context**: `test-kg-02` got halfway through concept generation (6,256/13,383 nodes) before SLURM killed it. A language bug means concepts were generated with English prompts. No GraphML output exists.

### Tasks

1. **Implement #77** — resumable concept generation wrapper + `language='pt'` bug fix
2. **Run KG pipeline to completion** — resume or fresh run with Portuguese prompts
3. **Inspect output quality** — are Portuguese entities/relations sensible? Knowledge islands?
4. **Analyze predicate explosion** — check for semantic duplicates in predicates (e.g., "lutar"/"combater"/"brigar" as separate predicates). AutoSchemaKG's conceptualization step should canonicalize these, but the language bug may have broken it. (Feedback: Joel, midway seminar)
5. **Close #75** — superseded by #77

### Success Criteria

- A `corpus_graph.graphml` file exists with meaningful Portuguese entities and relations
- Concept generation can survive SLURM timeouts and resume
- Predicate space is reasonable (no massive explosion of semantically equivalent predicates)

---

## Phase B: Judge Refactor

**Goal**: Composable multi-stage evaluation architecture usable across all pipeline domains.

### Architecture

```
shared/judge/
├── criterion.py      JudgeCriterion protocol (same interface for heuristic and LLM)
├── step.py           JudgeStep: runs N criteria with individual thresholds
├── pipeline.py       JudgePipeline: chains steps with filtering between them
└── registry.py       JudgeRegistry: criterion factory

Domain-specific criteria (all implementing JudgeCriterion):

transcription/criteria/
├── script_match.py        (heuristic)
├── repetition.py          (heuristic)
├── content_density.py     (heuristic)
├── language_drift.py      (LLM-based)
└── hallucination.py       (LLM-based)

qa/criteria/
├── faithfulness.py        (LLM-based)
├── bloom_calibration.py   (LLM-based)
├── informativeness.py     (LLM-based)
├── self_containedness.py  (LLM-based)
└── human_comparable.py    (LLM-based — designed for kappa comparison)
```

### Multi-Stage Pattern

Each domain uses the same pattern:

```
Stage 1: Fast filter (heuristic criteria or cheap LLM checks)
    ↓ items passing individual thresholds
Stage 2: Deep evaluation (LLM-based criteria)
    ↓ items passing individual thresholds
Stage 3: Human-comparable evaluation (single LLM criterion)
    → scores stored for later offline kappa comparison
```

### Tasks

1. **Individual thresholds** — replace weighted avg with per-criterion minimums
2. **Rename**: `JudgePipeline` → `JudgeStep`, create new `JudgePipeline` (multi-stage)
3. **Move judge to `shared/judge/`** — cross-domain service like `llm_client`
4. **#35: `generate_structured()` on LLMClient** — DRY across all LLM criteria
5. **Extract heuristic validators as criteria** — transcription heuristics implement `JudgeCriterion`
6. **New transcription criteria** — language drift, hallucination loop detection
7. **Human-comparable QA criterion** — designed backwards from annotation protocol
8. **Remove `--validate` flag** from `generate-cep-qa` (clean break, no backwards compat)
9. **Standalone CLI**: `arandu judge <domain> <input>`

### Key Decisions

- Heuristics are just criteria — `JudgeStep` doesn't care if a criterion is heuristic or LLM-based
- Each CLI command loads only the models it needs — no co-loading Whisper + LLM
- Generator generates, judge judges — separate commands, separate concerns

---

## Phase C: RAG Evaluation

**Goal**: Compare retrieval strategies using CEP QA pairs as the benchmark dataset.

### Design

The evaluation IS the experiment:
- CEP QA pairs are the **benchmark questions**
- Retrievers attempt to answer each question from source material
- The **same judge pipeline** scores the retriever answers
- Comparison: which retriever produces answers that the judge rates highest?

### Components

1. **Retriever protocol** — abstract interface with pluggable implementations
2. **BM25 baseline** — sparse retrieval over raw transcriptions
3. **GraphRAG retriever** — retrieval over the constructed KG
4. **CLI commands**:
   - `arandu retrieve` — runs a retriever against QA questions, produces answers
   - `arandu judge answers` — judges answer quality using the judge pipeline

### No Traditional Metrics

Evaluation does **not** use EM, F1, or BLEU. The judge module scores retriever answers with the same criteria used for QA validation (faithfulness, informativeness, etc.). This is consistent — judge all the way down.

### Non-answerable Questions Experiment

Generate a subset of questions whose answers are **not** present in the KG, to test whether the retriever falls back to parametric knowledge or correctly abstains. This detects hallucination at the retrieval level and validates that the evaluation measures graph coverage, not LLM memorization. (Feedback: Luciana, midway seminar)

### Tasks

1. Define retriever protocol
2. Implement BM25 baseline retriever
3. Implement GraphRAG retriever (using Phase A graph)
4. Design non-answerable question subset (questions the KG cannot answer)
5. Implement `arandu retrieve` CLI command
6. Implement `arandu judge answers` CLI command (reuses Phase B judge)
7. Run experiments, collect results
8. Analyze: separate graph quality limitations from retrieval tool limitations (Feedback: Joel, midway seminar)

---

## Phase D: Writing & Human Evaluation

**Goal**: Dissertation (monography), human evaluation with domain specialists, and article submissions.

### Dissertation Structure

#### Chapter 1 — Introdução

Problem statement, objectives, justification. Frames the evidence gap at the intersection of AI, traditional ecological knowledge, and climate resilience.

| What exists | What's needed |
|-------------|---------------|
| SLR findings (Reckziegel & Costa, 2025b) documenting the evidence gap | Write the problem statement connecting the SLR gap to the Arandu pipeline |
| `docs/related-works.md` §10 (Ethnographic Knowledge, AI, Climate Resilience) | Frame the research questions and objectives |
| `docs/related-works.md` §11 (Positioning: What This Work Adds) | Justification — why this integration is novel |
| CARE Principles / decolonial frameworks (Carroll et al., BlackDeer, Tapu & Fa'agau) | Ethical grounding section |

**Can start**: now

#### Chapter 2 — Referencial Teórico / Trabalhos Relacionados

Literature review organized by thematic axes. `docs/related-works.md` (45+ references, 11 sections) is the primary source — needs consolidation into a cohesive narrative.

| Thematic Axis | Source | Pipeline Phase |
|---------------|--------|----------------|
| Tacit knowledge elicitation with LLMs | `related-works.md` §1 | Overarching |
| LLM-assisted interview/narrative analysis | `related-works.md` §2 | Phase 1-2 |
| Bloom's Taxonomy and QA generation | `related-works.md` §3 | Phase 2 (CEP) |
| LLM-as-a-Judge evaluation | `related-works.md` §4 | Phase 2 (Judge) |
| Self-containedness and decontextualization | `related-works.md` §5 | Phase 2 (CEP) |
| KG construction from text with LLMs | `related-works.md` §6 | Phase 3 |
| Value of KGs as knowledge distillation | Argue independently of RAG performance (Feedback: Joel) | Phase 3 / Discussion |
| Predicate explosion in triple extraction | Known KGC problem — does AutoSchemaKG's conceptualization handle it? (Feedback: Joel) | Phase 3 |
| GraphRAG and graph-based retrieval | `related-works.md` §7 | Phase 4 (Eval) |
| QA-based KG evaluation | `related-works.md` §8 | Phase 4 (Eval) |
| ASR quality and Whisper | `related-works.md` §9 | Phase 1 |
| TEK, AI, and climate resilience | `related-works.md` §10 | Context |
| Network science foundations | `related-works.md` §10.3 | Phase 3 analysis |

**Can start**: now (consolidate `related-works.md` into chapter form)

#### Chapter 3 — Metodologia

Pipeline design, each phase's approach and rationale. `docs/methodology.md` (62K chars, 9 sections) is nearly a complete draft.

| Section | Source | Status |
|---------|--------|--------|
| Pipeline architecture (overview) | `methodology.md` §1-2 | Draft ready |
| Phase 1: Automated transcription | `methodology.md` §3 | Draft ready |
| Phase 2: Cognitive Elicitation Pipeline | `methodology.md` §4 | Draft ready |
| Phase 3: Knowledge Graph Construction | `methodology.md` §5 | Draft ready |
| Phase 4: Evaluation (retrieve + judge) | `methodology.md` §6 | Needs rewrite — current draft uses intrinsic metrics, new design uses judge-based RAG evaluation |
| Human evaluation protocol | — | Not written — depends on Phase B (annotation protocol design) |
| Technical infrastructure | `methodology.md` §7 | Draft ready |

**Can start**: now for sections 1-5, 7. Section 6 blocked on Phase B/C design finalization.

#### Chapter 4 — Resultados e Discussão

Experiment results, analysis, human evaluation findings.

| Section | Blocked on |
|---------|------------|
| Transcription results (353/354 files, quality analysis) | Nothing — data exists |
| CEP QA results (241/309 records, Bloom distribution, judge scores) | Nothing — data exists |
| KG construction results (graph structure, Portuguese entities/relations) | Phase A (no graph yet) |
| RAG evaluation (BM25 vs GraphRAG comparison) | Phase C |
| Non-answerable questions (parametric knowledge detection) | Phase C |
| Human evaluation (kappa analysis, LLM vs specialist agreement) | Phase D human eval sessions |
| Graph value argument (knowledge distillation independent of RAG) | Phase A (graph must exist) |
| Predicate explosion analysis (semantic duplicates in predicates) | Phase A (graph must exist) |
| Technique vs data limitations discussion | All above |
| Discussion (Bloom-stratified depth profiling, tacit knowledge layers) | All above |

**Can start**: transcription and CEP results sections now. Rest blocked.

#### Chapter 5 — Conclusão

Summary, contributions, limitations, future work. Written last.

---

### Human Evaluation with Domain Specialists

The human-comparable judge criterion (Phase B) produces LLM scores that must be compared against human judgments to validate the pipeline. This is a research activity, not a coding task.

1. **Design annotation protocol** — define the evaluation question (same one the LLM judge answers), annotation guidelines, scoring rubric
2. **Select QA sample** — stratified sample across Bloom levels, participants, and validation scores
3. **Recruit specialists** — domain experts familiar with riverine communities / climate events
4. **Run evaluation sessions** — specialists score the same QA pairs the LLM judge scored
5. **Compute agreement** — Cohen's kappa (LLM vs human), inter-annotator agreement
6. **Analyze results** — where does the LLM judge agree/disagree with humans? At which Bloom levels?

**Depends on**: Phase B (human-comparable criterion must exist to define what question specialists answer)
**Blocks**: Phase E (Bloom taxonomy article needs this data)

### Articles

- Consolidate 2025/2 article with new experiments and discoveries
- Bloom taxonomy for cognitive-calibrated QA generation — standalone publication candidate
- Conference submissions

---

## Phase E: Polish & Extensions (if time permits)

- **Bloom taxonomy article** — needs human eval data from Phase D
- **KG framework comparison** — prototype 1-2 alternatives to AutoSchemaKG
- **Justfile** for pipeline orchestration (low effort, nice-to-have)
- **SLURM cheatsheet** — `RUNNING_ON_SLURM.md` with exact commands in order

---

## Command Structure

Each pipeline step is an atomic CLI command. No model co-loading. Pipeline orchestration is command sequencing (Justfile or shell script).

| Command | Input | Output | Models |
|---------|-------|--------|--------|
| `arandu transcribe` | audio/video | EnrichedRecord | Whisper |
| `arandu judge transcription` | EnrichedRecord | scored EnrichedRecord | LLM (heuristics need no model) |
| `arandu generate-cep-qa` | EnrichedRecord | QAPairCEP | LLM |
| `arandu judge qa` | QAPairCEP | scored QAPairCEP | LLM |
| `arandu build-kg` | EnrichedRecord | GraphML | LLM |
| `arandu retrieve` | QA pairs + source | retriever answers | depends on retriever |
| `arandu judge answers` | QA pairs + answers | judge scores | LLM |

---

## Explicitly Skipped

| Item | Reason |
|------|--------|
| Pipeline orchestrator framework (Luigi/Airflow/scikit) | YAGNI — Justfile is sufficient |
| Autonomous SLURM runners | Manual 5-command process is fine for ~20 runs |
| Frontend/report issues (#40, #59–#67) | Not on critical path for dissertation |
| Heuristic/judge module unification | Solved by making heuristics into criteria |
| Backwards compatibility for `--validate` flag | Clean break, minimal codebase |
| EM/F1/BLEU evaluation metrics | Replaced by judge-based evaluation |

---

## Midway Seminar Feedback

Feedback from Luciana and Joel incorporated into the roadmap:

| Feedback | Who | Addressed In |
|----------|-----|-------------|
| Validate QA ground truth with humans, measure concordance | Luciana | Phase D: Human eval with specialists (Cohen's kappa) |
| BM25 baseline — is RAG without graph sufficient? | Luciana | Phase C: BM25 vs GraphRAG comparison |
| Non-answerable questions to detect parametric knowledge use | Luciana | Phase C: Non-answerable questions experiment |
| Argue the value of the graph itself (knowledge distillation) | Joel | Chapter 2 + Chapter 4 discussion |
| Separate graph quality limitations from retrieval tool limitations | Joel | Phase C task 8 + Chapter 4 discussion |
| Predicate explosion — does AutoSchemaKG handle it? | Joel | Phase A task 4 (inspect graph) + Chapter 4 |
| Distinguish technique limitations from data limitations | Joel | Chapter 4 discussion |

---

## Open Issues

| # | Title | Phase | Priority |
|---|-------|-------|----------|
| [#77](https://github.com/FredDsR/etno-kgc-preprocessing/issues/77) | Resumable concept generation + language bug | A | Critical |
| [#35](https://github.com/FredDsR/etno-kgc-preprocessing/issues/35) | Extract `generate_structured()` to LLMClient | B | High |
| ~~[#75](https://github.com/FredDsR/etno-kgc-preprocessing/issues/75)~~ | ~~Concept gen resume~~ | — | Closed (superseded by #77) |

---

**Last Updated**: 2026-03-23
