# Arandu - The Knowledge Discovery Tooling

[![.github/workflows/ci.yml](https://github.com/FredDsR/etno-kgc-preprocessing/actions/workflows/ci.yml/badge.svg)](https://github.com/FredDsR/etno-kgc-preprocessing/actions/workflows/ci.yml)
![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/94bfd9f7de8e4f16abcdc62811a81cd0/raw/tests-badge.json)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/94bfd9f7de8e4f16abcdc62811a81cd0/raw/coverage-badge.json)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)

Arandu turns recorded interviews into a knowledge graph and then measures that
graph without a reference to compare it against. It transcribes the recordings,
generates a cognitively stratified question-answer benchmark from them, builds a
knowledge graph with an LLM, and asks how well each retrieval strategy over that
graph lets an answerer answer the benchmark, or abstain when it should. The case
study is a corpus of interviews with riverine communities in southern Brazil
affected by the May 2024 floods.

## About the Name

**Arandu** comes from the Guarani language (*ara* = time + *endu* = to hear/feel),
meaning "wisdom through perceiving time": knowledge gained through listening and
experiencing. The project's aim is the same: eliciting tacit knowledge from
ethnographic interviews.

## What Arandu evaluates

A knowledge graph extracted from oral narrative has no ground-truth graph, the
concepts it uses emerge from the same corpus, and an unanswered question is not
necessarily an error. Arandu therefore evaluates the graph functionally, through
a benchmark derived from the corpus itself. The framework has three parts.

- **Cognitively stratified benchmark.** For each chunk of a transcript an LLM
  generates question-answer pairs at four levels of Bloom's revised taxonomy
  (Remember, Understand, Analyze, Evaluate). Each level receives the pairs of the
  levels below as scaffolding context, so that higher-order questions build on
  established facts. An LLM judge scores every pair on faithfulness, Bloom
  calibration, informativeness and self-containment, and the gate is
  conjunctive: a pair passes only if every criterion clears the threshold. A
  separate emic-validity judge scores, from 1 to 5, whether a pair keeps the
  speaker's frame or restates it in the analyst's categories; it stratifies the
  metrics instead of filtering the benchmark.
- **Unanswerable probes.** Entity perturbation swaps a named entity in a
  benchmark question for one found neither in the graph nor in the corpus, so the
  correct behaviour on the probe is to abstain.
- **Metrics.** A shared answerer receives each question and only the evidence a
  retrieval arm returned. Answers fall into four cells: TC (commits on an
  answerable question), FC (commits on an unanswerable one), FA (abstains on an
  answerable one) and TA (abstains on an unanswerable one). See
  [Metrics](#metrics).

## Pipeline

Every stage reads and writes under `results/<run-id>/<stage>/`, checkpoints its
progress and can be resumed. Run the stages in this order with the same
`--id`:

| # | Stage | Command | Output in `results/<run-id>/` |
|---|-------|---------|-------------------------------|
| 1 | Transcription | `batch-transcribe` (also `transcribe`, `drive-transcribe`) | `transcription/` |
| 2 | Transcription quality | `judge-transcription` | verdicts written into the transcription records |
| 3 | Chunking | `chunk` | `chunk/` |
| 4 | Benchmark generation with the scaffolding gate | `generate-cep-qa` | `cep/` |
| 5 | Pair judging (resume or re-judge) | `judge-qa` | verdicts in `cep/`, metadata in `judge_qa/` |
| 6 | Emic validity | `emic-judge` | `emic_judge/` |
| 7 | Knowledge graph | `build-kg`, then `kg-link-passages` and `kg-build-retriever-index` | `kg/` |
| 8 | Unanswerable probes | `generate-non-answerable` | `non_answerable/` |
| 9 | Retrieval, one run per arm | `retrieve` | `retrieve/` |
| 10 | Answering | `answer` | `answers/` |
| 11 | Answer judging | `judge-answers` | `judge_answers/` |
| 12 | Cross-arm report | `rag-analysis` | `analysis/` |
| 13 | Human evaluation sample and annotation | `build-human-eval-sample`, `emic-annotation-build`, `emic-annotation-push`, `emic-annotation-pull` | `human_eval/`, `annotation/` |

A run end to end, with the defaults of each stage:

```bash
RUN=my-run

arandu batch-transcribe input/catalog.csv --id $RUN --credentials credentials.json
arandu judge-transcription results/$RUN/transcription/outputs
arandu chunk results/$RUN/transcription/outputs --id $RUN --view cep_4k
arandu generate-cep-qa results/ --id $RUN
arandu emic-judge --id $RUN
arandu build-kg results/ --id $RUN
arandu kg-link-passages --id $RUN
arandu kg-build-retriever-index --id $RUN
arandu generate-non-answerable --id $RUN
arandu retrieve --id $RUN
arandu answer --id $RUN
arandu judge-answers --id $RUN
arandu rag-analysis --id $RUN
```

`generate-cep-qa` judges each pair during generation (the scaffolding gate) and
needs a validator model, set with `ARANDU_JUDGE_VALIDATOR_MODEL` or
`--validator-model`. `judge-qa results/$RUN/cep/outputs` only judges the pairs
that still have no clean verdict, or all of them with `--rejudge`.

Run management: `list-runs`, `run-info`, `replicate` and `rebuild-index`.
Reports: `report` writes an HTML dashboard and PNG charts, and `serve-report`
serves it. Each command documents its options with `arandu <command> --help`,
and the [CLI reference](docs/guide/cli-reference.md) collects them.

## Retrieval arms

All arms return their top 10 evidence items to the same answerer, under the
same prompt and judges, so differences between arms come from the access
strategy. Select them with `arandu retrieve --arm <id>` (repeatable); the
default runs all five.

| Arm | Evidence returned | How it retrieves |
|-----|-------------------|------------------|
| `null` | nothing | Control that always abstains; anchors the abstention scale |
| `bm25` | 4k-character transcript chunks | Lexical BM25 over the chunks; separates the value of graph structure from the value of having passages |
| `khop_passage` | graph passages | Matches the question's lemmas to node labels, expands a 2-hop neighbourhood over every edge type, and returns the passages its entities and events came from, ranked by hop distance |
| `khop_triple` | linearized triples | Same seeding and neighbourhood as `khop_passage`, but returns the relation edges themselves |
| `atlas_rag` | graph passages | HippoRAG-style retriever shipped with AutoSchemaKG: Personalized PageRank from the question's entities over node embeddings |

Arm settings come from `ARANDU_BM25_*`, `ARANDU_KHOP_*` and
`ARANDU_ATLAS_RAG_*`. The arms are compared in the
[RAG analysis guide](docs/guide/pipelines/rag-analysis.md).

## Metrics

`rag-analysis` writes `report.json` and `tables.md` under `analysis/outputs/`,
per arm and per Bloom level.

- **Knowledge coverage (KC):** mean of correctness times faithfulness over the
  TC items, the questions an arm chose to answer.
- **Hallucination:** FC / (FC + TA).
- **Over-caution:** FA / (FA + TC).
- **F1 of abstention:** 2 TA / (2 TA + FA + FC), with 95% Wilson intervals on
  the rates.
- **Evidence support:** share of answerable questions whose retrieved evidence
  supports the reference answer, committed or not; it separates missing evidence
  from an answerer that declines sufficient evidence.

A question counts as abstained only when the answerer's structured flag and an
abstention judge agree. Every metric can be stratified by emic validity
(EV >= 4) to check whether a comparison between arms holds for references that
keep the speaker's frame. Judge reliability is measured by test-retest with
Krippendorff's alpha (`analysis/analyze_judge_retest.py`).

## Installation

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # base dependencies
uv sync --all-extras --all-groups    # with the KG, report and dashboard extras and the test tools
uv run arandu --help
```

## Running on Docker and SLURM

Each step runs as a Docker Compose profile, with an Ollama sidecar for the LLM
steps:

| Profile | Services | Pipeline |
|---------|----------|----------|
| `cep` (`cep-gpu`) | ollama, arandu-cep | CEP QA generation (`generate-cep-qa`); also needs the `ARANDU_JUDGE_VALIDATOR_*` vars for the gate |
| `judge` (`judge-gpu`) | ollama, arandu-judge | LLM-as-a-Judge (`judge-transcription` / `judge-qa`) |
| `emic` (`emic-gpu`) | ollama, arandu-emic | Emic-validity judge (`emic-judge`) |
| `kg` (`kg-gpu`) | ollama, arandu-kg | KG construction (`build-kg`) |
| `rag` (`rag-gpu` / `rag-cpu`) | ollama, arandu-rag | Retrieval evaluation chain |
| `cpu` | arandu-cpu | Transcription (CPU) |
| `rocm` | arandu-rocm | Transcription (AMD GPU) |

On the pcad cluster the SLURM scripts live in `scripts/slurm/<step>/`, each
sourcing a shared `<step>_common.sh`. Submit with `PIPELINE_ID` set:

```bash
PIPELINE_ID=run-01 sbatch scripts/slurm/transcription/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/cep/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/judge/qa/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/emic/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/kg/tupi.slurm
PIPELINE_ID=run-01 sbatch scripts/slurm/rag/chunk.slurm   # also retrieve, answer, judge-answers, ...
```

[scripts/slurm/AGENTS.md](scripts/slurm/AGENTS.md) maps each command to its
config class, compose service, image and SLURM directory, and lists the
partition rules. The [SLURM guide](docs/operations/slurm.md) covers submission.

## Configuration

Command-line options take precedence over environment variables, which can live
in a `.env` file (see [.env.example](.env.example)). Each LLM role is configured
on its own, so the generator, the judges and the answerer can use different
models:

| Prefix | Configures |
|--------|------------|
| `ARANDU_` | Transcription (Whisper model, device, workers) |
| `ARANDU_QA_`, `ARANDU_CEP_` | Benchmark generation and the scaffolding gate |
| `ARANDU_JUDGE_` | QA pair and transcription judges, and the gate validator |
| `ARANDU_EMIC_JUDGE_` | Emic-validity judge |
| `ARANDU_KG_`, `ARANDU_EMBEDDER_` | Graph construction and node embeddings |
| `ARANDU_NONANSWERABLE_` | Unanswerable probes |
| `ARANDU_BM25_`, `ARANDU_KHOP_`, `ARANDU_ATLAS_RAG_` | Retrieval arms |
| `ARANDU_ANSWERER_`, `ARANDU_JUDGE_ANSWERS_` | Answerer and answer judges |
| `ARANDU_LABEL_STUDIO_` | Human annotation |
| `ARANDU_RESULTS_` | Results directory and versioning |

The [configuration guide](docs/guide/configuration.md) lists every variable.

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/guide/getting-started.md)
- [CLI reference](docs/guide/cli-reference.md)
- [Configuration](docs/guide/configuration.md)
- [Methodology](docs/research/methodology.md)
- [Canonical results: thesis-run-02](docs/research/results/thesis-run-02.md)
- [Analyses of finished runs](analysis/README.md)

## License

MIT
