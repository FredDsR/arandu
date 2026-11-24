# Arandu

[![CI](https://github.com/FredDsR/arandu/actions/workflows/ci.yml/badge.svg)](https://github.com/FredDsR/arandu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24453915.svg)](https://doi.org/10.5281/zenodo.24453915)
[![Dataset DOI](https://img.shields.io/badge/Dataset_DOI-10.5281%2Fzenodo.24453595-blue)](https://doi.org/10.5281/zenodo.24453595)

Arandu turns recorded oral interviews into a knowledge graph and evaluates that
graph without a reference graph to compare it against. It transcribes the
recordings, generates a cognitively stratified question-answer benchmark from
them, builds an open-schema knowledge graph with an LLM, and measures how well
different retrieval strategies over that graph (and over the underlying
transcripts) allow a shared answerer to answer the benchmark—or abstain when it
should.

The accompanying dataset and study examine a corpus of oral interviews with
riverside communities in southern Brazil affected by the May 2024 floods.

## About the Name

**Arandu** comes from Guarani (*ara* = time + *endu* = to hear/feel), meaning
"wisdom through perceiving time": knowledge gained through listening and
experiencing.

## Framework Overview

A knowledge graph induced from oral narrative has no gold-standard graph, its
schema emerges from the corpus itself, and an unanswered question is not
necessarily an error. Arandu therefore evaluates the graph functionally through
a corpus-derived benchmark with three components:

- **Cognitively stratified benchmark (CEP).** For each transcript chunk, an LLM
  generates question-answer pairs across four levels of Bloom's revised taxonomy
  (*Remember*, *Understand*, *Analyze*, *Evaluate*). Each level receives pairs
  from lower levels as scaffolding context so higher-order questions build on
  established facts. An LLM judge scores every pair on faithfulness, Bloom
  calibration, informativeness, and self-containment under a conjunctive gate
  (all applicable criteria must clear the threshold). A separate emic-validity
  judge scores from 1 to 5 whether each pair preserves the speaker's frame or
  recasts it into analyst categories; emic validity stratifies downstream
  metrics rather than filtering the benchmark.
- **Unanswerable probes.** Entity perturbation swaps a named entity in an
  answerable question for one absent from both the graph and the corpus, so the
  correct behavior is abstention.
- **Open-world evaluation metrics.** A shared answerer receives each question
  and only the top-10 evidence items returned by a retrieval arm. Predictions
  fall into four cells: **TC** (commits on an answerable question), **FC**
  (commits on an unanswerable probe), **FA** (abstains on an answerable
  question), and **TA** (abstains on an unanswerable probe). See
  [Metrics](#metrics).

## Pipeline Stages

Every stage reads and writes under `results/<run-id>/<stage>/`, checkpoints its
progress, and can be resumed by passing the same `--id`:

| # | Stage | Command | Output under `results/<run-id>/` |
|---|-------|---------|----------------------------------|
| 1 | Transcription | `batch-transcribe` (also `transcribe`, `drive-transcribe`) | `transcription/` |
| 2 | Transcription quality | `judge-transcription` | verdicts written into transcription records |
| 3 | Chunking | `chunk` | `chunk/` |
| 4 | Benchmark generation + scaffolding gate | `generate-cep-qa` | `cep/` |
| 5 | Pair judging (resume or re-judge) | `judge-qa` | verdicts in `cep/`, metadata in `judge_qa/` |
| 6 | Emic validity | `emic-judge` | `emic_judge/` |
| 7 | Knowledge graph | `build-kg`, `kg-link-passages`, `kg-build-retriever-index` | `kg/` |
| 8 | Unanswerable probes | `generate-non-answerable` | `non_answerable/` |
| 9 | Retrieval (one pass per arm) | `retrieve` | `retrieve/` |
| 10 | Answering | `answer` | `answers/` |
| 11 | Answer judging | `judge-answers` | `judge_answers/` |
| 12 | Cross-arm analysis | `rag-analysis` | `analysis/` |
| 13 | Human evaluation & annotation | `build-human-eval-sample`, `emic-annotation-build`, `emic-annotation-push`, `emic-annotation-pull` | `human_eval/`, `annotation/` |

End-to-end execution with stage defaults:

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
requires a validator model (`ARANDU_JUDGE_VALIDATOR_MODEL` or
`--validator-model`). `judge-qa results/$RUN/cep/outputs` scores pairs without a
clean verdict, or all pairs with `--rejudge`.

Run management commands: `list-runs`, `run-info`, `replicate`, and
`rebuild-index`. Reporting commands: `report` and `serve-report`. Every command
exposes `--help`, and the [CLI reference](docs/guide/cli-reference.md) lists
all options.

## Retrieval Arms

All arms return their top 10 evidence items to the same answerer under identical
prompts and judges, isolating the effect of the retrieval strategy. Select arms
with `arandu retrieve --arm <id>` (repeatable); the default runs all eight:

| Arm | Evidence returned | Strategy |
|-----|-------------------|----------|
| `null` | none | Lower-bound control that always abstains; anchors the abstention scale |
| `oracle` | source 4k-char chunk | Upper-bound control returning the exact chunk that generated the QA pair (or the seed pair's chunk for a probe) |
| `bm25` | 4k-char transcript chunks | Lexical BM25 over chunks; separates graph structure from passage access |
| `dense_bgem3_256` | 4k-char transcript chunks | Dense retrieval with BGE-M3 over 256-token sliding windows (25% overlap), max-cosine chunk scoring |
| `dense_mpnet128` | 4k-char transcript chunks | Dense retrieval with `paraphrase-multilingual-mpnet-base-v2` over 128-token windows (same encoder as `atlas_rag`) |
| `khop_passage` | graph passages | Lemmatized smoothed-IDF seeding + 2-hop neighborhood expansion, returning source passages ranked by proximity |
| `khop_triple` | linearized triples | Same seeding and 2-hop neighborhood as `khop_passage`, returning linearized relation edges |
| `atlas_rag` | graph passages | HippoRAG-style Personalized PageRank from question entities over AutoSchemaKG node embeddings |

Additional named dense presets (`dense_bgem3_128`, `dense_bgem3_512`,
`dense_bgem3_whole`, `dense_e5_*`) are available on demand. Arm settings are
controlled via `ARANDU_BM25_*`, `ARANDU_DENSE_*`, `ARANDU_KHOP_*`, and
`ARANDU_ATLAS_RAG_*`. See the [RAG analysis guide](docs/guide/pipelines/rag-analysis.md).

## Metrics

`rag-analysis` writes `report.json` and `tables.md` under `analysis/outputs/`,
per arm and per Bloom level:

- **Knowledge Coverage (KC):** mean of `answer_correctness * answer_faithfulness`
  over committed answerable items (**TC**).
- **Effective Knowledge Coverage (EKC):** `KC * (1 - OC)`, combining answer
  quality with abstention rate.
- **Hallucination:** `FC / (FC + TA)`.
- **Over-caution (OC):** `FA / (FA + TC)`.
- **Abstention F1:** `2 * TA / (2 * TA + FA + FC)`, with 95% Wilson confidence
  intervals.
- **Evidence support:** share of answerable questions whose retrieved evidence
  supports the reference answer (`passage_coverage >= threshold`), separating
  missing evidence from an answerer declining sufficient evidence.

A response counts as abstained only when the answerer's structured flag and the
abstention judge agree. All metrics can be stratified by emic validity
(`EV >= 4`), and judge stability is measured via test-retest Krippendorff's
alpha and Gwet's AC2 (`analysis/analyze_judge_retest.py`).

## Installation

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                              # core dependencies
uv sync --all-extras --all-groups    # KG, report, dashboard, analysis extras + test/lint tools
uv run arandu --help
```

## Reproducing Published Analyses

The restricted-access dataset archive on Zenodo (CC BY-NC-SA 4.0, DOI:
[10.5281/zenodo.24453595](https://doi.org/10.5281/zenodo.24453595)) preserves
Arandu's native `results/<run-id>/<stage>/outputs/` layout. Once extracted into
`results/`, every analysis module in [analysis/](analysis/README.md) runs
read-only from the repository root without calling any LLM:

```bash
# Verify checksums and extract the Zenodo archives into results/
sha256sum -c CHECKSUMS.sha256
mkdir -p results
tar -xzf arandu-rs2024-v1.tar.gz -C results/
tar -xzf arandu-rs2024-v1-retest.tar.gz -C results/

# Examples (see analysis/README.md for the full command index)
RUN=results/arandu-rs2024-v1
R=results/arandu-rs2024-v1-retest

uv run python -m analysis.analyze_bloom_yield --id arandu-rs2024-v1
uv run python -m analysis.analyze_conditional_overcaution --id arandu-rs2024-v1
uv run python -m analysis.kg_structure_profile --id arandu-rs2024-v1
uv run python -m analysis.analyze_corpus_coverage \
    --id arandu-rs2024-v1 \
    --replica "$RUN"
uv run --with krippendorff python -m analysis.analyze_judge_retest \
    --replica "$R/r1" --replica "$R/r2" --replica "$R/r3" \
    --gate-from "3=$R/r3-gate" \
    --nonans-dataset "$RUN/non_answerable/outputs/dataset.json" \
    --kg-graphml "$RUN/kg/outputs/atlas_output/kg_graphml/transcriptions.json_graph.graphml" \
    --out-json "$R/out/retest.json" --out-md "$R/out/retest.md" \
    --n-boot 2000 --seed 20261005
```

To package a sanitized release bundle from a completed run, see
[scripts/export_zenodo_release.py](scripts/export_zenodo_release.py).

## Running with Docker and HPC (SLURM)

Each pipeline stage is defined as a Docker Compose profile in
[docker-compose.yml](docker-compose.yml), paired with an Ollama sidecar for
local LLM inference:

| Profile | Services | Pipeline |
|---------|----------|----------|
| `cep` (`cep-gpu`) | `ollama`, `arandu-cep` | CEP QA generation (`generate-cep-qa`) |
| `judge` (`judge-gpu`) | `ollama`, `arandu-judge` | LLM-as-a-Judge (`judge-transcription` / `judge-qa`) |
| `emic` (`emic-gpu`) | `ollama`, `arandu-emic` | Emic-validity judge (`emic-judge`) |
| `kg` (`kg-gpu`) | `ollama`, `arandu-kg` | KG construction (`build-kg`) |
| `rag` (`rag-gpu` / `rag-cpu`) | `ollama`, `arandu-rag` | Retrieval evaluation chain |
| `cpu` | `arandu-cpu` | Transcription (CPU) |
| `rocm` | `arandu-rocm` | Transcription (AMD GPU) |

See [docs/operations/docker.md](docs/operations/docker.md) and
[docs/operations/slurm.md](docs/operations/slurm.md). Cluster-specific SLURM
deployment scripts live in the companion
[`arandu-deploy`](https://github.com/FredDsR/arandu-deploy) repository.

## Configuration

CLI options override environment variables (which may live in `.env`; see
[.env.example](.env.example)). Each LLM role is configured independently so
generation, judging, and answering can target different models or providers
(Ollama, OpenAI-compatible APIs, DeepSeek, etc.):

| Prefix | Configures |
|--------|------------|
| `ARANDU_` | Transcription (Whisper model, device, workers) |
| `ARANDU_QA_`, `ARANDU_CEP_` | Benchmark generation and scaffolding gate |
| `ARANDU_JUDGE_` | QA and transcription judges, plus scaffolding gate validator |
| `ARANDU_EMIC_JUDGE_` | Emic-validity judge |
| `ARANDU_KG_`, `ARANDU_EMBEDDER_` | Graph construction and node embeddings |
| `ARANDU_NONANSWERABLE_` | Unanswerable probes |
| `ARANDU_BM25_`, `ARANDU_DENSE_`, `ARANDU_KHOP_`, `ARANDU_ATLAS_RAG_` | Retrieval arms |
| `ARANDU_ANSWERER_`, `ARANDU_JUDGE_ANSWERS_` | Answerer and answer judges |
| `ARANDU_LABEL_STUDIO_` | Human annotation integration |
| `ARANDU_RESULTS_` | Results base directory and run layout |

See [docs/guide/configuration.md](docs/guide/configuration.md) for the full
variable reference.

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/guide/getting-started.md)
- [CLI reference](docs/guide/cli-reference.md)
- [Configuration](docs/guide/configuration.md)
- [Analysis modules](analysis/README.md)
- [Operational scripts](scripts/README.md)
- [Emic annotator instructions (pt-BR)](docs/instruments/annotator-instructions.pt.md)
- [Contributing](CONTRIBUTING.md)

## Citation

If you use Arandu in your research, please cite the software using
[CITATION.cff](CITATION.cff) (and the restricted-access dataset at
<https://doi.org/10.5281/zenodo.24453595> when using the corpus or evaluation
outputs):

```bibtex
@software{reckziegel2026arandu,
  author    = {Reckziegel, Frederico Dal Soglio and Tavares, Anderson Rocha},
  title     = {Arandu: Open-World Knowledge Graph Construction and Evaluation from Oral Interviews},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24453915},
  url       = {https://doi.org/10.5281/zenodo.24453915},
  license   = {MIT}
}
```

## License

Released under the [MIT License](LICENSE).
