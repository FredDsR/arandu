# SLURM Scripts: Agent Guide

Job scripts for the pcad cluster (`pcad.inf.ufrgs.br`). Each pipeline step has a
directory with thin partition-specific scripts that source a shared
`<step>_common.sh`. The cluster checkout at `~/etno-kgc-preprocessing/` is a
flat working copy with NO `.git`: deploy changes via `rsync` + `md5sum` verify,
never `git pull`.

## How the layers connect (code ↔ config ↔ compose ↔ image ↔ SLURM)

Every step is the same chain. A SLURM partition script exports env and sources
its `<step>_common.sh`; the common script runs `docker compose --profile <profile>
up <service>`; the service (in `docker-compose.yml`) runs `arandu <command>` off
the image `ENTRYPOINT`; the CLI loads a `*Config` class from `ARANDU_<PREFIX>_*`
env and writes `results/<id>/<stage>/outputs/`.

| Step | `arandu` command(s) | Config class · env prefix | Compose service | Profiles | Image · Dockerfile | SLURM dir |
| ---- | ------------------- | ------------------------- | --------------- | -------- | ------------------ | --------- |
| Transcription | `batch-transcribe` | `TranscriberConfig` · `ARANDU_` | `arandu` / `arandu-cpu` / `arandu-rocm` | (runtime GPU) | `arandu:latest` · `Dockerfile`; `arandu:rocm` · `Dockerfile.rocm` | `transcription/` |
| QA | `generate-qa` †legacy | `QAConfig` · `ARANDU_QA_` | `arandu-qa` | `qa` / `qa-gpu` | `arandu:latest` · `Dockerfile` | `qa/` |
| CEP | `generate-cep-qa` | `QAConfig` + `CEPConfig` · `ARANDU_QA_`, `ARANDU_CEP_` | `arandu-cep` | `cep` / `cep-gpu` | `arandu:latest` · `Dockerfile` | `cep/` |
| Judge | `judge-transcription`, `judge-qa` | `JudgeConfig` (+ `CEPConfig` weights) · `ARANDU_JUDGE_` | `arandu-judge` | `judge` / `judge-gpu` | `arandu:latest` · `Dockerfile` | `judge/{transcription,qa}/` |
| KG | `build-kg`, `kg-link-passages`, `kg-build-retriever-index` | `KGConfig` · `ARANDU_KG_` | `arandu-kg` | `kg` / `kg-gpu` | `arandu-kg:latest` · `Dockerfile.kg` | `kg/` |
| RAG (Phase C) | `chunk`, `retrieve`, `answer`, `judge-answers`, `generate-non-answerable`, `rag-analysis` | rag settings + `RAG_*` runner vars | `arandu-rag` / `arandu-rag-cpu` | `rag` / `rag-gpu` / `rag-cpu` | `arandu-kg:latest` · `Dockerfile.kg` | `rag/` |
| Evaluation | `evaluate` †legacy | `EvaluationConfig` · `ARANDU_EVAL_` | `arandu-eval` | `evaluate` | `arandu:latest` · `Dockerfile` | `evaluation/` |

> **†legacy**: the `arandu-qa` (`generate-qa`) and `arandu-eval` (`evaluate`)
> compose services name commands that are **not currently registered** in
> `cli/app.py`, so they will not run as written. The live QA path is
> `generate-cep-qa` (+ `judge-qa`); Phase C retrieval evaluation is the `rag-*`
> chain. The rows are kept because the compose services + SLURM dirs still exist.

Config-class fields + env prefixes are the contract; see
[docs/user-guide/configuration.md](../../docs/user-guide/configuration.md).

**Two images.** `arandu:latest` (`Dockerfile`) covers transcription/qa/cep/judge/eval.
`arandu-kg:latest` (`Dockerfile.kg` = base **+ `uv sync --extra kg`**, i.e. atlas-rag)
covers **kg and rag**. A dep bump in the `kg` extra needs a `Dockerfile.kg` rebuild,
not the base one. Both images `COPY` `src/`, `pyproject.toml`+`uv.lock`, and `prompts/`
at build time, so a deploy only takes effect after a rebuild (this is why the deploy
rules below ship those trees).

**An env var only takes effect when all three agree:** (1) it is a real field on the
step's `*Config` class, (2) `docker-compose.yml` forwards it in the service's
`environment:`, and (3) the SLURM script exports it. Break any link and the var is
silently ignored (`*Config` uses `extra="ignore"`). This has bitten us: a stale
`ARANDU_CEP_VALIDATOR_*` once lingered in the CEP script + compose with no matching
config field, so it did nothing — CEP-pair validation lives in the separate
`judge-qa` step under `ARANDU_JUDGE_*`, not in `generate-cep-qa`.

## Layout

| Path | Role |
| ---- | ---- |
| `transcription/<partition>.slurm` + `job_common.sh` | Whisper transcription (GPU) |
| `cep/`, `qa/`, `judge/` + `*_common.sh` | CEP/QA generation and LLM judges |
| `kg/<partition>.slurm` + `kg_common.sh` | atlas-rag KG construction (extraction + concept gen) |
| `rag/<stage>.slurm` + `rag_common.sh` | Phase C eval chain (chunk, link-passages, retriever-index, non-answerable, retrieve, answer, judge-answers, rag-analysis) |
| `general/cleanup.slurm`, `kg/pipeline-cleanup.slurm` | Docker/disk cleanup jobs |
| `dashboard.py`, `watch-job.sh` | Local monitoring helpers |

The `rag/` per-stage scripts set `RAG_CLI_ARGS` (the `arandu` subcommand) and
`RAG_NEEDS_OLLAMA`, then source `rag_common.sh`. CPU-only stages override
`RAG_SERVICE=arandu-rag-cpu` / `RAG_PROFILE=rag-cpu` to avoid GPU contention.

## Submitting

```bash
PIPELINE_ID=<run-id> [overrides] sbatch [--exclude=<nodes>] scripts/slurm/<step>/<partition>.slurm
```

| Variable | Default | Notes |
| -------- | ------- | ----- |
| `PIPELINE_ID` | empty | REQUIRED. Without it output lands in an auto-generated id (e.g. `20260604_210325_local`) |
| `ARANDU_KG_MODEL_ID` | `llama3.1:8b` | Override for non-default models (e.g. `qwen3:14b`) |
| `RAG_OLLAMA_MODEL` | `qwen3:14b` | Model pulled by `rag_common.sh` LLM stages |
| `JUDGE_REJUDGE` | unset | Force re-judge instead of resume |
| `MIN_DISK_GB` | 15 | Disk-floor preflight on the Docker root partition |
| `USE_GPU_OLLAMA` | set by partition script | Selects `kg-gpu` vs `kg` compose profile |

## Partitions: access + GPU (confirmed 2026-06-15 — do NOT re-test)

- **tupi** — the ONLY GPU partition we can use. Every GPU/ollama stage
  (build-kg, cep, judge-qa, answer, judge-answers, atlas_rag retrieve) runs
  here. Frequently saturated by other users; when it is, **wait** — there is
  no faster GPU alternative for this account.
- **draco** — **CPU-only, no GPU.** Idle and fast to schedule; use it for
  CPU-only work (root-container result prep, the khop retrieve, rag-analysis,
  chunk, kg-link-passages). NEVER submit a GPU/ollama stage here.
- **grace** — NO ACCESS (uid not in the allowed group); jobs PEND forever
  with `uid_..._not_in_group_permitted_to_use_this_partition`. Do not submit.

Never reroute a stuck GPU job to grace or draco to "beat the queue" — grace is
denied and draco has no GPU. The lever is: wait, or ask the user.

## Node selection (tupi partition)

Prefer `--exclude=<broken-nodes>` over pinning a node with `--nodelist`: a pin
queues behind whatever occupies that node; an exclude grabs any healthy one.
Known failure modes: broken Docker socket, full root FS (`/var/lib/docker`).
Observed state 2026-06-11 (verify before trusting): tupi1/tupi2 good,
tupi3/tupi6 docker-broken, tupi5 disk-full, tupi4 untested.

## What a job run looks like

`kg_common.sh` (and the analogous common scripts) execute, in order:

1. Aggressive Docker cleanup: `docker system prune -af --volumes` plus builder
   prune. Named volumes on the node DO get removed. Partial ollama downloads
   (`*-partial`, `*.tmp`) are deleted; installed ollama models not required by
   this job are removed.
2. Disk preflight: abort if the Docker root partition has under `MIN_DISK_GB`.
3. Image build, ollama sidecar startup (30 x 5s readiness retries), model pull
   (qwen3:14b is ~9.3 GB; the first run on a freshly pruned node re-pulls it).
4. The stage container runs with `--abort-on-container-exit`, then compose down.

## What to expect (read this before declaring a job hung)

- Logs land in `logs/<step>_<partition>_<jobname>_<jobid>.{out,err}`.
- Container log timestamps are UTC; the head node is UTC-3.
- **A silent log is not a hang.** Long LLM loops (KG concept generation
  especially) buffer stdout and write progress to result files instead. Check
  artifact mtimes under `results/<id>/...` first, or attach to the allocation:

  ```bash
  srun --jobid=<jobid> --overlap bash -c \
    'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; \
     docker logs --tail 15 ollama-gpu-<jobid>'
  ```

- Wall limits are 24h on tupi. The KG step survives a timeout: extraction and
  concept generation both checkpoint (see `src/arandu/kg/AGENTS.md`), so the
  correct response to a `TIMEOUT` is to resubmit the exact same command.
- `results/` is root-owned (containers write as root). The head-node user
  cannot edit it directly; mutate it via a root container, e.g.
  `srun ... docker run --entrypoint python3 -v .../results:/app/results arandu:latest ...`.

## Operational rules for agents

- One bundled ssh per checkpoint. Repeated ssh connections cause the head node
  to close ALL of the user's sessions. Never poll in a loop.
- Every deploy is `rsync` the specific files + `md5sum` verify on both ends.
  A code deploy is NOT just `src/`: ship `pyproject.toml` + `uv.lock` when
  deps changed (stale manifests broke cep once) and `prompts/` when criteria
  changed (a missing criterion config.json broke judge-answers once - the
  image COPYs both trees from the cluster checkout at build time).
- Before resubmitting a "failed" KG job fresh, check
  `results/<id>/kg/outputs/atlas_output/concepts/` for partial state: a
  resubmit resumes automatically and a fresh start throws hours away.
