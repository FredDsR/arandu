# Shared Module: Agent Guide

Cross-domain infrastructure every pipeline depends on: the unified LLM client,
settings base classes, the composable judge pipeline, results versioning +
checkpointing, and the Phase C RAG stages (chunk / retrieve / answer /
judge-answers / analysis). Edit here with care: a change ripples into kg, qa,
transcription, and rag at once.

## Module map

| Path | Role |
| ---- | ---- |
| `config.py` | `LLMConfig`, `ResultsConfig` (shared, cross-pipeline) |
| `llm_settings.py` | `LLMSettings` base + `REASONING_MODEL_MAX_TOKENS` (8192, the shared token headroom); subclass this per stage |
| `llm_client.py` | Unified OpenAI-SDK client (`build_llm_client_from_settings`, `parse_provider`); provider dispatch + backoff. Never construct `OpenAI()` directly |
| `results_manager.py` | `ResultsManager`: versioned `results/<id>/<stage>/outputs/`, run metadata, `index.json` |
| `checkpoint.py` | `CheckpointManager`: batched save-interval state for resume |
| `schemas.py` | `InputRecord`, `EnrichedRecord`, `SourceMetadata`, `JudgeResultMixin`, run/pipeline metadata |
| `judge/` | The multi-stage judge: `JudgePipeline` → `JudgeStage` (filter/score/always) → `JudgeStep` → criteria; `LLMCriterion` + factory |
| `rag/` | Phase C: `retrieve/` (factory over bm25/atlas_rag/khop_passage/khop_triple/null), `answer/`, `judge_answers/`, `analysis/`, `retrievers/` |
| `agreement/`, `chunking/`, `embeddings/`, `emic/`, `human_eval/` | Krippendorff/Scott/Cohen agreement; chunk registry+chonkie; sentence-transformer wrappers; emic categories; human-eval sampling |

## Patterns to follow

- **New LLM-backed stage**: subclass `LLMSettings` with its own
  `env_prefix="ARANDU_<STAGE>_"`, add only the extra fields, then build the
  client with `build_llm_client_from_settings(settings)` (with `parse_provider`).
  Shared/cross-pipeline knobs go in
  `config.py` (no prefix); stage-specific knobs in that stage's `settings.py`.
- **Judge composition**: assemble `JudgePipeline(stages=[JudgeStage(name=...,
  step=JudgeStep(criteria=[...]), mode="filter"|"score"|"always")])`. A `filter`
  stage rejects on the first failing criterion; `always` runs even after an
  earlier rejection. Criteria are `LLMCriterion`s (router/facade over
  Range/Ordinal engines) loaded from `prompts/<...>/<lang>/<name>/config.json`.
- **Retrievers**: implement the `Retriever` protocol (`retriever_id`,
  `retrieve(question, top_k)`) and wire it through `rag/retrieve/factory.py`'s
  `build_retriever(arm, ...)`. Don't special-case arms at call sites.
- **Persistence**: go through `ResultsManager` (`create_run` →
  `update_progress` → `complete_run`); never hand-build a `results/` path.
- **Token budget**: default `max_tokens` to `REASONING_MODEL_MAX_TOKENS`;
  thinking models burn the budget before the JSON, so a tight cap truncates output.

## Complex logic worth knowing

- **Ordinal vs continuous criteria.** Continuous: `passed = score >= threshold`.
  Ordinal (`scale="ordinal"`): emits an integer label, runs in `score` mode only,
  and a filter stage containing an ordinal criterion is rejected at construction
  (a filter would never reject it). Emic validity is the ordinal case.
- **Response-key aliases.** Local models vary key names wildly
  (`score`/`rating`/`value`, `rationale`/`reasoning`/`justificativa`...). The
  criterion response models accept synonyms via `AliasChoices`; do not strip them
  or ~half of qwen responses fail parsing.
- **Pipeline id is run-level.** All stages of one run share `results/<id>/`; a new
  id mid-run orphans later stages from earlier outputs.

## Gotchas

- Forgetting `env_prefix` on an `LLMSettings` subclass → it silently shares the
  base prefix and reads the wrong env.
- All `*Config`/`*Settings` use `extra="ignore"`: an unknown/renamed env var is
  silently dropped (see the [deployment wiring contract](../../../scripts/slurm/AGENTS.md)).
- `CheckpointManager` with `save_interval > 1`: call `flush()` at the end or the
  last batch isn't persisted.
- `atlas_rag` retrieval needs the precomputed index from
  `arandu kg-build-retriever-index`; missing → failure at retriever construction.

**Deployment surface**: ships in both images (`arandu:latest` and
`arandu-kg:latest`) since every service imports `shared`. Full map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
