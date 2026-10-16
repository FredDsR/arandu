# QA / CEP Module: Agent Guide

Cognitive Elicitation Pipeline: generate Bloom-scaffolded QA pairs from
transcriptions, then validate them with LLM-as-a-Judge. `arandu generate-cep-qa`
(`cli/qa.py` → `batch.run_batch_cep_generation`) generates pairs and, with the
scaffolding gate on (default), judges each one before it can scaffold later pairs,
persisting the verdict; `arandu judge-qa` judges whatever has no clean verdict (or
everything with `--rejudge`).
Config: `QAConfig`/`CEPConfig`/`JudgeConfig` (`config.py`). Output:
`results/<id>/cep/outputs/*_cep_qa.json`.

## Module map

| Path | Role |
| ---- | ---- |
| `config.py` | `QAConfig` (`ARANDU_QA_`, provider/model), `CEPConfig` (`ARANDU_CEP_`, bloom_distribution + judge weights), `JudgeConfig` (`ARANDU_JUDGE_`, validator client) |
| `schemas.py` | `QAPairCEP` (extends `QAPair` + `JudgeResultMixin`: bloom_level, reasoning_trace, multi-hop, `validation`), `QARecordCEP` (doc wrapper, `validation_rate`) |
| `batch.py` | Orchestrator: load transcriptions, task filtering, parallel generation, checkpoint |
| `cep/generator.py` | Per-chunk driver: runs Module I; stamps `chunk_id` |
| `cep/bloom_scaffolding.py` | Module I: per-level generation with scaffolding context + retries |
| `cep/judge.py` | `QAJudge`: wraps the shared `JudgePipeline` (stage `cep_validation`, four criteria) |
| `non_answerable/` | Derive the non-answerable benchmark (stratified seeds, perturbation, KG/corpus absence check) |

## Patterns to follow

- **`bloom_distribution` is absolute per-chunk pair counts, not weights**
  (`{"remember": 3, ...}` = 3 remember pairs per chunk). `pairs_per_chunk` is its
  sum; the document total scales with chunk count. It is the single source of
  truth — there is no `bloom_levels` field.
- **Scaffolding context**: with `enable_scaffolding_context`, levels run in Bloom
  order and each level's prompt receives prior pairs (capped at
  `max_scaffolding_pairs`), so higher levels build on lower ones. With
  `enable_scaffolding_gate` (default on, needs `ARANDU_JUDGE_VALIDATOR_MODEL`),
  only pairs with `is_valid is True` are eligible; rejected pairs stay in the
  output. The validator is injected as a `PairValidator` built in
  `batch._build_cep_generator`.
- **Judge composition**: `QAJudge` runs one `cep_validation` filter stage with
  four `LLMCriterion`s (faithfulness, bloom_calibration, informativeness,
  self_containedness) at temp 0.1, the `JudgeConfig.temperature` default. A separate remember-only pipeline drops
  self_containedness (remember pairs are extractive).
- **Config cascade** (`cli/qa.py`): each config loads env defaults, then merges
  CLI overrides via `model_validate({**dump(), **overrides})`. `judge-qa`'s
  validator model resolves `--model` → `ARANDU_JUDGE_VALIDATOR_MODEL` → error.

## Complex logic worth knowing

- A judged pair carries `validation: JudgePipelineResult` (from
  `JudgeResultMixin`); `is_valid` is computed from `validation.passed`. There is
  no aggregate `overall_score` on the pair — the four criteria are individual
  `CriterionScore`s under the stage.
- `judge-qa` resumes by default (skips pairs with a clean verdict; a verdict with a
  judge error, see `has_judge_error`, is re-judged); `--rejudge` forces a fresh
  pass — required after changing the validator model.
- `batch.load_transcription_tasks` skips judged-failed records and re-uses the
  transcription `ContentLengthFloorCriterion` to skip too-short text (so QA
  doesn't spend LLM budget on records the transcription judge would reject).

## Gotchas

- `CEPConfig` has no thresholds or weights; the only gate setting is
  `enable_scaffolding_gate`. A pair passes when **every** criterion
  passes (`JudgeStepResult.passed` is an `all()`), each against the `threshold`
  in `prompts/judge/criteria/<criterion>/config.json`. There is no aggregate
  score and no criterion weights; zeroing a criterion is not possible from config.
- The generation parse (`bloom_scaffolding.py`) always sets `validation` to
  `None`: a verdict echoed by the LLM is discarded, only the judge writes one.
- `max_scaffolding_pairs` caps the pairs left after the gate filter
  (`_scaffolding_pool` runs before `_format_prior_pairs`), not the raw prior pairs.
- `MAX_BLOOM_PAIRS_PER_CHUNK = 50` caps the distribution sum (typo guard against
  runaway cost like `remember:9999`).
- When generation and the gate's validator use different Ollama models in the same
  sidecar, both weights must fit in VRAM alongside the KV cache; otherwise Ollama
  swaps models between every generate and judge call.
- `extra="ignore"` on the configs silently drops unknown/misspelled env vars.
- `chunk_id` is stamped after generation; changing the chunker view between
  generations breaks provenance.

**Deployment surface**: `arandu:latest` (`Dockerfile`); `arandu-cep` service
under `cep`/`cep-gpu` (the scaffolding gate runs inside it with
`ARANDU_JUDGE_*`); the `arandu-judge` service runs `judge-qa`. Full map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
