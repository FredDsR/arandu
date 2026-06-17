# QA / CEP Module: Agent Guide

Cognitive Elicitation Pipeline: generate Bloom-scaffolded QA pairs from
transcriptions, then validate them with LLM-as-a-Judge. **Generation and
validation are separate steps** — `arandu generate-cep-qa` (`cli/qa.py` →
`batch.run_batch_cep_generation`) only generates; `arandu judge-qa` only judges.
Config: `QAConfig`/`CEPConfig`/`JudgeConfig` (`config.py`). Output:
`results/<id>/cep/outputs/*_cep_qa.json`.

## Module map

| Path | Role |
| ---- | ---- |
| `config.py` | `QAConfig` (`ARANDU_QA_`, provider/model), `CEPConfig` (`ARANDU_CEP_`, bloom_distribution + judge weights), `JudgeConfig` (`ARANDU_JUDGE_`, validator client) |
| `schemas.py` | `QAPairCEP` (extends `QAPair` + `JudgeResultMixin`: bloom_level, reasoning_trace, multi-hop, `validation`), `QARecordCEP` (doc wrapper, `validation_rate`) |
| `batch.py` | Orchestrator: load transcriptions, task filtering, parallel generation, checkpoint |
| `cep/generator.py` | Per-chunk driver: Module I then Module II; stamps `chunk_id` |
| `cep/bloom_scaffolding.py` | Module I: per-level generation with scaffolding context + retries |
| `cep/reasoning.py` | Module II: reasoning_trace + multi-hop detection (analyze/evaluate/create only) |
| `cep/judge.py` | `QAJudge`: wraps the shared `JudgePipeline` (stage `cep_validation`, four criteria) |
| `non_answerable/` | Derive the non-answerable benchmark (stratified seeds, perturbation, KG/corpus absence check) |

## Patterns to follow

- **`bloom_distribution` is absolute per-chunk pair counts, not weights**
  (`{"remember": 3, ...}` = 3 remember pairs per chunk). `pairs_per_chunk` is its
  sum; the document total scales with chunk count. It is the single source of
  truth — there is no `bloom_levels` field.
- **Scaffolding context**: with `enable_scaffolding_context`, levels run in Bloom
  order and each level's prompt receives prior pairs (capped at
  `max_scaffolding_pairs`), so higher levels build on lower ones.
- **Judge composition**: `QAJudge` runs one `cep_validation` filter stage with
  four `LLMCriterion`s — faithfulness, bloom_calibration, informativeness,
  self_containedness — at temp 0.3. A separate remember-only pipeline drops
  self_containedness (remember pairs are extractive).
- **Config cascade** (`cli/qa.py`): each config loads env defaults, then merges
  CLI overrides via `model_validate({**dump(), **overrides})`. `judge-qa`'s
  validator model resolves `--model` → `ARANDU_JUDGE_VALIDATOR_MODEL` → error.

## Complex logic worth knowing

- A judged pair carries `validation: JudgePipelineResult` (from
  `JudgeResultMixin`); `is_valid` is computed from `validation.passed`. There is
  no aggregate `overall_score` on the pair — the four criteria are individual
  `CriterionScore`s under the stage.
- `judge-qa` resumes by default (skips pairs that already have `validation`);
  `--rejudge` forces a fresh pass — required after changing the validator model.
- `batch.load_transcription_tasks` skips judged-failed records and re-uses the
  transcription `ContentLengthFloorCriterion` to skip too-short text (so QA
  doesn't spend LLM budget on records the transcription judge would reject).

## Gotchas

- `CEPConfig.validate_scoring_weights` requires the four weights to sum to 1.0
  (defaults 0.30/0.25/0.25/0.20); changing one means rebalancing the rest.
- `MAX_BLOOM_PAIRS_PER_CHUNK = 50` caps the distribution sum (typo guard against
  runaway cost like `remember:9999`).
- `extra="ignore"` on the configs silently drops unknown/misspelled env vars.
- `chunk_id` is stamped after enrichment; changing the chunker view between
  generations breaks provenance.

**Deployment surface**: `arandu:latest` (`Dockerfile`); `arandu-cep` service
under `cep`/`cep-gpu`, validation via the `arandu-judge` service. Full map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
