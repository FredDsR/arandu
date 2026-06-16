---
title: CEP QA Generation
description: Cognitive Elicitation Pipeline generation flow and post-generation judging.
---

CEP generation creates Bloom-scaffolded QA pairs with reasoning-oriented metadata.
Quality judging is performed afterward using `judge-qa`.

## Workflow

```bash
# Generate CEP QA pairs
arandu generate-cep-qa results/ --output-dir cep_dataset/

# Judge generated QA pairs
arandu judge-qa cep_dataset/ --model qwen3:14b
```

## CEP modules

1. Bloom-level scaffolding (`remember` → `evaluate` by default)
2. Reasoning/tacit enrichment (reasoning traces, multi-hop markers)
3. Independent judge stage (`judge-qa`) persisted per pair in `validation`

## Generation configuration

Main `ARANDU_CEP_*` knobs:

- `ARANDU_CEP_BLOOM_LEVELS`
- `ARANDU_CEP_BLOOM_DISTRIBUTION`
- `ARANDU_CEP_ENABLE_REASONING_TRACES`
- `ARANDU_CEP_MAX_HOP_COUNT`
- `ARANDU_CEP_REASONING_MAX_TOKENS`
- `ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT`
- `ARANDU_CEP_MAX_SCAFFOLDING_PAIRS`
- `ARANDU_CEP_ENABLE_SOURCE_METADATA_CONTEXT`
- `ARANDU_CEP_LANGUAGE`

Legacy weighted-score fields still exist in `CEPConfig` for compatibility/reporting:

- `ARANDU_CEP_VALIDATION_THRESHOLD`
- `ARANDU_CEP_FAITHFULNESS_WEIGHT`
- `ARANDU_CEP_BLOOM_CALIBRATION_WEIGHT`
- `ARANDU_CEP_INFORMATIVENESS_WEIGHT`
- `ARANDU_CEP_SELF_CONTAINEDNESS_WEIGHT`

## Judge configuration

`judge-qa` uses `ARANDU_JUDGE_*` settings (model/provider/base URL/language/token/temperature)
and can run in resume mode or full rejudge mode.

## Output layout

```text
results/<pipeline_id>/cep/outputs/
├── <file_id>_cep_qa.json
└── cep_checkpoint.json
```

After `judge-qa`, each QA pair in `*_cep_qa.json` can contain:

- `validation: JudgePipelineResult`
- computed `is_valid`

---

See also: [QA Generation](/guides/qa-generation/) · [Transcription Judging](/guides/transcription-validation/)
