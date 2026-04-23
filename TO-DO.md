# TO-DO

Items not tracked elsewhere (issues or roadmap).

## Bad Transcription Records

- [ ] Checkout record `1PHylF_SMTsDwuj-iYnxzGDxophmPhKdk` — seems to be a bad transcription
- [ ] Checkout record `1_8u1qiKm5lvpVp6p6yvtWwXqMksr6vI7` — repetition of "Eu não falei muito no chão"
- [ ] Checkout record `17iWGMvnk4ajK2RT0rE3eyP9f4IEcx7bd` — repeats a sentence over and over

## CEP QA Prompt Refinement

- [ ] Check the self-containedness prompt to catch places without names (example: test-cep-01 | 20250531_165132.mp4 | understand)

## QA Domain — Use `generate_structured()`

- [ ] `qa/cep/bloom_scaffolding.py` — replace manual `json.loads()` + `_parse_response()` with `generate_structured()` and Pydantic response models
- [ ] `qa/cep/reasoning.py` — same pattern, replace manual JSON parsing with `generate_structured()`

## Shared Task Processing

- [ ] **Shared `BatchProcessor` abstraction** — `qa/batch.py`, `kg/batch.py`, and `transcription/` all duplicate the same pattern: load config → load records → filter via checkpoint → process items with error handling → track progress → save results. Consider extracting a shared `BatchProcessor` or `PipelineRunner` in `shared/` when Phase C (RAG evaluation) adds a third consumer and the real shared pattern is clear. Each domain currently has slightly different needs (KG = all-at-once batch, QA = per-record, transcription = audio-specific retries), so premature abstraction is risky.

## Judge Refactor — Future Improvements

- [ ] Abstract validate/validate_batch into `BaseJudge` with generic item processing (`_extract_kwargs`, `_wrap_result`) when second domain judge exists (TranscriptionJudge). See sub-project 2 in [#79](https://github.com/FredDsR/arandu/issues/79) and `docs/planning/2026-04-06-judge-refactor-design.md`.
