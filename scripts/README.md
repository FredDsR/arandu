# Scripts

Operational utilities for packaging, inspecting, and smoke-testing pipeline
runs. Read-only research analyses live in [analysis/](../analysis/README.md),
and cluster-specific SLURM deployment manifests live in
[`arandu-deploy`](https://github.com/FredDsR/arandu-deploy).

Run scripts from the repository root with `uv run python -m scripts.<name>` or
`uv run python scripts/<path>.py`. Contribution rules are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Tools

- [`export_zenodo_release`](export_zenodo_release.py): package a sanitized pipeline run and judge-retest replicas in native `results/` layout for Zenodo release.
- [`import_results`](import_results.py): import externally produced results into the versioned `results/<run-id>/` layout.
- [`export_cep_pairs_markdown`](export_cep_pairs_markdown.py): export CEP QA pairs to Markdown for qualitative or emic-validity review.
- [`export_rag_answers_markdown`](export_rag_answers_markdown.py): export judged RAG answers to Markdown for manual inspection.
- [`render_prompt_review`](render_prompt_review.py): render all prompt templates and criterion configs into a single Markdown snapshot.
- [`estimate_pipeline_cost`](estimate_pipeline_cost.py): estimate LLM API cost across pipeline stages before executing a run.
- [`check_failed_files`](check_failed_files.py): inspect audio and video files that failed transcription.

## Smoke Checks

Run individual components end to end on a small sample to verify wiring:

- [`smoke_atlas_rag_retriever`](smoke/smoke_atlas_rag_retriever.py): build and query the AutoSchemaKG/HippoRAG retriever against a KG.
- [`smoke_bm25_retriever`](smoke/smoke_bm25_retriever.py): build an on-disk BM25 index and query it.
- [`smoke_khop_seeding`](smoke/smoke_khop_seeding.py): verify k-hop lexical seeding against a KG.
- [`smoke_qa_judge`](smoke/smoke_qa_judge.py): run the QA judge on a sample against a live LLM endpoint.
- [`smoke_transcription_judge`](smoke/smoke_transcription_judge.py): run the transcription judge on a sample.
