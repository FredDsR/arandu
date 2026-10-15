# Scripts

Code that operates the pipeline. Analyses of finished runs live in
[`analysis/`](../analysis/README.md). Rules for adding a script are in
[AGENTS.md](AGENTS.md). Run from the repo root with
`uv run python -m scripts.<name>`.

## Tools

- [`import_results`](import_results.py): import externally produced results into the versioned results layout.
- [`export_cep_pairs_markdown`](export_cep_pairs_markdown.py): export every CEP QA pair to Markdown for emic-validity review.
- [`export_rag_answers_markdown`](export_rag_answers_markdown.py): export every judged RAG answer to Markdown for manual review.
- [`render_prompt_review`](render_prompt_review.py): render all pt prompts and criteria configs into one Markdown snapshot.
- [`estimate_pipeline_cost`](estimate_pipeline_cost.py): estimate full-pipeline cost before a run.
- [`check_failed_files`](check_failed_files.py): inspect audio and video files that failed transcription.

## Cluster

- [`slurm/`](slurm/AGENTS.md): job scripts for the pcad cluster.

## Smoke checks

Run one component end to end on a small real sample.

- [`smoke_atlas_rag_retriever`](smoke/smoke_atlas_rag_retriever.py): build and query the AtlasRag retriever against a real KG.
- [`smoke_bm25_retriever`](smoke/smoke_bm25_retriever.py): build a real on-disk BM25 index and query it.
- [`smoke_khop_seeding`](smoke/smoke_khop_seeding.py): compare old and new k-hop seeding on the test-kg-04 KG.
- [`smoke_qa_judge`](smoke/smoke_qa_judge.py): run the QA judge on a sample against a real LLM.
- [`smoke_transcription_judge`](smoke/smoke_transcription_judge.py): run the transcription judge on a sample.

## One-offs

Tied to one run or migration. Delete each once its run is retired.

- [`cap_cep_pairs`](oneoff/cap_cep_pairs.py): cap each CEP file to a Bloom-stratified sample (dry-run helper).
- [`verify_cep_load`](oneoff/verify_cep_load.py): check that every capped CEP file loads through `QARecordCEP`.
- [`migrate_chunk_id_namespace`](oneoff/migrate_chunk_id_namespace.py): move a run's `chunk_id` namespace onto the canonical coordinate space.
- [`_khop_revalidate_prep`](oneoff/_khop_revalidate_prep.py): back up and clear k-hop state before re-validation (runs inside a cluster container).
- [`kg_relabel_predicate`](oneoff/kg_relabel_predicate.py): relabel atlas-rag's hardcoded event-participation predicate in a GraphML.
- [`seed_thesis_transcripts`](oneoff/seed_thesis_transcripts.py): seed a fresh thesis run with reusable transcripts.
