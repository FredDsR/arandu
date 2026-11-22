# Analyses

Read-only analyses over finished runs under `results/<run-id>/`. No module here
makes LLM calls or modifies pipeline outputs. Run each module from the
repository root with `uv run python -m analysis.<module> --help`.

Contribution and style rules are documented in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Analysis Index

| Question / Artifact | Module | Example Command |
|---|---|---|
| Bloom-stratified CEP yield, rejection decomposition, Gini concentration, and metadata echo | [`analyze_bloom_yield`](analyze_bloom_yield.py) | `uv run python -m analysis.analyze_bloom_yield --id thesis-run-02` |
| Corpus coverage, graph representation, community/quartile retrieval, and pooled bootstraps | [`analyze_corpus_coverage`](analyze_corpus_coverage.py) | `uv run python -m analysis.analyze_corpus_coverage --id thesis-run-02 --replica results/judge-retest/r1-oracle` |
| Concentration curves and yield-by-quartile figure from corpus coverage JSON | [`plot_corpus_coverage`](plot_corpus_coverage.py) | `uv run python -m analysis.plot_corpus_coverage --input corpus_coverage.json --output corpus_coverage.pdf` |
| Retrieval-only BM25 vs. dense encoder × window grid comparison | [`analyze_dense_retrieval`](analyze_dense_retrieval.py) | `uv run python -m analysis.analyze_dense_retrieval --id thesis-run-02 --replica results/judge-retest/r1-oracle --ranks-file ranks.json` |
| Intra-judge test-retest reliability (Krippendorff's α, Gwet's AC2) and cross-arm retest tables | [`analyze_judge_retest`](analyze_judge_retest.py) | `uv run python -m analysis.analyze_judge_retest --help` |
| Over-caution split by evidence support (`passage_coverage`) | [`analyze_conditional_overcaution`](analyze_conditional_overcaution.py) | `uv run python -m analysis.analyze_conditional_overcaution --id thesis-run-02` |
| Knowledge coverage stratified by emic validity (`EV >= 4`) | [`analyze_kc_by_emic`](analyze_kc_by_emic.py) | `uv run python -m analysis.analyze_kc_by_emic --id thesis-run-02` |
| Effect of adding `EV >= 4` as a benchmark admission gate | [`analyze_emic_gate`](analyze_emic_gate.py) | `uv run python -m analysis.analyze_emic_gate --id thesis-run-02` |
| Emic-validity score distribution across Bloom levels | [`analyze_emic_scores`](analyze_emic_scores.py) | `uv run python -m analysis.analyze_emic_scores --id thesis-run-02` |
| Induced KG structural profile (components, Louvain communities, degree/hub structure) | [`kg_structure_profile`](kg_structure_profile.py) | `uv run python -m analysis.kg_structure_profile --id thesis-run-02` |
| KG summary report and structural metrics | [`kg_report`](kg_report.py), [`kg_structural_metrics`](kg_structural_metrics.py) | `uv run python -m analysis.kg_report --help` |
| Cross-arm ranking sensitivity to the `judge-qa` threshold | [`analyze_judge_thresholds`](analyze_judge_thresholds.py) | `uv run python -m analysis.analyze_judge_thresholds --id thesis-run-02` |
| `judge-qa` score distributions relative to thresholds | [`plot_judge_score_distributions`](plot_judge_score_distributions.py) | `uv run python -m analysis.plot_judge_score_distributions --id thesis-run-02` |
| RAG outcomes for pairs admitted only under a looser `judge-qa` threshold | [`analyze_qa_cohort_rag_outcome`](analyze_qa_cohort_rag_outcome.py) | `uv run python -m analysis.analyze_qa_cohort_rag_outcome --id thesis-run-02` |
| Benchmark stability across two `judge-qa` gate configurations | [`analyze_cep_gate_flip`](analyze_cep_gate_flip.py) | `uv run python -m analysis.analyze_cep_gate_flip --help` |
| Audit of rejected QA pairs near the admission threshold | [`audit_near_threshold_qa`](audit_near_threshold_qa.py) | `uv run python -m analysis.audit_near_threshold_qa --id thesis-run-02` |

Shared judge helper utilities live in [`_judge_common.py`](_judge_common.py).
