# Analyses

Read-only analyses of finished runs. Run each one from the repo root with
`uv run python -m analysis.<module> --id <run-id>`; outputs default to
`results/<run-id>/analysis/`. Rules for adding one are in [AGENTS.md](AGENTS.md).

| Question | Module or notebook | Cited in |
|---|---|---|
| Is the cross-arm ranking robust to the judge-qa threshold? | [`analyze_judge_thresholds`](analyze_judge_thresholds.py) | [thesis-run-02 §4.2, §4.4](../docs/research/results/thesis-run-02.md) |
| Where do judge-qa scores sit relative to the threshold? | [`plot_judge_score_distributions`](plot_judge_score_distributions.py) | [thesis-run-02 §4.4](../docs/research/results/thesis-run-02.md) |
| How does the cohort admitted only at threshold 0.5 fare in RAG? | [`analyze_qa_cohort_rag_outcome`](analyze_qa_cohort_rag_outcome.py) | [thesis-run-02 §4.6](../docs/research/results/thesis-run-02.md) |
| What do the emic-validity scores look like? | [`analyze_emic_scores`](analyze_emic_scores.py) | [thesis-run-02 §4.7](../docs/research/results/thesis-run-02.md) |
| What is the structure of a run's knowledge graph? | [`kg_structure_profile`](kg_structure_profile.py) | [thesis-run-02 §4.9](../docs/research/results/thesis-run-02.md) |
| How stable is the benchmark under two judge-qa gates? | [`analyze_cep_gate_flip`](analyze_cep_gate_flip.py) | [thesis-run-02 §5](../docs/research/results/thesis-run-02.md) |
| How reliable are the judges on retest? | [`analyze_judge_retest`](analyze_judge_retest.py) | |
| What changes if useful pairs need emic validity >= 4? | [`analyze_emic_gate`](analyze_emic_gate.py) | |
| Does knowledge coverage depend on the reference pair's emic validity? | [`analyze_kc_by_emic`](analyze_kc_by_emic.py) | |
| Is over-caution explained by unsupported evidence? | [`analyze_conditional_overcaution`](analyze_conditional_overcaution.py) | |
| Which rejected QA pairs missed the threshold narrowly? | [`audit_near_threshold_qa`](audit_near_threshold_qa.py) | |
| What did the KG pipeline produce? | [`kg_report`](kg_report.py), [`kg_structural_metrics`](kg_structural_metrics.py) | |
| How do Bloom levels approve, chunk by chunk? | [`bloom.ipynb`](notebooks/bloom.ipynb), [`bloom-thesis-run-02.ipynb`](notebooks/bloom-thesis-run-02.ipynb) | |

Shared judge helpers live in [`_judge_common`](_judge_common.py).
