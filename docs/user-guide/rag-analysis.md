# RAG Analysis Guide

Interpret the cross-arm comparison report produced by `arandu rag-analysis`.

This is the **last** stage of the Phase C RAG-evaluation pipeline. It reads
the judged answers from `arandu judge-answers` and emits per-arm tables you
can compare across retrieval strategies (BM25, atlas-rag, NetworkX subgraph,
NetworkX triple, Null).

> **What changed (2026-05-26)**: Phase C replaces the legacy QA / KG
> evaluation tables ([evaluation.md](evaluation.md)) for retrieval comparison.
> Use `rag-analysis` when you want to compare retrievers; use `evaluation.md`
> for QA-dataset and KG-graph quality (unchanged).

---

## Quick start

```bash
# Prerequisite: judge_answers must be populated
arandu judge-answers --id <pipeline_id>

# Aggregate and report
arandu rag-analysis --id <pipeline_id>
```

Outputs land under `results/<pipeline_id>/analysis/outputs/`:

```
analysis/
└── outputs/
    ├── report.json    # Typed AnalysisReport (programmatic)
    └── tables.md      # Thesis-ready Markdown tables
```

The CEP outputs (`results/<pipeline_id>/cep/outputs/`) are consulted for
Bloom-level and question-type cross-cuts. If they're absent the run still
succeeds — you get the joint table only and a warning in the log.

---

## The four classes (TA / TC / FA / FC)

Every judged answer is sorted into one of four cells by crossing two binary
signals: the question's ground-truth `is_answerable` flag and a unified
"abstained" signal that combines the answerer's `abstained` field with the
abstention judge's verdict.

|                      | **Answerable**                             | **Non-answerable**                |
|----------------------|--------------------------------------------|-----------------------------------|
| **Abstained**        | False Abstention (**FA**) — over-cautious  | True Abstention (**TA**) — correct |
| **Committed**        | True Commitment (**TC**) — scored by KC    | False Commitment (**FC**) — **hallucination** |

A record is "abstained for analysis" only when **both** the answerer and the
judge agree. If they disagree the record is treated as committed (the
disagreement is independently logged by `judge-answers` in
`abstention_audit.jsonl`).

**`unknown`** is a fifth class for records where the abstention judge errored
or did not run. Unknown records are kept in the confusion-matrix count but
excluded from rate and mean denominators — they would otherwise skew the
headline numbers from a single judge failure.

---

## Reading Table 1 (per-arm joint metrics)

This is the headline thesis table. Each row is one retrieval arm; columns
are the per-arm rates and means.

| Column | Direction | Formula | What it tells you |
|---|---|---|---|
| **n_TA+TC** | — | TC + FA count | Size of the answerable subset that produced data (TC) plus over-caution (FA). |
| **n_TA+FC** | — | TA + FC count | Size of the non-answerable subset that produced data. |
| **KC ↑** | higher is better | `mean(correctness × faithfulness)` over TC | Knowledge coverage — does the system get the *right* answer using the *retrieved* context? |
| **Hallucination ↓** | lower is better | `FC / (FC + TA)` | When the question is unanswerable, how often does the system fabricate an answer? |
| **Over-caution ↓** | lower is better | `FA / (FA + TC)` | When the question is answerable, how often does the system refuse? |
| **Abstention F1 ↑** | higher is better | F1 over TA / FA / FC | One number summarising "how well does this arm know when to abstain?" |
| **Passage cov ↑** | higher is better | `mean(passage_coverage)` | LLM-judged: do the retrieved passages contain the supporting evidence? |

Each proportion cell is rendered as `value [lower, upper]` — the Wilson 95%
confidence interval. `n/a` means the denominator was zero (e.g. an arm with
no non-answerable items has no hallucination rate to compute).

### What "good" looks like

| Metric | Strong | Weak |
|---|---|---|
| KC | > 0.65 | < 0.40 |
| Hallucination | < 0.10 | > 0.30 |
| Over-caution | < 0.15 | > 0.40 |
| Abstention F1 | > 0.75 | < 0.50 |
| Passage coverage | > 0.65 | < 0.40 |

These ranges are heuristic targets for the ethnographic corpus — they are
**not** thresholds for accepting an arm. The thesis claim is comparative
(arm A vs arm B on the same benchmark), not absolute.

### Knowledge Coverage (KC) deep dive

`KC = correctness × faithfulness` is the multiplicative form preserved from
methodology §6. The product means **both factors must be high** to score
well:

- Correctness 0.9, faithfulness 0.4 → KC 0.36 (faithful retrieval, wrong answer).
- Correctness 0.5, faithfulness 0.9 → KC 0.45 (close answer, supported by passages).
- Correctness 0.9, faithfulness 0.9 → KC 0.81 (the target).

The mean is taken over **TC records only** — abstained records don't contribute
to KC because there's no answer to score.

### The hallucination / over-caution trade-off

These two rates are in tension. An arm that abstains aggressively will have
**low hallucination** (almost no FC) but **high over-caution** (lots of FA).
An arm that never abstains will have the opposite. **Abstention F1 ↑**
balances them into one number; use the individual rates when you care about
the underlying behaviour.

---

## Reading confidence intervals

The Wilson 95% CI tells you the range in which the true proportion likely
sits given the sample size. Two rules of thumb:

1. **Overlapping CIs ≈ no significant difference.** If arm A reports
   `0.12 [0.08, 0.17]` and arm B reports `0.15 [0.10, 0.20]`, the bands
   overlap — you can't conclude A < B from this run.
2. **Narrow CIs require many records.** If `n_TA+FC = 50` the bands will
   be wide regardless of how low the rate is. Don't read precision into
   small strata.

For rigorous pairwise testing the spec calls for paired McNemar's tests
(§8.4), which is deferred to a follow-up PR (needs `statsmodels`). For now,
treat CI overlap as the test.

---

## Bloom-stratified KC (Table 2)

The same KC metric, re-computed per Bloom-taxonomy level
(`remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`).

Use this table to answer questions like:

- *Does the triple-arm retriever underperform on lower-Bloom (factual recall) questions?*
- *Does atlas-rag's advantage concentrate on multi-hop (`analyze` / `evaluate`) items?*

Cells are scalar KC means — no confidence intervals here. With small per-stratum
`n`, single-cell comparisons are suggestive, not conclusive.

---

## Question-type-stratified KC (Table 3)

Same idea, stratified by `question_type` (`factual`, `conceptual`,
`temporal`, `entity`). Useful when the thesis claim concerns *what kind* of
question favours which retrieval paradigm.

---

## Output schema (`report.json`)

```python
from arandu.shared.rag.analysis.report import AnalysisReport

report = AnalysisReport.model_validate_json(
    open("results/<id>/analysis/outputs/report.json").read()
)

for arm_id, metrics in report.joint.items():
    print(arm_id, metrics.confusion, metrics.knowledge_coverage.mean)
    # metrics.hallucination_rate.{value, ci_lower, ci_upper, numerator, denominator}
    # metrics.over_cautiousness_rate.*
    # metrics.abstention_precision.*  /  abstention_recall.*  /  abstention_f1
    # metrics.answer_correctness.{mean, n}  /  answer_faithfulness  /  passage_coverage

# Cross-cuts
report.by_bloom["bm25"]["remember"].knowledge_coverage.mean
report.by_question_type["atlas_rag"]["conceptual"].knowledge_coverage.mean
```

`ProportionMetric.value` is `None` when the denominator is zero (rendered as
`n/a` in `tables.md`). `MeanMetric.mean` is `None` when no records contributed.

---

## When something looks wrong

| Symptom | Likely cause | What to check |
|---|---|---|
| All `n/a` in a row | The arm produced no judged records | `results/<id>/judge_answers/outputs/<arm>/` exists and is non-empty |
| Many `unknown` records | Abstention judge errored frequently | Inspect `judge_answers/outputs/abstention_audit.jsonl` and the judge logs |
| Bloom / question-type tables missing | CEP outputs absent | `results/<id>/cep/outputs/` should hold the `QARecordCEP` files |
| KC near 0 for every arm | Either correctness or faithfulness collapsed | Sample a few judged records and inspect their `validation.stage_results` |
| Hallucination = 0 but over-caution = 1.0 | Answerer abstaining on everything | Check the Answerer prompt and the abstention threshold (`ARANDU_JUDGE_ANSWERS_*`) |
| All arms tie within CI | Either sample too small or arms truly indistinguishable | Raise the benchmark size; check `n_TA+TC` and `n_TA+FC` in Table 1 |

---

## Known limitations

These are intentional first-cut scope — tracked for follow-up PRs:

- **Paired McNemar's tests** for arm-vs-arm significance (spec §8.4). Needs
  `statsmodels`; today, use CI overlap as a heuristic.
- **Paired bootstrap CIs** for continuous metrics (spec §8.5). Same
  dependency story.
- **Methodology §6 composite score** (`0.40·KC + 0.20·D_e + 0.20·RD + 0.20·CR`).
  Needs KG structural metrics joined onto the analysis report.
- **`--compare-runs run_a run_b`** for cross-run composite-score comparison.
- **Matplotlib figures.**
- **Non-answerable item handling**: the benchmark currently includes
  non-answerable items only when the CEP / non-answerable source provides
  them; the wider experiment is tracked under the `nonanswerable-experiment` task.
- **Deterministic offset-overlap** (`offset_coverage`) was dropped from the
  judge stage during PR #110 review — the thesis question is semantic, not
  extractive, and the triple-arm retriever has no character offsets. The
  judge LLM scores passage coverage instead; `passage_coverage` in the
  report comes from that judge.

---

## Related

- [CLI reference](cli-reference.md) — full command list.
- [Evaluation](evaluation.md) — legacy QA + KG evaluation (unchanged).
- [`docs/superpowers/specs/2026-05-08-phase-c-rag-evaluation-design.md`](../superpowers/specs/2026-05-08-phase-c-rag-evaluation-design.md)
  — Phase C design spec; §8 is the analysis-stage methodology.
