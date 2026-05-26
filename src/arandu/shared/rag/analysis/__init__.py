"""``arandu rag-analysis`` machinery — confusion-matrix + per-arm metrics over judged AnswerRecords.

Spec §8. Public entry points:

- :func:`run_rag_analysis_batch` — CLI driver.
- :func:`classify_record` — TA/TC/FA/FC labelling per spec §8.1.
- :func:`wilson_ci` — Wilson 95% confidence interval for a binary proportion.

Out of scope in this PR's first cut (deferred to follow-ups):

- Paired McNemar's tests (needs ``statsmodels`` dep).
- Paired bootstrap CIs.
- Methodology §6 composite score (needs KG structural metrics integration).
- ``--compare-runs`` cross-run hook.
- Matplotlib figures.
"""

from arandu.shared.rag.analysis.batch import run_rag_analysis_batch
from arandu.shared.rag.analysis.classifier import classify_record
from arandu.shared.rag.analysis.wilson import wilson_ci

__all__ = ["classify_record", "run_rag_analysis_batch", "wilson_ci"]
