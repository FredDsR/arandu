"""Non-answerable benchmark construction (spec §7).

Derives non-answerable QA items from validated CEP pairs by swapping one
named entity for a plausible alternative that is absent from both the KG
and the source corpus. Each item keeps a ``parent_qa_pair_id`` link to
its answerable twin so the analysis stage can run paired comparisons.

Public entry point:

- :func:`run_generate_non_answerable_batch` - CLI driver.
"""

from arandu.qa.non_answerable.batch import run_generate_non_answerable_batch

__all__ = ["run_generate_non_answerable_batch"]
