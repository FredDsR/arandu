"""CLI command: ``arandu retrieve`` — drive Phase C retriever arms over a run.

Iterates a run's CEP QA + (when present) non-answerable items, runs
retrieval per requested arm, and persists one :class:`RetrievalRecord`
per (arm, question) tuple under
``results/<id>/retrieve/outputs/<arm>/<source>/``.
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from arandu.shared.rag.retrieve.batch import run_retrieve_batch
from arandu.shared.rag.retrieve.settings import (
    ALL_ARMS,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)
from arandu.utils.logger import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


# All ALL_ARMS members EXCEPT atlas_rag; atlas_rag joins the default set in a
# follow-up PR once its LLM-client wiring lands.
_DEFAULT_ARMS: tuple[str, ...] = tuple(a for a in ALL_ARMS if a != "atlas_rag")


def retrieve(
    pipeline_id: Annotated[
        str,
        typer.Option(
            "--id",
            help=(
                "Pipeline ID for the run. The cep/ stage must already be populated; "
                "each arm has its own prerequisites (chunk/ for bm25, kg/ for k-hop)."
            ),
        ),
    ],
    arms: Annotated[
        list[str] | None,
        typer.Option(
            "--arm",
            help=(
                "Retriever arm; repeatable. Known: "
                + ", ".join(ALL_ARMS)
                + f". Defaults to: {', '.join(_DEFAULT_ARMS)}. "
                + "atlas_rag is currently disabled (LLM-client wiring deferred)."
            ),
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Passages per question per arm.",
            min=1,
        ),
    ] = 10,
    rebuild_index: Annotated[
        bool,
        typer.Option(
            "--rebuild-index",
            help=(
                "Force-rebuild arm-side indexes (BM25 pickle only; "
                "k-hop arms build state in-memory at construction)."
            ),
        ),
    ] = False,
) -> None:
    """Run Phase C retrievers over a populated run.

    Reads QA pairs from ``results/<id>/cep/outputs/`` and (when present)
    non-answerable items from ``results/<id>/nonanswerable/outputs/``;
    emits one ``RetrievalRecord`` per (arm, question) tuple under
    ``results/<id>/retrieve/outputs/<arm>/<source>/<safe_qa_pair_id>.json``.

    The on-disk ``safe_qa_pair_id`` is the schema's ``qa_pair_id`` with
    ``":"`` replaced by ``"__"`` for cross-platform path safety (the
    composite carries ``"<file_id>:<chunk_id>:<idx>"``). The
    ``RetrievalRecord.qa_pair_id`` field inside each file preserves the
    original colons.

    Arm-specific knobs (chunker view, k-hop radius, ...) are read from
    ``ARANDU_BM25_*`` / ``ARANDU_KHOP_*`` env vars; see the per-arm
    settings classes for the fields.
    """
    selected_arms = list(arms) if arms else list(_DEFAULT_ARMS)
    bm25_settings = Bm25RetrieveSettings()
    khop_settings = KHopRetrieveSettings()

    print_info(f"Run: {pipeline_id}")
    print_info(f"Arms: {', '.join(selected_arms)}")
    if "bm25" in selected_arms:
        print_info(f"BM25 chunker: {bm25_settings.chunker_id}")
    if any(a.startswith("khop_") for a in selected_arms):
        print_info(f"K-hop: k_hop={khop_settings.k_hop}, max_postings={khop_settings.max_postings}")

    try:
        result = run_retrieve_batch(
            pipeline_id=pipeline_id,
            arms=selected_arms,
            top_k=top_k,
            bm25_settings=bm25_settings,
            khop_settings=khop_settings,
            rebuild_index=rebuild_index,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        print_error(f"Invalid retrieve configuration: {exc}")
        raise typer.Exit(code=1) from exc

    if result.retrievals_failed:
        print_warning(
            f"Failed retrievals: {result.retrievals_failed} "
            f"(check logs for per-(arm, qa_pair) errors)."
        )
    if result.retrievals_resumed:
        print_info(f"Resumed: {result.retrievals_resumed} (arm, qa_pair) tuple(s).")
    print_success(
        f"Wrote {result.retrievals_written} RetrievalRecord(s) to {result.run_dir}/outputs/"
    )
