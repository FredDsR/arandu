"""Build the atlas-rag retriever's precomputed index for a given pipeline run.

Thin domain-layer wrapper around :meth:`AtlasRagRetriever.build_index`. The
heavy embedding work lives in atlas-rag's ``create_embeddings_and_index``;
this module just resolves paths via ``ResultsConfig`` (matching
:func:`arandu.kg.passage_offsets.link_passages`'s style) and dispatches.

Lives next to ``passage_offsets.py`` rather than under
``arandu.shared.rag.retrievers/`` because the precompute is intrinsic to
the *KG* (depends on the graphml's sha), not to the benchmark run that
consumes it. Co-locating with the KG-stage helpers matches where the
precompute lands on disk: ``results/<id>/kg/outputs/atlas_output/precompute/``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arandu.shared.config import ResultsConfig
from arandu.shared.embeddings import EmbedderSettings, build_embedder
from arandu.shared.rag.retrievers.atlas_rag import (
    MANIFEST_FILENAME,
    PRECOMPUTE_DIR_NAME,
    AtlasRagRetriever,
)

if TYPE_CHECKING:
    from pathlib import Path


def _check_manifest_compatible(
    manifest_path: Path,
    *,
    keyword: str,
    include_events: bool,
    include_concept: bool,
    sentence_encoder_model: str,
) -> None:
    """Raise ValueError if the on-disk manifest disagrees with the requested params.

    Without this check, a precompute built with one set of flags would be
    silently reused when ``arandu kg-build-retriever-index`` is invoked
    with different ones. ``AtlasRagRetriever.__init__`` would later catch
    the mismatch, but the operator's mental model "I just rebuilt this"
    would be wrong — surface it here instead, pointing them at ``--rebuild``.
    """
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Existing manifest at {manifest_path} is unreadable: {exc}. "
            f"Pass --rebuild to regenerate."
        ) from exc

    mismatches: list[str] = []
    for field, expected in (
        ("keyword", keyword),
        ("include_events", include_events),
        ("include_concept", include_concept),
        ("sentence_encoder_model", sentence_encoder_model),
    ):
        if manifest.get(field) != expected:
            mismatches.append(
                f"  {field}: manifest={manifest.get(field)!r}, requested={expected!r}"
            )
    if mismatches:
        joined = "\n".join(mismatches)
        raise ValueError(
            f"Existing precompute at {manifest_path.parent}/ was built with different "
            f"parameters than requested:\n{joined}\n"
            f"Pass --rebuild to regenerate, or invoke with matching parameters."
        )


def build_retriever_index(
    pipeline_id: str,
    *,
    base_dir: Path | None = None,
    embedder_settings: EmbedderSettings | None = None,
    keyword: str = "transcriptions.json",
    include_events: bool = True,
    include_concept: bool = True,
    rebuild: bool = False,
) -> Path:
    """Build (or refresh) the atlas-rag precompute for run ``pipeline_id``.

    Resolves the KG output directory under ``<base_dir>/<pipeline_id>/kg/
    outputs/atlas_output/``, then invokes
    :meth:`AtlasRagRetriever.build_index` with an embedder built from
    ``embedder_settings``.

    Args:
        pipeline_id: The run identifier. Inputs are resolved relative to
            ``<base_dir>/<pipeline_id>/kg/outputs/atlas_output/``.
        base_dir: Project ``results/`` root. Defaults to
            ``ResultsConfig().base_dir``.
        embedder_settings: Provider + model choice for the encoder.
            Defaults to ``EmbedderSettings()`` (reads ``ARANDU_EMBEDDER_*``
            env vars).
        keyword: atlas-rag's filename pattern. Defaults to project
            convention ``"transcriptions.json"``.
        include_events: Whether to include event nodes in the embedded
            node set. Defaults to True (matches project KG construction).
        include_concept: Whether to include concept nodes. Defaults to
            True. atlas-rag rejects the ``(False, True)`` combination;
            :meth:`AtlasRagRetriever.build_index` defends that with a
            clearer error before invoking upstream.
        rebuild: If False (default), skip the build when a manifest already
            exists at the target precompute path AND its recorded
            parameters match the requested ones. If the existing manifest
            disagrees with the request, raise :class:`ValueError` rather
            than silently reusing an incompatible precompute. If True,
            force a fresh build regardless (atlas-rag overwrites the
            existing pickles).

    Returns:
        The precompute directory path (``.../atlas_output/precompute/``).

    Raises:
        FileNotFoundError: If the KG outputs or the atlas-rag GraphML
            for ``pipeline_id`` are missing.
        RuntimeError: From :func:`build_embedder` if a cloud embedder is
            requested without its API key env var set.
        ValueError: If an existing manifest at the target path was built
            with different ``keyword`` / ``include_events`` /
            ``include_concept`` / ``sentence_encoder_model`` than
            requested. Pass ``rebuild=True`` to regenerate.
    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir
    kg_outputs_dir = base / pipeline_id / "kg" / "outputs"
    atlas_output_dir = kg_outputs_dir / "atlas_output"
    graphml_path = atlas_output_dir / "kg_graphml" / f"{keyword}_graph.graphml"

    if not kg_outputs_dir.exists():
        raise FileNotFoundError(
            f"kg outputs not found for pipeline_id {pipeline_id!r}: {kg_outputs_dir}. "
            f"Run `arandu build-kg --id {pipeline_id} ...` first."
        )
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"atlas-rag GraphML not found for pipeline_id {pipeline_id!r}: {graphml_path}. "
            f"The KG stage either used a non-atlas backend or was interrupted "
            f"before graphml emission."
        )

    settings = embedder_settings if embedder_settings is not None else EmbedderSettings()

    precompute_dir = atlas_output_dir / PRECOMPUTE_DIR_NAME
    manifest_path = precompute_dir / MANIFEST_FILENAME
    if manifest_path.exists() and not rebuild:
        # Validate the on-disk manifest matches what was requested. Silent
        # reuse on mismatch would defer the error to retrieve-time when
        # AtlasRagRetriever.__init__ runs its own validation — surface it
        # now so the operator's "I just rebuilt this" mental model is right.
        _check_manifest_compatible(
            manifest_path,
            keyword=keyword,
            include_events=include_events,
            include_concept=include_concept,
            sentence_encoder_model=settings.model,
        )
        return precompute_dir

    encoder = build_embedder(settings)

    AtlasRagRetriever.build_index(
        kg_outputs_dir=atlas_output_dir,
        sentence_encoder=encoder,
        sentence_encoder_model=settings.model,
        keyword=keyword,
        include_events=include_events,
        include_concept=include_concept,
    )
    return precompute_dir
