"""Entity-swap perturbation + stratified seeding (spec §7.3, §7.6).

One LLM call rewrites a CEP question by swapping a single named entity
for a plausible same-type alternative. The LLM never sees the KG node
list; verification is entirely code-side: the replacement must be absent
from both the KG and the source corpus, and the claimed original entity
must really appear in the question.
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

from arandu.qa.non_answerable.prompts import render_perturbation_prompt
from arandu.qa.non_answerable.schemas import (
    NonAnswerableItem,
    PerturbationOutput,
    SwapRecord,
)
from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.llm_client import StructuredOutputError

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.qa.non_answerable.corpus_index import SourceCorpusIndex
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SeedPair(BaseModel):
    """A sampled CEP pair plus the provenance needed to build its twin.

    ``parent_qa_pair_id`` is constructed the same way the gold lookup and
    analysis loader build it (``<source_file_id>:<chunk_id|none>:<idx>``)
    so the non-answerable item's ``parent_qa_pair_id`` joins cleanly to
    the answerable pair in the analysis stage.
    """

    parent_qa_pair_id: str
    source_file_id: str
    chunker_id: str
    pair: QAPairCEP


def stratified_seed_sample(
    cep_dir: Path,
    *,
    seeds_per_bloom: int = 100,
    rng_seed: int = 42,
) -> list[SeedPair]:
    """Sample up to ``seeds_per_bloom`` validated CEP pairs per Bloom level.

    Only pairs whose judge validation passed are eligible (spec §7.6) -
    perturbing an already-rejected pair would compound noise. Sampling is
    seeded so the benchmark is reproducible.

    Args:
        cep_dir: ``results/<id>/cep/outputs/`` holding ``QARecordCEP`` files.
        seeds_per_bloom: Target seeds per Bloom level. Strata smaller than
            this contribute their whole eligible pool.
        rng_seed: Seed for the sampler (reproducibility).

    Returns:
        Flat list of :class:`SeedPair`, grouped-then-flattened by Bloom level.
    """
    rng = random.Random(rng_seed)
    by_bloom: dict[str, list[SeedPair]] = defaultdict(list)
    for path in sorted(cep_dir.glob("*.json")):
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable CEP file %s: %s", path, exc)
            continue
        for idx, pair in enumerate(record.qa_pairs):
            if pair.validation is None or not pair.validation.passed:
                continue
            chunk_id_segment = pair.chunk_id or "none"
            parent_id = f"{record.source_file_id}:{chunk_id_segment}:{idx}"
            by_bloom[str(pair.bloom_level)].append(
                SeedPair(
                    parent_qa_pair_id=parent_id,
                    source_file_id=record.source_file_id,
                    chunker_id=record.chunker_id,
                    pair=pair,
                )
            )

    seeds: list[SeedPair] = []
    for pool in by_bloom.values():
        n = min(seeds_per_bloom, len(pool))
        seeds.extend(rng.sample(pool, n))
    return seeds


def perturb_to_non_answerable(
    seed: SeedPair,
    *,
    kg_node_set: set[str],
    corpus_index: SourceCorpusIndex,
    llm: LLMClient,
    language: str = "pt",
    base_temperature: float = 0.7,
    max_retries: int = 3,
) -> NonAnswerableItem | None:
    """Perturb one seed into a non-answerable item, or ``None`` on collision.

    Retries up to ``max_retries`` times, nudging temperature up each attempt
    so a fresh sample is drawn when the replacement collides with the corpus
    or KG. Returns ``None`` when every attempt collides (or the question has
    no perturbable entity) - the caller records the miss in ``success_rate``.

    Args:
        seed: The CEP pair to perturb.
        kg_node_set: Normalized KG label set; replacement must be absent.
        corpus_index: Source-corpus membership gate; replacement must be absent.
        llm: Unified LLM client.
        language: Prompt language (``"pt"`` default).
        base_temperature: Temperature for attempt 0; each retry adds 0.1.
        max_retries: Total attempts before giving up.

    Returns:
        A populated :class:`NonAnswerableItem`, or ``None`` if no valid swap
        was produced.
    """
    prompt = render_perturbation_prompt(language, question=seed.pair.question)
    for attempt in range(max_retries):
        try:
            output = llm.generate_structured(
                prompt,
                PerturbationOutput,
                temperature=base_temperature + 0.1 * attempt,
            )
        except StructuredOutputError as exc:
            logger.warning(
                "Perturbation parse failed for %s (attempt %d/%d): %s",
                seed.parent_qa_pair_id,
                attempt + 1,
                max_retries,
                exc,
            )
            continue
        if _is_valid_swap(output, seed.pair.question, kg_node_set, corpus_index):
            return _build_item(seed, output)
    return None


# Entity types too weak to yield genuinely non-answerable items: years are
# densely present in the corpus and a year swap invites generalization rather
# than abstention (every year-swap leaked in the dry-run). Rejected regardless
# of the (LLM-assigned, unreliable) entity_type label.
_WEAK_ENTITY_TYPES = {"year", "ano"}


def _is_valid_swap(
    output: PerturbationOutput,
    question: str,
    kg_node_set: set[str],
    corpus_index: SourceCorpusIndex,
) -> bool:
    """Verify the swap: not a weak type, replacement absent, original present."""
    replacement = output.replacement_entity.strip()
    if output.entity_type.strip().lower() in _WEAK_ENTITY_TYPES:
        return False
    if re.fullmatch(r"\d{4}", replacement):  # bare year, even if mislabeled
        return False
    replacement_norm = replacement.lower()
    return (
        replacement_norm not in kg_node_set
        and output.replacement_entity not in corpus_index
        and output.original_entity.strip().lower() in question.lower()
    )


def _build_item(seed: SeedPair, output: PerturbationOutput) -> NonAnswerableItem:
    """Assemble the persisted :class:`NonAnswerableItem` from a valid swap."""
    return NonAnswerableItem(
        qa_pair_id=f"{seed.parent_qa_pair_id}:nonans",
        question=output.new_question,
        bloom_level=seed.pair.bloom_level,
        question_type=seed.pair.question_type,
        source_file_id=seed.source_file_id,
        chunker_id=seed.chunker_id,
        parent_qa_pair_id=seed.parent_qa_pair_id,
        swapped_entity=SwapRecord(
            original_entity=output.original_entity,
            replacement_entity=output.replacement_entity,
            entity_type=output.entity_type,
        ),
    )
