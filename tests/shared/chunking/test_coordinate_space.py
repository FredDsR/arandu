"""The chunk stage and CEP generation must agree on one coordinate space.

``chunk_id`` is sha1 over ``source_file_id|chunker_id|start_char|end_char``, so
a one-character disagreement about which text is being chunked makes every id
diverge. This file pins the two producers together (issue #166).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from arandu.qa.cep.generator import CEP_CHUNKER_ID, CEPQAGenerator
from arandu.qa.config import CEPConfig, QAConfig
from arandu.qa.schemas import QAPairCEP
from arandu.shared.chunking.batch import run_chunk_batch
from arandu.shared.chunking.schemas import ChunkSet
from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture


# Long enough that cep_4k (RecursiveChunker, chunk_size=4000) emits several
# chunks, so the test exercises interior boundaries and not just chunk 0.
_BODY = (
    "O pescador contou que quando o rio sobe ele guarda o barco no barranco alto. "
    "Depois falou da prefeitura, do ciclone e da ajuda que veio da universidade. "
) * 120


def _write_transcription(directory: Path, file_id: str, text: str) -> None:
    """Write a minimal-but-valid EnrichedRecord transcription JSON."""
    payload: dict[str, Any] = {
        "file_id": file_id,
        "name": f"{file_id}.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 10.0,
        "transcription_status": "completed",
        "validation": {"stage_results": {}, "passed": True, "rejected_at": None},
    }
    (directory / f"{file_id}_transcription.json").write_text(json.dumps(payload))


@pytest.fixture
def generator(mocker: MockerFixture) -> CEPQAGenerator:
    """A CEP generator whose Bloom module returns one pair per chunk, no LLM.

    Reasoning traces are off so the enricher never runs. One pair per chunk is
    enough: the assertion is about the ``chunk_id`` stamped on each pair, not
    about the ladder.
    """
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "qwen3:14b"
    generator = CEPQAGenerator(
        llm_client=client,
        qa_config=QAConfig(),
        cep_config=CEPConfig(
            language="pt",
            enable_reasoning_traces=False,
            bloom_distribution={"remember": 1},
        ),
    )
    mocker.patch.object(
        generator._bloom_generator,
        "generate",
        side_effect=lambda context, source_metadata=None: [
            QAPairCEP(
                question="O que aconteceu?",
                answer="Uma resposta.",
                context=context,
                question_type="factual",
                bloom_level="remember",
            )
        ],
    )
    return generator


def test_chunk_stage_and_cep_generation_stamp_the_same_chunk_ids(
    tmp_path: Path, monkeypatch: MonkeyPatch, generator: CEPQAGenerator
) -> None:
    """A leading space must not split the two producers into separate namespaces.

    Whisper prefixes transcriptions with a space. Before the EnrichedRecord
    validator, the chunk stage chunked that raw text while CEP generation
    chunked it stripped, so 0 of 2670 pairs in thesis-run-01 resolved against
    their persisted ChunkSet.

    The generation side goes through ``generate_qa_pairs``, not through
    ``_chunk_with_offsets``. That matters: the stripping this test exists to
    catch lived inside ``generate_qa_pairs``, so a test that chunked directly
    would pass while the bug was still there.
    """
    monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # The leading space is the whole point: it is what Whisper emits.
    _write_transcription(input_dir, "file-1", f" {_BODY}")

    result = run_chunk_batch(
        input_dir=input_dir, views=[CEP_CHUNKER_ID], pipeline_id="invariant-run"
    )

    chunk_set = ChunkSet.load(Path(result.run_dir) / "outputs" / CEP_CHUNKER_ID / "file-1.json")
    stage_ids = [c.chunk_id for c in chunk_set.view(CEP_CHUNKER_ID)]

    record = EnrichedRecord.model_validate_json(
        (input_dir / "file-1_transcription.json").read_text()
    )
    qa_record = generator.generate_qa_pairs(record)
    generation_ids = [pair.chunk_id for pair in qa_record.qa_pairs]

    assert len(stage_ids) > 1, "fixture must produce several chunks"
    assert stage_ids == generation_ids
