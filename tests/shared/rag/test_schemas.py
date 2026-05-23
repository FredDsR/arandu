"""Tests for shared/rag/schemas.py — RetrievedPassage, RetrievalRecord, AnswerRecord."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used as a parameter type at runtime

import pytest
from pydantic import ValidationError

from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeResultMixin,
    JudgeStepResult,
)
from arandu.shared.rag.schemas import AnswerRecord, RetrievalRecord, RetrievedPassage


def _passage(
    chunk_id: str = "abcdef1234567890",
    rank: int = 0,
    score: float = 1.2,
    retriever_meta: dict[str, object] | None = None,
    payload: str | None = None,
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        rank=rank,
        score=score,
        retriever_meta=retriever_meta or {},
        payload=payload,
    )


def _record(**overrides: object) -> RetrievalRecord:
    defaults: dict[str, object] = {
        "qa_pair_id": "src1:cep_4k:0",
        "question": "Em que ano ocorreu a enchente?",
        "retriever_id": "bm25_bm25_512t",
        "chunker_id": "bm25_512t",
        "top_k": 5,
        "passages": [_passage()],
        "elapsed_ms": 12.5,
        "is_answerable": True,
    }
    defaults.update(overrides)
    return RetrievalRecord(**defaults)  # type: ignore[arg-type]


class TestRetrievedPassage:
    def test_minimal_passage_validates(self) -> None:
        p = _passage()
        assert p.chunk_id == "abcdef1234567890"
        assert p.rank == 0
        assert p.score == 1.2
        assert p.retriever_meta == {}

    def test_rank_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            _passage(rank=-1)

    def test_retriever_meta_carries_backend_specific_keys(self) -> None:
        p = _passage(retriever_meta={"score_method": "bm25_okapi", "ppr_weight": 0.42})
        assert p.retriever_meta["score_method"] == "bm25_okapi"
        assert p.retriever_meta["ppr_weight"] == 0.42

    def test_score_can_be_negative(self) -> None:
        # BM25 / cosine variants can produce negative or zero scores; don't gate on sign.
        p = _passage(score=-0.5)
        assert p.score == -0.5

    def test_payload_defaults_none(self) -> None:
        # Existing retrievers (BM25, atlas-rag, NetworkX, Null) don't set payload;
        # the field must default to None so they continue working unchanged.
        p = _passage()
        assert p.payload is None

    def test_payload_carries_arbitrary_string(self) -> None:
        # Used by retrievers that emit non-chunk content (e.g. linearized KG
        # triples). When set, the Answerer uses payload verbatim instead of
        # resolving chunk_id → source-text via offsets.
        triple = "[PERSON] Maria --[VIVE_EM]--> [LOCATION] Barra do Ribeiro"
        p = _passage(payload=triple)
        assert p.payload == triple

    def test_payload_roundtrips_through_json(self) -> None:
        # The container persists RetrievedPassage to JSONL via RetrievalRecord;
        # payload must survive serialize/deserialize for the Answerer to read it.
        original = _passage(payload="some triple text")
        roundtripped = type(original).model_validate_json(original.model_dump_json())
        assert roundtripped.payload == "some triple text"


class TestRetrievalRecord:
    def test_minimal_record_validates(self) -> None:
        r = _record()
        assert r.retriever_id == "bm25_bm25_512t"
        assert r.chunker_id == "bm25_512t"
        assert r.top_k == 5
        assert r.is_answerable is True
        assert r.elapsed_ms == 12.5
        assert len(r.passages) == 1

    def test_top_k_rejects_non_positive(self) -> None:
        with pytest.raises(ValidationError):
            _record(top_k=0)
        with pytest.raises(ValidationError):
            _record(top_k=-1)

    def test_elapsed_ms_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            _record(elapsed_ms=-0.001)

    def test_empty_passages_allowed(self) -> None:
        # NullRetriever returns []; rank-0 passage list is a valid retrieval outcome.
        r = _record(passages=[])
        assert r.passages == []

    def test_passages_serialize_round_trip(self) -> None:
        r = _record(
            passages=[_passage(chunk_id="a" * 16, rank=0), _passage(chunk_id="b" * 16, rank=1)]
        )
        dumped = r.model_dump_json()
        loaded = RetrievalRecord.model_validate_json(dumped)
        assert [p.chunk_id for p in loaded.passages] == ["a" * 16, "b" * 16]
        assert [p.rank for p in loaded.passages] == [0, 1]

    def test_rejects_passages_longer_than_top_k(self) -> None:
        # Documented invariant: len(passages) <= top_k.
        many = [_passage(chunk_id=f"c{i:015d}", rank=i) for i in range(5)]
        with pytest.raises(ValidationError, match=r"top_k"):
            _record(top_k=3, passages=many)

    def test_accepts_passages_equal_to_top_k(self) -> None:
        many = [_passage(chunk_id=f"c{i:015d}", rank=i) for i in range(3)]
        r = _record(top_k=3, passages=many)
        assert len(r.passages) == 3

    def test_accepts_fewer_passages_than_top_k(self) -> None:
        # Retriever may run out of candidates before reaching top_k.
        r = _record(top_k=10, passages=[_passage()])
        assert len(r.passages) == 1

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        r = _record(
            passages=[
                _passage(chunk_id="a" * 16, rank=0, score=2.5),
                _passage(chunk_id="b" * 16, rank=1, score=1.5),
            ],
        )
        r.save(path)
        loaded = RetrievalRecord.load(path)
        assert loaded.retriever_id == r.retriever_id
        assert loaded.chunker_id == r.chunker_id
        assert loaded.top_k == r.top_k
        assert [p.chunk_id for p in loaded.passages] == ["a" * 16, "b" * 16]
        assert [p.score for p in loaded.passages] == [2.5, 1.5]


class TestAnswerRecord:
    def _answer(self, **overrides: object) -> AnswerRecord:
        defaults: dict[str, object] = {
            "qa_pair_id": "src1:cep_4k:0",
            "question": "Em que ano ocorreu a enchente?",
            "retriever_id": "bm25_bm25_512t",
            "chunker_id": "bm25_512t",
            "top_k": 5,
            "passages": [_passage()],
            "elapsed_ms": 12.5,
            "is_answerable": True,
            "answer_text": "2024",
            "abstained": False,
            "rationale": "Found in passage 0.",
            "answerer_model": "qwen3:14b",
            "answerer_temperature": 0.2,
            "answerer_meta": {},
        }
        defaults.update(overrides)
        return AnswerRecord(**defaults)  # type: ignore[arg-type]

    def test_answer_record_inherits_retrieval_fields(self) -> None:
        a = self._answer()
        assert isinstance(a, RetrievalRecord)
        assert a.retriever_id == "bm25_bm25_512t"
        assert a.answer_text == "2024"
        assert a.abstained is False

    def test_answer_record_inherits_judge_mixin(self) -> None:
        a = self._answer()
        assert isinstance(a, JudgeResultMixin)
        # No judge run yet → validation is None, is_valid is None
        assert a.validation is None
        assert a.is_valid is None

    def test_abstained_requires_null_answer_text(self) -> None:
        # answer_text MUST be None when abstained=True
        with pytest.raises(ValidationError, match="answer_text"):
            self._answer(abstained=True, answer_text="2024")

    def test_non_abstained_requires_non_null_answer_text(self) -> None:
        with pytest.raises(ValidationError, match="answer_text"):
            self._answer(abstained=False, answer_text=None)

    def test_abstained_with_null_answer_text_validates(self) -> None:
        a = self._answer(abstained=True, answer_text=None, rationale="Question not answerable.")
        assert a.abstained is True
        assert a.answer_text is None

    def test_carries_judge_verdict_when_validation_set(self) -> None:
        verdict = JudgePipelineResult(
            stage_results={
                "answer_correctness": JudgeStepResult(
                    criterion_scores={
                        "answer_correctness": CriterionScore(
                            score=1.0, threshold=0.5, rationale="Answer matches gold."
                        ),
                    },
                ),
            },
            passed=True,
        )
        a = self._answer(validation=verdict)
        assert a.is_valid is True
        assert a.validation is not None
        assert a.validation.stage_results["answer_correctness"].passed is True

    def test_answerer_temperature_bounded(self) -> None:
        with pytest.raises(ValidationError):
            self._answer(answerer_temperature=-0.1)
        with pytest.raises(ValidationError):
            self._answer(answerer_temperature=2.5)
