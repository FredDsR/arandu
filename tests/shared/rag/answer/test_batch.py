"""Integration test for the answerer batch runner.

Stubs ``LLMClient`` so the tests never actually hit a model — verifies
the read/write contract end-to-end against a fixture run that mimics
what ``arandu retrieve`` produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from arandu.shared.rag.answer.batch import run_answer_batch
from arandu.shared.rag.answer.schemas import AnswererOutput
from arandu.shared.rag.answer.settings import AnswererSettings
from arandu.shared.rag.schemas import AnswerRecord, RetrievalRecord, RetrievedPassage

if TYPE_CHECKING:
    from pathlib import Path


def _seed_retrieve_outputs(base: Path, pipeline_id: str = "run_x") -> None:
    """Lay out a minimal retrieve stage with one record under the null arm."""
    out_dir = base / pipeline_id / "retrieve" / "outputs" / "null" / "cep"
    out_dir.mkdir(parents=True)
    record = RetrievalRecord(
        qa_pair_id="src_a:chk_00:0",
        question="Onde Maria mora?",
        retriever_id="null",
        chunker_id="cep_4k",
        top_k=5,
        passages=[],
        elapsed_ms=1.5,
        is_answerable=True,
    )
    record.save(out_dir / "src_a__chk_00__0.json")


class TestRunAnswerBatch:
    def test_end_to_end_with_mocked_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The answerer's only network seam is `LLMClient.generate_structured`.
        # Stub it via the unified LLMClient patch — confirms the batch runner
        # reads retrieval records, runs the answerer, writes AnswerRecords
        # with the right layout, and updates the checkpoint.
        _seed_retrieve_outputs(tmp_path)
        # Need an api_key for the runtime check, but we patch the LLMClient
        # construction so the value doesn't matter.
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        settings = AnswererSettings(provider="ollama")  # no api key needed

        with patch("arandu.shared.rag.answer.batch.LLMClient") as mock_llm_cls:
            inner = MagicMock()
            inner.generate_structured.return_value = AnswererOutput(
                abstained=True,
                answer=None,
                rationale="No passages retrieved (null arm).",
            )
            mock_llm_cls.return_value = inner

            result = run_answer_batch(
                pipeline_id="run_x",
                settings=settings,
                base_dir=tmp_path,
            )

        assert result.answers_written == 1
        assert result.answers_failed == 0

        # Layout mirror: <outputs>/null/cep/<file>.json
        answer_path = (
            tmp_path / "run_x" / "answers" / "outputs" / "null" / "cep" / "src_a__chk_00__0.json"
        )
        assert answer_path.exists()

        # Round-trip via AnswerRecord schema.
        rec = AnswerRecord.load(answer_path)
        assert rec.qa_pair_id == "src_a:chk_00:0"
        assert rec.retriever_id == "null"
        assert rec.abstained is True
        assert rec.answer_text is None
        assert rec.rationale.startswith("No passages")
        # answerer_meta carries the audit trail + settings snapshot.
        assert rec.answerer_meta["attempts"] == 1
        assert rec.answerer_meta["language"] == "pt"
        assert rec.answerer_meta["packed_passages"] == 0
        assert rec.answerer_meta["passages_after_top_k"] == 0
        # Settings snapshot persisted per-record so attribution survives
        # even if run-level metadata is later moved or split.
        assert rec.answerer_meta["provider"] == "ollama"
        assert rec.answerer_meta["top_k"] == 10
        assert rec.answerer_meta["max_tokens"] == 1024
        assert rec.answerer_meta["max_context_tokens"] == 8192
        assert rec.answerer_meta["prompt_overhead_tokens"] == 350
        assert rec.answerer_model == "qwen3:14b"

    def test_missing_retrieve_dir_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Retrieve outputs not found"):
            run_answer_batch(
                pipeline_id="never_retrieved",
                settings=AnswererSettings(provider="ollama"),
                base_dir=tmp_path,
            )

    def test_resume_skips_completed_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_retrieve_outputs(tmp_path)
        settings = AnswererSettings(provider="ollama")

        with patch("arandu.shared.rag.answer.batch.LLMClient") as mock_llm_cls:
            inner = MagicMock()
            inner.generate_structured.return_value = AnswererOutput(
                abstained=True, answer=None, rationale="r"
            )
            mock_llm_cls.return_value = inner

            first = run_answer_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)
            assert first.answers_written == 1

            second = run_answer_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)
            assert second.answers_written == 0
            assert second.answers_resumed == 1


class TestBuildLlmClientFailures:
    def test_unknown_provider_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_retrieve_outputs(tmp_path)
        settings = AnswererSettings.model_construct(
            provider="not_a_real_provider",
            model_id="x",
            api_key_env="OPENAI_API_KEY",
            base_url=None,
            temperature=0.2,
            max_tokens=1024,
            language="pt",
            top_k=10,
            max_context_tokens=8192,
            prompt_overhead_tokens=350,
        )
        with pytest.raises(ValueError, match="Unknown answerer provider"):
            run_answer_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)

    def test_cloud_provider_without_api_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_retrieve_outputs(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = AnswererSettings(provider="openai", api_key_env="OPENAI_API_KEY")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            run_answer_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)


class TestTopKCap:
    def test_settings_top_k_caps_passages_before_packing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Retrieval returned 5 passages; answerer settings top_k=2 must
        # cap to the first 2 BEFORE the token-budget pack runs. Without
        # this cap, the answerer would consider every passage the
        # retriever returned regardless of the per-run constant from
        # spec §5.7's ARANDU_ANSWERER_TOP_K.
        out_dir = tmp_path / "run_x" / "retrieve" / "outputs" / "khop_triple" / "cep"
        out_dir.mkdir(parents=True)
        passages = [
            RetrievedPassage(
                chunk_id=f"triple:{i}",
                rank=i,
                score=1.0 - i * 0.1,
                payload=f"triple_payload_{i}",
            )
            for i in range(5)
        ]
        record = RetrievalRecord(
            qa_pair_id="src_a:chk_00:0",
            question="q",
            retriever_id="khop_triple",
            chunker_id="cep_4k",
            top_k=5,
            passages=passages,
            elapsed_ms=1.0,
            is_answerable=True,
        )
        record.save(out_dir / "src_a__chk_00__0.json")
        settings = AnswererSettings(provider="ollama", top_k=2)

        with patch("arandu.shared.rag.answer.batch.LLMClient") as mock_llm_cls:
            inner = MagicMock()
            inner.generate_structured.return_value = AnswererOutput(
                abstained=False, answer="a", rationale="r"
            )
            mock_llm_cls.return_value = inner
            run_answer_batch(pipeline_id="run_x", settings=settings, base_dir=tmp_path)

        # Inspect what the answerer saw — only the first 2 payloads.
        prompt = inner.generate_structured.call_args.kwargs["prompt"]
        assert "triple_payload_0" in prompt
        assert "triple_payload_1" in prompt
        assert "triple_payload_2" not in prompt
        assert "triple_payload_3" not in prompt

        # And the persisted record reflects the cap.
        rec = AnswerRecord.load(
            tmp_path
            / "run_x"
            / "answers"
            / "outputs"
            / "khop_triple"
            / "cep"
            / "src_a__chk_00__0.json"
        )
        assert rec.answerer_meta["passages_after_top_k"] == 2
        assert rec.answerer_meta["top_k"] == 2


class TestPayloadPassThrough:
    def test_triple_payload_routed_to_answerer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Triple-arm records carry passage content in `payload`. The
        # batch runner's pack_passages call must surface it to the
        # answerer prompt without consulting the chunk_id resolver.
        out_dir = tmp_path / "run_x" / "retrieve" / "outputs" / "khop_triple" / "cep"
        out_dir.mkdir(parents=True)
        passage = RetrievedPassage(
            chunk_id="triple:abc123",
            rank=0,
            score=1.0,
            payload="Maria --[vive_em]--> Itaqui",
        )
        record = RetrievalRecord(
            qa_pair_id="src_a:chk_00:0",
            question="Onde Maria mora?",
            retriever_id="khop_triple",
            chunker_id="cep_4k",
            top_k=5,
            passages=[passage],
            elapsed_ms=2.5,
            is_answerable=True,
        )
        record.save(out_dir / "src_a__chk_00__0.json")
        settings = AnswererSettings(provider="ollama")

        with patch("arandu.shared.rag.answer.batch.LLMClient") as mock_llm_cls:
            inner = MagicMock()
            inner.generate_structured.return_value = AnswererOutput(
                abstained=False,
                answer="Maria mora em Itaqui.",
                rationale="Triple says vive_em → Itaqui.",
            )
            mock_llm_cls.return_value = inner
            run_answer_batch(
                pipeline_id="run_x",
                settings=settings,
                base_dir=tmp_path,
            )
            # Verify the answerer saw the triple's payload in its prompt.
            prompt = inner.generate_structured.call_args.kwargs["prompt"]
            assert "Maria --[vive_em]--> Itaqui" in prompt
