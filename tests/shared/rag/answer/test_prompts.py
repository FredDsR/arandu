"""Tests for the answerer Jinja prompt loader."""

from __future__ import annotations

import pytest

from arandu.shared.rag.answer.prompts import render_prompt


class TestRenderPrompt:
    def test_pt_renders_question(self) -> None:
        out = render_prompt("pt", question="Onde Maria mora?", passages=[])
        assert "Onde Maria mora?" in out
        assert "REGRAS:" in out

    def test_en_renders_question(self) -> None:
        out = render_prompt("en", question="Where does Maria live?", passages=[])
        assert "Where does Maria live?" in out
        assert "RULES:" in out

    def test_no_passages_branch(self) -> None:
        # The template's else-branch tells the model no passages were
        # retrieved (so it knows to abstain). Lock the wording.
        pt = render_prompt("pt", question="q", passages=[])
        assert "nenhuma passagem foi recuperada" in pt
        en = render_prompt("en", question="q", passages=[])
        assert "no passages were retrieved" in en

    def test_passages_enumerated_in_order(self) -> None:
        # Jinja loop renders ranked passages in order with 1-indexed
        # labels. Lock that ordering so future template edits don't
        # silently shuffle them.
        out = render_prompt(
            "pt",
            question="q",
            passages=["primeira passagem", "segunda passagem", "terceira passagem"],
        )
        idx_p1 = out.index("primeira passagem")
        idx_p2 = out.index("segunda passagem")
        idx_p3 = out.index("terceira passagem")
        assert idx_p1 < idx_p2 < idx_p3
        assert "[Passagem 1]" in out
        assert "[Passagem 2]" in out
        assert "[Passagem 3]" in out

    def test_unknown_language_raises(self) -> None:
        # The Literal type covers this at typecheck time, but at runtime
        # we want the StrictUndefined / TemplateNotFound path to fire
        # cleanly rather than silently rendering an empty body.
        with pytest.raises(Exception, match="answerer_fr"):
            render_prompt("fr", question="q", passages=[])  # type: ignore[arg-type]
