"""Tests for KGConstructor protocol."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime in method signatures
from typing import TYPE_CHECKING

from arandu.core.kg.protocol import KGConstructor
from arandu.core.kg.schemas import KGConstructionResult
from arandu.kg.schemas import KGMetadata

if TYPE_CHECKING:
    from arandu.shared.schemas import EnrichedRecord


class _DummyConstructor:
    """Minimal implementation that satisfies KGConstructor structurally."""

    def build_graph(
        self,
        records: list[EnrichedRecord],
        output_dir: Path,
    ) -> KGConstructionResult:
        return KGConstructionResult(
            graph_file=output_dir / "test.graphml",
            metadata=KGMetadata(
                graph_id="test",
                source_documents=[],
                model_id="test-model",
                provider="ollama",
            ),
            node_count=0,
            edge_count=0,
            source_record_ids=[],
        )


class TestKGConstructorProtocol:
    """Tests for KGConstructor protocol compliance."""

    def test_isinstance_check(self) -> None:
        """Test that a class implementing build_graph satisfies the protocol."""
        constructor = _DummyConstructor()
        assert isinstance(constructor, KGConstructor)

    def test_non_conforming_class_fails(self) -> None:
        """Test that a class without build_graph does NOT satisfy the protocol."""

        class NotAConstructor:
            pass

        assert not isinstance(NotAConstructor(), KGConstructor)

    def test_wrong_method_name_fails(self) -> None:
        """Test that a class with a differently-named method fails."""

        class WrongMethod:
            def make_graph(self, records: list, output_dir: Path) -> KGConstructionResult: ...

        assert not isinstance(WrongMethod(), KGConstructor)
