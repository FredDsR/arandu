"""Tests for the emic-judge descriptive analysis script.

Guards the load-bearing arithmetic (a pair whose LLM call errored must never
enter a mean) and the two reporting decisions that would otherwise mislead the
results chapter: an ``approved``-scope run must not report zero rejected pairs,
and an unreadable output file must be counted rather than silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arandu.shared.emic.schemas import EmicScore, EmicSourceScores
from scripts.analyze_emic_scores import (
    build_analysis,
    distribution,
    load_source_scores,
    render_markdown,
    resolve_scope,
)

if TYPE_CHECKING:
    from pathlib import Path


def _score(
    index: int,
    value: int | None,
    bloom: str = "remember",
    is_valid: bool | None = True,
) -> EmicScore:
    return EmicScore(
        pair_index=index,
        bloom_level=bloom,
        emic_score=value,
        rationale="",
        error=None if value is not None else "sidecar down",
        is_valid=is_valid,
    )


@pytest.fixture
def sources() -> list[EmicSourceScores]:
    """Two sources: five scored pairs (1..5) plus one errored pair."""
    return [
        EmicSourceScores(
            source_file_id="a",
            source_filename="entrevista-a.m4a",
            scope="all",
            scores=[
                _score(0, 5, "remember", True),
                _score(1, 4, "understand", True),
                _score(2, 2, "apply", False),
                _score(3, None, "analyze", True),
            ],
        ),
        EmicSourceScores(
            source_file_id="b",
            source_filename="entrevista-b.m4a",
            scope="all",
            scores=[
                _score(0, 3, "remember", None),
                _score(1, 1, "remember", True),
            ],
        ),
    ]


class TestDistribution:
    def test_reports_counts_mean_median_and_iqr(self) -> None:
        dist = distribution([1, 2, 3, 4, 5])

        assert dist.n == 5
        assert dist.counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        assert dist.mean == pytest.approx(3.0)
        assert dist.median == pytest.approx(3.0)
        assert (dist.p25, dist.p75) == (pytest.approx(2.0), pytest.approx(4.0))
        assert dist.pct_ge_4 == pytest.approx(0.4)

    def test_single_value_has_a_degenerate_iqr(self) -> None:
        dist = distribution([4])

        assert (dist.n, dist.mean, dist.median) == (1, 4.0, 4.0)
        assert (dist.p25, dist.p75) == (4.0, 4.0)

    def test_empty_reports_no_central_tendency(self) -> None:
        dist = distribution([])

        assert dist.n == 0
        assert dist.counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        assert (dist.mean, dist.median, dist.p25, dist.p75, dist.pct_ge_4) == (
            None,
            None,
            None,
            None,
            None,
        )


class TestBuildAnalysis:
    def test_errored_pairs_are_counted_but_never_averaged(
        self, sources: list[EmicSourceScores]
    ) -> None:
        analysis = build_analysis(sources, unreadable=0, scope="all")

        assert analysis.health.pairs_total == 6
        assert analysis.health.pairs_scored == 5
        assert analysis.health.pairs_failed == 1
        assert analysis.overall.n == 5
        assert analysis.overall.mean == pytest.approx(3.0)

    def test_bloom_breakdown_keeps_a_level_whose_only_pair_errored(
        self, sources: list[EmicSourceScores]
    ) -> None:
        by_bloom = build_analysis(sources, unreadable=0, scope="all").by_bloom

        assert by_bloom["remember"].n == 3
        assert by_bloom["remember"].mean == pytest.approx(3.0)
        assert by_bloom["analyze"].n == 0
        assert by_bloom["analyze"].mean is None

    def test_verdict_crosstab_splits_approved_rejected_and_unjudged(
        self, sources: list[EmicSourceScores]
    ) -> None:
        by_verdict = build_analysis(sources, unreadable=0, scope="all").by_verdict

        assert by_verdict["approved"].counts == {1: 1, 2: 0, 3: 0, 4: 1, 5: 1}
        assert by_verdict["rejected"].counts == {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
        assert by_verdict["unjudged"].counts == {1: 0, 2: 0, 3: 1, 4: 0, 5: 0}

    def test_sources_are_ordered_worst_mean_first(self, sources: list[EmicSourceScores]) -> None:
        rows = build_analysis(sources, unreadable=0, scope="all").by_source

        assert [row.filename for row in rows] == ["entrevista-b.m4a", "entrevista-a.m4a"]
        assert rows[0].dist.mean == pytest.approx(2.0)


class TestLoadSourceScores:
    def test_missing_outputs_directory_names_the_stage(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="emic_judge"):
            load_source_scores(tmp_path / "emic_judge" / "outputs")

    def test_unreadable_file_is_skipped_and_counted(
        self, tmp_path: Path, sources: list[EmicSourceScores]
    ) -> None:
        outputs = tmp_path / "emic_judge" / "outputs"
        outputs.mkdir(parents=True)
        sources[0].save(outputs / "a.json")
        (outputs / "b.json").write_text("{ not json", encoding="utf-8")

        loaded, unreadable = load_source_scores(outputs)

        assert [s.source_file_id for s in loaded] == ["a"]
        assert unreadable == 1


class TestResolveScope:
    def test_prefers_the_run_metadata_snapshot(
        self, tmp_path: Path, sources: list[EmicSourceScores]
    ) -> None:
        (tmp_path / "run_metadata.json").write_text(
            '{"config": {"config_values": {"scope": "approved"}}}', encoding="utf-8"
        )

        assert resolve_scope(tmp_path, sources) == "approved"

    def test_falls_back_to_the_unanimous_scope_on_the_outputs(
        self, tmp_path: Path, sources: list[EmicSourceScores]
    ) -> None:
        assert resolve_scope(tmp_path, sources) == "all"

    def test_reports_unknown_when_the_outputs_disagree(
        self, tmp_path: Path, sources: list[EmicSourceScores]
    ) -> None:
        sources[1].scope = None

        assert resolve_scope(tmp_path, sources) == "unknown"


class TestRenderMarkdown:
    def test_renders_every_section(self, sources: list[EmicSourceScores]) -> None:
        rendered = render_markdown(build_analysis(sources, unreadable=0, scope="all"), "run-01")

        assert "# Emic-validity scores - run-01" in rendered
        assert "## Run health" in rendered
        assert "## Score distribution" in rendered
        assert "## By Bloom level" in rendered
        assert "## Emic score x judge-qa verdict" in rendered
        assert "## By source" in rendered
        assert "entrevista-b.m4a" in rendered

    def test_approved_scope_says_unscored_instead_of_zero(
        self, sources: list[EmicSourceScores]
    ) -> None:
        approved_only = [
            EmicSourceScores(
                source_file_id="a",
                source_filename="entrevista-a.m4a",
                scope="approved",
                scores=[_score(0, 5), _score(1, 4)],
            )
        ]

        rendered = render_markdown(
            build_analysis(approved_only, unreadable=0, scope="approved"), "run-01"
        )

        assert "never scored" in rendered
        assert "| rejected |" not in rendered
