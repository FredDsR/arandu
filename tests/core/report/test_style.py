"""Tests for style helper functions."""

from __future__ import annotations

from arandu.core.report.style import (
    CATEGORICAL_COLORS,
    CRITERION_COLORS,
    WONG_PALETTE,
    get_bloom_color,
    get_color_palette,
    get_criterion_color,
)


class TestGetColorPalette:
    """Tests for get_color_palette function."""

    def test_returns_correct_count(self) -> None:
        """Test that the correct number of colors is returned."""
        assert len(get_color_palette(3)) == 3
        assert len(get_color_palette(5)) == 5
        assert len(get_color_palette(8)) == 8

    def test_returns_subset_of_categorical(self) -> None:
        """Test that colors come from CATEGORICAL_COLORS."""
        palette = get_color_palette(4)
        assert palette == CATEGORICAL_COLORS[:4]

    def test_handles_more_than_eight(self) -> None:
        """Test that requesting more than 8 colors repeats the palette."""
        palette = get_color_palette(12)
        assert len(palette) == 12
        # Colors wrap around
        assert palette[8] == CATEGORICAL_COLORS[0]

    def test_single_color(self) -> None:
        """Test requesting a single color."""
        palette = get_color_palette(1)
        assert len(palette) == 1
        assert palette[0] == CATEGORICAL_COLORS[0]


class TestGetBloomColor:
    """Tests for get_bloom_color function."""

    def test_known_levels(self) -> None:
        """Test that known levels return expected colors."""
        assert get_bloom_color("remember") == WONG_PALETTE["blue"]
        assert get_bloom_color("understand") == WONG_PALETTE["green"]
        assert get_bloom_color("analyze") == WONG_PALETTE["orange"]
        assert get_bloom_color("evaluate") == WONG_PALETTE["red"]

    def test_unknown_level_returns_grey(self) -> None:
        """Test that unknown levels return grey."""
        assert get_bloom_color("create") == WONG_PALETTE["grey"]
        assert get_bloom_color("nonexistent") == WONG_PALETTE["grey"]

    def test_case_insensitive(self) -> None:
        """Test that lookup is case-insensitive."""
        assert get_bloom_color("Analyze") == WONG_PALETTE["orange"]
        assert get_bloom_color("REMEMBER") == WONG_PALETTE["blue"]


class TestGetCriterionColor:
    """Tests for get_criterion_color function."""

    def test_known_criteria(self) -> None:
        """Test that known criteria return expected colors."""
        assert get_criterion_color("faithfulness") == WONG_PALETTE["blue"]
        assert get_criterion_color("bloom_calibration") == WONG_PALETTE["orange"]
        assert get_criterion_color("informativeness") == WONG_PALETTE["green"]
        assert get_criterion_color("self_containedness") == WONG_PALETTE["red"]

    def test_unknown_criterion_returns_grey(self) -> None:
        """Test that unknown criteria return grey."""
        assert get_criterion_color("nonexistent") == WONG_PALETTE["grey"]

    def test_case_insensitive(self) -> None:
        """Test that lookup is case-insensitive after .lower() fix."""
        assert get_criterion_color("Faithfulness") == WONG_PALETTE["blue"]
        assert get_criterion_color("BLOOM_CALIBRATION") == WONG_PALETTE["orange"]


class TestConstants:
    """Tests for module-level constants."""

    def test_criterion_colors_count(self) -> None:
        """Test that CRITERION_COLORS has exactly 4 entries."""
        assert len(CRITERION_COLORS) == 4

    def test_categorical_colors_count(self) -> None:
        """Test that CATEGORICAL_COLORS has exactly 8 entries."""
        assert len(CATEGORICAL_COLORS) == 8

    def test_all_colors_are_hex(self) -> None:
        """Test that all colors are valid hex codes."""
        for color in CATEGORICAL_COLORS:
            assert color.startswith("#")
            assert len(color) == 7
