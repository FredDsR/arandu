"""Shared style configuration for consistent visualization across all figures.

Provides colorblind-friendly palettes, publication-quality defaults,
and theme settings for Plotly charts.
"""

from __future__ import annotations

# Colorblind-friendly palette (Wong 2011, Nature Methods)
# https://www.nature.com/articles/nmeth.1618
WONG_PALETTE = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC3311",
    "purple": "#6B1E7C",
    "brown": "#6F4E37",
    "pink": "#EE6677",
    "grey": "#949494",
}

# Ordered list for categorical plots
CATEGORICAL_COLORS = [
    WONG_PALETTE["blue"],
    WONG_PALETTE["orange"],
    WONG_PALETTE["green"],
    WONG_PALETTE["red"],
    WONG_PALETTE["purple"],
    WONG_PALETTE["pink"],
    WONG_PALETTE["brown"],
    WONG_PALETTE["grey"],
]

# Bloom taxonomy level colors (semantic mapping)
BLOOM_COLORS = {
    "remember": WONG_PALETTE["blue"],
    "understand": WONG_PALETTE["green"],
    "analyze": WONG_PALETTE["orange"],
    "evaluate": WONG_PALETTE["red"],
}

# Semantic color mapping for validation criteria (consistent across all charts)
CRITERION_COLORS = {
    "faithfulness": WONG_PALETTE["blue"],
    "bloom_calibration": WONG_PALETTE["orange"],
    "informativeness": WONG_PALETTE["green"],
    "self_containedness": WONG_PALETTE["red"],
}

# Plotly template settings
PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Arial, sans-serif", "size": 12},
        "title": {"font": {"size": 16}},
        "xaxis": {"title": {"font": {"size": 13}}, "tickfont": {"size": 11}},
        "yaxis": {"title": {"font": {"size": 13}}, "tickfont": {"size": 11}},
        "colorway": CATEGORICAL_COLORS,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }
}


def get_plotly_template() -> dict:
    """Get Plotly template configuration for interactive charts.

    Returns:
        Dictionary of Plotly layout settings.
    """
    return PLOTLY_TEMPLATE.copy()


def get_color_palette(n: int) -> list[str]:
    """Get a colorblind-friendly color palette with n colors.

    Args:
        n: Number of colors needed.

    Returns:
        List of hex color codes.
    """
    if n <= len(CATEGORICAL_COLORS):
        return CATEGORICAL_COLORS[:n]
    # Repeat colors if more needed
    return (CATEGORICAL_COLORS * ((n // len(CATEGORICAL_COLORS)) + 1))[:n]


def get_bloom_color(level: str) -> str:
    """Get the semantic color for a Bloom taxonomy level.

    Args:
        level: Bloom level (remember, understand, analyze, evaluate).

    Returns:
        Hex color code.
    """
    return BLOOM_COLORS.get(level.lower(), WONG_PALETTE["grey"])


def get_criterion_color(criterion: str) -> str:
    """Get the semantic color for a validation criterion.

    Args:
        criterion: Criterion name (faithfulness, bloom_calibration, etc.).

    Returns:
        Hex color code.
    """
    return CRITERION_COLORS.get(criterion, WONG_PALETTE["grey"])
