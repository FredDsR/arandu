"""Results visualization and metrics reporting module.

This module provides tools for:
- Discovering and aggregating pipeline results from the filesystem
- Generating interactive HTML reports with Plotly charts
- Exporting publication-quality static figures (PNG, SVG, PDF)
"""

from __future__ import annotations

from .collector import ResultsCollector, RunReport

__all__ = [
    "ResultsCollector",
    "RunReport",
]
