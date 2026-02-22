"""Results visualization and interactive dashboard reporting module.

This module provides tools for:
- Discovering and aggregating pipeline results from the filesystem
- Building flat datasets for visualization and JSON serialization
- Generating interactive HTML dashboards with client-side filtering
- Exporting publication-quality PNG charts via Plotly + kaleido
- Serving a FastAPI-based interactive dashboard (serve-report command)
"""

from __future__ import annotations

from .api import create_app
from .collector import ResultsCollector, RunReport
from .dataset import ReportDataset, build_dataset
from .exporter import export_charts_as_png
from .generator import generate_html_report
from .service import ReportService

__all__ = [
    "ReportDataset",
    "ReportService",
    "ResultsCollector",
    "RunReport",
    "build_dataset",
    "create_app",
    "export_charts_as_png",
    "generate_html_report",
]
