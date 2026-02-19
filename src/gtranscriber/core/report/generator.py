"""HTML report generation with embedded Plotly charts and interactive dashboard.

Assembles self-contained HTML reports using Jinja2 FileSystemLoader with
client-side filtering and chart rendering.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

from .dataset import build_dataset

if TYPE_CHECKING:
    from .collector import RunReport

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_html_report(
    reports: list[RunReport],
    output_path: Path,
    self_contained: bool = True,
) -> None:
    """Generate a self-contained HTML report with interactive dashboard.

    Builds a flat ReportDataset from reports, serializes it to JSON, and
    renders the Jinja2 template with embedded Plotly.js for offline use.

    Args:
        reports: List of RunReport objects to include in the report.
        output_path: Path to save the HTML file.
        self_contained: If True, embed Plotly.js inline for offline use.
    """
    from datetime import datetime

    dataset = build_dataset(reports)
    dataset_json = dataset.model_dump_json()

    plotly_js = None
    if self_contained:
        try:
            import plotly.offline

            plotly_js = plotly.offline.get_plotlyjs()
        except Exception:
            pass

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template("report.html.j2")

    html_content = template.render(
        dataset_json=dataset_json,
        plotly_js=plotly_js,
        timestamp=datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
