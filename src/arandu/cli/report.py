"""Report generation and dashboard CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from arandu.shared.config import ResultsConfig
from arandu.utils.logger import print_error, print_info, print_success, print_warning, setup_logging

logger = logging.getLogger(__name__)

_results_config = ResultsConfig()


def report(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", "--id", help="Generate report for a specific pipeline run."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for HTML report."),
    ] = Path("report.html"),
    no_png: Annotated[
        bool,
        typer.Option("--no-png", help="Skip automatic PNG export of charts."),
    ] = False,
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            help="Path to results directory (defaults to configured results_dir).",
        ),
    ] = _results_config.base_dir,
) -> None:
    """Generate interactive HTML dashboard and PNG charts for pipeline results.

    Creates a self-contained HTML report with client-side filtering,
    tabbed navigation, and interactive Plotly charts. By default, also
    exports publication-quality PNG files via kaleido.

    The report discovers results by scanning the filesystem directly,
    without relying on index.json which can become stale.

    .. deprecated::
        Use ``serve-report`` for the interactive dashboard.
        This command will be removed in a future version.

    Examples:
        # Generate HTML report + PNGs for all runs
        arandu report

        # Generate report for specific run
        arandu report --run-id 20250101_120000

        # Generate HTML only (skip PNG export)
        arandu report --no-png

        # Custom output path
        arandu report --output ./reports/dashboard.html
    """
    from arandu.report import ResultsCollector
    from arandu.report.dataset import build_dataset
    from arandu.report.exporter import export_charts_as_png
    from arandu.report.generator import generate_html_report

    print_warning(
        "The 'report' command is deprecated and will be removed in a future version. "
        "Use 'serve-report' for the interactive dashboard with drill-down, "
        "filtering, and export capabilities."
    )
    setup_logging()

    # Initialize collector
    collector = ResultsCollector(results_dir)

    # Load reports
    if run_id:
        print_info(f"Loading run: {run_id}")
        try:
            reports = [collector.load_run(run_id)]
        except FileNotFoundError:
            print_error(f"Run not found: {run_id}")
            raise typer.Exit(code=1) from None
    else:
        print_info("Discovering all pipeline runs...")
        reports = collector.load_all_runs()

    if not reports:
        print_error("No pipeline runs found.")
        raise typer.Exit(code=1)

    print_info(f"Loaded {len(reports)} pipeline run(s)")

    # Generate HTML report
    print_info(f"Generating HTML report: {output}")
    try:
        generate_html_report(reports, output)
        print_success(f"HTML report saved to: {output}")
    except Exception as e:
        print_error(f"Failed to generate HTML report: {e}")
        logger.exception("HTML report generation failed")
        raise typer.Exit(code=1) from e

    # Export PNG charts (unless --no-png)
    if not no_png:
        figures_dir = output.parent / "figures"
        print_info(f"Exporting PNG charts to: {figures_dir}")
        try:
            dataset = build_dataset(reports)
            generated_files = export_charts_as_png(dataset, figures_dir)
            print_success(f"Exported {len(generated_files)} chart(s):")
            for fig_path in generated_files:
                print_info(f"  - {fig_path.name}")
        except Exception as e:
            print_error(f"Failed to export PNG charts: {e}")
            logger.exception("PNG export failed")
            raise typer.Exit(code=1) from e


def serve_report(
    results_dir: Annotated[
        Path,
        typer.Argument(help="Path to results directory containing pipeline runs."),
    ],
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port for the dashboard server.")
    ] = 8050,
    host: Annotated[str, typer.Option("--host", help="Host address to bind.")] = "127.0.0.1",
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not automatically open browser on startup."),
    ] = False,
) -> None:
    """Launch interactive dashboard for pipeline results exploration.

    Starts a local FastAPI server serving an interactive dashboard with
    charts, data tables, and drill-down views for pipeline results.

    Examples:
        arandu serve-report results/
        arandu serve-report results/ --port 9000 --no-browser
    """
    import webbrowser

    import uvicorn

    from arandu.report.api import create_app
    from arandu.report.collector import ResultsCollector

    setup_logging()

    if not results_dir.exists():
        print_error(f"Results directory not found: {results_dir}")
        raise typer.Exit(code=1)

    url = f"http://{host}:{port}"
    print_info(f"Starting report dashboard at [bold]{url}[/bold]")

    fastapi_app = create_app(results_dir)

    # Override the dependency so all handlers use the correct results_dir.
    from arandu.report.api import get_report_service
    from arandu.report.service import ReportService

    _cached_service: ReportService | None = None

    def _service_override() -> ReportService:
        nonlocal _cached_service
        if _cached_service is None:
            collector = ResultsCollector(results_dir)
            _cached_service = ReportService(collector)
        return _cached_service

    fastapi_app.dependency_overrides[get_report_service] = _service_override

    if not no_browser:
        import threading

        def _open_browser() -> None:
            import time

            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(fastapi_app, host=host, port=port)
