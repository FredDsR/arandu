"""HTML report generation with embedded Plotly charts.

Assembles self-contained HTML reports with interactive visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Template

from . import charts

if TYPE_CHECKING:
    from pathlib import Path

    from .collector import RunReport


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>G-Transcriber Pipeline Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .run-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .run-table th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .run-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        .run-table tr:hover {
            background-color: #f8f9fa;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-completed {
            background-color: #2ecc71;
            color: white;
        }
        .status-failed {
            background-color: #e74c3c;
            color: white;
        }
        .status-in_progress {
            background-color: #f39c12;
            color: white;
        }
        footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>📊 G-Transcriber Pipeline Report</h1>

    <div class="summary-stats">
        <div class="stat-card">
            <h3>Total Runs</h3>
            <div class="value">{{ summary.total_runs }}</div>
        </div>
        <div class="stat-card">
            <h3>Total Transcriptions</h3>
            <div class="value">{{ summary.total_transcriptions }}</div>
        </div>
        <div class="stat-card">
            <h3>Total QA Pairs</h3>
            <div class="value">{{ summary.total_qa_pairs }}</div>
        </div>
        <div class="stat-card">
            <h3>Avg Success Rate</h3>
            <div class="value">{{ "%.1f"|format(summary.avg_success_rate) }}%</div>
        </div>
    </div>

    <h2>📋 Run Summary Table</h2>
    <table class="run-table">
        <thead>
            <tr>
                <th>Pipeline ID</th>
                <th>Steps</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Success Rate</th>
                <th>Items</th>
            </tr>
        </thead>
        <tbody>
            {% for run in runs %}
            <tr>
                <td><strong>{{ run.pipeline_id }}</strong></td>
                <td>{{ run.steps }}</td>
                <td><span class="status-badge status-{{ run.status }}">{{ run.status }}</span></td>
                <td>{{ "%.1f"|format(run.duration) if run.duration else "N/A" }}</td>
                <td>{{ "%.1f"|format(run.success_rate) if run.success_rate else "N/A" }}%</td>
                <td>{{ run.completed }}/{{ run.total }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% if charts.pipeline_overview %}
    <h2>📈 Pipeline Overview</h2>
    <div class="chart-container">
        <div id="pipeline-overview"></div>
    </div>
    {% endif %}

    {% if charts.bloom_distribution %}
    <h2>🎓 Bloom's Taxonomy Distribution</h2>
    <div class="chart-container">
        <div id="bloom-distribution"></div>
    </div>
    {% endif %}

    {% if charts.validation_scores %}
    <h2>✅ Validation Score Distributions</h2>
    <div class="chart-container">
        <div id="validation-scores"></div>
    </div>
    {% endif %}

    {% if charts.confidence_distribution %}
    <h2>🎯 Confidence Score Distribution</h2>
    <div class="chart-container">
        <div id="confidence-distribution"></div>
    </div>
    {% endif %}

    {% if charts.transcription_quality %}
    <h2>🎙️ Transcription Quality Metrics</h2>
    <div class="chart-container">
        <div id="transcription-quality"></div>
    </div>
    {% endif %}

    {% if charts.multihop_ratio %}
    <h2>🔗 Multi-hop vs Single-hop Questions</h2>
    <div class="chart-container">
        <div id="multihop-ratio"></div>
    </div>
    {% endif %}

    <footer>
        <p>Generated by G-Transcriber Report Generator</p>
        <p>Report created: {{ timestamp }}</p>
    </footer>

    <script>
        {% if charts.pipeline_overview %}
        var pipelineOverviewData = {{ charts.pipeline_overview | safe }};
        Plotly.newPlot('pipeline-overview', pipelineOverviewData.data, pipelineOverviewData.layout);
        {% endif %}

        {% if charts.bloom_distribution %}
        var bloomData = {{ charts.bloom_distribution | safe }};
        Plotly.newPlot('bloom-distribution', bloomData.data, bloomData.layout);
        {% endif %}

        {% if charts.validation_scores %}
        var validationData = {{ charts.validation_scores | safe }};
        Plotly.newPlot('validation-scores', validationData.data, validationData.layout);
        {% endif %}

        {% if charts.confidence_distribution %}
        var confidenceData = {{ charts.confidence_distribution | safe }};
        Plotly.newPlot('confidence-distribution', confidenceData.data, confidenceData.layout);
        {% endif %}

        {% if charts.transcription_quality %}
        var qualityData = {{ charts.transcription_quality | safe }};
        Plotly.newPlot('transcription-quality', qualityData.data, qualityData.layout);
        {% endif %}

        {% if charts.multihop_ratio %}
        var multihopData = {{ charts.multihop_ratio | safe }};
        Plotly.newPlot('multihop-ratio', multihopData.data, multihopData.layout);
        {% endif %}
    </script>
</body>
</html>
"""


def generate_html_report(reports: list[RunReport], output_path: Path) -> None:
    """Generate a self-contained HTML report with interactive charts.

    Args:
        reports: List of RunReport objects to include in the report.
        output_path: Path to save the HTML file.
    """
    from datetime import datetime

    # Calculate summary statistics
    total_runs = len(reports)
    total_transcriptions = sum(len(r.transcription_records) for r in reports)
    total_qa_pairs = sum(sum(len(cep.qa_pairs) for cep in r.cep_records) for r in reports)

    success_rates = []
    for report in reports:
        if report.transcription_metadata or report.cep_metadata:
            metadata = report.cep_metadata or report.transcription_metadata
            if metadata and metadata.success_rate is not None:
                success_rates.append(metadata.success_rate)

    avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0

    # Prepare run table data
    run_data = []
    for report in reports:
        metadata = report.cep_metadata or report.transcription_metadata
        steps = ", ".join(report.pipeline.steps_run) if report.pipeline else "N/A"

        run_data.append(
            {
                "pipeline_id": report.pipeline_id,
                "steps": steps,
                "status": metadata.status.value if metadata else "unknown",
                "duration": metadata.duration_seconds if metadata else None,
                "success_rate": metadata.success_rate if metadata else None,
                "completed": metadata.completed_items if metadata else 0,
                "total": metadata.total_items if metadata else 0,
            }
        )

    # Generate charts
    chart_data = {}

    if reports:
        try:
            fig = charts.create_pipeline_overview_chart(reports)
            chart_data["pipeline_overview"] = fig.to_json()
        except Exception:
            pass

        try:
            fig = charts.create_bloom_distribution_chart(reports)
            chart_data["bloom_distribution"] = fig.to_json()
        except Exception:
            pass

        try:
            fig = charts.create_validation_scores_boxplot(reports)
            chart_data["validation_scores"] = fig.to_json()
        except Exception:
            pass

        try:
            fig = charts.create_confidence_distribution_chart(reports)
            chart_data["confidence_distribution"] = fig.to_json()
        except Exception:
            pass

        try:
            fig = charts.create_transcription_quality_chart(reports)
            chart_data["transcription_quality"] = fig.to_json()
        except Exception:
            pass

        try:
            fig = charts.create_multihop_ratio_chart(reports)
            chart_data["multihop_ratio"] = fig.to_json()
        except Exception:
            pass

    # Render template
    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        summary={
            "total_runs": total_runs,
            "total_transcriptions": total_transcriptions,
            "total_qa_pairs": total_qa_pairs,
            "avg_success_rate": avg_success_rate,
        },
        runs=run_data,
        charts=chart_data,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # Write to file
    output_path.write_text(html_content, encoding="utf-8")
