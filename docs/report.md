# Results Visualization & Metrics Reporting

This module provides interactive visualization and reporting capabilities for G-Transcriber pipeline results.

## Features

- **Interactive HTML Dashboard**: Self-contained report with embedded Plotly charts, client-side filtering, and tabbed navigation
- **Publication-Quality PNG Export**: Automatic PNG export via Plotly + kaleido
- **Filesystem-Based Discovery**: Discovers results by scanning the filesystem directly (no dependency on index.json)
- **Headless Compatibility**: Works on HPC/SLURM environments without display servers
- **Colorblind-Friendly**: Uses Wong (2011) palette for accessibility

## Installation

Install the optional `report` dependency group:

```bash
uv sync --group report
```

Or with pip:

```bash
pip install plotly jinja2 kaleido
```

## Quick Start

### Generate HTML Report + PNG Charts

```bash
# Generate report for all pipeline runs (HTML + PNGs)
gtranscriber report

# Generate report for a specific run
gtranscriber report --run-id 20250101_120000

# Custom output path
gtranscriber report --output ./reports/dashboard.html
```

### Skip PNG Export

```bash
# Generate HTML only (no PNG files)
gtranscriber report --no-png
```

## CLI Options

```
gtranscriber report [OPTIONS]

Options:
  --run-id, --id TEXT      Generate report for a specific pipeline run
  --output, -o PATH        Output path for HTML report [default: report.html]
  --all-runs               Generate report for all runs (default if no --run-id)
  --no-png                 Skip automatic PNG export of charts
  --results-dir PATH       Path to results directory [default: from config]
  --help                   Show this message and exit
```

## Report Sections

### 1. Overview

- Run summary table (pipeline ID, steps run, status, duration, success rate, item count)
- Success rate and duration bar charts
- Run timeline (creation date vs. success rate, marker size = item count)

### 2. QA Analysis

- **Bloom Taxonomy Distribution**: Stacked bar chart per run
- **Validation Score Distributions**: Violin plots for faithfulness, bloom calibration, informativeness, self-containedness
- **Bloom Level x Criterion Heatmap**: Mean ± std per combination
- **Confidence Score Distribution**: Histogram of LLM generation confidence
- **Multi-hop vs Single-hop**: Horizontal bar chart
- **Correlation Matrix**: Pearson r between validation criteria and confidence
- **Parallel Coordinates**: Multi-dimensional quality profile colored by Bloom level

### 3. Transcriptions

- Quality score histograms (overall, script match, repetition, segment quality, content density)
- Validity rate bar chart
- Quality radar chart comparing mean scores across runs

### 4. Source Data

- Location > Participant treemap
- Participant breakdown (documents vs QA pairs)
- Validation scores by location (violin plots)
- Document-level detail table with expandable QA pairs

### 5. Cross-Run Comparison

- Side-by-side violin plots for each validation criterion
- Radar chart comparing mean quality profiles

## Architecture

### Module Structure

```
src/gtranscriber/core/report/
├── __init__.py           # Public API
├── collector.py          # Results discovery and aggregation
├── charts.py             # Plotly chart builders
├── dataset.py            # Flat data models and dataset builder
├── exporter.py           # PNG export via kaleido
├── generator.py          # HTML report assembly (Jinja2)
├── style.py              # Shared theme and color palettes
└── templates/
    ├── report.html.j2    # Main HTML template
    ├── _styles.css        # Dashboard CSS
    └── _filter_engine.js  # Client-side filtering and chart rendering
```

### Key Classes

#### ResultsCollector

```python
from gtranscriber.core.report import ResultsCollector

collector = ResultsCollector("results/")
runs = collector.discover_runs()           # List all pipeline IDs
report = collector.load_run("run_001")     # Load specific run
reports = collector.load_all_runs()        # Load all runs
```

#### Dataset Building and Report Generation

```python
from pathlib import Path
from gtranscriber.core.report import build_dataset, generate_html_report
from gtranscriber.core.report.exporter import export_charts_as_png

# Generate HTML report
generate_html_report(reports, Path("report.html"))

# Build dataset and export PNGs
dataset = build_dataset(reports)
export_charts_as_png(dataset, Path("figures/"))
```

## PNG Export Naming Convention

| File | Description |
|------|-------------|
| `pipeline_overview.png` | Success rate and duration bar charts |
| `bloom_distribution.png` | Stacked bar chart of QA pairs per Bloom level |
| `validation_scores.png` | Violin plots of validation criteria |
| `confidence_distribution.png` | Histogram of QA confidence scores |
| `transcription_quality.png` | Multi-panel histogram of quality sub-scores |
| `multihop.png` | Multi-hop vs single-hop bar chart |
| `correlation_heatmap.png` | Pearson correlation matrix |
| `quality_radar.png` | Mean quality profile radar chart |
| `parallel_coordinates.png` | Multi-dimensional QA quality profile |
| `run_timeline.png` | Run progression over time |
| `participant_breakdown.png` | Documents and QA pairs per participant |
| `location_treemap.png` | Location > participant hierarchy |
| `bloom_validation_heatmap.png` | Mean score by Bloom level and criterion |
| `location_quality.png` | Validation scores by recording location |

## Customization

### Color Palette

```python
from gtranscriber.core.report.style import (
    WONG_PALETTE,
    CATEGORICAL_COLORS,
    get_color_palette,
    get_bloom_color,
    get_criterion_color,
)

# Access color codes
blue = WONG_PALETTE["blue"]
colors = get_color_palette(5)  # First 5 colorblind-friendly colors
```

## Development

### Running Tests

```bash
pytest tests/core/report/ -v
```

### Linting

```bash
ruff check src/gtranscriber/core/report/
ruff format src/gtranscriber/core/report/
```

## Troubleshooting

### Missing Dependency

```bash
# Install report dependencies
uv sync --group report
# Or manually:
pip install plotly jinja2 kaleido
```

### No Results Found

Ensure your results directory structure matches the expected format:

```
results/
  {pipeline_id}/
    pipeline.json
    transcription/
      run_metadata.json
      outputs/*.json
    cep/
      run_metadata.json
      outputs/*_cep_qa.json
```

## References

- [Plotly Documentation](https://plotly.com/python/)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8(6), 441.
