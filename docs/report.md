# Results Visualization & Metrics Reporting

This module provides comprehensive visualization and reporting capabilities for G-Transcriber pipeline results.

## Features

- **Interactive HTML Reports**: Self-contained reports with embedded Plotly charts
- **Publication-Quality Figures**: Export static images (PNG, SVG, PDF) for papers and presentations
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
pip install plotly jinja2 matplotlib seaborn scipy
```

## Quick Start

### Generate HTML Report

```bash
# Generate report for all pipeline runs
gtranscriber report

# Generate report for specific run
gtranscriber report --run-id 20250101_120000 --output my_report.html
```

### Export Static Figures

```bash
# Export figures as PNG
gtranscriber report --export-figures ./figures/ --figure-format png

# Export as SVG for publications
gtranscriber report --export-figures ./figures/ --figure-format svg

# Export as PDF for LaTeX documents
gtranscriber report --export-figures ./figures/ --figure-format pdf
```

### Combined Usage

```bash
# Generate both HTML and figures
gtranscriber report --output report.html --export-figures ./figures/
```

## Report Sections

### 1. Pipeline Overview Dashboard
- Run summary table (run_id, pipeline_type, status, duration, success_rate)
- Success rate comparison across runs (bar chart)
- Processing timeline (duration per run)

### 2. CEP QA Validation Metrics
- **Bloom's Taxonomy Distribution**: Stacked bar chart showing QA pairs per cognitive level
- **Validation Score Distributions**: Box plots for faithfulness, bloom_calibration, informativeness, self_containedness
- **Confidence Score Distribution**: Histogram with KDE overlay
- **Multi-hop Ratio**: Pie chart showing single-hop vs multi-hop questions

### 3. Transcription Quality
- Distribution plots for quality sub-scores
- Overall score histogram
- Validity rate visualization

### 4. Source Metadata Summary
- Document count by participant and location
- Recording date timeline
- Processing coverage

### 5. Hardware & Execution Context
- Device type breakdown (CPU/CUDA/MPS)
- GPU model comparison
- SLURM vs local execution stats

## CLI Options

```
gtranscriber report [OPTIONS]

Options:
  --run-id TEXT              Generate report for specific pipeline run
  --output PATH              Output path for HTML report [default: report.html]
  --all-runs                 Generate report for all runs (default behavior)
  --export-figures PATH      Directory to export static figures
  --figure-format TEXT       Output format: png, svg, or pdf [default: png]
  --results-dir PATH         Path to results directory [default: from config]
  --help                     Show this message and exit
```

## Architecture

### Module Structure

```
src/gtranscriber/core/report/
├── __init__.py           # Public API
├── collector.py          # Results discovery and aggregation
├── charts.py             # Plotly chart builders for HTML
├── figures.py            # Matplotlib/Seaborn builders for static export
├── generator.py          # HTML report assembly
└── style.py              # Shared theme and color palettes
```

### Data Models

- **ResultsCollector**: Discovers and loads pipeline results from filesystem
- **RunReport**: Aggregates pipeline metadata and outputs for a single run

### Key Classes

#### ResultsCollector

```python
from gtranscriber.core.report import ResultsCollector

collector = ResultsCollector("results/")
runs = collector.discover_runs()           # List all pipeline IDs
report = collector.load_run("run_001")     # Load specific run
reports = collector.load_all_runs()        # Load all runs
```

#### Report Generation

```python
from gtranscriber.core.report.generator import generate_html_report
from gtranscriber.core.report.figures import export_all_figures
from pathlib import Path

# Generate HTML report
generate_html_report(reports, Path("report.html"))

# Export static figures
export_all_figures(reports, Path("figures/"), format="png")
```

## Figure Naming Convention

| Figure File | Description |
|-------------|-------------|
| `bloom_distribution.{ext}` | Stacked bar chart of QA pairs per Bloom level |
| `validation_scores_boxplot.{ext}` | Box plots of validation criteria |
| `confidence_distribution.{ext}` | Histogram of QA confidence scores |
| `transcription_quality.{ext}` | Multi-panel histogram of quality sub-scores |
| `multihop_ratio.{ext}` | Pie chart of multi-hop vs single-hop questions |
| `run_comparison.{ext}` | Bar chart comparing success rates across runs |

## Customization

### Custom Color Palette

```python
from gtranscriber.core.report.style import WONG_PALETTE, CATEGORICAL_COLORS

# Access color codes
blue = WONG_PALETTE["blue"]
colors = CATEGORICAL_COLORS[:5]  # First 5 colors
```

### Matplotlib Style

```python
from gtranscriber.core.report.style import get_matplotlib_style
import matplotlib.pyplot as plt

style = get_matplotlib_style()
plt.rcParams.update(style)
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

### Display Backend Error

If you encounter "no display name and no $DISPLAY environment variable" errors:

```bash
# The module automatically uses Agg backend for headless environments
# If issues persist, explicitly set before importing:
export MPLBACKEND=Agg
```

### Missing Dependency

```bash
# Install report dependencies
pip install plotly jinja2 matplotlib seaborn scipy
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

## Future Enhancements (Phase 2)

- Interactive dashboard with `--serve` flag (Streamlit/Dash/Panel)
- Document-level drill-down with QA pair inspection
- Cross-run filtering and comparison
- Real-time monitoring of in-progress runs
- CSV export of filtered data

## References

- [Plotly Documentation](https://plotly.com/python/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8(6), 441.
