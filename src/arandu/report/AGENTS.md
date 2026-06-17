# Report Module: Agent Guide

Aggregates pipeline results off the filesystem and renders them as an interactive
FastAPI dashboard (`arandu serve-report`) or a static HTML file (`arandu report`,
`generator.py`). Plotly charts, Jinja2 templating. No pipeline logic here — it is
a read-only view over `results/`.

## Module map

| Path | Role |
| ---- | ---- |
| `collector.py` | `ResultsCollector`: discovers runs by **walking `results/`** and merges per-run artifacts into a `RunReport` |
| `service.py` | `ReportService`: stateless filtering/pagination/aggregation over a cached `ReportDataset` |
| `api.py` | FastAPI app factory + thin routes (inject `ReportService` via `Depends`) |
| `generator.py` | `generate_html_report()`: Jinja2 (`templates/report.html.j2`) + embedded Plotly for offline use |
| `exporter.py` | CSV / static exports |
| `dataset.py`, `schemas.py` | `ReportDataset` (flat, template-ready) + API response/pagination/filter models |
| `charts/*.py` | Plotly figure builders (`quality`, `distribution`, `comparison`, `timeline`, `multihop`, `validation`); all import `charts/style.py` |

## Patterns to follow

- **Discover runs by walking `results/`, never `index.json`** — the PCAD sync can
  leave the index stale (`collector.py` is explicit about this).
- **New chart**: add `charts/<name>.py` with a `create_*` function (the existing
  convention: `create_transcription_quality_chart`, `create_bloom_distribution_chart`,
  ...) returning a Plotly `go.Figure`; pull colors/layout from `charts/style.py` —
  no magic hex, so the colorblind palette stays consistent.
- **New endpoint**: keep the `api.py` handler thin, put logic in a `ReportService`
  method, and return a `schemas.*` model (not a raw dict).
- **Template edits**: `templates/report.html.j2` keeps autoescape on; pass data
  pre-serialized into the context.

## Complex logic worth knowing

- `ResultsCollector.load_run()` merges several optional artifacts
  (`pipeline.json`, `run_metadata.json`, transcription + CEP records); missing
  files degrade gracefully. Results are cached, so refreshing the filesystem
  needs a new collector/service instance.
- Rationale extraction digs into `validation.stage_results[...].criterion_scores`
  — that nested shape is not type-checked here, so a judge-schema change can
  silently blank out rationales. Keep it in sync with `shared/judge/schemas.py`.

## Gotchas

- Hard-coded hex in a chart ⇒ palette drift; always import from `style.py`.
- Large runs: rely on the lazy dataset cache; don't eager-load everything at
  startup.
- A chart for a stage absent from the run renders empty rather than erroring —
  check the run actually has those records.

**Deployment surface**: `arandu:latest` (`Dockerfile`); `arandu serve-report` /
`arandu report` run on the head node, not a GPU job. Full map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
