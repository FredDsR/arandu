# Analysis: Agent Guide

`analysis/` holds everything that reads a finished run under `results/<run-id>/`
and answers a question for the dissertation: modules that print or write
tables and plots, and notebooks. If the code changes pipeline state or prepares
a run, it belongs in `scripts/`, not here.

Modules sit directly in `analysis/`, without subpackages by domain. Name
prefixes (`analyze_judge_*`, `analyze_emic_*`, `kg_*`, `plot_*`) do the
grouping, and modules import from each other freely. Notebooks go in
`notebooks/` and their figures in `figures/` (gitignored). `README.md` is the
index that maps each question to its module or notebook.

## Rules for a module

1. **Read-only.** Never write into a stage's `outputs/` or touch checkpoints.
   When a module writes files, new modules default to
   `results/<run-id>/analysis/`; explicit `--out`, `--out-json` or `--out-md`
   flags (several existing modules require them) are fine.
2. **No LLM calls.** Analyses recompute from stored verdicts and scores. Anything
   that re-judges or regenerates is a pipeline step and goes through the CLI.
3. **Run as a module from the repo root:** `uv run python -m analysis.<module>`.
   Do not hardcode a run id or an absolute path. New modules that analyse one
   run take it through `--id <run-id>`.
4. **The module docstring states the question** it answers and shows the exact
   command above. Reuse production aggregators from `arandu.*` instead of
   reimplementing metrics.
5. **Shared helpers go in an `_`-prefixed module** here. If a helper is needed
   by `src/arandu` too, move it into the package instead.
6. **Tests live in `tests/analysis/`**, one file per module that has logic worth
   testing, built on `tmp_path` fixtures rather than real runs.
7. **Register it.** Add a line to `analysis/README.md`. If a number it produces
   is cited in `docs/research/results/<run-id>.md`, the command there must match.

## Notebooks

- Resolve paths from the repo root and honour `ARANDU_RESULTS_BASE_DIR`. Save
  figures to `analysis/figures/` with the run id in the filename
  (`<figure>-<run-id>.pdf` plus a 300 dpi `.png`).
- Per-run copies of a notebook carry the run id as a suffix
  (`<notebook>-<run-id>.ipynb`). Prefer parametrizing the run id over copying.
- Logic that two notebooks need, or that a results number depends on, moves
  into a module here so it can be tested and called from the command line.

## Generated Markdown

Markdown exported from a run (pair listings, chunk dumps, qualitative review
sheets) is an output, not a source file. Write it under
`results/<run-id>/analysis/`, never into the repository, and regenerate it
instead of editing it by hand.

## Keeping it organized

Do not draw a file tree of this directory here or in `README.md`. The index
lists questions and links to modules; the filesystem is the source of truth for
what exists.
