# Scripts: Agent Guide

`scripts/` holds code that operates the pipeline: launching jobs, moving run
artifacts in and out, checking a deployment, or repairing state. Code that only
reads a finished run to answer a research question belongs in `analysis/`.
Cluster job scripts live in `slurm/`, which has its own guide.

## Where a new script goes

| It... | Put it in |
|---|---|
| reads `results/<run-id>/` and produces a table, figure or report | `analysis/` |
| will be run again on future runs (import, export, cost estimate, prompt review) | `scripts/` root |
| runs one component end to end on a small real sample to see that it works | `scripts/smoke/` |
| changes the state of an existing run, or exists for a single migration | `scripts/oneoff/` |
| submits or supports a SLURM job | `scripts/slurm/<step>/` |

## Rules

1. **Run from the repo root,** either as a module
   (`uv run python -m scripts.<package>.<name>`, for example
   `scripts.oneoff.migrate_chunk_id_namespace`) or by path
   (`uv run python scripts/<path>.py`). A script that imports from `scripts.*`
   or `analysis.*` must be run with `-m`. The directories are packages.
2. **Smoke scripts are named `smoke_*.py`, never `test_*.py`.** The `test_`
   prefix belongs to the pytest suite in `tests/`.
3. **One-offs record their reason.** The docstring names the run, the PR or
   issue, and the date. A one-off that mutates a run makes a backup first and
   supports a dry run. Once its run is retired, delete it; Git keeps the history.
4. **Reusable tools follow the package rules:** typed, Google docstrings, Rich
   console output (no `print()`), and tests in `tests/scripts/`.
5. **Register it.** Add or remove the matching line in `scripts/README.md` in
   the same commit.
6. **Logic that grows a second caller moves into `src/arandu`.** Scripts are
   thin entry points, not a second library.
7. **No file trees.** Do not draw the layout of this directory here or in
   `README.md`; the filesystem is the source of truth for what exists.
