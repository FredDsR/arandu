# Docs: Agent Guide

`docs/` is the only documentation source. There is no docs site: GitHub renders
these files directly, so plain Markdown with relative `.md` links is the whole
toolchain. Never create a second copy of a page anywhere else in the repo; link
to it instead.

## Where a file goes

Place a page by the kind of content it carries:

| Content | Directory |
|---|---|
| Why the project exists and what it found | `research/` |
| The results of one canonical run | `research/results/<run-id>.md` |
| Notes that inform a decision but are not results | `research/notes/` |
| Material handed to human participants (annotators, raters) | `research/instruments/` |
| How to install, configure and call the CLI | `guide/` |
| How one pipeline step works and what it writes | `guide/pipelines/` |
| Running the pipeline on Docker or the pcad cluster | `operations/` |
| Internals for contributors: architecture, schemas, tests, CI | `development/` |
| Roadmap and design records that must outlive a branch | `planning/` |
| Local scratch data for research notes (gitignored, only `.gitkeep` is tracked) | `research/data/` |
| Mermaid sources referenced by the pages | `assets/diagrams/` |

If a new page fits none of these, ask before creating a new directory.

## Rules

1. **Update the index.** Every added, moved or deleted page gets its line in
   `docs/README.md` changed in the same commit.
2. **Results documents cite their commands.** Every number in
   `research/results/<run-id>.md` carries the command that reproduces it,
   usually `uv run python -m analysis.<module> --id <run-id>`. When an analysis
   module is renamed, update the commands here in the same commit.
3. **`docs/superpowers/` is local and gitignored.** Agent specs and plans
   written there are working files for one branch. Never link to them from
   tracked files, code or docstrings. A decision that must outlive the branch
   goes in the PR description, in the page it affects, or as a dated design
   record in `planning/` (`YYYY-MM-DD-<topic>-design.md`), which is not
   rewritten after merge.
4. **Keep the language of the neighbours.** Software docs (`guide/`,
   `operations/`, `development/`) are in English. Results, design records and
   instruments are in Portuguese (pt-BR). Instruments carry the language in the
   filename (`*.pt.md`).
5. **Move files with `git mv` and keep the content edits of that change
   small** (links and paths only). Git detects a rename on merge by content
   similarity between the merge base and the branch tip, so a moved file that
   was also rewritten shows up as a delete plus an add, and open branches that
   edit it will conflict.
6. **Document code changes where the reader looks for them.** A new CLI flag
   goes in `guide/cli-reference.md` and the matching `guide/pipelines/` page; a
   new env var goes in `guide/configuration.md`; a schema change goes in
   `development/schemas.md`.
7. **No file trees of the repository.** Do not draw the layout of the repo, of
   `src/`, `tests/`, `scripts/` or of `docs/` itself, in any page or index; it
   goes stale with the next move. Link to the directory or to the module
   instead. The one exception is the layout a command *writes* under
   `results/<run-id>/`, which is part of that command's contract.
   `docs/README.md` is a list of links grouped by audience, not a tree.

## What does not belong here

- Reports or tables generated from a run: `results/<run-id>/analysis/`.
- Figures: `analysis/figures/` (regenerated, gitignored).
- Run artifacts: `results/<run-id>/`.
- LaTeX drafts of the dissertation or papers: they live in the separate
  `dissertacao-tex` repository.
- Loose to-do lists: GitHub issues or `planning/ROADMAP.md`.
