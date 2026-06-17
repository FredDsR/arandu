# Utils Module: Agent Guide

Cross-cutting helpers used everywhere: Rich logging/console, path discovery,
bounded concurrency with rate-limit adaptation, and LLM-response text parsing.
No pipeline logic lives here — keep it dependency-light and side-effect-free.

## Module map

| File | Role |
| ---- | ---- |
| `logger.py` | `setup_logging`, `get_logger`, and the `print_info`/`print_error`/`print_success`/`print_warning` helpers |
| `console.py` | Shared Rich `console` (stdout) + stderr console |
| `paths.py` | `get_project_root()` (walks up to `pyproject.toml`) |
| `concurrency.py` | `map_concurrent` (bounded thread pool, rate-limit retries) + `AdaptiveThrottle` |
| `text.py` | `extract_thinking` (`<think>` stripping), `validate_score` / `validate_ordinal_score`, codeblock stripping |

## Patterns to follow

- **User-facing output goes through `print_*`** from `logger.py`, never bare
  `print()`. Use `logging.getLogger("arandu.<mod>")` only for debug/trace.
- **Batch LLM work uses `map_concurrent(fn, items, workers=N, ...)`** — do not
  hand-roll a retry loop. Pass the rate-limit predicate so rate-limited items
  requeue (adapting via `AdaptiveThrottle`) instead of failing the record.
- **Parse model output** with `extract_thinking()` before JSON-decoding, and
  clamp numbers through `validate_score` / `validate_ordinal_score` even when you
  trust the model.
- **Resolve the project root** with `get_project_root()`; don't compute paths
  relative to `__file__` by hand.

## Complex logic worth knowing

- `AdaptiveThrottle` is AIMD: a rate-limit event multiplicatively drops the
  in-flight limit and pauses acquisition; consecutive successes additively
  restore slots. It is thread-based (`threading.Condition`), so it does not apply
  to async code.
- `map_concurrent(workers=1)` runs inline (no pool); `workers>1` keeps a bounded
  submission window so memory stays flat across tens of thousands of items.
- `validate_ordinal_score` rounds reasoning-model fractional labels half-up
  (e.g. `3.5 → 4`) so a stray decimal doesn't poison the result into an error.

## Gotchas

- Forgetting `setup_logging()` at the entrypoint ⇒ logs render without Rich
  styling.
- Not passing the rate-limit predicate to `map_concurrent` against a throttled
  provider ⇒ the batch dies on the first 429 instead of adapting.
- `get_project_root()` walks up for `pyproject.toml`; code run outside the repo
  tree won't find it.

This module is image-agnostic (imported by every service). Deployment map:
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
