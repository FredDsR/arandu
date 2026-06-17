# CLI Module: Agent Guide

Thin Typer command layer. Each `cli/<feature>.py` defines one command function;
`cli/app.py` imports and registers it with `app.command()(func)`. Commands parse
args, load a config/settings object from env + CLI overrides, call a domain batch
orchestrator, and report via Rich. They are fire-and-forget: side effects land in
`results/<id>/<stage>/outputs/`, not in a return value.

## Module map

| File | Role |
| ---- | ---- |
| `app.py` | Typer app + the single registration list (`app.command()(fn)` per command) |
| `transcribe.py`, `qa.py`, `kg.py`, `chunking.py`, `retrieve.py`, `answer.py`, `judge_answers.py`, `emic_prepass.py`, `human_eval.py`, `rag_analysis.py`, `non_answerable.py`, `manage.py`, `report.py` | One command (or a few related commands) each |
| `_helpers.py` | Shared output/sanitization helpers |

## Patterns to follow

- **Add a command**: write `def <name>(...)` in `cli/<name>.py` with
  `Annotated[...]` Typer params and a Google-style docstring, then import it in
  `app.py` and add `app.command()(<name>)`. Forgetting the registration line
  means the command silently doesn't exist.
- **Config/settings**: load the stage's config (`QAConfig`, `KGConfig`, an
  `LLMSettings` subclass, ...) at the top of the function; merge CLI overrides
  with `Config.model_validate({**cfg.model_dump(), **overrides})`. For LLM
  stages use `build_llm_client_from_settings` / `parse_provider` from
  `shared.llm_client` — never instantiate a provider SDK directly.
- **Output**: only `print_info` / `print_success` / `print_warning` /
  `print_error` from `arandu.utils.logger` — never bare `print()`. Echo run id +
  key config before work; counts/success at the end. Exit failures with
  `typer.Exit(code=1)`.
- **Errors**: catch `FileNotFoundError` (missing input) and
  `ValueError`/`RuntimeError` (bad config) and surface them via `print_error`
  before exiting; let truly unexpected errors propagate.

## Complex logic worth knowing

- CLI functions are exempt from the strict type-annotation rule (Typer's
  `Annotated` wrapping obscures inference) — see the root `AGENTS.md`. The
  exemption is a Ruff per-file-ignore on the whole `cli/*.py` glob (ANN001/ANN201),
  so it also covers `_helpers.py`; still annotate helper logic where practical.
- Settings are constructed fresh per invocation (no global singleton), so env
  changes take effect across reruns in the same shell. CLI flags always override
  env, which overrides `.env`, which overrides defaults.

## Gotchas

- Inconsistent `env_prefix` on a new settings subclass ⇒ values silently fall
  back to the base class's env.
- Mixing `logging` and `print_*` in one command ⇒ out-of-order, unstyled output.
- `setup_logging()` runs in the app callback; standalone scripts that import a
  command function must call it themselves for Rich output.

**Deployment surface**: commands run as the image `ENTRYPOINT ["arandu"]`; which
service/profile/image runs each command is in
[scripts/slurm/AGENTS.md](../../../scripts/slurm/AGENTS.md).
