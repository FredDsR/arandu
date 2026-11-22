"""Skip the report test suite when the optional ``report`` extra is absent.

The ``arandu.report`` package depends on ``plotly`` (and ``kaleido``), which
live in the optional ``report`` extra (``pyproject.toml``
``[project.optional-dependencies]``). Without that extra installed, importing
the report test modules fails at collection time. ``importorskip`` turns that
hard collection error into a clean skip, so ``pytest`` over the whole tree
works after a bare ``uv sync``; the tests still run in CI / after
``uv sync --extra report``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")
