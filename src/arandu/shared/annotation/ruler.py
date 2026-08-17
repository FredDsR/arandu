"""Load the signed-off emic-validity ruler and enforce its gate.

``prompts/judge/criteria/emic_validity/ruler.pt.yaml`` is the single source of
the construct, the 1-5 scale and the loss types. The judge prompt and the
annotator sheet already render it; the Label Studio labeling config is the third
consumer.

The ``signed_off`` flag is the mechanical form of the
``anthropologist-validation-of-readings`` gate: while it is false the annotation
build refuses to run, so an unreviewed ruler cannot reach the annotators.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import yaml

from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

RULER_PATH: Path = (
    get_project_root() / "prompts" / "judge" / "criteria" / "emic_validity" / "ruler.pt.yaml"
)

GATE_NAME = "anthropologist-validation-of-readings"


class RulerNotSignedOffError(RuntimeError):
    """Raised when the ruler has not been signed off by the anthropologist."""


def load_ruler(path: Path | None = None) -> dict[str, Any]:
    """Load the emic-validity ruler.

    Args:
        path: Override the shipped ruler location (tests and audits).

    Returns:
        The parsed ruler mapping.

    Raises:
        FileNotFoundError: If the ruler file does not exist.
        ValueError: If the file does not parse into a mapping.
    """
    target = path if path is not None else RULER_PATH
    if not target.exists():
        raise FileNotFoundError(f"Emic ruler not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Emic ruler at {target} did not parse into a mapping.")
    return data


def ruler_sha256(path: Path | None = None) -> str:
    """Return the SHA-256 of the ruler file bytes.

    Recorded in the annotation manifest so an auditor can tell which ruler the
    annotators actually saw. A ruler edit after the project is pushed would
    otherwise be invisible.

    Args:
        path: Override the shipped ruler location (tests and audits).

    Returns:
        The hex-encoded SHA-256 digest of the ruler file contents.
    """
    target = path if path is not None else RULER_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


def require_signed_off(ruler: dict[str, Any]) -> None:
    """Raise unless the ruler carries an explicit boolean sign-off.

    Only ``True`` passes. A missing flag, a string, or any other truthy value is
    treated as unsigned: a typo must not open the gate.

    Args:
        ruler: The parsed ruler mapping, as returned by `load_ruler`.

    Raises:
        RulerNotSignedOffError: If the ruler is not signed off.
    """
    if ruler.get("signed_off") is not True:
        gate = ruler.get("gate", GATE_NAME)
        raise RulerNotSignedOffError(
            f"The emic ruler is not signed off (signed_off is not true). The gate {gate!r} "
            f"must close first: the anchors the annotators read have to be the reviewed ones, "
            f"and a mismatch with the judge prompt is not recoverable after annotation. "
            f"Ruler: {RULER_PATH}"
        )
