"""The ruler loader is the mechanical form of the anthropologist gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from arandu.shared.annotation.ruler import (
    RULER_PATH,
    RulerNotSignedOffError,
    load_ruler,
    require_signed_off,
    ruler_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_ruler(tmp_path: Path, *, signed_off: bool) -> Path:
    path = tmp_path / "ruler.pt.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "signed_off": signed_off,
                "gate": "anthropologist-validation-of-readings",
                "scale": [{"score": 5, "label": "Preserva o sentido."}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


class TestLoadRuler:
    def test_default_path_points_at_the_shipped_ruler(self) -> None:
        assert RULER_PATH.name == "ruler.pt.yaml"
        assert RULER_PATH.parent.name == "emic_validity"
        assert RULER_PATH.exists()

    def test_loads_the_shipped_ruler(self) -> None:
        ruler = load_ruler()
        assert ruler["signed_off"] is True
        assert [entry["score"] for entry in ruler["scale"]] == [5, 4, 3, 2, 1]

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"ruler\.pt\.yaml"):
            load_ruler(tmp_path / "absent" / "ruler.pt.yaml")


class TestSignOffGate:
    def test_signed_off_ruler_passes(self, tmp_path: Path) -> None:
        require_signed_off(load_ruler(_write_ruler(tmp_path, signed_off=True)))

    def test_unsigned_ruler_raises_and_names_the_gate(self, tmp_path: Path) -> None:
        ruler = load_ruler(_write_ruler(tmp_path, signed_off=False))
        with pytest.raises(RulerNotSignedOffError) as excinfo:
            require_signed_off(ruler)
        assert "anthropologist-validation-of-readings" in str(excinfo.value)

    def test_absent_flag_is_treated_as_unsigned(self, tmp_path: Path) -> None:
        path = tmp_path / "ruler.pt.yaml"
        path.write_text(yaml.safe_dump({"scale": []}), encoding="utf-8")
        with pytest.raises(RulerNotSignedOffError):
            require_signed_off(load_ruler(path))

    def test_truthy_non_bool_is_not_accepted_as_signed(self, tmp_path: Path) -> None:
        """`signed_off: "yes"` is a typo, not a signature."""
        path = tmp_path / "ruler.pt.yaml"
        path.write_text(yaml.safe_dump({"signed_off": "yes"}), encoding="utf-8")
        with pytest.raises(RulerNotSignedOffError):
            require_signed_off(load_ruler(path))


class TestRulerHash:
    def test_hash_is_stable_and_content_addressed(self, tmp_path: Path) -> None:
        first = _write_ruler(tmp_path / "a", signed_off=True)
        second = _write_ruler(tmp_path / "b", signed_off=True)
        assert ruler_sha256(first) == ruler_sha256(second)

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        signed = _write_ruler(tmp_path / "a", signed_off=True)
        unsigned = _write_ruler(tmp_path / "b", signed_off=False)
        assert ruler_sha256(signed) != ruler_sha256(unsigned)
