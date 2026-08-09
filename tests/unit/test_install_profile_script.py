"""Tests for install profile smoke script."""

from __future__ import annotations

from scripts.check_install_profiles import check_base


def test_base_profile_check_passes() -> None:
    report = check_base()
    assert report["profile"] == "base"
    assert report["ok"] is True
    names = {c["name"] for c in report["checks"] if c["ok"]}
    assert "import_neural_memory" in names
    assert "sqlite_encode_recall" in names
