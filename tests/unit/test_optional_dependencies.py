"""Tests for optional capability gates (Phase 7)."""

from __future__ import annotations

import pytest

from neural_memory.utils.optional_dependencies import (
    MissingCapabilityError,
    capability_status,
    has_capability,
    require_capability,
)


class TestRequireCapability:
    def test_stdlib_ok(self) -> None:
        mod = require_capability("json", "none", "JSON")
        assert mod is not None

    def test_missing_raises_install_hint(self) -> None:
        with pytest.raises(MissingCapabilityError) as ei:
            require_capability("definitely_not_a_real_pkg_xyz", "sync", "Fake Feature")
        err = ei.value
        assert "neural-memory[sync]" in str(err)
        assert err.extra == "sync"
        assert err.feature == "Fake Feature"

    def test_has_capability_json(self) -> None:
        assert has_capability("json") is True
        assert has_capability("definitely_not_a_real_pkg_xyz") is False

    def test_capability_status_dict(self) -> None:
        status = capability_status()
        assert isinstance(status, dict)
        assert "aiohttp" in status
