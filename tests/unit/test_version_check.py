"""Tests for version check handler — editable install detection."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from neural_memory.mcp.version_check_handler import (
    VersionInfo,
    _is_editable_install,
    _is_newer,
    _parse_version,
)


class TestParseVersion:
    def test_simple_version(self) -> None:
        assert _parse_version("2.21.0") == (2, 21, 0)

    def test_prerelease_stripped(self) -> None:
        assert _parse_version("2.5.0a1") == (2, 5, 0)

    def test_dev_stripped(self) -> None:
        assert _parse_version("3.0.0dev1") == (3, 0, 0)


class TestIsNewer:
    def test_newer(self) -> None:
        assert _is_newer("2.22.0", "2.21.0") is True

    def test_same(self) -> None:
        assert _is_newer("2.21.0", "2.21.0") is False

    def test_older(self) -> None:
        assert _is_newer("2.20.0", "2.21.0") is False


class TestIsEditableInstall:
    @patch("importlib.metadata.distribution")
    def test_detects_editable_from_direct_url(self, mock_dist_fn: MagicMock) -> None:
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = '{"url": "file:///path", "dir_info": {"editable": true}}'
        mock_dist_fn.return_value = mock_dist
        assert _is_editable_install() is True

    def test_returns_bool(self) -> None:
        # In test environment (editable install), should return True
        # In CI (pip install), may return either. Just ensure no crash.
        result = _is_editable_install()
        assert isinstance(result, bool)


class TestGetUpdateHint:
    def test_no_hint_when_no_update(self) -> None:
        handler = _make_handler(update_available=False)
        assert handler.get_update_hint() is None

    def test_no_hint_when_not_checked(self) -> None:
        handler = _make_handler(version_info=None)
        assert handler.get_update_hint() is None

    def test_normal_hint_for_pip_install(self) -> None:
        handler = _make_handler(update_available=True)
        with patch(
            "neural_memory.mcp.version_check_handler._is_editable_install", return_value=False
        ):
            hint = handler.get_update_hint()
        assert hint is not None
        assert "pip install --upgrade" in hint["message"]
        assert "editable" not in hint

    def test_editable_hint_for_dev_install(self) -> None:
        handler = _make_handler(update_available=True)
        with patch(
            "neural_memory.mcp.version_check_handler._is_editable_install", return_value=True
        ):
            hint = handler.get_update_hint()
        assert hint is not None
        assert hint["editable"] is True
        assert "editable install" in hint["message"]
        assert "pip install --upgrade" not in hint["message"]


def _make_handler(
    *,
    update_available: bool = False,
    version_info: VersionInfo | None | str = "auto",
) -> MagicMock:
    """Create a mock handler with VersionCheckHandler.get_update_hint."""
    from neural_memory.mcp.version_check_handler import VersionCheckHandler

    handler = MagicMock(spec=VersionCheckHandler)
    handler.get_update_hint = VersionCheckHandler.get_update_hint.__get__(handler)

    if version_info == "auto":
        handler._version_info = VersionInfo(
            current="2.21.0",
            latest="2.22.0",
            update_available=update_available,
            checked_at=datetime(2026, 3, 3),
        )
    else:
        handler._version_info = version_info

    return handler
