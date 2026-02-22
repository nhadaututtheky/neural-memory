"""Tests for device identity generation and persistence."""

from __future__ import annotations

import pathlib
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from neural_memory.sync.device import DeviceInfo, get_device_id, get_device_info, get_device_name


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def config_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary directory for device ID storage."""
    return tmp_path / "config"


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGetDeviceId:
    """Test get_device_id generates and persists a stable device ID."""

    def test_get_device_id_generates_new(self, config_dir: pathlib.Path) -> None:
        """First call creates the file and returns a 16-character hex string."""
        device_id = get_device_id(config_dir)

        assert isinstance(device_id, str)
        assert len(device_id) == 16
        # Must be valid hexadecimal
        int(device_id, 16)  # raises ValueError if not hex

        # File must be created
        id_file = config_dir / "device_id"
        assert id_file.exists()
        assert id_file.read_text(encoding="utf-8").strip() == device_id

    def test_get_device_id_persistent(self, config_dir: pathlib.Path) -> None:
        """Second call returns the same ID as the first call."""
        first_id = get_device_id(config_dir)
        second_id = get_device_id(config_dir)

        assert first_id == second_id

    def test_get_device_id_handles_empty_file(self, config_dir: pathlib.Path) -> None:
        """If device_id file exists but is empty, a new ID is generated."""
        config_dir.mkdir(parents=True, exist_ok=True)
        id_file = config_dir / "device_id"
        id_file.write_text("", encoding="utf-8")

        new_id = get_device_id(config_dir)

        # Should have generated a valid 16-char hex ID
        assert isinstance(new_id, str)
        assert len(new_id) == 16
        int(new_id, 16)

        # File must now contain the new ID
        assert id_file.read_text(encoding="utf-8").strip() == new_id

    def test_get_device_id_handles_whitespace_only_file(
        self, config_dir: pathlib.Path
    ) -> None:
        """Whitespace-only file is treated as empty → generates new ID."""
        config_dir.mkdir(parents=True, exist_ok=True)
        id_file = config_dir / "device_id"
        id_file.write_text("   \n\t  ", encoding="utf-8")

        new_id = get_device_id(config_dir)
        assert len(new_id) == 16

    def test_get_device_id_creates_config_dir(self, tmp_path: pathlib.Path) -> None:
        """get_device_id creates the config dir if it does not exist."""
        deep_dir = tmp_path / "nested" / "config" / "dir"
        assert not deep_dir.exists()

        get_device_id(deep_dir)
        assert deep_dir.exists()

    def test_get_device_id_uniqueness(self, tmp_path: pathlib.Path) -> None:
        """Different config dirs produce different device IDs."""
        dir_a = tmp_path / "config_a"
        dir_b = tmp_path / "config_b"

        id_a = get_device_id(dir_a)
        id_b = get_device_id(dir_b)

        # Statistically near-impossible to collide (UUID4 prefix)
        assert id_a != id_b


class TestGetDeviceName:
    """Test get_device_name returns a non-empty string."""

    def test_get_device_name_returns_non_empty(self) -> None:
        """get_device_name returns a non-empty string."""
        name = get_device_name()

        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_device_name_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If platform.node() returns empty string, falls back to 'unknown'."""
        import platform

        monkeypatch.setattr(platform, "node", lambda: "")
        name = get_device_name()

        assert name == "unknown"

    def test_get_device_name_uses_platform_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_device_name returns the value of platform.node()."""
        import platform

        monkeypatch.setattr(platform, "node", lambda: "my-test-machine")
        name = get_device_name()

        assert name == "my-test-machine"


class TestGetDeviceInfo:
    """Test get_device_info returns a fully-populated DeviceInfo."""

    def test_get_device_info_returns_device_info(
        self, config_dir: pathlib.Path
    ) -> None:
        """get_device_info returns a DeviceInfo instance."""
        info = get_device_info(config_dir)
        assert isinstance(info, DeviceInfo)

    def test_get_device_info_all_fields_populated(
        self, config_dir: pathlib.Path
    ) -> None:
        """All fields on DeviceInfo are non-empty / non-None."""
        info = get_device_info(config_dir)

        assert isinstance(info.device_id, str)
        assert len(info.device_id) == 16

        assert isinstance(info.device_name, str)
        assert len(info.device_name) > 0

        assert isinstance(info.registered_at, datetime)

    def test_get_device_info_device_id_stable(
        self, config_dir: pathlib.Path
    ) -> None:
        """Calling get_device_info twice returns the same device_id."""
        info1 = get_device_info(config_dir)
        info2 = get_device_info(config_dir)

        assert info1.device_id == info2.device_id

    def test_get_device_info_registered_at_is_utcnow(
        self, config_dir: pathlib.Path
    ) -> None:
        """registered_at is close to the current UTC time (within 5 seconds)."""
        from neural_memory.utils.timeutils import utcnow

        before = utcnow()
        info = get_device_info(config_dir)
        after = utcnow()

        assert before <= info.registered_at <= after


class TestDeviceInfoImmutable:
    """Test DeviceInfo is a frozen dataclass."""

    def test_device_info_immutable(self) -> None:
        """DeviceInfo fields cannot be mutated after creation."""
        info = DeviceInfo(
            device_id="abc123def456abcd",
            device_name="test-machine",
            registered_at=datetime(2026, 1, 1),
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            info.device_id = "new-id"  # type: ignore[misc]

    def test_device_info_fields_accessible(self) -> None:
        """All DeviceInfo fields are readable."""
        ts = datetime(2026, 2, 22, 12, 0, 0)
        info = DeviceInfo(
            device_id="1234567890abcdef",
            device_name="laptop",
            registered_at=ts,
        )
        assert info.device_id == "1234567890abcdef"
        assert info.device_name == "laptop"
        assert info.registered_at == ts
