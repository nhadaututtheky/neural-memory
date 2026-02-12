"""Tests for unified_config.py — legacy DB migration."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from neural_memory.unified_config import (
    UnifiedConfig,
    _MIN_LEGACY_DB_BYTES,
    _migrate_legacy_db,
)


def _create_fake_db(path: Path, *, size: int = 0) -> None:
    """Create a minimal SQLite database at *path*.

    If *size* is given and larger than a bare DB, pad with extra data so
    ``stat().st_size >= size``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY)")
    conn.execute("INSERT OR IGNORE INTO neurons VALUES ('test-neuron-1')")
    conn.commit()
    conn.close()

    # Pad to requested size if needed.
    current = path.stat().st_size
    if size > current:
        with open(path, "ab") as f:
            f.write(b"\x00" * (size - current))


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary NeuralMemory data directory."""
    return tmp_path / ".neuralmemory"


def _make_config(data_dir: Path) -> UnifiedConfig:
    """Build a UnifiedConfig pointing at *data_dir* with brain='default'."""
    return UnifiedConfig(data_dir=data_dir, current_brain="default")


# ── Happy path ───────────────────────────────────────────────────


class TestMigrateLegacyDb:
    def test_copies_when_old_exists_and_new_does_not(
        self, tmp_data_dir: Path
    ) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert new_db.exists()
        assert new_db.stat().st_size == old_db.stat().st_size

        # Old file still exists (backup).
        assert old_db.exists()

    def test_copies_wal_and_shm_if_present(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        # Create fake WAL/SHM companions.
        wal = old_db.with_name("default.db-wal")
        shm = old_db.with_name("default.db-shm")
        wal.write_bytes(b"wal-data")
        shm.write_bytes(b"shm-data")

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        brains = tmp_data_dir / "brains"
        assert (brains / "default.db-wal").read_bytes() == b"wal-data"
        assert (brains / "default.db-shm").read_bytes() == b"shm-data"

    # ── Skip conditions ──────────────────────────────────────────

    def test_skips_when_new_already_exists(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        new_db = tmp_data_dir / "brains" / "default.db"
        new_db.parent.mkdir(parents=True, exist_ok=True)
        new_db.write_text("existing")

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        # Should NOT overwrite existing new DB.
        assert new_db.read_text() == "existing"

    def test_skips_non_default_brain(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, "my-custom-brain")

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    def test_skips_small_file(self, tmp_data_dir: Path) -> None:
        """An empty-schema DB (< _MIN_LEGACY_DB_BYTES) is not migrated."""
        old_db = tmp_data_dir / "default.db"
        old_db.parent.mkdir(parents=True, exist_ok=True)
        old_db.write_bytes(b"\x00" * 4096)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    def test_skips_when_old_does_not_exist(self, tmp_data_dir: Path) -> None:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    # ── Error resilience ─────────────────────────────────────────

    def test_handles_copy_error_gracefully(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)

        with patch("neural_memory.unified_config.shutil.copy2", side_effect=OSError("disk full")):
            # Should not raise — logs warning instead.
            _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    # ── Config brain name resolution ─────────────────────────────

    def test_uses_config_current_brain_when_none(
        self, tmp_data_dir: Path
    ) -> None:
        """When brain_name is None, uses config.current_brain."""
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        assert config.current_brain == "default"

        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert new_db.exists()
