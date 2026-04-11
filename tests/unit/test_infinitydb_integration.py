"""Tests for InfinityDB integration in unified_config.py.

Covers:
- Per-brain caching (no singleton)
- list_brains() detects InfinityDB directories
- Graceful fallback when InfinityDB open() fails
- Auto-migration hint logging when SQLite data exists
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.unified_config import UnifiedConfig, _get_infinitydb_storage

_pro_deps_available = True
try:
    import numpy
except ImportError:
    _pro_deps_available = False


def _make_config(data_dir: Path, brain: str = "default") -> UnifiedConfig:
    """Build a UnifiedConfig pointing at *data_dir*."""
    return UnifiedConfig(data_dir=data_dir, current_brain=brain, storage_backend="infinitydb")


# ── list_brains ──────────────────────────────────────────────────


class TestListBrainsInfinityDB:
    """list_brains() should detect both SQLite .db files and InfinityDB dirs."""

    def test_detects_sqlite_only(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)
        (brains_dir / "work.db").touch()
        (brains_dir / "personal.db").touch()

        result = cfg.list_brains()
        assert result == ["personal", "work"]

    def test_detects_infinitydb_only(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        (brains_dir / "my-brain").mkdir(parents=True)
        (brains_dir / "my-brain" / "brain.inf").touch()

        result = cfg.list_brains()
        assert result == ["my-brain"]

    def test_detects_both_backends(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)

        # SQLite brain
        (brains_dir / "sqlite-brain.db").touch()

        # InfinityDB brain
        (brains_dir / "inf-brain").mkdir()
        (brains_dir / "inf-brain" / "brain.inf").touch()

        result = cfg.list_brains()
        assert result == ["inf-brain", "sqlite-brain"]

    def test_deduplicates_when_both_exist(self, tmp_path: Path) -> None:
        """Brain with both .db and InfinityDB dir should appear once."""
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)

        (brains_dir / "shared.db").touch()
        (brains_dir / "shared").mkdir()
        (brains_dir / "shared" / "brain.inf").touch()

        result = cfg.list_brains()
        assert result == ["shared"]

    def test_ignores_dirs_without_brain_inf(self, tmp_path: Path) -> None:
        """Random directories in brains/ should not appear."""
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        (brains_dir / "random-dir").mkdir(parents=True)
        (brains_dir / "random-dir" / "some-file.txt").touch()

        result = cfg.list_brains()
        assert result == []

    def test_empty_brains_dir(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = cfg.list_brains()
        assert result == []


# ── _get_infinitydb_storage ──────────────────────────────────────


class TestGetInfinityDBStorage:
    """Per-brain caching, fallback, and migration hints."""

    @pytest.mark.asyncio
    async def test_falls_back_to_sqlite_when_pro_missing(self, tmp_path: Path) -> None:
        """When Pro deps are not installed, should fall back to SQLite."""
        import builtins

        cfg = _make_config(tmp_path, brain="test-brain")
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "neural_memory.pro.storage_adapter":
                raise ImportError("Simulated missing Pro deps")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("neural_memory.plugins.get_storage_class", return_value=None),
            patch(
                "neural_memory.unified_config._get_sqlite_storage", new_callable=AsyncMock
            ) as mock_sqlite,
        ):
            mock_sqlite.return_value = MagicMock()
            await _get_infinitydb_storage(cfg, "test-brain")
            mock_sqlite.assert_called_once_with(cfg, "test-brain", None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _pro_deps_available, reason="Pro deps (numpy/hnswlib) not installed")
    async def test_falls_back_to_sqlite_when_open_fails(self, tmp_path: Path) -> None:
        """When InfinityDB open() raises, should fall back to SQLite."""
        cfg = _make_config(tmp_path, brain="broken-brain")
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)

        mock_storage = MagicMock()
        mock_storage.return_value.open = AsyncMock(side_effect=RuntimeError("WAL lock"))
        mock_storage.return_value._db = None

        with (
            patch("neural_memory.pro.storage_adapter.InfinityDBStorage", mock_storage),
            patch(
                "neural_memory.unified_config._get_sqlite_storage", new_callable=AsyncMock
            ) as mock_sqlite,
        ):
            mock_sqlite.return_value = MagicMock()
            # Clear cache to force fresh creation
            from neural_memory.unified_config import _storage_cache

            _storage_cache.pop("inf:broken-brain", None)

            await _get_infinitydb_storage(cfg, "broken-brain")
            mock_sqlite.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _pro_deps_available, reason="Pro deps (numpy/hnswlib) not installed")
    async def test_per_brain_caching(self, tmp_path: Path) -> None:
        """Each brain should get its own cached InfinityDB instance."""
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)

        call_count = 0

        def make_storage(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            s = MagicMock()
            s.open = AsyncMock()
            s._db = MagicMock()  # engine is "open"
            return s

        mock_cls = MagicMock(side_effect=make_storage)

        from neural_memory.unified_config import _storage_cache

        _storage_cache.pop("inf:brain-a", None)
        _storage_cache.pop("inf:brain-b", None)

        with patch("neural_memory.pro.storage_adapter.InfinityDBStorage", mock_cls):
            storage_a = await _get_infinitydb_storage(cfg, "brain-a")
            storage_b = await _get_infinitydb_storage(cfg, "brain-b")

            # Two different instances created
            assert call_count == 2
            assert storage_a is not storage_b

            # Cached — no new instance
            storage_a2 = await _get_infinitydb_storage(cfg, "brain-a")
            assert call_count == 2
            assert storage_a2 is storage_a

        # Cleanup
        _storage_cache.pop("inf:brain-a", None)
        _storage_cache.pop("inf:brain-b", None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _pro_deps_available, reason="Pro deps (numpy/hnswlib) not installed")
    async def test_migration_hint_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log migration hint when SQLite exists but InfinityDB doesn't."""
        cfg = _make_config(tmp_path)
        brains_dir = tmp_path / "brains"
        brains_dir.mkdir(parents=True)
        # Create SQLite file
        (brains_dir / "my-brain.db").write_bytes(b"SQLite data")

        mock_storage = MagicMock()
        mock_storage.return_value.open = AsyncMock()
        mock_storage.return_value._db = MagicMock()

        from neural_memory.unified_config import _storage_cache

        _storage_cache.pop("inf:my-brain", None)

        import logging

        with (
            patch("neural_memory.pro.storage_adapter.InfinityDBStorage", mock_storage),
            caplog.at_level(logging.INFO),
        ):
            await _get_infinitydb_storage(cfg, "my-brain")

        assert any("migrate" in r.message.lower() for r in caplog.records)

        _storage_cache.pop("inf:my-brain", None)
