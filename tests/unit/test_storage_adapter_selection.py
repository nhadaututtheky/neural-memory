"""Tests for the reversible legacy/unified SQLite adapter pilot."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.core.brain_mode import BrainModeConfig
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.unified_config import UnifiedConfig


def test_unified_config_defaults_to_unified_adapter(tmp_path: Path) -> None:
    """New installs / bare UnifiedConfig default to the unified SQL adapter."""
    config = UnifiedConfig(data_dir=tmp_path)

    assert config.storage_backend == "sqlite"
    assert config.storage_adapter == "unified"


def test_fresh_config_file_writes_unified_adapter(tmp_path: Path) -> None:
    loaded = UnifiedConfig.load(tmp_path / "config.toml")
    assert loaded.storage_adapter == "unified"
    text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert 'storage_adapter = "unified"' in text


def test_existing_config_missing_adapter_key_stays_legacy(tmp_path: Path) -> None:
    """Upgrade path: configs written before storage_adapter stay on legacy."""
    path = tmp_path / "config.toml"
    path.write_text(
        'version = "1.0"\ncurrent_brain = "default"\nstorage_backend = "sqlite"\n',
        encoding="utf-8",
    )
    loaded = UnifiedConfig.load(path)
    assert loaded.storage_adapter == "legacy"


def test_unified_config_adapter_round_trip(tmp_path: Path) -> None:
    config = UnifiedConfig(
        data_dir=tmp_path,
        current_brain="pilot",
        storage_adapter="unified",
    )

    config.save()
    loaded = UnifiedConfig.load(tmp_path / "config.toml")

    assert loaded.storage_backend == "sqlite"
    assert loaded.storage_adapter == "unified"
    assert 'storage_adapter = "unified"' in (tmp_path / "config.toml").read_text(encoding="utf-8")


def test_invalid_persisted_adapter_warns_and_falls_back(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = UnifiedConfig(data_dir=tmp_path, current_brain="pilot")
    config.save()
    config_path = tmp_path / "config.toml"
    content = config_path.read_text(encoding="utf-8").replace(
        'storage_adapter = "unified"',
        'storage_adapter = "mystery"',
    )
    config_path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="neural_memory.core.brain_mode"):
        loaded = UnifiedConfig.load(config_path)

    assert loaded.storage_adapter == "legacy"
    assert any(
        "Unknown storage_adapter 'mystery'" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_local_factory_honors_unified_adapter(tmp_path: Path) -> None:
    from neural_memory.storage.factory import create_storage

    storage = await create_storage(
        BrainModeConfig.local(storage_adapter="unified"),
        "pilot",
        local_path=str(tmp_path / "pilot.db"),
    )
    try:
        assert isinstance(storage, SQLStorage)
        assert storage.brain_id == "pilot"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_hybrid_factory_honors_unified_adapter(tmp_path: Path) -> None:
    from neural_memory.storage.factory import HybridStorage, create_storage

    storage = await create_storage(
        BrainModeConfig.hybrid_mode(
            local_path=str(tmp_path / "hybrid.db"),
            server_url="http://localhost:8000",
            storage_adapter="unified",
        ),
        "pilot",
    )
    try:
        assert isinstance(storage, HybridStorage)
        assert isinstance(storage._local, SQLStorage)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_explicit_unified_init_failure_never_falls_back(tmp_path: Path) -> None:
    from neural_memory.storage.factory import create_storage

    with (
        patch(
            "neural_memory.storage.factory.SQLStorage.initialize",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unified init failed"),
        ),
        patch("neural_memory.storage.factory.SQLiteStorage") as legacy_storage,
        pytest.raises(RuntimeError, match="unified init failed"),
    ):
        await create_storage(
            BrainModeConfig.local(storage_adapter="unified"),
            "pilot",
            local_path=str(tmp_path / "pilot.db"),
        )

    legacy_storage.assert_not_called()


@pytest.mark.asyncio
async def test_shared_storage_cache_identity_includes_adapter(tmp_path: Path) -> None:
    import neural_memory.unified_config as unified_config

    config = UnifiedConfig(
        data_dir=tmp_path,
        current_brain="pilot",
        storage_adapter="legacy",
    )
    unified_config._storage_cache.clear()
    with (
        patch("neural_memory.unified_config.get_config", return_value=config),
        patch("neural_memory.unified_config._read_current_brain_from_toml", return_value=None),
    ):
        legacy = await unified_config.get_shared_storage("pilot")
        config.storage_adapter = "unified"
        unified = await unified_config.get_shared_storage("pilot")

    try:
        assert isinstance(legacy, SQLiteStorage)
        assert isinstance(unified, SQLStorage)
        assert legacy is not unified
        assert any(key.startswith("sqlite:legacy:") for key in unified_config._storage_cache)
        assert any(key.startswith("sqlite:unified:") for key in unified_config._storage_cache)
    finally:
        await legacy.close()
        await unified.close()
        unified_config._storage_cache.clear()
