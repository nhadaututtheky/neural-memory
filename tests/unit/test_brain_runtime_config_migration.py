"""Tests for _migrate_brain_runtime_config (issue #168).

When a brain was created on an older version, its stored `BrainConfig` JSON
is missing fields added later (e.g. `bm25_enabled`, `high_signal_memory_boost`).
On the next storage open, `config.toml [brain]` extras must be layered on top
and the patched brain saved back — otherwise users have to backfill via raw
SQL after every upgrade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.unified_config import (
    BrainSettings,
    EmbeddingSettings,
    UnifiedConfig,
    _migrate_brain_runtime_config,
)


@pytest.mark.asyncio
async def test_migration_applies_extras_to_stored_brain(tmp_path: Path) -> None:
    """Extras from config.toml override stored BrainConfig defaults."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    try:
        # Brain created with defaults (legacy state — bm25 disabled, etc.)
        legacy_brain = Brain.create(name="legacy", config=BrainConfig(), brain_id="legacy")
        await storage.save_brain(legacy_brain)
        storage.set_brain(legacy_brain.id)

        # User edits config.toml — flips bm25 on, raises boost
        config = UnifiedConfig(data_dir=tmp_path)
        config.brain = BrainSettings.from_dict(
            {
                "bm25_enabled": True,
                "bm25_tokenizer": "vietnamese",
                "high_signal_memory_boost": 1.5,
            }
        )

        # Re-fetch then migrate
        loaded = await storage.get_brain("legacy")
        assert loaded is not None
        assert loaded.config.bm25_enabled is False  # legacy default

        await _migrate_brain_runtime_config(storage, loaded, config)

        migrated = await storage.get_brain("legacy")
        assert migrated is not None
        assert migrated.config.bm25_enabled is True
        assert migrated.config.bm25_tokenizer == "vietnamese"
        assert migrated.config.high_signal_memory_boost == 1.5
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_migration_noop_when_no_extras(tmp_path: Path) -> None:
    """Empty [brain] section in config.toml leaves brain untouched."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    try:
        original = Brain.create(name="b", config=BrainConfig(), brain_id="b")
        await storage.save_brain(original)
        storage.set_brain(original.id)

        config = UnifiedConfig(data_dir=tmp_path)
        # No extras — runtime_overrides() returns {}
        assert config.brain.runtime_overrides() == {}

        loaded = await storage.get_brain("b")
        assert loaded is not None
        original_updated_at = loaded.updated_at

        await _migrate_brain_runtime_config(storage, loaded, config)

        # No write happened — updated_at unchanged
        after = await storage.get_brain("b")
        assert after is not None
        assert after.updated_at == original_updated_at
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_migration_skips_explicit_brainsettings_fields(
    tmp_path: Path,
) -> None:
    """The 7 explicit BrainSettings fields are NOT auto-overridden.

    Legacy brains may carry per-brain customization on these fields; we only
    propagate the newer extras keys via the upgrade path.
    """
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    try:
        # Brain has a per-brain decay_rate
        custom = Brain.create(
            name="custom",
            config=BrainConfig(decay_rate=0.99, max_spread_hops=2),
            brain_id="custom",
        )
        await storage.save_brain(custom)
        storage.set_brain(custom.id)

        # config.toml sets a different decay_rate but the migration should
        # leave the brain's per-brain customization intact.
        config = UnifiedConfig(data_dir=tmp_path)
        config.brain = BrainSettings.from_dict({"decay_rate": 0.05, "bm25_enabled": True})

        loaded = await storage.get_brain("custom")
        assert loaded is not None
        await _migrate_brain_runtime_config(storage, loaded, config)

        after = await storage.get_brain("custom")
        assert after is not None
        # decay_rate (explicit) unchanged, bm25_enabled (extras) applied
        assert after.config.decay_rate == 0.99
        assert after.config.max_spread_hops == 2
        assert after.config.bm25_enabled is True
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_migration_safe_on_save_failure(tmp_path: Path) -> None:
    """Storage write failures are swallowed — recall must never break."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    try:
        brain = Brain.create(name="b", config=BrainConfig(), brain_id="b")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Close storage to force save_brain() failures
        await storage.close()

        config = UnifiedConfig(data_dir=tmp_path)
        config.brain = BrainSettings.from_dict({"bm25_enabled": True})

        # Should not raise even though storage is closed
        await _migrate_brain_runtime_config(storage, brain, config)
    finally:
        try:
            await storage.close()
        except Exception:
            pass


class TestNewBrainCreationUsesFullKwargs:
    """When new brain created via _get_sqlite_storage, BrainConfig must
    include extras from config.toml — not just the historical 7 fields."""

    def test_full_kwargs_includes_extras(self) -> None:
        """Smoke test the helper used by storage open."""
        settings = BrainSettings.from_dict(
            {
                "decay_rate": 0.2,
                "bm25_enabled": True,
                "high_signal_memory_boost": 1.5,
                "creation_recency_boost": 0.1,
            }
        )
        embedding = EmbeddingSettings(
            enabled=True,
            provider="sentence_transformer",
            model="all-MiniLM-L6-v2",
            similarity_threshold=0.7,
        )
        config = BrainConfig(**settings.to_brain_config_kwargs(embedding))
        # 7 explicit
        assert config.decay_rate == 0.2
        # extras
        assert config.bm25_enabled is True
        assert config.high_signal_memory_boost == 1.5
        assert config.creation_recency_boost == 0.1
        # embedding-derived
        assert config.embedding_enabled is True
        assert config.embedding_provider == "sentence_transformer"
