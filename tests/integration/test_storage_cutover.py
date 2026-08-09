"""P3-T5: new-install unified cutover + upgrade compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.brain_mode import BrainModeConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.factory import create_storage
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.unified_config import UnifiedConfig


def test_fresh_install_config_selects_unified(tmp_path: Path) -> None:
    cfg = UnifiedConfig.load(tmp_path / "config.toml")
    assert cfg.storage_adapter == "unified"
    assert 'storage_adapter = "unified"' in (tmp_path / "config.toml").read_text(encoding="utf-8")


def test_upgrade_missing_key_stays_legacy(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'version = "1.0"\ncurrent_brain = "default"\nstorage_backend = "sqlite"\n',
        encoding="utf-8",
    )
    cfg = UnifiedConfig.load(path)
    assert cfg.storage_adapter == "legacy"


@pytest.mark.asyncio
async def test_create_storage_follows_config_adapter(tmp_path: Path) -> None:
    unified = await create_storage(
        BrainModeConfig.local(storage_adapter="unified"),
        "cutover",
        local_path=str(tmp_path / "u.db"),
    )
    try:
        assert isinstance(unified, SQLStorage)
    finally:
        await unified.close()

    legacy = await create_storage(
        BrainModeConfig.local(storage_adapter="legacy"),
        "cutover",
        local_path=str(tmp_path / "l.db"),
    )
    try:
        assert isinstance(legacy, SQLiteStorage)
    finally:
        await legacy.close()


@pytest.mark.asyncio
async def test_fresh_shared_storage_path_uses_unified(tmp_path: Path) -> None:
    import neural_memory.unified_config as uc

    cfg = UnifiedConfig(data_dir=tmp_path, current_brain="pilot", storage_adapter="unified")
    cfg.save()
    uc._storage_cache.clear()
    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        # Patch get_config to our temp config
        mp.setattr(uc, "get_config", lambda: cfg)
        mp.setattr(uc, "_read_current_brain_from_toml", lambda *a, **k: None)
        storage = await uc.get_shared_storage("pilot")
        try:
            assert isinstance(storage, SQLStorage)
            brain = Brain.create(name="pilot", brain_id="pilot")
            await storage.save_brain(brain)
            storage.set_brain("pilot")
            await storage.add_neuron(
                Neuron.create(type=NeuronType.CONCEPT, content="cutover-neuron")
            )
            found = await storage.find_neurons(content_exact="cutover-neuron")
            assert len(found) == 1
        finally:
            await storage.close()
            uc._storage_cache.clear()


@pytest.mark.asyncio
async def test_brain_switch_round_trip(tmp_path: Path) -> None:
    store = SQLStorage(
        __import__(
            "neural_memory.storage.sql.sqlite_dialect", fromlist=["SQLiteDialect"]
        ).SQLiteDialect(tmp_path / "switch.db")
    )
    await store.initialize()
    try:
        a = Brain.create(name="a", brain_id="a")
        b = Brain.create(name="b", brain_id="b")
        await store.save_brain(a)
        await store.save_brain(b)
        store.set_brain("a")
        await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="only-a"))
        store.set_brain("b")
        found_b = await store.find_neurons(content_exact="only-a")
        assert found_b == []
        store.set_brain("a")
        found_a = await store.find_neurons(content_exact="only-a")
        assert len(found_a) == 1
    finally:
        await store.close()
