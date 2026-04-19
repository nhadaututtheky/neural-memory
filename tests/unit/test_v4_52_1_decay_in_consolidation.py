"""Tests for v4.52.1: DecayManager wired into consolidation Tier 0.

Prior to v4.52.1, neuron activation + synapse weight only decayed on the
scheduled 12h decay cycle, so consolidation runs between cycles left stale
activation in place. Now DECAY runs as Tier 0 of consolidation — before
PRUNE — so activation reflects actual recency when PRUNE considers
thresholds.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationStrategy,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


@pytest_asyncio.fixture
async def decay_storage() -> InMemoryStorage:
    """Storage with one neuron and one synapse — enough to exercise decay."""
    from dataclasses import replace as dc_replace

    store = InMemoryStorage()
    brain = Brain.create(name="decay_test", brain_id="decay-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    old_time = utcnow() - timedelta(days=10)
    old = Neuron.create(type=NeuronType.ENTITY, content="old memory", neuron_id="n-old")
    old = dc_replace(old, created_at=old_time)
    await store.add_neuron(old)

    partner = Neuron.create(type=NeuronType.ENTITY, content="partner", neuron_id="n-partner")
    await store.add_neuron(partner)

    s = Synapse.create(
        source_id="n-old",
        target_id="n-partner",
        type=SynapseType.RELATED_TO,
        weight=0.8,
        synapse_id="syn-old",
    )
    await store.add_synapse(s)

    return store


class TestDecayStrategyExists:
    """Structural checks — DECAY enum + tier ordering."""

    def test_decay_enum_exists(self) -> None:
        assert ConsolidationStrategy.DECAY.value == "decay"

    def test_decay_is_tier_zero(self) -> None:
        """DECAY must run before PRUNE — i.e. live in an earlier tier."""
        tiers = ConsolidationEngine.STRATEGY_TIERS
        decay_tier_idx: int | None = None
        prune_tier_idx: int | None = None
        for i, tier in enumerate(tiers):
            if ConsolidationStrategy.DECAY in tier:
                decay_tier_idx = i
            if ConsolidationStrategy.PRUNE in tier:
                prune_tier_idx = i
        assert decay_tier_idx is not None, "DECAY missing from any tier"
        assert prune_tier_idx is not None, "PRUNE missing from any tier"
        assert decay_tier_idx < prune_tier_idx, (
            "DECAY must be in an earlier tier than PRUNE so decayed "
            "activation can drop items below prune thresholds"
        )

    def test_decay_dispatch_registered(self) -> None:
        """ConsolidationEngine must have a _decay method attached."""
        assert hasattr(ConsolidationEngine, "_decay")


@pytest.mark.asyncio
class TestDecayInConsolidation:
    """Runtime behavior — decay actually runs during consolidation."""

    async def test_decay_pass_applies_to_old_neuron(self, decay_storage: InMemoryStorage) -> None:
        """Running consolidation with DECAY should drop old neuron activation."""
        engine = ConsolidationEngine(decay_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DECAY], dry_run=False)
        # Report should have decay stats under extra
        assert "decay" in report.extra
        decay_stats = report.extra["decay"]
        assert decay_stats["neurons_processed"] >= 1

    async def test_decay_runs_before_prune_in_all(self, decay_storage: InMemoryStorage) -> None:
        """Running ALL strategies should record decay in the extra dict."""
        engine = ConsolidationEngine(decay_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.ALL], dry_run=False)
        # If DECAY ran during ALL, the extra.decay key should exist
        assert "decay" in report.extra, (
            "DECAY must run when ALL is requested — missing from report.extra"
        )

    async def test_decay_dry_run_returns_stats(self, decay_storage: InMemoryStorage) -> None:
        """dry_run=True should still produce stats in extra."""
        engine = ConsolidationEngine(decay_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DECAY], dry_run=True)
        assert "decay" in report.extra
        assert report.dry_run is True

    async def test_decay_failure_is_non_fatal(
        self, decay_storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If DecayManager.apply_decay raises, consolidation should continue."""
        from neural_memory.engine import lifecycle

        async def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("synthetic decay failure")

        monkeypatch.setattr(lifecycle.DecayManager, "apply_decay", _boom)

        engine = ConsolidationEngine(decay_storage)
        # Should not raise — failure is swallowed with warning
        report = await engine.run(strategies=[ConsolidationStrategy.DECAY], dry_run=False)
        # Decay key absent because apply_decay raised before report update
        assert "decay" not in report.extra
