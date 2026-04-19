"""Tests for v4.52.2: hub-aware graph density + PPR hub edge dampening.

Prior to v4.52.2:
- `get_graph_density()` counted ALL synapses including DREAM hub links
  (metadata `_hub=True`), inflating density and biasing auto-strategy
  selection toward PPR even when the *real* user graph is sparse.
- PPR pushed activation through hub edges at full weight, letting
  synthetic hub links dominate fiber ranking.

v4.52.2 adds:
- `get_graph_density(exclude_hubs=True)` filters `_hub=True` synapses
- `BrainConfig.hub_edge_dampening` (default 0.5) multiplies the effective
  weight of `_hub=True` edges during PPR push
- Retrieval's `_auto_select_strategy()` uses hub-excluded density
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.ppr_activation import PPRActivation
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest_asyncio.fixture
async def store_with_hubs(tmp_path: pathlib.Path) -> SQLiteStorage:
    """Brain with 4 neurons, 3 plain synapses + 2 DREAM hub synapses.

    Raw density = 5/4 = 1.25
    Hub-excluded density = 3/4 = 0.75
    """
    db_path = tmp_path / "hubs.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    brain = Brain.create(name="hub-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    for nid in ("a", "b", "c", "d"):
        await storage.add_neuron(Neuron.create(type=NeuronType.ENTITY, content=nid, neuron_id=nid))

    plain = [
        Synapse.create(source_id="a", target_id="b", type=SynapseType.RELATED_TO, weight=0.5),
        Synapse.create(source_id="b", target_id="c", type=SynapseType.RELATED_TO, weight=0.5),
        Synapse.create(source_id="c", target_id="d", type=SynapseType.RELATED_TO, weight=0.5),
    ]
    hub = [
        Synapse.create(
            source_id="a",
            target_id="d",
            type=SynapseType.RELATED_TO,
            weight=0.4,
            metadata={"_dream": True, "_hub": True},
        ),
        Synapse.create(
            source_id="b",
            target_id="d",
            type=SynapseType.RELATED_TO,
            weight=0.4,
            metadata={"_dream": True, "_hub": True},
        ),
    ]
    for s in plain + hub:
        await storage.add_synapse(s)

    return storage


class TestHubExcludedDensity:
    """get_graph_density(exclude_hubs=True) skips synapses with _hub metadata."""

    @pytest.mark.asyncio
    async def test_density_default_counts_all_synapses(
        self, store_with_hubs: SQLiteStorage
    ) -> None:
        density = await store_with_hubs.get_graph_density()
        # 5 synapses / 4 neurons
        assert density == pytest.approx(1.25)

    @pytest.mark.asyncio
    async def test_density_excluding_hubs(self, store_with_hubs: SQLiteStorage) -> None:
        density = await store_with_hubs.get_graph_density(exclude_hubs=True)
        # 3 plain synapses / 4 neurons
        assert density == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_density_excluding_hubs_lower_than_raw(
        self, store_with_hubs: SQLiteStorage
    ) -> None:
        raw = await store_with_hubs.get_graph_density()
        excluded = await store_with_hubs.get_graph_density(exclude_hubs=True)
        assert excluded < raw

    @pytest.mark.asyncio
    async def test_empty_brain_returns_zero(self, tmp_path: pathlib.Path) -> None:
        db_path = tmp_path / "empty.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        brain = Brain.create(name="empty", config=BrainConfig(), owner_id="test")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        assert await storage.get_graph_density(exclude_hubs=True) == 0.0


class TestAutoStrategyUsesHubExcluded:
    """retrieval._auto_select_strategy passes exclude_hubs=True to density."""

    @pytest.mark.asyncio
    async def test_auto_strategy_calls_with_exclude_hubs(self) -> None:
        from neural_memory.engine.retrieval import ReflexPipeline

        engine = object.__new__(ReflexPipeline)
        engine._storage = MagicMock()
        engine._storage.get_graph_density = AsyncMock(return_value=1.0)
        engine._config = BrainConfig()
        engine._ppr_activator = MagicMock()

        await engine._auto_select_strategy()
        # Must have been called with exclude_hubs=True so raw dream hub
        # density doesn't bias the strategy selector.
        call_kwargs = engine._storage.get_graph_density.call_args.kwargs
        assert call_kwargs.get("exclude_hubs") is True


class TestPPRHubDampening:
    """Hub synapses contribute less activation push than regular edges."""

    def _make_storage(
        self,
        graph: dict[str, list[tuple[str, float, bool]]],
    ) -> AsyncMock:
        """Graph entry: (target, weight, is_hub)."""
        storage = AsyncMock()

        async def get_synapses_for_neurons(
            neuron_ids: list[str], direction: str = "out"
        ) -> dict[str, list[Synapse]]:
            result: dict[str, list[Synapse]] = {}
            for nid in neuron_ids:
                edges = graph.get(nid, [])
                syns: list[Synapse] = []
                for tgt, w, is_hub in edges:
                    meta = {"_hub": True} if is_hub else {}
                    syns.append(
                        Synapse.create(
                            source_id=nid,
                            target_id=tgt,
                            type=SynapseType.RELATED_TO,
                            weight=w,
                            metadata=meta,
                        )
                    )
                result[nid] = syns
            return result

        storage.get_synapses_for_neurons = get_synapses_for_neurons
        return storage

    @pytest.mark.asyncio
    async def test_hub_edge_receives_less_push_than_plain_edge(self) -> None:
        """Two edges with equal weight — hub edge target gets less activation."""
        graph = {
            "seed": [("plain_tgt", 0.5, False), ("hub_tgt", 0.5, True)],
            "plain_tgt": [],
            "hub_tgt": [],
        }
        storage = self._make_storage(graph)
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.001,
            ppr_damping=0.15,
            ppr_iterations=20,
            hub_edge_dampening=0.5,
        )
        ppr = PPRActivation(storage, config)
        results, _trace = await ppr.activate(["seed"])

        assert "plain_tgt" in results
        assert "hub_tgt" in results
        # Plain target should have strictly more activation than hub target
        # since their base weights are equal but hub edge is dampened.
        assert results["plain_tgt"].activation_level > results["hub_tgt"].activation_level

    @pytest.mark.asyncio
    async def test_hub_dampening_disabled_when_factor_is_one(self) -> None:
        """hub_edge_dampening=1.0 → hub and plain edges behave identically."""
        graph = {
            "seed": [("plain_tgt", 0.5, False), ("hub_tgt", 0.5, True)],
            "plain_tgt": [],
            "hub_tgt": [],
        }
        storage = self._make_storage(graph)
        config = BrainConfig(
            activation_strategy="ppr",
            activation_threshold=0.001,
            ppr_damping=0.15,
            ppr_iterations=20,
            hub_edge_dampening=1.0,
        )
        ppr = PPRActivation(storage, config)
        results, _trace = await ppr.activate(["seed"])

        # With factor=1.0, both targets receive equal activation
        assert results["plain_tgt"].activation_level == pytest.approx(
            results["hub_tgt"].activation_level, rel=1e-6
        )

    @pytest.mark.asyncio
    async def test_brain_config_default_dampening(self) -> None:
        """Default hub_edge_dampening should be 0.5 (documented)."""
        config = BrainConfig()
        assert config.hub_edge_dampening == 0.5
