"""Tests for BM25 candidate fusion in the retrieval pipeline.

Item #1 from plan-tllr-learnings: covers the `_bm25_anchors` helper
and the `_get_lexical_index` lazy-build path. The full `query()`
integration is exercised by existing retrieval tests; this module pins
the contract that BM25 is OFF by default and produces well-formed
RankedAnchor lists when ON.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.score_fusion import RankedAnchor


def _make_neuron(neuron_id: str, content: str) -> Neuron:
    return Neuron(id=neuron_id, type=NeuronType.CONCEPT, content=content)


class _FakeEngine:
    """Lightweight harness exposing the BM25 helper methods directly."""

    def __init__(self, neurons: list[Neuron], *, bm25_enabled: bool = True) -> None:
        import asyncio

        self._storage = AsyncMock()
        self._storage.find_neurons = AsyncMock(return_value=neurons)
        # Ghost-neuron defense (Item #1 C1): validate hits exist via
        # get_neurons_batch before returning. Default fixture mirrors the
        # full neuron set so existing tests continue to pass.
        self._storage.get_neurons_batch = AsyncMock(return_value={n.id: n for n in neurons})
        self._config = BrainConfig(bm25_enabled=bm25_enabled, bm25_limit=10)
        self._lexical_index: Any = None
        self._lexical_index_lock = asyncio.Lock()

    from neural_memory.engine.retrieval import ReflexPipeline

    _bm25_anchors = ReflexPipeline._bm25_anchors
    _get_lexical_index = ReflexPipeline._get_lexical_index
    _invalidate_lexical_index = ReflexPipeline._invalidate_lexical_index


async def _make_engine(
    neurons: list[Neuron],
    *,
    bm25_enabled: bool = True,
    bm25_limit: int = 10,
) -> _FakeEngine:
    """Default engine + storage that returns the same neurons via both
    `find_neurons` (for index build) AND `get_neurons_batch` (for
    Item #1 review C1 ghost-neuron validation)."""
    engine = _FakeEngine(neurons, bm25_enabled=bm25_enabled)
    engine._config = BrainConfig(bm25_enabled=bm25_enabled, bm25_limit=bm25_limit)
    engine._storage.get_neurons_batch = AsyncMock(return_value={n.id: n for n in neurons})
    return engine


@pytest.mark.asyncio
async def test_bm25_anchors_return_ranked_list() -> None:
    neurons = [
        _make_neuron("n1", "neural memory provenance footer"),
        _make_neuron("n2", "neural memory unrelated content"),
        _make_neuron("n3", "completely separate topic"),
    ]
    engine = await _make_engine(neurons)
    anchors = await engine._bm25_anchors("provenance footer")

    assert anchors
    assert anchors[0].neuron_id == "n1"
    assert all(isinstance(a, RankedAnchor) for a in anchors)
    assert all(a.retriever == "bm25" for a in anchors)
    # Ranks are 1-indexed and ascending.
    assert [a.rank for a in anchors] == list(range(1, len(anchors) + 1))


@pytest.mark.asyncio
async def test_bm25_anchors_respects_limit() -> None:
    neurons = [_make_neuron(f"n{i}", f"shared term content {i}") for i in range(20)]
    engine = await _make_engine(neurons)
    engine._config = BrainConfig(bm25_enabled=True, bm25_limit=5)
    anchors = await engine._bm25_anchors("shared")
    assert len(anchors) == 5


@pytest.mark.asyncio
async def test_bm25_anchors_filters_exclude_ids() -> None:
    neurons = [
        _make_neuron("n1", "alpha beta"),
        _make_neuron("n2", "alpha gamma"),
    ]
    engine = await _make_engine(neurons)
    anchors = await engine._bm25_anchors("alpha", exclude_ids={"n1"})
    assert all(a.neuron_id != "n1" for a in anchors)


@pytest.mark.asyncio
async def test_bm25_anchors_empty_query_returns_empty() -> None:
    neurons = [_make_neuron("n1", "alpha beta")]
    engine = await _make_engine(neurons)
    assert await engine._bm25_anchors("") == []
    assert await engine._bm25_anchors("!!! ,,,") == []


@pytest.mark.asyncio
async def test_bm25_anchors_no_neurons_returns_empty() -> None:
    engine = _FakeEngine([])
    assert await engine._bm25_anchors("anything") == []


@pytest.mark.asyncio
async def test_lexical_index_lazy_built_once() -> None:
    """First call builds; second reuses the cache (one storage scan total)."""
    neurons = [_make_neuron("n1", "alpha beta gamma")]
    engine = await _make_engine(neurons)

    index1 = await engine._get_lexical_index()
    index2 = await engine._get_lexical_index()
    assert index1 is index2
    engine._storage.find_neurons.assert_called_once()


@pytest.mark.asyncio
async def test_lexical_index_invalidate_triggers_rebuild() -> None:
    neurons = [_make_neuron("n1", "alpha")]
    engine = await _make_engine(neurons)

    await engine._get_lexical_index()
    engine._invalidate_lexical_index()
    await engine._get_lexical_index()
    assert engine._storage.find_neurons.call_count == 2


@pytest.mark.asyncio
async def test_lexical_index_handles_storage_failure_gracefully() -> None:
    """A storage error must NOT propagate — BM25 is opt-in candidate source."""
    engine = _FakeEngine([])
    engine._storage.find_neurons = AsyncMock(side_effect=RuntimeError("storage down"))

    index = await engine._get_lexical_index()
    assert index is None
    # Recall must still work; BM25 just contributes no anchors.
    anchors = await engine._bm25_anchors("anything")
    assert anchors == []


@pytest.mark.asyncio
async def test_bm25_skips_neurons_without_content() -> None:
    neurons = [
        _make_neuron("n1", "alpha beta"),
        _make_neuron("n2", ""),  # empty content
    ]
    engine = await _make_engine(neurons)
    anchors = await engine._bm25_anchors("alpha")
    assert all(a.neuron_id != "n2" for a in anchors)


@pytest.mark.asyncio
async def test_ghost_neuron_dropped_when_storage_lacks_record() -> None:
    """Item #1 review C1: a BM25 hit whose backing neuron was deleted is dropped."""
    neurons = [
        _make_neuron("alive", "alpha beta gamma"),
        _make_neuron("ghost", "alpha beta gamma"),
    ]
    engine = _FakeEngine(neurons)
    # Simulate ghost: get_neurons_batch returns only the live neuron.
    engine._storage.get_neurons_batch = AsyncMock(return_value={"alive": neurons[0]})
    anchors = await engine._bm25_anchors("alpha")
    assert all(a.neuron_id != "ghost" for a in anchors)
    assert any(a.neuron_id == "alive" for a in anchors)


@pytest.mark.asyncio
async def test_bm25_limit_capped_at_max() -> None:
    """Item #1 review C2: a misconfigured `bm25_limit=10000` must not produce 10000 results."""
    from neural_memory.engine.lexical_index import MAX_BM25_LIMIT

    neurons = [_make_neuron(f"n{i}", f"shared {i}") for i in range(MAX_BM25_LIMIT + 50)]
    engine = _FakeEngine(neurons)
    engine._config = BrainConfig(bm25_enabled=True, bm25_limit=10000)
    anchors = await engine._bm25_anchors("shared")
    assert len(anchors) <= MAX_BM25_LIMIT


@pytest.mark.asyncio
async def test_lexical_index_lock_serializes_concurrent_first_query() -> None:
    """Item #1 review H3: concurrent first-queries must not each scan the corpus."""
    import asyncio

    neurons = [_make_neuron("n1", "alpha")]
    engine = _FakeEngine(neurons)

    # Two coroutines both invoke `_get_lexical_index` simultaneously. Lock
    # must serialize them so storage scan happens exactly once.
    indexes = await asyncio.gather(
        engine._get_lexical_index(),
        engine._get_lexical_index(),
        engine._get_lexical_index(),
    )
    assert all(idx is not None for idx in indexes)
    assert all(idx is indexes[0] for idx in indexes)
    engine._storage.find_neurons.assert_called_once()


@pytest.mark.asyncio
async def test_bm25_score_descending() -> None:
    neurons = [
        _make_neuron("low", "rare term once"),
        _make_neuron("high", "rare term rare term rare term"),
    ]
    engine = await _make_engine(neurons)
    anchors = await engine._bm25_anchors("rare term")
    assert anchors[0].neuron_id == "high"
    assert anchors[0].score >= anchors[1].score
