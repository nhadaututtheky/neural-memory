"""Tests for the lifecycle status filter wired into the retrieval pipeline.

Covers `_filter_by_status` directly. The full `query()` integration is
exercised by existing retrieval tests; this module pins the contract that
non-active neurons are dropped by default and surfaced via
`include_status` overrides.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronStatus, NeuronType
from neural_memory.engine.activation import ActivationResult


def _make_neuron(neuron_id: str, status: NeuronStatus = NeuronStatus.ACTIVE) -> Neuron:
    md: dict[str, Any] = {}
    if status != NeuronStatus.ACTIVE:
        md["_status"] = status.value
    return Neuron(id=neuron_id, type=NeuronType.CONCEPT, content=neuron_id, metadata=md)


def _make_activation(neuron_id: str, level: float = 0.5) -> ActivationResult:
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=0,
        path=[neuron_id],
        source_anchor=neuron_id,
    )


class _FakeEngine:
    """Lightweight harness exposing `_filter_by_status` without booting the full RetrievalEngine."""

    def __init__(self, neurons: dict[str, Neuron]) -> None:
        self._storage = AsyncMock()
        self._storage.get_neurons_batch = AsyncMock(return_value=neurons)

    # Bind the real filter implementation as an unbound method.
    from neural_memory.engine.retrieval import ReflexPipeline

    _filter_by_status = ReflexPipeline._filter_by_status


@pytest.mark.asyncio
async def test_default_drops_superseded() -> None:
    neurons = {
        "n1": _make_neuron("n1", NeuronStatus.ACTIVE),
        "n2": _make_neuron("n2", NeuronStatus.SUPERSEDED),
    }
    activations = {
        "n1": _make_activation("n1"),
        "n2": _make_activation("n2"),
    }
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(activations)
    assert "n1" in kept
    assert "n2" not in kept
    assert dropped == ["n2"]


@pytest.mark.asyncio
async def test_default_drops_expired() -> None:
    neurons = {
        "n1": _make_neuron("n1", NeuronStatus.ACTIVE),
        "n2": _make_neuron("n2", NeuronStatus.EXPIRED),
    }
    activations = {nid: _make_activation(nid) for nid in neurons}
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(activations)
    assert "n2" not in kept
    assert dropped == ["n2"]


@pytest.mark.asyncio
async def test_include_status_surfaces_superseded() -> None:
    neurons = {
        "n1": _make_neuron("n1", NeuronStatus.ACTIVE),
        "n2": _make_neuron("n2", NeuronStatus.SUPERSEDED),
    }
    activations = {nid: _make_activation(nid) for nid in neurons}
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(
        activations, include_status=frozenset({"superseded"})
    )
    assert "n2" in kept
    assert "n1" not in kept  # active not in allowed → dropped
    assert dropped == ["n1"]


@pytest.mark.asyncio
async def test_include_status_active_and_superseded() -> None:
    neurons = {
        "n1": _make_neuron("n1", NeuronStatus.ACTIVE),
        "n2": _make_neuron("n2", NeuronStatus.SUPERSEDED),
        "n3": _make_neuron("n3", NeuronStatus.EXPIRED),
    }
    activations = {nid: _make_activation(nid) for nid in neurons}
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(
        activations, include_status=frozenset({"active", "superseded"})
    )
    assert set(kept.keys()) == {"n1", "n2"}
    assert dropped == ["n3"]


@pytest.mark.asyncio
async def test_legacy_superseded_flag_filtered_by_default() -> None:
    """Pre-Item#2 neurons with `_superseded` flag should also drop out."""
    legacy = Neuron(
        id="legacy",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={"_superseded": True},
    )
    neurons = {"n1": _make_neuron("n1"), "legacy": legacy}
    activations = {nid: _make_activation(nid) for nid in neurons}
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(activations)
    assert "legacy" not in kept
    assert dropped == ["legacy"]


@pytest.mark.asyncio
async def test_empty_activations_short_circuits() -> None:
    engine = _FakeEngine({})
    kept, dropped = await engine._filter_by_status({})
    assert kept == {}
    assert dropped == []
    engine._storage.get_neurons_batch.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_include_status_raises_value_error() -> None:
    """Item #2 review M3: typos at the boundary fail loudly, not silently drop everything."""
    engine = _FakeEngine({"n1": _make_neuron("n1")})
    activations = {"n1": _make_activation("n1")}
    with pytest.raises(ValueError, match="Invalid include_status"):
        await engine._filter_by_status(activations, include_status=frozenset({"actve"}))


@pytest.mark.asyncio
async def test_empty_frozenset_treated_as_default() -> None:
    """Item #2 review M4: empty frozenset is a probable caller bug — fall back to default."""
    neurons = {
        "n1": _make_neuron("n1", NeuronStatus.ACTIVE),
        "n2": _make_neuron("n2", NeuronStatus.SUPERSEDED),
    }
    activations = {nid: _make_activation(nid) for nid in neurons}
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(activations, include_status=frozenset())
    assert "n1" in kept
    assert "n2" not in kept
    assert dropped == ["n2"]


@pytest.mark.asyncio
async def test_unknown_stored_status_logs_warning_and_falls_back(caplog: Any) -> None:
    """Item #2 review L1: corrupt `_status` value gets observable signal, not silent ACTIVE."""
    legacy = Neuron(
        id="legacy",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={"_status": "garbage-status-name"},
    )
    activations = {"legacy": _make_activation("legacy")}
    engine = _FakeEngine({"legacy": legacy})
    import logging

    with caplog.at_level(logging.WARNING, logger="neural_memory.engine.retrieval"):
        kept, _ = await engine._filter_by_status(activations)
    assert "legacy" in kept  # falls back to ACTIVE → kept by default
    assert any("Unknown _status" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_missing_neuron_kept_by_default() -> None:
    """If get_neurons_batch returns no record for an ID, keep it.

    Better to err on the side of including unknown data than silently
    dropping (matches the conservative pattern used by `_deprioritize_disputed`).
    """
    neurons = {"n1": _make_neuron("n1")}  # n_missing absent
    activations = {
        "n1": _make_activation("n1"),
        "n_missing": _make_activation("n_missing"),
    }
    engine = _FakeEngine(neurons)
    kept, dropped = await engine._filter_by_status(activations)
    assert "n_missing" in kept
    assert dropped == []
