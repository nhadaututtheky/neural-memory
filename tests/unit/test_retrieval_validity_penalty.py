"""Tests for the validity-window penalty step in the retrieval pipeline.

Covers `_apply_validity_penalty` directly. Out-of-window neurons get a
0.1x score multiplier (not dropped) so callers can still surface history
via an explicit `as_of` time-travel query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.activation import ActivationResult


def _make_neuron(neuron_id: str, **md: Any) -> Neuron:
    return Neuron(id=neuron_id, type=NeuronType.CONCEPT, content=neuron_id, metadata=dict(md))


def _make_activation(neuron_id: str, level: float = 0.9) -> ActivationResult:
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=0,
        path=[neuron_id],
        source_anchor=neuron_id,
    )


class _FakeEngine:
    """Lightweight harness exposing `_apply_validity_penalty`."""

    def __init__(self, neurons: dict[str, Neuron], threshold: float = 0.05) -> None:
        self._storage = AsyncMock()
        self._storage.get_neurons_batch = AsyncMock(return_value=neurons)
        self._config = BrainConfig(activation_threshold=threshold)

    from neural_memory.engine.retrieval import ReflexPipeline

    _apply_validity_penalty = ReflexPipeline._apply_validity_penalty


@pytest.mark.asyncio
async def test_no_validity_metadata_no_change() -> None:
    n = _make_neuron("n1")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=None)
    assert out["n1"].activation_level == 0.9


@pytest.mark.asyncio
async def test_within_window_no_penalty() -> None:
    n = _make_neuron("n1", _valid_from="2026-01-01", _valid_until="2026-12-31")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=datetime(2026, 6, 15))
    assert out["n1"].activation_level == 0.9


@pytest.mark.asyncio
async def test_after_valid_until_gets_penalty() -> None:
    n = _make_neuron("n1", _valid_until="2026-06-01T00:00:00")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=datetime(2026, 12, 1))
    assert "n1" in out
    assert out["n1"].activation_level == pytest.approx(0.09, rel=0.01)


@pytest.mark.asyncio
async def test_before_valid_from_gets_penalty() -> None:
    n = _make_neuron("n1", _valid_from="2026-06-01T00:00:00")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=datetime(2026, 1, 1))
    assert out["n1"].activation_level == pytest.approx(0.09, rel=0.01)


@pytest.mark.asyncio
async def test_below_threshold_pruned() -> None:
    """Penalty that drops below activation_threshold removes the neuron."""
    n = _make_neuron("n1", _valid_until="2026-01-01T00:00:00")
    engine = _FakeEngine({"n1": n}, threshold=0.5)
    activations = {"n1": _make_activation("n1", 0.4)}
    out = await engine._apply_validity_penalty(activations, as_of=datetime(2026, 12, 1))
    assert "n1" not in out


@pytest.mark.asyncio
async def test_as_of_time_travel_keeps_old_memory() -> None:
    """Querying `as_of` a moment inside the window keeps full activation."""
    n = _make_neuron("n1", _valid_until="2026-06-01T00:00:00")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=datetime(2026, 3, 1))
    assert out["n1"].activation_level == 0.9


@pytest.mark.asyncio
async def test_default_as_of_none_uses_utcnow() -> None:
    """When `as_of=None`, current time is used to evaluate validity."""
    # Pick a far-past valid_until so penalty must apply at "now".
    n = _make_neuron("n1", _valid_until="2020-01-01T00:00:00")
    engine = _FakeEngine({"n1": n})
    activations = {"n1": _make_activation("n1", 0.9)}
    out = await engine._apply_validity_penalty(activations, as_of=None)
    assert out["n1"].activation_level == pytest.approx(0.09, rel=0.01)


@pytest.mark.asyncio
async def test_empty_activations_short_circuits() -> None:
    engine = _FakeEngine({})
    out = await engine._apply_validity_penalty({}, as_of=None)
    assert out == {}
    engine._storage.get_neurons_batch.assert_not_called()


@pytest.mark.asyncio
async def test_missing_neuron_kept_unchanged() -> None:
    engine = _FakeEngine({})  # storage returns nothing
    activations = {"n_missing": _make_activation("n_missing", 0.7)}
    out = await engine._apply_validity_penalty(activations, as_of=None)
    assert out["n_missing"].activation_level == 0.7
