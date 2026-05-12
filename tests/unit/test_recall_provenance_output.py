"""Integration test: provenance footer surfaces through format_context.

Pins the contract that BudgetConfig.show_provenance gates the per-neuron
`[src=… · YYYY-MM-DD · conf=…]` footer in the "## Related Information"
section of recall output.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.token_budget import BudgetConfig
from neural_memory.utils.timeutils import utcnow


def _make_neuron(neuron_id: str, content: str, **md: object) -> Neuron:
    return Neuron(
        id=neuron_id,
        type=NeuronType.CONCEPT,
        content=content,
        metadata=dict(md),
        created_at=utcnow() - timedelta(days=1),
    )


def _make_activation(neuron_id: str, level: float = 0.9) -> ActivationResult:
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=0,
        path=[neuron_id],
        source_anchor=neuron_id,
    )


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_provenance_appears_when_flag_on(mock_storage: AsyncMock) -> None:
    from neural_memory.engine.retrieval_context import format_context_budgeted

    n = _make_neuron("n1", "alpha beta gamma", source="workflow", _confidence=0.85)
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})

    activations = {"n1": _make_activation("n1")}
    cfg = BudgetConfig(show_provenance=True)
    context, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
        budget_config=cfg,
    )
    assert "[src=workflow" in context
    assert "conf=0.85" in context


@pytest.mark.asyncio
async def test_provenance_hidden_when_flag_off(mock_storage: AsyncMock) -> None:
    from neural_memory.engine.retrieval_context import format_context_budgeted

    n = _make_neuron("n1", "alpha beta gamma", source="workflow", _confidence=0.85)
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})

    activations = {"n1": _make_activation("n1")}
    cfg = BudgetConfig(show_provenance=False)
    context, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
        budget_config=cfg,
    )
    assert "[src=" not in context
    assert "conf=" not in context


@pytest.mark.asyncio
async def test_provenance_default_is_on(mock_storage: AsyncMock) -> None:
    """Default BudgetConfig() emits provenance — agents get trust signals out of the box."""
    from neural_memory.engine.retrieval_context import format_context

    n = _make_neuron("n1", "alpha beta", source="auto-capture")
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})

    activations = {"n1": _make_activation("n1")}
    context, _ = await format_context(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
    )
    assert "[src=auto-capture" in context


@pytest.mark.asyncio
async def test_provenance_works_in_clean_for_prompt_mode(mock_storage: AsyncMock) -> None:
    """clean_for_prompt strips headers/type-tags but provenance is still useful for agents."""
    from neural_memory.engine.retrieval_context import format_context_budgeted

    n = _make_neuron("n1", "delta epsilon", source="workflow")
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})

    activations = {"n1": _make_activation("n1")}
    cfg = BudgetConfig(show_provenance=True)
    context, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
        budget_config=cfg,
        clean_for_prompt=True,
    )
    # No [CONCEPT] type tag, no headers, but provenance footer present.
    assert "[concept]" not in context.lower()
    assert "## Related Information" not in context
    assert "[src=workflow" in context


@pytest.mark.asyncio
async def test_provenance_token_overhead_under_8_percent_for_realistic_neurons(
    mock_storage: AsyncMock,
) -> None:
    """Plan acceptance criterion: < 8% overhead for standard recall.

    The 8% target is meaningful for realistic atomic-fact neurons (50-150
    tokens each), not for 6-word toy neurons. The footer is fixed-size
    (~10 tokens), so overhead = footer_tokens / content_tokens — small
    when content is properly sized, large for tiny test fixtures.
    """
    from neural_memory.engine.retrieval_context import (
        _estimate_tokens,
        format_context_budgeted,
    )

    # Realistic memory-sized content: 80-100 tokens each (sentence-level facts).
    realistic_content = (
        "The Reflex Arc subsystem watches for SimHash collisions across pinned "
        "neurons; on conflict it unpins the older entry, flips its lifecycle "
        "status to superseded, records the winner via metadata, and emits a "
        "SUPERSEDES synapse so audit reconstructions can replay the swap. "
        "Without the explicit status flip, recall would still surface the "
        "outdated rule via decay only, which is too gradual to override "
        "high-priority writes."
    )
    neurons = {
        f"n{i}": _make_neuron(f"n{i}", realistic_content, _source="workflow") for i in range(10)
    }
    mock_storage.get_neurons_batch = AsyncMock(return_value=neurons)
    activations = {nid: _make_activation(nid, 0.9 - i * 0.05) for i, nid in enumerate(neurons)}

    base_cfg = BudgetConfig(show_provenance=False)
    base_ctx, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=5000,
        budget_config=base_cfg,
    )

    enriched_cfg = BudgetConfig(show_provenance=True)
    enriched_ctx, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=5000,
        budget_config=enriched_cfg,
    )

    base_tokens = _estimate_tokens(base_ctx)
    enriched_tokens = _estimate_tokens(enriched_ctx)
    overhead = (enriched_tokens - base_tokens) / max(base_tokens, 1)
    assert overhead < 0.08, f"Provenance overhead too high: {overhead:.2%} (spec: <8%)"


@pytest.mark.asyncio
async def test_provenance_overhead_bounded_even_for_tiny_neurons(
    mock_storage: AsyncMock,
) -> None:
    """For abnormally short neurons, overhead is high in % but bounded in absolute tokens.

    The footer is ~10 tokens regardless of neuron size, so 6-word neurons
    will see >100% overhead — that's mathematically expected and still
    acceptable because absolute token cost stays small.
    """
    from neural_memory.engine.retrieval_context import (
        _estimate_tokens,
        format_context_budgeted,
    )

    neurons = {
        f"n{i}": _make_neuron(f"n{i}", f"tiny {i} alpha beta", _source="x") for i in range(10)
    }
    mock_storage.get_neurons_batch = AsyncMock(return_value=neurons)
    activations = {nid: _make_activation(nid, 0.9 - i * 0.05) for i, nid in enumerate(neurons)}

    enriched_cfg = BudgetConfig(show_provenance=True)
    enriched_ctx, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=5000,
        budget_config=enriched_cfg,
    )
    # Bound absolute cost: 10 footers, ~14 tokens each = 140 token cap.
    assert _estimate_tokens(enriched_ctx) < 500


@pytest.mark.asyncio
async def test_bare_format_context_call_emits_provenance_by_default(
    mock_storage: AsyncMock,
) -> None:
    """Pin the silent-behavior-change contract from review feedback.

    `format_context()` is called from `engine/retrieval.py` without
    `_budget_config` (ReflexPipeline + familiarity paths). The internal
    default-config materialization activates `show_provenance=True`.
    Document the contract here so future callers are not surprised.
    """
    from neural_memory.engine.retrieval_context import format_context

    n = _make_neuron("n1", "alpha beta gamma delta", _source="bare-caller")
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})
    activations = {"n1": _make_activation("n1")}

    context, _ = await format_context(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
    )
    assert "[src=bare-caller" in context, (
        "Bare format_context() must default-emit provenance — "
        "ReflexPipeline / familiarity callers rely on this contract"
    )


@pytest.mark.asyncio
async def test_provenance_handles_missing_metadata(mock_storage: AsyncMock) -> None:
    """Neurons saved before provenance tracking → graceful 'manual' fallback."""
    from neural_memory.engine.retrieval_context import format_context_budgeted

    n = Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="legacy content",
        metadata={},
        created_at=datetime(2024, 1, 15, 12, 0, 0),
    )
    mock_storage.get_neurons_batch = AsyncMock(return_value={"n1": n})

    activations = {"n1": _make_activation("n1")}
    cfg = BudgetConfig(show_provenance=True)
    context, _, _ = await format_context_budgeted(
        storage=mock_storage,
        activations=activations,
        fibers=[],
        max_tokens=2000,
        budget_config=cfg,
    )
    assert "[src=manual" in context
    assert "2024-01-15" in context
