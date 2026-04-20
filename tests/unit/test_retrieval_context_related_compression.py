"""Tests for Related Information section compression in format_context().

Covers the new `_compress_related_neurons()` helper and its wiring through
`format_context_budgeted` → `format_context`.

Baseline measurement (plan-related-info-compression.md) found 85.9% of recall
context tokens go to the "## Related Information" section. This test suite
enforces the compression contract: age-tier compression, hard content cap,
SimHash dedup (intra-section + cross-section vs fibers).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.token_budget import BudgetConfig
from neural_memory.utils.timeutils import utcnow

# ─────────────────────── Helpers ───────────────────────


def _make_neuron(
    neuron_id: str,
    content: str,
    neuron_type: NeuronType = NeuronType.CONCEPT,
    age_days: float = 0.0,
) -> Neuron:
    created = utcnow() - timedelta(days=age_days)
    return Neuron(
        id=neuron_id,
        type=neuron_type,
        content=content,
        created_at=created,
    )


def _make_activation(neuron_id: str, level: float) -> ActivationResult:
    return ActivationResult(
        neuron_id=neuron_id,
        activation_level=level,
        hop_distance=0,
        path=[neuron_id],
        source_anchor=neuron_id,
    )


# ─────────────────────── _compress_related_neurons ───────────────────────


class TestCompressRelatedNeurons:
    """Direct tests for the helper."""

    def test_old_neuron_age_compressed(self) -> None:
        """A 100-day-old multi-sentence neuron should be truncated to 1 sentence."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        n = _make_neuron(
            "n1",
            "First sentence here. Second sentence next. Third sentence last.",
            age_days=100.0,
        )
        results = [_make_activation("n1", 0.9)]
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"n1": n},
            emitted_fiber_texts=[],
            config=BudgetConfig(),
            max_neurons=20,
        )
        assert len(kept) == 1
        _, compressed = kept[0]
        # Expect truncation to roughly the first sentence
        assert "First sentence" in compressed
        assert "Third sentence" not in compressed

    def test_long_content_is_capped(self) -> None:
        """Content exceeding per_neuron_max_tokens should be clipped."""
        from neural_memory.engine.retrieval_context import (
            _compress_related_neurons,
            _estimate_tokens,
        )

        long_content = " ".join(["word"] * 400)  # ~520 tokens
        n = _make_neuron("n1", long_content, age_days=0.5)
        results = [_make_activation("n1", 0.9)]
        cfg = BudgetConfig(related_neuron_max_tokens=100)
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"n1": n},
            emitted_fiber_texts=[],
            config=cfg,
            max_neurons=20,
        )
        assert len(kept) == 1
        _, compressed = kept[0]
        assert _estimate_tokens(compressed) <= 100 + 5  # small tolerance

    def test_time_neurons_skipped(self) -> None:
        """TIME neurons must still be dropped (matches format_context behavior)."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        n = _make_neuron("t1", "yesterday", neuron_type=NeuronType.TIME)
        results = [_make_activation("t1", 0.9)]
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"t1": n},
            emitted_fiber_texts=[],
            config=BudgetConfig(),
            max_neurons=20,
        )
        assert len(kept) == 0

    def test_cross_section_dedup_against_fiber(self) -> None:
        """Neuron with content near an already-emitted fiber is dropped."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        fiber_text = (
            "The context compiler runs SimHash dedup across fibers and merges "
            "near-duplicates before formatting the final recall output."
        )
        neuron_text = (
            "The context compiler runs SimHash dedup across fibers and merges "
            "near-duplicates before formatting the final recall output."
        )
        n = _make_neuron("n1", neuron_text)
        results = [_make_activation("n1", 0.9)]
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"n1": n},
            emitted_fiber_texts=[fiber_text],
            config=BudgetConfig(),
            max_neurons=20,
        )
        assert len(kept) == 0

    def test_intra_section_dedup(self) -> None:
        """Two near-duplicate neurons → keep only one (highest activation)."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        text = (
            "Predictive priming boosts recall activation for neurons tied to "
            "topics that frequently co-occur with the current query."
        )
        n1 = _make_neuron("n1", text)
        n2 = _make_neuron("n2", text)
        results = [_make_activation("n1", 0.9), _make_activation("n2", 0.5)]
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"n1": n1, "n2": n2},
            emitted_fiber_texts=[],
            config=BudgetConfig(),
            max_neurons=20,
        )
        assert len(kept) == 1
        assert kept[0][0].id == "n1"

    def test_max_neurons_honored(self) -> None:
        """max_neurons caps the output list."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        # Use distinct vocabulary so SimHash does not dedup them.
        vocab = [
            "quantum mechanics wavefunction collapse",
            "postgresql transaction isolation levels",
            "rust borrow checker lifetime annotations",
            "kubernetes pod disruption budget",
            "typescript discriminated union pattern",
            "webgl fragment shader uniform buffer",
            "neural graph spreading activation decay",
            "openapi schema code generation pipeline",
            "docker multi-stage build cache mount",
            "redis cluster hash slot migration",
        ]
        neurons = {f"n{i}": _make_neuron(f"n{i}", vocab[i]) for i in range(10)}
        results = [_make_activation(f"n{i}", 1.0 - i * 0.01) for i in range(10)]
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map=neurons,
            emitted_fiber_texts=[],
            config=BudgetConfig(),
            max_neurons=3,
        )
        assert len(kept) == 3

    def test_disabled_returns_all_uncompressed(self) -> None:
        """With enable_related_compression=False, no compression/dedup applied."""
        from neural_memory.engine.retrieval_context import _compress_related_neurons

        text = "Exact same text here for both neurons in test."
        n1 = _make_neuron("n1", text)
        n2 = _make_neuron("n2", text)
        results = [_make_activation("n1", 0.9), _make_activation("n2", 0.5)]
        cfg = BudgetConfig(enable_related_compression=False)
        kept = _compress_related_neurons(
            sorted_results=results,
            neuron_map={"n1": n1, "n2": n2},
            emitted_fiber_texts=[],
            config=cfg,
            max_neurons=20,
        )
        # Both kept, raw content preserved
        assert len(kept) == 2
        assert kept[0][1] == text
        assert kept[1][1] == text


# ─────────────────────── Integration: format_context_budgeted ───────────────────────


class TestFormatContextBudgetedWithRelatedCompression:
    """End-to-end: ensure new behavior reaches format_context via budget_config."""

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.get_neurons_batch = AsyncMock(return_value={})
        return storage

    def _make_fiber(self, fid: str, summary: str) -> Any:
        f = MagicMock()
        f.id = fid
        f.summary = summary
        f.salience = 0.0
        f.conductivity = 1.0
        f.neuron_ids = {fid}
        f.anchor_neuron_id = fid
        f.metadata = {}
        f.created_at = utcnow()
        return f

    @pytest.mark.asyncio
    async def test_long_neuron_in_related_is_capped(self, mock_storage: AsyncMock) -> None:
        """Long neurons in Related Info section get clipped when compression enabled."""
        from neural_memory.engine.retrieval_context import format_context_budgeted

        long_content = " ".join(["foo"] * 500)
        n = _make_neuron("n_long", long_content, age_days=0.1)
        mock_storage.get_neurons_batch = AsyncMock(return_value={"n_long": n})

        activations = {"n_long": _make_activation("n_long", 0.9)}
        cfg = BudgetConfig(
            enable_related_compression=True,
            related_neuron_max_tokens=80,
        )
        context, tokens, _ = await format_context_budgeted(
            storage=mock_storage,
            activations=activations,
            fibers=[],
            max_tokens=2000,
            budget_config=cfg,
        )
        # Context should contain a truncated mention, not 500 "foo"s
        foo_count = context.count("foo")
        assert foo_count < 150  # Much less than 500

    @pytest.mark.asyncio
    async def test_disabled_preserves_legacy_behavior(self, mock_storage: AsyncMock) -> None:
        """With enable_related_compression=False, no truncation."""
        from neural_memory.engine.retrieval_context import format_context_budgeted

        long_content = " ".join(["bar"] * 300)
        n = _make_neuron("n_long", long_content, age_days=0.1)
        mock_storage.get_neurons_batch = AsyncMock(return_value={"n_long": n})

        activations = {"n_long": _make_activation("n_long", 0.9)}
        cfg = BudgetConfig(enable_related_compression=False)
        context, _, _ = await format_context_budgeted(
            storage=mock_storage,
            activations=activations,
            fibers=[],
            max_tokens=5000,
            budget_config=cfg,
        )
        # Legacy path: all 300 bars should fit within the generous budget
        assert context.count("bar") >= 250
