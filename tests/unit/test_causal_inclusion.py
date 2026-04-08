"""Tests for engine/causal_inclusion.py — causal auto-inclusion for recall."""

from __future__ import annotations

import pytest

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.causal_inclusion import (
    CausalContext,
    format_causal_supplement,
)
from neural_memory.engine.causal_traversal import CausalChain, CausalStep

# ---------------------------------------------------------------------------
# format_causal_supplement
# ---------------------------------------------------------------------------


class TestFormatCausalSupplement:
    """Tests for format_causal_supplement()."""

    def test_empty_chains(self) -> None:
        result = format_causal_supplement([], max_chars=500)
        assert result == ""

    def test_chain_with_no_steps(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n1",
            direction="causes",
            steps=(),
            total_weight=0.0,
        )
        result = format_causal_supplement([chain])
        assert result == ""

    def test_causes_direction_label(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n1",
            direction="causes",
            steps=(
                CausalStep(
                    neuron_id="n2",
                    content="Server timeout occurred",
                    synapse_type=SynapseType.CAUSED_BY,
                    weight=0.8,
                    depth=0,
                ),
            ),
            total_weight=0.8,
        )
        result = format_causal_supplement([chain])
        assert "[Caused by]" in result
        assert "Server timeout occurred" in result

    def test_effects_direction_label(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n1",
            direction="effects",
            steps=(
                CausalStep(
                    neuron_id="n3",
                    content="Users lost data",
                    synapse_type=SynapseType.LEADS_TO,
                    weight=0.9,
                    depth=0,
                ),
            ),
            total_weight=0.9,
        )
        result = format_causal_supplement([chain])
        assert "[Led to]" in result
        assert "Users lost data" in result

    def test_multiple_chains(self) -> None:
        chains = [
            CausalChain(
                seed_neuron_id="n1",
                direction="causes",
                steps=(CausalStep("n2", "Cause A", SynapseType.CAUSED_BY, 0.8, 0),),
                total_weight=0.8,
            ),
            CausalChain(
                seed_neuron_id="n1",
                direction="effects",
                steps=(CausalStep("n3", "Effect B", SynapseType.LEADS_TO, 0.7, 0),),
                total_weight=0.7,
            ),
        ]
        result = format_causal_supplement(chains, max_chars=5000)
        assert "[Caused by] Cause A" in result
        assert "[Led to] Effect B" in result

    def test_budget_truncation(self) -> None:
        long_content = "x" * 100
        chains = [
            CausalChain(
                seed_neuron_id="n1",
                direction="causes",
                steps=(
                    CausalStep("n2", long_content, SynapseType.CAUSED_BY, 0.8, 0),
                    CausalStep("n3", long_content, SynapseType.CAUSED_BY, 0.7, 1),
                    CausalStep("n4", long_content, SynapseType.CAUSED_BY, 0.6, 2),
                ),
                total_weight=0.3,
            ),
        ]
        result = format_causal_supplement(chains, max_chars=150)
        # Should fit at most 1 line (~112 chars for "[Caused by] " + 100 "x"s)
        assert len(result) <= 150


# ---------------------------------------------------------------------------
# CausalContext dataclass
# ---------------------------------------------------------------------------


class TestCausalContext:
    """Tests for CausalContext frozen dataclass."""

    def test_creation(self) -> None:
        ctx = CausalContext(
            chains=(),
            neuron_ids=frozenset(),
            supplement_text="",
        )
        assert ctx.chains == ()
        assert ctx.neuron_ids == frozenset()
        assert ctx.supplement_text == ""

    def test_immutable(self) -> None:
        ctx = CausalContext(
            chains=(),
            neuron_ids=frozenset({"n1"}),
            supplement_text="test",
        )
        with pytest.raises(AttributeError):
            ctx.supplement_text = "modified"  # type: ignore[misc]
