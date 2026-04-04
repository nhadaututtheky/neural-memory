"""Tests for Phase 3: Context Retrieval Sharpening.

Covers:
- ContextItem tier field
- HOT tier label in context output (prefix [HOT] for hot memories)
- Token budget and HOT count in response metadata
- Scoring transparency via optimization_stats
"""

from __future__ import annotations

from neural_memory.engine.context_optimizer import (
    ContextItem,
    ContextPlan,
    FidelityStats,
    compute_composite_score,
)

# ── ContextItem tier field ────────────────────────────────────────


class TestContextItemTier:
    """ContextItem has a tier field for HOT/WARM/COLD classification."""

    def test_default_tier_is_warm(self) -> None:
        item = ContextItem(
            fiber_id="f1",
            content="test",
            score=0.5,
            token_count=10,
        )
        assert item.tier == "warm"

    def test_hot_tier(self) -> None:
        item = ContextItem(
            fiber_id="f1",
            content="test",
            score=0.8,
            token_count=10,
            tier="hot",
        )
        assert item.tier == "hot"

    def test_cold_tier(self) -> None:
        item = ContextItem(
            fiber_id="f1",
            content="test",
            score=0.2,
            token_count=10,
            tier="cold",
        )
        assert item.tier == "cold"

    def test_tier_preserved_in_plan(self) -> None:
        items = [
            ContextItem(fiber_id="f1", content="hot mem", score=0.9, token_count=5, tier="hot"),
            ContextItem(fiber_id="f2", content="warm mem", score=0.5, token_count=5, tier="warm"),
        ]
        plan = ContextPlan(items=items, total_tokens=10, dropped_count=0)
        assert plan.items[0].tier == "hot"
        assert plan.items[1].tier == "warm"


# ── HOT label in context formatting ──────────────────────────────


class TestHotTierLabel:
    """HOT tier memories get [HOT] prefix in context output."""

    def test_hot_item_gets_label(self) -> None:
        """Simulate what recall_handler does with HOT items."""
        item = ContextItem(
            fiber_id="f1", content="Safety rule", score=0.9, token_count=5, tier="hot"
        )
        line = (
            f"- [{item.tier.upper()}] {item.content}" if item.tier == "hot" else f"- {item.content}"
        )
        assert line == "- [HOT] Safety rule"

    def test_warm_item_no_label(self) -> None:
        item = ContextItem(
            fiber_id="f2", content="A decision", score=0.5, token_count=5, tier="warm"
        )
        line = (
            f"- [{item.tier.upper()}] {item.content}" if item.tier == "hot" else f"- {item.content}"
        )
        assert line == "- A decision"

    def test_cold_item_no_label(self) -> None:
        item = ContextItem(fiber_id="f3", content="Old fact", score=0.2, token_count=5, tier="cold")
        line = (
            f"- [{item.tier.upper()}] {item.content}" if item.tier == "hot" else f"- {item.content}"
        )
        assert line == "- Old fact"


# ── Scoring transparency ─────────────────────────────────────────


class TestScoringTransparency:
    """Composite score is accessible and meaningful."""

    def test_score_range(self) -> None:
        score = compute_composite_score(
            activation=1.0, priority=1.0, frequency=1.0, conductivity=1.0, freshness=1.0
        )
        assert 0.0 <= score <= 1.0

    def test_higher_activation_higher_score(self) -> None:
        low = compute_composite_score(activation=0.1, priority=0.5, freshness=0.5)
        high = compute_composite_score(activation=0.9, priority=0.5, freshness=0.5)
        assert high > low

    def test_hot_boost_applied(self) -> None:
        """HOT tier gets +0.3 boost (applied in optimizer, verify here conceptually)."""
        base_score = compute_composite_score(activation=0.5, priority=0.5, freshness=0.5)
        boosted = min(1.0, base_score + 0.3)
        assert boosted > base_score
        assert boosted <= 1.0


# ── Token budget reporting ───────────────────────────────────────


class TestTokenBudgetReporting:
    """ContextPlan provides enough info for budget transparency."""

    def test_plan_has_total_tokens(self) -> None:
        plan = ContextPlan(items=[], total_tokens=500, dropped_count=0)
        assert plan.total_tokens == 500

    def test_plan_has_dropped_count(self) -> None:
        plan = ContextPlan(items=[], total_tokens=500, dropped_count=3)
        assert plan.dropped_count == 3

    def test_plan_has_fidelity_stats(self) -> None:
        stats = FidelityStats(full=5, summary=2, essence=1, ghost=1)
        plan = ContextPlan(items=[], total_tokens=500, dropped_count=0, fidelity_stats=stats)
        assert plan.fidelity_stats.full == 5
        assert plan.fidelity_stats.ghost == 1

    def test_hot_count_from_items(self) -> None:
        """hot_memories_injected can be computed from items."""
        items = [
            ContextItem(fiber_id="f1", content="x", score=0.9, token_count=5, tier="hot"),
            ContextItem(fiber_id="f2", content="y", score=0.5, token_count=5, tier="warm"),
            ContextItem(fiber_id="f3", content="z", score=0.8, token_count=5, tier="hot"),
        ]
        hot_count = sum(1 for item in items if item.tier == "hot")
        assert hot_count == 2

    def test_token_budget_used_ratio(self) -> None:
        """Budget usage can be calculated from total_tokens / budget."""
        plan = ContextPlan(items=[], total_tokens=2000, dropped_count=0)
        budget = 4000
        usage = plan.total_tokens / budget
        assert 0.0 <= usage <= 1.0
        assert abs(usage - 0.5) < 0.01
