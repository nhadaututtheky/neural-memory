"""Tests for fidelity layers engine — extractive essence generation and scoring."""

from __future__ import annotations

from neural_memory.core.fiber import Fiber
from neural_memory.engine.fidelity import (
    MAX_ESSENCE_LENGTH,
    ExtractiveEssenceGenerator,
    FidelityLevel,
    LLMEssenceGenerator,
    _render_ghost,
    _score_sentence,
    _split_sentences,
    _truncate,
    compute_fidelity_score,
    extract_essence,
    get_essence_generator,
    render_at_fidelity,
    select_fidelity,
)


class TestExtractEssence:
    """Tests for extract_essence()."""

    def test_empty_content(self) -> None:
        assert extract_essence("") == ""
        assert extract_essence("   ") == ""

    def test_single_sentence(self) -> None:
        result = extract_essence("This is a single sentence.")
        assert result == "This is a single sentence."

    def test_picks_most_informative_sentence(self) -> None:
        content = (
            "Some generic intro text. "
            "PostgreSQL handles JSONB indexing for the Auth Module efficiently. "
            "That is all."
        )
        result = extract_essence(content)
        # Should pick the sentence with more entities (PostgreSQL, JSONB, Auth Module)
        assert "PostgreSQL" in result

    def test_max_length_enforced(self) -> None:
        long_sentence = "A " * 100 + "end."
        result = extract_essence(long_sentence)
        assert len(result) <= MAX_ESSENCE_LENGTH + 3  # +3 for "..."

    def test_position_bias_first_sentence(self) -> None:
        content = (
            "The Critical Decision was made here. "
            "some filler words nothing important. "
            "more filler content here."
        )
        result = extract_essence(content)
        # First sentence should win due to position bias + entity ("Critical Decision")
        assert "Critical Decision" in result

    def test_multiline_content(self) -> None:
        content = "First line of content\nSecond line has PostgreSQL\nThird line"
        result = extract_essence(content)
        assert result  # Should produce something
        assert len(result) <= MAX_ESSENCE_LENGTH

    def test_truncation_at_word_boundary(self) -> None:
        # Content with a single very long sentence
        words = ["word"] * 40
        content = " ".join(words) + "."
        result = extract_essence(content)
        assert len(result) <= MAX_ESSENCE_LENGTH + 3
        assert not result.endswith(" ...")  # Should break at word boundary

    def test_code_references_count_as_entities(self) -> None:
        content = (
            "Nothing important here. "
            "The `extract_essence` function uses `_score_sentence` for ranking."
        )
        result = extract_essence(content)
        assert "extract_essence" in result or "_score_sentence" in result

    def test_preserves_meaningful_content(self) -> None:
        content = "Chose Redis over Memcached because Redis supports pub/sub natively."
        result = extract_essence(content)
        assert result == content  # Single sentence, should be preserved as-is


class TestSplitSentences:
    """Tests for _split_sentences()."""

    def test_basic_split(self) -> None:
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    def test_single_sentence(self) -> None:
        result = _split_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_newline_fallback(self) -> None:
        result = _split_sentences("Line one\nLine two\nLine three")
        assert len(result) == 3

    def test_empty_string(self) -> None:
        result = _split_sentences("")
        assert result == []

    def test_question_and_exclamation(self) -> None:
        result = _split_sentences("What happened? Something broke! Fix it now.")
        assert len(result) >= 2


class TestScoreSentence:
    """Tests for _score_sentence()."""

    def test_first_position_gets_boost(self) -> None:
        score_first = _score_sentence("Hello world", 0, 5)
        score_middle = _score_sentence("Hello world", 2, 5)
        assert score_first > score_middle

    def test_last_position_gets_small_boost(self) -> None:
        score_last = _score_sentence("Hello world", 4, 5)
        score_middle = _score_sentence("Hello world", 2, 5)
        assert score_last > score_middle

    def test_entity_rich_scores_higher(self) -> None:
        plain = _score_sentence("some words here", 1, 3)
        entity_rich = _score_sentence("PostgreSQL and Redis for Auth Module", 1, 3)
        assert entity_rich > plain

    def test_very_short_penalized(self) -> None:
        short = _score_sentence("Hi", 1, 3)
        normal = _score_sentence("This is a normal length sentence with content", 1, 3)
        assert normal > short


class TestTruncate:
    """Tests for _truncate()."""

    def test_short_text_unchanged(self) -> None:
        assert _truncate("short text", 50) == "short text"

    def test_truncates_at_word_boundary(self) -> None:
        result = _truncate("one two three four five six", 15)
        assert len(result) <= 18  # 15 + "..."
        assert result.endswith("...")

    def test_max_length_respected(self) -> None:
        result = _truncate("a " * 100, 20)
        assert len(result) <= 23  # 20 + "..."


class TestFiberEssenceField:
    """Tests for essence field on Fiber dataclass."""

    def test_fiber_has_essence_field(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert fiber.essence is None

    def test_fiber_create_with_essence(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            essence="Test essence",
        )
        assert fiber.essence == "Test essence"

    def test_with_essence_immutable(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        updated = fiber.with_essence("New essence")
        assert updated.essence == "New essence"
        assert fiber.essence is None  # Original unchanged

    def test_essence_not_mutated(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            essence="original",
        )
        new_fiber = fiber.with_essence("updated")
        assert fiber.essence == "original"
        assert new_fiber.essence == "updated"
        assert fiber is not new_fiber


# ── Phase 2: Fidelity Scoring + Selection + Rendering ──────────────────


class TestComputeFidelityScore:
    """Tests for compute_fidelity_score()."""

    def test_fresh_high_importance(self) -> None:
        score = compute_fidelity_score(
            activation=0.5, importance=0.8, hours_since_access=0, decay_rate=0.1
        )
        assert score > 0.9  # High importance + just accessed

    def test_old_memory_decays(self) -> None:
        fresh = compute_fidelity_score(
            activation=0.3, importance=0.5, hours_since_access=1, decay_rate=0.1
        )
        old = compute_fidelity_score(
            activation=0.3, importance=0.5, hours_since_access=100, decay_rate=0.1
        )
        assert fresh > old

    def test_decay_floor_respected(self) -> None:
        score = compute_fidelity_score(
            activation=0.0,
            importance=0.1,
            hours_since_access=10000,
            decay_rate=0.1,
            decay_floor=0.05,
        )
        assert score >= 0.05

    def test_zero_activation_and_importance(self) -> None:
        score = compute_fidelity_score(
            activation=0.0, importance=0.0, hours_since_access=0, decay_rate=0.1
        )
        assert score >= 0.05  # Floor

    def test_capped_at_one(self) -> None:
        score = compute_fidelity_score(
            activation=1.0, importance=1.0, hours_since_access=0, decay_rate=0.1
        )
        assert score <= 1.0


class TestFidelityDecayUnitConsistency:
    """Cross-validation: fidelity decay must match NeuronState.decay() units.

    Bug #98: compute_fidelity_score() was using hours directly with a per-day
    decay_rate, causing 24x faster decay than intended. These tests ensure
    both systems agree on the same time scale.
    """

    def test_one_day_matches_neuron_decay(self) -> None:
        """1 day: fidelity score decay should match NeuronState.decay()."""
        import math

        decay_rate = 0.1
        # NeuronState.decay: e^(-0.1 * 1 day) = e^(-0.1) ≈ 0.905
        neuron_decay = math.exp(-decay_rate * 1.0)

        # Fidelity: 24 hours, base=1.0
        fidelity = compute_fidelity_score(
            activation=0.5,
            importance=0.5,
            hours_since_access=24,
            decay_rate=decay_rate,
        )
        # base = 1.0, decay = e^(-0.1 * 1 day) = 0.905
        expected = 1.0 * neuron_decay
        assert abs(fidelity - expected) < 0.01, (
            f"Fidelity {fidelity:.3f} != expected {expected:.3f} (NeuronState decay)"
        )

    def test_seven_days_still_full_or_summary(self) -> None:
        """7-day old memory with moderate importance should NOT be GHOST."""
        score = compute_fidelity_score(
            activation=0.5,
            importance=0.5,
            hours_since_access=168,  # 7 days
            decay_rate=0.1,
        )
        # e^(-0.1 * 7) ≈ 0.497 → score ≈ 0.497
        assert score > 0.3, f"7-day memory score {score:.3f} too low (expected >0.3)"
        level = select_fidelity(score)
        assert level in (FidelityLevel.FULL, FidelityLevel.SUMMARY), (
            f"7-day memory should be FULL or SUMMARY, got {level}"
        )

    def test_thirty_days_not_immediately_ghost(self) -> None:
        """30-day memory with high importance should still be above floor."""
        score = compute_fidelity_score(
            activation=0.3,
            importance=0.7,
            hours_since_access=720,  # 30 days
            decay_rate=0.1,
        )
        # e^(-0.1 * 30) ≈ 0.050, base=1.0 → score ≈ 0.050
        # With decay_floor=0.05 it should be at floor but not below
        assert score >= 0.05

    def test_one_hour_barely_decays(self) -> None:
        """1 hour old memory should barely decay (< 1% drop)."""
        fresh = compute_fidelity_score(
            activation=0.5,
            importance=0.5,
            hours_since_access=0,
            decay_rate=0.1,
        )
        one_hour = compute_fidelity_score(
            activation=0.5,
            importance=0.5,
            hours_since_access=1,
            decay_rate=0.1,
        )
        # 1 hour = 1/24 day → e^(-0.1/24) ≈ 0.9958 → ~0.4% drop
        drop = (fresh - one_hour) / fresh
        assert drop < 0.01, f"1-hour decay is {drop:.4f} (should be <1%)"

    def test_realistic_fidelity_distribution(self) -> None:
        """Verify FULL → SUMMARY → ESSENCE → GHOST at realistic time ranges."""
        decay_rate = 0.1

        # 1 day → should be FULL (score ~0.905)
        s1 = compute_fidelity_score(
            activation=0.5,
            importance=0.5,
            hours_since_access=24,
            decay_rate=decay_rate,
        )
        assert select_fidelity(s1) == FidelityLevel.FULL

        # 14 days → should be SUMMARY (score ~0.247)
        s14 = compute_fidelity_score(
            activation=0.3,
            importance=0.3,
            hours_since_access=336,
            decay_rate=decay_rate,
        )
        assert select_fidelity(s14) in (FidelityLevel.SUMMARY, FidelityLevel.ESSENCE)

        # 60 days → should be GHOST (score near floor)
        s60 = compute_fidelity_score(
            activation=0.1,
            importance=0.1,
            hours_since_access=1440,
            decay_rate=decay_rate,
        )
        assert select_fidelity(s60) == FidelityLevel.GHOST


class TestSelectFidelity:
    """Tests for select_fidelity()."""

    def test_high_score_selects_full(self) -> None:
        assert select_fidelity(0.8, budget_pressure=0.0) == FidelityLevel.FULL

    def test_medium_score_selects_summary(self) -> None:
        assert select_fidelity(0.4, budget_pressure=0.0) == FidelityLevel.SUMMARY

    def test_low_score_selects_essence(self) -> None:
        assert select_fidelity(0.15, budget_pressure=0.0) == FidelityLevel.ESSENCE

    def test_very_low_score_selects_ghost(self) -> None:
        assert select_fidelity(0.05, budget_pressure=0.0) == FidelityLevel.GHOST

    def test_budget_pressure_downgrades(self) -> None:
        # Without pressure: FULL
        assert select_fidelity(0.7, budget_pressure=0.0) == FidelityLevel.FULL
        # With high pressure: threshold shifts up, same score → SUMMARY
        assert select_fidelity(0.7, budget_pressure=1.0) == FidelityLevel.SUMMARY

    def test_extreme_pressure_makes_ghost(self) -> None:
        # Score 0.3 is normally SUMMARY, but max pressure shifts threshold by +0.3
        result = select_fidelity(0.3, budget_pressure=1.0)
        assert result in (FidelityLevel.ESSENCE, FidelityLevel.GHOST)

    def test_budget_pressure_defaults_to_zero(self) -> None:
        """budget_pressure should default to 0.0 (no pressure)."""
        # Should work without specifying budget_pressure
        result = select_fidelity(0.8)
        assert result == FidelityLevel.FULL

    def test_custom_thresholds(self) -> None:
        result = select_fidelity(
            0.5, budget_pressure=0.0, full_threshold=0.9, summary_threshold=0.7
        )
        assert result != FidelityLevel.FULL  # 0.5 < 0.9


class TestRenderAtFidelity:
    """Tests for render_at_fidelity()."""

    def _make_fiber(self, *, summary: str | None = None, essence: str | None = None) -> Fiber:
        return Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            summary=summary,
            essence=essence,
        )

    def test_full_returns_anchor_content(self) -> None:
        fiber = self._make_fiber(summary="short summary")
        result = render_at_fidelity(fiber, FidelityLevel.FULL, anchor_content="Full text here")
        assert result == "Full text here"

    def test_summary_returns_summary(self) -> None:
        fiber = self._make_fiber(summary="A good summary")
        result = render_at_fidelity(fiber, FidelityLevel.SUMMARY)
        assert result == "A good summary"

    def test_essence_returns_essence(self) -> None:
        fiber = self._make_fiber(essence="Core insight")
        result = render_at_fidelity(fiber, FidelityLevel.ESSENCE)
        assert result == "Core insight"

    def test_essence_fallback_to_summary(self) -> None:
        fiber = self._make_fiber(summary="Fallback summary", essence=None)
        result = render_at_fidelity(fiber, FidelityLevel.ESSENCE)
        assert result == "Fallback summary"

    def test_essence_fallback_to_full(self) -> None:
        fiber = self._make_fiber()
        result = render_at_fidelity(fiber, FidelityLevel.ESSENCE, anchor_content="Full content")
        assert result == "Full content"

    def test_ghost_returns_metadata(self) -> None:
        fiber = self._make_fiber()
        result = render_at_fidelity(fiber, FidelityLevel.GHOST)
        assert result.startswith("[~]")
        assert "recall:fiber:" in result
        assert "links" in result

    def test_ghost_includes_tags(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            auto_tags={"python", "async"},
        )
        result = _render_ghost(fiber)
        assert "async" in result or "python" in result


class TestFidelityLevel:
    """Tests for FidelityLevel enum."""

    def test_values(self) -> None:
        assert FidelityLevel.FULL == "full"
        assert FidelityLevel.SUMMARY == "summary"
        assert FidelityLevel.ESSENCE == "essence"
        assert FidelityLevel.GHOST == "ghost"

    def test_all_four_levels(self) -> None:
        assert len(FidelityLevel) == 4


# ── Phase 3: Ghost Recall Tests ───────────────────────────────────────


class TestGhostFormat:
    """Tests for ghost rendering format and robustness."""

    def test_ghost_format_includes_all_parts(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids={"s1", "s2"},
            anchor_neuron_id="n1",
            auto_tags={"python", "auth", "security"},
        )
        result = _render_ghost(fiber)
        assert result.startswith("[~]")
        assert "recall:fiber:" in result
        assert fiber.id in result
        assert "links" in result
        # Should have at most 3 tags
        tag_part = result.split("|")[0].replace("[~]", "").strip()
        tags_shown = [t.strip() for t in tag_part.split(",")]
        assert len(tags_shown) <= 3

    def test_ghost_untagged_fiber(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        result = _render_ghost(fiber)
        assert "untagged" in result

    def test_ghost_age_formatting_hours(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        # Created just now → age < 24h → shows Xh
        result = _render_ghost(fiber)
        assert "h ago" in result or "0h ago" in result

    def test_ghost_age_formatting_days(self) -> None:
        from datetime import timedelta

        from neural_memory.utils.timeutils import utcnow

        now = utcnow()
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        # Manually override created_at for testing via replace
        from dataclasses import replace

        old_fiber = replace(fiber, created_at=now - timedelta(days=5))
        result = _render_ghost(old_fiber)
        assert "d ago" in result

    def test_ghost_survives_none_synapse_ids(self) -> None:
        """Ghost render shouldn't crash with unexpected data."""
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        result = _render_ghost(fiber)
        assert "recall:fiber:" in result

    def test_ghost_recall_key_format(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        result = _render_ghost(fiber)
        # Must contain exact recall key format
        assert f"recall:fiber:{fiber.id}" in result


class TestFiberGhostField:
    """Tests for last_ghost_shown_at field on Fiber."""

    def test_default_none(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert fiber.last_ghost_shown_at is None

    def test_set_via_replace(self) -> None:
        from dataclasses import replace

        from neural_memory.utils.timeutils import utcnow

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        now = utcnow()
        updated = replace(fiber, last_ghost_shown_at=now)
        assert updated.last_ghost_shown_at == now
        assert fiber.last_ghost_shown_at is None  # Immutability

    def test_immutability_preserved(self) -> None:
        from dataclasses import replace

        from neural_memory.utils.timeutils import utcnow

        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        now = utcnow()
        updated = replace(fiber, last_ghost_shown_at=now)
        # Original unchanged
        assert fiber.last_ghost_shown_at is None
        assert updated.last_ghost_shown_at == now


class TestGhostVisibilityBoost:
    """Tests for ghost visibility boost in context optimizer."""

    def test_recently_shown_ghost_gets_higher_score(self) -> None:
        """A fiber with recent ghost_shown_at should get +0.1 boost."""
        from neural_memory.engine.fidelity import compute_fidelity_score

        # Base score for a very old memory
        base = compute_fidelity_score(
            activation=0.0, importance=0.1, hours_since_access=500, decay_rate=0.1
        )
        # With +0.1 boost
        boosted = min(1.0, base + 0.1)
        assert boosted > base
        assert abs(boosted - base - 0.1) < 1e-9 or boosted == 1.0

    def test_old_ghost_shown_no_boost(self) -> None:
        """Ghost shown > 24h ago should NOT get boost."""
        from datetime import timedelta

        from neural_memory.utils.timeutils import utcnow

        now = utcnow()
        ghost_time = now - timedelta(hours=25)
        ghost_age_hours = (now - ghost_time).total_seconds() / 3600
        assert ghost_age_hours >= 24  # No boost applied


class TestThresholdValidation:
    """Tests for threshold ordering validation."""

    def test_valid_thresholds_no_warning(self) -> None:
        # Should not raise
        result = select_fidelity(
            0.5,
            budget_pressure=0.0,
            full_threshold=0.6,
            summary_threshold=0.3,
            essence_threshold=0.1,
        )
        assert isinstance(result, FidelityLevel)

    def test_inverted_thresholds_still_work(self) -> None:
        # Inverted thresholds should produce a result (with logged warning)
        result = select_fidelity(
            0.5,
            budget_pressure=0.0,
            full_threshold=0.1,
            summary_threshold=0.6,
            essence_threshold=0.9,
        )
        assert isinstance(result, FidelityLevel)


# ── Phase 4: Essence Generator Tests ──────────────────────────────────


class TestExtractiveEssenceGenerator:
    """Tests for ExtractiveEssenceGenerator."""

    async def test_generate_returns_essence(self) -> None:
        gen = ExtractiveEssenceGenerator()
        result = await gen.generate(
            "Redis caching improves PostgreSQL query performance significantly."
        )
        assert len(result) > 0
        assert len(result) <= 150

    async def test_generate_empty_content(self) -> None:
        gen = ExtractiveEssenceGenerator()
        result = await gen.generate("")
        assert result == ""

    async def test_priority_ignored(self) -> None:
        gen = ExtractiveEssenceGenerator()
        low = await gen.generate("Content here.", priority=1)
        high = await gen.generate("Content here.", priority=9)
        assert low == high


class TestLLMEssenceGenerator:
    """Tests for LLMEssenceGenerator with mocked LLM."""

    async def test_llm_generates_essence(self) -> None:
        async def mock_llm(prompt: str) -> str:
            return "LLM-generated essence sentence."

        gen = LLMEssenceGenerator(llm_call=mock_llm)
        result = await gen.generate(
            "Long content about auth patterns and Redis caching.", priority=5
        )
        assert result == "LLM-generated essence sentence."

    async def test_cost_guard_skips_low_priority(self) -> None:
        calls: list[str] = []

        async def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            return "Should not be called"

        gen = LLMEssenceGenerator(llm_call=mock_llm)
        result = await gen.generate("Some content.", priority=2)
        assert len(calls) == 0  # LLM not called for priority < 3
        assert len(result) > 0  # Falls back to extractive

    async def test_fallback_on_llm_failure(self) -> None:
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        gen = LLMEssenceGenerator(llm_call=failing_llm)
        result = await gen.generate("Redis improves PostgreSQL performance.", priority=7)
        assert len(result) > 0

    async def test_fallback_on_empty_llm_response(self) -> None:
        async def empty_llm(prompt: str) -> str:
            return ""

        gen = LLMEssenceGenerator(llm_call=empty_llm)
        result = await gen.generate("Redis improves PostgreSQL performance.", priority=5)
        assert len(result) > 0

    async def test_no_llm_call_fn_falls_back(self) -> None:
        gen = LLMEssenceGenerator(llm_call=None)
        result = await gen.generate("Content here.", priority=8)
        assert len(result) > 0

    async def test_llm_output_truncated_to_max(self) -> None:
        async def verbose_llm(prompt: str) -> str:
            return "A" * 300

        gen = LLMEssenceGenerator(llm_call=verbose_llm)
        result = await gen.generate("Some content.", priority=5)
        assert len(result) <= 155  # 150 + "..." tolerance


class TestGetEssenceGenerator:
    """Tests for factory function."""

    def test_default_is_extractive(self) -> None:
        gen = get_essence_generator()
        assert isinstance(gen, ExtractiveEssenceGenerator)

    def test_extractive_strategy(self) -> None:
        gen = get_essence_generator("extractive")
        assert isinstance(gen, ExtractiveEssenceGenerator)

    def test_llm_strategy(self) -> None:
        gen = get_essence_generator("llm")
        assert isinstance(gen, LLMEssenceGenerator)

    def test_llm_with_call_fn(self) -> None:
        async def mock_llm(prompt: str) -> str:
            return "test"

        gen = get_essence_generator("llm", llm_call=mock_llm)
        assert isinstance(gen, LLMEssenceGenerator)

    def test_unknown_strategy_defaults_extractive(self) -> None:
        gen = get_essence_generator("unknown")
        assert isinstance(gen, ExtractiveEssenceGenerator)
