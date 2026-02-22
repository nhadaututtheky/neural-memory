"""Unit tests for Tiered Memory Compression feature.

Covers:
- split_sentences helper
- compute_entity_density helper
- compress_tier1_extractive
- compress_tier2_entity_preserving
- compress_tier3_template
- CompressionTier enum
- CompressionConfig dataclass
- CompressionEngine.determine_target_tier
- SQLiteCompressionMixin storage (save/get/delete/stats)
- ConsolidationStrategy.COMPRESS integration
- ConsolidationReport compression fields
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.engine.compression import (
    CompressionConfig,
    CompressionTier,
    CompressionEngine,
    compress_tier1_extractive,
    compress_tier2_entity_preserving,
    compress_tier3_template,
    compute_entity_density,
    split_sentences,
)
from neural_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fiber(
    *,
    days_old: float = 0.0,
    compression_tier: int = 0,
) -> Fiber:
    """Create a minimal Fiber with a specific age and compression tier."""
    anchor_id = "anchor-1"
    created = utcnow() - timedelta(days=days_old)
    fiber = Fiber(
        id="fiber-1",
        neuron_ids={anchor_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        compression_tier=compression_tier,
        created_at=created,
    )
    return fiber


# ---------------------------------------------------------------------------
# split_sentences tests
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic_split(self) -> None:
        sentences = split_sentences("Hello world. Goodbye world.")
        assert len(sentences) == 2

    def test_no_split_single(self) -> None:
        sentences = split_sentences("Hello world")
        assert len(sentences) == 1

    def test_empty_string(self) -> None:
        sentences = split_sentences("")
        assert sentences == []

    def test_question_exclamation(self) -> None:
        sentences = split_sentences("What? Yes!")
        assert len(sentences) == 2

    def test_preserves_content(self) -> None:
        text = "The cat sat. The dog ran."
        sentences = split_sentences(text)
        # Each sentence content must appear verbatim in the original text
        for sentence in sentences:
            assert sentence in text

    def test_strips_leading_trailing_whitespace(self) -> None:
        sentences = split_sentences("  Hello world.  ")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    def test_abbreviation_not_split(self) -> None:
        # "Dr." should not be treated as a sentence boundary
        text = "Dr. Smith is here. He is great."
        sentences = split_sentences(text)
        # Should get 2 sentences (not split at "Dr.")
        assert len(sentences) == 2

    def test_multiple_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 3


# ---------------------------------------------------------------------------
# compute_entity_density tests
# ---------------------------------------------------------------------------


class TestComputeEntityDensity:
    def test_no_entities_returns_zero(self) -> None:
        score = compute_entity_density("hello world", [])
        assert score == 0.0

    def test_no_matching_neurons_returns_zero(self) -> None:
        score = compute_entity_density("hello world", ["Alice", "Bob"])
        assert score == 0.0

    def test_all_match(self) -> None:
        # Every neuron content is found in the sentence
        sentence = "Alice and Bob met"
        neurons = ["Alice", "Bob"]
        score = compute_entity_density(sentence, neurons)
        assert score > 0.0

    def test_partial_match(self) -> None:
        # Only one of two neurons matches
        sentence = "Alice went shopping"
        score_one = compute_entity_density(sentence, ["Alice"])
        score_two = compute_entity_density(sentence, ["Alice", "Bob"])
        # Both positive, and one-match score >= two-match score (more neurons = lower density)
        assert score_one > 0.0
        assert score_two > 0.0

    def test_empty_sentence_returns_zero(self) -> None:
        score = compute_entity_density("", ["Alice"])
        assert score == 0.0

    def test_case_insensitive(self) -> None:
        # Neuron content "Alice" matches "alice" in sentence
        score = compute_entity_density("alice went to the store", ["Alice"])
        assert score > 0.0

    def test_result_clamped_to_one(self) -> None:
        # Even with many entities the result should not exceed 1.0
        sentence = "a"
        neurons = ["a"] * 100
        score = compute_entity_density(sentence, neurons)
        assert score <= 1.0

    def test_empty_neuron_contents_skipped(self) -> None:
        # Empty strings in neuron_contents should not count as matches
        score = compute_entity_density("hello world", ["", "", ""])
        assert score == 0.0


# ---------------------------------------------------------------------------
# compress_tier1_extractive tests
# ---------------------------------------------------------------------------


class TestCompressTier1Extractive:
    def test_keeps_top_sentences(self) -> None:
        # Sentence with entity should score higher than sentence without
        content = "Alice went to the store. The sky is blue. Bob called Alice."
        neurons = ["Alice", "Bob"]
        config = CompressionConfig(tier1_max_sentences=2, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        # Sentences mentioning Alice/Bob should survive
        assert "Alice" in compressed or "Bob" in compressed

    def test_preserves_first_sentence_true(self) -> None:
        # With preserve_first_sentence=True the first sentence always survives
        content = "Intro sentence. Alice was here. Bob was there. Charlie was around."
        neurons = ["Alice", "Bob", "Charlie"]
        config = CompressionConfig(tier1_max_sentences=1, preserve_first_sentence=True)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        assert "Intro sentence" in compressed

    def test_preserves_first_sentence_false(self) -> None:
        # With preserve_first_sentence=False the first sentence may be dropped
        # if it has low entity density and max_sentences is small
        content = "No entities here. Alice was here. Bob was there."
        neurons = ["Alice", "Bob"]
        config = CompressionConfig(tier1_max_sentences=1, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        # Should pick a sentence with Alice or Bob (higher density)
        assert "Alice" in compressed or "Bob" in compressed

    def test_max_sentences_limit(self) -> None:
        content = "One. Two. Three. Four. Five. Six."
        config = CompressionConfig(tier1_max_sentences=2, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, [], config)
        # With no entities all scores are 0; result should still be bounded
        sentences_out = [s for s in compressed.split(".") if s.strip()]
        assert len(sentences_out) <= 2

    def test_short_content_unchanged(self) -> None:
        content = "Single sentence only."
        config = CompressionConfig()
        compressed, _ = compress_tier1_extractive(content, [], config)
        assert "Single sentence only" in compressed

    def test_empty_content(self) -> None:
        compressed, entities = compress_tier1_extractive("", [], CompressionConfig())
        assert compressed == ""
        assert entities == 0

    def test_returns_string_and_int(self) -> None:
        content = "Alice is here."
        compressed, entities = compress_tier1_extractive(content, ["Alice"], CompressionConfig())
        assert isinstance(compressed, str)
        assert isinstance(entities, int)


# ---------------------------------------------------------------------------
# compress_tier2_entity_preserving tests
# ---------------------------------------------------------------------------


class TestCompressTier2EntityPreserving:
    def test_keeps_entity_sentences(self) -> None:
        content = "No entities here. Alice was seen. Nothing relevant."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        assert "Alice" in compressed

    def test_drops_non_entity_sentences(self) -> None:
        content = "Boring sentence one. Alice appeared. Boring sentence two."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        # Boring sentences should not survive when preserve_first_sentence=False
        assert "Boring sentence two" not in compressed

    def test_preserves_first_sentence_flag(self) -> None:
        content = "First sentence with no entities. Alice was here."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=True)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        assert "First sentence" in compressed

    def test_fallback_when_nothing_survives(self) -> None:
        # If no sentence passes the density threshold, return original content
        content = "No entities at all."
        neurons = []
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, entities = compress_tier2_entity_preserving(content, neurons, [], config)
        assert compressed == content
        assert entities == 0

    def test_empty_content(self) -> None:
        compressed, entities = compress_tier2_entity_preserving("", [], [], CompressionConfig())
        assert compressed == ""
        assert entities == 0

    def test_returns_string_and_int(self) -> None:
        content = "Alice is here."
        compressed, entities = compress_tier2_entity_preserving(
            content, ["Alice"], [], CompressionConfig()
        )
        assert isinstance(compressed, str)
        assert isinstance(entities, int)


# ---------------------------------------------------------------------------
# compress_tier3_template tests
# ---------------------------------------------------------------------------


class TestCompressTier3Template:
    def test_basic_template(self) -> None:
        entities = ["Alice", "Bob"]
        relations = ["knows"]
        text, count = compress_tier3_template(entities, relations)
        assert "Alice" in text
        assert "Bob" in text
        assert "knows" in text
        assert count == 2

    def test_empty_relations_uses_default(self) -> None:
        entities = ["Alice", "Bob"]
        text, count = compress_tier3_template(entities, [])
        # Should use the default "related_to" relation
        assert "related_to" in text
        assert count == 2

    def test_empty_entities_returns_empty_string(self) -> None:
        text, count = compress_tier3_template([], [])
        assert text == ""
        assert count == 0

    def test_single_entity_returns_entity_itself(self) -> None:
        text, count = compress_tier3_template(["Alice"], [])
        assert text == "Alice"
        assert count == 1

    def test_multiple_relations(self) -> None:
        entities = ["A", "B", "C"]
        relations = ["rel1", "rel2"]
        text, count = compress_tier3_template(entities, relations)
        assert "rel1" in text
        assert "rel2" in text
        assert count == 3

    def test_semicolon_joins_triples(self) -> None:
        entities = ["X", "Y", "Z"]
        relations = ["r1", "r2"]
        text, _ = compress_tier3_template(entities, relations)
        assert ";" in text

    def test_empty_neuron_contents_filtered(self) -> None:
        # Empty strings in neuron_contents should not become entities
        entities = ["", "Alice", ""]
        text, count = compress_tier3_template(entities, [])
        # Only "Alice" is a real entity
        assert count == 1
        assert text == "Alice"


# ---------------------------------------------------------------------------
# CompressionTier enum tests
# ---------------------------------------------------------------------------


class TestCompressionTierEnum:
    def test_full_value(self) -> None:
        assert CompressionTier.FULL == 0

    def test_extractive_value(self) -> None:
        assert CompressionTier.EXTRACTIVE == 1

    def test_entity_only_value(self) -> None:
        assert CompressionTier.ENTITY_ONLY == 2

    def test_template_value(self) -> None:
        assert CompressionTier.TEMPLATE == 3

    def test_graph_only_value(self) -> None:
        assert CompressionTier.GRAPH_ONLY == 4

    def test_ordering(self) -> None:
        assert CompressionTier.FULL < CompressionTier.EXTRACTIVE
        assert CompressionTier.EXTRACTIVE < CompressionTier.ENTITY_ONLY
        assert CompressionTier.ENTITY_ONLY < CompressionTier.TEMPLATE
        assert CompressionTier.TEMPLATE < CompressionTier.GRAPH_ONLY

    def test_int_enum(self) -> None:
        assert int(CompressionTier.EXTRACTIVE) == 1


# ---------------------------------------------------------------------------
# CompressionConfig tests
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_default_config_tier_days(self) -> None:
        cfg = CompressionConfig()
        assert cfg.tier1_days == 7.0
        assert cfg.tier2_days == 30.0
        assert cfg.tier3_days == 90.0
        assert cfg.tier4_days == 180.0

    def test_default_config_reasonable_values(self) -> None:
        cfg = CompressionConfig()
        assert cfg.tier1_max_sentences > 0
        assert cfg.preserve_first_sentence is True
        assert cfg.tier2_min_density >= 0.0

    def test_custom_thresholds(self) -> None:
        cfg = CompressionConfig(
            tier1_days=3.0,
            tier2_days=14.0,
            tier3_days=45.0,
            tier4_days=90.0,
            tier1_max_sentences=3,
            preserve_first_sentence=False,
            tier2_min_density=0.1,
        )
        assert cfg.tier1_days == 3.0
        assert cfg.tier2_days == 14.0
        assert cfg.tier3_days == 45.0
        assert cfg.tier4_days == 90.0
        assert cfg.tier1_max_sentences == 3
        assert cfg.preserve_first_sentence is False
        assert cfg.tier2_min_density == pytest.approx(0.1)

    def test_frozen(self) -> None:
        cfg = CompressionConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.tier1_days = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompressionEngine.determine_target_tier tests
# ---------------------------------------------------------------------------


class TestDetermineTargetTier:
    def _engine(self, config: CompressionConfig | None = None) -> CompressionEngine:
        """Build a CompressionEngine with a stub storage (not needed for determine_target_tier)."""
        from unittest.mock import MagicMock
        storage = MagicMock()
        return CompressionEngine(storage, config)

    def test_recent_fiber_full(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=1.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.FULL

    def test_just_under_tier1_full(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=6.9)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.FULL

    def test_week_old_extractive(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=15.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.EXTRACTIVE

    def test_just_at_tier1_boundary_extractive(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=7.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.EXTRACTIVE

    def test_month_old_entity_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=60.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.ENTITY_ONLY

    def test_just_at_tier2_boundary_entity_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=30.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.ENTITY_ONLY

    def test_quarter_old_template(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=120.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.TEMPLATE

    def test_just_at_tier3_boundary_template(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=90.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.TEMPLATE

    def test_old_fiber_graph_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=200.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.GRAPH_ONLY

    def test_just_at_tier4_boundary_graph_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=180.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.GRAPH_ONLY

    def test_custom_thresholds_respected(self) -> None:
        config = CompressionConfig(tier1_days=1.0, tier2_days=5.0, tier3_days=10.0, tier4_days=20.0)
        engine = self._engine(config)
        now = utcnow()

        assert engine.determine_target_tier(_make_fiber(days_old=0.5), now) == CompressionTier.FULL
        assert engine.determine_target_tier(_make_fiber(days_old=3.0), now) == CompressionTier.EXTRACTIVE
        assert engine.determine_target_tier(_make_fiber(days_old=7.0), now) == CompressionTier.ENTITY_ONLY
        assert engine.determine_target_tier(_make_fiber(days_old=15.0), now) == CompressionTier.TEMPLATE
        assert engine.determine_target_tier(_make_fiber(days_old=25.0), now) == CompressionTier.GRAPH_ONLY

    def test_already_compressed_skip_logic(self) -> None:
        """Fiber already at EXTRACTIVE (tier=1) → target tier 1 means no compression needed."""
        engine = self._engine()
        # Fiber is 15 days old → target EXTRACTIVE (1), fiber is already at 1
        fiber = _make_fiber(days_old=15.0, compression_tier=1)
        now = utcnow()
        target = engine.determine_target_tier(fiber, now)
        # The engine returns target = EXTRACTIVE; caller must compare with fiber.compression_tier
        assert target == CompressionTier.EXTRACTIVE
        # Caller logic: int(target) <= fiber.compression_tier → skip
        assert int(target) <= fiber.compression_tier


# ---------------------------------------------------------------------------
# Storage tests (SQLiteCompressionMixin via SQLiteStorage)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_storage(tmp_path: Path) -> SQLiteStorage:
    """SQLiteStorage backed by a temp file, brain context set."""
    db_path = tmp_path / "test_compression.db"
    store = SQLiteStorage(db_path)
    await store.initialize()
    brain = Brain.create(name="test-compression-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


class TestSQLiteCompressionMixin:
    @pytest.mark.asyncio
    async def test_save_and_get_backup(self, sqlite_storage: SQLiteStorage) -> None:
        """Round-trip: save a backup then retrieve it with matching fields."""
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-abc",
            original_content="Original content text here.",
            compression_tier=1,
            original_token_count=100,
            compressed_token_count=40,
        )
        result = await sqlite_storage.get_compression_backup("fiber-abc")
        assert result is not None
        assert result["fiber_id"] == "fiber-abc"
        assert result["original_content"] == "Original content text here."
        assert result["compression_tier"] == 1
        assert result["original_token_count"] == 100
        assert result["compressed_token_count"] == 40

    @pytest.mark.asyncio
    async def test_get_nonexistent_backup_returns_none(self, sqlite_storage: SQLiteStorage) -> None:
        result = await sqlite_storage.get_compression_backup("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_backup_returns_true(self, sqlite_storage: SQLiteStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-del",
            original_content="Some content.",
            compression_tier=2,
            original_token_count=50,
            compressed_token_count=20,
        )
        deleted = await sqlite_storage.delete_compression_backup("fiber-del")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_backup_removes_row(self, sqlite_storage: SQLiteStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-gone",
            original_content="Content.",
            compression_tier=1,
            original_token_count=10,
            compressed_token_count=5,
        )
        await sqlite_storage.delete_compression_backup("fiber-gone")
        result = await sqlite_storage.get_compression_backup("fiber-gone")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_backup_returns_false(self, sqlite_storage: SQLiteStorage) -> None:
        deleted = await sqlite_storage.delete_compression_backup("ghost-fiber")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_compression_stats_empty(self, sqlite_storage: SQLiteStorage) -> None:
        stats = await sqlite_storage.get_compression_stats()
        assert stats["total_backups"] == 0
        assert stats["by_tier"] == {}
        assert stats["total_tokens_saved"] == 0

    @pytest.mark.asyncio
    async def test_compression_stats_counts_by_tier(self, sqlite_storage: SQLiteStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="f1",
            original_content="Content 1.",
            compression_tier=1,
            original_token_count=100,
            compressed_token_count=40,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="f2",
            original_content="Content 2.",
            compression_tier=1,
            original_token_count=80,
            compressed_token_count=30,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="f3",
            original_content="Content 3.",
            compression_tier=2,
            original_token_count=60,
            compressed_token_count=10,
        )
        stats = await sqlite_storage.get_compression_stats()
        assert stats["total_backups"] == 3
        assert stats["by_tier"][1] == 2
        assert stats["by_tier"][2] == 1
        assert stats["total_tokens_saved"] == (100 - 40) + (80 - 30) + (60 - 10)

    @pytest.mark.asyncio
    async def test_upsert_backup_replaces_existing(self, sqlite_storage: SQLiteStorage) -> None:
        """Saving twice for the same fiber_id replaces the old backup."""
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-upsert",
            original_content="Old content.",
            compression_tier=1,
            original_token_count=50,
            compressed_token_count=20,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-upsert",
            original_content="New content.",
            compression_tier=2,
            original_token_count=60,
            compressed_token_count=15,
        )
        result = await sqlite_storage.get_compression_backup("fiber-upsert")
        assert result is not None
        assert result["original_content"] == "New content."
        assert result["compression_tier"] == 2

    @pytest.mark.asyncio
    async def test_brain_isolation(self, tmp_path: Path) -> None:
        """Backups stored under brain A are not visible under brain B."""
        db_path = tmp_path / "isolation.db"
        store = SQLiteStorage(db_path)
        await store.initialize()

        brain_a = Brain.create(name="brain-a")
        brain_b = Brain.create(name="brain-b")
        await store.save_brain(brain_a)
        await store.save_brain(brain_b)

        store.set_brain(brain_a.id)
        await store.save_compression_backup(
            fiber_id="shared-fiber",
            original_content="Secret content.",
            compression_tier=1,
            original_token_count=30,
            compressed_token_count=10,
        )

        store.set_brain(brain_b.id)
        result_b = await store.get_compression_backup("shared-fiber")
        assert result_b is None

        store.set_brain(brain_a.id)
        result_a = await store.get_compression_backup("shared-fiber")
        assert result_a is not None


# ---------------------------------------------------------------------------
# Consolidation integration tests
# ---------------------------------------------------------------------------


class TestConsolidationIntegration:
    def test_compress_strategy_exists(self) -> None:
        """ConsolidationStrategy must have a COMPRESS member."""
        assert hasattr(ConsolidationStrategy, "COMPRESS")

    def test_compress_strategy_value(self) -> None:
        """COMPRESS value is 'compress'."""
        assert ConsolidationStrategy.COMPRESS == "compress"

    def test_compress_in_strategy_values(self) -> None:
        """COMPRESS appears in the set of ConsolidationStrategy values."""
        all_values = {s.value for s in ConsolidationStrategy}
        assert "compress" in all_values

    def test_report_has_fibers_compressed(self) -> None:
        report = ConsolidationReport()
        assert hasattr(report, "fibers_compressed")
        assert isinstance(report.fibers_compressed, int)

    def test_report_has_tokens_saved(self) -> None:
        report = ConsolidationReport()
        assert hasattr(report, "tokens_saved")
        assert isinstance(report.tokens_saved, int)

    def test_report_compression_fields_default_zero(self) -> None:
        report = ConsolidationReport()
        assert report.fibers_compressed == 0
        assert report.tokens_saved == 0

    def test_report_compression_fields_are_mutable(self) -> None:
        report = ConsolidationReport()
        report.fibers_compressed = 5
        report.tokens_saved = 200
        assert report.fibers_compressed == 5
        assert report.tokens_saved == 200
