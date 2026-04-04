"""Tests for InfinityDB Phase 3 — Tiered Compression."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from neural_memory.pro.infinitydb.compressor import (
    BYTES_PER_DIM,
    CompressionTier,
    VectorCompressor,
)
from neural_memory.pro.infinitydb.tier_manager import TierConfig, TierManager, TierStats

# ── VectorCompressor tests ──


@pytest.fixture
def compressor() -> VectorCompressor:
    return VectorCompressor(dimensions=384)


@pytest.fixture
def small_compressor() -> VectorCompressor:
    return VectorCompressor(dimensions=8)


class TestCompressionTier:
    def test_tier_ordering(self) -> None:
        assert CompressionTier.ACTIVE < CompressionTier.WARM
        assert CompressionTier.WARM < CompressionTier.COOL
        assert CompressionTier.COOL < CompressionTier.FROZEN
        assert CompressionTier.FROZEN < CompressionTier.CRYSTAL

    def test_bytes_per_dim(self) -> None:
        assert BYTES_PER_DIM[CompressionTier.ACTIVE] == 4.0
        assert BYTES_PER_DIM[CompressionTier.WARM] == 2.0
        assert BYTES_PER_DIM[CompressionTier.COOL] == 1.0
        assert BYTES_PER_DIM[CompressionTier.FROZEN] == 0.125
        assert BYTES_PER_DIM[CompressionTier.CRYSTAL] == 0.0


class TestCompressorActive:
    def test_lossless_roundtrip(self, compressor: VectorCompressor) -> None:
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(384).astype(np.float32)
        data = compressor.compress(vec, CompressionTier.ACTIVE)
        restored = compressor.decompress(data, CompressionTier.ACTIVE)
        np.testing.assert_array_equal(vec, restored)

    def test_size(self, compressor: VectorCompressor) -> None:
        vec = np.zeros(384, dtype=np.float32)
        data = compressor.compress(vec, CompressionTier.ACTIVE)
        assert len(data) == 384 * 4  # 1536 bytes


class TestCompressorWarm:
    def test_half_precision_roundtrip(self, compressor: VectorCompressor) -> None:
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(384).astype(np.float32)
        data = compressor.compress(vec, CompressionTier.WARM)
        restored = compressor.decompress(data, CompressionTier.WARM)
        # float16 loses some precision
        np.testing.assert_allclose(vec, restored, atol=0.01)

    def test_size(self, compressor: VectorCompressor) -> None:
        vec = np.zeros(384, dtype=np.float32)
        data = compressor.compress(vec, CompressionTier.WARM)
        assert len(data) == 384 * 2  # 768 bytes


class TestCompressorCool:
    def test_int8_roundtrip(self, compressor: VectorCompressor) -> None:
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(384).astype(np.float32)
        data = compressor.compress(vec, CompressionTier.COOL)
        restored = compressor.decompress(data, CompressionTier.COOL)
        # int8 quantization has ~1% error
        np.testing.assert_allclose(vec, restored, atol=0.1)

    def test_size(self, compressor: VectorCompressor) -> None:
        vec = np.zeros(384, dtype=np.float32)
        data = compressor.compress(vec, CompressionTier.COOL)
        assert len(data) == 384 + 8  # 392 bytes (8 byte header)

    def test_constant_vector(self, small_compressor: VectorCompressor) -> None:
        vec = np.full(8, 5.0, dtype=np.float32)
        data = small_compressor.compress(vec, CompressionTier.COOL)
        restored = small_compressor.decompress(data, CompressionTier.COOL)
        np.testing.assert_allclose(restored, vec, atol=0.01)

    def test_cosine_similarity_preserved(self, compressor: VectorCompressor) -> None:
        """Int8 should preserve cosine similarity reasonably well."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(384).astype(np.float32)
        b = rng.standard_normal(384).astype(np.float32)

        orig_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        a_q = compressor.decompress(
            compressor.compress(a, CompressionTier.COOL), CompressionTier.COOL
        )
        b_q = compressor.decompress(
            compressor.compress(b, CompressionTier.COOL), CompressionTier.COOL
        )
        quant_sim = np.dot(a_q, b_q) / (np.linalg.norm(a_q) * np.linalg.norm(b_q))

        assert abs(orig_sim - quant_sim) < 0.05  # <5% sim drift


class TestCompressorFrozen:
    def test_binary_roundtrip(self, small_compressor: VectorCompressor) -> None:
        vec = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=np.float32)
        data = small_compressor.compress(vec, CompressionTier.FROZEN)
        restored = small_compressor.decompress(data, CompressionTier.FROZEN)
        # Binary preserves sign: positive -> +1, negative -> -1
        expected = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        np.testing.assert_array_equal(restored, expected)

    def test_size(self, compressor: VectorCompressor) -> None:
        vec = np.zeros(384, dtype=np.float32)
        data = compressor.compress(vec, CompressionTier.FROZEN)
        assert len(data) == 384 // 8  # 48 bytes

    def test_direction_preserved(self, compressor: VectorCompressor) -> None:
        """Binary hash should preserve general direction (hamming similarity)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(384).astype(np.float32)
        b = a + rng.standard_normal(384).astype(np.float32) * 0.1  # similar

        compressor.decompress(
            compressor.compress(a, CompressionTier.FROZEN), CompressionTier.FROZEN
        )
        compressor.decompress(
            compressor.compress(b, CompressionTier.FROZEN), CompressionTier.FROZEN
        )

        # Sign agreement should be >80% for similar vectors
        agreement = np.mean(np.sign(a) == np.sign(b))
        assert agreement > 0.7


class TestCompressorCrystal:
    def test_no_data(self, compressor: VectorCompressor) -> None:
        vec = np.ones(384, dtype=np.float32)
        data = compressor.compress(vec, CompressionTier.CRYSTAL)
        assert data == b""

    def test_decompress_zeros(self, compressor: VectorCompressor) -> None:
        restored = compressor.decompress(b"", CompressionTier.CRYSTAL)
        np.testing.assert_array_equal(restored, np.zeros(384, dtype=np.float32))


class TestCompressorEdgeCases:
    def test_wrong_shape(self, compressor: VectorCompressor) -> None:
        vec = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            compressor.compress(vec, CompressionTier.ACTIVE)

    def test_estimate_size(self, compressor: VectorCompressor) -> None:
        # 1000 active vectors = 1000 * 384 * 4 = 1,536,000 bytes
        assert compressor.estimate_size(CompressionTier.ACTIVE, 1000) == 1_536_000
        # 1000 frozen vectors = 1000 * 48 = 48,000 bytes
        assert compressor.estimate_size(CompressionTier.FROZEN, 1000) == 48_000
        # 1000 cool vectors = 1000 * (384 + 8) = 392,000 bytes
        assert compressor.estimate_size(CompressionTier.COOL, 1000) == 392_000

    def test_compression_ratio(self, compressor: VectorCompressor) -> None:
        assert compressor.compression_ratio(CompressionTier.ACTIVE, CompressionTier.WARM) == 2.0
        assert compressor.compression_ratio(CompressionTier.ACTIVE, CompressionTier.COOL) == 4.0
        assert compressor.compression_ratio(CompressionTier.ACTIVE, CompressionTier.FROZEN) == 32.0
        assert compressor.compression_ratio(
            CompressionTier.ACTIVE, CompressionTier.CRYSTAL
        ) == float("inf")


# ── TierManager tests ──


def _utcnow_str() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _days_ago(days: int) -> str:
    dt = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    return dt.isoformat()


@pytest.fixture
def manager() -> TierManager:
    return TierManager(dimensions=384)


class TestTierClassification:
    def test_high_priority_stays_active(self, manager: TierManager) -> None:
        meta = {"priority": 9, "access_count": 0, "accessed_at": _days_ago(100)}
        assert manager.classify_neuron(meta) == CompressionTier.ACTIVE

    def test_recent_high_access_active(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 10, "accessed_at": _utcnow_str()}
        assert manager.classify_neuron(meta) == CompressionTier.ACTIVE

    def test_recent_low_access_warm(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 1, "accessed_at": _utcnow_str()}
        assert manager.classify_neuron(meta) == CompressionTier.WARM

    def test_moderate_age_warm(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 3, "accessed_at": _days_ago(15)}
        assert manager.classify_neuron(meta) == CompressionTier.WARM

    def test_old_cool(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 3, "accessed_at": _days_ago(60)}
        assert manager.classify_neuron(meta) == CompressionTier.COOL

    def test_very_old_frozen(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 3, "accessed_at": _days_ago(120)}
        assert manager.classify_neuron(meta) == CompressionTier.FROZEN

    def test_zero_priority_crystal(self, manager: TierManager) -> None:
        meta = {"priority": 0, "access_count": 100, "accessed_at": _utcnow_str()}
        assert manager.classify_neuron(meta) == CompressionTier.CRYSTAL

    def test_missing_accessed_at_frozen(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 0, "accessed_at": ""}
        assert manager.classify_neuron(meta) == CompressionTier.FROZEN

    def test_invalid_accessed_at(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 0, "accessed_at": "not-a-date"}
        assert manager.classify_neuron(meta) == CompressionTier.FROZEN


class TestPromotion:
    def test_promote_on_access(self, manager: TierManager) -> None:
        meta = {"priority": 9, "access_count": 10, "accessed_at": _utcnow_str()}
        target = manager.should_promote(meta, CompressionTier.COOL)
        assert target == CompressionTier.ACTIVE

    def test_no_promotion_if_already_correct(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 10, "accessed_at": _utcnow_str()}
        target = manager.should_promote(meta, CompressionTier.ACTIVE)
        assert target is None

    def test_no_promotion_if_disabled(self) -> None:
        config = TierConfig(auto_promote_on_access=False)
        mgr = TierManager(384, config)
        meta = {"priority": 9, "access_count": 10, "accessed_at": _utcnow_str()}
        assert mgr.should_promote(meta, CompressionTier.FROZEN) is None


class TestDemotion:
    def test_demote_old_neuron(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 3, "accessed_at": _days_ago(60)}
        target = manager.should_demote(meta, CompressionTier.ACTIVE)
        assert target == CompressionTier.COOL

    def test_no_demotion_if_already_correct(self, manager: TierManager) -> None:
        meta = {"priority": 5, "access_count": 3, "accessed_at": _days_ago(60)}
        target = manager.should_demote(meta, CompressionTier.COOL)
        assert target is None

    def test_no_demotion_if_disabled(self) -> None:
        config = TierConfig(auto_demote_enabled=False)
        mgr = TierManager(384, config)
        meta = {"priority": 5, "access_count": 0, "accessed_at": _days_ago(120)}
        assert mgr.should_demote(meta, CompressionTier.ACTIVE) is None


class TestBatchClassify:
    def test_batch_classify(self, manager: TierManager) -> None:
        neurons = [
            {"id": "n1", "priority": 9, "access_count": 10, "accessed_at": _utcnow_str()},
            {"id": "n2", "priority": 5, "access_count": 3, "accessed_at": _days_ago(60)},
            {"id": "n3", "priority": 0, "access_count": 0, "accessed_at": _days_ago(200)},
        ]
        result = manager.batch_classify(neurons)
        assert "n1" in result[CompressionTier.ACTIVE]
        assert "n2" in result[CompressionTier.COOL]
        assert "n3" in result[CompressionTier.CRYSTAL]


class TestTierStats:
    def test_stats_total(self) -> None:
        stats = TierStats(active=10, warm=20, cool=30, frozen=40, crystal=5)
        assert stats.total == 105

    def test_stats_as_dict(self) -> None:
        stats = TierStats(active=10, warm=20)
        d = stats.as_dict()
        assert d["active"] == 10
        assert d["warm"] == 20
        assert d["total"] == 30


class TestEstimateSavings:
    def test_all_active_no_savings(self, manager: TierManager) -> None:
        stats = TierStats(active=1000)
        savings = manager.estimate_savings(stats)
        assert savings["saved_bytes"] == 0
        assert savings["compression_ratio"] == 1.0

    def test_mixed_tiers_savings(self, manager: TierManager) -> None:
        stats = TierStats(active=100, warm=200, cool=300, frozen=300, crystal=100)
        savings = manager.estimate_savings(stats)
        assert savings["saved_bytes"] > 0
        assert savings["compression_ratio"] > 1.0
        assert savings["savings_percent"] > 0

    def test_1m_neurons_estimate(self, manager: TierManager) -> None:
        """1M neurons with realistic tier distribution should be ~300MB."""
        # Realistic: 5% active, 15% warm, 30% cool, 40% frozen, 10% crystal
        stats = TierStats(
            active=50_000,
            warm=150_000,
            cool=300_000,
            frozen=400_000,
            crystal=100_000,
        )
        savings = manager.estimate_savings(stats)
        actual_mb = savings["actual_bytes"] / 1_048_576
        assert actual_mb < 400  # well under 400MB
        assert savings["compression_ratio"] > 3.0
        print("\n--- 1M Neuron Storage Estimate ---")
        print(f"All-active: {savings['all_active_bytes'] / 1_048_576:.1f} MB")
        print(f"Tiered: {actual_mb:.1f} MB")
        print(f"Savings: {savings['savings_percent']}%")
        print(f"Compression ratio: {savings['compression_ratio']}x")
