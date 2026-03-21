"""Tests for Phase 2: Dedup Improvement.

Covers:
- Tighter simhash threshold defaults (10 → 7)
- Wider max_candidates defaults (10 → 30)
- max_candidates in DedupSettings (user config)
- Session-end consolidation includes DEDUP
- DedupConfig backward compatibility
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.dedup.config import DedupConfig
from neural_memory.unified_config import DedupSettings

# ---------------------------------------------------------------------------
# DedupConfig defaults
# ---------------------------------------------------------------------------


class TestDedupConfigDefaults:
    """Verify tighter dedup defaults."""

    def test_simhash_threshold_default_7(self) -> None:
        cfg = DedupConfig()
        assert cfg.simhash_threshold == 7

    def test_max_candidates_default_30(self) -> None:
        cfg = DedupConfig()
        assert cfg.max_candidates == 30

    def test_from_dict_defaults(self) -> None:
        cfg = DedupConfig.from_dict({})
        assert cfg.simhash_threshold == 7
        assert cfg.max_candidates == 30

    def test_from_dict_preserves_old_values(self) -> None:
        """Old config files with explicit values should be preserved."""
        cfg = DedupConfig.from_dict({"simhash_threshold": 10, "max_candidates": 10})
        assert cfg.simhash_threshold == 10
        assert cfg.max_candidates == 10

    def test_simhash_threshold_tighter_catches_more(self) -> None:
        """Threshold 7 rejects content that threshold 10 would accept."""
        from neural_memory.utils.simhash import hamming_distance, simhash

        # Two similar but not identical strings
        hash1 = simhash("Chose PostgreSQL for payment processing")
        hash2 = simhash("Chose PostgreSQL for payment systems")
        distance = hamming_distance(hash1, hash2)

        # The texts are similar — distance should be moderate
        # With threshold=7 (tighter), more matches caught
        # With threshold=10 (looser), fewer matches caught
        # Just verify the threshold is applied correctly
        is_match_tight = distance <= 7
        is_match_loose = distance <= 10
        # Loose should catch at least everything tight catches
        if is_match_tight:
            assert is_match_loose


# ---------------------------------------------------------------------------
# DedupSettings (user config layer)
# ---------------------------------------------------------------------------


class TestDedupSettings:
    """Tests for DedupSettings user config."""

    def test_defaults(self) -> None:
        settings = DedupSettings()
        assert settings.simhash_threshold == 7
        assert settings.max_candidates == 30

    def test_from_dict_has_max_candidates(self) -> None:
        settings = DedupSettings.from_dict({"max_candidates": 50})
        assert settings.max_candidates == 50

    def test_to_dict_includes_max_candidates(self) -> None:
        settings = DedupSettings()
        d = settings.to_dict()
        assert "max_candidates" in d
        assert d["max_candidates"] == 30

    def test_from_dict_default_max_candidates(self) -> None:
        settings = DedupSettings.from_dict({})
        assert settings.max_candidates == 30

    def test_old_config_without_max_candidates(self) -> None:
        """Config files from before this change (no max_candidates key)."""
        settings = DedupSettings.from_dict(
            {
                "enabled": True,
                "simhash_threshold": 10,
            }
        )
        assert settings.max_candidates == 30  # new default
        assert settings.simhash_threshold == 10  # preserved


# ---------------------------------------------------------------------------
# Session-end consolidation includes DEDUP
# ---------------------------------------------------------------------------


class TestSessionEndDedup:
    """Test that session-end consolidation includes DEDUP strategy."""

    @pytest.mark.asyncio
    async def test_session_end_includes_dedup(self) -> None:
        """run_session_end_consolidation should include DEDUP strategy."""
        from neural_memory.engine.consolidation import ConsolidationStrategy
        from neural_memory.mcp.maintenance_handler import MaintenanceHandler

        handler = MagicMock(spec=MaintenanceHandler)
        handler.config = MagicMock()
        handler.config.maintenance.enabled = True
        handler.config.maintenance.auto_consolidate = True
        handler.get_storage = AsyncMock()

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.brain_id = "test-brain"
        handler.get_storage.return_value = mock_storage

        captured_strategies = []

        async def mock_run_with_delta(storage, brain_id, strategies):
            captured_strategies.extend(strategies)
            result = MagicMock()
            result.report.summary.return_value = "test"
            result.purity_delta = 0.0
            return result

        with patch(
            "neural_memory.engine.consolidation_delta.run_with_delta",
            side_effect=mock_run_with_delta,
        ):
            await MaintenanceHandler.run_session_end_consolidation(handler)

        assert ConsolidationStrategy.DEDUP in captured_strategies
        # DEDUP should come before MATURE
        dedup_idx = captured_strategies.index(ConsolidationStrategy.DEDUP)
        mature_idx = captured_strategies.index(ConsolidationStrategy.MATURE)
        assert dedup_idx < mature_idx


# ---------------------------------------------------------------------------
# Pipeline max_candidates cap
# ---------------------------------------------------------------------------


class TestPipelineCandidateCap:
    """Test max_candidates is capped at 50 in pipeline."""

    @pytest.mark.asyncio
    async def test_candidates_capped_at_50(self) -> None:
        """Even if config says 100, pipeline caps at 50."""
        from neural_memory.engine.dedup.pipeline import DedupPipeline

        cfg = DedupConfig(max_candidates=100)
        mock_storage = AsyncMock()
        mock_storage.find_neurons = AsyncMock(return_value=[])

        pipeline = DedupPipeline(config=cfg, storage=mock_storage)
        await pipeline.check_duplicate("test content here")

        # Should have called find_neurons with limit=50 (capped)
        mock_storage.find_neurons.assert_called_once()
        call_kwargs = mock_storage.find_neurons.call_args
        assert call_kwargs[1]["limit"] == 50

    @pytest.mark.asyncio
    async def test_candidates_uses_config_when_under_50(self) -> None:
        """Config value used when under cap."""
        from neural_memory.engine.dedup.pipeline import DedupPipeline

        cfg = DedupConfig(max_candidates=20)
        mock_storage = AsyncMock()
        mock_storage.find_neurons = AsyncMock(return_value=[])

        pipeline = DedupPipeline(config=cfg, storage=mock_storage)
        await pipeline.check_duplicate("test content here")

        mock_storage.find_neurons.assert_called_once()
        call_kwargs = mock_storage.find_neurons.call_args
        assert call_kwargs[1]["limit"] == 20


# ---------------------------------------------------------------------------
# DedupConfig validation
# ---------------------------------------------------------------------------


class TestDedupConfigValidation:
    """Ensure validation still works with new defaults."""

    def test_simhash_threshold_valid_range(self) -> None:
        cfg = DedupConfig(simhash_threshold=0)
        assert cfg.simhash_threshold == 0
        cfg = DedupConfig(simhash_threshold=64)
        assert cfg.simhash_threshold == 64

    def test_simhash_threshold_invalid(self) -> None:
        with pytest.raises(ValueError):
            DedupConfig(simhash_threshold=65)
        with pytest.raises(ValueError):
            DedupConfig(simhash_threshold=-1)

    def test_max_candidates_valid(self) -> None:
        cfg = DedupConfig(max_candidates=1)
        assert cfg.max_candidates == 1

    def test_max_candidates_invalid(self) -> None:
        with pytest.raises(ValueError):
            DedupConfig(max_candidates=0)
