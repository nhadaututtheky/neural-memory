"""Tests for BrainSettings extras pass-through (issue #168).

`config.toml [brain]` keys that map onto `BrainConfig` fields added after
`BrainSettings` was first defined (e.g. `bm25_enabled`,
`high_signal_memory_boost`, `creation_recency_boost`) must reach the
constructed `BrainConfig` instead of being silently dropped.
"""

from __future__ import annotations

from neural_memory.core.brain import BrainConfig
from neural_memory.unified_config import BrainSettings, EmbeddingSettings


class TestBrainSettingsExtras:
    """Verify [brain] keys outside the historical 7 land in BrainConfig."""

    def test_extras_captured_from_dict(self) -> None:
        settings = BrainSettings.from_dict(
            {
                "decay_rate": 0.2,
                "bm25_enabled": True,
                "high_signal_memory_boost": 1.5,
                "creation_recency_boost": 0.1,
                "creation_recency_halflife_hrs": 48.0,
            }
        )
        assert settings.decay_rate == 0.2
        assert settings.extras == {
            "bm25_enabled": True,
            "high_signal_memory_boost": 1.5,
            "creation_recency_boost": 0.1,
            "creation_recency_halflife_hrs": 48.0,
        }

    def test_to_dict_round_trips_extras(self) -> None:
        original = {
            "decay_rate": 0.15,
            "bm25_enabled": True,
            "bm25_tokenizer": "vietnamese",
            "goal_proximity_boost": 0.5,
        }
        roundtrip = BrainSettings.from_dict(original).to_dict()
        assert roundtrip["bm25_enabled"] is True
        assert roundtrip["bm25_tokenizer"] == "vietnamese"
        assert roundtrip["goal_proximity_boost"] == 0.5
        assert roundtrip["decay_rate"] == 0.15

    def test_unknown_extras_dropped_when_building_brainconfig(self) -> None:
        settings = BrainSettings.from_dict(
            {
                "bm25_enabled": True,
                "this_field_does_not_exist": "garbage",
                "another_typo": 42,
            }
        )
        # Should not raise — unknown keys are filtered against BrainConfig fields
        config = BrainConfig(**settings.to_brain_config_kwargs())
        assert config.bm25_enabled is True

    def test_brainconfig_built_with_all_known_extras(self) -> None:
        settings = BrainSettings.from_dict(
            {
                "bm25_enabled": True,
                "bm25_tokenizer": "vietnamese",
                "high_signal_memory_boost": 1.5,
                "creation_recency_boost": 0.1,
                "creation_recency_halflife_hrs": 48.0,
                "goal_proximity_boost": 0.4,
                "anti_redundancy_penalty": 0.6,
            }
        )
        config = BrainConfig(**settings.to_brain_config_kwargs())
        assert config.bm25_enabled is True
        assert config.bm25_tokenizer == "vietnamese"
        assert config.high_signal_memory_boost == 1.5
        assert config.creation_recency_boost == 0.1
        assert config.creation_recency_halflife_hrs == 48.0
        assert config.goal_proximity_boost == 0.4
        assert config.anti_redundancy_penalty == 0.6

    def test_embedding_settings_merged_into_kwargs(self) -> None:
        settings = BrainSettings.from_dict({"bm25_enabled": True})
        embedding = EmbeddingSettings(
            enabled=True,
            provider="sentence_transformer",
            model="all-MiniLM-L6-v2",
            similarity_threshold=0.8,
        )
        kwargs = settings.to_brain_config_kwargs(embedding)
        assert kwargs["embedding_enabled"] is True
        assert kwargs["embedding_provider"] == "sentence_transformer"
        assert kwargs["embedding_similarity_threshold"] == 0.8
        assert kwargs["bm25_enabled"] is True

    def test_runtime_overrides_excludes_explicit_fields(self) -> None:
        """Runtime overrides for upgrade migration only touch new extras keys.

        Explicit 7 fields may have per-brain customizations on legacy brains;
        we must not clobber those just because config.toml has a default for
        the same key.
        """
        settings = BrainSettings.from_dict(
            {
                "decay_rate": 0.9,  # explicit — excluded
                "bm25_enabled": True,  # extras — included
                "high_signal_memory_boost": 2.0,  # extras — included
            }
        )
        overrides = settings.runtime_overrides()
        assert "decay_rate" not in overrides
        assert overrides == {
            "bm25_enabled": True,
            "high_signal_memory_boost": 2.0,
        }

    def test_runtime_overrides_filters_unknown_keys(self) -> None:
        settings = BrainSettings.from_dict(
            {
                "bm25_enabled": True,
                "totally_made_up_field": "ignore me",
            }
        )
        overrides = settings.runtime_overrides()
        assert overrides == {"bm25_enabled": True}

    def test_empty_extras_means_empty_overrides(self) -> None:
        settings = BrainSettings()
        assert settings.runtime_overrides() == {}

    def test_default_brainconfig_unaffected_by_empty_brainsettings(self) -> None:
        """Existing behaviour preserved: empty BrainSettings → BrainConfig defaults."""
        config = BrainConfig(**BrainSettings().to_brain_config_kwargs())
        defaults = BrainConfig()
        # The kwargs path explicitly sets the 7 BrainSettings fields,
        # but those defaults match BrainConfig's own defaults — no drift.
        assert config.decay_rate == defaults.decay_rate
        assert config.max_spread_hops == defaults.max_spread_hops
        assert config.freshness_weight == defaults.freshness_weight
        # Extras-driven fields keep BrainConfig defaults
        assert config.bm25_enabled == defaults.bm25_enabled
        assert config.high_signal_memory_boost == defaults.high_signal_memory_boost
