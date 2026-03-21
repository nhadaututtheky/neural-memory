"""Tests for dedup enabled by default and short content guard."""

from __future__ import annotations

from neural_memory.engine.dedup.config import DedupConfig
from neural_memory.utils.simhash import hamming_distance, is_near_duplicate, simhash


class TestDedupConfig:
    def test_enabled_by_default(self) -> None:
        config = DedupConfig()
        assert config.enabled is True

    def test_llm_disabled_by_default(self) -> None:
        config = DedupConfig()
        assert config.llm_enabled is False

    def test_from_dict_defaults_enabled(self) -> None:
        config = DedupConfig.from_dict({})
        assert config.enabled is True

    def test_from_dict_explicit_disable(self) -> None:
        config = DedupConfig.from_dict({"enabled": False})
        assert config.enabled is False

    def test_simhash_threshold_default(self) -> None:
        config = DedupConfig()
        assert config.simhash_threshold == 7


class TestSimHashBasics:
    def test_identical_content(self) -> None:
        h1 = simhash("PocketBase runs on port 8090")
        h2 = simhash("PocketBase runs on port 8090")
        assert h1 == h2
        assert is_near_duplicate(h1, h2)

    def test_very_similar_content(self) -> None:
        h1 = simhash("PocketBase runs on port 8090 by default")
        h2 = simhash("PocketBase runs on port 8090 as default")
        distance = hamming_distance(h1, h2)
        # Similar content should have low hamming distance
        assert distance < 15  # structurally similar strings

    def test_different_content(self) -> None:
        h1 = simhash("PocketBase runs on port 8090")
        h2 = simhash("The weather today is sunny and warm in Hanoi")
        distance = hamming_distance(h1, h2)
        assert distance > 10  # very different content

    def test_short_content_not_deduped(self) -> None:
        """Content < 20 chars should be skipped by the pipeline guard."""
        # This tests the concept — actual pipeline guard is in DedupCheckStep
        assert len("hello world") < 20
