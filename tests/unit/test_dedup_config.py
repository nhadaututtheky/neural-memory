"""Tests for dedup configuration."""

from __future__ import annotations

import pytest

from neural_memory.engine.dedup.config import DedupConfig


class TestDedupConfig:
    def test_defaults_enabled(self) -> None:
        cfg = DedupConfig()
        assert cfg.enabled is True
        assert cfg.simhash_threshold == 7
        assert cfg.max_candidates == 30
        assert cfg.embedding_threshold == 0.85
        assert cfg.embedding_ambiguous_low == 0.75
        assert cfg.llm_enabled is False
        assert cfg.llm_provider == "none"
        assert cfg.merge_strategy == "keep_newer"

    def test_validation_ambiguous_low_gt_threshold(self) -> None:
        with pytest.raises(ValueError, match="embedding_ambiguous_low"):
            DedupConfig(embedding_ambiguous_low=0.9, embedding_threshold=0.8)

    def test_validation_embedding_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="embedding_threshold"):
            DedupConfig(embedding_threshold=1.5)

    def test_validation_ambiguous_low_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="embedding_ambiguous_low"):
            DedupConfig(embedding_ambiguous_low=-0.1)

    def test_to_dict_roundtrip(self) -> None:
        cfg = DedupConfig(
            enabled=True,
            simhash_threshold=8,
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        d = cfg.to_dict()
        restored = DedupConfig.from_dict(d)
        assert restored.enabled is True
        assert restored.simhash_threshold == 8
        assert restored.llm_enabled is True
        assert restored.llm_provider == "openai"
        assert restored.llm_model == "gpt-4o-mini"

    def test_from_dict_defaults(self) -> None:
        cfg = DedupConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.simhash_threshold == 7

    def test_frozen(self) -> None:
        cfg = DedupConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]
