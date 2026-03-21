"""Integration tests for dedup with unified config and consolidation."""

from __future__ import annotations

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.consolidation import ConsolidationStrategy
from neural_memory.engine.dedup.prompts import DEDUP_SYSTEM_PROMPT, format_dedup_prompt
from neural_memory.unified_config import DedupSettings


class TestDedupSettings:
    def test_defaults_disabled(self) -> None:
        cfg = DedupSettings()
        assert cfg.enabled is False
        assert cfg.llm_enabled is False
        assert cfg.llm_provider == "none"

    def test_to_dict_roundtrip(self) -> None:
        cfg = DedupSettings(
            enabled=True,
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o",
        )
        d = cfg.to_dict()
        restored = DedupSettings.from_dict(d)
        assert restored.enabled is True
        assert restored.llm_enabled is True
        assert restored.llm_provider == "openai"
        assert restored.llm_model == "gpt-4o"

    def test_from_dict_defaults(self) -> None:
        cfg = DedupSettings.from_dict({})
        assert cfg.enabled is False
        assert cfg.simhash_threshold == 7


class TestConsolidationDedupStrategy:
    def test_dedup_strategy_exists(self) -> None:
        assert ConsolidationStrategy.DEDUP == "dedup"
        assert "dedup" in [s.value for s in ConsolidationStrategy]


class TestAliasSynapseType:
    def test_alias_type_exists(self) -> None:
        assert SynapseType.ALIAS == "alias"
        assert "alias" in [s.value for s in SynapseType]


class TestDedupPrompts:
    def test_system_prompt_not_empty(self) -> None:
        assert len(DEDUP_SYSTEM_PROMPT) > 50

    def test_format_user_prompt(self) -> None:
        result = format_dedup_prompt("Memory A content", "Memory B content")
        assert "Memory A content" in result
        assert "Memory B content" in result

    def test_format_truncates_long_content(self) -> None:
        long_content = "x" * 1000
        result = format_dedup_prompt(long_content, "short")
        assert len(result) < 1500  # Should be truncated
