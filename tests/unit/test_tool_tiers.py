"""Tests for MCP tool tier filtering and ToolTierConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.mcp.tool_schemas import (
    TOOL_TIERS,
    get_tool_schemas,
    get_tool_schemas_for_tier,
)
from neural_memory.unified_config import ToolTierConfig


class TestToolTierConfig:
    """Tests for ToolTierConfig dataclass."""

    def test_default_tier_is_full(self) -> None:
        cfg = ToolTierConfig()
        assert cfg.tier == "full"

    def test_from_dict_valid(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "standard"})
        assert cfg.tier == "standard"

    def test_from_dict_minimal(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "minimal"})
        assert cfg.tier == "minimal"

    def test_from_dict_full(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "full"})
        assert cfg.tier == "full"

    def test_from_dict_invalid_defaults_to_full(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "nonexistent"})
        assert cfg.tier == "full"

    def test_from_dict_missing_defaults_to_full(self) -> None:
        cfg = ToolTierConfig.from_dict({})
        assert cfg.tier == "full"

    def test_from_dict_case_insensitive(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "STANDARD"})
        assert cfg.tier == "standard"

    def test_from_dict_strips_whitespace(self) -> None:
        cfg = ToolTierConfig.from_dict({"tier": "  minimal  "})
        assert cfg.tier == "minimal"

    def test_to_dict(self) -> None:
        cfg = ToolTierConfig(tier="standard")
        assert cfg.to_dict() == {"tier": "standard"}

    def test_frozen(self) -> None:
        cfg = ToolTierConfig(tier="full")
        with pytest.raises(AttributeError):
            cfg.tier = "minimal"  # type: ignore[misc]


class TestToolTiers:
    """Tests for tier-based tool filtering."""

    def test_full_tier_returns_all(self) -> None:
        tools = get_tool_schemas_for_tier("full")
        assert len(tools) == 26

    def test_full_tier_matches_get_tool_schemas(self) -> None:
        full = get_tool_schemas_for_tier("full")
        all_tools = get_tool_schemas()
        assert len(full) == len(all_tools)
        assert {t["name"] for t in full} == {t["name"] for t in all_tools}

    def test_standard_tier_count(self) -> None:
        tools = get_tool_schemas_for_tier("standard")
        assert len(tools) == 8

    def test_standard_tier_correct_names(self) -> None:
        tools = get_tool_schemas_for_tier("standard")
        names = {t["name"] for t in tools}
        assert names == {
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_recap",
            "nmem_todo",
            "nmem_session",
            "nmem_auto",
            "nmem_eternal",
        }

    def test_minimal_tier_count(self) -> None:
        tools = get_tool_schemas_for_tier("minimal")
        assert len(tools) == 4

    def test_minimal_tier_correct_names(self) -> None:
        tools = get_tool_schemas_for_tier("minimal")
        names = {t["name"] for t in tools}
        assert names == {
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_recap",
        }

    def test_invalid_tier_defaults_to_full(self) -> None:
        tools = get_tool_schemas_for_tier("bogus")
        assert len(tools) == 26

    def test_tier_hierarchy_minimal_subset_of_standard(self) -> None:
        assert TOOL_TIERS["minimal"] < TOOL_TIERS["standard"]

    def test_tier_hierarchy_standard_subset_of_full(self) -> None:
        all_names = {t["name"] for t in get_tool_schemas()}
        assert TOOL_TIERS["standard"] < all_names

    def test_all_schemas_have_required_fields(self) -> None:
        for tool in get_tool_schemas():
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_get_tool_schemas_returns_copy(self) -> None:
        """Ensure mutations don't affect the original."""
        a = get_tool_schemas()
        b = get_tool_schemas()
        a.pop()
        assert len(b) == 26

    def test_get_tool_schemas_for_tier_returns_copy(self) -> None:
        a = get_tool_schemas_for_tier("standard")
        b = get_tool_schemas_for_tier("standard")
        a.pop()
        assert len(b) == 8


class TestServerTierIntegration:
    """Test that MCPServer.get_tools() respects tier config."""

    def _make_server(self, tier: str) -> MCPServer:  # noqa: F821
        from neural_memory.mcp.server import MCPServer

        with patch("neural_memory.mcp.server.get_config") as mock:
            mock.return_value = MagicMock(
                current_brain="test",
                get_brain_db_path=MagicMock(return_value="/tmp/test.db"),
                tool_tier=ToolTierConfig(tier=tier),
            )
            return MCPServer()

    def test_server_full_tier(self) -> None:
        server = self._make_server("full")
        assert len(server.get_tools()) == 26

    def test_server_standard_tier(self) -> None:
        server = self._make_server("standard")
        assert len(server.get_tools()) == 8

    def test_server_minimal_tier(self) -> None:
        server = self._make_server("minimal")
        assert len(server.get_tools()) == 4

    @pytest.mark.asyncio
    async def test_hidden_tools_still_callable(self) -> None:
        """Tools hidden by tier should still be callable via dispatch."""
        server = self._make_server("minimal")
        tools = server.get_tools()
        exposed_names = {t["name"] for t in tools}
        assert "nmem_stats" not in exposed_names

        with patch.object(server, "_stats", return_value={"status": "ok"}) as mock_stats:
            result = await server.call_tool("nmem_stats", {})
            mock_stats.assert_called_once_with({})
            assert result == {"status": "ok"}


class TestConfigRoundTrip:
    """Test ToolTierConfig save/load round-trip via UnifiedConfig."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(
            data_dir=tmp_path,
            current_brain="default",
            tool_tier=ToolTierConfig(tier="standard"),
        )
        config.save()

        loaded = UnifiedConfig.load(tmp_path / "config.toml")
        assert loaded.tool_tier.tier == "standard"

    def test_save_load_default_tier(self, tmp_path: Path) -> None:
        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path, current_brain="default")
        config.save()

        loaded = UnifiedConfig.load(tmp_path / "config.toml")
        assert loaded.tool_tier.tier == "full"
