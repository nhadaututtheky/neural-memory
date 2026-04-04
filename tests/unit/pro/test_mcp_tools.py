"""Tests for Pro MCP tool schemas and handlers."""

from __future__ import annotations

from neural_memory.pro.mcp_tools import PRO_TOOL_SCHEMAS, TOOL_HANDLERS


class TestToolSchemas:
    def test_three_pro_tools(self) -> None:
        assert len(PRO_TOOL_SCHEMAS) == 3

    def test_tool_names(self) -> None:
        names = {t["name"] for t in PRO_TOOL_SCHEMAS}
        assert names == {"nmem_cone_query", "nmem_tier_info", "nmem_pro_merge"}

    def test_all_have_input_schema(self) -> None:
        for tool in PRO_TOOL_SCHEMAS:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_all_have_description(self) -> None:
        for tool in PRO_TOOL_SCHEMAS:
            assert "description" in tool
            assert len(tool["description"]) > 20


class TestToolHandlers:
    def test_handlers_match_schemas(self) -> None:
        schema_names = {t["name"] for t in PRO_TOOL_SCHEMAS}
        handler_names = set(TOOL_HANDLERS.keys())
        assert schema_names == handler_names

    def test_handlers_are_callable(self) -> None:
        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"{name} handler is not callable"
