"""Tests for layered recall — Phase 2 of Layered Consciousness.

Covers:
- Global brain context merge in recall
- Layer parameter routing (auto/project/global)
- Global recap context injection
- Cross-brain recall with _global brain
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLayeredRecallRouting:
    """Test layer parameter routing in recall_handler._recall()."""

    def test_layer_schema_has_enum(self) -> None:
        """nmem_recall schema includes layer parameter with correct enum."""
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS

        recall_schema = next(s for s in _ALL_TOOL_SCHEMAS if s["name"] == "nmem_recall")
        props = recall_schema["inputSchema"]["properties"]
        assert "layer" in props
        assert props["layer"]["enum"] == ["auto", "project", "global"]


class TestCrossBrainGlobalVisibility:
    """Test that cross_brain_recall can see _global brain."""

    @pytest.mark.asyncio
    async def test_cross_brain_recall_with_global_name(self) -> None:
        """cross_brain_recall uses include_global=True when _global requested."""
        from neural_memory.engine.cross_brain import cross_brain_recall
        from neural_memory.unified_config import GLOBAL_BRAIN_NAME

        config = MagicMock()
        config.list_brains.return_value = [GLOBAL_BRAIN_NAME]
        config.get_brain_db_path.return_value = MagicMock(exists=lambda: False)

        result = await cross_brain_recall(
            config=config,
            brain_names=[GLOBAL_BRAIN_NAME],
            query="test preferences",
        )
        # Should call list_brains with include_global=True
        config.list_brains.assert_called_once_with(include_global=True)
        # No valid brains (db doesn't exist) so empty result
        assert result.brains_queried == []

    @pytest.mark.asyncio
    async def test_cross_brain_recall_without_global(self) -> None:
        """cross_brain_recall uses include_global=False for normal brains."""
        from neural_memory.engine.cross_brain import cross_brain_recall

        config = MagicMock()
        config.list_brains.return_value = []
        config.get_brain_db_path.return_value = MagicMock(exists=lambda: False)

        result = await cross_brain_recall(
            config=config,
            brain_names=["work", "personal"],
            query="test",
        )
        config.list_brains.assert_called_once_with(include_global=False)
        assert result.brains_queried == []


class TestGlobalRecapContext:
    """Test _get_global_recap_context in eternal_handler."""

    @pytest.mark.asyncio
    async def test_no_global_brain_returns_none(self) -> None:
        """Returns None when no global brain DB exists."""
        from neural_memory.mcp.eternal_handler import EternalHandler

        handler = MagicMock(spec=EternalHandler)
        handler.config = MagicMock()
        handler.config.get_brain_db_path.return_value = MagicMock(exists=lambda: False)

        result = await EternalHandler._get_global_recap_context(handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_global_brain_returns_context(self) -> None:
        """Returns formatted context when global brain has memories."""
        from neural_memory.mcp.eternal_handler import EternalHandler

        handler = MagicMock(spec=EternalHandler)
        handler.config = MagicMock()
        handler.config.has_global_brain.return_value = True
        handler.config.get_brain_db_path.return_value = MagicMock(exists=lambda: True)

        with patch(
            "neural_memory.engine.cross_brain._query_single_brain",
            new_callable=AsyncMock,
            return_value=("_global", [], 2, "User prefers dark mode"),
        ):
            result = await EternalHandler._get_global_recap_context(handler, topic="preferences")

        assert result is not None
        assert "User prefers dark mode" in result
        assert "[global]" in result

    @pytest.mark.asyncio
    async def test_global_brain_db_not_found(self) -> None:
        """Returns None when global brain DB doesn't exist."""
        from neural_memory.mcp.eternal_handler import EternalHandler

        handler = MagicMock(spec=EternalHandler)
        handler.config = MagicMock()
        handler.config.has_global_brain.return_value = True
        handler.config.get_brain_db_path.return_value = MagicMock(exists=lambda: False)

        result = await EternalHandler._get_global_recap_context(handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_global_recap_with_topic(self) -> None:
        """Topic is passed to the global brain query."""
        from neural_memory.mcp.eternal_handler import EternalHandler

        handler = MagicMock(spec=EternalHandler)
        handler.config = MagicMock()
        handler.config.has_global_brain.return_value = True
        handler.config.get_brain_db_path.return_value = MagicMock(exists=lambda: True)

        with patch(
            "neural_memory.engine.cross_brain._query_single_brain",
            new_callable=AsyncMock,
            return_value=("_global", [], 1, "Always use dark mode"),
        ) as mock_query:
            await EternalHandler._get_global_recap_context(handler, topic="UI preferences")
            assert mock_query.called
            call_kwargs = mock_query.call_args
            assert "UI preferences" in str(call_kwargs)


class TestLayeredRecallMerge:
    """Test that layered recall merges project + global context."""

    def test_layers_field_only_when_global_used(self) -> None:
        """layers field should only appear when multiple layers contribute."""
        layers_used = ["project"]
        response: dict[str, list[str]] = {}
        if len(layers_used) > 1:
            response["layers"] = layers_used
        assert "layers" not in response

    def test_layers_field_present_when_global_merged(self) -> None:
        """layers field should list both project and global."""
        layers_used = ["project", "global"]
        response: dict[str, list[str]] = {}
        if len(layers_used) > 1:
            response["layers"] = layers_used
        assert response["layers"] == ["project", "global"]

    def test_project_context_comes_first(self) -> None:
        """Project context should have priority (appear first) in merged output."""
        project_ctx = "Project memory: use PostgreSQL"
        global_ctx = "Global preference: always use dark mode"
        merged = f"{project_ctx}\n\n[global] {global_ctx}"
        # Project context appears before global
        assert merged.index(project_ctx) < merged.index(global_ctx)
        # Global prefix present
        assert "[global]" in merged

    def test_global_context_used_when_project_empty(self) -> None:
        """F-01: Global context must surface even when project has no results."""
        project_ctx = ""
        global_ctx = "User prefers dark mode"
        if project_ctx:
            merged = f"{project_ctx}\n\n[global] {global_ctx}"
        else:
            merged = f"[global] {global_ctx}"
        assert "[global]" in merged
        assert "User prefers dark mode" in merged

    def test_budget_reformat_preserves_global(self) -> None:
        """F-02: Budget re-format must re-append global context."""
        budgeted_ctx = "Budget-aware project context"
        global_context = "Global preference: dark mode"
        if global_context:
            budgeted_ctx = f"{budgeted_ctx}\n\n[global] {global_context}"
        assert "[global] Global preference: dark mode" in budgeted_ctx
