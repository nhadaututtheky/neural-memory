"""Tests for Knowledge Surface MCP integration.

Tests surface loading, instruction injection, depth-aware recall routing,
brain switch reload, and resolver path resolution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.surface.models import (
    Cluster,
    DepthHint,
    DepthLevel,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceEdge,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)
from neural_memory.surface.resolver import (
    detect_project_root,
    get_surface_path,
    load_surface_text,
    save_surface_text,
)
from neural_memory.surface.serializer import serialize

# ── Fixtures ───────────────────────────────────────────


def _make_surface() -> KnowledgeSurface:
    """Build a minimal test surface with all sections populated."""
    return KnowledgeSurface(
        frontmatter=SurfaceFrontmatter(
            brain="testbrain",
            updated="2026-03-16T10:00:00",
            neurons=100,
            synapses=300,
            token_budget=1200,
        ),
        graph=(
            GraphEntry(
                node=SurfaceNode(
                    id="d1", content="Chose PostgreSQL", node_type="decision", priority=8
                ),
                edges=(SurfaceEdge(edge_type="caused", target_id="f1", target_text="ACID needed"),),
            ),
            GraphEntry(
                node=SurfaceNode(
                    id="f1", content="Payment transactions", node_type="fact", priority=6
                ),
                edges=(),
            ),
        ),
        clusters=(
            Cluster(name="payments", node_ids=("d1", "f1"), description="Payment processing"),
        ),
        signals=(
            Signal(level=SignalLevel.URGENT, node_id="f1", text="Payment migration deadline"),
        ),
        depth_map=(
            DepthHint(node_id="d1", level=DepthLevel.SUFFICIENT, context="8 synapses"),
            DepthHint(
                node_id="f1", level=DepthLevel.NEEDS_DEEP, context="recall payment transactions"
            ),
        ),
        meta=SurfaceMeta(coverage=0.7, staleness=0.1),
    )


def _make_surface_text() -> str:
    """Serialize the test surface to .nm text."""
    return serialize(_make_surface())


# ── Resolver Tests ─────────────────────────────────────


class TestResolver:
    """Tests for surface path resolution and file I/O."""

    def test_global_surface_path(self, tmp_path: Path) -> None:
        """Global surface path uses brain name."""
        with (
            patch("neural_memory.surface.resolver.detect_project_root", return_value=None),
            patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=tmp_path),
        ):
            path = get_surface_path("myproject")
            assert path == tmp_path / "surfaces" / "myproject.nm"

    def test_project_surface_takes_priority(self, tmp_path: Path) -> None:
        """Project-level surface.nm takes priority over global."""
        project_dir = tmp_path / "myproject"
        nm_dir = project_dir / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        surface_file = nm_dir / "surface.nm"
        surface_file.write_text("test", encoding="utf-8")

        with (
            patch("neural_memory.surface.resolver.detect_project_root", return_value=project_dir),
            patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=tmp_path),
        ):
            path = get_surface_path("default")
            assert path == surface_file

    def test_project_surface_for_write_creates_at_project_level(self, tmp_path: Path) -> None:
        """for_write=True returns project path even when file doesn't exist yet."""
        project_dir = tmp_path / "myproject"
        nm_dir = project_dir / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        # Note: surface.nm does NOT exist yet

        with (
            patch("neural_memory.surface.resolver.detect_project_root", return_value=project_dir),
            patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=tmp_path),
        ):
            path = get_surface_path("default", for_write=True)
            assert path == nm_dir / "surface.nm"

    def test_project_surface_read_falls_back_to_global(self, tmp_path: Path) -> None:
        """Read mode falls back to global when project surface doesn't exist."""
        project_dir = tmp_path / "myproject"
        nm_dir = project_dir / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        # Note: surface.nm does NOT exist

        with (
            patch("neural_memory.surface.resolver.detect_project_root", return_value=project_dir),
            patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=tmp_path),
        ):
            path = get_surface_path("default")
            assert path == tmp_path / "surfaces" / "default.nm"

    def test_load_surface_text_returns_none_when_missing(self, tmp_path: Path) -> None:
        """load_surface_text returns None when file doesn't exist."""
        with patch(
            "neural_memory.surface.resolver.get_surface_path", return_value=tmp_path / "nope.nm"
        ):
            result = load_surface_text("default")
            assert result is None

    def test_load_surface_text_reads_file(self, tmp_path: Path) -> None:
        """load_surface_text reads existing file content."""
        nm_file = tmp_path / "test.nm"
        nm_file.write_text("hello surface", encoding="utf-8")

        with patch("neural_memory.surface.resolver.get_surface_path", return_value=nm_file):
            result = load_surface_text("default")
            assert result == "hello surface"

    def test_save_surface_text_creates_directories(self, tmp_path: Path) -> None:
        """save_surface_text creates parent directories."""
        target = tmp_path / "deep" / "path" / "brain.nm"
        with patch("neural_memory.surface.resolver.get_surface_path", return_value=target):
            result = save_surface_text("content here", "mybrain")
            assert result == target
            assert target.read_text(encoding="utf-8") == "content here"

    def test_save_surface_text_overwrites_existing(self, tmp_path: Path) -> None:
        """save_surface_text overwrites existing file."""
        target = tmp_path / "brain.nm"
        target.write_text("old content", encoding="utf-8")

        with patch("neural_memory.surface.resolver.get_surface_path", return_value=target):
            save_surface_text("new content", "default")
            assert target.read_text(encoding="utf-8") == "new content"

    def test_detect_project_root_finds_git(self, tmp_path: Path) -> None:
        """detect_project_root finds .git directory."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        with patch("neural_memory.surface.resolver.Path.cwd", return_value=tmp_path):
            result = detect_project_root()
            assert result == tmp_path

    def test_detect_project_root_returns_none_at_home(self, tmp_path: Path) -> None:
        """detect_project_root returns None when no markers found."""
        with (
            patch("neural_memory.surface.resolver.Path.cwd", return_value=tmp_path),
            patch("neural_memory.surface.resolver.Path.home", return_value=tmp_path),
        ):
            result = detect_project_root()
            assert result is None


# ── Server Surface Loading Tests ───────────────────────


class TestServerSurfaceLoading:
    """Tests for MCP server surface loading and instructions injection."""

    def test_load_surface_caches_by_brain(self) -> None:
        """load_surface caches result and doesn't re-read for same brain."""
        from neural_memory.mcp.server import MCPServer

        server = MCPServer()
        surface_text = _make_surface_text()

        with patch(
            "neural_memory.surface.resolver.load_surface_text", return_value=surface_text
        ) as mock_load:
            # First load
            result1 = server.load_surface("testbrain")
            assert result1 == surface_text
            assert mock_load.call_count == 1

            # Second load — cached, no re-read
            result2 = server.load_surface("testbrain")
            assert result2 == surface_text
            assert mock_load.call_count == 1

    def test_load_surface_reloads_on_brain_change(self) -> None:
        """load_surface reloads when brain name changes."""
        from neural_memory.mcp.server import MCPServer

        server = MCPServer()

        with patch(
            "neural_memory.surface.resolver.load_surface_text", return_value="brain1 content"
        ) as mock_load:
            server.load_surface("brain1")
            assert mock_load.call_count == 1

        with patch(
            "neural_memory.surface.resolver.load_surface_text", return_value="brain2 content"
        ) as mock_load:
            result = server.load_surface("brain2")
            assert result == "brain2 content"
            assert mock_load.call_count == 1

    def test_load_surface_returns_empty_when_missing(self) -> None:
        """load_surface returns empty string when file doesn't exist."""
        from neural_memory.mcp.server import MCPServer

        server = MCPServer()

        with patch("neural_memory.surface.resolver.load_surface_text", return_value=None):
            result = server.load_surface("missing")
            assert result == ""

    @pytest.mark.asyncio
    async def test_initialize_injects_surface(self) -> None:
        """initialize response includes surface in instructions."""
        from neural_memory.mcp.server import MCPServer, handle_message

        server = MCPServer()
        surface_text = _make_surface_text()

        with patch.object(server, "load_surface", return_value=surface_text):
            response = await handle_message(server, {"method": "initialize", "id": 1, "params": {}})

        instructions = response["result"]["instructions"]
        assert "## Knowledge Surface" in instructions
        assert "Chose PostgreSQL" in instructions

    @pytest.mark.asyncio
    async def test_initialize_works_without_surface(self) -> None:
        """initialize response works when no surface exists."""
        from neural_memory.mcp.server import MCPServer, handle_message

        server = MCPServer()

        with patch.object(server, "load_surface", return_value=""):
            response = await handle_message(server, {"method": "initialize", "id": 1, "params": {}})

        instructions = response["result"]["instructions"]
        assert "## Knowledge Surface" not in instructions
        assert "Neural Memory" in instructions  # Base instructions still present


# ── Depth-Aware Recall Routing Tests ───────────────────


class TestDepthRouting:
    """Tests for surface-based depth routing in recall."""

    def _make_handler_with_surface(self, surface_text: str) -> MagicMock:
        """Create a mock ToolHandler with surface loaded."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._surface_text = surface_text
        handler._surface_brain = "testbrain"
        handler._check_surface_depth = ToolHandler._check_surface_depth.__get__(handler)
        return handler

    def test_sufficient_entity_returns_surface_context(self) -> None:
        """SUFFICIENT entity answered from surface without brain.db."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, depth = handler._check_surface_depth("Chose PostgreSQL")
        assert response is not None
        assert response["source"] == "knowledge_surface"
        assert response["depth_hint"] == "SUFFICIENT"
        assert "Chose PostgreSQL" in response["answer"]
        assert depth is None

    def test_needs_deep_entity_returns_depth_override(self) -> None:
        """NEEDS_DEEP entity returns depth=2 override."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, depth = handler._check_surface_depth("Payment transactions")
        assert response is None
        assert depth == 2

    def test_unknown_entity_returns_none(self) -> None:
        """Unknown entity returns no routing."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, depth = handler._check_surface_depth("something completely different")
        assert response is None
        assert depth is None

    def test_empty_surface_returns_none(self) -> None:
        """No surface loaded returns no routing."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._surface_text = ""
        handler._check_surface_depth = ToolHandler._check_surface_depth.__get__(handler)

        response, depth = handler._check_surface_depth("anything")
        assert response is None
        assert depth is None

    def test_sufficient_includes_cluster_context(self) -> None:
        """SUFFICIENT response includes cluster info when available."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, _ = handler._check_surface_depth("Chose PostgreSQL")
        assert response is not None
        assert "@payments" in response["answer"]
        assert "Payment processing" in response["answer"]

    def test_sufficient_includes_edge_context(self) -> None:
        """SUFFICIENT response includes edges from the matched node."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, _ = handler._check_surface_depth("PostgreSQL")
        assert response is not None
        assert "ACID needed" in response["answer"]
        assert "caused" in response["answer"]

    def test_case_insensitive_matching(self) -> None:
        """Entity matching is case-insensitive."""
        surface_text = _make_surface_text()
        handler = self._make_handler_with_surface(surface_text)

        response, _ = handler._check_surface_depth("chose postgresql")
        assert response is not None
        assert response["source"] == "knowledge_surface"

    def test_corrupt_surface_returns_none(self) -> None:
        """Corrupt surface text returns no routing (graceful degradation)."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._surface_text = "this is not valid .nm format at all"
        handler._check_surface_depth = ToolHandler._check_surface_depth.__get__(handler)

        response, depth = handler._check_surface_depth("anything")
        assert response is None
        assert depth is None
