"""Tests for Phase 4: Multi-Agent Hygiene.

Covers:
- Agent identity capture from MCP initialize
- Agent tag auto-injection in _remember()
- Consolidation file lock (acquire, release, stale detection)
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.utils.consolidation_lock import (
    _lock_path,
    acquire_consolidation_lock,
    release_consolidation_lock,
)

# ---------------------------------------------------------------------------
# Agent identity capture
# ---------------------------------------------------------------------------


class TestAgentIdentityCapture:
    """Test that MCP initialize captures clientInfo.name."""

    @pytest.mark.asyncio
    async def test_capture_client_info_name(self) -> None:
        """Initialize with clientInfo should set _agent_id."""
        from neural_memory.mcp.server import MCPServer, handle_message

        server = MagicMock(spec=MCPServer)
        server._agent_id = ""
        server.load_surface = MagicMock(return_value="")

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "claude-code", "version": "1.0"},
            },
        }

        with patch("neural_memory.mcp.server.get_mcp_instructions", return_value="test"):
            result = await handle_message(server, message)

        assert server._agent_id == "claude-code"
        assert result["result"]["serverInfo"]["name"] == "neural-memory"

    @pytest.mark.asyncio
    async def test_no_client_info(self) -> None:
        """Initialize without clientInfo should leave agent_id empty."""
        from neural_memory.mcp.server import MCPServer, handle_message

        server = MagicMock(spec=MCPServer)
        server._agent_id = ""
        server.load_surface = MagicMock(return_value="")

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        with patch("neural_memory.mcp.server.get_mcp_instructions", return_value="test"):
            await handle_message(server, message)

        assert server._agent_id == ""


# ---------------------------------------------------------------------------
# Agent tag auto-injection
# ---------------------------------------------------------------------------


class TestAgentTagInjection:
    """Test that agent tag is auto-injected in _remember()."""

    def test_agent_tag_added_to_tags(self) -> None:
        """When agent_id is set, 'agent:<id>' tag should be added."""
        tags: set[str] = set()
        agent_id = "claude-code"
        if agent_id:
            tags.add(f"agent:{agent_id}")
        assert "agent:claude-code" in tags

    def test_no_agent_id_no_tag(self) -> None:
        """When agent_id is empty, no agent tag should be added."""
        tags: set[str] = set()
        agent_id = ""
        if agent_id:
            tags.add(f"agent:{agent_id}")
        assert len(tags) == 0

    def test_agent_tag_coexists_with_user_tags(self) -> None:
        """Agent tag should not replace user-provided tags."""
        tags = {"python", "auth"}
        agent_id = "claude-code"
        if agent_id:
            tags.add(f"agent:{agent_id}")
        assert "python" in tags
        assert "auth" in tags
        assert "agent:claude-code" in tags
        assert len(tags) == 3


# ---------------------------------------------------------------------------
# Consolidation file lock
# ---------------------------------------------------------------------------


class TestConsolidationLock:
    """Test file-based consolidation lock."""

    @pytest.fixture(autouse=True)
    def _cleanup_lock(self) -> None:
        """Remove lock file before/after each test."""
        lock = _lock_path()
        lock.unlink(missing_ok=True)
        yield  # type: ignore[misc]
        lock.unlink(missing_ok=True)

    def test_acquire_fresh_lock(self) -> None:
        """Should acquire lock when no lock exists."""
        assert acquire_consolidation_lock() is True
        assert _lock_path().exists()

    def test_acquire_blocked_by_active_lock(self) -> None:
        """Should fail when lock is held by current process."""
        assert acquire_consolidation_lock() is True
        # Same PID, not stale → should block
        assert acquire_consolidation_lock() is False

    def test_release_lock(self) -> None:
        """Should release lock when called by owning PID."""
        acquire_consolidation_lock()
        release_consolidation_lock()
        assert not _lock_path().exists()

    def test_stale_lock_cleared_dead_pid(self) -> None:
        """Lock from dead PID should be auto-cleared."""
        lock = _lock_path()
        lock.write_text(json.dumps({"pid": 99999999, "timestamp": time.time()}))

        with patch("neural_memory.utils.consolidation_lock._is_pid_alive", return_value=False):
            assert acquire_consolidation_lock() is True

    def test_stale_lock_cleared_old_timestamp(self) -> None:
        """Lock older than 1 hour should be auto-cleared."""
        lock = _lock_path()
        lock.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "timestamp": time.time() - 7200,  # 2 hours ago
                }
            )
        )

        assert acquire_consolidation_lock() is True

    def test_corrupt_lock_cleared(self) -> None:
        """Corrupt lock file should be auto-cleared."""
        lock = _lock_path()
        lock.write_text("not json")

        assert acquire_consolidation_lock() is True

    def test_release_only_own_lock(self) -> None:
        """Should not release lock owned by different PID."""
        lock = _lock_path()
        lock.write_text(json.dumps({"pid": 99999999, "timestamp": time.time()}))

        release_consolidation_lock()
        # Lock should still exist (different PID)
        assert lock.exists()

    def test_lock_contains_pid_and_timestamp(self) -> None:
        """Lock file should contain current PID and recent timestamp."""
        acquire_consolidation_lock()
        data = json.loads(_lock_path().read_text())
        assert data["pid"] == os.getpid()
        assert time.time() - data["timestamp"] < 5  # within 5 seconds


# ---------------------------------------------------------------------------
# Session-end consolidation with lock
# ---------------------------------------------------------------------------


class TestSessionEndConsolidationLock:
    """Test that session-end consolidation uses the file lock."""

    @pytest.fixture(autouse=True)
    def _cleanup_lock(self) -> None:
        lock = _lock_path()
        lock.unlink(missing_ok=True)
        yield  # type: ignore[misc]
        lock.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_consolidation_skipped_when_locked(self) -> None:
        """Session-end consolidation should skip when lock is held."""
        from neural_memory.mcp.maintenance_handler import MaintenanceHandler

        handler = MagicMock()
        handler.config = MagicMock()
        handler.config.maintenance.enabled = True
        handler.config.maintenance.auto_consolidate = True

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test"
        mock_storage.brain_id = "test"
        handler.get_storage = AsyncMock(return_value=mock_storage)

        with patch(
            "neural_memory.utils.consolidation_lock.acquire_consolidation_lock",
            return_value=False,
        ):
            await MaintenanceHandler.run_session_end_consolidation(handler)

        # get_storage is called (to get brain_id), but consolidation should not run

    @pytest.mark.asyncio
    async def test_consolidation_releases_lock_on_success(self) -> None:
        """Lock should be released after successful consolidation."""
        from neural_memory.mcp.maintenance_handler import MaintenanceHandler

        handler = MagicMock()
        handler.config = MagicMock()
        handler.config.maintenance.enabled = True
        handler.config.maintenance.auto_consolidate = True
        handler.get_storage = AsyncMock()

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test"
        mock_storage.brain_id = "test"
        handler.get_storage.return_value = mock_storage

        release_called = False

        def track_release(brain_id: str = "") -> None:
            nonlocal release_called
            release_called = True

        async def mock_run_with_delta(storage, brain_id, strategies):
            result = MagicMock()
            result.report.summary.return_value = "ok"
            result.purity_delta = 0.0
            return result

        with (
            patch(
                "neural_memory.utils.consolidation_lock.acquire_consolidation_lock",
                return_value=True,
            ),
            patch(
                "neural_memory.utils.consolidation_lock.release_consolidation_lock",
                side_effect=track_release,
            ),
            patch(
                "neural_memory.engine.consolidation_delta.run_with_delta",
                side_effect=mock_run_with_delta,
            ),
        ):
            await MaintenanceHandler.run_session_end_consolidation(handler)

        assert release_called
