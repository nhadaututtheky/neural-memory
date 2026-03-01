"""Tests for MCP auto-configuration (setup_mcp_claude)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from neural_memory.cli.setup import (
    _add_via_claude_json,
    _claude_json_has_server,
    _cleanup_stale_mcp_servers_json,
    setup_mcp_claude,
)


class TestClaudeJsonHasServer:
    def test_file_not_exists(self, tmp_path: Path) -> None:
        assert _claude_json_has_server(tmp_path / "nope.json", "neural-memory") is False

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text("{}")
        assert _claude_json_has_server(f, "neural-memory") is False

    def test_server_present(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"mcpServers": {"neural-memory": {"command": "nmem-mcp"}}}))
        assert _claude_json_has_server(f, "neural-memory") is True

    def test_different_server(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}))
        assert _claude_json_has_server(f, "neural-memory") is False

    def test_corrupt_json(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text("not json")
        assert _claude_json_has_server(f, "neural-memory") is False


class TestAddViaClaudeJson:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        assert _add_via_claude_json(f, {"command": "nmem-mcp"}) is True
        data = json.loads(f.read_text())
        assert data["mcpServers"]["neural-memory"]["command"] == "nmem-mcp"

    def test_preserves_existing_data(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"numStartups": 5, "mcpServers": {"other": {"command": "x"}}}))
        assert _add_via_claude_json(f, {"command": "nmem-mcp"}) is True
        data = json.loads(f.read_text())
        assert data["numStartups"] == 5
        assert data["mcpServers"]["other"]["command"] == "x"
        assert data["mcpServers"]["neural-memory"]["command"] == "nmem-mcp"


class TestCleanupStaleMcpServersJson:
    def test_removes_stale_file(self, tmp_path: Path) -> None:
        stale = tmp_path / ".claude" / "mcp_servers.json"
        stale.parent.mkdir(parents=True)
        stale.write_text('{"neural-memory": {}}')
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            _cleanup_stale_mcp_servers_json()
        assert not stale.exists()

    def test_no_error_if_missing(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            _cleanup_stale_mcp_servers_json()  # Should not raise


class TestSetupMcpClaude:
    def test_not_found_no_claude_dir(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            assert setup_mcp_claude() == "not_found"

    def test_already_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(
            json.dumps({"mcpServers": {"neural-memory": {"command": "nmem-mcp"}}})
        )
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            assert setup_mcp_claude() == "exists"

    def test_adds_via_fallback(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        with (
            patch("neural_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("neural_memory.cli.setup._add_via_claude_cli", return_value=False),
            patch("neural_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"
        data = json.loads((tmp_path / ".claude.json").read_text())
        assert "neural-memory" in data["mcpServers"]

    def test_cleans_stale_on_success(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        stale = tmp_path / ".claude" / "mcp_servers.json"
        stale.write_text('{"old": true}')
        with (
            patch("neural_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("neural_memory.cli.setup._add_via_claude_cli", return_value=False),
            patch("neural_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"
        assert not stale.exists()

    def test_adds_via_cli_when_available(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        with (
            patch("neural_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("neural_memory.cli.setup._add_via_claude_cli", return_value=True),
            patch("neural_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"
