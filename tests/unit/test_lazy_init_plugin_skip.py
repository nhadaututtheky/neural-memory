"""Tests for _lazy_init plugin detection (issue #169).

When the MCP server is launched from a Claude Code plugin install, the
plugin's own .claude-plugin/hooks/hooks.json already registers the four
lifecycle hooks. _lazy_init must not duplicate them into
~/.claude/settings.json — otherwise every hook fires twice per event.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from neural_memory.mcp import server as mcp_server


class TestPluginDetection:
    """`_running_under_plugin()` heuristics."""

    def test_returns_false_when_plugin_cache_dir_missing(self, tmp_path: Path) -> None:
        with patch.object(Path, "home", return_value=tmp_path):
            assert mcp_server._running_under_plugin() is False

    def test_returns_false_when_no_neural_memory_plugin_installed(self, tmp_path: Path) -> None:
        # Some other plugin is installed, but not neural-memory
        other_plugin = tmp_path / ".claude" / "plugins" / "cache" / "some-market" / "other-plugin"
        other_plugin.mkdir(parents=True)
        with patch.object(Path, "home", return_value=tmp_path):
            assert mcp_server._running_under_plugin() is False

    def test_returns_true_when_neural_memory_plugin_installed(self, tmp_path: Path) -> None:
        nm_plugin = tmp_path / ".claude" / "plugins" / "cache" / "neural-memory" / "neural-memory"
        nm_plugin.mkdir(parents=True)
        with patch.object(Path, "home", return_value=tmp_path):
            assert mcp_server._running_under_plugin() is True


class TestLazyInitSkipsHooksUnderPlugin:
    """End-to-end: _lazy_init in plugin context must NOT call setup_hooks_claude.

    Issue #169 cause: settings.json had 3 hooks injected by _lazy_init, AND
    the plugin's hooks.json had the same 3 hooks. Every event fired twice.
    """

    def test_hook_install_skipped_when_plugin_detected(self, tmp_path: Path) -> None:
        # Simulate: config.toml doesn't exist (so _lazy_init runs full setup)
        # AND plugin cache directory exists (so plugin context detected).
        nm_plugin = tmp_path / ".claude" / "plugins" / "cache" / "neural-memory" / "neural-memory"
        nm_plugin.mkdir(parents=True)

        data_dir = tmp_path / ".neuralmemory"

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch(
                "neural_memory.unified_config.get_neuralmemory_dir",
                return_value=data_dir,
            ),
            patch("neural_memory.cli.setup.setup_config") as mock_config,
            patch("neural_memory.cli.setup.setup_brain") as mock_brain,
            patch("neural_memory.cli.setup.setup_hooks_claude") as mock_hooks,
        ):
            mock_config.return_value = True
            mock_brain.return_value = "default"

            mcp_server._lazy_init()

            mock_config.assert_called_once()
            mock_brain.assert_called_once()
            # Critical: hook injection must be skipped under plugin context
            mock_hooks.assert_not_called()

    def test_hook_install_runs_for_pip_only_user(self, tmp_path: Path) -> None:
        """Without the plugin installed, pip users still get hook auto-setup."""
        data_dir = tmp_path / ".neuralmemory"
        # No plugin cache directory at all

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch(
                "neural_memory.unified_config.get_neuralmemory_dir",
                return_value=data_dir,
            ),
            patch("neural_memory.cli.setup.setup_config") as mock_config,
            patch("neural_memory.cli.setup.setup_brain") as mock_brain,
            patch("neural_memory.cli.setup.setup_hooks_claude") as mock_hooks,
        ):
            mock_config.return_value = True
            mock_brain.return_value = "default"
            mock_hooks.return_value = "added"

            mcp_server._lazy_init()

            mock_hooks.assert_called_once()

    def test_lazy_init_no_op_when_config_exists(self, tmp_path: Path) -> None:
        """Fast path: when config.toml already exists, _lazy_init does nothing."""
        data_dir = tmp_path / ".neuralmemory"
        data_dir.mkdir()
        (data_dir / "config.toml").write_text("# pre-existing\n")

        with (
            patch(
                "neural_memory.unified_config.get_neuralmemory_dir",
                return_value=data_dir,
            ),
            patch("neural_memory.cli.setup.setup_hooks_claude") as mock_hooks,
        ):
            mcp_server._lazy_init()
            mock_hooks.assert_not_called()
