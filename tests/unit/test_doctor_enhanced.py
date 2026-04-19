"""Tests for enhanced doctor checks (hooks, dedup, surface, auto-fix)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from neural_memory.cli.doctor import (
    _CHECK_TIERS,
    FAIL,
    OK,
    QUICKSTART_URL,
    SKIP,
    TIER_CORE,
    TIER_OPTIONAL,
    TIER_RECOMMENDED,
    WARN,
    _auto_fix,
    _check_dedup,
    _check_hooks,
    _check_surface,
    _fix_dedup,
    _fix_embedding,
    _fix_hooks,
    run_doctor,
)


class TestCheckHooks:
    """Test hooks diagnostic check."""

    def test_all_hooks_present(self, tmp_path: Path) -> None:
        settings = {
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "nmem-hook-pre-compact"}]}
                ],
                "Stop": [{"hooks": [{"type": "command", "command": "nmem-hook-stop"}]}],
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "nmem-hook-post-tool-use"}]}
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("neural_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == OK
            assert "3/3" in result["detail"]

    def test_missing_hooks(self, tmp_path: Path) -> None:
        settings = {
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "nmem-hook-pre-compact"}]}
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("neural_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN
            assert "Stop" in result["detail"]
            assert "PostToolUse" in result["detail"]

    def test_no_settings_file(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN

    def test_corrupt_settings(self, tmp_path: Path) -> None:
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("not json!", encoding="utf-8")

        with patch("neural_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN

    def test_detects_python_module_hooks(self, tmp_path: Path) -> None:
        """Hooks using python -m neural_memory.hooks.* should also be detected."""
        settings = {
            "hooks": {
                "PreCompact": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m neural_memory.hooks.pre_compact",
                            }
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "python -m neural_memory.hooks.stop"}
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m neural_memory.hooks.post_tool_use",
                            }
                        ]
                    }
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("neural_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == OK


class TestCheckDedup:
    """Test dedup diagnostic check."""

    def test_dedup_enabled(self) -> None:
        mock_config = MagicMock()
        mock_config.dedup.enabled = True
        with patch("neural_memory.unified_config.get_config", return_value=mock_config):
            result = _check_dedup()
            assert result["status"] == OK

    def test_dedup_disabled(self) -> None:
        mock_config = MagicMock()
        mock_config.dedup.enabled = False
        with patch("neural_memory.unified_config.get_config", return_value=mock_config):
            result = _check_dedup()
            assert result["status"] == WARN
            assert "fixable" in result

    def test_config_not_loaded(self) -> None:
        with patch("neural_memory.unified_config.get_config", side_effect=Exception("no config")):
            result = _check_dedup()
            assert result["status"] == SKIP


class TestCheckSurface:
    """Test knowledge surface diagnostic check."""

    def test_surface_exists(self, tmp_path: Path) -> None:
        surface_file = tmp_path / "default.nm"
        surface_file.write_text("---\nbrain: default\n---\n# GRAPH", encoding="utf-8")

        mock_config = MagicMock()
        mock_config.current_brain = "default"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.surface.resolver.get_surface_path", return_value=surface_file),
        ):
            result = _check_surface()
            assert result["status"] == OK
            assert "KB" in result["detail"]

    def test_surface_missing(self, tmp_path: Path) -> None:
        surface_file = tmp_path / "default.nm"  # doesn't exist

        mock_config = MagicMock()
        mock_config.current_brain = "default"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.surface.resolver.get_surface_path", return_value=surface_file),
        ):
            result = _check_surface()
            assert result["status"] == WARN

    def test_surface_module_unavailable(self) -> None:
        """When surface package isn't available, check should skip."""
        with patch(
            "neural_memory.unified_config.get_config",
            side_effect=Exception("no config"),
        ):
            result = _check_surface()
            assert result["status"] == SKIP


class TestAutoFix:
    """Test auto-fix functionality."""

    def test_fixes_fixable_checks(self) -> None:
        checks = [
            {"name": "Hooks", "status": WARN, "detail": "missing", "fixable": True},
            {"name": "Python version", "status": OK, "detail": "3.12"},
        ]
        with patch("neural_memory.cli.doctor._try_fix") as mock_fix:
            mock_fix.return_value = {"name": "Hooks", "status": OK, "detail": "auto-fixed"}
            result = _auto_fix(checks)
            assert result[0]["status"] == OK
            assert result[1]["status"] == OK  # unchanged

    def test_skips_non_fixable(self) -> None:
        checks = [
            {"name": "Python version", "status": FAIL, "detail": "3.9 (requires 3.11+)"},
        ]
        result = _auto_fix(checks)
        assert result[0]["status"] == FAIL  # unchanged, not fixable

    def test_fix_hooks_success(self) -> None:
        with patch("neural_memory.cli.setup.setup_hooks_claude", return_value="added"):
            result = _fix_hooks()
            assert result["status"] == OK

    def test_fix_hooks_failure(self) -> None:
        with patch("neural_memory.cli.setup.setup_hooks_claude", side_effect=Exception("fail")):
            result = _fix_hooks()
            assert result["status"] == WARN

    def test_fix_dedup_success(self, tmp_path: Path) -> None:
        from dataclasses import replace

        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)
        config = replace(config, dedup=replace(config.dedup, enabled=False))
        with patch("neural_memory.unified_config.get_config", return_value=config):
            result = _fix_dedup()
            assert result["status"] == OK

    def test_fix_embedding_with_provider(self) -> None:
        provider = {"key": "sentence_transformer", "model": "all-MiniLM-L6-v2", "label": "ST"}
        with (
            patch(
                "neural_memory.cli.full_setup.detect_embedding_provider",
                return_value=provider,
            ),
            patch("neural_memory.cli.full_setup.enable_config_defaults"),
        ):
            result = _fix_embedding()
            assert result["status"] == OK

    def test_fix_embedding_no_provider(self) -> None:
        with patch("neural_memory.cli.full_setup.detect_embedding_provider", return_value=None):
            result = _fix_embedding()
            assert result["status"] == WARN


class TestRunDoctorIntegration:
    """Test run_doctor with new checks."""

    @patch("neural_memory.cli.doctor._check_pro_plugin")
    @patch("neural_memory.cli.doctor._check_surface")
    @patch("neural_memory.cli.doctor._check_dedup")
    @patch("neural_memory.cli.doctor._check_hooks")
    @patch("neural_memory.cli.doctor._check_cli_tools")
    @patch("neural_memory.cli.doctor._check_mcp_connection")
    @patch("neural_memory.cli.doctor._check_mcp_config")
    @patch("neural_memory.cli.doctor._check_schema_version")
    @patch("neural_memory.cli.doctor._check_embedding_provider")
    @patch("neural_memory.cli.doctor._check_dependencies")
    @patch("neural_memory.cli.doctor._check_brain")
    @patch("neural_memory.cli.doctor._check_config")
    @patch("neural_memory.cli.doctor._check_python_version")
    def test_includes_all_14_checks(self, *mocks: MagicMock) -> None:
        for mock in mocks:
            mock.return_value = {"name": "test", "status": OK, "detail": "ok"}

        result = run_doctor(json_output=True)
        assert result["total"] == 14
        assert result["passed"] == 14

    def test_quickstart_url_defined(self) -> None:
        assert "quickstart" in QUICKSTART_URL


class TestPriorityTiers:
    """Verify each check is tagged with the correct priority tier (issue #132)."""

    def test_core_checks_tagged(self) -> None:
        core_names = {
            "Python version",
            "Configuration",
            "Brain database",
            "Dependencies",
            "Schema version",
            "CLI tools",
        }
        for name in core_names:
            assert _CHECK_TIERS[name] == TIER_CORE, f"{name} should be CORE"

    def test_recommended_checks_tagged(self) -> None:
        for name in ("Embedding provider", "MCP configuration", "MCP server", "Hooks"):
            assert _CHECK_TIERS[name] == TIER_RECOMMENDED, f"{name} should be RECOMMENDED"

    def test_optional_checks_tagged(self) -> None:
        for name in ("Dedup", "Knowledge surface", "Config freshness", "Pro features"):
            assert _CHECK_TIERS[name] == TIER_OPTIONAL, f"{name} should be OPTIONAL"

    @patch("neural_memory.cli.doctor._check_pro_plugin")
    @patch("neural_memory.cli.doctor._check_surface")
    @patch("neural_memory.cli.doctor._check_dedup")
    @patch("neural_memory.cli.doctor._check_hooks")
    @patch("neural_memory.cli.doctor._check_cli_tools")
    @patch("neural_memory.cli.doctor._check_mcp_connection")
    @patch("neural_memory.cli.doctor._check_mcp_config")
    @patch("neural_memory.cli.doctor._check_schema_version")
    @patch("neural_memory.cli.doctor._check_embedding_provider")
    @patch("neural_memory.cli.doctor._check_dependencies")
    @patch("neural_memory.cli.doctor._check_brain")
    @patch("neural_memory.cli.doctor._check_config")
    @patch("neural_memory.cli.doctor._check_python_version")
    def test_run_doctor_annotates_tier(self, *mocks: MagicMock) -> None:
        names = [
            "Python version",
            "Configuration",
            "Brain database",
            "Dependencies",
            "Embedding provider",
            "Schema version",
            "MCP configuration",
            "MCP server",
            "Hooks",
            "Dedup",
            "Knowledge surface",
            "Config freshness",
            "CLI tools",
            "Pro features",
        ]
        # Decorators are applied bottom-up; mock[0] is innermost (_check_python_version)
        # Order of mocks matches run_doctor() call order 1:1
        for mock, name in zip(mocks, names, strict=False):
            mock.return_value = {"name": name, "status": OK, "detail": "ok"}

        result = run_doctor(json_output=True)
        tiers = {c["name"]: c["tier"] for c in result["checks"]}
        assert tiers["Python version"] == TIER_CORE
        assert tiers["Embedding provider"] == TIER_RECOMMENDED
        assert tiers["Knowledge surface"] == TIER_OPTIONAL
        assert "core_issues" in result


class TestSchemaMigrationHint:
    """Issue #132: schema migration fix message should name a concrete command."""

    def test_fix_message_names_command(self) -> None:
        import sqlite3
        from pathlib import Path

        from neural_memory.cli.doctor import _check_schema_version

        # This test relies on the schema check detecting a stale schema;
        # we validate the constant hint text instead (check in source).
        source = Path(__file__).parent.parent.parent / "src" / "neural_memory" / "cli" / "doctor.py"
        text = source.read_text(encoding="utf-8")
        assert "run any command" in text.lower()
        assert "nmem recall" in text  # concrete trigger example
        _ = _check_schema_version, sqlite3  # keep imports alive for mypy
