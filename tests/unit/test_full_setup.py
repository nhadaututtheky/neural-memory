"""Tests for nmem init --full (extended setup)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from neural_memory.cli.full_setup import (
    _PROVIDER_PRIORITY,
    QUICKSTART_URL,
    detect_embedding_provider,
    enable_config_defaults,
    generate_maintenance_script,
    run_full_setup,
)


class TestDetectEmbeddingProvider:
    """Test auto-detection of embedding providers."""

    def test_detects_sentence_transformers(self) -> None:
        with patch(
            "neural_memory.cli.full_setup._is_module_available",
            side_effect=lambda m: m == "sentence_transformers",
        ):
            result = detect_embedding_provider()
            assert result is not None
            assert result["key"] == "sentence_transformer"

    def test_detects_gemini_with_key(self) -> None:
        with (
            patch(
                "neural_memory.cli.full_setup._is_module_available",
                side_effect=lambda m: m == "google.genai",
            ),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            result = detect_embedding_provider()
            assert result is not None
            assert result["key"] == "gemini"

    def test_skips_gemini_without_key(self) -> None:
        with (
            patch(
                "neural_memory.cli.full_setup._is_module_available",
                side_effect=lambda m: m == "google.genai",
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_embedding_provider()
            assert result is None

    def test_detects_ollama(self) -> None:
        with patch(
            "neural_memory.cli.full_setup._is_module_available",
            side_effect=lambda m: m == "ollama",
        ):
            result = detect_embedding_provider()
            assert result is not None
            assert result["key"] == "ollama"

    def test_returns_none_when_nothing_available(self) -> None:
        with patch(
            "neural_memory.cli.full_setup._is_module_available",
            return_value=False,
        ):
            result = detect_embedding_provider()
            assert result is None

    def test_priority_order_prefers_local(self) -> None:
        """sentence_transformers should win over gemini even if both available."""
        with (
            patch(
                "neural_memory.cli.full_setup._is_module_available",
                return_value=True,
            ),
            patch.dict("os.environ", {"GEMINI_API_KEY": "key", "OPENAI_API_KEY": "key"}),
        ):
            result = detect_embedding_provider()
            assert result is not None
            assert result["key"] == "sentence_transformer"


class TestEnableConfigDefaults:
    """Test config defaults enablement."""

    def _make_config(
        self, tmp_path: Path, *, embedding_enabled: bool = False, dedup_enabled: bool = False
    ) -> Any:
        """Create a real UnifiedConfig with specified settings."""
        from dataclasses import replace

        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)
        config = replace(
            config,
            embedding=replace(config.embedding, enabled=embedding_enabled),
            dedup=replace(config.dedup, enabled=dedup_enabled),
        )
        return config

    def test_enables_embedding_and_dedup(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path, embedding_enabled=False, dedup_enabled=False)

        with patch("neural_memory.unified_config.get_config", return_value=config):
            provider = _PROVIDER_PRIORITY[0]  # sentence_transformer
            changes = enable_config_defaults(embedding_provider=provider)
            assert "embedding" in changes
            assert "dedup" in changes

    def test_skips_already_enabled(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path, embedding_enabled=True, dedup_enabled=True)

        with patch("neural_memory.unified_config.get_config", return_value=config):
            provider = _PROVIDER_PRIORITY[0]
            changes = enable_config_defaults(embedding_provider=provider)
            assert changes == {}

    def test_enables_dedup_without_provider(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path, embedding_enabled=True, dedup_enabled=False)

        with patch("neural_memory.unified_config.get_config", return_value=config):
            changes = enable_config_defaults(embedding_provider=None)
            assert "dedup" in changes
            assert "embedding" not in changes


class TestGenerateMaintenanceScript:
    """Test maintenance script generation."""

    def test_creates_bash_script_on_unix(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.full_setup.os.name", "posix"):
            result = generate_maintenance_script(tmp_path)
            assert result is not None
            assert result.name == "maintenance.sh"
            assert result.exists()
            content = result.read_text(encoding="utf-8")
            assert "nmem decay" in content
            assert "nmem consolidate" in content
            assert "nmem doctor" in content

    def test_creates_powershell_on_windows(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.full_setup.os.name", "nt"):
            result = generate_maintenance_script(tmp_path)
            assert result is not None
            assert result.name == "maintenance.ps1"
            content = result.read_text(encoding="utf-8")
            assert "nmem decay" in content

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        # Create existing script
        (tmp_path / "maintenance.sh").write_text("existing", encoding="utf-8")
        with patch("neural_memory.cli.full_setup.os.name", "posix"):
            result = generate_maintenance_script(tmp_path)
            assert result is None


class TestRunFullSetup:
    """Test the full setup orchestrator."""

    @patch("neural_memory.cli.full_setup.generate_maintenance_script", return_value=None)
    @patch("neural_memory.cli.full_setup.enable_config_defaults", return_value={"dedup": "enabled"})
    @patch("neural_memory.cli.full_setup.detect_embedding_provider", return_value=None)
    @patch("neural_memory.cli.full_setup._prompt_install_embeddings", return_value=None)
    @patch("neural_memory.cli.setup.setup_skills", return_value={"memory-intake": "installed"})
    @patch("neural_memory.cli.setup.setup_hooks_claude", return_value="added")
    @patch("neural_memory.cli.setup.setup_mcp_cursor", return_value="not_found")
    @patch("neural_memory.cli.setup.setup_mcp_claude", return_value="added")
    @patch("neural_memory.cli.setup.setup_brain", return_value="default")
    @patch("neural_memory.cli.setup.setup_config", return_value=True)
    @patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=Path("/tmp/nm"))
    @patch("neural_memory.cli.setup.print_summary")
    @patch("neural_memory.cli.full_setup.print_full_banner")
    def test_full_flow(
        self,
        mock_banner: MagicMock,
        mock_summary: MagicMock,
        mock_dir: MagicMock,
        mock_config: MagicMock,
        mock_brain: MagicMock,
        mock_claude: MagicMock,
        mock_cursor: MagicMock,
        mock_hooks: MagicMock,
        mock_skills: MagicMock,
        mock_prompt: MagicMock,
        mock_detect: MagicMock,
        mock_defaults: MagicMock,
        mock_maintenance: MagicMock,
    ) -> None:
        result = run_full_setup()
        assert "results" in result
        assert result["results"]["Brain"] == "default (ready)"
        assert result["results"]["Claude Code"] == "MCP server configured"
        assert result["results"]["Hooks"] == "3 hooks installed (PreCompact, Stop, PostToolUse)"
        mock_summary.assert_called_once()
        mock_banner.assert_called_once()

    @patch("neural_memory.cli.full_setup.generate_maintenance_script", return_value=None)
    @patch("neural_memory.cli.full_setup.enable_config_defaults", return_value={})
    @patch("neural_memory.cli.full_setup.detect_embedding_provider")
    @patch("neural_memory.cli.setup.setup_skills", return_value={"s1": "exists"})
    @patch("neural_memory.cli.setup.setup_hooks_claude", return_value="exists")
    @patch("neural_memory.cli.setup.setup_mcp_cursor", return_value="not_found")
    @patch("neural_memory.cli.setup.setup_mcp_claude", return_value="exists")
    @patch("neural_memory.cli.setup.setup_brain", return_value="default")
    @patch("neural_memory.cli.setup.setup_config", return_value=False)
    @patch("neural_memory.unified_config.get_neuralmemory_dir", return_value=Path("/tmp/nm"))
    @patch("neural_memory.cli.setup.print_summary")
    @patch("neural_memory.cli.full_setup.print_full_banner")
    def test_skip_mcp(
        self,
        mock_banner: MagicMock,
        mock_summary: MagicMock,
        mock_dir: MagicMock,
        mock_config: MagicMock,
        mock_brain: MagicMock,
        mock_claude: MagicMock,
        mock_cursor: MagicMock,
        mock_hooks: MagicMock,
        mock_skills: MagicMock,
        mock_detect: MagicMock,
        mock_defaults: MagicMock,
        mock_maintenance: MagicMock,
    ) -> None:
        mock_detect.return_value = _PROVIDER_PRIORITY[0]
        result = run_full_setup(skip_mcp=True)
        assert result["results"]["Claude Code"] == "skipped"
        assert "Hooks" not in result["results"]


class TestQuickstartUrl:
    """Verify the URL constant is set."""

    def test_url_is_valid(self) -> None:
        assert "github.io" in QUICKSTART_URL
        assert "quickstart" in QUICKSTART_URL
