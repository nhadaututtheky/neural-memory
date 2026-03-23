"""Tests for DX Sprint features: wizard, doctor, embedding setup (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ──────────────────── Doctor ────────────────────


class TestDoctor:
    """Test nmem doctor diagnostics."""

    def test_check_python_version_ok(self) -> None:
        from neural_memory.cli.doctor import _check_python_version

        result = _check_python_version()
        assert result["status"] == "ok"
        assert result["name"] == "Python version"

    def test_check_dependencies_ok(self) -> None:
        from neural_memory.cli.doctor import _check_dependencies

        result = _check_dependencies()
        assert result["status"] == "ok"

    def test_check_dependencies_missing(self) -> None:
        from neural_memory.cli.doctor import _check_dependencies

        with patch("neural_memory.cli.doctor.importlib.import_module", side_effect=ImportError):
            result = _check_dependencies()
            assert result["status"] == "fail"
            assert "Missing" in result["detail"]

    def test_check_embedding_disabled(self) -> None:
        from neural_memory.cli.doctor import _check_embedding_provider

        mock_config = MagicMock()
        mock_config.embedding.enabled = False

        with patch("neural_memory.unified_config.get_config", return_value=mock_config):
            result = _check_embedding_provider()
            assert result["status"] == "warn"
            assert "nmem setup embeddings" in result.get("fix", "")

    def test_check_embedding_enabled_ok(self) -> None:
        from neural_memory.cli.doctor import _check_embedding_provider

        mock_config = MagicMock()
        mock_config.embedding.enabled = True
        mock_config.embedding.provider = "sentence_transformer"
        mock_config.embedding.model = "test-model"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.cli.doctor.importlib.import_module"),
        ):
            result = _check_embedding_provider()
            assert result["status"] == "ok"

    def test_check_embedding_not_installed(self) -> None:
        from neural_memory.cli.doctor import _check_embedding_provider

        mock_config = MagicMock()
        mock_config.embedding.enabled = True
        mock_config.embedding.provider = "openai"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.cli.doctor.importlib.import_module", side_effect=ImportError),
        ):
            result = _check_embedding_provider()
            assert result["status"] == "fail"
            assert "not installed" in result["detail"]

    def test_check_embedding_openrouter_not_installed(self) -> None:
        from neural_memory.cli.doctor import _check_embedding_provider

        mock_config = MagicMock()
        mock_config.embedding.enabled = True
        mock_config.embedding.provider = "openrouter"
        mock_config.embedding.model = "openai/text-embedding-3-small"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.cli.doctor.importlib.import_module", side_effect=ImportError),
        ):
            result = _check_embedding_provider()
            assert result["status"] == "fail"
            assert "not installed" in result["detail"]
            assert result["fix"] == "Run: pip install neural-memory[embeddings-openrouter]"

    def test_check_embedding_openai_not_installed_uses_embeddings_extra(self) -> None:
        from neural_memory.cli.doctor import _check_embedding_provider

        mock_config = MagicMock()
        mock_config.embedding.enabled = True
        mock_config.embedding.provider = "openai"
        mock_config.embedding.model = "text-embedding-3-small"

        with (
            patch("neural_memory.unified_config.get_config", return_value=mock_config),
            patch("neural_memory.cli.doctor.importlib.import_module", side_effect=ImportError),
        ):
            result = _check_embedding_provider()
            assert result["status"] == "fail"
            assert result["fix"] == "Run: pip install neural-memory[embeddings-openai]"

    def test_check_mcp_config_registered(self, tmp_path: Path) -> None:
        from neural_memory.cli.doctor import _check_mcp_config

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"mcpServers": {"neural-memory": {}}}))

        with patch("neural_memory.cli.doctor.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            result = _check_mcp_config()
            assert result["status"] == "ok"

    def test_check_mcp_config_missing(self, tmp_path: Path) -> None:
        from neural_memory.cli.doctor import _check_mcp_config

        with patch("neural_memory.cli.doctor.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            result = _check_mcp_config()
            assert result["status"] == "warn"

    def test_check_cli_tools_all_found(self) -> None:
        from neural_memory.cli.doctor import _check_cli_tools

        with patch("neural_memory.cli.doctor.shutil.which", return_value="/usr/bin/nmem"):
            result = _check_cli_tools()
            assert result["status"] == "ok"

    def test_check_cli_tools_nmem_missing(self) -> None:
        from neural_memory.cli.doctor import _check_cli_tools

        with patch("neural_memory.cli.doctor.shutil.which", return_value=None):
            result = _check_cli_tools()
            assert result["status"] == "fail"

    def test_run_doctor_returns_summary(self) -> None:
        from neural_memory.cli.doctor import run_doctor

        with (
            patch("neural_memory.cli.doctor._check_python_version") as m1,
            patch("neural_memory.cli.doctor._check_config") as m2,
            patch("neural_memory.cli.doctor._check_brain") as m3,
            patch("neural_memory.cli.doctor._check_dependencies") as m4,
            patch("neural_memory.cli.doctor._check_embedding_provider") as m5,
            patch("neural_memory.cli.doctor._check_schema_version") as m6,
            patch("neural_memory.cli.doctor._check_mcp_config") as m7,
            patch("neural_memory.cli.doctor._check_cli_tools") as m8,
            patch("neural_memory.cli.doctor._check_hooks") as m9,
            patch("neural_memory.cli.doctor._check_dedup") as m10,
            patch("neural_memory.cli.doctor._check_surface") as m11,
            patch("neural_memory.cli.doctor._check_config_freshness") as m12,
        ):
            for m in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]:
                m.return_value = {"name": "test", "status": "ok", "detail": "ok"}

            result = run_doctor(json_output=True)
            assert result["passed"] == 12
            assert result["total"] == 12
            assert result["failed"] == 0

    def test_run_doctor_with_failures(self) -> None:
        from neural_memory.cli.doctor import run_doctor

        with (
            patch("neural_memory.cli.doctor._check_python_version") as m1,
            patch("neural_memory.cli.doctor._check_config") as m2,
            patch("neural_memory.cli.doctor._check_brain") as m3,
            patch("neural_memory.cli.doctor._check_dependencies") as m4,
            patch("neural_memory.cli.doctor._check_embedding_provider") as m5,
            patch("neural_memory.cli.doctor._check_schema_version") as m6,
            patch("neural_memory.cli.doctor._check_mcp_config") as m7,
            patch("neural_memory.cli.doctor._check_cli_tools") as m8,
            patch("neural_memory.cli.doctor._check_hooks") as m9,
            patch("neural_memory.cli.doctor._check_dedup") as m10,
            patch("neural_memory.cli.doctor._check_surface") as m11,
            patch("neural_memory.cli.doctor._check_config_freshness") as m12,
        ):
            m1.return_value = {"name": "Python", "status": "ok", "detail": "ok"}
            m2.return_value = {"name": "Config", "status": "fail", "detail": "missing"}
            for m in [m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]:
                m.return_value = {"name": "test", "status": "ok", "detail": "ok"}

            result = run_doctor(json_output=True)
            assert result["failed"] == 1
            assert result["passed"] == 11


# ──────────────────── Embedding Setup ────────────────────


class TestEmbeddingSetup:
    """Test embedding setup helpers."""

    def test_is_installed_true(self) -> None:
        from neural_memory.cli.embedding_setup import _is_installed

        assert _is_installed("json") is True

    def test_is_installed_false(self) -> None:
        from neural_memory.cli.embedding_setup import _is_installed

        assert _is_installed("nonexistent_module_xyz") is False

    def test_has_env_key_none(self) -> None:
        from neural_memory.cli.embedding_setup import _has_env_key

        assert _has_env_key(None) is True

    def test_has_env_key_missing(self) -> None:
        from neural_memory.cli.embedding_setup import _has_env_key

        with patch.dict("os.environ", {}, clear=True):
            assert _has_env_key("NONEXISTENT_KEY_XYZ") is False

    def test_has_env_key_set(self) -> None:
        from neural_memory.cli.embedding_setup import _has_env_key

        with patch.dict("os.environ", {"TEST_KEY": "value"}):
            assert _has_env_key("TEST_KEY") is True

    def test_save_embedding_config(self, tmp_path) -> None:
        from neural_memory.cli.embedding_setup import _save_embedding_config
        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)

        with (
            patch("neural_memory.unified_config.get_config", return_value=config),
            patch.object(UnifiedConfig, "save") as mock_save,
        ):
            _save_embedding_config("gemini", "models/text-embedding-004")
            mock_save.assert_called_once()

    def test_save_embedding_disabled(self, tmp_path) -> None:
        from neural_memory.cli.embedding_setup import _save_embedding_disabled
        from neural_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)

        with (
            patch("neural_memory.unified_config.get_config", return_value=config),
            patch.object(UnifiedConfig, "save") as mock_save,
        ):
            _save_embedding_disabled()
            mock_save.assert_called_once()

    def test_provider_list_has_all_providers(self) -> None:
        from neural_memory.cli.embedding_setup import _PROVIDERS

        keys = {p["key"] for p in _PROVIDERS}
        assert keys == {"sentence_transformer", "gemini", "ollama", "openai", "openrouter"}

    def test_cloud_provider_install_hints_use_embeddings_extra(self) -> None:
        from neural_memory.cli.embedding_setup import _PROVIDERS

        install_by_key = {provider["key"]: provider["install"] for provider in _PROVIDERS}
        assert install_by_key["openai"] == "pip install neural-memory[embeddings-openai]"
        assert install_by_key["openrouter"] == "pip install neural-memory[embeddings-openrouter]"

    def test_pyproject_embedding_extras_include_cloud_sdk(self) -> None:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        embeddings = data["project"]["optional-dependencies"]["embeddings"]
        assert "openai>=1.0" not in embeddings

        embeddings_openai = data["project"]["optional-dependencies"]["embeddings-openai"]
        assert "openai>=1.0" in embeddings_openai

        embeddings_openrouter = data["project"]["optional-dependencies"]["embeddings-openrouter"]
        assert "openai>=1.0" in embeddings_openrouter

    def test_all_providers_have_required_fields(self) -> None:
        from neural_memory.cli.embedding_setup import _PROVIDERS

        required = {"key", "name", "type", "module", "install", "default_model", "note"}
        for p in _PROVIDERS:
            assert required.issubset(p.keys()), f"Provider {p.get('key')} missing fields"


# ──────────────────── Wizard ────────────────────


class TestWizard:
    """Test wizard helpers."""

    def test_format_mcp_result_added(self) -> None:
        from neural_memory.cli.wizard import _format_mcp_result

        results: dict[str, str] = {}
        _format_mcp_result(results, "Claude Code", "added")
        assert results["Claude Code"] == "configured"

    def test_format_mcp_result_exists(self) -> None:
        from neural_memory.cli.wizard import _format_mcp_result

        results: dict[str, str] = {}
        _format_mcp_result(results, "Cursor", "exists")
        assert results["Cursor"] == "already configured"

    def test_format_mcp_result_not_found(self) -> None:
        from neural_memory.cli.wizard import _format_mcp_result

        results: dict[str, str] = {}
        _format_mcp_result(results, "Test", "not_found")
        assert results["Test"] == "not detected"

    def test_format_mcp_result_failed(self) -> None:
        from neural_memory.cli.wizard import _format_mcp_result

        results: dict[str, str] = {}
        _format_mcp_result(results, "Test", "failed")
        assert results["Test"] == "failed to configure"

    def test_setup_brain_with_name(self, tmp_path: Path) -> None:
        from neural_memory.cli.wizard import _setup_brain_with_name

        _setup_brain_with_name(tmp_path, "work")
        assert (tmp_path / "brains" / "work.db").exists()

    def test_setup_brain_with_name_default(self, tmp_path: Path) -> None:
        from neural_memory.cli.wizard import _setup_brain_with_name

        _setup_brain_with_name(tmp_path, "default")
        assert (tmp_path / "brains" / "default.db").exists()

    def test_provider_metadata(self) -> None:
        from neural_memory.cli.wizard import _PROVIDERS

        assert len(_PROVIDERS) == 5
        for p in _PROVIDERS:
            assert "key" in p
            assert "name" in p
            assert "install" in p


# ──────────────────── Init Wizard Flag ────────────────────


class TestInitWizardFlag:
    """Test that init --wizard flag is wired correctly."""

    def test_init_has_wizard_param(self) -> None:
        """Verify init function accepts wizard parameter."""
        import inspect

        from neural_memory.cli.commands.tools import init

        sig = inspect.signature(init)
        assert "wizard" in sig.parameters
        assert "defaults" in sig.parameters

    def test_init_wizard_dispatches(self) -> None:
        """Verify --wizard flag dispatches to run_wizard."""
        from neural_memory.cli.commands.tools import init

        with patch("neural_memory.cli.wizard.run_wizard") as mock_wizard:
            init(wizard=True, force=False, skip_mcp=False, skip_skills=False, defaults=False)
            mock_wizard.assert_called_once_with(force=False)


# ──────────────────── Doctor Command ────────────────────


class TestDoctorCommand:
    """Test doctor CLI command wiring."""

    def test_doctor_function_exists(self) -> None:
        from neural_memory.cli.commands.tools import doctor

        assert callable(doctor)

    def test_setup_function_exists(self) -> None:
        from neural_memory.cli.commands.tools import setup

        assert callable(setup)

    def test_setup_no_arg_does_not_raise(self) -> None:
        from neural_memory.cli.commands.tools import setup

        # Should not raise when called with empty string
        setup(component="")


# ──────────────────── Error Message Improvement ────────────────────


class TestErrorMessages:
    """Verify error messages don't leak internals."""

    def test_batch_remember_error_no_leak(self) -> None:
        """Batch remember errors should not expose exception details."""
        import inspect

        from neural_memory.mcp import tool_handlers

        source = inspect.getsource(tool_handlers.ToolHandler._remember_batch)
        assert 'reason": "failed to store"' in source
        assert "str(e)" not in source
