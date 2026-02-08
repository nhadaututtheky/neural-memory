"""Tests for the nmem update CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from neural_memory.cli.commands.update import (
    _get_source_dir,
    _run_command,
)

runner = CliRunner()


# ── _run_command ─────────────────────────────────────────────────


class TestRunCommand:
    """Tests for shell command execution."""

    def test_successful_command(self) -> None:
        """Successful command returns 0 and output."""
        returncode, output = _run_command(["python", "-c", "print('hello')"])
        assert returncode == 0
        assert "hello" in output

    def test_failing_command(self) -> None:
        """Failing command returns non-zero."""
        returncode, _output = _run_command(["python", "-c", "raise SystemExit(1)"])
        assert returncode == 1

    def test_command_not_found(self) -> None:
        """Missing command returns error message."""
        returncode, output = _run_command(["nonexistent_command_xyz_99"])
        assert returncode == 1
        assert "not found" in output.lower() or "Command not found" in output


# ── _get_source_dir ──────────────────────────────────────────────


class TestGetSourceDir:
    """Tests for source directory detection."""

    def test_finds_source_dir_for_current_project(self) -> None:
        """Source dir detection works for the current NeuralMemory project."""
        source_dir = _get_source_dir()
        assert source_dir is not None
        assert (source_dir / "pyproject.toml").exists()
        assert (source_dir / ".git").exists()


# ── CLI integration ──────────────────────────────────────────────


class TestUpdateCLI:
    """Integration tests for the update command via Typer CLI runner."""

    def test_check_only_shows_version(self) -> None:
        """--check flag shows versions without updating."""
        from neural_memory.cli.main import app

        with patch(
            "neural_memory.cli.commands.update._fetch_latest_version",
            return_value="0.19.0",
        ):
            result = runner.invoke(app, ["update", "--check"])
            assert "Current version:" in result.output
            assert "Latest version:" in result.output
            assert result.exit_code == 0

    def test_already_up_to_date(self) -> None:
        """Shows 'up to date' when local >= remote."""
        from neural_memory.cli.main import app

        with patch(
            "neural_memory.cli.commands.update._fetch_latest_version",
            return_value="0.1.0",
        ):
            result = runner.invoke(app, ["update"])
            assert "Already up to date" in result.output
            assert result.exit_code == 0

    def test_pypi_check_failure(self) -> None:
        """Shows error when PyPI check fails."""
        from neural_memory.cli.main import app

        with patch(
            "neural_memory.cli.commands.update._fetch_latest_version",
            return_value=None,
        ):
            result = runner.invoke(app, ["update"])
            assert "Failed to check" in result.output
            assert result.exit_code == 1

    def test_update_from_pip(self) -> None:
        """Pip update runs correct command."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="pip",
            ),
            patch(
                "neural_memory.cli.commands.update._run_command",
                return_value=(0, "Successfully installed neural-memory-99.0.0"),
            ) as mock_run,
            patch("neural_memory.cli.commands.update._show_new_version"),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 0
            assert "Updated successfully" in result.output
            cmd = mock_run.call_args[0][0]
            assert "neural-memory" in cmd

    def test_update_from_source(self) -> None:
        """Source update runs git pull + pip install -e ."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="editable",
            ),
            patch(
                "neural_memory.cli.commands.update._get_source_dir",
                return_value=MagicMock(__str__=lambda _: "/fake/dir"),
            ),
            patch(
                "neural_memory.cli.commands.update._run_command",
                side_effect=[
                    (0, "Updating files..."),  # git pull
                    (0, "Successfully installed"),  # pip install -e .
                ],
            ) as mock_run,
            patch("neural_memory.cli.commands.update._show_new_version"),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 0
            assert "Updated successfully" in result.output
            assert mock_run.call_count == 2

    def test_update_unknown_mode(self) -> None:
        """Unknown install mode shows help message."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="unknown",
            ),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "Could not detect" in result.output

    def test_force_flag_triggers_update(self) -> None:
        """--force flag updates even if already latest."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="0.1.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="pip",
            ),
            patch(
                "neural_memory.cli.commands.update._run_command",
                return_value=(0, "Reinstalled"),
            ),
            patch("neural_memory.cli.commands.update._show_new_version"),
        ):
            result = runner.invoke(app, ["update", "--force"])
            assert result.exit_code == 0
            assert "Force reinstalling" in result.output

    def test_git_pull_failure(self) -> None:
        """Git pull failure shows error."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="editable",
            ),
            patch(
                "neural_memory.cli.commands.update._get_source_dir",
                return_value=MagicMock(__str__=lambda _: "/fake/dir"),
            ),
            patch(
                "neural_memory.cli.commands.update._run_command",
                return_value=(1, "merge conflict"),
            ),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "git pull failed" in result.output

    def test_source_dir_not_found(self) -> None:
        """Shows error when source dir cannot be found."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="editable",
            ),
            patch(
                "neural_memory.cli.commands.update._get_source_dir",
                return_value=None,
            ),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "Could not find git source" in result.output

    def test_pip_install_failure(self) -> None:
        """Shows error when pip install fails."""
        from neural_memory.cli.main import app

        with (
            patch(
                "neural_memory.cli.commands.update._fetch_latest_version",
                return_value="99.0.0",
            ),
            patch(
                "neural_memory.cli.commands.update._detect_install_mode",
                return_value="pip",
            ),
            patch(
                "neural_memory.cli.commands.update._run_command",
                return_value=(1, "Permission denied"),
            ),
        ):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "Update failed" in result.output

    def test_check_with_update_available(self) -> None:
        """--check shows update instruction when newer version exists."""
        from neural_memory.cli.main import app

        with patch(
            "neural_memory.cli.commands.update._fetch_latest_version",
            return_value="99.0.0",
        ):
            result = runner.invoke(app, ["update", "--check"])
            assert "Update available" in result.output
            assert "To update, run: nmem update" in result.output
            assert result.exit_code == 0
