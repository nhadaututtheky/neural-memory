"""Self-update command for neural-memory CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from neural_memory.cli.update_check import _fetch_latest_version, _is_newer


def _detect_install_mode() -> str:
    """Detect how neural-memory was installed.

    Returns one of: 'editable', 'pip', 'unknown'.
    """
    try:
        from importlib.metadata import distribution

        dist = distribution("neural-memory")
        # Editable installs have a direct_url.json with "editable" key
        direct_url = dist.read_text("direct_url.json")
        if direct_url and '"editable"' in direct_url:
            return "editable"
    except Exception:
        pass

    # Check if we're running from a git repo with pyproject.toml
    try:
        import neural_memory

        pkg_dir = Path(neural_memory.__file__).resolve().parent
        # Walk up to find pyproject.toml + .git
        for parent in [pkg_dir] + list(pkg_dir.parents):
            if (parent / "pyproject.toml").exists() and (parent / ".git").exists():
                return "editable"
            # Stop at reasonable depth
            if parent == parent.parent:
                break
    except Exception:
        pass

    # If we can import it and it's not editable, assume pip
    try:
        from importlib.metadata import distribution

        distribution("neural-memory")
        return "pip"
    except Exception:
        return "unknown"


def _get_source_dir() -> Path | None:
    """Find the git source directory for editable installs."""
    try:
        import neural_memory

        pkg_dir = Path(neural_memory.__file__).resolve().parent
        for parent in [pkg_dir] + list(pkg_dir.parents):
            if (parent / "pyproject.toml").exists() and (parent / ".git").exists():
                return parent
            if parent == parent.parent:
                break
    except Exception:
        pass
    return None


def _run_command(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    """Run a shell command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return 1, "Command timed out after 120 seconds"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def update(
    force: bool = typer.Option(False, "--force", "-f", help="Force update even if already latest"),
    check_only: bool = typer.Option(
        False, "--check", "-c", help="Only check for updates, don't install"
    ),
) -> None:
    """Update neural-memory to the latest version.

    Automatically detects install method (pip or git source) and updates accordingly.

    Examples:
        nmem update              # Update to latest
        nmem update --check      # Just check for updates
        nmem update --force      # Force reinstall
    """
    from neural_memory import __version__

    current = __version__
    typer.secho(f"Current version: {current}", fg=typer.colors.CYAN)

    # Check latest version
    typer.echo("Checking for updates...")
    latest = _fetch_latest_version()

    if latest is None:
        typer.secho("Failed to check PyPI for updates.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"Latest version:  {latest}", fg=typer.colors.CYAN)

    if not force and not _is_newer(latest, current):
        typer.secho("\nAlready up to date!", fg=typer.colors.GREEN)
        return

    if _is_newer(latest, current):
        typer.secho(f"\nUpdate available: {current} -> {latest}", fg=typer.colors.YELLOW)
    elif force:
        typer.secho("\nForce reinstalling current version...", fg=typer.colors.YELLOW)

    if check_only:
        if _is_newer(latest, current):
            typer.echo("\nTo update, run: nmem update")
        return

    # Detect install mode
    mode = _detect_install_mode()
    typer.secho(f"Install mode: {mode}", fg=typer.colors.BRIGHT_BLACK)

    if mode == "editable":
        _update_from_source(force)
    elif mode == "pip":
        _update_from_pip(force)
    else:
        typer.secho(
            "Could not detect install method. Try one of:\n"
            "  pip install -U neural-memory\n"
            "  cd <source-dir> && git pull && pip install -e .",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


def _update_from_pip(force: bool) -> None:
    """Update via pip install -U."""
    typer.echo("\nUpdating via pip...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "neural-memory"]
    if force:
        cmd.insert(-1, "--force-reinstall")

    returncode, output = _run_command(cmd)

    if returncode == 0:
        typer.secho("\nUpdated successfully!", fg=typer.colors.GREEN)
        # Show new version
        _show_new_version()
    else:
        typer.secho(f"\nUpdate failed:\n{output}", fg=typer.colors.RED)
        raise typer.Exit(1)


def _update_from_source(force: bool) -> None:
    """Update via git pull + pip install -e ."""
    source_dir = _get_source_dir()
    if source_dir is None:
        typer.secho("Could not find git source directory.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"Source directory: {source_dir}", fg=typer.colors.BRIGHT_BLACK)

    # Step 1: git pull
    typer.echo("\nPulling latest changes...")
    returncode, output = _run_command(["git", "pull", "origin", "main"], cwd=str(source_dir))

    if returncode != 0:
        typer.secho(f"\ngit pull failed:\n{output}", fg=typer.colors.RED)
        typer.echo("You may need to resolve conflicts manually.")
        raise typer.Exit(1)

    if "Already up to date" in output and not force:
        typer.secho("Already up to date (git).", fg=typer.colors.GREEN)
    else:
        typer.echo(output)

    # Step 2: pip install -e .
    typer.echo("\nReinstalling package...")
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    returncode, output = _run_command(cmd, cwd=str(source_dir))

    if returncode == 0:
        typer.secho("\nUpdated successfully!", fg=typer.colors.GREEN)
        _show_new_version()
    else:
        typer.secho(f"\npip install -e . failed:\n{output}", fg=typer.colors.RED)
        raise typer.Exit(1)


def _show_new_version() -> None:
    """Show the version after update by running a fresh check."""
    returncode, output = _run_command(
        [sys.executable, "-c", "from neural_memory import __version__; print(__version__)"]
    )
    if returncode == 0 and output:
        typer.secho(f"Now running: neural-memory {output}", fg=typer.colors.GREEN)


def register(app: typer.Typer) -> None:
    """Register the update command on the app."""
    app.command(name="update")(update)
