"""Extended init — nmem init --full.

Orchestrates a complete NeuralMemory setup in one command:
1. Standard init (config, brain, MCP, hooks, skills)
2. Auto-detect and enable best embedding provider
3. Enable recommended config defaults (dedup, etc.)
4. Generate maintenance script
5. Print summary with quickstart guide URL
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import typer

QUICKSTART_URL = "https://nhadaututtheky.github.io/neural-memory/guides/quickstart/"


# ---------------------------------------------------------------------------
# Embedding auto-detection
# ---------------------------------------------------------------------------

_PROVIDER_PRIORITY: list[dict[str, str]] = [
    {
        "key": "sentence_transformer",
        "module": "sentence_transformers",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "label": "Sentence Transformers (local, multilingual)",
        "install": "pip install neural-memory[embeddings]",
    },
    {
        "key": "gemini",
        "module": "google.generativeai",
        "model": "models/text-embedding-004",
        "label": "Google Gemini (free tier)",
        "env_key": "GEMINI_API_KEY",
    },
    {
        "key": "ollama",
        "module": "ollama",
        "model": "nomic-embed-text",
        "label": "Ollama (local)",
    },
    {
        "key": "openai",
        "module": "openai",
        "model": "text-embedding-3-small",
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
    },
    {
        "key": "openrouter",
        "module": "openai",
        "model": "openai/text-embedding-3-small",
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
    },
]


def _is_module_available(module_name: str) -> bool:
    """Check if a Python module is importable without actually importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def detect_embedding_provider() -> dict[str, str] | None:
    """Auto-detect the best available embedding provider.

    Returns provider dict or None if nothing is available.
    Priority: installed local > cloud with key > nothing.
    """
    for provider in _PROVIDER_PRIORITY:
        module = provider["module"]
        env_key = provider.get("env_key", "")

        if not _is_module_available(module):
            continue

        # Cloud providers need API key
        if env_key and not os.environ.get(env_key):
            continue

        return provider

    return None


def _prompt_install_embeddings() -> dict[str, str] | None:
    """Ask user to install sentence-transformers if nothing is available."""
    typer.echo()
    typer.secho("  No embedding provider detected.", fg=typer.colors.YELLOW)
    typer.echo("  Embeddings enable semantic search — find memories by meaning, not just keywords.")
    typer.echo()
    typer.echo("  Options:")
    typer.echo("    1. Install sentence-transformers (~440MB, local, free)")
    typer.echo("    2. Skip for now (keyword-only search)")
    typer.echo()

    choice = typer.prompt("  Choose", default="1").strip()

    if choice == "1":
        import subprocess

        typer.echo("  Installing sentence-transformers (~440MB)...")
        typer.echo("  This may take a few minutes. Use Ctrl+C to cancel.")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "neural-memory[embeddings]"],
                capture_output=False,  # Show pip output for progress
                text=True,
                timeout=300,
                check=False,
            )
            if result.returncode == 0:
                typer.secho("  Installed successfully.", fg=typer.colors.GREEN)
                return _PROVIDER_PRIORITY[0]  # sentence_transformer
            else:
                typer.secho("  Installation failed. You can retry later:", fg=typer.colors.RED)
                typer.echo(f"    {_PROVIDER_PRIORITY[0]['install']}")
                return None
        except subprocess.TimeoutExpired:
            typer.secho(
                "\n  Installation timed out (5 min). Retry manually:", fg=typer.colors.YELLOW
            )
            typer.echo(f"    {_PROVIDER_PRIORITY[0]['install']}")
            return None
        except KeyboardInterrupt:
            typer.secho("\n  Installation cancelled.", fg=typer.colors.YELLOW)
            return None

    return None


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def enable_config_defaults(*, embedding_provider: dict[str, str] | None = None) -> dict[str, str]:
    """Enable recommended defaults in config.toml.

    Returns dict of what was changed.
    """
    from dataclasses import replace

    from neural_memory.unified_config import get_config

    config = get_config(reload=True)
    changes: dict[str, str] = {}

    new_embedding = config.embedding
    new_dedup = config.dedup

    # Embedding
    if embedding_provider and not config.embedding.enabled:
        new_embedding = replace(
            config.embedding,
            enabled=True,
            provider=embedding_provider["key"],
            model=embedding_provider["model"],
        )
        changes["embedding"] = f"{embedding_provider['key']} ({embedding_provider['model']})"

    # Dedup
    if not config.dedup.enabled:
        new_dedup = replace(config.dedup, enabled=True)
        changes["dedup"] = "enabled"

    if changes:
        updated = replace(config, embedding=new_embedding, dedup=new_dedup)
        updated.save()

    return changes


# ---------------------------------------------------------------------------
# Maintenance script generation
# ---------------------------------------------------------------------------

_BASH_MAINTENANCE = """\
#!/bin/bash
# Neural Memory maintenance script — generated by nmem init --full
# Add to crontab: crontab -e → 0 3 * * * {script_path}
set -e

echo "[$(date)] Neural Memory maintenance starting..."

# 1. Memory decay (forgetting curve)
nmem decay

# 2. Consolidate (prune + mature + dedup)
nmem consolidate --strategy all

# 3. Health check
nmem doctor

echo "[$(date)] Neural Memory maintenance complete."
"""

_POWERSHELL_MAINTENANCE = """\
# Neural Memory maintenance script — generated by nmem init --full
# Schedule via Task Scheduler or run manually

Write-Host "[$(Get-Date)] Neural Memory maintenance starting..."

# 1. Memory decay (forgetting curve)
nmem decay

# 2. Consolidate (prune + mature + dedup)
nmem consolidate --strategy all

# 3. Health check
nmem doctor

Write-Host "[$(Get-Date)] Neural Memory maintenance complete."
"""


def generate_maintenance_script(data_dir: Path) -> Path | None:
    """Generate platform-appropriate maintenance script.

    Returns path to generated script, or None if already exists.
    """
    if os.name == "nt":
        script_name = "maintenance.ps1"
        content = _POWERSHELL_MAINTENANCE
    else:
        script_name = "maintenance.sh"
        content = _BASH_MAINTENANCE

    script_path = data_dir / script_name
    if script_path.exists():
        return None

    content = content.replace("{script_path}", str(script_path))
    script_path.write_text(content, encoding="utf-8")

    # Make executable on Unix
    if os.name != "nt":
        import stat

        st = script_path.stat()
        script_path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP)

    return script_path


# ---------------------------------------------------------------------------
# Summary banner
# ---------------------------------------------------------------------------


def print_full_banner(results: dict[str, str]) -> None:
    """Print post-init banner with guide link."""
    typer.echo()

    # Use ASCII box-drawing that works everywhere
    typer.secho("  +--------------------------------------------------+", dim=True)
    typer.secho("  |                                                  |", dim=True)
    typer.echo("  |  " + typer.style("Neural Memory is ready!", bold=True) + "                   |")
    typer.secho("  |                                                  |", dim=True)
    typer.echo("  |  Quickstart Guide:                                |")
    typer.echo(f"  |  {QUICKSTART_URL}")
    typer.secho("  |                                                  |", dim=True)
    typer.echo(
        "  |  Run " + typer.style("nmem doctor", bold=True) + " to verify your setup        |"
    )
    typer.echo("  |  including MCP server connectivity.            |")
    typer.secho("  |                                                  |", dim=True)
    typer.secho("  +--------------------------------------------------+", dim=True)
    typer.echo()
    typer.echo("  Restart your AI tool to activate memory.")
    typer.echo()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_full_setup(
    *,
    force: bool = False,
    skip_mcp: bool = False,
    skip_skills: bool = False,
) -> dict[str, Any]:
    """Run the full extended setup.

    Returns summary dict with all results.
    """
    from neural_memory.cli.setup import (
        print_summary,
        setup_brain,
        setup_config,
        setup_hooks_claude,
        setup_mcp_claude,
        setup_mcp_cursor,
        setup_skills,
    )
    from neural_memory.unified_config import get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()
    results: dict[str, str] = {}

    # ── Phase 1: Standard init ──────────────────────────────────────────

    # 1. Config
    created = setup_config(data_dir, force=force)
    results["Config"] = f"{data_dir / 'config.toml'} (created)" if created else "already exists"

    # 2. Brain
    brain_name = setup_brain(data_dir)
    results["Brain"] = f"{brain_name} (ready)"

    # 3. MCP
    if skip_mcp:
        results["Claude Code"] = "skipped"
        results["Cursor"] = "skipped"
    else:
        claude_status = setup_mcp_claude()
        status_labels = {
            "added": "MCP server configured",
            "exists": "already configured",
            "not_found": "not detected",
            "failed": "failed",
        }
        results["Claude Code"] = status_labels.get(claude_status, claude_status)

        cursor_status = setup_mcp_cursor()
        cursor_labels = {
            "added": "MCP server configured",
            "exists": "already configured",
            "not_found": "not detected",
            "failed": "failed",
        }
        results["Cursor"] = cursor_labels.get(cursor_status, cursor_status)

    # 4. Hooks
    if not skip_mcp:
        hook_status = setup_hooks_claude()
        hook_labels = {
            "added": "3 hooks installed (PreCompact, Stop, PostToolUse)",
            "exists": "already configured",
            "not_found": "Claude Code not detected",
            "failed": "failed",
        }
        results["Hooks"] = hook_labels.get(hook_status, hook_status)

    # 5. Skills
    if skip_skills:
        results["Skills"] = "skipped"
    else:
        skill_results = setup_skills(force=force)
        if "Skills" in skill_results:
            results["Skills"] = skill_results["Skills"]
        else:
            installed = sum(
                1 for s in skill_results.values() if s in ("installed", "updated", "exists")
            )
            results["Skills"] = f"{installed} skills ready"

    # ── Phase 2: Embedding auto-detect ──────────────────────────────────

    provider = detect_embedding_provider()
    if provider is None:
        provider = _prompt_install_embeddings()

    if provider:
        results["Embeddings"] = f"{provider['label']}"
    else:
        results["Embeddings"] = "skipped (keyword search only)"

    # ── Phase 3: Config defaults ────────────────────────────────────────

    changes = enable_config_defaults(embedding_provider=provider)
    if changes:
        parts = [f"{k}: {v}" for k, v in changes.items()]
        results["Config defaults"] = ", ".join(parts)
    else:
        results["Config defaults"] = "already optimal"

    # ── Phase 4: Maintenance script ─────────────────────────────────────

    script_path = generate_maintenance_script(data_dir)
    if script_path:
        results["Maintenance"] = f"script generated: {script_path.name}"
    else:
        results["Maintenance"] = "script already exists"

    # ── Print ───────────────────────────────────────────────────────────

    print_summary(results)
    print_full_banner(results)

    return {
        "results": results,
        "embedding_provider": provider.get("key") if provider else None,
        "data_dir": str(data_dir),
    }
