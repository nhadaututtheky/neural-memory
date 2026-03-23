"""Interactive first-run wizard for NeuralMemory.

Guides new users through brain setup, embedding provider selection,
and a test encode/recall cycle. Called by `nmem init --wizard`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from neural_memory.cli.setup import (
    print_summary,
    setup_config,
    setup_hooks_claude,
    setup_mcp_claude,
    setup_mcp_cursor,
    setup_skills,
)
from neural_memory.unified_config import get_neuralmemory_dir

# Provider metadata for display
_PROVIDERS: list[dict[str, str]] = [
    {
        "key": "sentence_transformer",
        "name": "Sentence Transformers (local)",
        "install": "pip install neural-memory[embeddings]",
        "note": "~440MB disk, runs locally, no API key",
    },
    {
        "key": "gemini",
        "name": "Google Gemini (cloud)",
        "install": "pip install neural-memory[embeddings-gemini]",
        "note": "Free tier available, needs GEMINI_API_KEY",
    },
    {
        "key": "ollama",
        "name": "Ollama (local)",
        "install": "pip install neural-memory[embeddings]",
        "note": "Requires Ollama running locally",
    },
    {
        "key": "openai",
        "name": "OpenAI (cloud)",
        "install": "pip install neural-memory[embeddings-openai]",
        "note": "Paid, needs OPENAI_API_KEY",
    },
    {
        "key": "openrouter",
        "name": "OpenRouter (cloud)",
        "install": "pip install neural-memory[embeddings-openrouter]",
        "note": "OpenAI-compatible, needs OPENROUTER_API_KEY",
    },
]


def run_wizard(*, force: bool = False) -> None:
    """Run the interactive first-run wizard."""
    typer.echo()
    typer.secho("  NeuralMemory Setup Wizard", bold=True)
    typer.secho("  ─────────────────────────", dim=True)
    typer.echo()

    data_dir = get_neuralmemory_dir()
    results: dict[str, str] = {}

    # Step 1: Brain name
    brain_name = _step_brain_name()

    # Step 2: Config + brain
    created = setup_config(data_dir, force=force)
    results["Config"] = f"{data_dir / 'config.toml'} (created)" if created else "already exists"

    _setup_brain_with_name(data_dir, brain_name)
    results["Brain"] = f"{brain_name} (ready)"

    # Step 3: Embedding provider
    provider = _step_embedding_provider()
    if provider:
        _save_embedding_config(data_dir, provider)
        results["Embedding"] = f"{provider} (configured)"
    else:
        results["Embedding"] = "skipped (can set up later with: nmem setup embeddings)"

    # Step 4: MCP auto-config
    claude_status = setup_mcp_claude()
    _format_mcp_result(results, "Claude Code", claude_status)

    cursor_status = setup_mcp_cursor()
    _format_mcp_result(results, "Cursor", cursor_status)

    # Step 5: Hooks
    hook_status = setup_hooks_claude()
    hook_labels = {
        "added": "hooks installed",
        "exists": "hooks already configured",
        "not_found": "Claude Code not detected",
        "failed": "failed to configure",
    }
    results["Hooks"] = hook_labels.get(hook_status, hook_status)

    # Step 6: Skills
    skill_results = setup_skills(force=force)
    if "Skills" in skill_results:
        results["Skills"] = skill_results["Skills"]
    else:
        installed = sum(1 for s in skill_results.values() if s in ("installed", "updated"))
        existing = sum(1 for s in skill_results.values() if s == "exists")
        results["Skills"] = f"{installed} installed, {existing} existing"

    # Step 7: Test memory (optional)
    _step_test_memory(brain_name)

    # Summary
    print_summary(results)
    typer.echo("  Restart your AI tool to activate memory.")
    typer.echo()


def _step_brain_name() -> str:
    """Ask user for brain name."""
    typer.secho("  Step 1: Brain Name", bold=True)
    typer.echo("  Your brain stores all memories. Most users use 'default'.")
    typer.echo()

    name = str(typer.prompt("  Brain name", default="default")).strip()
    if not name:
        name = "default"
    typer.echo()
    return name


def _step_embedding_provider() -> str:
    """Ask user to pick an embedding provider."""
    typer.secho("  Step 2: Embedding Provider (optional)", bold=True)
    typer.echo("  Embeddings enable semantic search (find similar memories).")
    typer.echo("  You can skip this and set up later.")
    typer.echo()

    for i, p in enumerate(_PROVIDERS, 1):
        typer.echo(f"    {i}. {p['name']}")
        typer.secho(f"       {p['note']}", dim=True)

    typer.echo(f"    {len(_PROVIDERS) + 1}. Skip for now")
    typer.echo()

    choice = typer.prompt(
        "  Choose provider",
        default=str(len(_PROVIDERS) + 1),
    ).strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(_PROVIDERS):
            provider = _PROVIDERS[idx]
            typer.echo()
            typer.secho(f"  Selected: {provider['name']}", fg=typer.colors.GREEN)
            typer.secho(f"  Install:  {provider['install']}", dim=True)
            typer.echo()
            return provider["key"]
    except ValueError:
        pass

    typer.echo()
    return ""


def _step_test_memory(brain_name: str) -> None:
    """Optionally store and recall a test memory."""
    typer.secho("  Step 3: Test Memory (optional)", bold=True)

    if not typer.confirm("  Store a test memory to verify setup?", default=True):
        typer.echo()
        return

    typer.echo()
    content = typer.prompt(
        "  Memory content",
        default="NeuralMemory was set up successfully!",
    ).strip()

    if not content:
        return

    try:
        from neural_memory.cli._helpers import get_config, get_storage, run_async

        async def _test() -> dict[str, Any]:
            config = get_config()
            storage = await get_storage(config)
            try:
                from neural_memory.core.brain import BrainConfig
                from neural_memory.engine.encoder import MemoryEncoder

                encoder = MemoryEncoder(storage, BrainConfig())
                result = await encoder.encode(content)
                return {"success": True, "neuron_id": result.fiber.anchor_neuron_id}
            finally:
                await storage.close()

        result = run_async(_test())
        if result.get("success"):
            typer.secho(
                f"  Memory stored! (neuron: {result['neuron_id'][:8]})",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("  Could not store memory.", fg=typer.colors.YELLOW)
    except Exception:
        typer.secho("  Could not store memory (setup may need restart).", fg=typer.colors.YELLOW)

    typer.echo()


def _setup_brain_with_name(data_dir: Path, name: str) -> None:
    """Ensure brain DB exists with given name."""
    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    db_path = brains_dir / f"{name}.db"
    if not db_path.exists():
        db_path.touch()

    # Update config to use this brain
    config_path = data_dir / "config.toml"
    if config_path.exists():
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        if config.current_brain != name:
            from dataclasses import replace as dc_replace

            updated = dc_replace(config, current_brain=name)
            updated.save()


def _save_embedding_config(data_dir: Path, provider: str) -> None:
    """Update config.toml with embedding provider."""
    from neural_memory.unified_config import get_config

    config = get_config(reload=True)
    from dataclasses import replace

    updated = replace(config, embedding=replace(config.embedding, enabled=True, provider=provider))
    updated.save()


def _format_mcp_result(results: dict[str, str], label: str, status: str) -> None:
    """Format MCP setup status."""
    labels = {
        "added": "configured",
        "exists": "already configured",
        "not_found": "not detected",
        "failed": "failed to configure",
    }
    results[label] = labels.get(status, status)
