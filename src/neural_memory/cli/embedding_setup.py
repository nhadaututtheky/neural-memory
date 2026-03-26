"""Embedding provider setup — nmem setup embeddings.

Interactive command to configure embedding provider. Lists available
providers, checks installation status, and validates configuration.
"""

from __future__ import annotations

import importlib
import os

import typer

# Provider definitions with detection logic
_PROVIDERS: list[dict[str, str]] = [
    {
        "key": "sentence_transformer",
        "name": "Sentence Transformers",
        "type": "local",
        "module": "sentence_transformers",
        "install": "pip install neural-memory[embeddings]",
        "default_model": "all-MiniLM-L6-v2",
        "multilingual_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "note": "~440MB, runs locally, no API key needed",
        "env_key": "",
    },
    {
        "key": "gemini",
        "name": "Google Gemini",
        "type": "cloud",
        "module": "google.genai",
        "install": "pip install neural-memory[embeddings-gemini]",
        "default_model": "gemini-embedding-001",
        "note": "Free tier, needs GEMINI_API_KEY",
        "env_key": "GEMINI_API_KEY",
    },
    {
        "key": "ollama",
        "name": "Ollama",
        "type": "local",
        "module": "ollama",
        "install": "pip install neural-memory[embeddings]",
        "default_model": "nomic-embed-text",
        "note": "Free, needs Ollama running locally",
        "env_key": "",
    },
    {
        "key": "openai",
        "name": "OpenAI",
        "type": "cloud",
        "module": "openai",
        "install": "pip install neural-memory[embeddings-openai]",
        "default_model": "text-embedding-3-small",
        "note": "Paid, needs OPENAI_API_KEY",
        "env_key": "OPENAI_API_KEY",
    },
    {
        "key": "openrouter",
        "name": "OpenRouter",
        "type": "cloud",
        "module": "openai",
        "install": "pip install neural-memory[embeddings-openrouter]",
        "default_model": "openai/text-embedding-3-small",
        "note": "OpenAI-compatible, needs OPENROUTER_API_KEY",
        "env_key": "OPENROUTER_API_KEY",
    },
]


def _is_installed(module_name: str) -> bool:
    """Check if a Python module is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _has_env_key(key: str) -> bool:
    """Check if an environment variable is set."""
    if not key:
        return True
    return bool(os.environ.get(key))


def run_embedding_setup() -> None:
    """Interactive embedding provider setup."""
    typer.echo()
    typer.secho("  Embedding Provider Setup", bold=True)
    typer.secho("  ────────────────────────", dim=True)
    typer.echo()
    typer.echo("  Embeddings enable semantic search — find similar memories")
    typer.echo("  even when the exact words differ.")
    typer.echo()

    # Show providers with status
    for i, p in enumerate(_PROVIDERS, 1):
        installed = _is_installed(p["module"])
        has_key = _has_env_key(p["env_key"])

        status_parts = []
        if installed:
            status_parts.append(typer.style("installed", fg=typer.colors.GREEN))
        else:
            status_parts.append(typer.style("not installed", fg=typer.colors.YELLOW))
        if p.get("env_key"):
            if has_key:
                status_parts.append(typer.style("key set", fg=typer.colors.GREEN))
            else:
                status_parts.append(typer.style(f"{p['env_key']} not set", fg=typer.colors.YELLOW))

        status = ", ".join(status_parts)
        typer.echo(f"  {i}. {p['name']} ({p['type']}) [{status}]")
        typer.secho(f"     {p['note']}", dim=True)

    typer.echo(f"  {len(_PROVIDERS) + 1}. Disable embeddings")
    typer.echo()

    choice = typer.prompt("  Choose provider", default="1").strip()

    try:
        idx = int(choice) - 1
    except ValueError:
        typer.secho("  Invalid choice.", fg=typer.colors.RED)
        return

    # Disable option
    if idx == len(_PROVIDERS):
        _save_embedding_disabled()
        typer.secho("  Embeddings disabled.", fg=typer.colors.YELLOW)
        return

    if not 0 <= idx < len(_PROVIDERS):
        typer.secho("  Invalid choice.", fg=typer.colors.RED)
        return

    provider = _PROVIDERS[idx]

    # Check installation
    if not _is_installed(provider["module"]):
        typer.echo()
        typer.secho(f"  {provider['name']} is not installed.", fg=typer.colors.YELLOW)
        typer.echo(f"  Install with: {provider['install']}")
        if not typer.confirm("  Continue anyway?", default=False):
            return

    # Check API key for cloud providers
    env_key = provider.get("env_key")
    if env_key and not _has_env_key(env_key):
        typer.echo()
        typer.secho(f"  {env_key} is not set.", fg=typer.colors.YELLOW)
        typer.echo(f"  Set it in your shell: export {env_key}=your-key-here")
        if not typer.confirm("  Continue anyway?", default=False):
            return

    # Model selection
    model = provider["default_model"]
    if provider["key"] == "sentence_transformer":
        typer.echo()
        typer.echo("  Available models:")
        typer.echo(f"    1. {provider['default_model']} (English, fast, 384D)")
        typer.echo(f"    2. {provider['multilingual_model']} (50+ languages, 384D)")
        model_choice = typer.prompt("  Choose model", default="1").strip()
        if model_choice == "2":
            model = provider["multilingual_model"]

    # Save config
    _save_embedding_config(provider["key"], model)

    typer.echo()
    typer.secho(f"  Configured: {provider['name']}", fg=typer.colors.GREEN)
    typer.secho(f"  Model: {model}", dim=True)
    typer.echo("  Restart your AI tool for changes to take effect.")
    typer.echo()


def _save_embedding_config(provider_key: str, model: str) -> None:
    """Save embedding provider to config.toml."""
    from dataclasses import replace

    from neural_memory.unified_config import get_config

    config = get_config(reload=True)
    updated = replace(
        config,
        embedding=replace(config.embedding, enabled=True, provider=provider_key, model=model),
    )
    updated.save()


def _save_embedding_disabled() -> None:
    """Disable embeddings in config.toml."""
    from dataclasses import replace

    from neural_memory.unified_config import get_config

    config = get_config(reload=True)
    updated = replace(config, embedding=replace(config.embedding, enabled=False))
    updated.save()
