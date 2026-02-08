"""Shared CLI helpers for configuration, storage, and output formatting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, TypeVar

import typer

from neural_memory.cli.config import CLIConfig
from neural_memory.cli.storage import PersistentStorage

T = TypeVar("T")

# Track storages created during a CLI command so we can close them before
# the event loop shuts down (prevents "Event loop is closed" noise from
# aiosqlite's background thread).
_active_storages: list[Any] = []


def get_config() -> CLIConfig:
    """Get CLI configuration."""
    return CLIConfig.load()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async CLI command with proper storage cleanup.

    Replaces bare ``asyncio.run()`` to ensure aiosqlite connections are
    closed *before* the event loop is torn down.
    """

    async def _with_cleanup() -> T:
        try:
            return await coro
        finally:
            for storage in _active_storages:
                if hasattr(storage, "close"):
                    try:
                        await storage.close()
                    except Exception:
                        pass
            _active_storages.clear()

    return asyncio.run(_with_cleanup())


async def get_storage(
    config: CLIConfig,
    *,
    force_shared: bool = False,
    force_local: bool = False,
    force_sqlite: bool = False,
) -> PersistentStorage:
    """
    Get storage for current brain.

    Args:
        config: CLI configuration
        force_shared: Override config to use remote shared mode
        force_local: Override config to use local JSON mode
        force_sqlite: Override config to use local SQLite mode

    Returns:
        Storage instance (local JSON, local SQLite, or remote shared)
    """
    # Remote shared mode (via server)
    use_shared = (config.is_shared_mode or force_shared) and not force_local
    if use_shared:
        from neural_memory.storage.shared_store import SharedStorage

        storage = SharedStorage(
            server_url=config.shared.server_url,
            brain_id=config.current_brain,
            timeout=config.shared.timeout,
            api_key=config.shared.api_key,
        )
        await storage.connect()
        _active_storages.append(storage)
        return storage  # type: ignore[return-value]

    # SQLite mode (unified config - shared file-based storage)
    if config.use_sqlite or force_sqlite:
        from neural_memory.unified_config import get_shared_storage

        storage = await get_shared_storage(config.current_brain)
        _active_storages.append(storage)
        return storage  # type: ignore[return-value]

    # Legacy JSON mode
    brain_path = config.get_brain_path()
    return await PersistentStorage.load(brain_path)


def output_result(data: dict, as_json: bool = False) -> None:
    """Output result in appropriate format."""
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        # Human-readable format
        if "error" in data:
            typer.secho(f"Error: {data['error']}", fg=typer.colors.RED)
        elif "answer" in data:
            typer.echo(data["answer"])

            # Show freshness warnings
            if data.get("freshness_warnings"):
                typer.echo("")
                for warning in data["freshness_warnings"]:
                    typer.secho(warning, fg=typer.colors.YELLOW)

            # Show metadata
            meta_parts = []
            if data.get("confidence") is not None:
                meta_parts.append(f"confidence: {data['confidence']:.2f}")
            if data.get("neurons_activated"):
                meta_parts.append(f"neurons: {data['neurons_activated']}")
            if data.get("oldest_memory_age"):
                meta_parts.append(f"oldest: {data['oldest_memory_age']}")

            if meta_parts:
                typer.secho(f"\n[{', '.join(meta_parts)}]", fg=typer.colors.BRIGHT_BLACK)

            # Show routing info if present
            if data.get("routing"):
                r = data["routing"]
                typer.secho(
                    f"\n[routing: {r['query_type']}, depth: {r['suggested_depth']}, "
                    f"confidence: {r['confidence']}]",
                    fg=typer.colors.BRIGHT_BLACK,
                )

        elif "message" in data:
            typer.secho(data["message"], fg=typer.colors.GREEN)

            # Show memory type info
            type_parts = []
            if data.get("memory_type"):
                type_parts.append(f"type: {data['memory_type']}")
            if data.get("priority"):
                type_parts.append(f"priority: {data['priority']}")
            if data.get("expires_in_days") is not None:
                type_parts.append(f"expires: {data['expires_in_days']}d")
            if data.get("project"):
                type_parts.append(f"project: {data['project']}")
            if type_parts:
                typer.secho(f"  [{', '.join(type_parts)}]", fg=typer.colors.BRIGHT_BLACK)

            # Show warnings if any
            if data.get("warnings"):
                for warning in data["warnings"]:
                    typer.secho(warning, fg=typer.colors.YELLOW)

        elif "context" in data:
            typer.echo(data["context"])
        else:
            typer.echo(str(data))
