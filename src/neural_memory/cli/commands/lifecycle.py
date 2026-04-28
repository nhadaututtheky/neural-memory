"""Memory lifecycle CLI commands: forget, show, edit, pin, unpin.

Mirrors the MCP tools nmem_forget, nmem_show, nmem_edit, and nmem_pin so that
the CLI can drive the same operations without an MCP server. Useful for
scripting (cron cleanup), debugging stale recall results, and operating on
a brain when MCP is unavailable.

Each command shares a small adapter that lets the CLI call into the existing
MCP handler bodies, so behavior stays identical across surfaces.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async
from neural_memory.mcp.lifecycle_handler import LifecycleHandler
from neural_memory.mcp.provenance_handler import ProvenanceHandler
from neural_memory.mcp.train_handler import TrainHandler

if TYPE_CHECKING:
    from neural_memory.cli.storage import PersistentStorage
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class _CliMcpFacade(LifecycleHandler, ProvenanceHandler, TrainHandler):
    """Bridge that lets the CLI invoke MCP handler bodies unchanged.

    The MCP handlers expect ``self.config`` and ``self.get_storage()``. Inheriting
    from the three mixin classes and providing those two attributes is the
    simplest way to reuse the existing implementations.
    """

    def __init__(self, storage: NeuralStorage, config: Any) -> None:
        self._storage = storage
        self.config = config

    async def get_storage(self) -> NeuralStorage:
        return self._storage


async def _build_facade(*, shared: bool = False) -> tuple[PersistentStorage, _CliMcpFacade]:
    from neural_memory.unified_config import get_config as get_unified_config

    cli_config = get_config()
    storage = await get_storage(cli_config, force_shared=shared)
    unified = get_unified_config()
    return storage, _CliMcpFacade(storage, unified)


def _exit_on_error(result: dict[str, Any], *, json_output: bool) -> None:
    """Handle ``{"error": ...}`` payloads consistently across commands."""
    if "error" not in result:
        return
    if json_output:
        output_result(result, as_json=True)
    else:
        typer.secho(f"Error: {result['error']}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def forget(
    memory_id: Annotated[str, typer.Argument(help="Fiber or neuron ID to forget")],
    hard: Annotated[
        bool,
        typer.Option(
            "--hard",
            help="Permanent deletion (cascade cleanup). Default is soft delete (expire now).",
        ),
    ] = False,
    reason: Annotated[
        str,
        typer.Option("--reason", "-r", help="Reason for forgetting (logged for audit)"),
    ] = "",
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Soft- or hard-delete a memory by fiber ID (or neuron ID with --hard).

    Soft delete sets expires_at=now; consolidation removes the fiber later.
    Hard delete drops the fiber + typed_memory immediately (CASCADE handles
    fiber_neurons).

    Examples:
        nmem forget 256545b4-0b83-42b3-9291-31ef8563f05b
        nmem forget 256545b4-... --hard --reason "duplicate of newer entry"
    """

    async def _forget() -> dict[str, Any]:
        _storage, facade = await _build_facade(shared=shared)
        return await facade._forget({"memory_id": memory_id, "hard": hard, "reason": reason or ""})

    result = run_async(_forget())
    _exit_on_error(result, json_output=json_output)

    if json_output:
        output_result(result, as_json=True)
        return

    typer.secho(result.get("message", result.get("status", "forgotten")), fg=typer.colors.GREEN)
    typer.secho(
        f"  [memory_id: {result.get('memory_id', memory_id)}]", fg=typer.colors.BRIGHT_BLACK
    )


def show(
    memory_id: Annotated[str, typer.Argument(help="Fiber or neuron ID to inspect")],
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show full content + metadata for a memory by ID.

    Works for typed fibers, untyped fibers, and bare neurons. Useful for
    debugging when recall returns an ID that other commands can't resolve.

    Examples:
        nmem show 256545b4-0b83-42b3-9291-31ef8563f05b
        nmem show 256545b4-... --json
    """

    async def _show() -> dict[str, Any]:
        _storage, facade = await _build_facade(shared=shared)
        return await facade._show({"memory_id": memory_id})

    result = run_async(_show())
    _exit_on_error(result, json_output=json_output)

    if json_output:
        output_result(result, as_json=True)
        return

    typer.secho(f"Memory {result['memory_id']}", fg=typer.colors.CYAN)
    if result.get("memory_type"):
        typer.echo(f"  type:        {result['memory_type']}")
    elif result.get("neuron_type"):
        typer.echo(f"  neuron_type: {result['neuron_type']}")
    if result.get("priority") is not None:
        typer.echo(f"  priority:    {result['priority']}")
    if result.get("tags"):
        typer.echo(f"  tags:        {', '.join(result['tags'])}")
    if result.get("created_at"):
        typer.echo(f"  created:     {result['created_at']}")
    if result.get("expires_at"):
        typer.echo(f"  expires:     {result['expires_at']}")
    if result.get("warning"):
        typer.secho(f"  warning:     {result['warning']}", fg=typer.colors.YELLOW)
    typer.echo("")
    typer.echo(result.get("content") or "(no content)")
    syns = result.get("synapses", []) or []
    if syns:
        typer.secho(f"\n[{len(syns)} synapse(s)]", fg=typer.colors.BRIGHT_BLACK)


def edit(
    memory_id: Annotated[str, typer.Argument(help="Fiber or neuron ID to edit")],
    new_type: Annotated[
        str | None,
        typer.Option("--type", "-T", help="New memory type"),
    ] = None,
    new_content: Annotated[
        str | None,
        typer.Option("--content", "-c", help="New content for the anchor neuron"),
    ] = None,
    new_priority: Annotated[
        int | None,
        typer.Option("--priority", "-p", help="New priority 0-10"),
    ] = None,
    new_tier: Annotated[
        str | None,
        typer.Option("--tier", help="New tier: hot, warm, or cold"),
    ] = None,
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Edit a memory's type, content, priority, or tier.

    Examples:
        nmem edit 256545b4-... --type decision
        nmem edit 256545b4-... --priority 8 --tier hot
        nmem edit 256545b4-... --content "Updated summary"
    """
    if all(v is None for v in (new_type, new_content, new_priority, new_tier)):
        typer.secho(
            "Error: provide at least one of --type, --content, --priority, --tier.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    args: dict[str, Any] = {"memory_id": memory_id}
    if new_type is not None:
        args["type"] = new_type
    if new_content is not None:
        args["content"] = new_content
    if new_priority is not None:
        args["priority"] = new_priority
    if new_tier is not None:
        args["tier"] = new_tier

    async def _edit() -> dict[str, Any]:
        _storage, facade = await _build_facade(shared=shared)
        return await facade._edit(args)

    result = run_async(_edit())
    _exit_on_error(result, json_output=json_output)

    if json_output:
        output_result(result, as_json=True)
        return

    typer.secho(f"Edited {result['memory_id']}", fg=typer.colors.GREEN)
    for change in result.get("changes", []):
        typer.echo(f"  - {change}")


def pin(
    fiber_id: Annotated[str, typer.Argument(help="Fiber ID to pin (or 'list' to list pinned)")],
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage")
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="When fiber_id='list', cap number of results"),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Pin a fiber as permanent KB (skips decay/prune; promotes tier to HOT).

    Use 'list' as the argument to list currently-pinned fibers.

    Examples:
        nmem pin 256545b4-0b83-42b3-9291-31ef8563f05b
        nmem pin list
        nmem pin list --limit 100
    """
    args: dict[str, Any]
    if fiber_id == "list":
        args = {"action": "list", "limit": limit}
    else:
        args = {"action": "pin", "fiber_ids": [fiber_id]}

    async def _pin() -> dict[str, Any]:
        _storage, facade = await _build_facade(shared=shared)
        return await facade._pin(args)

    result = run_async(_pin())
    _exit_on_error(result, json_output=json_output)

    if json_output:
        output_result(result, as_json=True)
        return

    if fiber_id == "list":
        fibers = result.get("fibers", [])
        typer.secho(
            f"Pinned fibers ({result.get('pinned_count', len(fibers))})", fg=typer.colors.CYAN
        )
        for f in fibers:
            typer.echo(
                f"  {f['fiber_id'][:8]}  [{f.get('type', 'unknown')}]  "
                f"{(f.get('summary') or '')[:60]}"
            )
        return

    typer.secho(result.get("message", "pinned"), fg=typer.colors.GREEN)


def unpin(
    fiber_id: Annotated[str, typer.Argument(help="Fiber ID to unpin")],
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Unpin a fiber (removes grounding; allows decay/prune again).

    Examples:
        nmem unpin 256545b4-0b83-42b3-9291-31ef8563f05b
    """

    async def _unpin() -> dict[str, Any]:
        _storage, facade = await _build_facade(shared=shared)
        return await facade._pin({"action": "unpin", "fiber_ids": [fiber_id]})

    result = run_async(_unpin())
    _exit_on_error(result, json_output=json_output)

    if json_output:
        output_result(result, as_json=True)
        return

    typer.secho(result.get("message", "unpinned"), fg=typer.colors.GREEN)


def register(app: typer.Typer) -> None:
    """Register lifecycle commands on the app."""
    app.command()(forget)
    app.command()(show)
    app.command()(edit)
    app.command()(pin)
    app.command()(unpin)
