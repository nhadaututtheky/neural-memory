"""Memory listing and cleanup commands."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result
from neural_memory.core.memory_types import MemoryType, Priority
from neural_memory.safety.freshness import evaluate_freshness, format_age


def list_memories(
    memory_type: Annotated[
        str | None,
        typer.Option("--type", "-T", help="Filter by memory type (fact, decision, todo, etc.)"),
    ] = None,
    min_priority: Annotated[
        int | None,
        typer.Option("--min-priority", "-p", help="Minimum priority (0-10)"),
    ] = None,
    project_name: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Filter by project name"),
    ] = None,
    show_expired: Annotated[
        bool,
        typer.Option("--expired", "-e", help="Show only expired memories"),
    ] = False,
    include_expired: Annotated[
        bool,
        typer.Option("--include-expired", help="Include expired memories in results"),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum number of results")] = 20,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List memories with filtering by type, priority, project, and status.

    Memory types: fact, decision, preference, todo, insight, context,
                  instruction, error, workflow, reference

    Examples:
        nmem list                               # List all recent memories
        nmem list --type todo                   # List all TODOs
        nmem list --type decision -p 7          # High priority decisions
        nmem list --expired                     # Show expired memories
        nmem list --type todo --expired         # Expired TODOs (need cleanup)
        nmem list --project "Q1 Sprint"         # Memories in a project
    """

    async def _list() -> dict:
        config = get_config()
        storage = await get_storage(config)

        # Parse memory type if provided
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type.lower())
            except ValueError:
                valid_types = ", ".join(t.value for t in MemoryType)
                return {"error": f"Invalid memory type. Valid types: {valid_types}"}

        # Parse priority
        priority = None
        if min_priority is not None:
            priority = Priority.from_int(min_priority)

        # Look up project if specified
        project_id = None
        if project_name:
            proj = await storage.get_project_by_name(project_name)
            if not proj:
                return {"error": f"Project '{project_name}' not found."}
            project_id = proj.id

        # Handle expired-only mode
        if show_expired:
            expired_memories = await storage.get_expired_memories()
            if mem_type:
                expired_memories = [tm for tm in expired_memories if tm.memory_type == mem_type]

            memories_data = []
            for tm in expired_memories[:limit]:
                fiber = await storage.get_fiber(tm.fiber_id)
                content = ""
                if fiber:
                    if fiber.summary:
                        content = fiber.summary
                    elif fiber.anchor_neuron_id:
                        anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                        if anchor:
                            content = anchor.content

                memories_data.append(
                    {
                        "fiber_id": tm.fiber_id,
                        "type": tm.memory_type.value,
                        "priority": tm.priority.name.lower(),
                        "content": content[:100] + "..." if len(content) > 100 else content,
                        "expired_days_ago": abs(tm.days_until_expiry)
                        if tm.days_until_expiry
                        else 0,
                        "created_at": tm.created_at.isoformat(),
                    }
                )

            return {
                "memories": memories_data,
                "count": len(memories_data),
                "filter": "expired",
                "type_filter": memory_type,
            }

        # Normal listing with filters
        typed_memories = await storage.find_typed_memories(
            memory_type=mem_type,
            min_priority=priority,
            include_expired=include_expired,
            project_id=project_id,
            limit=limit,
        )

        # If no typed memories, fall back to listing fibers
        if not typed_memories:
            fibers = await storage.get_fibers(limit=limit)
            memories_data = []
            for fiber in fibers:
                content = fiber.summary
                if not content and fiber.anchor_neuron_id:
                    anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        content = anchor.content

                freshness = evaluate_freshness(fiber.created_at)
                memories_data.append(
                    {
                        "fiber_id": fiber.id,
                        "type": "unknown",
                        "priority": "normal",
                        "content": content[:100] + "..."
                        if content and len(content) > 100
                        else content or "",
                        "age": format_age(freshness.age_days),
                        "created_at": fiber.created_at.isoformat(),
                    }
                )

            return {
                "memories": memories_data,
                "count": len(memories_data),
                "note": "No typed memories found. Showing raw fibers.",
            }

        # Build response with typed memories
        memories_data = []
        for tm in typed_memories:
            fiber = await storage.get_fiber(tm.fiber_id)
            content = ""
            if fiber:
                if fiber.summary:
                    content = fiber.summary
                elif fiber.anchor_neuron_id:
                    anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        content = anchor.content

            freshness = evaluate_freshness(tm.created_at)
            expiry_info = None
            if tm.expires_at:
                days = tm.days_until_expiry
                if days is not None:
                    expiry_info = f"{days}d" if days > 0 else "EXPIRED"

            memories_data.append(
                {
                    "fiber_id": tm.fiber_id,
                    "type": tm.memory_type.value,
                    "priority": tm.priority.name.lower(),
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "age": format_age(freshness.age_days),
                    "expires": expiry_info,
                    "verified": tm.provenance.verified,
                    "created_at": tm.created_at.isoformat(),
                }
            )

        return {
            "memories": memories_data,
            "count": len(memories_data),
            "type_filter": memory_type,
            "min_priority": min_priority,
            "project_filter": project_name,
        }

    result = asyncio.run(_list())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
            return

        memories = result.get("memories", [])
        if not memories:
            typer.echo("No memories found.")
            return

        if result.get("note"):
            typer.secho(result["note"], fg=typer.colors.YELLOW)
            typer.echo("")

        # Display header
        filter_parts = []
        if result.get("type_filter"):
            filter_parts.append(f"type={result['type_filter']}")
        if result.get("min_priority"):
            filter_parts.append(f"priority>={result['min_priority']}")
        if result.get("project_filter"):
            filter_parts.append(f"project={result['project_filter']}")
        if result.get("filter") == "expired":
            filter_parts.append("EXPIRED")

        header = f"Memories ({result['count']})"
        if filter_parts:
            header += f" [{', '.join(filter_parts)}]"
        typer.secho(header, fg=typer.colors.CYAN, bold=True)
        typer.echo("-" * 60)

        # Display memories
        for mem in memories:
            # Type indicator
            type_colors = {
                "todo": typer.colors.YELLOW,
                "decision": typer.colors.BLUE,
                "error": typer.colors.RED,
                "fact": typer.colors.WHITE,
                "preference": typer.colors.MAGENTA,
                "insight": typer.colors.GREEN,
            }
            type_color = type_colors.get(mem["type"], typer.colors.WHITE)

            # Priority indicator
            priority_indicators = {
                "critical": "[!!!]",
                "high": "[!!]",
                "normal": "[+]",
                "low": "[.]",
                "lowest": "[_]",
            }
            priority_ind = priority_indicators.get(mem["priority"], "[+]")

            # Build line
            type_badge = f"[{mem['type'][:4].upper()}]"
            content = mem.get("content", "")[:60]
            if len(mem.get("content", "")) > 60:
                content += "..."

            typer.echo(f"{priority_ind} ", nl=False)
            typer.secho(type_badge, fg=type_color, nl=False)
            typer.echo(f" {content}")

            # Second line with metadata
            meta_parts = []
            if mem.get("age"):
                meta_parts.append(mem["age"])
            if mem.get("expires"):
                if mem["expires"] == "EXPIRED":
                    meta_parts.append(typer.style("EXPIRED", fg=typer.colors.RED))
                else:
                    meta_parts.append(f"expires: {mem['expires']}")
            if mem.get("verified"):
                meta_parts.append("verified")

            if meta_parts:
                typer.secho(f"     {' | '.join(meta_parts)}", fg=typer.colors.BRIGHT_BLACK)

        typer.echo("-" * 60)


def cleanup(
    expired_only: Annotated[
        bool,
        typer.Option("--expired", "-e", help="Only clean up expired memories"),
    ] = True,
    memory_type: Annotated[
        str | None,
        typer.Option("--type", "-T", help="Only clean up specific memory type"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be deleted without deleting"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Clean up expired or old memories.

    Examples:
        nmem cleanup --expired              # Remove all expired memories
        nmem cleanup --expired --dry-run    # Preview what would be removed
        nmem cleanup --type context         # Remove expired context memories
    """

    async def _cleanup() -> dict:
        config = get_config()
        storage = await get_storage(config)

        # Parse memory type if provided
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type.lower())
            except ValueError:
                valid_types = ", ".join(t.value for t in MemoryType)
                return {"error": f"Invalid memory type. Valid types: {valid_types}"}

        # Get expired memories
        expired_memories = await storage.get_expired_memories()

        # Filter by type if specified
        if mem_type:
            expired_memories = [tm for tm in expired_memories if tm.memory_type == mem_type]

        if not expired_memories:
            return {"message": "No expired memories to clean up.", "deleted": 0}

        # Build preview
        to_delete = []
        for tm in expired_memories:
            fiber = await storage.get_fiber(tm.fiber_id)
            content = ""
            if fiber:
                if fiber.summary:
                    content = fiber.summary[:50]
                elif fiber.anchor_neuron_id:
                    anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        content = anchor.content[:50]

            to_delete.append(
                {
                    "fiber_id": tm.fiber_id,
                    "type": tm.memory_type.value,
                    "content": content,
                    "expired_at": tm.expires_at.isoformat() if tm.expires_at else None,
                }
            )

        if dry_run:
            return {
                "dry_run": True,
                "would_delete": to_delete,
                "count": len(to_delete),
            }

        # Actually delete
        deleted_count = 0
        for tm in expired_memories:
            # Delete typed memory
            await storage.delete_typed_memory(tm.fiber_id)
            # Optionally delete the fiber too
            await storage.delete_fiber(tm.fiber_id)
            deleted_count += 1

        await storage.batch_save()

        return {
            "message": f"Cleaned up {deleted_count} expired memories.",
            "deleted": deleted_count,
            "details": to_delete,
        }

    # Confirmation for non-dry-run
    if not dry_run and not force:
        # First do a dry run to show count
        async def _preview() -> int:
            config = get_config()
            storage = await get_storage(config)
            expired = await storage.get_expired_memories()
            if memory_type:
                try:
                    mem_type = MemoryType(memory_type.lower())
                    expired = [tm for tm in expired if tm.memory_type == mem_type]
                except ValueError:
                    pass
            return len(expired)

        count = asyncio.run(_preview())
        if count == 0:
            typer.echo("No expired memories to clean up.")
            return

        if not typer.confirm(f"Delete {count} expired memories? This cannot be undone."):
            typer.echo("Cancelled.")
            return

    result = asyncio.run(_cleanup())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
            return

        if result.get("dry_run"):
            typer.secho(f"Would delete {result['count']} memories:", fg=typer.colors.YELLOW)
            for item in result["would_delete"][:10]:
                typer.echo(f"  [{item['type']}] {item['content']}...")
            if result["count"] > 10:
                typer.echo(f"  ... and {result['count'] - 10} more")
        else:
            typer.secho(result["message"], fg=typer.colors.GREEN)


def register(app: typer.Typer) -> None:
    """Register listing commands on the app."""
    app.command(name="list")(list_memories)
    app.command()(cleanup)
