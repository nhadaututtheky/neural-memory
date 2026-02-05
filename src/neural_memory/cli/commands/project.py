"""Project management commands."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result
from neural_memory.core.project import Project

project_app = typer.Typer(help="Project scoping for memory organization")


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name")],
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Project description"),
    ] = None,
    duration: Annotated[
        int | None,
        typer.Option("--duration", "-D", help="Duration in days (creates end date)"),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Project tags"),
    ] = None,
    priority: Annotated[
        float,
        typer.Option("--priority", "-p", help="Project priority (default: 1.0)"),
    ] = 1.0,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Create a new project for organizing memories.

    Projects group related memories and enable time-bounded retrieval.

    Examples:
        nmem project create "Q1 Sprint"
        nmem project create "Auth Refactor" --duration 14 --tag backend
        nmem project create "Research" -d "ML exploration" --priority 2.0
    """

    async def _create() -> dict:
        config = get_config()
        storage = await get_storage(config)

        # Check if project with same name exists
        existing = await storage.get_project_by_name(name)
        if existing:
            return {"error": f"Project '{name}' already exists."}

        proj = Project.create(
            name=name,
            description=description or "",
            duration_days=duration,
            tags=set(tags) if tags else None,
            priority=priority,
        )

        await storage.add_project(proj)
        await storage.batch_save()

        response = {
            "message": f"Created project: {name}",
            "project_id": proj.id,
            "name": proj.name,
            "is_ongoing": proj.is_ongoing,
        }

        if proj.end_date:
            response["ends_in_days"] = proj.days_remaining
            response["end_date"] = proj.end_date.isoformat()

        if tags:
            response["tags"] = tags

        return response

    result = asyncio.run(_create())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
            return

        typer.secho(result["message"], fg=typer.colors.GREEN)
        if result.get("ends_in_days") is not None:
            typer.secho(f"  Ends in {result['ends_in_days']} days", fg=typer.colors.BRIGHT_BLACK)
        else:
            typer.secho("  Ongoing (no end date)", fg=typer.colors.BRIGHT_BLACK)


@project_app.command("list")
def project_list(
    active_only: Annotated[
        bool,
        typer.Option("--active", "-a", help="Show only active projects"),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List all projects.

    Examples:
        nmem project list
        nmem project list --active
    """

    async def _list() -> dict:
        config = get_config()
        storage = await get_storage(config)

        projects = await storage.list_projects(active_only=active_only)

        projects_data = []
        for proj in projects:
            # Count memories in project
            memories = await storage.get_project_memories(proj.id)

            proj_data = {
                "id": proj.id,
                "name": proj.name,
                "description": proj.description,
                "is_active": proj.is_active,
                "is_ongoing": proj.is_ongoing,
                "memory_count": len(memories),
                "priority": proj.priority,
                "tags": list(proj.tags),
                "start_date": proj.start_date.isoformat(),
                "end_date": proj.end_date.isoformat() if proj.end_date else None,
            }

            if proj.end_date and proj.is_active:
                proj_data["days_remaining"] = proj.days_remaining

            projects_data.append(proj_data)

        return {
            "projects": projects_data,
            "count": len(projects_data),
            "filter": "active" if active_only else "all",
        }

    result = asyncio.run(_list())

    if json_output:
        output_result(result, True)
    else:
        projects = result.get("projects", [])
        if not projects:
            typer.echo("No projects found. Create one with: nmem project create <name>")
            return

        typer.secho(f"Projects ({result['count']})", fg=typer.colors.CYAN, bold=True)
        typer.echo("-" * 50)

        for proj in projects:
            # Status indicator
            if proj["is_active"]:
                proj_status = typer.style("[ACTIVE]", fg=typer.colors.GREEN)
            else:
                proj_status = typer.style("[ENDED]", fg=typer.colors.BRIGHT_BLACK)

            typer.echo(f"{proj_status} {proj['name']}")

            # Details line
            details = []
            details.append(f"{proj['memory_count']} memories")
            if proj.get("days_remaining") is not None:
                details.append(f"{proj['days_remaining']}d remaining")
            elif proj["is_ongoing"]:
                details.append("ongoing")

            if proj["tags"]:
                details.append(f"tags: {', '.join(proj['tags'])}")

            typer.secho(f"       {' | '.join(details)}", fg=typer.colors.BRIGHT_BLACK)

        typer.echo("-" * 50)


@project_app.command("show")
def project_show(
    name: Annotated[str, typer.Argument(help="Project name")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show project details and its memories.

    Examples:
        nmem project show "Q1 Sprint"
    """

    async def _show() -> dict:
        config = get_config()
        storage = await get_storage(config)

        proj = await storage.get_project_by_name(name)
        if not proj:
            return {"error": f"Project '{name}' not found."}

        # Get memories in project
        memories = await storage.get_project_memories(proj.id)

        # Get memory content
        memories_data = []
        for tm in memories[:20]:  # Limit to 20
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
                    "type": tm.memory_type.value,
                    "priority": tm.priority.name.lower(),
                    "content": content[:80] + "..." if len(content) > 80 else content,
                    "created_at": tm.created_at.isoformat(),
                }
            )

        # Count by type
        type_counts: dict[str, int] = {}
        for tm in memories:
            type_name = tm.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "project": {
                "id": proj.id,
                "name": proj.name,
                "description": proj.description,
                "is_active": proj.is_active,
                "is_ongoing": proj.is_ongoing,
                "priority": proj.priority,
                "tags": list(proj.tags),
                "start_date": proj.start_date.isoformat(),
                "end_date": proj.end_date.isoformat() if proj.end_date else None,
                "days_remaining": proj.days_remaining,
                "duration_days": proj.duration_days,
            },
            "memory_count": len(memories),
            "by_type": type_counts,
            "recent_memories": memories_data,
        }

    result = asyncio.run(_show())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
            return

        proj = result["project"]
        typer.secho(f"\nProject: {proj['name']}", fg=typer.colors.CYAN, bold=True)

        if proj["description"]:
            typer.echo(f"  {proj['description']}")

        typer.echo("")

        # Status
        if proj["is_active"]:
            typer.secho("  Status: ACTIVE", fg=typer.colors.GREEN)
        else:
            typer.secho("  Status: ENDED", fg=typer.colors.BRIGHT_BLACK)

        # Timeline
        typer.echo(f"  Started: {proj['start_date'][:10]}")
        if proj["end_date"]:
            typer.echo(f"  Ends: {proj['end_date'][:10]}")
            if proj["days_remaining"] is not None and proj["days_remaining"] > 0:
                typer.echo(f"  Days remaining: {proj['days_remaining']}")
        else:
            typer.echo("  End: ongoing")

        # Tags
        if proj["tags"]:
            typer.echo(f"  Tags: {', '.join(proj['tags'])}")

        typer.echo("")

        # Memory stats
        typer.secho(f"  Memories: {result['memory_count']}", fg=typer.colors.WHITE, bold=True)
        if result["by_type"]:
            for mem_type, count in sorted(result["by_type"].items(), key=lambda x: -x[1]):
                typer.echo(f"    {mem_type}: {count}")

        # Recent memories
        if result["recent_memories"]:
            typer.echo("")
            typer.secho("  Recent:", fg=typer.colors.WHITE, bold=True)
            for mem in result["recent_memories"][:5]:
                type_badge = f"[{mem['type'][:4].upper()}]"
                typer.echo(f"    {type_badge} {mem['content']}")


@project_app.command("delete")
def project_delete(
    name: Annotated[str, typer.Argument(help="Project name to delete")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Delete a project (memories are preserved but unlinked).

    Examples:
        nmem project delete "Old Project"
        nmem project delete "Temp" --force
    """

    async def _delete() -> dict:
        config = get_config()
        storage = await get_storage(config)

        proj = await storage.get_project_by_name(name)
        if not proj:
            return {"error": f"Project '{name}' not found."}

        # Count memories
        memories = await storage.get_project_memories(proj.id)

        deleted = await storage.delete_project(proj.id)
        if deleted:
            await storage.batch_save()
            return {
                "message": f"Deleted project: {name}",
                "memories_preserved": len(memories),
            }
        else:
            return {"error": "Failed to delete project."}

    # Confirmation
    if not force:

        async def _preview() -> int:
            config = get_config()
            storage = await get_storage(config)
            proj = await storage.get_project_by_name(name)
            if not proj:
                return -1
            memories = await storage.get_project_memories(proj.id)
            return len(memories)

        count = asyncio.run(_preview())
        if count < 0:
            typer.secho(f"Project '{name}' not found.", fg=typer.colors.RED)
            return

        msg = f"Delete project '{name}'?"
        if count > 0:
            msg += f" ({count} memories will be preserved but unlinked)"
        if not typer.confirm(msg):
            typer.echo("Cancelled.")
            return

    result = asyncio.run(_delete())

    if "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        typer.secho(result["message"], fg=typer.colors.GREEN)
        if result.get("memories_preserved", 0) > 0:
            typer.secho(
                f"  {result['memories_preserved']} memories preserved (use 'nmem list' to see them)",
                fg=typer.colors.BRIGHT_BLACK,
            )


@project_app.command("extend")
def project_extend(
    name: Annotated[str, typer.Argument(help="Project name")],
    days: Annotated[int, typer.Argument(help="Days to extend by")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Extend a project's deadline.

    Examples:
        nmem project extend "Q1 Sprint" 7
    """

    async def _extend() -> dict:
        config = get_config()
        storage = await get_storage(config)

        proj = await storage.get_project_by_name(name)
        if not proj:
            return {"error": f"Project '{name}' not found."}

        if proj.is_ongoing:
            return {"error": "Cannot extend ongoing project - it has no end date."}

        try:
            updated = proj.with_extended_deadline(days)
            await storage.update_project(updated)
            await storage.batch_save()

            return {
                "message": f"Extended '{name}' by {days} days",
                "new_end_date": updated.end_date.isoformat() if updated.end_date else None,
                "days_remaining": updated.days_remaining,
            }
        except ValueError as e:
            return {"error": str(e)}

    result = asyncio.run(_extend())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
        else:
            typer.secho(result["message"], fg=typer.colors.GREEN)
            if result.get("days_remaining") is not None:
                typer.secho(
                    f"  New deadline: {result['new_end_date'][:10]} ({result['days_remaining']} days remaining)",
                    fg=typer.colors.BRIGHT_BLACK,
                )
