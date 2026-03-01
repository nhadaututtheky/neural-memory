"""Brain version management commands."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import (
    get_brain_path_auto,
    get_config,
    get_storage,
    output_result,
    run_async,
)

logger = logging.getLogger(__name__)

version_app = typer.Typer(help="Brain version control commands")


@version_app.command("create")
def version_create(
    name: Annotated[str, typer.Argument(help="Version name (must be unique)")],
    description: Annotated[str, typer.Option("--description", "-d", help="Description")] = "",
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Create a version snapshot of the current brain state.

    Examples:
        nmem version create v1-baseline
        nmem version create pre-refactor -d "Before cleanup"
    """

    async def _create() -> dict[str, Any]:
        config = get_config()
        brain_name = config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await get_storage(config)
        try:
            brain = await storage.get_brain(storage._current_brain_id or "")
            if not brain:
                return {"error": "No brain configured"}

            from neural_memory.engine.brain_versioning import VersioningEngine

            engine = VersioningEngine(storage)
            try:
                version = await engine.create_version(brain.id, name, description)
            except ValueError as e:
                logger.error("Version create failed: %s", e)
                return {"error": "Failed to create version: invalid parameters"}

            return {
                "success": True,
                "version_id": version.id,
                "version_name": version.version_name,
                "version_number": version.version_number,
                "neuron_count": version.neuron_count,
                "synapse_count": version.synapse_count,
                "fiber_count": version.fiber_count,
            }
        finally:
            await storage.close()

    result = run_async(_create())
    if json_output:
        output_result(result, True)
    elif "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        typer.secho(
            f"Created version '{result['version_name']}' (#{result['version_number']})",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  Neurons: {result['neuron_count']}")
        typer.echo(f"  Synapses: {result['synapse_count']}")
        typer.echo(f"  Fibers: {result['fiber_count']}")


@version_app.command("list")
def version_list(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max versions")] = 20,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List brain versions.

    Examples:
        nmem version list
        nmem version list --limit 5
    """

    async def _list() -> dict[str, Any]:
        config = get_config()
        brain_name = config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await get_storage(config)
        try:
            brain = await storage.get_brain(storage._current_brain_id or "")
            if not brain:
                return {"error": "No brain configured"}

            from neural_memory.engine.brain_versioning import VersioningEngine

            engine = VersioningEngine(storage)
            versions = await engine.list_versions(brain.id, limit=limit)

            return {
                "versions": [
                    {
                        "id": v.id,
                        "name": v.version_name,
                        "number": v.version_number,
                        "description": v.description,
                        "neurons": v.neuron_count,
                        "synapses": v.synapse_count,
                        "fibers": v.fiber_count,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
                "count": len(versions),
            }
        finally:
            await storage.close()

    result = run_async(_list())
    if json_output:
        output_result(result, True)
    elif "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        versions = result["versions"]
        if not versions:
            typer.echo("No versions found. Create one with: nmem version create <name>")
            return
        typer.echo(f"Brain versions ({result['count']}):")
        for v in versions:
            desc = f" — {v['description']}" if v["description"] else ""
            typer.echo(
                f"  #{v['number']} {v['name']}{desc} ({v['neurons']}n/{v['synapses']}s/{v['fibers']}f)"
            )
            typer.echo(f"       id={v['id'][:12]}...  {v['created_at']}")


@version_app.command("rollback")
def version_rollback(
    version_id: Annotated[str, typer.Argument(help="Version ID to rollback to")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Rollback brain to a previous version.

    Examples:
        nmem version rollback abc123
    """

    async def _rollback() -> dict[str, Any]:
        config = get_config()
        brain_name = config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await get_storage(config)
        try:
            brain = await storage.get_brain(storage._current_brain_id or "")
            if not brain:
                return {"error": "No brain configured"}

            from neural_memory.engine.brain_versioning import VersioningEngine

            engine = VersioningEngine(storage)
            try:
                rollback_v = await engine.rollback(brain.id, version_id)
            except ValueError as e:
                logger.error("Version rollback failed: %s", e)
                return {"error": "Rollback failed: version not found or invalid"}

            return {
                "success": True,
                "version_name": rollback_v.version_name,
                "neuron_count": rollback_v.neuron_count,
                "synapse_count": rollback_v.synapse_count,
                "fiber_count": rollback_v.fiber_count,
            }
        finally:
            await storage.close()

    result = run_async(_rollback())
    if json_output:
        output_result(result, True)
    elif "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        typer.secho(f"Rolled back: {result['version_name']}", fg=typer.colors.GREEN)


@version_app.command("diff")
def version_diff(
    from_version: Annotated[str, typer.Argument(help="Source version ID")],
    to_version: Annotated[str, typer.Argument(help="Target version ID")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Compare two brain versions.

    Examples:
        nmem version diff abc123 def456
    """

    async def _diff() -> dict[str, Any]:
        config = get_config()
        brain_name = config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await get_storage(config)
        try:
            brain = await storage.get_brain(storage._current_brain_id or "")
            if not brain:
                return {"error": "No brain configured"}

            from neural_memory.engine.brain_versioning import VersioningEngine

            engine = VersioningEngine(storage)
            try:
                diff = await engine.diff(brain.id, from_version, to_version)
            except ValueError as e:
                logger.error("Version diff failed: %s", e)
                return {"error": "Diff failed: one or both versions not found"}

            return {
                "summary": diff.summary,
                "neurons_added": len(diff.neurons_added),
                "neurons_removed": len(diff.neurons_removed),
                "neurons_modified": len(diff.neurons_modified),
                "synapses_added": len(diff.synapses_added),
                "synapses_removed": len(diff.synapses_removed),
                "synapses_weight_changed": len(diff.synapses_weight_changed),
                "fibers_added": len(diff.fibers_added),
                "fibers_removed": len(diff.fibers_removed),
            }
        finally:
            await storage.close()

    result = run_async(_diff())
    if json_output:
        output_result(result, True)
    elif "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        typer.echo(f"Diff: {result['summary']}")
        typer.echo(
            f"  Neurons: +{result['neurons_added']} -{result['neurons_removed']} ~{result['neurons_modified']}"
        )
        typer.echo(
            f"  Synapses: +{result['synapses_added']} -{result['synapses_removed']} ~{result['synapses_weight_changed']}"
        )
        typer.echo(f"  Fibers: +{result['fibers_added']} -{result['fibers_removed']}")
