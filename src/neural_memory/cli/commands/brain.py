"""Brain management commands."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import (
    get_brain_path_auto,
    get_config,
    get_storage,
    output_result,
    run_async,
)
from neural_memory.safety.freshness import analyze_freshness
from neural_memory.safety.sensitive import check_sensitive_content

brain_app = typer.Typer(help="Brain management commands")


@brain_app.command("list")
def brain_list(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List available brains.

    Examples:
        nmem brain list
        nmem brain list --json
    """
    config = get_config()
    brains = config.list_brains()
    current = config.current_brain

    if json_output:
        output_result({"brains": brains, "current": current}, True)
    else:
        if not brains:
            typer.echo("No brains found. Create one with: nmem brain create <name>")
            return

        typer.echo("Available brains:")
        for brain in brains:
            marker = " *" if brain == current else ""
            typer.echo(f"  {brain}{marker}")


@brain_app.command("use")
def brain_use(
    name: Annotated[str, typer.Argument(help="Brain name to switch to")],
) -> None:
    """Switch to a different brain.

    Examples:
        nmem brain use work
        nmem brain use personal
    """
    config = get_config()

    if name not in config.list_brains():
        typer.secho(
            f"Brain '{name}' not found. Create it with: nmem brain create {name}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    config.current_brain = name
    config.save()
    typer.secho(f"Switched to brain: {name}", fg=typer.colors.GREEN)


@brain_app.command("create")
def brain_create(
    name: Annotated[str, typer.Argument(help="Name for the new brain")],
    use: Annotated[
        bool, typer.Option("--use", "-u", help="Switch to the new brain after creating")
    ] = True,
) -> None:
    """Create a new brain.

    Examples:
        nmem brain create work
        nmem brain create personal --no-use
    """

    async def _create() -> None:
        config = get_config()

        if name in config.list_brains():
            typer.secho(f"Brain '{name}' already exists.", fg=typer.colors.RED)
            raise typer.Exit(1)

        # Create new brain by loading storage (which creates if not exists)
        await get_storage(config, brain_name=name)

        if use:
            config.current_brain = name
            config.save()

        typer.secho(f"Created brain: {name}", fg=typer.colors.GREEN)
        if use:
            typer.echo(f"Now using: {name}")

    run_async(_create())


@brain_app.command("export")
def brain_export(
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path")] = None,
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Brain name (default: current)")
    ] = None,
    exclude_sensitive: Annotated[
        bool,
        typer.Option("--exclude-sensitive", "-s", help="Exclude memories with sensitive content"),
    ] = False,
) -> None:
    """Export brain to JSON file.

    Examples:
        nmem brain export
        nmem brain export -o backup.json
        nmem brain export --exclude-sensitive -o safe.json
    """

    async def _export() -> None:
        config = get_config()
        brain_name = name or config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            typer.secho(f"Brain '{brain_name}' not found.", fg=typer.colors.RED)
            raise typer.Exit(1)

        storage = await get_storage(config, brain_name=brain_name)
        snapshot = await storage.export_brain(storage._current_brain_id or "")

        # Filter sensitive content if requested
        neurons = snapshot.neurons
        excluded_count = 0

        if exclude_sensitive:
            filtered_neurons = []
            excluded_neuron_ids = set()

            for neuron in neurons:
                content = neuron.get("content", "")
                matches = check_sensitive_content(content, min_severity=2)
                if matches:
                    excluded_neuron_ids.add(neuron["id"])
                    excluded_count += 1
                else:
                    filtered_neurons.append(neuron)

            neurons = filtered_neurons

            # Also filter synapses connected to excluded neurons
            synapses = [
                s
                for s in snapshot.synapses
                if s["source_id"] not in excluded_neuron_ids
                and s["target_id"] not in excluded_neuron_ids
            ]

            # Update fiber neuron references
            fibers = []
            for fiber in snapshot.fibers:
                fiber_neuron_ids = set(fiber.get("neuron_ids", []))
                if not fiber_neuron_ids.intersection(excluded_neuron_ids):
                    fibers.append(fiber)
        else:
            synapses = snapshot.synapses
            fibers = snapshot.fibers

        export_data = {
            "brain_id": snapshot.brain_id,
            "brain_name": snapshot.brain_name,
            "exported_at": snapshot.exported_at.isoformat(),
            "version": snapshot.version,
            "neurons": neurons,
            "synapses": synapses,
            "fibers": fibers,
            "config": snapshot.config,
            "metadata": snapshot.metadata,
        }

        if output:
            from pathlib import Path

            output_path = Path(output).resolve()
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, default=str)
            except OSError as exc:
                typer.secho(f"Failed to write export file: {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(1) from exc
            typer.secho(f"Exported to: {output_path}", fg=typer.colors.GREEN)
            if excluded_count > 0:
                typer.secho(
                    f"Excluded {excluded_count} neurons with sensitive content",
                    fg=typer.colors.YELLOW,
                )
        else:
            typer.echo(json.dumps(export_data, indent=2, default=str))

    run_async(_export())


@brain_app.command("import")
def brain_import(
    file: Annotated[str, typer.Argument(help="JSON file to import")],
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Name for imported brain")
    ] = None,
    use: Annotated[bool, typer.Option("--use", "-u", help="Switch to imported brain")] = True,
    scan_sensitive: Annotated[
        bool, typer.Option("--scan", help="Scan for sensitive content before importing")
    ] = True,
) -> None:
    """Import brain from JSON file.

    Examples:
        nmem brain import backup.json
        nmem brain import shared-brain.json --name shared
        nmem brain import untrusted.json --scan
    """
    from neural_memory.core.brain import BrainSnapshot

    async def _import() -> None:
        from pathlib import Path

        file_path = Path(file).resolve()
        if not file_path.is_file():
            typer.secho("File not found or not a regular file", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Scan for sensitive content
        if scan_sensitive:
            sensitive_count = 0
            for neuron in data.get("neurons", []):
                content = neuron.get("content", "")
                matches = check_sensitive_content(content, min_severity=2)
                if matches:
                    sensitive_count += 1

            if sensitive_count > 0:
                typer.secho(
                    f"[!] Found {sensitive_count} neurons with potentially sensitive content",
                    fg=typer.colors.YELLOW,
                )
                if not typer.confirm("Continue importing?"):
                    raise typer.Exit(0)

        brain_name = name or data.get("brain_name", "imported")
        config = get_config()

        if brain_name in config.list_brains():
            typer.secho(
                f"Brain '{brain_name}' already exists. Use --name to specify different name.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        # Create snapshot
        snapshot = BrainSnapshot(
            brain_id=data["brain_id"],
            brain_name=brain_name,
            exported_at=datetime.fromisoformat(data["exported_at"]),
            version=data["version"],
            neurons=data["neurons"],
            synapses=data["synapses"],
            fibers=data["fibers"],
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )

        # Load/create storage and import
        storage = await get_storage(config, brain_name=brain_name)
        await storage.import_brain(snapshot, storage._current_brain_id)
        await storage.save()

        if use:
            config.current_brain = brain_name
            config.save()

        typer.secho(f"Imported brain: {brain_name}", fg=typer.colors.GREEN)
        typer.echo(f"  Neurons: {len(data['neurons'])}")
        typer.echo(f"  Synapses: {len(data['synapses'])}")
        typer.echo(f"  Fibers: {len(data['fibers'])}")

    run_async(_import())


@brain_app.command("delete")
def brain_delete(
    name: Annotated[str, typer.Argument(help="Brain name to delete")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Delete a brain.

    Examples:
        nmem brain delete old-brain
        nmem brain delete temp --force
    """
    config = get_config()

    if name not in config.list_brains():
        typer.secho(f"Brain '{name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if name == config.current_brain:
        typer.secho(
            "Cannot delete current brain. Switch to another brain first.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete brain '{name}'? This cannot be undone.")
        if not confirm:
            typer.echo("Cancelled.")
            return

    brain_path = get_brain_path_auto(config, name)
    brain_path.unlink()
    typer.secho(f"Deleted brain: {name}", fg=typer.colors.GREEN)


def _scan_sensitive_neurons(neurons: list[Any]) -> list[dict[str, Any]]:
    """Scan neurons for sensitive content, return summary dicts."""
    result = []
    for neuron in neurons:
        matches = check_sensitive_content(neuron.content, min_severity=2)
        if matches:
            result.append(
                {
                    "id": neuron.id,
                    "type": neuron.type.value,
                    "sensitive_types": [m.type.value for m in matches],
                }
            )
    return result


def _compute_health_score(sensitive_count: int, freshness_report: Any) -> tuple[int, list[str]]:
    """Compute health score (0-100) and list of issues."""
    score = 100
    issues: list[str] = []

    if sensitive_count:
        penalty = min(30, sensitive_count * 5)
        score -= penalty
        issues.append(f"{sensitive_count} neurons with sensitive content")

    stale_ratio = (freshness_report.stale + freshness_report.ancient) / max(
        1, freshness_report.total
    )
    if stale_ratio > 0.5:
        score -= 20
        issues.append(f"{stale_ratio * 100:.0f}% of memories are stale/ancient")
    elif stale_ratio > 0.2:
        score -= 10
        issues.append(f"{stale_ratio * 100:.0f}% of memories are stale/ancient")

    return max(0, score), issues


def _display_health(result: dict[str, Any]) -> None:
    """Pretty-print health report to terminal."""
    if "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
        return

    score = result["health_score"]
    color, indicator = (
        (typer.colors.GREEN, "[OK]")
        if score >= 80
        else (typer.colors.YELLOW, "[~]")
        if score >= 50
        else (typer.colors.RED, "[!!]")
    )

    typer.echo(f"\nBrain: {result['brain']}")
    typer.secho(f"Health Score: {indicator} {score}/100", fg=color)

    if result["issues"]:
        typer.echo("\nIssues:")
        for issue in result["issues"]:
            typer.secho(f"  [!] {issue}", fg=typer.colors.YELLOW)

    if result["sensitive_content"]["count"] > 0:
        typer.echo(f"\nSensitive content: {result['sensitive_content']['count']} neurons")
        typer.secho(
            "  Run 'nmem brain export --exclude-sensitive' for safe export",
            fg=typer.colors.BRIGHT_BLACK,
        )

    f = result["freshness"]
    if f["total"] > 0:
        typer.echo(f"\nMemory freshness ({f['total']} total):")
        typer.echo(f"  [+] Fresh/Recent: {f['fresh'] + f['recent']}")
        typer.echo(f"  [~] Aging: {f['aging']}")
        typer.echo(f"  [!!] Stale/Ancient: {f['stale'] + f['ancient']}")


@brain_app.command("transplant")
def brain_transplant(
    source: Annotated[str, typer.Argument(help="Source brain name to transplant from")],
    tags: Annotated[list[str] | None, typer.Option("--tag", "-t", help="Filter by tags")] = None,
    memory_types: Annotated[
        list[str] | None, typer.Option("--type", help="Filter by memory types")
    ] = None,
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="Conflict resolution strategy")
    ] = "prefer_local",
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Transplant memories from another brain into the current brain.

    Examples:
        nmem brain transplant expert-brain --tag python --tag api
        nmem brain transplant work-brain --type fact --type decision
        nmem brain transplant shared --strategy prefer_recent
    """

    async def _transplant() -> dict[str, Any]:
        from neural_memory.engine.brain_transplant import TransplantFilter, transplant
        from neural_memory.engine.merge import ConflictStrategy

        config = get_config()
        target_name = config.current_brain
        source_path = get_brain_path_auto(config, source)
        target_path = get_brain_path_auto(config, target_name)

        if not source_path.exists():
            return {"error": f"Source brain '{source}' not found."}
        if not target_path.exists():
            return {"error": f"Target brain '{target_name}' not found."}

        source_storage = await get_storage(config, brain_name=source)
        target_storage = await get_storage(config, brain_name=target_name)
        try:
            source_brain = await source_storage.get_brain(source_storage._current_brain_id or "")
            target_brain = await target_storage.get_brain(target_storage._current_brain_id or "")

            if not source_brain:
                return {"error": f"Source brain '{source}' has no data."}
            if not target_brain:
                return {"error": f"Target brain '{target_name}' has no data."}

            try:
                merge_strategy = ConflictStrategy(strategy)
            except ValueError:
                valid = [s.value for s in ConflictStrategy]
                return {"error": f"Unknown strategy '{strategy}'. Use: {valid}"}

            filt = TransplantFilter(
                tags=frozenset(tags) if tags else None,
                memory_types=frozenset(memory_types) if memory_types else None,
            )

            result = await transplant(
                source_storage=source_storage,
                target_storage=target_storage,
                source_brain_id=source_brain.id,
                target_brain_id=target_brain.id,
                filt=filt,
                strategy=merge_strategy,
            )

            return {
                "success": True,
                "neurons_transplanted": result.neurons_transplanted,
                "synapses_transplanted": result.synapses_transplanted,
                "fibers_transplanted": result.fibers_transplanted,
                "source": source,
                "target": target_name,
            }
        finally:
            await source_storage.close()
            await target_storage.close()

    result = run_async(_transplant())
    if json_output:
        output_result(result, True)
    elif "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED)
    else:
        typer.secho(
            f"Transplanted from '{result['source']}' → '{result['target']}'",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  Neurons: {result['neurons_transplanted']}")
        typer.echo(f"  Synapses: {result['synapses_transplanted']}")
        typer.echo(f"  Fibers: {result['fibers_transplanted']}")


@brain_app.command("health")
def brain_health(
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Brain name (default: current)")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Check brain health (freshness, sensitive content).

    Examples:
        nmem brain health
        nmem brain health --name work --json
    """

    async def _health() -> dict[str, Any]:
        config = get_config()
        brain_name = name or config.current_brain
        brain_path = get_brain_path_auto(config, brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await get_storage(config, brain_name=brain_name)
        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        neurons = await storage.find_neurons(limit=10000)
        fibers = await storage.get_fibers(limit=10000)

        sensitive_neurons = _scan_sensitive_neurons(neurons)
        freshness_report = analyze_freshness([f.created_at for f in fibers])
        health_score, issues = _compute_health_score(len(sensitive_neurons), freshness_report)

        return {
            "brain": brain_name,
            "health_score": health_score,
            "issues": issues,
            "sensitive_content": {
                "count": len(sensitive_neurons),
                "neurons": sensitive_neurons[:5],
            },
            "freshness": {
                "total": freshness_report.total,
                "fresh": freshness_report.fresh,
                "recent": freshness_report.recent,
                "aging": freshness_report.aging,
                "stale": freshness_report.stale,
                "ancient": freshness_report.ancient,
            },
        }

    result = run_async(_health())

    if json_output:
        output_result(result, True)
    else:
        _display_health(result)
