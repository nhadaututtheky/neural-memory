"""Information commands: stats, check, status, version."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async
from neural_memory.core.memory_types import MemoryType
from neural_memory.safety.freshness import analyze_freshness, format_age
from neural_memory.safety.sensitive import check_sensitive_content, format_sensitive_warning
from neural_memory.utils.timeutils import utcnow


def stats(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show brain statistics including freshness and memory type analysis.

    Examples:
        nmem stats
        nmem stats --json
    """

    async def _stats() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        enhanced = await storage.get_enhanced_stats(brain.id)

        # Get fibers for freshness analysis
        fibers = await storage.get_fibers(limit=1000)
        created_dates = [f.created_at for f in fibers]
        freshness_report = analyze_freshness(created_dates)

        # Get typed memory statistics
        typed_memories = await storage.find_typed_memories(include_expired=True, limit=10000)
        expired_memories = await storage.get_expired_memories()

        # Count by type
        type_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "normal": 0,
            "low": 0,
            "lowest": 0,
        }

        for tm in typed_memories:
            type_name = tm.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            priority_counts[tm.priority.name.lower()] += 1

        return {
            "brain": brain.name,
            "brain_id": brain.id,
            "neuron_count": enhanced["neuron_count"],
            "synapse_count": enhanced["synapse_count"],
            "fiber_count": enhanced["fiber_count"],
            "db_size_bytes": enhanced.get("db_size_bytes", 0),
            "hot_neurons": enhanced.get("hot_neurons", []),
            "today_fibers_count": enhanced.get("today_fibers_count", 0),
            "synapse_stats": enhanced.get("synapse_stats", {}),
            "neuron_type_breakdown": enhanced.get("neuron_type_breakdown", {}),
            "oldest_memory": enhanced.get("oldest_memory"),
            "newest_memory": enhanced.get("newest_memory"),
            "typed_memory_count": len(typed_memories),
            "expired_count": len(expired_memories),
            "created_at": brain.created_at.isoformat(),
            "freshness": {
                "fresh": freshness_report.fresh,
                "recent": freshness_report.recent,
                "aging": freshness_report.aging,
                "stale": freshness_report.stale,
                "ancient": freshness_report.ancient,
                "average_age_days": round(freshness_report.average_age_days, 1),
            },
            "by_type": type_counts,
            "by_priority": priority_counts,
        }

    result = run_async(_stats())

    if json_output:
        output_result(result, True)
    else:
        typer.echo(f"Brain: {result['brain']}")
        typer.echo(f"Neurons: {result['neuron_count']}")
        typer.echo(f"Synapses: {result['synapse_count']}")
        typer.echo(f"Fibers (memories): {result['fiber_count']}")

        # DB size
        db_size = result.get("db_size_bytes", 0)
        if db_size > 0:
            if db_size >= 1_048_576:
                typer.echo(f"DB size: {db_size / 1_048_576:.1f} MB")
            else:
                typer.echo(f"DB size: {db_size / 1024:.1f} KB")

        # Today's activity
        today_count = result.get("today_fibers_count", 0)
        typer.echo(f"Today's memories: {today_count}")

        # Neuron type breakdown
        neuron_types = result.get("neuron_type_breakdown", {})
        if neuron_types:
            typer.echo("\nNeuron Types:")
            for ntype, count in sorted(neuron_types.items(), key=lambda x: -x[1]):
                typer.echo(f"  {ntype}: {count}")

        # Hot neurons
        hot_neurons = result.get("hot_neurons", [])
        if hot_neurons:
            typer.echo("\nHot Neurons (most accessed):")
            for hn in hot_neurons[:5]:
                content = hn["content"][:50] + "..." if len(hn["content"]) > 50 else hn["content"]
                typer.echo(f"  [{hn['type']}] {content} (freq: {hn['access_frequency']})")

        # Synapse stats
        synapse_stats = result.get("synapse_stats", {})
        if synapse_stats and synapse_stats.get("by_type"):
            typer.echo("\nSynapse Stats:")
            typer.echo(f"  Avg weight: {synapse_stats['avg_weight']}")
            typer.echo(f"  Total reinforcements: {synapse_stats['total_reinforcements']}")

        # Memory time range
        oldest = result.get("oldest_memory")
        newest = result.get("newest_memory")
        if oldest and newest:
            typer.echo(f"\nMemory Range: {oldest[:10]} to {newest[:10]}")

        # Show typed memory stats
        if result.get("typed_memory_count", 0) > 0:
            typer.echo(f"\nTyped Memories: {result['typed_memory_count']}")

            # By type
            by_type = result.get("by_type", {})
            if by_type:
                typer.echo("  By type:")
                for mem_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
                    typer.echo(f"    {mem_type}: {count}")

            # By priority (only show non-zero)
            by_priority = result.get("by_priority", {})
            non_zero_priority = {k: v for k, v in by_priority.items() if v > 0}
            if non_zero_priority:
                typer.echo("  By priority:")
                for pri in ["critical", "high", "normal", "low", "lowest"]:
                    if pri in non_zero_priority:
                        typer.echo(f"    {pri}: {non_zero_priority[pri]}")

            # Expired warning
            if result.get("expired_count", 0) > 0:
                typer.secho(
                    f"\n  [!] {result['expired_count']} expired memories - run 'nmem cleanup' to remove",
                    fg=typer.colors.YELLOW,
                )

        if result.get("freshness") and result["fiber_count"] > 0:
            f = result["freshness"]
            typer.echo("\nMemory Freshness:")
            typer.echo(f"  [+] Fresh (<7d): {f['fresh']}")
            typer.echo(f"  [+] Recent (7-30d): {f['recent']}")
            typer.echo(f"  [~] Aging (30-90d): {f['aging']}")
            typer.echo(f"  [!] Stale (90-365d): {f['stale']}")
            typer.echo(f"  [!!] Ancient (>365d): {f['ancient']}")
            typer.echo(f"  Average age: {f['average_age_days']} days")


def check(
    content: Annotated[str, typer.Argument(help="Content to check for sensitive data")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Check content for sensitive information without storing.

    Examples:
        nmem check "My API_KEY=sk-xxx123"
        nmem check "password: secret123" --json
    """
    matches = check_sensitive_content(content)

    if json_output:
        output_result(
            {
                "sensitive": len(matches) > 0,
                "matches": [
                    {
                        "type": m.type.value,
                        "pattern": m.pattern_name,
                        "severity": m.severity,
                        "redacted": m.redacted(),
                    }
                    for m in matches
                ],
            },
            True,
        )
    else:
        if matches:
            typer.echo(format_sensitive_warning(matches))
        else:
            typer.secho("[OK] No sensitive content detected", fg=typer.colors.GREEN)


def status(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show current brain status, recent activity, and actionable suggestions.

    Unlike 'stats' which shows raw counts, 'status' gives a quick
    overview of what's happening and what you should do next.

    Examples:
        nmem status
        nmem status --json
    """

    async def _status() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured. Run: nmem init"}

        stats_data = await storage.get_stats(brain.id)

        # Last memory timestamp
        fibers = await storage.get_fibers(limit=1)
        last_memory_at = fibers[0].created_at if fibers else None

        # Pending TODOs
        todos = await storage.find_typed_memories(
            memory_type=MemoryType.TODO,
            include_expired=False,
            limit=100,
        )

        # Expired memories
        expired = await storage.get_expired_memories()

        # Today's activity
        all_recent = await storage.get_fibers(limit=100)
        today = utcnow().date()
        today_count = sum(1 for f in all_recent if f.created_at.date() == today)

        # Build suggestions
        suggestions: list[str] = []

        if last_memory_at is not None:
            hours_since = (utcnow() - last_memory_at).total_seconds() / 3600
            if hours_since > 4:
                suggestions.append(
                    f"Last save was {format_age(int(hours_since / 24))} ago — consider saving progress"
                )

        if len(expired) > 0:
            suggestions.append(f"{len(expired)} expired memories — run 'nmem cleanup' to remove")

        if len(todos) > 0:
            suggestions.append(f"{len(todos)} pending TODOs")

        if stats_data["fiber_count"] == 0:
            suggestions.append('No memories yet — try: nmem remember "your first memory"')

        return {
            "brain": brain.name,
            "brain_id": brain.id,
            "neurons": stats_data["neuron_count"],
            "synapses": stats_data["synapse_count"],
            "fibers": stats_data["fiber_count"],
            "today_count": today_count,
            "last_memory_at": last_memory_at.isoformat() if last_memory_at else None,
            "pending_todos": len(todos),
            "expired_count": len(expired),
            "suggestions": suggestions,
        }

    result = run_async(_status())

    if "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if json_output:
        output_result(result, True)
        return

    # Header
    typer.secho(f"Brain: {result['brain']}", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"  Neurons: {result['neurons']}  "
        f"Synapses: {result['synapses']}  "
        f"Fibers: {result['fibers']}"
    )

    # Activity
    typer.echo()
    typer.echo(f"  Today: {result['today_count']} memories")
    if result["last_memory_at"]:
        last_dt = datetime.fromisoformat(result["last_memory_at"])
        hours_ago = (utcnow() - last_dt).total_seconds() / 3600
        if hours_ago < 1:
            age_str = f"{int(hours_ago * 60)}m ago"
        elif hours_ago < 24:
            age_str = f"{int(hours_ago)}h ago"
        else:
            age_str = format_age(int(hours_ago / 24))
        typer.echo(f"  Last save: {age_str}")
    else:
        typer.echo("  Last save: never")

    # TODOs & expired
    if result["pending_todos"] > 0:
        typer.echo(f"  Pending TODOs: {result['pending_todos']}")
    if result["expired_count"] > 0:
        typer.secho(
            f"  Expired: {result['expired_count']}",
            fg=typer.colors.YELLOW,
        )

    # Suggestions
    if result["suggestions"]:
        typer.echo()
        for suggestion in result["suggestions"]:
            typer.secho(f"  > {suggestion}", fg=typer.colors.YELLOW)


def health(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show brain health diagnostics with purity score and recommendations.

    Analyzes connectivity, diversity, freshness, consolidation,
    orphan rate, and activation efficiency to produce a composite
    health grade (A-F) and actionable recommendations.

    Examples:
        nmem health
        nmem health --json
    """

    async def _health() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured. Run: nmem init"}

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(storage)
        report = await engine.analyze(brain.id)

        return {
            "brain": brain.name,
            "grade": report.grade,
            "purity_score": report.purity_score,
            "connectivity": report.connectivity,
            "diversity": report.diversity,
            "freshness": report.freshness,
            "consolidation_ratio": report.consolidation_ratio,
            "orphan_rate": report.orphan_rate,
            "activation_efficiency": report.activation_efficiency,
            "recall_confidence": report.recall_confidence,
            "neuron_count": report.neuron_count,
            "synapse_count": report.synapse_count,
            "fiber_count": report.fiber_count,
            "warnings": [
                {"severity": w.severity.value, "code": w.code, "message": w.message}
                for w in report.warnings
            ],
            "recommendations": list(report.recommendations),
        }

    result = run_async(_health())

    if "error" in result:
        typer.secho(result["error"], fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if json_output:
        output_result(result, True)
        return

    # Header
    grade = result["grade"]
    grade_colors = {
        "A": typer.colors.GREEN,
        "B": typer.colors.CYAN,
        "C": typer.colors.YELLOW,
        "D": typer.colors.RED,
        "F": typer.colors.RED,
    }
    typer.secho(f"Brain: {result['brain']}", fg=typer.colors.CYAN, bold=True)
    typer.secho(
        f"Grade: {grade} ({result['purity_score']}/100)",
        fg=grade_colors.get(grade, typer.colors.WHITE),
        bold=True,
    )
    typer.echo()

    # Metrics with progress bars
    _render_bar("Connectivity", result["connectivity"])
    _render_bar("Diversity", result["diversity"])
    _render_bar("Freshness", result["freshness"])
    _render_bar("Consolidation", result["consolidation_ratio"])
    _render_bar("Orphan rate", result["orphan_rate"], invert=True)
    _render_bar("Activation", result["activation_efficiency"])
    _render_bar("Recall conf.", result["recall_confidence"])

    # Counts
    typer.echo(
        f"\n  Neurons: {result['neuron_count']}  "
        f"Synapses: {result['synapse_count']}  "
        f"Fibers: {result['fiber_count']}"
    )

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        typer.echo("\nWarnings:")
        for w in warnings:
            severity = w["severity"]
            icon = {"critical": "!!", "warning": " !", "info": " ~"}[severity]
            color = {
                "critical": typer.colors.RED,
                "warning": typer.colors.YELLOW,
                "info": typer.colors.BRIGHT_BLACK,
            }[severity]
            typer.secho(f"  [{icon}] {w['message']}", fg=color)

    # Recommendations
    recommendations = result.get("recommendations", [])
    if recommendations:
        typer.echo("\nRecommendations:")
        for rec in recommendations:
            typer.echo(f"  > {rec}")


def _render_bar(label: str, value: float, *, invert: bool = False) -> None:
    """Render an ASCII progress bar for a metric."""
    filled = round(value * 10)
    filled = max(0, min(10, filled))
    bar = "#" * filled + "-" * (10 - filled)
    pct = f"{value:.0%}"
    quality = _quality_label(value, invert=invert)
    typer.echo(f"  {label:<16} [{bar}]  {pct:>4} ({quality})")


def _quality_label(value: float, *, invert: bool = False) -> str:
    """Map a 0-1 score to a quality label."""
    effective = (1.0 - value) if invert else value
    if effective >= 0.8:
        return "excellent"
    if effective >= 0.6:
        return "good"
    if effective >= 0.4:
        return "moderate"
    if effective >= 0.2:
        return "low"
    return "poor"


def version() -> None:
    """Show version information."""
    from neural_memory import __version__

    typer.echo(f"neural-memory v{__version__}")


def register(app: typer.Typer) -> None:
    """Register info commands on the app."""
    app.command()(stats)
    app.command()(check)
    app.command()(status)
    app.command()(health)
    app.command()(version)
