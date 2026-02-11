"""Terminal UI components for NeuralMemory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from neural_memory.cli.storage import PersistentStorage

console = Console()


# =============================================================================
# Color Schemes
# =============================================================================

MEMORY_TYPE_COLORS = {
    "fact": "white",
    "decision": "blue",
    "preference": "magenta",
    "todo": "yellow",
    "insight": "green",
    "context": "cyan",
    "instruction": "bright_blue",
    "error": "red",
    "workflow": "bright_cyan",
    "reference": "bright_black",
}

PRIORITY_COLORS = {
    "critical": "bold red",
    "high": "red",
    "normal": "white",
    "low": "bright_black",
    "lowest": "dim",
}

FRESHNESS_COLORS = {
    "fresh": "green",
    "recent": "bright_green",
    "aging": "yellow",
    "stale": "red",
    "ancient": "bright_black",
}


# =============================================================================
# Dashboard
# =============================================================================


async def render_dashboard(storage: PersistentStorage) -> None:
    """Render a rich dashboard with brain stats and recent activity."""
    from neural_memory.safety.freshness import analyze_freshness, evaluate_freshness

    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        console.print("[red]No brain configured[/red]")
        return

    # Gather data
    stats = await storage.get_stats(brain.id)
    fibers = await storage.get_fibers(limit=100)
    typed_memories = await storage.find_typed_memories(limit=1000)
    expired_memories = await storage.get_expired_memories()

    # Analyze freshness
    created_dates = [f.created_at for f in fibers]
    freshness_report = analyze_freshness(created_dates)

    # Count by type and priority
    type_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for tm in typed_memories:
        type_name = tm.memory_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        pri_name = tm.priority.name.lower()
        priority_counts[pri_name] = priority_counts.get(pri_name, 0) + 1

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(name="stats", ratio=1),
        Layout(name="types", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="freshness", ratio=1),
        Layout(name="recent", ratio=1),
    )

    # Header
    header_text = Text()
    header_text.append("[*] ", style="bold")
    header_text.append("NeuralMemory Dashboard", style="bold cyan")
    header_text.append(f" - {brain.name}", style="bright_black")
    layout["header"].update(Panel(header_text, style="cyan"))

    # Stats panel
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Metric", style="bright_black")
    stats_table.add_column("Value", style="bold")
    stats_table.add_row("Neurons", f"[cyan]{stats['neuron_count']:,}[/cyan]")
    stats_table.add_row("Synapses", f"[blue]{stats['synapse_count']:,}[/blue]")
    stats_table.add_row("Fibers", f"[green]{stats['fiber_count']:,}[/green]")
    stats_table.add_row("Typed Memories", f"[yellow]{len(typed_memories):,}[/yellow]")
    if expired_memories:
        stats_table.add_row(
            "Expired", f"[red]{len(expired_memories)}[/red] [dim](run cleanup)[/dim]"
        )
    layout["stats"].update(Panel(stats_table, title="Brain Stats", border_style="cyan"))

    # Types panel
    types_table = Table(show_header=False, box=None, padding=(0, 1))
    types_table.add_column("Type")
    types_table.add_column("Count", justify="right")
    for mem_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:8]:
        color = MEMORY_TYPE_COLORS.get(mem_type, "white")
        types_table.add_row(f"[{color}]*[/{color}] {mem_type}", str(count))
    layout["types"].update(Panel(types_table, title="By Type", border_style="blue"))

    # Freshness panel
    fresh_table = Table(show_header=False, box=None, padding=(0, 1))
    fresh_table.add_column("Age")
    fresh_table.add_column("Count", justify="right")
    fresh_table.add_column("Bar")

    total = max(freshness_report.total, 1)
    for label, count, color in [
        ("Fresh (<7d)", freshness_report.fresh, "green"),
        ("Recent (7-30d)", freshness_report.recent, "bright_green"),
        ("Aging (30-90d)", freshness_report.aging, "yellow"),
        ("Stale (90-365d)", freshness_report.stale, "red"),
        ("Ancient (>365d)", freshness_report.ancient, "bright_black"),
    ]:
        pct = count / total
        bar = "#" * int(pct * 15) + "-" * (15 - int(pct * 15))
        fresh_table.add_row(f"[{color}]{label}[/{color}]", str(count), f"[{color}]{bar}[/{color}]")

    layout["freshness"].update(Panel(fresh_table, title="Memory Freshness", border_style="green"))

    # Recent memories panel
    recent_table = Table(show_header=False, box=None, padding=(0, 1))
    recent_table.add_column("Memory", overflow="ellipsis", max_width=40)

    recent_typed = sorted(typed_memories, key=lambda x: x.created_at, reverse=True)[:6]
    for tm in recent_typed:
        fiber = await storage.get_fiber(tm.fiber_id)
        content = ""
        if fiber:
            if fiber.summary:
                content = fiber.summary[:35]
            elif fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content[:35]

        if len(content) > 35:
            content = content[:32] + "..."

        type_color = MEMORY_TYPE_COLORS.get(tm.memory_type.value, "white")
        freshness = evaluate_freshness(tm.created_at)
        age_color = FRESHNESS_COLORS.get(freshness.level.value, "white")

        row_text = Text()
        row_text.append(f"[{tm.memory_type.value[:4].upper()}] ", style=type_color)
        row_text.append(content, style="white")
        row_text.append(f" ({freshness.level.value})", style=age_color)
        recent_table.add_row(row_text)

    layout["recent"].update(Panel(recent_table, title="Recent Memories", border_style="yellow"))

    # Footer
    footer_text = Text()
    footer_text.append("Commands: ", style="bright_black")
    footer_text.append("nmem ui", style="cyan")
    footer_text.append(" (browse) ", style="bright_black")
    footer_text.append("nmem graph", style="cyan")
    footer_text.append(" (visualize) ", style="bright_black")
    footer_text.append("nmem recall", style="cyan")
    footer_text.append(" (search)", style="bright_black")
    layout["footer"].update(Panel(footer_text, style="bright_black"))

    console.print(layout)


# =============================================================================
# Memory Browser (UI)
# =============================================================================


async def render_memory_browser(
    storage: PersistentStorage,
    memory_type: str | None = None,
    limit: int = 20,
    search: str | None = None,
) -> None:
    """Render an interactive memory browser."""
    from neural_memory.safety.freshness import evaluate_freshness, format_age

    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        console.print("[red]No brain configured[/red]")
        return

    # Get memories
    mem_type_filter = None
    if memory_type:
        from neural_memory.core.memory_types import MemoryType

        try:
            mem_type_filter = MemoryType(memory_type.lower())
        except ValueError:
            console.print(f"[red]Invalid memory type: {memory_type}[/red]")
            return

    typed_memories = await storage.find_typed_memories(
        memory_type=mem_type_filter,
        limit=limit * 2,  # Get extra for filtering
    )

    # Filter by search if provided
    if search:
        search_lower = search.lower()
        filtered = []
        for tm in typed_memories:
            fiber = await storage.get_fiber(tm.fiber_id)
            if fiber:
                content = fiber.summary or ""
                if not content and fiber.anchor_neuron_id:
                    anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        content = anchor.content
                if search_lower in content.lower():
                    filtered.append((tm, content))
        typed_memories_with_content = filtered[:limit]
    else:
        typed_memories_with_content = []
        for tm in typed_memories[:limit]:
            fiber = await storage.get_fiber(tm.fiber_id)
            content = ""
            if fiber:
                if fiber.summary:
                    content = fiber.summary
                elif fiber.anchor_neuron_id:
                    anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        content = anchor.content
            typed_memories_with_content.append((tm, content))

    if not typed_memories_with_content:
        console.print("[yellow]No memories found.[/yellow]")
        return

    # Create table
    table = Table(
        title="Memory Browser",
        show_lines=True,
        title_style="bold cyan",
        border_style="bright_black",
    )

    table.add_column("#", style="bright_black", width=3)
    table.add_column("Type", style="bold", width=10)
    table.add_column("Priority", width=8)
    table.add_column("Content", overflow="fold")
    table.add_column("Age", width=10)
    table.add_column("Tags", width=15)

    for idx, (tm, content) in enumerate(typed_memories_with_content, 1):
        type_color = MEMORY_TYPE_COLORS.get(tm.memory_type.value, "white")
        priority_color = PRIORITY_COLORS.get(tm.priority.name.lower(), "white")
        freshness = evaluate_freshness(tm.created_at)
        age_color = FRESHNESS_COLORS.get(freshness.level.value, "white")

        # Truncate content
        display_content = content[:80] + "..." if len(content) > 80 else content

        # Format tags
        tags_str = ", ".join(list(tm.tags)[:3]) if tm.tags else "-"
        if tm.tags and len(tm.tags) > 3:
            tags_str += f" +{len(tm.tags) - 3}"

        table.add_row(
            str(idx),
            f"[{type_color}]{tm.memory_type.value}[/{type_color}]",
            f"[{priority_color}]{tm.priority.name.lower()}[/{priority_color}]",
            display_content,
            f"[{age_color}]{format_age(freshness.age_days)}[/{age_color}]",
            f"[bright_black]{tags_str}[/bright_black]",
        )

    # Print summary
    filter_info = []
    if memory_type:
        filter_info.append(f"type={memory_type}")
    if search:
        filter_info.append(f"search='{search}'")

    if filter_info:
        console.print(f"[bright_black]Filters: {', '.join(filter_info)}[/bright_black]")

    console.print(table)
    console.print(
        f"\n[bright_black]Showing {len(typed_memories_with_content)} memories. "
        f"Use --limit N to show more.[/bright_black]"
    )


# =============================================================================
# Graph Visualization
# =============================================================================


async def render_graph(
    storage: PersistentStorage,
    query: str | None = None,
    depth: int = 2,
) -> None:
    """Render a text-based graph visualization."""
    from neural_memory.core.synapse import SynapseType

    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        console.print("[red]No brain configured[/red]")
        return

    # If query provided, find related neurons
    if query:
        from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=query,
            depth=DepthLevel(min(depth, 3)),
            max_tokens=1000,
        )

        if result.confidence < 0.1:
            console.print("[yellow]No relevant memories found for query.[/yellow]")
            return

        # Get fibers matched
        fibers = []
        for fiber_id in result.fibers_matched[:5]:
            fiber = await storage.get_fiber(fiber_id)
            if fiber:
                fibers.append(fiber)
    else:
        # Get recent fibers
        fibers = await storage.get_fibers(limit=5)

    if not fibers:
        console.print("[yellow]No memories to visualize.[/yellow]")
        return

    # Build tree visualization
    tree = Tree(
        "[bold cyan][*] Neural Graph[/bold cyan]",
        guide_style="bright_black",
    )

    synapse_icons = {
        SynapseType.CAUSED_BY: "<-",
        SynapseType.LEADS_TO: "->",
        SynapseType.CO_OCCURS: "<->",
        SynapseType.RELATED_TO: "~",
        SynapseType.SIMILAR_TO: "~~",
        SynapseType.HAPPENED_AT: "@",
        SynapseType.AT_LOCATION: "#",
        SynapseType.INVOLVES: "&",
    }

    for fiber in fibers:
        # Get fiber content
        content = fiber.summary or ""
        if not content and fiber.anchor_neuron_id:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            if anchor:
                content = anchor.content

        display_content = content[:50] + "..." if len(content) > 50 else content

        fiber_branch = tree.add(
            f"[green]*[/green] {display_content}",
        )

        # Get connected neurons
        if fiber.anchor_neuron_id:
            neighbors = await storage.get_neighbors(fiber.anchor_neuron_id, direction="both")

            for neighbor, synapse in neighbors[:5]:
                icon = synapse_icons.get(synapse.type, "─")
                neighbor_content = neighbor.content[:30]
                if len(neighbor.content) > 30:
                    neighbor_content += "..."

                synapse_color = "blue" if synapse.weight > 0.5 else "bright_black"
                fiber_branch.add(
                    f"[{synapse_color}]{icon}[/{synapse_color}] "
                    f"[bright_black]{synapse.type.value}[/bright_black] "
                    f"[white]{neighbor_content}[/white]"
                )

    console.print()
    console.print(tree)
    console.print()

    # Legend
    legend = Table(show_header=False, box=None, padding=(0, 2))
    legend.add_column("Symbol")
    legend.add_column("Meaning")
    legend.add_row("[green]*[/green]", "Memory (Fiber)")
    legend.add_row("->", "leads_to")
    legend.add_row("<-", "caused_by")
    legend.add_row("<->", "co_occurs")
    legend.add_row("@", "happened_at")
    legend.add_row("#", "at_location")

    console.print(Panel(legend, title="Legend", border_style="bright_black"))


# =============================================================================
# Quick Stats (for prompt injection)
# =============================================================================


async def render_quick_stats(storage: PersistentStorage) -> str:
    """Render a compact stats string for context injection."""
    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        return "No brain configured"

    stats = await storage.get_stats(brain.id)
    typed_memories = await storage.find_typed_memories(limit=1000)

    # Count by type
    type_counts: dict[str, int] = {}
    for tm in typed_memories:
        type_name = tm.memory_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    # Format
    parts = [f"Brain: {brain.name}"]
    parts.append(f"Memories: {stats['fiber_count']}")

    if type_counts:
        type_summary = ", ".join(
            f"{count} {t}" for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:3]
        )
        parts.append(f"Types: {type_summary}")

    return " | ".join(parts)
