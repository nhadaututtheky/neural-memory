"""Core memory commands: remember, todo, recall, context."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result
from neural_memory.core.memory_types import (
    DEFAULT_EXPIRY_DAYS,
    MemoryType,
    Priority,
    TypedMemory,
    suggest_memory_type,
)
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.extraction.parser import QueryParser
from neural_memory.extraction.router import QueryRouter
from neural_memory.safety.freshness import (
    FreshnessLevel,
    analyze_freshness,
    evaluate_freshness,
    format_age,
    get_freshness_indicator,
)
from neural_memory.safety.sensitive import (
    check_sensitive_content,
    filter_sensitive_content,
    format_sensitive_warning,
)


def remember(
    content: Annotated[str, typer.Argument(help="Content to remember")],
    tags: Annotated[
        list[str] | None, typer.Option("--tag", "-t", help="Tags for the memory")
    ] = None,
    memory_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-T",
            help="Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detected if not specified)",
        ),
    ] = None,
    priority: Annotated[
        int | None,
        typer.Option("--priority", "-p", help="Priority 0-10 (0=lowest, 5=normal, 10=critical)"),
    ] = None,
    expires: Annotated[
        int | None,
        typer.Option("--expires", "-e", help="Days until this memory expires"),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Associate with a project (by name)"),
    ] = None,
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage for this command")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Store even if sensitive content detected")
    ] = False,
    redact: Annotated[
        bool, typer.Option("--redact", "-r", help="Auto-redact sensitive content before storing")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Store a new memory (type auto-detected if not specified).

    Examples:
        nmem remember "Fixed auth bug by adding null check"
        nmem remember "We decided to use PostgreSQL" --type decision
        nmem remember "Need to refactor auth module" --type todo --priority 7
        nmem remember "API_KEY=xxx" --redact
    """
    # Check for sensitive content
    sensitive_matches = check_sensitive_content(content, min_severity=2)

    if sensitive_matches and not force and not redact:
        warning = format_sensitive_warning(sensitive_matches)
        typer.echo(warning)
        raise typer.Exit(1)

    # Redact if requested
    store_content = content
    if redact and sensitive_matches:
        store_content, _ = filter_sensitive_content(content)
        typer.secho(f"Redacted {len(sensitive_matches)} sensitive item(s)", fg=typer.colors.YELLOW)

    # Determine memory type
    if memory_type:
        try:
            mem_type = MemoryType(memory_type.lower())
        except ValueError:
            valid_types = ", ".join(t.value for t in MemoryType)
            typer.secho(f"Invalid memory type. Valid types: {valid_types}", fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        mem_type = suggest_memory_type(store_content)

    # Determine expiry
    expiry_days = expires
    if expiry_days is None:
        expiry_days = DEFAULT_EXPIRY_DAYS.get(mem_type)

    # Determine priority
    mem_priority = Priority.from_int(priority) if priority is not None else Priority.NORMAL

    async def _remember() -> dict:
        config = get_config()
        storage = await get_storage(config, force_shared=shared)

        brain_id = (
            storage._current_brain_id
            if hasattr(storage, "_current_brain_id")
            else config.current_brain
        )
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        # Look up project if specified
        project_id = None
        if project:
            proj = await storage.get_project_by_name(project)
            if not proj:
                return {
                    "error": f"Project '{project}' not found. Create it with: nmem project create \"{project}\""
                }
            project_id = proj.id

        encoder = MemoryEncoder(storage, brain.config)

        # Disable auto-save for batch operations during encoding
        storage.disable_auto_save()

        result = await encoder.encode(
            content=store_content,
            timestamp=datetime.now(),
            tags=set(tags) if tags else None,
        )

        # Create and store typed memory metadata
        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=mem_type,
            priority=mem_priority,
            source="user_input",
            expires_in_days=expiry_days,
            tags=set(tags) if tags else None,
            project_id=project_id,
        )
        await storage.add_typed_memory(typed_mem)

        # Save once after encoding
        await storage.batch_save()

        response = {
            "message": f"Remembered: {store_content[:50]}{'...' if len(store_content) > 50 else ''}",
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "priority": mem_priority.name.lower(),
            "neurons_created": len(result.neurons_created),
            "neurons_linked": len(result.neurons_linked),
            "synapses_created": len(result.synapses_created),
        }

        # Add project info
        if project_id:
            response["project"] = project

        # Add expiry info
        if typed_mem.expires_at:
            response["expires_in_days"] = typed_mem.days_until_expiry

        # Add warnings
        warnings = []
        if force and sensitive_matches:
            warnings.append(
                f"[!] Stored with {len(sensitive_matches)} sensitive item(s) - consider using --redact"
            )
        if warnings:
            response["warnings"] = warnings

        return response

    result = asyncio.run(_remember())
    output_result(result, json_output)


def todo(
    task: Annotated[str, typer.Argument(help="Task to remember")],
    priority: Annotated[
        int,
        typer.Option(
            "--priority", "-p", help="Priority 0-10 (default: 5=normal, 7=high, 10=critical)"
        ),
    ] = 5,
    project: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Associate with a project"),
    ] = None,
    expires: Annotated[
        int | None,
        typer.Option("--expires", "-e", help="Days until expiry (default: 30)"),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Tags for the task"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Quick shortcut to add a TODO memory.

    Examples:
        nmem todo "Fix the login bug"
        nmem todo "Review PR #123" --priority 7
    """
    # Determine expiry (default 30 days for todos)
    expiry_days = expires if expires is not None else 30
    mem_priority = Priority.from_int(priority)

    async def _todo() -> dict:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        # Look up project if specified
        project_id = None
        if project:
            proj = await storage.get_project_by_name(project)
            if not proj:
                return {
                    "error": f"Project '{project}' not found. Create it with: nmem project create \"{project}\""
                }
            project_id = proj.id

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        result = await encoder.encode(
            content=task,
            timestamp=datetime.now(),
            tags=set(tags) if tags else None,
        )

        # Create TODO typed memory
        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=MemoryType.TODO,
            priority=mem_priority,
            source="user_input",
            expires_in_days=expiry_days,
            tags=set(tags) if tags else None,
            project_id=project_id,
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        response = {
            "message": f"TODO: {task[:50]}{'...' if len(task) > 50 else ''}",
            "fiber_id": result.fiber.id,
            "memory_type": "todo",
            "priority": mem_priority.name.lower(),
            "expires_in_days": typed_mem.days_until_expiry,
        }

        if project_id:
            response["project"] = project

        return response

    result = asyncio.run(_todo())
    output_result(result, json_output)


def recall(
    query: Annotated[str, typer.Argument(help="Query to search memories")],
    depth: Annotated[
        int | None,
        typer.Option("--depth", "-d", help="Search depth (0=instant, 1=context, 2=habit, 3=deep)"),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", "-m", help="Max tokens in response")
    ] = 500,
    min_confidence: Annotated[
        float, typer.Option("--min-confidence", "-c", help="Minimum confidence threshold (0.0-1.0)")
    ] = 0.0,
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage for this command")
    ] = False,
    show_age: Annotated[
        bool, typer.Option("--show-age", "-a", help="Show memory ages in results")
    ] = True,
    show_routing: Annotated[
        bool, typer.Option("--show-routing", "-R", help="Show query routing info")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Query memories with intelligent routing (query type auto-detected).

    Examples:
        nmem recall "What did I do with auth?"
        nmem recall "meetings with Alice" --depth 2
        nmem recall "Why did the build fail?" --show-routing
        nmem recall "project status" --min-confidence 0.5
    """

    async def _recall() -> dict:
        config = get_config()
        storage = await get_storage(config, force_shared=shared)

        brain_id = (
            storage._current_brain_id
            if hasattr(storage, "_current_brain_id")
            else config.current_brain
        )
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        # Parse and route query
        parser = QueryParser()
        router = QueryRouter()
        stimulus = parser.parse(query, reference_time=datetime.now())
        route = router.route(stimulus)

        # Use router's suggested depth if not specified
        if depth is not None:
            depth_level = DepthLevel(depth)
        else:
            depth_level = DepthLevel(min(route.suggested_depth, 3))

        pipeline = ReflexPipeline(storage, brain.config)

        result = await pipeline.query(
            query=query,
            depth=depth_level,
            max_tokens=max_tokens,
            reference_time=datetime.now(),
        )

        # Check confidence threshold
        if result.confidence < min_confidence:
            return {
                "answer": f"No memories found with confidence >= {min_confidence:.2f}",
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "below_threshold": True,
            }

        # Gather freshness information from matched fibers
        freshness_warnings: list[str] = []
        oldest_age = 0

        if result.fibers_matched:
            for fiber_id in result.fibers_matched:
                fiber = await storage.get_fiber(fiber_id)
                if fiber:
                    freshness = evaluate_freshness(fiber.created_at)
                    if freshness.warning:
                        freshness_warnings.append(freshness.warning)
                    if freshness.age_days > oldest_age:
                        oldest_age = freshness.age_days

        response = {
            "answer": result.context or "No relevant memories found.",
            "confidence": result.confidence,
            "depth_used": result.depth_used.value,
            "neurons_activated": result.neurons_activated,
            "fibers_matched": result.fibers_matched,
            "latency_ms": result.latency_ms,
        }

        # Add routing info if requested
        if show_routing:
            response["routing"] = {
                "query_type": route.primary.value,
                "confidence": route.confidence.name.lower(),
                "suggested_depth": route.suggested_depth,
                "use_embeddings": route.use_embeddings,
                "time_weighted": route.time_weighted,
                "signals": list(route.signals)[:5],  # Limit signals shown
            }

        if show_age and oldest_age > 0:
            response["oldest_memory_age"] = format_age(oldest_age)

        if freshness_warnings:
            # Deduplicate warnings
            unique_warnings = list(dict.fromkeys(freshness_warnings))[:3]
            response["freshness_warnings"] = unique_warnings

        return response

    result = asyncio.run(_recall())
    output_result(result, json_output)


def context(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of recent memories")] = 10,
    fresh_only: Annotated[
        bool, typer.Option("--fresh-only", help="Only include memories < 30 days old")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Get recent context (for injecting into AI conversations).

    Examples:
        nmem context
        nmem context --limit 5 --json
        nmem context --fresh-only
    """

    async def _context() -> dict:
        config = get_config()
        storage = await get_storage(config)

        # Get recent fibers
        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)

        if not fibers:
            return {"context": "No memories stored yet.", "count": 0}

        # Filter by freshness if requested
        now = datetime.now()
        if fresh_only:
            fresh_fibers = []
            for fiber in fibers:
                freshness = evaluate_freshness(fiber.created_at, now)
                if freshness.level in (FreshnessLevel.FRESH, FreshnessLevel.RECENT):
                    fresh_fibers.append(fiber)
            fibers = fresh_fibers[:limit]

        # Build context string with age indicators
        context_parts = []
        fiber_data = []

        for fiber in fibers:
            freshness = evaluate_freshness(fiber.created_at, now)
            indicator = get_freshness_indicator(freshness.level)
            age_str = format_age(freshness.age_days)

            fiber_content = fiber.summary
            if not fiber_content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    fiber_content = anchor.content

            if fiber_content:
                context_parts.append(f"{indicator} [{age_str}] {fiber_content}")
                fiber_data.append(
                    {
                        "id": fiber.id,
                        "summary": fiber_content,
                        "created_at": fiber.created_at.isoformat(),
                        "age": age_str,
                        "freshness": freshness.level.value,
                    }
                )

        context_str = "\n".join(context_parts) if context_parts else "No context available."

        # Analyze overall freshness
        created_dates = [f.created_at for f in fibers]
        freshness_report = analyze_freshness(created_dates, now)

        return {
            "context": context_str,
            "count": len(fiber_data),
            "fibers": fiber_data,
            "freshness_summary": {
                "fresh": freshness_report.fresh,
                "recent": freshness_report.recent,
                "aging": freshness_report.aging,
                "stale": freshness_report.stale,
                "ancient": freshness_report.ancient,
            },
        }

    result = asyncio.run(_context())
    output_result(result, json_output)


def register(app: typer.Typer) -> None:
    """Register memory commands on the app."""
    app.command()(remember)
    app.command()(todo)
    app.command()(recall)
    app.command()(context)
