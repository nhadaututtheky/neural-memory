"""Neural Memory CLI main entry point."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import Annotated, Optional

import typer

from neural_memory.cli.config import CLIConfig
from neural_memory.cli.storage import PersistentStorage
from neural_memory.core.memory_types import (
    DEFAULT_EXPIRY_DAYS,
    MemoryType,
    Priority,
    TypedMemory,
    suggest_memory_type,
)
from neural_memory.core.project import Project
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.extraction.parser import QueryParser
from neural_memory.extraction.router import QueryRouter, QueryType
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

# Main app
app = typer.Typer(
    name="nmem",
    help="Neural Memory - Reflex-based memory for AI agents",
    no_args_is_help=True,
)

# Brain subcommand
brain_app = typer.Typer(help="Brain management commands")
app.add_typer(brain_app, name="brain")

# Project subcommand
project_app = typer.Typer(help="Project scoping for memory organization")
app.add_typer(project_app, name="project")

# Shared mode subcommand
shared_app = typer.Typer(help="Real-time brain sharing configuration")
app.add_typer(shared_app, name="shared")


def get_config() -> CLIConfig:
    """Get CLI configuration."""
    return CLIConfig.load()


async def get_storage(
    config: CLIConfig,
    *,
    force_shared: bool = False,
    force_local: bool = False,
) -> PersistentStorage:
    """
    Get storage for current brain.

    Args:
        config: CLI configuration
        force_shared: Override config to use shared mode
        force_local: Override config to use local mode

    Returns:
        Storage instance (local or shared based on config)
    """
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
        return storage  # type: ignore

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
                    fg=typer.colors.BRIGHT_BLACK
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


# =============================================================================
# Core Commands
# =============================================================================


@app.command()
def remember(
    content: Annotated[str, typer.Argument(help="Content to remember")],
    tags: Annotated[
        Optional[list[str]], typer.Option("--tag", "-t", help="Tags for the memory")
    ] = None,
    memory_type: Annotated[
        Optional[str],
        typer.Option(
            "--type",
            "-T",
            help="Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detected if not specified)",
        ),
    ] = None,
    priority: Annotated[
        Optional[int],
        typer.Option(
            "--priority", "-p", help="Priority 0-10 (0=lowest, 5=normal, 10=critical)"
        ),
    ] = None,
    expires: Annotated[
        Optional[int],
        typer.Option("--expires", "-e", help="Days until this memory expires"),
    ] = None,
    project: Annotated[
        Optional[str],
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
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Store a new memory.

    Memory types (auto-detected if not specified):
        fact        - Objective information
        decision    - Choices made
        preference  - User preferences
        todo        - Action items (expires in 30 days by default)
        insight     - Learned patterns
        context     - Situational info (expires in 7 days by default)
        instruction - User guidelines
        error       - Error patterns
        workflow    - Process patterns
        reference   - External references

    Examples:
        nmem remember "Fixed auth bug by adding null check"
        nmem remember "We decided to use PostgreSQL" --type decision
        nmem remember "Need to refactor auth module" --type todo --priority 7
        nmem remember "Meeting context" --type context --expires 7
        nmem remember "API_KEY=xxx" --redact  # Will redact sensitive content
        nmem remember "Sprint task done" --project "Q1 Sprint"  # Associate with project
        nmem remember "Shared memory" --shared  # Store in shared brain
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

        brain_id = storage._current_brain_id if hasattr(storage, "_current_brain_id") else config.current_brain
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        # Look up project if specified
        project_id = None
        if project:
            proj = await storage.get_project_by_name(project)
            if not proj:
                return {"error": f"Project '{project}' not found. Create it with: nmem project create \"{project}\""}
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
            warnings.append(f"[!] Stored with {len(sensitive_matches)} sensitive item(s) - consider using --redact")
        if warnings:
            response["warnings"] = warnings

        return response

    result = asyncio.run(_remember())
    output_result(result, json_output)


@app.command()
def todo(
    task: Annotated[str, typer.Argument(help="Task to remember")],
    priority: Annotated[
        int,
        typer.Option("--priority", "-p", help="Priority 0-10 (default: 5=normal, 7=high, 10=critical)"),
    ] = 5,
    project: Annotated[
        Optional[str],
        typer.Option("--project", "-P", help="Associate with a project"),
    ] = None,
    expires: Annotated[
        Optional[int],
        typer.Option("--expires", "-e", help="Days until expiry (default: 30)"),
    ] = None,
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Tags for the task"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Quick shortcut to add a TODO memory.

    Equivalent to: nmem remember --type todo "task"

    Examples:
        nmem todo "Fix the login bug"
        nmem todo "Review PR #123" --priority 7
        nmem todo "Deploy to prod" -p 10 --project "Q1 Sprint"
        nmem todo "Update docs" --expires 7
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
                return {"error": f"Project '{project}' not found. Create it with: nmem project create \"{project}\""}
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


@app.command()
def recall(
    query: Annotated[str, typer.Argument(help="Query to search memories")],
    depth: Annotated[
        Optional[int],
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
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Query memories with intelligent routing.

    Query types (auto-detected):
        semantic    - Conceptual queries ("What do I know about auth?")
        temporal    - Time-based queries ("What did I do yesterday?")
        causal      - Why/how queries ("Why did the build fail?")
        direct      - Exact recall ("What's Alice's email?")
        pattern     - Habit queries ("What do I usually do on Mondays?")
        comparative - Comparison ("Compare React and Vue")

    Examples:
        nmem recall "What did I do with auth?"
        nmem recall "meetings with Alice" --depth 2
        nmem recall "Why did the build fail?" --show-routing
        nmem recall "project status" --min-confidence 0.5
        nmem recall "shared knowledge" --shared  # Query from shared brain
    """

    async def _recall() -> dict:
        config = get_config()
        storage = await get_storage(config, force_shared=shared)

        brain_id = storage._current_brain_id if hasattr(storage, "_current_brain_id") else config.current_brain
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


@app.command()
def context(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Number of recent memories")
    ] = 10,
    fresh_only: Annotated[
        bool, typer.Option("--fresh-only", help="Only include memories < 30 days old")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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

            content = fiber.summary
            if not content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content

            if content:
                context_parts.append(f"{indicator} [{age_str}] {content}")
                fiber_data.append({
                    "id": fiber.id,
                    "summary": content,
                    "created_at": fiber.created_at.isoformat(),
                    "age": age_str,
                    "freshness": freshness.level.value,
                })

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


@app.command("list")
def list_memories(
    memory_type: Annotated[
        Optional[str],
        typer.Option("--type", "-T", help="Filter by memory type (fact, decision, todo, etc.)"),
    ] = None,
    min_priority: Annotated[
        Optional[int],
        typer.Option("--min-priority", "-p", help="Minimum priority (0-10)"),
    ] = None,
    project_name: Annotated[
        Optional[str],
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
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum number of results")
    ] = 20,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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

                memories_data.append({
                    "fiber_id": tm.fiber_id,
                    "type": tm.memory_type.value,
                    "priority": tm.priority.name.lower(),
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "expired_days_ago": abs(tm.days_until_expiry) if tm.days_until_expiry else 0,
                    "created_at": tm.created_at.isoformat(),
                })

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
                memories_data.append({
                    "fiber_id": fiber.id,
                    "type": "unknown",
                    "priority": "normal",
                    "content": content[:100] + "..." if content and len(content) > 100 else content or "",
                    "age": format_age(freshness.age_days),
                    "created_at": fiber.created_at.isoformat(),
                })

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

            memories_data.append({
                "fiber_id": tm.fiber_id,
                "type": tm.memory_type.value,
                "priority": tm.priority.name.lower(),
                "content": content[:100] + "..." if len(content) > 100 else content,
                "age": format_age(freshness.age_days),
                "expires": expiry_info,
                "verified": tm.provenance.verified,
                "created_at": tm.created_at.isoformat(),
            })

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


@app.command()
def cleanup(
    expired_only: Annotated[
        bool,
        typer.Option("--expired", "-e", help="Only clean up expired memories"),
    ] = True,
    memory_type: Annotated[
        Optional[str],
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
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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

            to_delete.append({
                "fiber_id": tm.fiber_id,
                "type": tm.memory_type.value,
                "content": content,
                "expired_at": tm.expires_at.isoformat() if tm.expires_at else None,
            })

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


@app.command()
def stats(
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Show brain statistics including freshness and memory type analysis.

    Examples:
        nmem stats
        nmem stats --json
    """

    async def _stats() -> dict:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        stats_data = await storage.get_stats(brain.id)

        # Get fibers for freshness analysis
        fibers = await storage.get_fibers(limit=1000)
        created_dates = [f.created_at for f in fibers]
        freshness_report = analyze_freshness(created_dates)

        # Get typed memory statistics
        typed_memories = await storage.find_typed_memories(include_expired=True, limit=10000)
        expired_memories = await storage.get_expired_memories()

        # Count by type
        type_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {"critical": 0, "high": 0, "normal": 0, "low": 0, "lowest": 0}

        for tm in typed_memories:
            type_name = tm.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            priority_counts[tm.priority.name.lower()] += 1

        return {
            "brain": brain.name,
            "brain_id": brain.id,
            "neuron_count": stats_data["neuron_count"],
            "synapse_count": stats_data["synapse_count"],
            "fiber_count": stats_data["fiber_count"],
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

    result = asyncio.run(_stats())

    if json_output:
        output_result(result, True)
    else:
        typer.echo(f"Brain: {result['brain']}")
        typer.echo(f"Neurons: {result['neuron_count']}")
        typer.echo(f"Synapses: {result['synapse_count']}")
        typer.echo(f"Fibers (memories): {result['fiber_count']}")

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
                    fg=typer.colors.YELLOW
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


@app.command()
def check(
    content: Annotated[str, typer.Argument(help="Content to check for sensitive data")],
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Check content for sensitive information without storing.

    Examples:
        nmem check "My API_KEY=sk-xxx123"
        nmem check "password: secret123" --json
    """
    matches = check_sensitive_content(content)

    if json_output:
        output_result({
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
        }, True)
    else:
        if matches:
            typer.echo(format_sensitive_warning(matches))
        else:
            typer.secho("[OK] No sensitive content detected", fg=typer.colors.GREEN)


# =============================================================================
# Brain Management Commands
# =============================================================================


@brain_app.command("list")
def brain_list(
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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
        typer.secho(f"Brain '{name}' not found. Create it with: nmem brain create {name}", fg=typer.colors.RED)
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
        brain_path = config.get_brain_path(name)
        await PersistentStorage.load(brain_path)

        if use:
            config.current_brain = name
            config.save()

        typer.secho(f"Created brain: {name}", fg=typer.colors.GREEN)
        if use:
            typer.echo(f"Now using: {name}")

    asyncio.run(_create())


@brain_app.command("export")
def brain_export(
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output file path")
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Brain name (default: current)")
    ] = None,
    exclude_sensitive: Annotated[
        bool, typer.Option("--exclude-sensitive", "-s", help="Exclude memories with sensitive content")
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
        brain_path = config.get_brain_path(brain_name)

        if not brain_path.exists():
            typer.secho(f"Brain '{brain_name}' not found.", fg=typer.colors.RED)
            raise typer.Exit(1)

        storage = await PersistentStorage.load(brain_path)
        snapshot = await storage.export_brain(storage._current_brain_id)

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
                s for s in snapshot.synapses
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
            with open(output, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            typer.secho(f"Exported to: {output}", fg=typer.colors.GREEN)
            if excluded_count > 0:
                typer.secho(f"Excluded {excluded_count} neurons with sensitive content", fg=typer.colors.YELLOW)
        else:
            typer.echo(json.dumps(export_data, indent=2, default=str))

    asyncio.run(_export())


@brain_app.command("import")
def brain_import(
    file: Annotated[str, typer.Argument(help="JSON file to import")],
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Name for imported brain")
    ] = None,
    use: Annotated[
        bool, typer.Option("--use", "-u", help="Switch to imported brain")
    ] = True,
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
        with open(file, encoding="utf-8") as f:
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
                typer.secho(f"[!] Found {sensitive_count} neurons with potentially sensitive content", fg=typer.colors.YELLOW)
                if not typer.confirm("Continue importing?"):
                    raise typer.Exit(0)

        brain_name = name or data.get("brain_name", "imported")
        config = get_config()

        if brain_name in config.list_brains():
            typer.secho(f"Brain '{brain_name}' already exists. Use --name to specify different name.", fg=typer.colors.RED)
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
        brain_path = config.get_brain_path(brain_name)
        storage = await PersistentStorage.load(brain_path)
        await storage.import_brain(snapshot, storage._current_brain_id)
        await storage.save()

        if use:
            config.current_brain = brain_name
            config.save()

        typer.secho(f"Imported brain: {brain_name}", fg=typer.colors.GREEN)
        typer.echo(f"  Neurons: {len(data['neurons'])}")
        typer.echo(f"  Synapses: {len(data['synapses'])}")
        typer.echo(f"  Fibers: {len(data['fibers'])}")

    asyncio.run(_import())


@brain_app.command("delete")
def brain_delete(
    name: Annotated[str, typer.Argument(help="Brain name to delete")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
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
        typer.secho("Cannot delete current brain. Switch to another brain first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete brain '{name}'? This cannot be undone.")
        if not confirm:
            typer.echo("Cancelled.")
            return

    brain_path = config.get_brain_path(name)
    brain_path.unlink()
    typer.secho(f"Deleted brain: {name}", fg=typer.colors.GREEN)


@brain_app.command("health")
def brain_health(
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Brain name (default: current)")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Check brain health (freshness, sensitive content).

    Examples:
        nmem brain health
        nmem brain health --name work --json
    """

    async def _health() -> dict:
        config = get_config()
        brain_name = name or config.current_brain
        brain_path = config.get_brain_path(brain_name)

        if not brain_path.exists():
            return {"error": f"Brain '{brain_name}' not found."}

        storage = await PersistentStorage.load(brain_path)
        brain = await storage.get_brain(storage._current_brain_id)

        if not brain:
            return {"error": "No brain configured"}

        # Get all neurons to check
        neurons = list(storage._neurons[storage._current_brain_id].values())
        fibers = await storage.get_fibers(limit=10000)

        # Check for sensitive content
        sensitive_neurons = []
        for neuron in neurons:
            matches = check_sensitive_content(neuron.content, min_severity=2)
            if matches:
                sensitive_neurons.append({
                    "id": neuron.id,
                    "type": neuron.type.value,
                    "sensitive_types": [m.type.value for m in matches],
                })

        # Analyze freshness
        created_dates = [f.created_at for f in fibers]
        freshness_report = analyze_freshness(created_dates)

        # Calculate health score (0-100)
        health_score = 100
        issues = []

        # Penalize for sensitive content
        if sensitive_neurons:
            penalty = min(30, len(sensitive_neurons) * 5)
            health_score -= penalty
            issues.append(f"{len(sensitive_neurons)} neurons with sensitive content")

        # Penalize for stale memories
        stale_ratio = (freshness_report.stale + freshness_report.ancient) / max(1, freshness_report.total)
        if stale_ratio > 0.5:
            health_score -= 20
            issues.append(f"{stale_ratio*100:.0f}% of memories are stale/ancient")
        elif stale_ratio > 0.2:
            health_score -= 10
            issues.append(f"{stale_ratio*100:.0f}% of memories are stale/ancient")

        health_score = max(0, health_score)

        return {
            "brain": brain_name,
            "health_score": health_score,
            "issues": issues,
            "sensitive_content": {
                "count": len(sensitive_neurons),
                "neurons": sensitive_neurons[:5],  # Show first 5
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

    result = asyncio.run(_health())

    if json_output:
        output_result(result, True)
    else:
        if "error" in result:
            typer.secho(result["error"], fg=typer.colors.RED)
            return

        score = result["health_score"]
        if score >= 80:
            color = typer.colors.GREEN
            indicator = "[OK]"
        elif score >= 50:
            color = typer.colors.YELLOW
            indicator = "[~]"
        else:
            color = typer.colors.RED
            indicator = "[!!]"

        typer.echo(f"\nBrain: {result['brain']}")
        typer.secho(f"Health Score: {indicator} {score}/100", fg=color)

        if result["issues"]:
            typer.echo("\nIssues:")
            for issue in result["issues"]:
                typer.secho(f"  [!] {issue}", fg=typer.colors.YELLOW)

        if result["sensitive_content"]["count"] > 0:
            typer.echo(f"\nSensitive content: {result['sensitive_content']['count']} neurons")
            typer.secho("  Run 'nmem brain export --exclude-sensitive' for safe export", fg=typer.colors.BRIGHT_BLACK)

        f = result["freshness"]
        if f["total"] > 0:
            typer.echo(f"\nMemory freshness ({f['total']} total):")
            typer.echo(f"  [+] Fresh/Recent: {f['fresh'] + f['recent']}")
            typer.echo(f"  [~] Aging: {f['aging']}")
            typer.echo(f"  [!!] Stale/Ancient: {f['stale'] + f['ancient']}")


# =============================================================================
# Project Commands
# =============================================================================


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name")],
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Project description"),
    ] = None,
    duration: Annotated[
        Optional[int],
        typer.Option("--duration", "-D", help="Duration in days (creates end date)"),
    ] = None,
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Project tags"),
    ] = None,
    priority: Annotated[
        float,
        typer.Option("--priority", "-p", help="Project priority (default: 1.0)"),
    ] = 1.0,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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

        project = Project.create(
            name=name,
            description=description or "",
            duration_days=duration,
            tags=set(tags) if tags else None,
            priority=priority,
        )

        await storage.add_project(project)
        await storage.batch_save()

        response = {
            "message": f"Created project: {name}",
            "project_id": project.id,
            "name": project.name,
            "is_ongoing": project.is_ongoing,
        }

        if project.end_date:
            response["ends_in_days"] = project.days_remaining
            response["end_date"] = project.end_date.isoformat()

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
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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
                status = typer.style("[ACTIVE]", fg=typer.colors.GREEN)
            else:
                status = typer.style("[ENDED]", fg=typer.colors.BRIGHT_BLACK)

            typer.echo(f"{status} {proj['name']}")

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
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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

            memories_data.append({
                "type": tm.memory_type.value,
                "priority": tm.priority.name.lower(),
                "content": content[:80] + "..." if len(content) > 80 else content,
                "created_at": tm.created_at.isoformat(),
            })

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
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation")
    ] = False,
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
                fg=typer.colors.BRIGHT_BLACK
            )


@project_app.command("extend")
def project_extend(
    name: Annotated[str, typer.Argument(help="Project name")],
    days: Annotated[int, typer.Argument(help="Days to extend by")],
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
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
                    fg=typer.colors.BRIGHT_BLACK
                )


# =============================================================================
# Shared Mode Commands
# =============================================================================


@shared_app.command("enable")
def shared_enable(
    server_url: Annotated[
        str, typer.Argument(help="NeuralMemory server URL (e.g., http://localhost:8000)")
    ],
    api_key: Annotated[
        Optional[str], typer.Option("--api-key", "-k", help="API key for authentication")
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", "-t", help="Request timeout in seconds")
    ] = 30.0,
) -> None:
    """Enable shared mode to connect to a remote NeuralMemory server.

    When shared mode is enabled, all memory operations (remember, recall, etc.)
    will use the remote server instead of local storage.

    Examples:
        nmem shared enable http://localhost:8000
        nmem shared enable https://memory.example.com --api-key mykey
        nmem shared enable http://server:8000 --timeout 60
    """
    config = get_config()
    config.shared.enabled = True
    config.shared.server_url = server_url.rstrip("/")
    config.shared.api_key = api_key
    config.shared.timeout = timeout
    config.save()

    typer.secho("Shared mode enabled!", fg=typer.colors.GREEN)
    typer.echo(f"  Server: {config.shared.server_url}")
    if api_key:
        typer.echo(f"  API Key: {'*' * 8}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    typer.echo(f"  Timeout: {timeout}s")
    typer.echo("")
    typer.secho("All memory commands will now use the remote server.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("Use 'nmem shared disable' to switch back to local storage.", fg=typer.colors.BRIGHT_BLACK)


@shared_app.command("disable")
def shared_disable() -> None:
    """Disable shared mode and use local storage.

    Examples:
        nmem shared disable
    """
    config = get_config()
    config.shared.enabled = False
    config.save()

    typer.secho("Shared mode disabled.", fg=typer.colors.GREEN)
    typer.echo("Memory commands will now use local storage.")


@shared_app.command("status")
def shared_status(
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Show shared mode status and configuration.

    Examples:
        nmem shared status
        nmem shared status --json
    """
    config = get_config()

    status = {
        "enabled": config.shared.enabled,
        "server_url": config.shared.server_url,
        "api_key_set": config.shared.api_key is not None,
        "timeout": config.shared.timeout,
    }

    if json_output:
        typer.echo(json.dumps(status, indent=2))
    else:
        if config.shared.enabled:
            typer.secho("[ENABLED] Shared mode is active", fg=typer.colors.GREEN)
        else:
            typer.secho("[DISABLED] Using local storage", fg=typer.colors.YELLOW)

        typer.echo(f"\nServer URL: {config.shared.server_url}")
        typer.echo(f"API Key: {'configured' if config.shared.api_key else 'not set'}")
        typer.echo(f"Timeout: {config.shared.timeout}s")


@shared_app.command("test")
def shared_test() -> None:
    """Test connection to the shared server.

    Examples:
        nmem shared test
    """
    config = get_config()

    if not config.shared.server_url:
        typer.secho("No server URL configured. Use 'nmem shared enable <url>' first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    async def _test() -> dict:
        import aiohttp

        url = f"{config.shared.server_url}/health"
        headers = {}
        if config.shared.api_key:
            headers["Authorization"] = f"Bearer {config.shared.api_key}"

        try:
            timeout = aiohttp.ClientTimeout(total=config.shared.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "status": data.get("status", "unknown"),
                            "version": data.get("version", "unknown"),
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Server returned status {response.status}",
                        }
        except aiohttp.ClientError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    typer.echo(f"Testing connection to {config.shared.server_url}...")
    result = asyncio.run(_test())

    if result["success"]:
        typer.secho("[OK] Connection successful!", fg=typer.colors.GREEN)
        typer.echo(f"  Server status: {result['status']}")
        typer.echo(f"  Server version: {result['version']}")
    else:
        typer.secho("[FAILED] Connection failed!", fg=typer.colors.RED)
        typer.echo(f"  Error: {result['error']}")
        raise typer.Exit(1)


@shared_app.command("sync")
def shared_sync(
    direction: Annotated[
        str, typer.Option("--direction", "-d", help="Sync direction: push, pull, or both")
    ] = "both",
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output as JSON")
    ] = False,
) -> None:
    """Manually sync local brain with remote server.

    Directions:
        push  - Upload local brain to server
        pull  - Download brain from server to local
        both  - Full bidirectional sync (default)

    Examples:
        nmem shared sync
        nmem shared sync --direction push
        nmem shared sync --direction pull
    """
    config = get_config()

    if not config.shared.server_url:
        typer.secho("No server URL configured. Use 'nmem shared enable <url>' first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    async def _sync() -> dict:
        from neural_memory.storage.shared_store import SharedStorage

        # Load local storage
        local_storage = await PersistentStorage.load(config.get_brain_path())

        # Connect to remote
        remote = SharedStorage(
            server_url=config.shared.server_url,
            brain_id=config.current_brain,
            api_key=config.shared.api_key,
            timeout=config.shared.timeout,
        )
        await remote.connect()

        result = {"direction": direction, "success": True}

        try:
            if direction in ("push", "both"):
                # Export local and push to remote
                snapshot = await local_storage.export_brain(local_storage._current_brain_id)
                await remote.import_brain(snapshot, config.current_brain)
                result["pushed"] = True
                result["neurons_pushed"] = len(snapshot.neurons)
                result["synapses_pushed"] = len(snapshot.synapses)
                result["fibers_pushed"] = len(snapshot.fibers)

            if direction in ("pull", "both"):
                # Pull from remote and import locally
                try:
                    snapshot = await remote.export_brain(config.current_brain)
                    await local_storage.import_brain(snapshot, local_storage._current_brain_id)
                    await local_storage.save()
                    result["pulled"] = True
                    result["neurons_pulled"] = len(snapshot.neurons)
                    result["synapses_pulled"] = len(snapshot.synapses)
                    result["fibers_pulled"] = len(snapshot.fibers)
                except Exception as e:
                    if direction == "pull":
                        raise
                    # For "both", pulling may fail if brain doesn't exist on server
                    result["pulled"] = False
                    result["pull_error"] = str(e)

        finally:
            await remote.disconnect()

        return result

    typer.echo(f"Syncing brain '{config.current_brain}' with {config.shared.server_url}...")
    result = asyncio.run(_sync())

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        if result.get("pushed"):
            typer.secho(f"[PUSHED] {result.get('neurons_pushed', 0)} neurons, "
                       f"{result.get('synapses_pushed', 0)} synapses, "
                       f"{result.get('fibers_pushed', 0)} fibers", fg=typer.colors.GREEN)

        if result.get("pulled"):
            typer.secho(f"[PULLED] {result.get('neurons_pulled', 0)} neurons, "
                       f"{result.get('synapses_pulled', 0)} synapses, "
                       f"{result.get('fibers_pulled', 0)} fibers", fg=typer.colors.GREEN)
        elif result.get("pull_error"):
            typer.secho(f"[PULL FAILED] {result['pull_error']}", fg=typer.colors.YELLOW)

        typer.secho("\nSync complete!", fg=typer.colors.GREEN)


# =============================================================================
# Utility Commands
# =============================================================================


@app.command()
def mcp() -> None:
    """Run the MCP (Model Context Protocol) server.

    This starts an MCP server over stdio that exposes NeuralMemory tools
    to Claude Code, Claude Desktop, and other MCP-compatible clients.

    Available tools:
        nmem_remember  - Store a memory
        nmem_recall    - Query memories
        nmem_context   - Get recent context
        nmem_todo      - Add a TODO memory
        nmem_stats     - Get brain statistics

    Examples:
        nmem mcp                    # Run MCP server
        python -m neural_memory.mcp # Alternative way

    Configuration for Claude Code (~/.claude/mcp_servers.json):
        {
            "neural-memory": {
                "command": "nmem",
                "args": ["mcp"]
            }
        }
    """
    from neural_memory.mcp.server import main as mcp_main

    mcp_main()


@app.command()
def version() -> None:
    """Show version information."""
    from neural_memory import __version__

    typer.echo(f"neural-memory v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
