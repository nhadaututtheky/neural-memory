"""Utility tool commands: mcp, dashboard, ui, graph, init, serve, decay, hooks."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage


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


def dashboard() -> None:
    """Show a rich dashboard with brain stats and recent activity.

    Displays:
        - Brain statistics (neurons, synapses, fibers)
        - Memory types distribution
        - Freshness analysis
        - Recent memories

    Examples:
        nmem dashboard
    """
    from neural_memory.cli.tui import render_dashboard

    async def _dashboard() -> None:
        config = get_config()
        storage = await get_storage(config)
        await render_dashboard(storage)

    asyncio.run(_dashboard())


def ui(
    memory_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by memory type"),
    ] = None,
    search: Annotated[
        str | None,
        typer.Option("--search", "-s", help="Search in memory content"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of memories to show"),
    ] = 20,
) -> None:
    """Interactive memory browser with rich formatting.

    Browse memories with color-coded types, priorities, and freshness.

    Examples:
        nmem ui                        # Browse all memories
        nmem ui --type decision        # Filter by type
        nmem ui --search "database"    # Search content
        nmem ui --limit 50             # Show more
    """
    from neural_memory.cli.tui import render_memory_browser

    async def _ui() -> None:
        config = get_config()
        storage = await get_storage(config)
        await render_memory_browser(
            storage,
            memory_type=memory_type,
            limit=limit,
            search=search,
        )

    asyncio.run(_ui())


def graph(
    query: Annotated[
        str | None,
        typer.Argument(help="Query to find related memories (optional)"),
    ] = None,
    depth: Annotated[
        int,
        typer.Option("--depth", "-d", help="Traversal depth (1-3)"),
    ] = 2,
) -> None:
    """Visualize neural connections as a tree graph.

    Shows memories and their relationships (caused_by, leads_to, etc.)

    Examples:
        nmem graph                     # Show recent memories
        nmem graph "database"          # Graph around query
        nmem graph "auth" --depth 3    # Deeper traversal
    """
    from neural_memory.cli.tui import render_graph

    async def _graph() -> None:
        config = get_config()
        storage = await get_storage(config)
        await render_graph(storage, query=query, depth=depth)

    asyncio.run(_graph())


def init(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing config"),
    ] = False,
) -> None:
    """Initialize unified config for cross-tool memory sharing.

    This sets up ~/.neuralmemory/ which enables memory sharing between:
    - CLI (nmem commands)
    - MCP (Claude Code, Cursor, AntiGravity)
    - Any other tool using NeuralMemory

    After running this, all tools will share the same brain database.

    Examples:
        nmem init              # Initialize unified config
        nmem init --force      # Overwrite existing config
    """
    from neural_memory.unified_config import UnifiedConfig, get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()
    config_path = data_dir / "config.toml"

    if config_path.exists() and not force:
        typer.secho(f"Config already exists at {config_path}", fg=typer.colors.YELLOW)
        typer.echo("Use --force to overwrite")
        return

    # Create unified config
    config = UnifiedConfig(data_dir=data_dir)
    config.save()

    # Ensure brains directory exists
    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(f"Initialized NeuralMemory at {data_dir}", fg=typer.colors.GREEN)
    typer.echo()
    typer.echo("Directory structure:")
    typer.echo(f"  {data_dir}/")
    typer.echo("  +-- config.toml        # Shared configuration")
    typer.echo("  +-- brains/")
    typer.echo("      +-- default.db     # SQLite brain database")
    typer.echo()
    typer.echo("This enables memory sharing between:")
    typer.echo("  - CLI: nmem commands")
    typer.echo("  - MCP: Claude Code, Cursor, AntiGravity")
    typer.echo()
    typer.echo("To use a specific brain, set NEURALMEMORY_BRAIN environment variable:")
    typer.echo("  export NEURALMEMORY_BRAIN=myproject")


def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host to bind to")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind to")
    ] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", "-r", help="Enable auto-reload for development")
    ] = False,
) -> None:
    """Run the NeuralMemory API server.

    Examples:
        nmem serve                    # Run on localhost:8000
        nmem serve -p 9000            # Run on port 9000
        nmem serve --host 0.0.0.0     # Expose to network
        nmem serve --reload           # Development mode
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo("Error: uvicorn not installed. Run: pip install neural-memory[server]", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting NeuralMemory API server on http://{host}:{port}")
    typer.echo(f"  UI:   http://{host}:{port}/ui")
    typer.echo(f"  Docs: http://{host}:{port}/docs")

    uvicorn.run(
        "neural_memory.server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


def decay(
    brain: Annotated[
        str | None, typer.Option("--brain", "-b", help="Brain to apply decay to")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Preview changes without applying")
    ] = False,
    prune_threshold: Annotated[
        float, typer.Option("--prune", "-p", help="Prune below this activation level")
    ] = 0.01,
) -> None:
    """Apply memory decay to simulate forgetting.

    Memories that haven't been accessed recently will have their
    activation levels reduced following the Ebbinghaus forgetting curve.

    Examples:
        nmem decay                    # Apply decay to current brain
        nmem decay -b work            # Apply to specific brain
        nmem decay --dry-run          # Preview without changes
        nmem decay --prune 0.05       # More aggressive pruning
    """
    from neural_memory.engine.lifecycle import DecayManager
    from neural_memory.unified_config import get_config, get_shared_storage

    async def _decay() -> None:
        config = get_config()
        brain_name = brain or config.current_brain

        typer.echo(f"Applying decay to brain '{brain_name}'...")
        if dry_run:
            typer.echo("(dry run - no changes will be saved)")

        storage = await get_shared_storage(brain_name)

        manager = DecayManager(
            decay_rate=config.brain.decay_rate,
            prune_threshold=prune_threshold,
        )

        report = await manager.apply_decay(storage, dry_run=dry_run)

        typer.echo("")
        typer.echo(report.summary())

        if report.neurons_pruned > 0 or report.synapses_pruned > 0:
            typer.echo("")
            typer.echo(
                f"Pruned {report.neurons_pruned} neurons and "
                f"{report.synapses_pruned} synapses below threshold {prune_threshold}"
            )

    asyncio.run(_decay())


def hooks(
    action: Annotated[
        str,
        typer.Argument(help="Action: install, uninstall, show"),
    ] = "install",
    path: Annotated[
        str | None,
        typer.Option("--path", "-p", help="Path to git repo (default: current dir)"),
    ] = None,
) -> None:
    """Install or manage git hooks for automatic memory capture.

    Installs a post-commit hook that suggests saving the commit
    as a memory after each git commit.

    Examples:
        nmem hooks install          # Install in current repo
        nmem hooks install -p .     # Explicit path
        nmem hooks uninstall        # Remove hooks
        nmem hooks show             # Show installed hooks
    """
    import os
    import stat
    from pathlib import Path

    repo_path = Path(path) if path else Path.cwd()
    git_dir = repo_path / ".git"

    if not git_dir.is_dir():
        typer.secho(
            f"Not a git repository: {repo_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    post_commit = hooks_dir / "post-commit"

    hook_marker = "# [neural-memory] auto-generated hook"

    hook_script = f"""#!/bin/sh
{hook_marker}
# Suggest saving git commit as a memory.
# Installed by: nmem hooks install
# Remove with:  nmem hooks uninstall

MSG=$(git log -1 --pretty=%B 2>/dev/null)
if [ -n "$MSG" ]; then
    echo ""
    echo "[NeuralMemory] Commit: $MSG"
    echo "  Save as memory? Run:"
    echo "    nmem remember \\"$MSG\\" --tag git --tag auto"
    echo ""
fi
"""

    if action == "install":
        # Check for existing non-nmem hook
        if post_commit.exists():
            existing = post_commit.read_text(encoding="utf-8")
            if hook_marker in existing:
                typer.secho(
                    "Hook already installed.",
                    fg=typer.colors.YELLOW,
                )
                return
            # Existing hook from another tool — don't overwrite
            typer.secho(
                "A post-commit hook already exists (not from neural-memory).",
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo("Manually merge or remove it, then try again.")
            raise typer.Exit(1)

        post_commit.write_text(hook_script, encoding="utf-8")

        # Make executable (Unix)
        if os.name != "nt":
            st = post_commit.stat()
            post_commit.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        typer.secho("Installed post-commit hook.", fg=typer.colors.GREEN)
        typer.echo(f"  Location: {post_commit}")
        typer.echo("  After each commit you'll see a reminder to save the commit message.")

    elif action == "uninstall":
        if not post_commit.exists():
            typer.secho("No post-commit hook found.", fg=typer.colors.YELLOW)
            return

        existing = post_commit.read_text(encoding="utf-8")
        if hook_marker not in existing:
            typer.secho(
                "Post-commit hook exists but wasn't installed by neural-memory. Skipping.",
                fg=typer.colors.YELLOW,
            )
            return

        post_commit.unlink()
        typer.secho("Removed post-commit hook.", fg=typer.colors.GREEN)

    elif action == "show":
        if post_commit.exists():
            existing = post_commit.read_text(encoding="utf-8")
            is_nmem = hook_marker in existing
            typer.echo(f"Post-commit hook: {'installed (neural-memory)' if is_nmem else 'exists (other)'}")
            typer.echo(f"  Path: {post_commit}")
        else:
            typer.echo("Post-commit hook: not installed")

    else:
        typer.secho(
            f"Unknown action: {action}. Use: install, uninstall, show",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Register tool commands on the app."""
    app.command()(mcp)
    app.command()(dashboard)
    app.command()(ui)
    app.command()(graph)
    app.command()(init)
    app.command()(serve)
    app.command()(decay)
    app.command()(hooks)
