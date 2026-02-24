"""Utility tool commands: mcp, dashboard, ui, graph, init, serve, decay, hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import get_config, get_storage, run_async


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

    run_async(_dashboard())


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

    run_async(_ui())


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

    run_async(_graph())


def init(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing config"),
    ] = False,
    skip_mcp: Annotated[
        bool,
        typer.Option("--skip-mcp", help="Skip MCP auto-configuration"),
    ] = False,
    skip_skills: Annotated[
        bool,
        typer.Option("--skip-skills", help="Skip skills installation"),
    ] = False,
) -> None:
    """Set up NeuralMemory in one command.

    Creates config, default brain, auto-configures MCP for
    Claude Code and Cursor, and installs agent skills.

    Examples:
        nmem init                # Full setup
        nmem init --force        # Overwrite existing config
        nmem init --skip-mcp     # Skip MCP auto-config
        nmem init --skip-skills  # Skip skills installation
    """
    from neural_memory.cli.setup import (
        print_summary,
        setup_brain,
        setup_config,
        setup_mcp_claude,
        setup_mcp_cursor,
        setup_skills,
    )
    from neural_memory.unified_config import get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()
    results: dict[str, str] = {}

    # 1. Config
    created = setup_config(data_dir, force=force)
    results["Config"] = f"{data_dir / 'config.toml'} (created)" if created else "already exists"

    # 2. Brain
    brain_name = setup_brain(data_dir)
    results["Brain"] = f"{brain_name} (ready)"

    # 3. MCP auto-config
    if skip_mcp:
        results["Claude Code"] = "skipped (--skip-mcp)"
        results["Cursor"] = "skipped (--skip-mcp)"
    else:
        claude_status = setup_mcp_claude()
        status_labels = {
            "added": "~/.claude/mcp_servers.json (added)",
            "exists": "~/.claude/mcp_servers.json (already configured)",
            "not_found": "not detected (~/.claude/ not found)",
            "failed": "failed to write config",
        }
        results["Claude Code"] = status_labels.get(claude_status, claude_status)

        cursor_status = setup_mcp_cursor()
        cursor_labels = {
            "added": "~/.cursor/mcp.json (added)",
            "exists": "~/.cursor/mcp.json (already configured)",
            "not_found": "not detected",
            "failed": "failed to write config",
        }
        results["Cursor"] = cursor_labels.get(cursor_status, cursor_status)

    # 4. Skills
    if skip_skills:
        results["Skills"] = "skipped (--skip-skills)"
    else:
        skill_results = setup_skills(force=force)
        if "Skills" in skill_results:
            # Error case (e.g. ~/.claude/ not found)
            results["Skills"] = skill_results["Skills"]
        else:
            counts: dict[str, int] = {}
            for status in skill_results.values():
                counts[status] = counts.get(status, 0) + 1
            parts = [f"{count} {status}" for status, count in counts.items()]
            results["Skills"] = ", ".join(parts) if parts else "none found"

    print_summary(results)

    typer.echo("  Restart your AI tool to activate memory.")
    typer.echo()


def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
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

    async def _decay() -> None:
        config = get_config()
        brain_name = brain or config.current_brain

        typer.echo(f"Applying decay to brain '{brain_name}'...")
        if dry_run:
            typer.echo("(dry run - no changes will be saved)")

        storage = await get_storage(config, brain_name=brain_name)

        from neural_memory.unified_config import get_config as get_unified_config

        unified = get_unified_config()
        manager = DecayManager(
            decay_rate=unified.brain.decay_rate,
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

    run_async(_decay())


def consolidate(
    strategy_positional: Annotated[
        str | None,
        typer.Argument(
            hidden=True,
            help="Deprecated positional syntax — use --strategy instead.",
        ),
    ] = None,
    brain: Annotated[str | None, typer.Option("--brain", "-b", help="Brain to consolidate")] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help=(
                "Consolidation strategy. Valid values: "
                "prune, merge, summarize, mature, infer, enrich, "
                "dream, learn_habits, dedup, semantic_link, compress, all. "
                "Default: all (runs every strategy in dependency order). "
                "'mature' advances episodic memories to semantic stage."
            ),
        ),
    ] = "all",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Preview changes without applying")
    ] = False,
    prune_threshold: Annotated[
        float, typer.Option("--prune-threshold", help="Weight threshold for pruning synapses")
    ] = 0.05,
    merge_overlap: Annotated[
        float, typer.Option("--merge-overlap", help="Jaccard overlap threshold for merging fibers")
    ] = 0.5,
    min_inactive_days: Annotated[
        float, typer.Option("--min-inactive-days", help="Minimum inactive days before pruning")
    ] = 7.0,
) -> None:
    """Consolidate brain memories by pruning, merging, or summarizing.

    Runs one or more consolidation passes on the current (or specified) brain.
    Use --strategy to pick a specific pass, or omit it to run everything.

    Strategies:
        prune        - Remove weak synapses and orphan neurons
        merge        - Combine overlapping fibers
        summarize    - Create concept neurons for topic clusters
        mature       - Advance episodic memories to semantic stage
        infer        - Add inferred synapses from co-activation patterns
        enrich       - Enrich neurons with extracted metadata
        dream        - Generate synthetic bridging memories
        learn_habits - Extract recurring workflow patterns
        dedup        - Merge near-duplicate memories
        semantic_link- Add cross-domain semantic connections
        compress     - Compress old low-activation fibers
        all          - Run all strategies in dependency order (default)

    Examples:
        nmem consolidate                         # Run all strategies
        nmem consolidate --strategy prune        # Only prune
        nmem consolidate -s mature               # Advance episodic memories
        nmem consolidate --dry-run               # Preview without changes
        nmem consolidate -s merge --merge-overlap 0.3
    """
    # Friendly error for old-style positional syntax (e.g. `nmem consolidate prune`)
    if strategy_positional is not None:
        typer.secho(
            f"Error: positional strategy argument is not supported.\n"
            f"  Did you mean: nmem consolidate --strategy {strategy_positional}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    from neural_memory.engine.consolidation import (
        ConsolidationConfig,
        ConsolidationStrategy,
    )
    from neural_memory.engine.consolidation_delta import run_with_delta

    async def _consolidate() -> None:
        config = get_config()
        brain_name = brain or config.current_brain

        typer.echo(f"Consolidating brain '{brain_name}' (strategy: {strategy})...")
        if dry_run:
            typer.echo("(dry run - no changes will be saved)")

        storage = await get_storage(config, brain_name=brain_name)
        brain_obj = await storage.get_brain(brain_name)
        brain_id = brain_obj.id if brain_obj else brain_name

        try:
            strategies = [ConsolidationStrategy(strategy)]
        except ValueError:
            valid = ", ".join(s.value for s in ConsolidationStrategy)
            typer.secho(
                f"Error: unknown strategy '{strategy}'.\n"
                f"  Valid strategies: {valid}\n"
                f"  Example: nmem consolidate --strategy prune",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        cons_config = ConsolidationConfig(
            prune_weight_threshold=prune_threshold,
            prune_min_inactive_days=min_inactive_days,
            merge_overlap_threshold=merge_overlap,
        )

        delta = await run_with_delta(
            storage,
            brain_id,
            strategies=strategies,
            dry_run=dry_run,
            config=cons_config,
        )

        typer.echo("")
        typer.echo(delta.summary())

    run_async(_consolidate())


_HOOK_MARKER = "# [neural-memory] auto-generated hook"

_HOOK_SCRIPT = f"""#!/bin/sh
{_HOOK_MARKER}
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


def _hooks_install(post_commit: Path) -> None:
    """Install the post-commit hook."""
    import os
    import stat

    if post_commit.exists():
        existing = post_commit.read_text(encoding="utf-8")
        if _HOOK_MARKER in existing:
            typer.secho("Hook already installed.", fg=typer.colors.YELLOW)
            return
        typer.secho(
            "A post-commit hook already exists (not from neural-memory).",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo("Manually merge or remove it, then try again.")
        raise typer.Exit(1)

    post_commit.write_text(_HOOK_SCRIPT, encoding="utf-8")
    if os.name != "nt":
        st = post_commit.stat()
        post_commit.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    typer.secho("Installed post-commit hook.", fg=typer.colors.GREEN)
    typer.echo(f"  Location: {post_commit}")
    typer.echo("  After each commit you'll see a reminder to save the commit message.")


def _hooks_uninstall(post_commit: Path) -> None:
    """Uninstall the post-commit hook."""
    if not post_commit.exists():
        typer.secho("No post-commit hook found.", fg=typer.colors.YELLOW)
        return
    existing = post_commit.read_text(encoding="utf-8")
    if _HOOK_MARKER not in existing:
        typer.secho(
            "Post-commit hook exists but wasn't installed by neural-memory. Skipping.",
            fg=typer.colors.YELLOW,
        )
        return
    post_commit.unlink()
    typer.secho("Removed post-commit hook.", fg=typer.colors.GREEN)


def _hooks_show(post_commit: Path) -> None:
    """Show installed hook status."""
    if post_commit.exists():
        existing = post_commit.read_text(encoding="utf-8")
        is_nmem = _HOOK_MARKER in existing
        typer.echo(
            f"Post-commit hook: {'installed (neural-memory)' if is_nmem else 'exists (other)'}"
        )
        typer.echo(f"  Path: {post_commit}")
    else:
        typer.echo("Post-commit hook: not installed")


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

    Examples:
        nmem hooks install          # Install in current repo
        nmem hooks install -p .     # Explicit path
        nmem hooks uninstall        # Remove hooks
        nmem hooks show             # Show installed hooks
    """
    from pathlib import Path

    repo_path = Path(path).resolve() if path else Path.cwd().resolve()
    if not repo_path.is_dir():
        typer.secho("Not a valid directory", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    git_dir = repo_path / ".git"

    if not git_dir.is_dir():
        typer.secho("Not a git repository", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    post_commit = hooks_dir / "post-commit"

    dispatch = {"install": _hooks_install, "uninstall": _hooks_uninstall, "show": _hooks_show}
    handler = dispatch.get(action)
    if handler:
        handler(post_commit)
    else:
        typer.secho(
            f"Unknown action: {action}. Use: install, uninstall, show",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def install_skills(
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing skills with latest version"),
    ] = False,
    list_only: Annotated[
        bool,
        typer.Option("--list", "-l", help="List available skills without installing"),
    ] = False,
) -> None:
    """Install NeuralMemory skills to ~/.claude/skills/.

    Skills add AI agent workflows for memory management:
      memory-intake     Structured memory creation from messy notes
      memory-audit      6-dimension quality review with graded findings
      memory-evolution   Evidence-based optimization from usage patterns

    Examples:
        nmem install-skills            # Install all skills
        nmem install-skills --force    # Overwrite with latest
        nmem install-skills --list     # Show available skills
    """
    from neural_memory.cli.setup import _discover_bundled_skills, setup_skills

    bundled = _discover_bundled_skills()
    if not bundled:
        typer.secho("No bundled skills found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if list_only:
        typer.secho("  Available skills", bold=True)
        typer.echo()
        for name, path in bundled.items():
            # Extract description from SKILL.md frontmatter
            desc = _extract_skill_description(path)
            typer.echo(f"  {name:<22}{desc}")
        typer.echo()
        return

    results = setup_skills(force=force)

    # Check for ~/.claude/ not found
    if "Skills" in results:
        typer.secho(f"  {results['Skills']}", fg=typer.colors.YELLOW, err=True)
        typer.echo("  Install Claude Code first, then retry.")
        raise typer.Exit(1)

    typer.echo()
    for name, status in results.items():
        if status in ("installed", "updated"):
            icon = typer.style("[OK]", fg=typer.colors.GREEN)
        elif status == "exists":
            icon = typer.style("[--]", fg=typer.colors.YELLOW)
        elif status == "update available":
            icon = typer.style("[!!]", fg=typer.colors.CYAN)
        else:
            icon = typer.style("[!!]", fg=typer.colors.RED)
        typer.echo(f"  {icon} {name:<22}{status}")

    update_available = [n for n, s in results.items() if s == "update available"]
    if update_available:
        typer.echo()
        typer.echo("  Use --force to overwrite with latest versions.")

    typer.echo()


def _extract_skill_description(skill_path: Path) -> str:
    """Extract description from SKILL.md YAML frontmatter."""
    try:
        content = skill_path.read_text(encoding="utf-8")
        in_frontmatter = False
        in_desc = False
        desc_lines: list[str] = []

        for line in content.splitlines():
            if line.strip() == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if not in_frontmatter:
                continue
            if line.startswith("description:"):
                rest = line[len("description:") :].strip()
                if rest and rest != "|":
                    return rest
                in_desc = True
                continue
            if in_desc:
                stripped = line.strip()
                if stripped and not line.startswith(" "):
                    break
                if stripped:
                    desc_lines.append(stripped)

        if desc_lines:
            return desc_lines[0]
    except OSError:
        pass
    return ""


def flush(
    transcript: Annotated[
        str | None,
        typer.Option("--transcript", "-t", help="Path to JSONL transcript file"),
    ] = None,
    text: Annotated[
        str | None,
        typer.Argument(help="Direct text to flush (reads stdin if omitted)"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Emergency flush: capture memories before context is lost.

    Aggressively detects and saves all memorable content with
    lowered confidence threshold. Designed for pre-compaction
    or session-end scenarios.

    Examples:
        nmem flush "We decided to use Redis for caching"
        nmem flush --transcript /path/to/transcript.jsonl
        echo "important text" | nmem flush
    """
    import json as json_mod
    import sys

    from neural_memory.hooks.pre_compact import flush_text, read_transcript_tail

    flush_content = ""

    if text:
        flush_content = text
    elif transcript:
        flush_content = read_transcript_tail(transcript)
        if not flush_content:
            typer.secho(f"No content found in transcript: {transcript}", fg=typer.colors.YELLOW)
            raise typer.Exit(0)
    else:
        # Try reading from stdin (piped input)
        if not sys.stdin.isatty():
            flush_content = sys.stdin.read()

    if not flush_content or len(flush_content.strip()) < 50:
        typer.secho("Not enough content to flush (minimum 50 chars).", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    async def _flush() -> dict[str, Any]:
        return await flush_text(flush_content)

    result = run_async(_flush())

    if json_output:
        typer.echo(json_mod.dumps(result, indent=2, default=str))
    else:
        saved = result.get("saved", 0)
        if saved > 0:
            typer.secho(
                f"Emergency flush: captured {saved} memories",
                fg=typer.colors.GREEN,
            )
            for mem in result.get("memories", []):
                typer.echo(f"  - {mem}")
        else:
            typer.secho(
                result.get("message", "No memories captured"),
                fg=typer.colors.YELLOW,
            )


def register(app: typer.Typer) -> None:
    """Register tool commands on the app."""
    app.command()(mcp)
    app.command()(dashboard)
    app.command()(ui)
    app.command()(graph)
    app.command()(init)
    app.command()(serve)
    app.command()(decay)
    app.command()(consolidate)
    app.command()(hooks)
    app.command()(flush)
    app.command(name="install-skills")(install_skills)
