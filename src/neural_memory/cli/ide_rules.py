"""IDE rules generator — creates rules files for AI coding assistants.

Generates rules files that instruct IDE agents (Cursor, Windsurf, Cline,
Gemini, etc.) to use Neural Memory effectively.

Usage:
    nmem setup rules              # Interactive: choose IDEs
    nmem setup rules --all        # Generate for all supported IDEs
    nmem setup rules --ide cursor # Single IDE
"""

from __future__ import annotations

from pathlib import Path

import typer

# -- Supported IDEs and their rules file paths --------------------------------

IDE_TARGETS: dict[str, dict[str, str]] = {
    "cursor": {
        "file": ".cursorrules",
        "name": "Cursor",
        "description": "Cursor AI rules file",
    },
    "windsurf": {
        "file": ".windsurfrules",
        "name": "Windsurf",
        "description": "Windsurf (Codeium) rules file",
    },
    "cline": {
        "file": ".clinerules",
        "name": "Cline",
        "description": "Cline rules file (also works as folder)",
    },
    "gemini": {
        "file": "GEMINI.md",
        "name": "Gemini",
        "description": "Gemini Code Assist / Gemini CLI rules",
    },
    "codex": {
        "file": "codex.md",
        "name": "Codex CLI (OpenAI)",
        "description": "OpenAI Codex CLI instructions file",
    },
    "agents": {
        "file": "AGENTS.md",
        "name": "AGENTS.md (cross-platform)",
        "description": "Cross-platform standard (Cline, Gemini, others)",
    },
}


def _get_rules_content() -> str:
    """Generate the NM rules content for IDE agents."""
    return """\
# Neural Memory Integration

This project uses [Neural Memory](https://github.com/nhadaututtheky/neural-memory) \
for persistent AI memory across sessions.

## Session Start (MANDATORY)

Before starting any code work, load context from previous sessions:

```
nmem_recall("current project context")
nmem_recall("<specific topic you're working on>")
```

This activates relevant neurons and loads cross-session decisions, patterns, and preferences.

## After Each Completed Task (MANDATORY)

Save what you learned, decided, or fixed:

```
nmem_remember(
    content="Chose PostgreSQL over MongoDB because ACID needed for payments",
    type="decision",
    tags=["database", "payments"],
    context={"reason": "ACID compliance", "alternatives": ["MongoDB"]}
)
```

### Memory Types
| Type | When to Use | Example |
|------|------------|---------|
| decision | Chose X over Y | "Chose React Query over SWR because..." |
| error | Bug root cause + fix | "Auth broke because new cookie format, fixed by..." |
| insight | Pattern or learning | "This codebase uses repository pattern for..." |
| workflow | Process or steps | "Deploy: build → test → push to staging" |
| preference | User/team preference | "User prefers dark mode on all dashboards" |
| instruction | Rule to follow | "Always run linter before commit because..." |
| fact | Verified information | "API endpoint is /v2/users, rate limit 100/min" |

### Quality Tips
- NEVER store flat facts like `"PostgreSQL"` alone
- ALWAYS include WHY: `"because..."`, `"after..."`, `"over X instead of Y"`
- ALWAYS include tags with project name: `["myproject", "auth", "bug-fix"]`
- Use `context` dict for structured data — NM merges it server-side into rich content
- Response includes `quality` score (0-10) and `hints` — aim for score >= 7

### Context Dict (Structured Enrichment)
Instead of crafting perfect sentences, send structured data:

```json
{
    "content": "Auth middleware broke",
    "type": "error",
    "context": {
        "root_cause": "new cookie format after v3 upgrade",
        "fix": "updated parser to handle SameSite attribute",
        "prevention": "add cookie format integration test"
    }
}
```

NM automatically merges into: "Auth middleware broke. Root cause: new cookie format \
after v3 upgrade. Fixed by updated parser to handle SameSite attribute. \
Prevention: add cookie format integration test."

## Memory Maintenance

| Action | When | Command |
|--------|------|---------|
| Check brain health | Weekly | `nmem_health()` |
| Consolidate memories | Weekly | `nmem consolidate` |
| Review queue | Monthly | `nmem_review(action="queue")` |

## What NOT to Remember
- Code patterns visible in the codebase (just read the code)
- Git history (use `git log`)
- Temporary debugging state
- Information already in project docs
"""


def _get_codex_rules_content() -> str:
    """Generate Codex CLI-specific rules with hook-like behavior.

    Codex CLI reads codex.md as system instructions. Since Codex doesn't
    support event hooks like Claude Code, we embed the hook behavior
    directly into the instructions so Codex triggers them proactively.
    """
    return """\
# Neural Memory Integration — Codex CLI

This project uses [Neural Memory](https://github.com/nhadaututtheky/neural-memory) \
for persistent AI memory across sessions. Neural Memory is available as an MCP server.

## Hooks (Execute Automatically)

Codex does not have built-in lifecycle hooks. Instead, treat these as \
**mandatory actions** that you MUST execute at the specified trigger points.

### Hook 1: Session Start (BEFORE any code work)

**Trigger**: Beginning of every session, before reading files or writing code.

```
nmem_recall("current project context")
nmem_recall("<topic related to user's request>")
nmem_recap()
```

This loads cross-session decisions, patterns, and known issues. \
Skipping this means you will repeat past mistakes or contradict prior decisions.

### Hook 2: After Each Task Completion

**Trigger**: After completing a feature, fixing a bug, or making a decision.

```
nmem_remember(
    content="<what happened and why — use causal language>",
    type="<decision|error|insight|workflow|preference|instruction|fact>",
    tags=["<project>", "<topic>"],
    context={"reason": "...", "alternatives": ["..."]}
)
```

### Hook 3: Session End

**Trigger**: Before finishing the session or when the user says "done".

```
nmem_auto(action="process", summary="<1-2 sentence session summary>")
```

## Memory Types

| Type | When to Use | Example |
|------|------------|---------|
| decision | Chose X over Y | "Chose React Query over SWR because..." |
| error | Bug root cause + fix | "Auth broke because new cookie format, fixed by..." |
| insight | Pattern or learning | "This codebase uses repository pattern for..." |
| workflow | Process or steps | "Deploy: build -> test -> push to staging" |
| preference | User/team preference | "User prefers dark mode on all dashboards" |
| instruction | Rule to follow | "Always run linter before commit because..." |
| fact | Verified information | "API endpoint is /v2/users, rate limit 100/min" |

## Quality Rules

- Max 1-3 sentences per memory
- ALWAYS include WHY: "because...", "after...", "over X instead of Y"
- ALWAYS include tags with project name
- Use `context` dict for structured data — NM merges it into rich content
- NEVER store flat facts without reasoning
- NEVER store code/file contents (derivable from codebase)

## Maintenance

| Action | When | Command |
|--------|------|---------|
| Check brain health | Weekly | `nmem_health()` |
| Consolidate memories | Weekly | `nmem_consolidate()` |
| Review queue | Monthly | `nmem_review(action="queue")` |
"""


def generate_rules_file(
    target_dir: Path,
    ide: str,
    *,
    force: bool = False,
) -> str:
    """Generate a rules file for the specified IDE.

    Args:
        target_dir: Directory to write the file in (usually project root).
        ide: IDE key from IDE_TARGETS.
        force: Overwrite existing file.

    Returns:
        Status string: "created", "exists", "unknown".
    """
    if ide not in IDE_TARGETS:
        return "unknown"

    target = IDE_TARGETS[ide]
    file_path = target_dir / target["file"]

    if file_path.exists() and not force:
        return "exists"

    content = _get_codex_rules_content() if ide == "codex" else _get_rules_content()
    file_path.write_text(content, encoding="utf-8")
    return "created"


def run_ide_rules_setup(
    *,
    ide: str | None = None,
    all_ides: bool = False,
    force: bool = False,
) -> None:
    """Run the IDE rules file generator.

    Args:
        ide: Specific IDE to generate for.
        all_ides: Generate for all supported IDEs.
        force: Overwrite existing files.
    """
    target_dir = Path.cwd()

    if ide:
        if ide not in IDE_TARGETS:
            typer.secho(
                f"Unknown IDE: {ide}. Available: {', '.join(IDE_TARGETS)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        targets = [ide]
    elif all_ides:
        targets = list(IDE_TARGETS.keys())
    else:
        # Interactive selection
        typer.echo("Select IDEs to generate rules for:\n")
        for i, (_key, info) in enumerate(IDE_TARGETS.items(), 1):
            typer.echo(f"  {i}. {info['name']:30s} -> {info['file']}")
        typer.echo(f"  {len(IDE_TARGETS) + 1}. All of the above")
        typer.echo()

        choice = typer.prompt("Enter numbers (comma-separated)", default="5")
        selected: list[str] = []
        keys = list(IDE_TARGETS.keys())

        for part in choice.split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            idx = int(part)
            if idx == len(IDE_TARGETS) + 1:
                selected = keys[:]
                break
            if 1 <= idx <= len(keys):
                selected.append(keys[idx - 1])

        if not selected:
            typer.secho("No IDEs selected.", fg=typer.colors.YELLOW)
            return

        targets = selected

    # Generate files
    typer.echo()
    for target_ide in targets:
        info = IDE_TARGETS[target_ide]
        status = generate_rules_file(target_dir, target_ide, force=force)

        if status == "created":
            typer.secho(
                f"  [+] {info['file']:25s} -- created",
                fg=typer.colors.GREEN,
            )
        elif status == "exists":
            typer.secho(
                f"  [=] {info['file']:25s} -- already exists (use --force to overwrite)",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"  [x] {info['file']:25s} -- unknown IDE",
                fg=typer.colors.RED,
            )

    typer.echo()
    typer.echo("  Rules tell your IDE agent to use Neural Memory automatically.")
    typer.echo("  Restart your IDE to activate.\n")
