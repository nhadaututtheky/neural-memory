# CLI Reference

Complete reference for the `nmem` command-line interface.
**77 commands** available.

!!! tip
    Run `nmem --help` or `nmem <command> --help` for the latest usage info.

## Table of Contents

- [Memory Operations](#memory)
  - [`nmem remember`](#nmem-remember)
  - [`nmem recall`](#nmem-recall)
  - [`nmem context`](#nmem-context)
  - [`nmem todo`](#nmem-todo)
  - [`nmem q`](#nmem-q)
  - [`nmem a`](#nmem-a)
  - [`nmem last`](#nmem-last)
  - [`nmem today`](#nmem-today)
- [Brain Management](#brain)
  - [`nmem brain list`](#nmem-brain-list)
  - [`nmem brain use`](#nmem-brain-use)
  - [`nmem brain create`](#nmem-brain-create)
  - [`nmem brain export`](#nmem-brain-export)
  - [`nmem brain import`](#nmem-brain-import)
  - [`nmem brain delete`](#nmem-brain-delete)
  - [`nmem brain health`](#nmem-brain-health)
  - [`nmem brain transplant`](#nmem-brain-transplant)
- [Information & Diagnostics](#info)
  - [`nmem stats`](#nmem-stats)
  - [`nmem status`](#nmem-status)
  - [`nmem health`](#nmem-health)
  - [`nmem check`](#nmem-check)
  - [`nmem doctor`](#nmem-doctor)
  - [`nmem dashboard`](#nmem-dashboard)
  - [`nmem ui`](#nmem-ui)
  - [`nmem graph`](#nmem-graph)
- [Training & Import/Export](#training)
  - [`nmem train`](#nmem-train)
  - [`nmem index`](#nmem-index)
  - [`nmem import`](#nmem-import)
  - [`nmem export`](#nmem-export)
- [Configuration & Setup](#config)
  - [`nmem init`](#nmem-init)
  - [`nmem setup`](#nmem-setup)
  - [`nmem mcp-config`](#nmem-mcp-config)
  - [`nmem prompt`](#nmem-prompt)
  - [`nmem hooks`](#nmem-hooks)
  - [`nmem config preset`](#nmem-config-preset)
  - [`nmem config tier`](#nmem-config-tier)
  - [`nmem install-skills`](#nmem-install-skills)
- [Server & MCP](#server)
  - [`nmem serve`](#nmem-serve)
  - [`nmem mcp`](#nmem-mcp)
- [Maintenance](#maintenance)
  - [`nmem decay`](#nmem-decay)
  - [`nmem consolidate`](#nmem-consolidate)
  - [`nmem cleanup`](#nmem-cleanup)
  - [`nmem flush`](#nmem-flush)
- [Project Management](#project)
  - [`nmem project create`](#nmem-project-create)
  - [`nmem project list`](#nmem-project-list)
  - [`nmem project show`](#nmem-project-show)
  - [`nmem project delete`](#nmem-project-delete)
  - [`nmem project extend`](#nmem-project-extend)
- [Advanced Features](#advanced)
  - [`nmem shared enable`](#nmem-shared-enable)
  - [`nmem shared disable`](#nmem-shared-disable)
  - [`nmem shared status`](#nmem-shared-status)
  - [`nmem shared test`](#nmem-shared-test)
  - [`nmem shared sync`](#nmem-shared-sync)
  - [`nmem habits list`](#nmem-habits-list)
  - [`nmem habits show`](#nmem-habits-show)
  - [`nmem habits clear`](#nmem-habits-clear)
  - [`nmem habits status`](#nmem-habits-status)
  - [`nmem version create`](#nmem-version-create)
  - [`nmem version list`](#nmem-version-list)
  - [`nmem version rollback`](#nmem-version-rollback)
  - [`nmem version diff`](#nmem-version-diff)
  - [`nmem telegram status`](#nmem-telegram-status)
  - [`nmem telegram test`](#nmem-telegram-test)
  - [`nmem telegram backup`](#nmem-telegram-backup)
  - [`nmem list`](#nmem-list)
  - [`nmem migrate`](#nmem-migrate)
  - [`nmem update`](#nmem-update)
- [Other](#other)
  - [`nmem lifecycle`](#nmem-lifecycle)
  - [`nmem shared activate`](#nmem-shared-activate)
  - [`nmem storage status`](#nmem-storage-status)
  - [`nmem storage switch`](#nmem-storage-switch)
  - [`nmem sync activate`](#nmem-sync-activate)
  - [`nmem sync disable`](#nmem-sync-disable)
  - [`nmem sync enable`](#nmem-sync-enable)
  - [`nmem sync status`](#nmem-sync-status)
  - [`nmem sync sync`](#nmem-sync-sync)
  - [`nmem sync test`](#nmem-sync-test)
  - [`nmem watch`](#nmem-watch)

---

## Memory Operations {#memory}

### `nmem remember`

Store a new memory (type auto-detected if not specified).

```
nmem remember [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | No | `` | (positional argument) |
| `--tag / -t` | text | No | — | Tags for the memory |
| `--type / -T` | text | No | — | Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detect... |
| `--priority / -p` | integer | No | — | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires / -e` | integer | No | — | Days until this memory expires |
| `--project / -P` | text | No | — | Associate with a project (by name) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--force / -f` | boolean | No | `False` | Store even if sensitive content detected |
| `--redact / -r` | boolean | No | `False` | Auto-redact sensitive content before storing |
| `--timestamp / --at` | text | No | — | ISO datetime of original event (e.g. '2026-03-02T08:00:00'). Defaults to now. |
| `--ephemeral` | boolean | No | `False` | Session-scoped memory: auto-expires after 24h, never synced |
| `--stdin` | boolean | No | `False` | Read content from stdin (safe for shell-special characters) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem recall`

Query memories with intelligent routing (query type auto-detected).

```
nmem recall [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `--depth / -d` | integer | No | — | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--limit / -l` | integer | No | — | Cap matched memories (approximate; maps to max_tokens ~200/memory) |
| `--max-tokens / -m` | integer | No | `500` | Max tokens in response |
| `--min-confidence / -c` | float | No | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--show-age / -a` | boolean | No | `True` | Show memory ages in results |
| `--show-routing / -R` | boolean | No | `False` | Show query routing info |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem context`

Get recent context (for injecting into AI conversations).

```
nmem context [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `10` | Number of recent memories |
| `--fresh-only` | boolean | No | `False` | Only include memories < 30 days old |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem todo`

Quick shortcut to add a TODO memory.

```
nmem todo [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `task` | text | Yes | — | (positional argument) |
| `--priority / -p` | integer | No | `5` | Priority 0-10 (default: 5=normal, 7=high, 10=critical) |
| `--project / -P` | text | No | — | Associate with a project |
| `--expires / -e` | integer | No | — | Days until expiry (default: 30) |
| `--tag / -t` | text | No | — | Tags for the task |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem q`

Quick recall - shortcut for 'nmem recall'.

```
nmem q [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `-d` | integer | No | — | — |

### `nmem a`

Quick add - shortcut for 'nmem remember' with auto-detect.

```
nmem a [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `-p` | integer | No | — | — |

### `nmem last`

Show last N memories - quick view of recent activity.

```
nmem last [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `-n` | integer | No | `5` | Number of memories to show |

### `nmem today`

Show today's memories.

```
nmem today [OPTIONS]
```

## Brain Management {#brain}

### `nmem brain list`

List available brains.

```
nmem brain list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem brain use`

Switch to a different brain.

```
nmem brain use [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |

### `nmem brain create`

Create a new brain.

```
nmem brain create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--use / -u` | boolean | No | `True` | Switch to the new brain after creating |

### `nmem brain export`

Export brain to JSON or markdown file.

```
nmem brain export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output / -o` | text | No | — | Output file path |
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--exclude-sensitive / -s` | boolean | No | `False` | Exclude memories with sensitive content |
| `--format / -f` | text | No | `json` | Export format: json or markdown |

### `nmem brain import`

Import brain from JSON file.

```
nmem brain import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `file` | text | Yes | — | (positional argument) |
| `--name / -n` | text | No | — | Name for imported brain |
| `--use / -u` | boolean | No | `True` | Switch to imported brain |
| `--scan` | boolean | No | `True` | Scan for sensitive content before importing |

### `nmem brain delete`

Delete a brain.

```
nmem brain delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `nmem brain health`

Check brain health (freshness, sensitive content).

```
nmem brain health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem brain transplant`

Transplant memories from another brain into the current brain.

```
nmem brain transplant [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | text | Yes | — | (positional argument) |
| `--tag / -t` | text | No | — | Filter by tags |
| `--type` | text | No | — | Filter by memory types |
| `--strategy / -s` | text | No | `prefer_local` | Conflict resolution strategy |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Information & Diagnostics {#info}

### `nmem stats`

Show brain statistics including freshness and memory type analysis.

```
nmem stats [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem status`

Show current brain status, recent activity, and actionable suggestions.

```
nmem status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem health`

Show brain health diagnostics with purity score and recommendations.

```
nmem health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem check`

Check content for sensitive information without storing.

```
nmem check [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem doctor`

Run system health diagnostics.

```
nmem doctor [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |
| `--fix` | boolean | No | `False` | Auto-fix available issues |

### `nmem dashboard`

Show a rich dashboard with brain stats and recent activity.

```
nmem dashboard [OPTIONS]
```

### `nmem ui`

Interactive memory browser with rich formatting.

```
nmem ui [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -t` | text | No | — | Filter by memory type |
| `--search / -s` | text | No | — | Search in memory content |
| `--limit / -n` | integer | No | `20` | Number of memories to show |

### `nmem graph`

Visualize neural connections as a tree graph.

```
nmem graph [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | No | — | (positional argument) |
| `--depth / -d` | integer | No | `2` | Traversal depth (1-3) |
| `--export / -e` | text | No | — | Export format: svg |
| `--output / -o` | text | No | — | Output file path (used with --export) |

## Training & Import/Export {#training}

### `nmem train`

Train a brain from documentation files (markdown).

```
nmem train [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--domain / -d` | text | No | `` | Domain tag (e.g., react, kubernetes) |
| `--brain / -b` | text | No | `` | Target brain name (default: current) |
| `--ext / -e` | text | No | — | File extensions (default: .md) |
| `--no-consolidate` | boolean | No | `False` | Skip ENRICH consolidation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem index`

Index a codebase into neural memory for code-aware recall.

```
nmem index [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--ext / -e` | text | No | — | File extensions to index (e.g. .py) |
| `--status / -s` | boolean | No | `False` | Show indexing status instead of scanning |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem import`

Import brain from JSON file.

```
nmem import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `input_file` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Target brain name (default: from file) |
| `--merge / -m` | boolean | No | `False` | Merge with existing brain |
| `--strategy` | text | No | `prefer_local` | Conflict resolution: prefer_local, prefer_remote, prefer_recent, prefer_stronger |

### `nmem export`

Export brain to JSON file for backup or sharing.

```
nmem export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `output` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to export (default: current) |

## Configuration & Setup {#config}

### `nmem init`

Set up NeuralMemory in one command.

```
nmem init [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing config |
| `--skip-mcp` | boolean | No | `False` | Skip MCP auto-configuration |
| `--skip-skills` | boolean | No | `False` | Skip skills installation |
| `--wizard / -w` | boolean | No | `False` | Interactive setup wizard |
| `--defaults` | boolean | No | `False` | Non-interactive with all defaults |
| `--full` | boolean | No | `False` | Extended setup: embeddings, dedup, maintenance script |
| `--embeddings` | boolean | No | `False` | Set up embedding provider |
| `--skip-embeddings` | boolean | No | `False` | Skip embedding provider setup |
| `--dedup` | boolean | No | `False` | Enable dedup |

### `nmem setup`

Set up optional components.

```
nmem setup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `component` | text | No | `` | (positional argument) |
| `--ide` | text | No | `` | Target IDE for rules: cursor, windsurf, cline, gemini, agents |
| `--all` | boolean | No | `False` | Generate rules for all supported IDEs |
| `--force / -f` | boolean | No | `False` | Overwrite existing files |

### `nmem mcp-config`

Generate MCP server configuration for Claude Code/Cursor.

```
nmem mcp-config [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--with-prompt / -p` | boolean | No | `False` | Include system prompt in config |
| `--compact / -c` | boolean | No | `False` | Use compact prompt (if --with-prompt) |

### `nmem prompt`

Show system prompt for AI tools.

```
nmem prompt [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--compact / -c` | boolean | No | `False` | Show compact version |
| `--copy` | boolean | No | `False` | Copy to clipboard (requires pyperclip) |

### `nmem hooks`

Install or manage git hooks for automatic memory capture.

```
nmem hooks [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `action` | text | No | `install` | (positional argument) |
| `--path / -p` | text | No | — | Path to git repo (default: current dir) |

### `nmem config preset`

Apply a configuration preset or list available presets.

```
nmem config preset [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--list / -l` | boolean | No | `False` | List available presets |
| `--dry-run / -n` | boolean | No | `False` | Show changes without applying |

### `nmem config tier`

Get or set the MCP tool tier to control token usage.

```
nmem config tier [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--show / -s` | boolean | No | `False` | Show current tier |

### `nmem install-skills`

Install NeuralMemory skills to ~/.claude/skills/.

```
nmem install-skills [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing skills with latest version |
| `--list / -l` | boolean | No | `False` | List available skills without installing |

## Server & MCP {#server}

### `nmem serve`

Run the NeuralMemory API server.

```
nmem serve [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--host / -h` | text | No | `127.0.0.1` | Host to bind to |
| `--port / -p` | integer | No | `8000` | Port to bind to |
| `--reload / -r` | boolean | No | `False` | Enable auto-reload for development |

### `nmem mcp`

Run the MCP (Model Context Protocol) server.

```
nmem mcp [OPTIONS]
```

## Maintenance {#maintenance}

### `nmem decay`

Apply memory decay to simulate forgetting.

```
nmem decay [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain to apply decay to |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune / -p` | float | No | `0.01` | Prune below this activation level |

### `nmem consolidate`

Consolidate brain memories by pruning, merging, or summarizing.

```
nmem consolidate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `strategy_positional` | text | No | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to consolidate |
| `--strategy / -s` | text | No | `all` | Consolidation strategy. Valid values: prune, merge, summarize, mature, infer, enrich, dream, learn_habits, dedup, sem... |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune-threshold` | float | No | `0.05` | Weight threshold for pruning synapses |
| `--merge-overlap` | float | No | `0.5` | Jaccard overlap threshold for merging fibers |
| `--min-inactive-days` | float | No | `7.0` | Minimum inactive days before pruning |

### `nmem cleanup`

Clean up expired or old memories.

```
nmem cleanup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--expired / -e` | boolean | No | `True` | Only clean up expired memories |
| `--type / -T` | text | No | — | Only clean up specific memory type |
| `--dry-run / -n` | boolean | No | `False` | Show what would be deleted without deleting |
| `--force / -f` | boolean | No | `False` | Skip confirmation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem flush`

Emergency flush: capture memories before context is lost.

```
nmem flush [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--transcript / -t` | text | No | — | Path to JSONL transcript file |
| `text` | text | No | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Project Management {#project}

### `nmem project create`

Create a new project for organizing memories.

```
nmem project create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | — | Project description |
| `--duration / -D` | integer | No | — | Duration in days (creates end date) |
| `--tag / -t` | text | No | — | Project tags |
| `--priority / -p` | float | No | `1.0` | Project priority (default: 1.0) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem project list`

List all projects.

```
nmem project list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--active / -a` | boolean | No | `False` | Show only active projects |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem project show`

Show project details and its memories.

```
nmem project show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem project delete`

Delete a project (memories are preserved but unlinked).

```
nmem project delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `nmem project extend`

Extend a project's deadline.

```
nmem project extend [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `days` | integer | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Advanced Features {#advanced}

### `nmem shared enable`

Enable shared mode to connect to a remote NeuralMemory server.

```
nmem shared enable [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `server_url` | text | Yes | — | (positional argument) |
| `--api-key / -k` | text | No | — | API key for authentication |
| `--timeout / -t` | float | No | `30.0` | Request timeout in seconds |

### `nmem shared disable`

Disable shared mode and use local storage.

```
nmem shared disable [OPTIONS]
```

### `nmem shared status`

Show shared mode status and configuration.

```
nmem shared status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem shared test`

Test connection to the shared server.

```
nmem shared test [OPTIONS]
```

### `nmem shared sync`

Manually sync local brain with remote server.

```
nmem shared sync [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--direction / -d` | text | No | `both` | Sync direction: push, pull, or both |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem habits list`

List learned workflow habits.

```
nmem habits list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem habits show`

Show details of a specific learned habit.

```
nmem habits show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem habits clear`

Clear all learned habits.

```
nmem habits clear [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `nmem habits status`

Show progress toward habit detection.

```
nmem habits status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem version create`

Create a version snapshot of the current brain state.

```
nmem version create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | `` | Description |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem version list`

List brain versions.

```
nmem version list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `20` | Max versions |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem version rollback`

Rollback brain to a previous version.

```
nmem version rollback [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `version_id` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem version diff`

Compare two brain versions.

```
nmem version diff [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `from_version` | text | Yes | — | (positional argument) |
| `to_version` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem telegram status`

Show Telegram integration status.

```
nmem telegram status [OPTIONS]
```

### `nmem telegram test`

Send a test message to verify configuration.

```
nmem telegram test [OPTIONS]
```

### `nmem telegram backup`

Send brain database file as backup to Telegram.

```
nmem telegram backup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain name (default: active brain) |

### `nmem list`

List memories with filtering by type, priority, project, and status.

```
nmem list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -T` | text | No | — | Filter by memory type (fact, decision, todo, etc.) |
| `--min-priority / -p` | integer | No | — | Minimum priority (0-10) |
| `--project / -P` | text | No | — | Filter by project name |
| `--expired / -e` | boolean | No | `False` | Show only expired memories |
| `--include-expired` | boolean | No | `False` | Include expired memories in results |
| `--limit / -l` | integer | No | `20` | Maximum number of results |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem migrate`

Migrate brain data between storage backends.

```
nmem migrate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `target` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Specific brain to migrate (default: current) |
| `--pg-host` | text | No | `localhost` | PostgreSQL host |
| `--pg-port` | integer | No | `5432` | PostgreSQL port |
| `--pg-database` | text | No | `neuralmemory` | PostgreSQL database name |
| `--pg-user` | text | No | `postgres` | PostgreSQL user |
| `--pg-password` | text | No | `` | PostgreSQL password (or use NEURAL_MEMORY_POSTGRES_PASSWORD env) |

### `nmem update`

Update neural-memory to the latest version.

```
nmem update [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Force update even if already latest |
| `--check / -c` | boolean | No | `False` | Only check for updates, don't install |

## Other {#other}

### `nmem lifecycle`

Manage memory lifecycle states.

```
nmem lifecycle [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `action` | text | No | `status` | (positional argument) |
| `neuron_id` | text | No | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain name |

### `nmem shared activate`

Activate a NeuralMemory Pro license key.

```
nmem shared activate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--key / -k` | text | Yes | — | License key (NM-PRO-XXXX-XXXX-XXXX) |

### `nmem storage status`

Show current storage backend status.

```
nmem storage status [OPTIONS]
```

### `nmem storage switch`

Switch active storage backend.

```
nmem storage switch [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `backend` | text | Yes | — | (positional argument) |

### `nmem sync activate`

Activate a NeuralMemory Pro license key.

```
nmem sync activate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--key / -k` | text | Yes | — | License key (NM-PRO-XXXX-XXXX-XXXX) |

### `nmem sync disable`

Disable shared mode and use local storage.

```
nmem sync disable [OPTIONS]
```

### `nmem sync enable`

Enable shared mode to connect to a remote NeuralMemory server.

```
nmem sync enable [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `server_url` | text | Yes | — | (positional argument) |
| `--api-key / -k` | text | No | — | API key for authentication |
| `--timeout / -t` | float | No | `30.0` | Request timeout in seconds |

### `nmem sync status`

Show shared mode status and configuration.

```
nmem sync status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem sync sync`

Manually sync local brain with remote server.

```
nmem sync sync [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--direction / -d` | text | No | `both` | Sync direction: push, pull, or both |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `nmem sync test`

Test connection to the shared server.

```
nmem sync test [OPTIONS]
```

### `nmem watch`

Watch a directory and auto-ingest files into memory.

```
nmem watch [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `directory` | text | No | `` | (positional argument) |
| `--action / -a` | text | No | `scan` | Action: scan, start, stop, status |
| `--json / -j` | boolean | No | `False` | Output as JSON |

---

*Auto-generated by `scripts/gen_cli_docs.py` from Typer app introspection — 77 commands.*
