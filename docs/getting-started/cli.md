# CLI Reference

Complete reference for all NeuralMemory CLI commands.

## Core Commands

### nmem remember

Store a memory in the brain.

```bash
nmem remember "content" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--tag` | `-t` | Tags for the memory (repeatable) |
| `--type` | `-T` | Memory type (auto-detected if not specified) |
| `--priority` | `-p` | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires` | `-e` | Days until expiry |
| `--project` | `-P` | Associate with project |
| `--shared` | `-S` | Use shared/remote storage |
| `--force` | `-f` | Store even if sensitive content detected |
| `--redact` | `-r` | Auto-redact sensitive content |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
nmem remember "Fixed auth bug with null check"
nmem remember "We decided to use PostgreSQL" --type decision
nmem remember "Refactor auth module" --type todo --priority 7
nmem remember "Meeting notes" --expires 7 --tag meeting
nmem remember "Team knowledge" --shared
```

### nmem recall

Query memories using spreading activation.

```bash
nmem recall "query" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--depth` | `-d` | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--max-tokens` | `-m` | Max tokens in response (default: 500) |
| `--min-confidence` | `-c` | Minimum confidence threshold |
| `--shared` | `-S` | Use shared/remote storage |
| `--show-age` | `-a` | Show memory ages (default: true) |
| `--show-routing` | `-R` | Show query routing info |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
nmem recall "auth bug fix"
nmem recall "meetings with Alice" --depth 2
nmem recall "Why did the build fail?" --show-routing
nmem recall "team decisions" --shared --min-confidence 0.5
```

### nmem todo

Quick shortcut for TODO items.

```bash
nmem todo "task" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--priority` | `-p` | Priority 0-10 (default: 5) |
| `--project` | `-P` | Associate with project |
| `--expires` | `-e` | Days until expiry (default: 30) |
| `--tag` | `-t` | Tags (repeatable) |
| `--json` | `-j` | Output as JSON |

### nmem context

Get recent memories for context injection.

```bash
nmem context [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-l` | Number of recent memories (default: 10) |
| `--fresh-only` | | Only include memories < 30 days old |
| `--json` | `-j` | Output as JSON |

### nmem list

List memories with filters.

```bash
nmem list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-T` | Filter by memory type |
| `--min-priority` | `-p` | Minimum priority |
| `--project` | `-P` | Filter by project |
| `--expired` | `-e` | Show only expired memories |
| `--include-expired` | | Include expired in results |
| `--limit` | `-l` | Maximum results (default: 20) |
| `--json` | `-j` | Output as JSON |

### nmem stats

Show brain statistics.

```bash
nmem stats [--json]
```

### nmem check

Check content for sensitive information.

```bash
nmem check "content" [--json]
```

### nmem cleanup

Clean expired memories.

```bash
nmem cleanup [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--expired` | `-e` | Only clean expired (default: true) |
| `--type` | `-T` | Only clean specific type |
| `--dry-run` | `-n` | Preview without deleting |
| `--force` | `-f` | Skip confirmation |

### nmem consolidate

Consolidate brain memories: prune weak links, merge overlapping fibers,
advance episodic memories to semantic stage, and more.

```bash
nmem consolidate [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--brain` | `-b` | Brain to consolidate (default: current) |
| `--strategy` | `-s` | Strategy to run (default: `all`) |
| `--dry-run` | `-n` | Preview without applying changes |
| `--prune-threshold` | | Synapse weight threshold for pruning (default: 0.05) |
| `--merge-overlap` | | Jaccard overlap threshold for merging (default: 0.5) |
| `--min-inactive-days` | | Minimum inactive days before pruning (default: 7.0) |

**Valid strategies:**

| Strategy | Description |
|----------|-------------|
| `prune` | Remove weak synapses and orphaned neurons |
| `merge` | Combine overlapping fibers |
| `summarize` | Create concept neurons for topic clusters |
| `mature` | Advance episodic memories to semantic stage |
| `infer` | Add inferred synapses from co-activation patterns |
| `enrich` | Enrich neurons with extracted metadata |
| `dream` | Generate synthetic bridging memories |
| `learn_habits` | Extract recurring workflow patterns |
| `dedup` | Merge near-duplicate memories |
| `semantic_link` | Add cross-domain semantic connections |
| `compress` | Compress old low-activation fibers |
| `all` | Run all strategies in dependency order (default) |

> **Note:** `mature` is a fully supported strategy. It advances episodic memories
> to the semantic stage, which improves recall quality. `nmem health` may recommend
> running it when the consolidation ratio is low.

**Examples:**

```bash
nmem consolidate                          # Run all strategies
nmem consolidate --strategy prune         # Only prune weak links
nmem consolidate -s mature                # Advance episodic memories
nmem consolidate --dry-run                # Preview without changes
nmem consolidate -s merge --merge-overlap 0.3
```

> **Tip:** Always use `--strategy <name>` (named flag). Positional syntax
> (`nmem consolidate prune`) is not supported and will produce a helpful error.

### nmem decay

Apply memory decay (Ebbinghaus forgetting curve).

```bash
nmem decay [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--brain` | Brain name (default: current) |
| `--dry-run` | Preview without applying |
| `--prune-threshold` | Threshold for pruning (default: 0.01) |

---

## Brain Commands

### nmem brain list

List all brains.

```bash
nmem brain list [--json]
```

### nmem brain create

Create a new brain.

```bash
nmem brain create NAME [--use/--no-use]
```

### nmem brain use

Switch to a brain.

```bash
nmem brain use NAME
```

### nmem brain export

Export brain to file.

```bash
nmem brain export [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path |
| `--name` | `-n` | Brain name (default: current) |
| `--exclude-sensitive` | `-s` | Exclude sensitive content |

### nmem brain import

Import brain from file.

```bash
nmem brain import FILE [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Name for imported brain |
| `--use` | `-u` | Switch to imported brain |
| `--merge` | | Merge with existing brain |
| `--scan` | | Scan for sensitive content |

### nmem brain delete

Delete a brain.

```bash
nmem brain delete NAME [--force]
```

### nmem brain health

Check brain health.

```bash
nmem brain health [--name NAME] [--json]
```

---

## Project Commands

### nmem project create

Create a project.

```bash
nmem project create NAME [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--description` | `-d` | Project description |
| `--duration` | `-D` | Duration in days |
| `--tag` | `-t` | Tags (repeatable) |
| `--priority` | `-p` | Priority (default: 1.0) |

### nmem project list

List projects.

```bash
nmem project list [--active] [--json]
```

### nmem project show

Show project details.

```bash
nmem project show NAME [--json]
```

### nmem project delete

Delete a project.

```bash
nmem project delete NAME [--force]
```

### nmem project extend

Extend project deadline.

```bash
nmem project extend NAME DAYS [--json]
```

---

## Shared Mode Commands

### nmem shared enable

Enable shared storage mode.

```bash
nmem shared enable URL [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--api-key` | `-k` | API key for authentication |
| `--timeout` | `-t` | Request timeout in seconds |

### nmem shared disable

Disable shared mode.

```bash
nmem shared disable
```

### nmem shared status

Show shared mode status.

```bash
nmem shared status [--json]
```

### nmem shared test

Test server connection.

```bash
nmem shared test
```

### nmem shared sync

Sync with server.

```bash
nmem shared sync [--direction push|pull|both] [--json]
```

---

## Server Commands

### nmem serve

Start the FastAPI server.

```bash
nmem serve [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--host` | Host to bind (default: 127.0.0.1) |
| `--port` | Port to bind (default: 8000) |
| `--reload` | Enable auto-reload for development |

### nmem mcp

Start MCP server for Claude integration.

```bash
nmem mcp
```

### nmem prompt

Show MCP system prompt.

```bash
nmem prompt [--compact] [--json]
```

### nmem mcp-config

Show MCP configuration.

```bash
nmem mcp-config
```

### nmem install-skills

Install bundled agent skills to `~/.claude/skills/`.

```bash
nmem install-skills [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Overwrite existing skills with latest version |
| `--list` | `-l` | List available skills without installing |

**Examples:**

```bash
nmem install-skills            # Install all skills
nmem install-skills --force    # Overwrite with latest
nmem install-skills --list     # Show available skills
```

---

## Memory Types

| Type | Description | Default Expiry |
|------|-------------|----------------|
| `fact` | Objective information | Never |
| `decision` | Choices made | Never |
| `preference` | User preferences | Never |
| `todo` | Action items | 30 days |
| `insight` | Learned patterns | Never |
| `context` | Situational info | 7 days |
| `instruction` | User guidelines | Never |
| `error` | Error patterns | Never |
| `workflow` | Process patterns | Never |
| `reference` | External references | Never |

## Depth Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Instant | Direct recall (who, what, where) |
| 1 | Context | Before/after context (2-3 hops) |
| 2 | Habit | Cross-time patterns |
| 3 | Deep | Full causal/emotional analysis |
