# Multi-Brain Setup

NeuralMemory stores each brain as a separate SQLite database. This gives you complete data isolation between agents, projects, or workspaces.

```
~/.neuralmemory/brains/
  default.db          ← shared brain (default)
  coder-agent.db      ← Agent 1
  researcher-agent.db ← Agent 2
  project-api.db      ← Project-specific
```

## Quick Setup

### Method 1: OpenClaw Plugin Config

Each OpenClaw profile can use a different brain via the `brain` field:

```json
{
  "neuralmemory": {
    "brain": "coder-agent"
  }
}
```

The brain is created automatically on first use.

### Method 2: MCP Server (Claude Code, Cursor, etc.)

Set the `NEURALMEMORY_BRAIN` environment variable in your MCP config:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"],
      "env": {
        "NEURALMEMORY_BRAIN": "my-project"
      }
    }
  }
}
```

### Method 3: CLI

```bash
# Create a new brain
nmem brain create research-brain

# Switch to it
nmem brain use research-brain

# List all brains
nmem brain list
```

## OpenClaw Multi-Profile Example

If you run multiple OpenClaw agents — each as a separate entity with its own files, memory, and keys — configure a different brain per profile.

**Profile: Coder**
```json
{
  "neuralmemory": {
    "brain": "coder",
    "autoContext": true,
    "autoCapture": true
  }
}
```

**Profile: Researcher**
```json
{
  "neuralmemory": {
    "brain": "researcher",
    "autoContext": true,
    "autoCapture": true,
    "contextDepth": 2
  }
}
```

**Profile: Security Reviewer**
```json
{
  "neuralmemory": {
    "brain": "security",
    "autoContext": true,
    "autoCapture": false
  }
}
```

Each agent gets a completely separate database file. No data leaks between brains.

## Per-Workspace MCP Config

For project-level isolation in Claude Code, create a `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"],
      "env": {
        "NEURALMEMORY_BRAIN": "work-api"
      }
    }
  }
}
```

This overrides the global config — memories stay scoped to that workspace.

## Sharing Knowledge Between Brains

Use the `nmem_transplant` tool to copy memories from one brain to another:

```
nmem_transplant(
  source_brain="researcher",
  tags=["architecture", "api-design"]
)
```

This copies matching fibers (with their neurons and synapses) into the current brain. Use it to share insights without merging entire brain histories.

Options:
- **tags** — only transplant fibers matching these tags
- **memory_types** — filter by type (`fact`, `decision`, `insight`, etc.)
- **strategy** — conflict resolution: `prefer_local`, `prefer_remote`, `prefer_recent`, `prefer_stronger`

## Concurrent Agents (Multi-Agent Isolation)

When running **multiple agents simultaneously** (e.g., 3 Claude Code sessions for 3 projects), you **must** use the `NMEM_BRAIN` environment variable. Using `nmem brain use` will cause race conditions because all agents read from the same `config.toml` file.

### The Problem

```
Agent A: nmem brain use brain-a  → config.toml: current_brain = "brain-a"
Agent B: nmem brain use brain-b  → config.toml: current_brain = "brain-b"  ← overwrites!
Agent A: nmem_recall "..."       → reads config.toml → gets brain-b data ← WRONG
```

### The Solution: Env Var Pinning

Each MCP server process is **pinned** to its brain via environment variable. It never reads `config.toml` for brain selection, so concurrent agents cannot interfere with each other.

**Project A — `.mcp.json`** (in project root):
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "env": {
        "NMEM_BRAIN": "project-alpha"
      }
    }
  }
}
```

**Project B — `.mcp.json`**:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "env": {
        "NMEM_BRAIN": "project-beta"
      }
    }
  }
}
```

**Project C — `.mcp.json`**:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "env": {
        "NMEM_BRAIN": "project-gamma"
      }
    }
  }
}
```

### Why This Works

| Factor | Explanation |
|--------|-------------|
| **Process isolation** | Each Claude Code session spawns its own MCP server as a separate OS process |
| **Env var is per-process** | `NMEM_BRAIN` is read from the process environment, not shared files |
| **No config mutation** | When env var is set, `get_shared_storage()` uses it directly without writing to `config.toml` |
| **Separate databases** | Each brain is a separate SQLite file — no lock contention on reads |

### Rules for Multi-Agent Users

1. **Always** set `NMEM_BRAIN` in `.mcp.json` for each project
2. **Never** use `nmem brain use` while agents are running — it only affects processes without env var
3. **Create brains first** via CLI: `nmem brain create project-alpha`
4. Brain names in env var are auto-created on first access if they don't exist

### Cross-Brain Knowledge Sharing

If Agent A discovers something useful for Agent B, use transplant:

```bash
# From CLI, copy architecture decisions from alpha to beta
nmem brain use project-beta
nmem brain transplant project-alpha --tag architecture --tag api-design
```

Or via MCP tool from any agent:
```
nmem_transplant(source_brain="project-alpha", tags=["architecture"])
```

## Best Practices

### When to Use Separate Brains

| Scenario | Recommendation |
|----------|---------------|
| Different agents with different roles | Separate brains |
| Different projects on the same machine | Separate brains |
| Same agent, different topics | Use tags instead |
| Security-sensitive isolation | Separate brains |
| Temporary experiments | Separate brain, delete when done |

### Naming Conventions

- **By agent role**: `coder`, `researcher`, `planner`, `security`
- **By project**: `work-api`, `side-project`, `open-source`
- **By environment**: `dev`, `staging`, `prod`

Valid characters: `a-z`, `A-Z`, `0-9`, `-`, `_`, `.` (max 64 chars).

### Maintenance

Each brain is independent. Run health checks per brain:

```bash
# Switch to a brain and check health
nmem brain use coder
nmem health
nmem stats
```

Or use the `nmem_health` / `nmem_stats` MCP tools — they always operate on the currently configured brain.
