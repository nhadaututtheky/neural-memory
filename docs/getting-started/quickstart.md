# Quick Start

This guide walks you through basic NeuralMemory usage in 5 minutes.

!!! tip "3 tools you need"
    NeuralMemory has 55 tools, but you only need three: **`nmem_remember`**, **`nmem_recall`**, and **`nmem_health`**. The agent handles the other 52 automatically. See [all tools](../guides/mcp-server.md#available-tools).

## 0. Setup

### Claude Code (Plugin)

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

### OpenClaw (Plugin)

```bash
pip install neural-memory
npm install -g neuralmemory
```

Then in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "neuralmemory"
    }
  }
}
```

Restart the gateway. The plugin auto-registers 6 tools (`nmem_remember`, `nmem_recall`, `nmem_context`, `nmem_todo`, `nmem_stats`, `nmem_health`) and injects memory context before each agent run. See the [full setup guide](../guides/openclaw-plugin.md).

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
```

Then add `nmem-mcp` to your editor's MCP config. No `nmem init` needed — the MCP server auto-initializes on first use.

### VS Code Extension

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory) for a visual interface — sidebar memory tree, interactive graph explorer, CodeLens on functions, and keyboard shortcuts for encode/recall.

### Optional: Explicit Init

```bash
nmem init    # Only needed if you want to pre-create config and brain
```

## 1. Store Your First Memory

```bash
nmem remember "Fixed auth bug with null check in login.py:42"
```

Output:
```
Stored memory with 4 neurons and 3 synapses
```

## 2. Query Memories

```bash
nmem recall "auth bug"
```

Output:
```
Fixed auth bug with null check in login.py:42
(confidence: 0.85, neurons activated: 4)
```

## 3. Use Memory Types

Different types help organize and retrieve memories:

```bash
# Decisions (never expire)
nmem remember "We decided to use PostgreSQL" --type decision

# TODOs (expire in 30 days)
nmem todo "Review PR #123" --priority 7

# Facts
nmem remember "API endpoint is /v2/users" --type fact

# Errors with solutions
nmem remember "ERROR: null pointer in auth. SOLUTION: add null check" --type error
```

## 4. Get Context

Retrieve recent memories for AI context injection:

```bash
nmem context --limit 5
```

With JSON output for programmatic use:

```bash
nmem context --limit 5 --json
```

## 5. View Statistics

```bash
nmem stats
```

Output:
```
Brain: default
Neurons: 12
Synapses: 18
Fibers: 4

Memory Types:
  fact: 2
  decision: 1
  todo: 1
```

## 6. Manage Brains

Create separate brains for different projects:

```bash
# List brains
nmem brain list

# Create new brain
nmem brain create work

# Switch to brain
nmem brain use work

# Export brain
nmem brain export -o backup.json
```

## 7. Web Visualization

Start the server to visualize your brain:

```bash
pip install neural-memory[server]
nmem serve
```

Open http://localhost:8000/ui to see:

- Interactive neural graph
- Color-coded neuron types
- Click nodes for details

## Example Workflow

Here's a typical workflow during a coding session:

```bash
# Start of session - get context
nmem context --limit 10

# During work - remember important things
nmem remember "UserService now uses async/await"
nmem remember "DECISION: Use JWT for auth. REASON: Stateless" --type decision
nmem todo "Add rate limiting to API" --priority 8

# When you need to recall
nmem recall "auth decision"
nmem recall "UserService changes"

# End of session - check what's pending
nmem list --type todo
```

## Next Steps

- [CLI Reference](cli.md) — All commands and options
- [Memory Types](../concepts/memory-types.md) — Understanding different memory types
- [Integration Guide](../guides/integration.md) — Integrate with Claude Code, Cursor, and other editors
- [OpenClaw Plugin Guide](../guides/openclaw-plugin.md) — Full setup for OpenClaw agents
