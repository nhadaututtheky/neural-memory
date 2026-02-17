# MCP Server Guide

NeuralMemory provides a Model Context Protocol (MCP) server for seamless integration with AI assistants like Claude.

## What is MCP?

MCP (Model Context Protocol) is a standard for AI tools to communicate with language models. NeuralMemory's MCP server exposes memory operations as tools that Claude can use directly.

## Setup

### Claude Code (Plugin — Recommended)

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

This configures the MCP server (via `uvx`), skills, commands, agent, and hooks — all automatically.

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
```

Add to your editor's MCP config (e.g. `~/.cursor/mcp.json`):

```json
{
  "neural-memory": {
    "command": "nmem-mcp"
  }
}
```

No `nmem init` needed — the MCP server auto-initializes on first use.

### Restart your editor

After restarting, your AI assistant will have access to NeuralMemory tools.

## Available Tools

### nmem_remember

Store a memory in the brain.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | Content to remember |
| `memory_type` | string | No | fact, decision, todo, etc. |
| `priority` | integer | No | 0-10 priority level |
| `tags` | array | No | Tags for organization |

**Example:**
```json
{
  "content": "We decided to use PostgreSQL",
  "memory_type": "decision",
  "priority": 7,
  "tags": ["database", "architecture"]
}
```

**Response includes:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether storage succeeded |
| `fiber_id` | string | ID of the created fiber |
| `neurons_created` | integer | Number of neurons created |
| `related_memories` | array | Up to 3 related existing memories (if any found) |
| `maintenance_hint` | string | Brain maintenance suggestion (if health check triggered) |

**Related Memories** — When storing a memory, NeuralMemory automatically discovers related existing memories via 2-hop spreading activation. Each related memory includes:
- `fiber_id` — ID of the related fiber
- `preview` — First 100 characters of content
- `similarity` — Activation-based similarity score (0.0–1.0)

### nmem_recall

Query memories using spreading activation.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Query to search for |
| `depth` | integer | No | Search depth 0-3 |
| `max_tokens` | integer | No | Max tokens in response |

**Example:**
```json
{
  "query": "database decision",
  "depth": 1,
  "max_tokens": 500
}
```

### nmem_context

Get recent memories for context injection.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Number of memories (default: 10) |
| `fresh_only` | boolean | No | Only memories < 30 days |

### nmem_todo

Quick shortcut for TODO items.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | Task description |
| `priority` | integer | No | 0-10 priority (default: 5) |

### nmem_stats

Get brain statistics.

**Parameters:** None

**Returns:**
```json
{
  "brain": "default",
  "neurons": 150,
  "synapses": 280,
  "fibers": 45,
  "memory_types": {
    "fact": 20,
    "decision": 15,
    "todo": 10
  }
}
```

### nmem_auto

Auto-capture memories from conversation text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | "analyze", "save", or "process" |
| `text` | string | Yes | Text to analyze |
| `detected` | array | No | Previously detected items (for save) |

**Actions:**

- `analyze` - Analyze text for memorable content
- `save` - Save previously detected items
- `process` - Analyze and save in one call

### nmem_alerts

View and manage brain health alerts.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | "list" or "acknowledge" |
| `alert_id` | string | No | Alert ID (required for acknowledge) |
| `limit` | integer | No | Max alerts to list (default: 50) |

**Actions:**

- `list` — View active and seen alerts, sorted by severity
- `acknowledge` — Mark an alert as handled (prevents auto-resolution)

**Example — List:**
```json
{
  "action": "list",
  "limit": 10
}
```

**Response:**
```json
{
  "alerts": [
    {
      "id": "a1b2c3d4",
      "alert_type": "high_neuron_count",
      "severity": "medium",
      "message": "High neuron count (5000). Consider running consolidation.",
      "recommended_action": "prune",
      "status": "active",
      "created_at": "2026-02-18T10:30:00"
    }
  ],
  "total": 1
}
```

**Alert Types:** `high_neuron_count`, `high_fiber_count`, `high_synapse_count`, `low_connectivity`, `high_orphan_ratio`, `expired_memories`, `stale_fibers`

**Alert Lifecycle:**

```
active → seen → acknowledged → resolved
  │         │         │              │
  │  (auto on    (manual via    (auto when
  │   next tool   acknowledge)   condition
  │   call)                      clears)
  │
  └── 6h dedup cooldown (same type suppressed)
```

Alerts are created automatically from health checks and surfaced as `pending_alerts` count in `nmem_remember`, `nmem_recall`, and `nmem_context` responses.

### nmem_train_db

Train a brain from database schema knowledge.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | No | "train" (default) or "status" |
| `connection_string` | string | Yes (train) | SQLite connection string (e.g., `sqlite:///path/to/db`) |
| `domain_tag` | string | No | Tag applied to all schema knowledge (e.g., "ecommerce") |
| `brain_name` | string | No | Target brain name (empty = current brain) |
| `consolidate` | boolean | No | Run ENRICH consolidation after training (default: true) |
| `max_tables` | integer | No | Maximum tables to process, 1-500 (default: 100) |

**Example — Train:**
```json
{
  "action": "train",
  "connection_string": "sqlite:///data/ecommerce.db",
  "domain_tag": "ecommerce",
  "max_tables": 50
}
```

**Example — Status:**
```json
{
  "action": "status"
}
```

**Response (train):**
```json
{
  "tables_processed": 12,
  "relationships_mapped": 8,
  "patterns_detected": 5,
  "neurons_created": 45,
  "synapses_created": 32,
  "schema_fingerprint": "a1b2c3d4e5f6g7h8",
  "message": "Trained schema: 12 tables, 8 relationships, 5 patterns detected"
}
```

## Resources

The MCP server also provides resources for system prompts:

### neuralmemory://prompt/system

Full system prompt explaining how to use NeuralMemory tools.

### neuralmemory://prompt/compact

Compact version for token-limited contexts.

## System Prompt

Claude receives guidance on when to use NeuralMemory:

```markdown
# When to REMEMBER

- Important decisions and their rationale
- Error solutions that might help later
- User preferences and patterns
- Architectural choices
- Meeting outcomes and action items

# When to RECALL

- Before making similar decisions
- When debugging familiar errors
- To check user preferences
- To understand project history

# Auto-Capture

After important conversations, use nmem_auto to automatically
capture decisions, errors, and insights.
```

## Configuration

### Get MCP Config

```bash
nmem mcp-config
```

Output:
```json
{
  "neural-memory": {
    "command": "nmem-mcp"
  }
}
```

### View System Prompt

```bash
# Full prompt
nmem prompt

# Compact version
nmem prompt --compact

# As JSON
nmem prompt --json
```

## Usage Examples

### Claude Remembering a Decision

```
User: Let's use Redis for caching instead of Memcached

Claude: [uses nmem_remember]
I've stored this decision:
- Content: "Using Redis for caching (chosen over Memcached)"
- Type: decision
- Tags: caching, infrastructure
```

### Claude Recalling Information

```
User: What caching solution did we pick?

Claude: [uses nmem_recall]
Based on my memory, you decided to use Redis for caching.
This was chosen over Memcached.
```

### Claude Auto-Capturing

```
User: We spent 3 hours debugging the auth issue. Turns out
      the JWT secret wasn't being loaded from env vars properly.

Claude: [uses nmem_auto with action="process"]
I've captured this for future reference:
- Error: JWT secret not loading from environment variables
- Solution: Ensure proper env var loading
- Type: error
```

## Troubleshooting

### Tools Not Appearing

1. Verify installation: `nmem --version`
2. Test MCP server: `nmem-mcp` (should wait for input)
3. Check Claude Code logs for errors
4. Restart Claude Code

### Connection Issues

1. Check Python path: `which python`
2. Verify nmem-mcp is in PATH: `which nmem-mcp`
3. Try Python module directly: `python -m neural_memory.mcp`

### Tool Errors

Enable debug logging:

```bash
NEURAL_MEMORY_DEBUG=1 nmem-mcp
```

Check the MCP communication:

```bash
# Test tool manually
echo '{"method":"tools/list"}' | nmem-mcp
```

## Best Practices

### 1. Let Claude Use Auto-Capture

After complex conversations, remind Claude:

```
"Can you capture any important decisions or errors from our discussion?"
```

### 2. Ask Claude to Check Memory

Before making decisions:

```
"Before we proceed, check if we've discussed this before."
```

### 3. Review with Stats

Periodically:

```
"Show me the memory statistics for this project."
```

### 4. Use Typed Memories

Encourage Claude to use appropriate types:

- `decision` for choices
- `error` for bug fixes
- `fact` for configurations
- `todo` for action items
