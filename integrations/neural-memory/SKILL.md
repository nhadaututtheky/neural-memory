---
name: neural-memory
description: |
  Associative memory with spreading activation for persistent, intelligent recall.
  Use PROACTIVELY when:
  (1) You need to remember facts, decisions, errors, or context across sessions
  (2) User asks "do you remember..." or references past conversations
  (3) Starting a new task — inject relevant context from memory
  (4) After making decisions or encountering errors — store for future reference
  (5) User asks "why did X happen?" — trace causal chains through memory
  Zero LLM dependency. Neural graph with Hebbian learning, memory decay, contradiction detection, and temporal reasoning.
homepage: https://github.com/nhadaututtheky/neural-memory
metadata: {"openclaw":{"emoji":"brain","primaryEnv":"NEURALMEMORY_BRAIN","requires":{"bins":["python3"],"env":["NEURALMEMORY_BRAIN"]},"os":["darwin","linux","win32"],"install":[{"id":"pip","kind":"node","package":"neural-memory","bins":["nmem"],"label":"pip install neural-memory"}]}}
---

# NeuralMemory — Associative Memory for AI Agents

A biologically-inspired memory system that uses spreading activation instead of keyword/vector search. Memories form a neural graph where neurons connect via 20 typed synapses. Frequently co-accessed memories strengthen their connections (Hebbian learning). Stale memories decay naturally. Contradictions are auto-detected.

**Why not just vector search?** Vector search finds documents similar to your query. NeuralMemory finds *conceptually related* memories through graph traversal — even when there's no keyword or embedding overlap. "What decision did we make about auth?" activates time + entity + concept neurons simultaneously and finds the intersection.

## Setup

### 1. Install NeuralMemory

```bash
pip install neural-memory
```

The brain and config at `~/.neuralmemory/` are auto-created on first use.

### 2. Install the OpenClaw Plugin (Recommended)

The plugin occupies the exclusive **memory slot** — auto-injects context before each agent run and auto-captures memories after.

```bash
# Install from npm
npm install -g @neuralmemory/openclaw-plugin
```

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "load": {
      "paths": ["<path-to-installed-plugin>"]
    },
    "entries": {
      "neuralmemory": {
        "enabled": true,
        "config": {
          "pythonPath": "python",
          "brain": "default",
          "autoContext": true,
          "autoCapture": true
        }
      }
    },
    "slots": {
      "memory": "neuralmemory"
    }
  }
}
```

**Plugin features:**
- 6 tools registered automatically (nmem_remember, nmem_recall, nmem_context, nmem_todo, nmem_stats, nmem_health)
- `before_agent_start` hook: injects tool instructions + relevant memories as context (persists across `/new`)
- `agent_end` hook: auto-extracts facts, decisions, and TODOs from the conversation
- Configurable: `contextDepth` (0-3), `maxContextTokens` (100-10000)

**After installing, build the plugin:**

```bash
cd <path-to-installed-plugin>
npm run build
```

This compiles TypeScript to JavaScript in `dist/`. The plugin entry point is `dist/index.js`.

#### Windows Installation

On Windows, use forward slashes or escaped backslashes in `openclaw.json` paths:

```json
{
  "plugins": {
    "load": {
      "paths": ["C:/Users/<you>/AppData/Roaming/npm/node_modules/@neuralmemory/openclaw-plugin"]
    }
  }
}
```

To find the installed path:

```powershell
npm list -g @neuralmemory/openclaw-plugin --parseable
```

If `openclaw plugins list` doesn't show the plugin:
1. Verify the path in `openclaw.json` points to the package root (where `package.json` is)
2. Ensure `npm run build` was run (the `dist/` folder must exist with compiled `.js` files)
3. Use `python` instead of `python3` in the plugin config (Windows default)

### Alternative: MCP Configuration (Manual)

If you prefer MCP over the plugin, add to `~/.openclaw/mcp.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"],
      "env": {
        "NEURALMEMORY_BRAIN": "default"
      }
    }
  }
}
```

On Windows, use `"python"` (not `"python3"`). This gives you all 20 MCP tools but without the auto-context/auto-capture hooks.

### 3. Verify

```bash
nmem stats
```

You should see brain statistics (neurons, synapses, fibers).

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `openclaw plugins list` doesn't show plugin | Plugin path wrong or not built | Run `npm run build`, verify path in `openclaw.json` |
| Agent runs `nmem remember` in terminal | Agent confused CLI vs tool | Plugin now auto-injects tool instructions via `systemPrompt` |
| Agent forgets tools after `/new` | No tool instructions in new session | Plugin now injects `systemPrompt` on every `before_agent_start` |
| `python3 not found` (Windows) | Windows uses `python` not `python3` | Set `pythonPath: "python"` in plugin config |
| Timeout errors | Slow machine or large brain | Increase `timeout` in plugin config (max 120000ms) |

## Tools Reference

### Core Memory Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `nmem_remember` | Store a memory | After decisions, errors, facts, insights, user preferences |
| `nmem_recall` | Query memories | Before tasks, when user references past context, "do you remember..." |
| `nmem_context` | Get recent memories | At session start, inject fresh context |
| `nmem_todo` | Quick TODO with 30-day expiry | Task tracking |

### Intelligence Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `nmem_auto` | Auto-extract memories from text | After important conversations — captures decisions, errors, TODOs automatically |
| `nmem_recall` (depth=3) | Deep associative recall | Complex questions requiring cross-domain connections |
| `nmem_habits` | Workflow pattern suggestions | When user repeats similar action sequences |

### Management Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `nmem_health` | Brain health diagnostics | Periodic checkup, before sharing brain |
| `nmem_stats` | Brain statistics | Quick overview of memory counts |
| `nmem_version` | Brain snapshots and rollback | Before risky operations, version checkpoints |
| `nmem_transplant` | Transfer memories between brains | Cross-project knowledge sharing |

## Workflow

### At Session Start
1. Call `nmem_context` to inject recent memories into your awareness
2. If user mentions a specific topic, call `nmem_recall` with that topic

### During Conversation
3. When a decision is made: `nmem_remember` with type="decision"
4. When an error occurs: `nmem_remember` with type="error"
5. When user states a preference: `nmem_remember` with type="preference"
6. When asked about past events: `nmem_recall` with appropriate depth

### At Session End
7. Call `nmem_auto` with action="process" on important conversation segments
8. This auto-extracts facts, decisions, errors, and TODOs

## Examples

### Remember a decision
```
nmem_remember(
  content="Use PostgreSQL for production, SQLite for development",
  type="decision",
  tags=["database", "infrastructure"],
  priority=8
)
```

### Recall with spreading activation
```
nmem_recall(
  query="database configuration for production",
  depth=1,
  max_tokens=500
)
```
Returns memories found via graph traversal, not keyword matching. Related memories (e.g., "deploy uses Docker with pg_dump backups") surface even without shared keywords.

### Trace causal chains
```
nmem_recall(
  query="why did the deployment fail last week?",
  depth=2
)
```
Follows CAUSED_BY and LEADS_TO synapses to trace cause-and-effect chains.

### Auto-capture from conversation
```
nmem_auto(
  action="process",
  text="We decided to switch from REST to GraphQL because the frontend needs flexible queries. The migration will take 2 sprints. TODO: update API docs."
)
```
Automatically extracts: 1 decision, 1 fact, 1 TODO.

## Key Features

- **Zero LLM dependency** — Pure algorithmic: regex, graph traversal, Hebbian learning
- **Spreading activation** — Associative recall through neural graph, not keyword/vector search
- **20 synapse types** — Temporal (BEFORE/AFTER), causal (CAUSED_BY/LEADS_TO), semantic (IS_A/HAS_PROPERTY), emotional (FELT/EVOKES), conflict (CONTRADICTS)
- **Memory lifecycle** — Short-term → Working → Episodic → Semantic with Ebbinghaus decay
- **Contradiction detection** — Auto-detects conflicting memories, deprioritizes outdated ones
- **Hebbian learning** — "Neurons that fire together wire together" — memory improves with use
- **Temporal reasoning** — Causal chain traversal, event sequences, temporal range queries
- **Brain versioning** — Snapshot, rollback, diff brain state
- **Brain transplant** — Transfer filtered knowledge between brains
- **Vietnamese + English** — Full bilingual support for extraction and sentiment

## Depth Levels

| Depth | Name | Speed | Use Case |
|-------|------|-------|----------|
| 0 | Instant | <10ms | Quick facts, recent context |
| 1 | Context | ~50ms | Standard recall (default) |
| 2 | Habit | ~200ms | Pattern matching, workflow suggestions |
| 3 | Deep | ~500ms | Cross-domain associations, causal chains |

## Notes

- Memories are stored locally in SQLite at `~/.neuralmemory/brains/<brain>.db`
- No data is sent to external services (unless optional embedding provider is configured)
- Brain isolation: each brain is independent, no cross-contamination
- `nmem_remember` returns fiber_id for reference tracking
- Priority scale: 0 (trivial) to 10 (critical), default 5
- Memory types: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference
