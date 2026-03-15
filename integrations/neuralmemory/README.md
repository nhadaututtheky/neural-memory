# NeuralMemory — OpenClaw Plugin

Brain-inspired persistent memory for AI agents. Stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works.

This is the **OpenClaw plugin** for [Neural Memory](https://github.com/nhadaututheky/neural-memory).

## Prerequisites

```bash
pip install neural-memory
```

Python 3.11+ required. Verify the install:

```bash
nmem-mcp --help
```

## Install

```bash
npm install -g neuralmemory
```

Or add to your OpenClaw config directly.

## OpenClaw Setup

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "neuralmemory"
    },
    "entries": {
      "neuralmemory": {
        "config": {
          "pythonPath": "python",
          "brain": "default",
          "autoContext": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

> **Important**: Setting `slots.memory = "neuralmemory"` disables the default `memory-core` plugin. Without this, agents may still use `memory_search` instead of NeuralMemory tools.

## Tools

**v1.7.0+**: The plugin dynamically fetches **all tools** from the MCP server at startup. Whatever version of `neural-memory` you have installed, the plugin automatically exposes every tool it provides — no plugin update needed when new tools are added.

With `neural-memory>=4.6.0`, this includes **45 tools**:

| Category | Tools |
|----------|-------|
| **Core** | `nmem_remember`, `nmem_remember_batch`, `nmem_recall`, `nmem_context`, `nmem_todo`, `nmem_stats` |
| **Management** | `nmem_edit`, `nmem_forget`, `nmem_pin`, `nmem_health`, `nmem_evolution`, `nmem_alerts` |
| **Recall** | `nmem_suggest`, `nmem_narrative`, `nmem_explain`, `nmem_recap` |
| **Workflow** | `nmem_session`, `nmem_eternal`, `nmem_auto`, `nmem_habits`, `nmem_review` |
| **Cognitive** | `nmem_hypothesize`, `nmem_evidence`, `nmem_predict`, `nmem_verify`, `nmem_cognitive`, `nmem_gaps`, `nmem_schema` |
| **Training** | `nmem_train`, `nmem_train_db`, `nmem_index`, `nmem_import` |
| **Sync** | `nmem_sync`, `nmem_sync_status`, `nmem_sync_config`, `nmem_telegram_backup` |
| **Infra** | `nmem_version`, `nmem_transplant`, `nmem_conflicts` |

If the MCP server is unreachable at startup, the plugin falls back to 5 core tools (remember, recall, context, stats, health) that auto-reconnect on first use.

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pythonPath` | string | `"python"` | Python executable with `neural-memory` installed |
| `brain` | string | `"default"` | Brain name (each workspace can have its own) |
| `autoContext` | boolean | `true` | Auto-inject relevant memories before each agent run |
| `autoCapture` | boolean | `true` | Auto-extract memories after each agent run |
| `contextDepth` | integer | `1` | Recall depth: 0=instant, 1=context, 2=habits, 3=deep |
| `maxContextTokens` | integer | `500` | Max tokens for auto-context injection |
| `timeout` | integer | `30000` | MCP request timeout (ms) |

## How It Works

```
OpenClaw Agent
    |
    v
NeuralMemory Plugin (this package)
    |  Spawns + manages lifecycle
    v
nmem-mcp (Python MCP server, stdio transport)
    |
    v
~/.neuralmemory/brains/<brain>.db (SQLite)
```

The plugin spawns `nmem-mcp` as a subprocess and communicates via JSON-RPC over stdio. Memories are stored in a local SQLite database.

## Troubleshooting

**Timeout on startup**: If you see `MCP timeout: initialize (30000ms)`, the Python process is slow to start. Fix:

```bash
# Pre-install to avoid cold start delays
pip install neural-memory

# Or increase the timeout in your config
"timeout": 60000
```

**"nmem-mcp not found"**: Ensure `neural-memory` is installed in the Python environment that `pythonPath` points to.

**Schema validation errors**: Upgrade to plugin `>=1.7.0` — schemas are now normalized for strict providers (Anthropic SDK, OpenAI strict mode, Gemini). The plugin strips constraint keywords, ensures `additionalProperties: false`, and adds missing `properties` fields automatically.

## How Schema Normalization Works

The plugin normalizes MCP schemas for cross-provider compatibility:

- Strips `minimum`, `maximum`, `maxLength`, `maxItems` (rejected by some providers)
- Replaces `integer` → `number` (Gemini compatibility)
- Adds `additionalProperties: false` to all objects (OpenAI strict mode)
- Ensures every object type has a `properties` field (Anthropic SDK requirement)

This means the MCP server can use full JSON Schema features while the plugin ensures the schemas work with any LLM provider.

## Claude Code (MCP Direct)

For Claude Code users, you can skip the plugin and use MCP directly for the full toolset:

```bash
claude mcp add --scope user neural-memory -- nmem-mcp
```

## Links

- [Neural Memory on GitHub](https://github.com/nhadaututheky/neural-memory)
- [Neural Memory on PyPI](https://pypi.org/project/neural-memory/)
- [Documentation](https://nhadaututheky.github.io/neural-memory/)

## License

MIT
