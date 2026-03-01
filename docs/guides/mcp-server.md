# MCP Server Setup — All Editors & Clients

> **Copy-paste the config for your editor and you're done.**
> No `nmem init` needed — the server auto-initializes on first use.

---

## Table of Contents

- [Requirements](#requirements)
- [Claude Code (Plugin)](#claude-code-plugin--recommended)
- [Claude Code (Manual MCP)](#claude-code-manual-mcp)
- [Cursor](#cursor)
- [Windsurf (Codeium)](#windsurf-codeium)
- [VS Code](#vs-code)
- [Claude Desktop](#claude-desktop)
- [Cline](#cline)
- [Zed](#zed)
- [Google Antigravity](#google-antigravity)
- [JetBrains IDEs](#jetbrains-ides-intellij-pycharm-webstorm)
- [Gemini CLI](#gemini-cli)
- [Amazon Q Developer](#amazon-q-developer)
- [Neovim](#neovim)
- [Warp Terminal](#warp-terminal)
- [Custom / Other MCP Clients](#custom--other-mcp-clients)
- [Alternative: Python Module](#alternative-python-module-directly)
- [Alternative: Docker](#alternative-docker)
- [Environment Variables](#environment-variables)
- [Resource Usage](#resource-usage)
- [Available Tools](#available-tools)
- [Resources](#resources)
- [Agent Instructions](#agent-instructions)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python 3.11+**
- **pip** or **uv** package manager

```bash
# Install via pip
pip install neural-memory

# Or via uv (faster)
uv pip install neural-memory
```

> **Note:** If using `uvx` (recommended for Claude Code), you don't need to install manually — `uvx` handles it automatically.

---

## Claude Code (Plugin — Recommended)

The easiest way. One command installs everything:

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

This auto-configures the MCP server, skills, commands, agent, and hooks.

**Done.** No further setup needed.

---

## Claude Code (CLI — Recommended for Manual Setup)

The official way to add MCP servers to Claude Code:

```bash
# Global (all projects):
claude mcp add --scope user neural-memory -- nmem-mcp

# Or with uvx (no pip install needed):
claude mcp add --scope user neural-memory -- uvx --from neural-memory nmem-mcp

# Project-only:
claude mcp add neural-memory -- nmem-mcp
```

**Alternatively**, add to your project's `.mcp.json` manually:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

Or if you installed via pip (no `uvx`):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

> **Note:** Do NOT add MCP servers to `~/.claude/settings.json` or `~/.claude/mcp_servers.json` — Claude Code does not read MCP config from those files. Use `claude mcp add` or `.mcp.json`.

---

## Cursor

Add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx (no pip install needed):**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

Restart Cursor after adding the config.

---

## Windsurf (Codeium)

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

Restart Windsurf after adding.

---

## VS Code

### With Continue Extension

Add to `~/.continue/config.json` under `mcpServers`:

```json
{
  "mcpServers": [
    {
      "name": "neural-memory",
      "command": "nmem-mcp"
    }
  ]
}
```

### With Copilot Chat (MCP support)

Add to VS Code `settings.json`:

```json
{
  "mcp": {
    "servers": {
      "neural-memory": {
        "command": "nmem-mcp"
      }
    }
  }
}
```

### VS Code Extension (GUI)

For a graphical experience, install the [NeuralMemory VS Code Extension](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory) from the marketplace.

---

## Claude Desktop

Add to `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

**Windows — full path (if `nmem-mcp` not in PATH):**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"]
    }
  }
}
```

Restart Claude Desktop after adding.

---

## Cline

Add to Cline MCP settings (`cline_mcp_settings.json` in your VS Code workspace):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "disabled": false
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"],
      "disabled": false
    }
  }
}
```

---

## Zed

Add to Zed `settings.json` (`~/.config/zed/settings.json`):

```json
{
  "language_models": {
    "mcp_servers": {
      "neural-memory": {
        "command": "nmem-mcp"
      }
    }
  }
}
```

---

## Google Antigravity

Google's AI-powered editor with built-in MCP Store.

### Option 1: MCP Store (GUI)

1. Open the **MCP Store** via the `...` dropdown at the top of the editor's agent panel
2. Browse & install servers directly
3. Authenticate if prompted

### Option 2: Custom Config (for NeuralMemory)

1. Open MCP Store → click **"Manage MCP Servers"**
2. Click **"View raw config"**
3. Add NeuralMemory to `mcp_config.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

4. Save and restart the editor.

> **Tip:** Antigravity also supports connecting to NeuralMemory's FastAPI server mode. Run `nmem serve` and connect via HTTP if you prefer server-side integration.

---

## JetBrains IDEs (IntelliJ, PyCharm, WebStorm)

JetBrains IDEs support MCP via the built-in AI Assistant or the JetBrains AI plugin.

Go to **Settings → Tools → AI Assistant → MCP Servers → Add**, or edit the config file directly:

- **Location**: `.idea/mcpServers.json` (project) or global settings

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

Restart the IDE after adding.

---

## Gemini CLI

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

---

## Amazon Q Developer

Add to `~/.aws/amazonq/mcp.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

---

## Neovim

With [mcp-hub.nvim](https://github.com/ravitemer/mcphub.nvim) or similar MCP plugin, add to your `mcpservers.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

Or configure in Lua:

```lua
require("mcphub").setup({
  servers = {
    ["neural-memory"] = {
      command = "nmem-mcp",
    },
  },
})
```

---

## Warp Terminal

Add to Warp's MCP config (`~/.warp/mcp.json`):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

---

## Custom / Other MCP Clients

NeuralMemory uses **stdio transport** (JSON-RPC 2.0 over stdin/stdout). Any MCP-compatible client can connect:

```json
{
  "name": "neural-memory",
  "transport": "stdio",
  "command": "nmem-mcp"
}
```

Or with explicit Python:

```json
{
  "name": "neural-memory",
  "transport": "stdio",
  "command": "python",
  "args": ["-m", "neural_memory.mcp"]
}
```

---

## Alternative: Python Module Directly

If `nmem-mcp` is not in your PATH, use the Python module:

```json
{
  "neural-memory": {
    "command": "python",
    "args": ["-m", "neural_memory.mcp"]
  }
}
```

**macOS/Linux with specific Python:**

```json
{
  "neural-memory": {
    "command": "python3",
    "args": ["-m", "neural_memory.mcp"]
  }
}
```

**Windows with full path:**

```json
{
  "neural-memory": {
    "command": "C:\\Users\\YOU\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
    "args": ["-m", "neural_memory.mcp"]
  }
}
```

---

## Alternative: Docker

```bash
docker run -i --rm -v neuralmemory:/root/.neuralmemory ghcr.io/nhadaututtheky/neural-memory:latest nmem-mcp
```

```json
{
  "neural-memory": {
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-v", "neuralmemory:/root/.neuralmemory",
      "ghcr.io/nhadaututtheky/neural-memory:latest",
      "nmem-mcp"
    ]
  }
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURALMEMORY_BRAIN` | `"default"` | Brain name to use |
| `NEURALMEMORY_DATA_DIR` | `~/.neuralmemory` | Data directory |
| `NEURAL_MEMORY_DEBUG` | `0` | Enable debug logging (`1` to enable) |
| `MEM0_API_KEY` | — | Mem0 API key (for import) |
| `COGNEE_API_KEY` | — | Cognee API key (for import) |

**Example with custom brain:**

```json
{
  "neural-memory": {
    "command": "nmem-mcp",
    "env": {
      "NEURALMEMORY_BRAIN": "work"
    }
  }
}
```

---

## Resource Usage

| Metric | Value |
|--------|-------|
| **RAM (idle)** | ~12-15 MB |
| **RAM (active, small brain)** | ~30-35 MB |
| **RAM (active, large brain)** | ~55-60 MB |
| **CPU** | Near 0% when idle |
| **Disk** | ~1-50 MB per brain (SQLite) |
| **Startup time** | < 2 seconds |

NeuralMemory is lightweight — it won't slow down your editor.

---

## Available Tools

Once configured, these 26 tools are available to your AI assistant:

### Core Memory

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory (fact, decision, insight, todo, error, etc.) |
| `nmem_recall` | Query with spreading activation (depth 0-3) |
| `nmem_context` | Inject recent context at session start |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_auto` | Auto-capture memories from conversation text |

### Brain Management

| Tool | Description |
|------|-------------|
| `nmem_stats` | Brain statistics and freshness |
| `nmem_health` | Brain health diagnostics (purity score, grade) |
| `nmem_evolution` | Brain evolution metrics (maturation, plasticity) |
| `nmem_version` | Brain version control (snapshot, rollback, diff) |
| `nmem_transplant` | Copy memories between brains |
| `nmem_conflicts` | View and resolve memory conflicts |
| `nmem_alerts` | Brain health alerts lifecycle |

### Session & Context

| Tool | Description |
|------|-------------|
| `nmem_session` | Track working session state and progress |
| `nmem_eternal` | Save project context, decisions, instructions |
| `nmem_recap` | Load saved context at session start |

### Learning & Training

| Tool | Description |
|------|-------------|
| `nmem_index` | Index codebase for code-aware recall |
| `nmem_train` | Train brain from documentation files |
| `nmem_train_db` | Train brain from database schema |
| `nmem_habits` | Manage learned workflow habits |
| `nmem_review` | Spaced repetition review system |

### Utilities

| Tool | Description |
|------|-------------|
| `nmem_suggest` | Autocomplete suggestions from brain neurons |
| `nmem_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `nmem_narrative` | Generate timeline/topic/causal narratives |

### Sync (Multi-Device)

| Tool | Description |
|------|-------------|
| `nmem_sync` | Trigger manual sync (push/pull/full) |
| `nmem_sync_status` | Show pending changes, devices, last sync |
| `nmem_sync_config` | Configure hub URL, auto-sync, conflict strategy |

---

## Tool Tiers

By default all 26 tools are exposed on every API turn. If you want to reduce token overhead, configure a **tool tier** in `~/.neuralmemory/config.toml`:

```toml
[tool_tier]
tier = "standard"   # minimal | standard | full
```

Or via CLI:

```bash
nmem config tier --show       # show current tier
nmem config tier standard     # set to standard
nmem config tier full         # reset to full
```

| Tier | Tools | Est. Tokens | Savings |
|------|-------|-------------|---------|
| `full` (default) | 26 | ~3,800 | — |
| `standard` | 8 | ~1,400 | ~63% |
| `minimal` | 4 | ~700 | ~82% |

**Tier contents:**

- **minimal** — `remember`, `recall`, `context`, `recap`
- **standard** — minimal + `todo`, `session`, `auto`, `eternal`
- **full** — all 26 tools

> Hidden tools remain callable — only the schema listing changes. If the AI model already knows a tool name, it can still call it even when the tool is not exposed in `tools/list`.

---

## Resources

The MCP server provides resources for system prompts:

| Resource URI | Description |
|-------------|-------------|
| `neuralmemory://prompt/system` | Full system prompt for AI assistants |
| `neuralmemory://prompt/compact` | Compact version for token-limited contexts |

### Get MCP Config via CLI

```bash
nmem mcp-config
```

### View System Prompt via CLI

```bash
nmem prompt            # Full prompt
nmem prompt --compact  # Compact version
nmem prompt --json     # As JSON
```

---

## Agent Instructions

Copy these instructions into your project's `CLAUDE.md` (for Claude Code) or `.cursorrules` (for Cursor) to teach your AI assistant how to use NeuralMemory proactively.

### For Claude Code

See [`docs/agent-instructions/CLAUDE.md`](../agent-instructions/CLAUDE.md) for the full template.

### For Cursor

See [`docs/agent-instructions/.cursorrules`](../agent-instructions/.cursorrules) for the full template.

### Quick Version (any editor)

```markdown
## Memory System — NeuralMemory

This workspace uses NeuralMemory for persistent memory.
Use nmem_* MCP tools PROACTIVELY.

### Session Start (ALWAYS)
1. nmem_recap() — Resume context
2. nmem_context(limit=20, fresh_only=true) — Recent memories
3. nmem_session(action="get") — Current task

### Auto-Remember
- Decision made → nmem_remember(content="...", type="decision", priority=7)
- Bug fixed → nmem_remember(content="...", type="error", priority=7)
- TODO found → nmem_todo(task="...", priority=6)

### Auto-Recall
Before asking user → nmem_recall(query="<topic>", depth=1)

### Session End
nmem_auto(action="process", text="<session summary>")
nmem_session(action="set", feature="...", progress=0.8)
```

---

## Troubleshooting

### "nmem-mcp" not found

```bash
# Check if installed
pip show neural-memory

# Check if nmem-mcp is in PATH
which nmem-mcp    # macOS/Linux
where nmem-mcp    # Windows

# If not found, use Python module instead
python -m neural_memory.mcp
```

### Tools not appearing in editor

1. Verify the MCP config file path is correct for your editor
2. Restart the editor completely
3. Check editor logs for MCP connection errors
4. Test manually: `echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | nmem-mcp`

### Python version mismatch

```bash
# NeuralMemory requires Python 3.11+
python --version

# If you have multiple Python versions, specify the full path
```

### Windows: encoding errors

NeuralMemory handles Windows stdio encoding automatically. If you still see encoding issues:

```json
{
  "neural-memory": {
    "command": "python",
    "args": ["-m", "neural_memory.mcp"],
    "env": {
      "PYTHONIOENCODING": "utf-8"
    }
  }
}
```

### Permission denied (macOS/Linux)

```bash
chmod +x $(which nmem-mcp)
```

### uvx not found

```bash
# Install uv first
pip install uv

# Or use pipx
pipx install neural-memory
```

### Debug mode

```bash
# Run with debug logging
NEURAL_MEMORY_DEBUG=1 nmem-mcp
```

### Reset to fresh state

```bash
# macOS/Linux
rm -rf ~/.neuralmemory

# Windows
rmdir /s /q %USERPROFILE%\.neuralmemory
```

---

## Quick Reference

| Editor | Config File | Config Format |
|--------|-------------|---------------|
| **Claude Code** | `claude mcp add` or `.mcp.json` | `{ "mcpServers": { ... } }` |
| **Cursor** | `~/.cursor/mcp.json` | `{ "mcpServers": { ... } }` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` | `{ "mcpServers": { ... } }` |
| **Claude Desktop** | See [path above](#claude-desktop) | `{ "mcpServers": { ... } }` |
| **VS Code (Continue)** | `~/.continue/config.json` | `{ "mcpServers": [ ... ] }` |
| **VS Code (Copilot)** | VS Code `settings.json` | `{ "mcp": { "servers": { ... } } }` |
| **Cline** | `cline_mcp_settings.json` | `{ "mcpServers": { ... } }` |
| **Zed** | `~/.config/zed/settings.json` | `{ "language_models": { "mcp_servers": { ... } } }` |
| **Antigravity** | `mcp_config.json` (via MCP Store) | `{ "mcpServers": { ... } }` |
| **JetBrains** | `.idea/mcpServers.json` | `{ "mcpServers": { ... } }` |
| **Gemini CLI** | `~/.gemini/settings.json` | `{ "mcpServers": { ... } }` |
| **Amazon Q** | `~/.aws/amazonq/mcp.json` | `{ "mcpServers": { ... } }` |
| **Neovim** | `mcpservers.json` (plugin-dependent) | `{ "mcpServers": { ... } }` |
| **Warp** | `~/.warp/mcp.json` | `{ "mcpServers": { ... } }` |

**Minimum config for any editor:**

```json
{
  "neural-memory": {
    "command": "nmem-mcp"
  }
}
```

That's it. Copy, paste, restart. Done.
