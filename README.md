# NeuralMemory

[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/neuralmem.neuralmemory?label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
[![OpenClaw Plugin](https://img.shields.io/npm/v/@neuralmemory/openclaw-plugin?label=OpenClaw)](https://www.npmjs.com/package/@neuralmemory/openclaw-plugin)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Reflex-based memory system for AI agents** — retrieval through activation, not search.

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories surface through associative recall — activating related concepts until the relevant memory emerges.

**28 MCP tools** · **11 memory types** · **24 synapse types** · **Schema v20** · **2860+ tests**

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search engine | Human brain |
| **LLM/Embedding** | Required (embedding API calls) | **None** — pure algorithmic graph traversal |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, etc. |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **Lifecycle** | Static | Decay, reinforcement, consolidation |
| **API Cost** | ~$0.02/1K queries | **$0.00** — fully offline |

**Example: "Why did Tuesday's outage happen?"**

- **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
- **NeuralMemory**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

---

## Installation

```bash
pip install neural-memory
```

With optional features:
```bash
pip install neural-memory[server]   # FastAPI server + dashboard
pip install neural-memory[extract]  # PDF/DOCX/PPTX/HTML/XLSX extraction
pip install neural-memory[nlp-vi]   # Vietnamese NLP
pip install neural-memory[all]      # All features
```

## Quick Setup

### Claude Code (Plugin — Recommended)

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

That's it. MCP server, skills, commands, and agent are all configured automatically via `uvx`.

### OpenClaw (Plugin)

```bash
pip install neural-memory
npm install -g @neuralmemory/openclaw-plugin
```

Then set the memory slot in `~/.openclaw/openclaw.json`:

```json
{ "plugins": { "slots": { "memory": "neuralmemory" } } }
```

Restart the gateway. See the [full setup guide](docs/guides/openclaw-plugin.md).

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
```

Then add to your editor's MCP config (Cursor: `.cursor/mcp.json`, Windsurf: `~/.codeium/windsurf/mcp_config.json`):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

The editor spawns `nmem-mcp` automatically via stdio — no manual server start needed. No `nmem init` needed — auto-initializes on first use.

## Usage

### CLI

```bash
# Store memories (type auto-detected)
nmem remember "Fixed auth bug with null check in login.py:42"
nmem remember "We decided to use PostgreSQL" --type decision
nmem todo "Review PR #123" --priority 7

# Recall memories
nmem recall "auth bug"
nmem recall "database decision" --depth 2

# Shortcuts
nmem a "quick note"           # Short for remember
nmem q "auth"                 # Short for recall
nmem last 5                   # Last 5 memories
nmem today                    # Today's memories

# Get context for AI injection
nmem context --limit 10 --json

# Brain management
nmem brain list
nmem brain create work
nmem brain use work
nmem brain health
nmem brain export -o backup.json
nmem brain import backup.json

# Codebase indexing
nmem index src/               # Index code into neural memory

# Memory lifecycle
nmem decay                    # Apply forgetting curve
nmem consolidate              # Prune, merge, summarize
nmem cleanup                  # Remove expired memories

# Visual tools
nmem serve                    # Start FastAPI server
# Then open http://localhost:8000/ui for React dashboard

# Telegram backup
nmem telegram status          # Show Telegram config status
nmem telegram test            # Send test message
nmem telegram backup          # Send brain .db to Telegram
```

### Python API

```python
import asyncio
from neural_memory import Brain
from neural_memory.storage import InMemoryStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def main():
    storage = InMemoryStorage()
    brain = Brain.create("my_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # Encode memories
    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice to discuss API design")
    await encoder.encode("Decided to use FastAPI for backend")

    # Query through activation
    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What did we decide about backend?")
    print(result.context)  # "Decided to use FastAPI for backend"

asyncio.run(main())
```

### MCP Tools (Claude Code / Cursor)

Once configured, these 28 tools are available to your AI assistant:

**Core Memory:**

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory (auto-detects type: fact, decision, insight, error, etc.) |
| `nmem_recall` | Query with spreading activation (4 depth levels: instant/context/habit/deep) |
| `nmem_context` | Get recent memories as session context |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_auto` | Auto-capture memories from conversation text |
| `nmem_suggest` | Autocomplete suggestions from brain neurons |

**Workflow:**

| Tool | Description |
|------|-------------|
| `nmem_session` | Track session state: task, feature, progress |
| `nmem_eternal` | Save project context, decisions, instructions |
| `nmem_recap` | Load saved context at session start |
| `nmem_stats` | Brain statistics and health metrics |
| `nmem_habits` | Workflow habit suggestions from usage patterns |

**Knowledge Base:**

| Tool | Description |
|------|-------------|
| `nmem_train` | Train brain from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD) |
| `nmem_train_db` | Train brain from database schema |
| `nmem_index` | Index codebase for code-aware recall |
| `nmem_pin` | Pin/unpin memories (pinned = permanent, skip decay/prune) |

**Advanced:**

| Tool | Description |
|------|-------------|
| `nmem_health` | Brain health: purity score, grade, warnings |
| `nmem_review` | Spaced repetition reviews (Leitner box system) |
| `nmem_conflicts` | Memory conflicts: list, resolve, or pre-check |
| `nmem_narrative` | Generate narratives: timeline, topic, or causal chain |
| `nmem_alerts` | Brain health alerts: list or acknowledge |
| `nmem_version` | Brain version control: snapshot, list, rollback, diff |
| `nmem_transplant` | Transplant memories between brains by tags/types |
| `nmem_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `nmem_sync` | Multi-device sync (push/pull/full) |
| `nmem_sync_status` | Sync status and pending changes |
| `nmem_sync_config` | Configure sync settings |
| `nmem_telegram_backup` | Send brain database backup to Telegram |

### VS Code Extension

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory).

- Memory tree view in the sidebar
- Interactive graph explorer with Cytoscape.js
- Encode from editor selections or comment triggers
- CodeLens memory counts on functions and classes
- Recap, eternal context, and codebase indexing commands
- Real-time WebSocket sync

## How It Works

```
Query: "What did Alice suggest?"
         │
         ▼
┌─────────────────────┐
│ 1. Decompose Query  │  → time hints, entities, intent
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Find Anchors     │  → "Alice" neuron
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Spread Activation│  → activate connected neurons
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. Find Intersection│  → high-activation subgraph
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. Extract Context  │  → "Alice suggested rate limiting"
└─────────────────────┘
```

### Key Concepts

| Concept | What it is |
|---------|------------|
| **Neuron** | A memory unit (concept, entity, action, time, state, spatial, sensory, intent) |
| **Synapse** | A weighted, typed connection between neurons (`CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, etc.) |
| **Fiber** | A memory trace — an ordered sequence of neurons forming a coherent experience |
| **Spreading activation** | Signal propagates from anchor neurons through synapses, decaying with distance |
| **Reflex pipeline** | Query → decompose → anchor → activate → intersect → extract context |
| **Decay** | Memories lose activation over time following the Ebbinghaus forgetting curve |
| **Consolidation** | Prune weak synapses, merge overlapping fibers, summarize topic clusters |

## Features

### Memory Types

```bash
nmem remember "Objective fact" --type fact
nmem remember "We chose X over Y" --type decision
nmem remember "User prefers dark mode" --type preference
nmem todo "Review the PR" --priority 7 --expires 30
nmem remember "Pattern: always validate input" --type insight
nmem remember "Meeting notes from standup" --type context --expires 7
nmem remember "Always run tests before push" --type instruction
nmem remember "Import failed: missing column" --type error
nmem remember "Deploy process: build → test → push" --type workflow
nmem remember "API docs: https://..." --type reference
```

### Knowledge Base Training

```bash
# Train from documents (permanent knowledge)
nmem_train(action="train", path="/docs/", domain_tag="project-docs")

# Supported formats: PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD, TXT, RST
# Trained memories are pinned — they never decay, prune, or compress

# Pin/unpin specific memories
nmem_pin(fiber_ids=["abc123"], pinned=True)
```

Install extraction dependencies:
```bash
pip install neural-memory[extract]
```

### Brain Health & Diagnostics

```bash
nmem_health()                       # Purity score, grade (A-F), warnings
nmem_alerts(action="list")          # Active health alerts
nmem_review(action="queue")         # Spaced repetition review queue
```

### Brain Versioning

```bash
nmem_version(action="create", name="v1-stable")  # Snapshot
nmem_version(action="list")                       # List versions
nmem_version(action="rollback", version_id="...")  # Restore
nmem_version(action="diff", from_version="...", to_version="...")
```

### Web Dashboard

```bash
nmem serve                         # Start server on localhost:8000
# Open http://localhost:8000/ui    # React dashboard (7 pages)
# Open http://localhost:8000/docs  # API docs (Swagger)
```

Pages: Overview (KPIs + brain list), Health (radar chart + warnings), Graph (placeholder for Sigma.js), Timeline, Evolution, Diagrams, Settings (brain files + Telegram backup).

### Telegram Backup

Send brain `.db` files to Telegram for offsite backup:

```bash
# Setup: set env var + config
export NMEM_TELEGRAM_BOT_TOKEN="your-bot-token"
# Add to config.toml:
# [telegram]
# enabled = true
# chat_ids = ["123456789"]

# CLI
nmem telegram status              # Check config
nmem telegram test                # Send test message
nmem telegram backup              # Send brain backup
nmem telegram backup --brain work # Specific brain

# MCP tool
nmem_telegram_backup(brain_name="work")
```

### Multi-Device Sync

```bash
nmem_sync(action="full")           # Bidirectional sync
nmem_sync_status()                 # Pending changes, devices
nmem_sync_config(action="set", hub_url="http://hub:8080")
```

### External Memory Import

Import from existing memory systems:

```bash
# ChromaDB
nmem import backup.json --source chromadb

# Via MCP tool
nmem_import(source="mem0")           # Uses MEM0_API_KEY env var
nmem_import(source="chromadb", connection="/path/to/chroma")
nmem_import(source="cognee")         # Uses COGNEE_API_KEY env var
nmem_import(source="graphiti", connection="bolt://localhost:7687")
nmem_import(source="llamaindex", connection="/path/to/index")
```

### Safety & Security

```bash
# Sensitive content detection
nmem check "API_KEY=sk-xxx"

# Auto-redact before storing
nmem remember "Config: API_KEY=sk-xxx" --redact

# Safe export (exclude sensitive neurons)
nmem brain export --exclude-sensitive -o safe.json

# Health check (freshness + sensitive scan)
nmem brain health
```

- Content length validation (100KB limit)
- ReDoS protection (text truncation before regex)
- Spreading activation queue cap (prevents memory exhaustion)
- API keys read from environment variables, never from tool parameters
- `max_tokens` clamped to 10,000

### Server Mode

```bash
pip install neural-memory[server]
nmem serve                    # localhost:8000
nmem serve -p 9000            # Custom port
nmem serve --host 0.0.0.0    # Expose to network
```

API endpoints:
```
POST /memory/encode     - Store memory
POST /memory/query      - Query memories
POST /brain/create      - Create brain
GET  /brain/{id}/export - Export brain
WS   /sync/ws           - Real-time sync
POST /hub/sync          - Multi-device incremental sync
GET  /hub/devices/{id}  - List registered devices
GET  /ui                - Web dashboard
GET  /docs              - API documentation
```

### Git Hooks

```bash
nmem hooks install          # Post-commit reminder to save commit messages
nmem hooks show             # Show installed hooks
nmem hooks uninstall        # Remove hooks
```

## Development

```bash
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory
pip install -e ".[dev]"

# Run tests (2830+ tests)
pytest tests/ -v

# Lint & format
ruff check src/ tests/
ruff format src/ tests/
```

## Documentation

- **[Complete Guide](docs/index.md)** — Full documentation
- **[Integration Guide](docs/guides/integration.md)** — AI assistant & tool integration
- **[Safety & Limitations](docs/guides/safety.md)** — Security best practices
- **[Architecture](docs/architecture/overview.md)** — Technical design

## Support

If you find NeuralMemory useful, consider supporting development:

**Solana:** `5XVY6dZDeyuZJy6Co9KeLDxY5RZ6EwCpjsUVkacMz7HF`

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
