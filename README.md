# NeuralMemory

[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/neuralmem.neuralmemory?label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Reflex-based memory system for AI agents** — retrieval through activation, not search.

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories surface through associative recall — activating related concepts until the relevant memory emerges.

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search engine | Human brain |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `DISCUSSED` |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **Lifecycle** | Static | Decay, reinforcement, consolidation |

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

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
```

Then add to your editor's MCP config:

```json
{
  "neural-memory": {
    "command": "nmem-mcp"
  }
}
```

No `nmem init` needed — the MCP server auto-initializes on first use.

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
nmem dashboard                # Rich terminal dashboard
nmem ui                       # Interactive memory browser
nmem graph "auth"             # Visualize neural connections
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

Once configured, these tools are available to your AI assistant:

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory (fact, decision, insight, todo, etc.) |
| `nmem_recall` | Query with spreading activation (auto depth detection) |
| `nmem_context` | Inject recent context at session start |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_stats` | Brain statistics and freshness |
| `nmem_auto` | Auto-capture memories from conversation text |
| `nmem_suggest` | Autocomplete suggestions from brain neurons |
| `nmem_session` | Track working session state and progress |
| `nmem_index` | Index codebase for code-aware recall |
| `nmem_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `nmem_eternal` | Save project context, decisions, instructions |
| `nmem_recap` | Load saved context at session start |

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
| **Synapse** | A weighted, typed connection between neurons (`CAUSED_BY`, `LEADS_TO`, `CO_OCCURRED`, etc.) |
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

# Run tests (584 tests)
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

If you find NeuralMemory useful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/vietnamit)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
