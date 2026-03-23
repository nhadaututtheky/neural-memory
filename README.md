# NeuralMemory

[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/neuralmem.neuralmemory?label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
[![OpenClaw Plugin](https://img.shields.io/npm/v/neuralmemory?label=OpenClaw)](https://www.npmjs.com/package/neuralmemory)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Reflex-based memory system for AI agents** — retrieval through activation, not search.

<p align="center">
  <img src="docs/assets/images/hero-brain.svg" alt="Neural Memory — spreading activation visualization" width="720"/>
</p>

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation — like the human brain. Instead of searching a database, memories surface through association.

**50 MCP tools** · **14 memory types** · **24 synapse types** · **4700+ tests** · **Cognitive reasoning layer**

---

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search engine | Human brain |
| **LLM/Embedding** | Required (API calls) | **Optional** — core is pure graph traversal |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY` |
| **Multi-hop** | Multiple queries | Natural graph traversal |
| **Lifecycle** | Static | Decay, reinforcement, consolidation |
| **API Cost** | ~$0.02/1K queries | **$0.00** — fully offline |

**Example: "Why did Tuesday's outage happen?"**
- **RAG**: Returns "JWT caused outage" (missing *why*)
- **NeuralMemory**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

### Benchmarks

| Metric | NeuralMemory | Mem0 | Cognee |
|--------|:---:|:---:|:---:|
| **Write 50 memories** | 1.2s | 148.2s (121x slower) | 290.6s (80x slower) |
| **Read 20 queries** | 1.8s | 2.9s | 34.6s |
| **API calls** | **0** | 70 | 149 |

> Zero LLM calls, zero API cost. [Full benchmarks →](docs/benchmarks.md)

---

## Get Started in 60 Seconds

```bash
pip install neural-memory
nmem init --full
```

One command sets up everything: config, brain, MCP server, hooks, embeddings, dedup, and maintenance.

Restart your AI tool — your brain is live.

> **New?** See the [Interactive Quickstart Guide](https://nhadaututtheky.github.io/neural-memory/guides/quickstart-guide/) with animated demos.

```bash
nmem doctor        # 11 diagnostic checks
nmem doctor --fix  # Auto-fix issues
```

### Setup by Tool

<details>
<summary><b>Claude Code (Plugin)</b></summary>

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

</details>

<details>
<summary><b>Cursor / Windsurf / Other MCP Clients</b></summary>

```bash
pip install neural-memory
```

Add to your editor's MCP config:

```json
{
  "mcpServers": {
    "neural-memory": { "command": "nmem-mcp" }
  }
}
```

</details>

<details>
<summary><b>OpenClaw (Plugin)</b></summary>

```bash
pip install neural-memory && npm install -g neuralmemory
```

Set memory slot in `~/.openclaw/openclaw.json`:
```json
{ "plugins": { "slots": { "memory": "neuralmemory" } } }
```

</details>

<details>
<summary><b>Installation extras</b></summary>

```bash
pip install neural-memory[server]       # FastAPI server + dashboard
pip install neural-memory[extract]      # PDF/DOCX/PPTX/HTML/XLSX extraction
pip install neural-memory[nlp-vi]       # Vietnamese NLP
pip install neural-memory[embeddings]   # Local + OpenAI/OpenRouter embedding support
pip install neural-memory[all]          # Everything
```

</details>

---

## 3 Tools You Need

Once configured, **50 tools** are available — but you only need three:

| Tool | What it does |
|------|-------------|
| `nmem_remember` | Store a memory — auto-detects type, tags, and connections |
| `nmem_recall` | Recall through spreading activation — related memories surface naturally |
| `nmem_health` | Brain health score (A-F) with actionable fix suggestions |

Everything else works transparently: sessions, context loading, habit tracking, maintenance.

> [All 50 tools →](https://nhadaututtheky.github.io/neural-memory/api/mcp-tools/)

---

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

---

## Quick Examples

```bash
# Store memories (type auto-detected)
nmem remember "Fixed auth bug with null check in login.py:42"
nmem remember "We decided to use PostgreSQL" --type decision
nmem todo "Review PR #123" --priority 7

# Recall
nmem recall "auth bug"
nmem recall "database decision" --depth 2

# Brain management
nmem brain list && nmem brain health
nmem brain export -o backup.json

# Web dashboard
nmem serve    # http://localhost:8000/dashboard
```

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

    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice to discuss API design")
    await encoder.encode("Decided to use FastAPI for backend")

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What did we decide about backend?")
    print(result.context)  # "Decided to use FastAPI for backend"

asyncio.run(main())
```

---

## Features

- **14 memory types** — fact, decision, preference, todo, insight, context, instruction, error, workflow, reference...
- **Knowledge base training** — Ingest PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV into permanent brain knowledge
- **Cognitive reasoning** — Hypothesize, submit evidence, make predictions, verify outcomes with Bayesian confidence
- **Brain versioning** — Snapshot, rollback, diff, transplant memories between brains
- **Cloud sync** — Multi-device sync via your own Cloudflare Worker (free tier). [Setup guide →](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/)
- **Web dashboard** — 7-page React dashboard with graph visualization, health radar, timeline, mindmap
- **VS Code extension** — Memory tree, graph explorer, CodeLens, WebSocket sync. [Marketplace →](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
- **PostgreSQL backend** — Optional pgvector backend for large-scale deployments
- **Telegram backup** — Send brain `.db` files to Telegram for offsite backup
- **Safety** — Fernet encryption, sensitive content auto-detection, parameterized SQL, path validation
- **Import adapters** — Migrate from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart Guide](https://nhadaututtheky.github.io/neural-memory/guides/quickstart-guide/) | Interactive guide with animated demos |
| [CLI Reference](https://nhadaututtheky.github.io/neural-memory/getting-started/cli-reference/) | All 66 CLI commands |
| [MCP Tools Reference](https://nhadaututtheky.github.io/neural-memory/api/mcp-tools/) | All 50 MCP tools with parameters |
| [Brain Health Guide](https://nhadaututtheky.github.io/neural-memory/guides/brain-health/) | Understanding and improving brain health |
| [Embedding Setup](https://nhadaututtheky.github.io/neural-memory/guides/embedding-setup/) | Configure embedding providers |
| [Cloud Sync](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/) | Multi-device sync setup |
| [Architecture](https://nhadaututtheky.github.io/neural-memory/architecture/overview/) | Technical design deep-dive |

## Development

```bash
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory
pip install -e ".[dev]"
pytest tests/ -v          # 4700+ tests
ruff check src/ tests/    # Lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Neural Memory Pro

Need more power? **[Neural Memory Pro](https://theio.vn)** replaces SQLite with InfinityDB — a purpose-built spatial database engine for neural graphs.

| | Free (SQLite) | Pro (InfinityDB) |
|--|---------------|-------------------|
| **Vector search** | Sequential scan | HNSW index, <5ms at 1M neurons |
| **Scale** | ~50K neurons | 2M+ tested |
| **Compression** | None | 5-tier adaptive (up to 89%) |
| **Graph traversal** | SQL JOINs | Native BFS, <1ms depth-3 |
| **MCP tools** | 52 | 52 + 3 Pro-exclusive |

```bash
pip install git+https://github.com/AIVN-Foundation/neural-memory-pro.git
```

One command. Auto-registers, auto-upgrades storage. All free tools keep working.

## Support

If NeuralMemory is useful to you, consider [sponsoring](https://github.com/sponsors/nhadaututtheky) or starring the repo.

## License

MIT — see [LICENSE](LICENSE).
