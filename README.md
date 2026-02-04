# NeuralMemory

[![CI](https://github.com/neural-memory/neural-memory/workflows/CI/badge.svg)](https://github.com/neural-memory/neural-memory/actions)
[![Coverage](https://codecov.io/gh/neural-memory/neural-memory/branch/main/graph/badge.svg)](https://codecov.io/gh/neural-memory/neural-memory)
[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![Python](https://img.shields.io/pypi/pyversions/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![License](https://img.shields.io/github/license/neural-memory/neural-memory.svg)](https://github.com/neural-memory/neural-memory/blob/main/LICENSE)

**Reflex-based memory system for AI agents** - retrieval through activation, not search.

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories are retrieved through associative recall - activating related concepts until the relevant memory emerges.

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search Engine | Human Brain |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `DISCUSSED` |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **Memory lifecycle** | Static | Decay, reinforcement, compression |

**Example: "Why did Tuesday's outage happen?"**

- **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
- **NeuralMemory**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

See [full comparison](docs/GUIDE.md#neuralmemory-vs-rag--vector-search) in the docs.

---

## The Problem

AI agents face fundamental memory limitations:

| Problem | Impact |
|---------|--------|
| **Limited context windows** | Cannot complete large projects across sessions |
| **Session amnesia** | Forget everything between conversations |
| **No knowledge sharing** | Cannot share learned patterns with other agents |
| **Context overflow** | Important early context gets lost |

## The Solution

| Feature | Benefit |
|---------|---------|
| **Persistent memory** | Survives across sessions |
| **Efficient retrieval** | Inject only relevant context, not everything |
| **Shareable brains** | Export/import patterns like Git repos |
| **Real-time sharing** | Multi-agent collaboration |
| **Project-bounded** | Optimize for active project timeframes |

## Installation

```bash
pip install neural-memory
```

With optional features:
```bash
pip install neural-memory[server]   # FastAPI server
pip install neural-memory[nlp-vi]   # Vietnamese NLP
pip install neural-memory[all]      # All features
```

## Quick Start

### CLI

```bash
# Store memories
nmem remember "Fixed auth bug with null check in login.py:42"
nmem remember "We decided to use PostgreSQL" --type decision
nmem todo "Review PR #123" --priority 7

# Query memories
nmem recall "auth bug"
nmem recall "database decision" --depth 2

# Get context for AI injection
nmem context --limit 10 --json

# Manage brains
nmem brain list
nmem brain create work
nmem brain use work

# Real-time sharing
nmem shared enable http://localhost:8000
nmem remember "Team knowledge" --shared
nmem recall "project status" --shared
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

## Features

### Memory Types
```bash
nmem remember "Objective fact" --type fact
nmem remember "We chose X" --type decision
nmem remember "User prefers Y" --type preference
nmem todo "Action item" --type todo --expires 30
nmem remember "Learned pattern" --type insight
nmem remember "Meeting notes" --type context --expires 7
```

### Project Scoping
```bash
nmem project create "Q1 Sprint" --duration 14
nmem remember "Sprint task" --project "Q1 Sprint"
nmem recall "sprint progress" --project "Q1 Sprint"
```

### Real-Time Brain Sharing
```bash
# Enable shared mode
nmem shared enable http://localhost:8000

# Per-command sharing
nmem remember "Team insight" --shared
nmem recall "shared knowledge" --shared

# Sync local with remote
nmem shared sync --direction push
```

### Safety Features
```bash
# Check for sensitive content
nmem check "API_KEY=sk-xxx"

# Auto-redact before storing
nmem remember "Config: API_KEY=sk-xxx" --redact

# Safe export
nmem brain export --exclude-sensitive -o safe.json

# Health check
nmem brain health
```

## Server Mode

```bash
pip install neural-memory[server]
uvicorn neural_memory.server:app --reload
```

API endpoints:
```
POST /memory/encode     - Store memory
POST /memory/query      - Query memories
POST /brain/create      - Create brain
GET  /brain/{id}/export - Export brain
WS   /sync/ws           - Real-time sync
```

## Documentation

- **[Complete Guide](docs/GUIDE.md)** - Full documentation with all features
- **[Integration Guide](docs/integration.md)** - AI assistant & tool integration
- **[Safety & Limitations](docs/safety.md)** - Security best practices
- **[Architecture & Scalability](docs/architecture.md)** - Technical design & future roadmap

## Development

```bash
git clone https://github.com/neural-memory/neural-memory
cd neural-memory
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v --cov=neural_memory

# Type check
mypy src/

# Lint
ruff check src/ tests/
```

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

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE).
