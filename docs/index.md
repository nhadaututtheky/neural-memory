# NeuralMemory

<p align="center">
  <strong>Reflex-based memory system for AI agents</strong><br>
  <em>Retrieval through activation, not search</em>
</p>

<p align="center">
  <a href="https://buymeacoffee.com/vietnamit"><img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

<p align="center">
  <a href="https://github.com/nhadaututtheky/neural-memory/actions"><img src="https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/neural-memory/"><img src="https://img.shields.io/pypi/v/neural-memory.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## What is NeuralMemory?

NeuralMemory stores experiences as interconnected neurons and recalls them through **spreading activation** - mimicking how the human brain works. Instead of searching a database, memories are retrieved through associative recall.

```bash
# Store a memory
nmem remember "Fixed auth bug with null check in login.py:42"

# Recall through association
nmem recall "auth bug fix"
# → "Fixed auth bug with null check in login.py:42"
```

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search Engine | Human Brain |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO` |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |

!!! example "Example: Causal Query"
    **Query:** "Why did Tuesday's outage happen?"

    - **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
    - **NeuralMemory**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

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

## Quick Start

### Installation

```bash
pip install neural-memory
```

With optional features:

```bash
pip install neural-memory[server]   # FastAPI server + Web UI
pip install neural-memory[nlp-vi]   # Vietnamese NLP
pip install neural-memory[all]      # All features
```

### Basic Usage

=== "CLI"

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
    ```

=== "Python"

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

        # Query through activation
        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query("What did we discuss?")
        print(result.context)

    asyncio.run(main())
    ```

=== "MCP Server"

    ```json
    // ~/.claude/mcp_servers.json
    {
      "neural-memory": {
        "command": "nmem-mcp"
      }
    }
    ```

    Claude will have access to:

    - `nmem_remember` - Store memories
    - `nmem_recall` - Query memories
    - `nmem_context` - Get recent context
    - `nmem_todo` - Quick TODO
    - `nmem_stats` - Brain statistics
    - `nmem_auto` - Auto-capture memories
    - `nmem_train_db` - Train brain from database schema
    - `nmem_alerts` - View and manage brain health alerts

## VS Code Extension

Install the NeuralMemory extension for a visual brain explorer directly in your editor:

- **Memory Tree View** — Browse neurons grouped by type in the activity bar
- **Graph Explorer** — Interactive Cytoscape.js force-directed graph
- **CodeLens** — Memory counts on functions/classes, comment trigger detection
- **Encode & Recall** — Store and query memories from the command palette
- **Real-time Sync** — WebSocket updates for tree, graph, and status bar

```bash
cd vscode-extension && npm run build
# Install from .vsix or use Extension Developer Host
```

## Web UI Visualization

Start the server and access the interactive brain visualization:

```bash
pip install neural-memory[server]
nmem serve
# Open http://localhost:8000/ui
```

## Features

- **Reflex Activation** - Trail-based retrieval through fiber pathways with conductivity (v0.6.0+)
- **Co-Activation** - Hebbian binding detects neurons activated by multiple sources (v0.6.0+)
- **Time-First Anchoring** - Time neurons as primary anchors for temporally-aware recall (v0.6.0+)
- **Spreading Activation** - Neural graph-based retrieval (classic mode)
- **Multi-language** - English + Vietnamese support
- **Typed Memories** - fact, decision, todo, insight, etc.
- **Priority System** - 0-10 priority levels
- **Expiry/TTL** - Auto-expire temporary memories
- **Project Scoping** - Organize memories by project
- **Sensitive Content Detection** - Auto-detect secrets, PII
- **Memory Decay** - Ebbinghaus forgetting curve
- **Brain Sharing** - Export, import, merge brains
- **DB-to-Brain Training** - Teach brains to understand database schemas (v1.6.0+)
- **AI Agent Skills** - Composable memory-intake, memory-audit, memory-evolution workflows (v1.6.0+)
- **Smart Context Optimizer** - 5-factor composite scoring + SimHash dedup + token budgeting (v2.6.0+)
- **Proactive Alerts** - Persistent brain health alerts with lifecycle management (v2.6.0+)
- **Recall Pattern Learning** - Topic co-occurrence mining + follow-up suggestions (v2.6.0+)

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install NeuralMemory and get started in minutes

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Learn the basics with a hands-on tutorial

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-brain:{ .lg .middle } **Concepts**

    ---

    Understand how NeuralMemory works

    [:octicons-arrow-right-24: Concepts](concepts/how-it-works.md)

-   :material-connection:{ .lg .middle } **Integration**

    ---

    Integrate with Claude, Cursor, and other tools

    [:octicons-arrow-right-24: Integration](guides/integration.md)

</div>
