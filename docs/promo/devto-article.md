# Dev.to Article

**Title:** Neural Memory: How Spreading Activation Gives AI Agents a Real Memory System

**Tags:** claudecode, mcp, python, ai, opensource

**Cover image:** (use dashboard-overview.png or dashboard-graph.png)

---

## The Problem

Every AI coding session starts from zero. You explain your project architecture, your conventions, your past decisions — and the AI forgets all of it when the session ends.

Most solutions reach for RAG: embed text into vectors, search by similarity, return chunks. It works for document retrieval, but it's a poor model for *memory*. When you remember something, you don't search a database — you *associate*. One thought triggers another, which triggers another, until the relevant memory surfaces.

## A Different Approach: Neural Graphs

[Neural Memory](https://github.com/nhadaututtheky/neural-memory) stores memories as a graph of typed neurons connected by typed synapses:

```
outage ← CAUSED_BY ← JWT_decision ← SUGGESTED_BY ← Alice ← DECIDED_AT ← Tuesday_meeting
```

When you ask "why did the outage happen?", it doesn't just find text containing "outage." It activates the outage neuron, and activation spreads through the graph following synapse weights. You get the full causal chain — not just the closest text match.

### RAG vs Spreading Activation

| Aspect | RAG / Vector Search | Neural Memory |
|--------|---------------------|---------------|
| Model | Search engine | Human brain |
| LLM/Embedding | Required | Optional — core recall is pure graph traversal |
| Query | "Find similar text" | "Recall through association" |
| Relationships | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY` |
| Multi-hop | Multiple queries | Natural graph traversal |
| API Cost | ~$0.02/1K queries | $0.00 — fully offline |

## How It Works

### 1. Encoding

When you tell the AI to remember something, Neural Memory:
- Extracts entities, keywords, temporal markers
- Creates typed neurons (ENTITY, CONCEPT, ACTION, TEMPORAL, etc.)
- Creates typed synapses between them (24 relationship types)
- Groups related neurons into a Fiber (episodic memory bundle)

### 2. Retrieval (Spreading Activation)

When you recall:
1. **Seed activation**: neurons matching your query get initial activation
2. **Spreading**: activation propagates through synapses, weighted by strength
3. **Decay**: activation decreases with each hop (configurable)
4. **Threshold**: only neurons above threshold are included in results
5. **Context assembly**: top-activated neurons are assembled into a coherent response

This naturally handles multi-hop queries. "Who suggested the thing that caused the outage?" follows the chain without explicit graph queries.

### 3. Consolidation

Memories have a lifecycle:
- **Decay**: unused synapses weaken over time
- **Reinforcement**: recalled memories get stronger
- **Pruning**: orphan neurons (no connections) get cleaned up
- **Merging**: duplicate information gets consolidated

## 50 MCP Tools

Neural Memory exposes 55 tools via the [Model Context Protocol](https://modelcontextprotocol.io/):

| Tool | What it does |
|------|-------------|
| `nmem_remember` | Store a memory with automatic extraction |
| `nmem_recall` | Retrieve memories through spreading activation |
| `nmem_context` | Load recent memories at session start |
| `nmem_explain` | Show WHY two concepts are connected (BFS path) |
| `nmem_habits` | Detect recurring patterns in your workflow |
| `nmem_consolidate` | Run memory lifecycle (decay, prune, merge) |
| `nmem_health` | Health diagnostics with actionable recommendations |
| `nmem_session` | Save/restore session state |

Plus 20 more for brain management, import/export, training, and diagnostics.

## Quick Start

```bash
pip install neural-memory
```

### Claude Code (Plugin)
```bash
/plugin marketplace add nhadaututtheky/neural-memory
```

### Manual MCP Config
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["neural-memory"]
    }
  }
}
```

### Optional: Cross-Language Embeddings

Core recall works without embeddings. Enable for cross-language search:

```toml
# ~/.neuralmemory/config.toml
[embedding]
enabled = true
provider = "ollama"          # or sentence_transformer, gemini, openai
model = "nomic-embed-text"
```

## Numbers

- **3,500+ tests**, 68% coverage
- **v2.29.0**, production-stable since v2.10
- **14 memory types**, 24 synapse types, schema v22
- **Python 3.11+**, async via aiosqlite
- **MIT license**
- **Dashboard**: FastAPI + React web UI for visualization

## Links

- **GitHub**: https://github.com/nhadaututtheky/neural-memory
- **Docs**: https://nhadaututtheky.github.io/neural-memory/
- **PyPI**: https://pypi.org/project/neural-memory/

---

*Neural Memory is open source and contributions are welcome. The spreading activation approach is particularly interesting if you've worked with cognitive architectures (ACT-R, Soar) — it's the same theoretical foundation applied to AI agent memory.*
