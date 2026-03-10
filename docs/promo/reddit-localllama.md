# Reddit r/LocalLLaMA Post

**Title:** Open-source graph-based memory for AI coding agents — spreading activation instead of RAG (no API keys needed)

**Body:**

I've been working on an alternative approach to giving AI agents persistent memory. Instead of the usual RAG pipeline (embed → vector search → return chunks), I built a system that stores memories as a neural graph and retrieves them through **spreading activation**.

## The problem with RAG for agent memory

RAG treats memory as a search problem: "find text similar to this query." It works, but it loses structure. When you ask "why did the outage happen?", RAG returns "JWT caused the outage" — but not *why* you chose JWT, who suggested it, or what it replaced.

## Spreading activation approach

Neural Memory stores everything as typed neurons connected by typed synapses (`CAUSED_BY`, `LEADS_TO`, `SUGGESTED_BY`, `RESOLVED_BY`, etc.). Recall works by:

1. Activate seed neurons matching your query
2. Activation spreads through synapses (weighted, with decay)
3. Most-activated neurons form the response context

This gives you multi-hop reasoning for free. "Why did the outage happen?" traces: `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice ← DECIDED_AT ← Tuesday meeting`.

**No embedding API calls needed.** Core recall is pure graph traversal with O(n) complexity on the local subgraph. Embeddings are optional (supports Ollama, sentence-transformers, Gemini free tier, OpenAI).

## Technical details

- **Storage**: SQLite via aiosqlite (async), FTS5 for text search
- **Graph**: 14 neuron types, 24 synapse types, fiber bundles for episodic grouping
- **Retrieval**: Spreading activation with configurable decay, threshold, and max hops
- **Consolidation**: Memory lifecycle — decay, reinforcement, pruning of orphan nodes
- **Extraction**: Entity/keyword/temporal extraction, Vietnamese NLP support
- **MCP server**: 39 tools (incl. cognitive reasoning), stdio transport, works with Claude Code, Cursor, etc.
- **Tests**: 3,500+, 67% coverage, CI with mypy + ruff + pytest

## What makes it interesting for local LLM users

- **Zero API cost**: core recall doesn't need embeddings or LLM calls
- **Ollama integration**: if you want embeddings, use your local Ollama instance
- **Works with any MCP client**: not locked to Claude — any agent that speaks MCP
- **Light**: single SQLite file, ~23MB for 1000+ memories with full graph

## Install

```bash
pip install neural-memory

# With local embeddings via Ollama
pip install neural-memory[embeddings]
```

Config for Ollama embeddings:
```toml
# ~/.neuralmemory/config.toml
[embedding]
enabled = true
provider = "ollama"
model = "nomic-embed-text"
```

GitHub: https://github.com/nhadaututtheky/neural-memory
Architecture docs: https://nhadaututtheky.github.io/neural-memory/

The codebase is MIT licensed. Contributions welcome — especially around alternative graph backends (Neo4j, FalkorDB adapters exist but are early).
