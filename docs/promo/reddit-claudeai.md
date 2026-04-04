# Reddit r/ClaudeAI Post

**Title:** I built a persistent memory system for Claude Code that works like a brain — Neural Memory (open source, 56 MCP tools)

**Body:**

Claude Code forgets everything between sessions. I got tired of re-explaining project context every time, so I built Neural Memory — an MCP server that gives Claude a persistent, associative memory.

## How it's different from other memory tools

Most memory MCP servers use RAG (embed text → vector search → return chunks). Neural Memory doesn't. It stores memories as a **neural graph** and retrieves them through **spreading activation** — the same mechanism the human brain uses for recall.

When you remember "Alice", it doesn't just find text containing "Alice". It activates the Alice neuron, which spreads to connected concepts: the meeting where you discussed rate limiting → the outage it caused → the JWT decision that led to it. You get the full causal chain, not just keyword matches.

**No LLM/embedding API required for core recall.** It's pure algorithmic graph traversal. Embeddings are optional for cross-language search.

## What it does

- **56 MCP tools**: `nmem_remember`, `nmem_recall`, `nmem_context`, `nmem_explain`, `nmem_habits`, and more
- **Spreading activation retrieval**: memories surface through association, not search
- **Connection explainer**: ask "how are X and Y connected?" and get the exact path through the knowledge graph
- **Habits tracking**: detects recurring patterns in your workflow
- **Multi-brain**: separate memory spaces for different projects
- **Proactive auto-save**: captures memories during session + saves summary on exit
- **Local-first**: SQLite, zero external deps, fully offline

## Quick start

```bash
pip install neural-memory
```

Add to Claude Code:
```bash
/plugin marketplace add nhadaututtheky/neural-memory
```

Or configure MCP manually:
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

## Numbers

- 3,500+ tests, 68% coverage
- v2.29.0, production-stable
- 14 memory types, 24 synapse types
- Python 3.11+, async, MIT license
- Optional embeddings: Ollama, Gemini (free), OpenAI, sentence-transformers

GitHub: https://github.com/nhadaututtheky/neural-memory
Docs: https://nhadaututtheky.github.io/neural-memory/

Happy to answer questions about the architecture or how spreading activation works in practice.
