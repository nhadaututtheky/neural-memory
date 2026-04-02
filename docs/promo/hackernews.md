# Hacker News — Show HN

**Title:** Show HN: Neural Memory – Graph-based persistent memory for AI agents (spreading activation, not RAG)

**URL:** https://github.com/nhadaututtheky/neural-memory

**Text (optional, for text post instead of URL post):**

Neural Memory is an open-source MCP server that gives AI coding agents persistent memory using a neural graph with spreading activation retrieval — instead of the usual RAG/vector search approach.

Core idea: memories are stored as typed neurons connected by typed synapses (CAUSED_BY, LEADS_TO, RESOLVED_BY, etc.). Recall works by activating seed neurons and letting activation spread through the graph, naturally surfacing related memories through association.

No embedding API calls needed for core recall — it's pure algorithmic graph traversal. Embeddings are optional for cross-language search (supports Ollama, sentence-transformers, Gemini, OpenAI).

55 MCP tools (incl. cognitive reasoning layer), 5800+ tests, SQLite storage, Python 3.11+, MIT license.
