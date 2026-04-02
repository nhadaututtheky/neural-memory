# neural-memory-mcp

Persistent memory for AI agents — 55 MCP tools, spreading activation recall, neuroscience-inspired consolidation.

## Install

```bash
pip install neural-memory
```

## Run as MCP server

```bash
npx neural-memory-mcp
```

## Configure in Claude Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp",
      "args": []
    }
  }
}
```

## Features

- **55 MCP tools** — remember, recall, consolidate, review, visualize, and more
- **Spreading activation recall** — graph traversal, not vector similarity RAG
- **10 neuroscience algorithms** — Hebbian learning, hippocampal replay, lateral inhibition
- **Tiered memory** — HOT/WARM/COLD with auto-tier promotion
- **3 storage backends** — SQLite, PostgreSQL, InfinityDB (Pro)
- **Cloud sync** — Cloudflare Workers hub with encryption
- **5800+ tests**, Python 3.11+, MIT licensed

## Links

- [Documentation](https://neuralmemory.theio.vn)
- [GitHub](https://github.com/nhadaututtheky/neural-memory)
- [PyPI](https://pypi.org/project/neural-memory/)
- [Get Pro](https://neuralmemory.theio.vn/landing/pro-landing.html)
