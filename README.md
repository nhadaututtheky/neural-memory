# NeuralMemory

[![GitHub stars](https://img.shields.io/github/stars/nhadaututtheky/neural-memory?style=social)](https://github.com/nhadaututtheky/neural-memory/stargazers)
[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![Downloads](https://img.shields.io/pypi/dm/neural-memory)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/neuralmem.neuralmemory?label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
[![OpenClaw Plugin](https://img.shields.io/npm/v/neuralmemory?label=OpenClaw)](https://www.npmjs.com/package/neuralmemory)

**Your AI agent forgets everything between sessions. Neural Memory gives it a brain.**

<p align="center">
  <img src="docs/assets/images/hero-brain.svg" alt="Neural Memory — spreading activation" width="720"/>
</p>

Memories are stored as interconnected neurons and recalled through spreading activation — the same way the human brain works. No vector database. No API calls. No monthly embedding bill.

```bash
pip install neural-memory
nmem init --full
```

Restart your AI tool. Your agent now remembers.

---

## 3 Tools. That's It.

55 MCP tools are available, but you only need three:

| Tool | What it does |
|------|-------------|
| `nmem_remember` | Store a memory — auto-detects type, tags, and connections |
| `nmem_recall` | Recall through spreading activation — related memories surface naturally |
| `nmem_health` | Brain health score (A–F) with actionable fix suggestions |

Everything else — sessions, context loading, habit tracking, maintenance — works transparently in the background.

> [All 55 MCP tools →](https://nhadaututtheky.github.io/neural-memory/api/mcp-tools/)

---

## What Makes This Different

Most memory tools are search engines. Neural Memory is a **graph that thinks**.

When you ask "Why did Tuesday's outage happen?", a vector database returns the most similar sentence. Neural Memory traces the chain:

```
outage ← CAUSED_BY ← JWT expiry ← SUGGESTED_BY ← Alice's review
```

**Relationships are explicit** — `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS` — so your agent doesn't just find memories, it *reasons* through them.

| | Search-based (RAG) | Neural Memory |
|--|---------------------|---------------|
| Retrieval | Similarity score | Graph traversal |
| Relationships | None | 24 explicit types |
| LLM required | Yes (embedding) | No — fully offline |
| Multi-hop reasoning | Multiple queries | One traversal |
| Memory lifecycle | Static | Decay, reinforcement, consolidation |
| Cost per 1K queries | ~$0.02 | **$0.00** |

---

## Cloud Sync — Your Data, Your Infrastructure

Sync your brain across every machine. Unlike other memory tools, **we never store your data**.

```
Laptop ←→ Your Cloudflare Worker ←→ Desktop
                  ↕
              Your Phone
```

You deploy the sync hub to **your own Cloudflare account** (free tier). Your D1 database, your encryption key, your data. We provide the code — you own the infrastructure.

```bash
nmem sync              # push/pull changes
nmem sync --auto       # auto-sync after every remember/recall
```

Sync uses **Merkle delta** — only diffs travel, not the full brain. Fast, efficient, private.

> [Cloud Sync setup guide →](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/)

---

## Features

#### Memory & Recall
- **14 memory types** — fact, decision, error, insight, preference, workflow, instruction, and more
- **Spreading activation** — memories surface by association, not keyword match
- **Cognitive reasoning** — hypothesize, submit evidence, make predictions, verify with Bayesian confidence

#### Knowledge Ingestion
- **Train from documents** — PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV ingested into permanent brain knowledge
- **Import adapters** — migrate from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex in one command

#### Lifecycle & Storage
- **Memory consolidation** — episodic memories mature into semantic knowledge over time
- **Compression tiers** — full → summary → essence → ghost → metadata (reclaim storage, keep meaning)
- **Brain versioning** — snapshot, rollback, diff, transplant memories between brains

#### Ecosystem
- **Web dashboard** — 7-page React UI with graph visualization, health radar, timeline, mindmap
- **VS Code extension** — memory tree, graph explorer, CodeLens, WebSocket sync ([Marketplace →](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory))
- **Safety** — Fernet encryption, sensitive content auto-detection, parameterized SQL, path validation
- **Telegram backup** — send brain `.db` files to Telegram for offsite backup

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

# Sync across devices
nmem sync --full

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

## Neural Memory Pro

Free Neural Memory is complete — 55 tools, unlimited memories, fully offline. **You never have to pay.**

But past 10K memories, things change. Keyword matching misses semantically related content. Consolidation slows to minutes. Storage grows unbounded. If your agent's brain is getting big, Pro makes it smart.

### Free recalls by keyword. Pro recalls by meaning.

```
Query: "authentication improvements"

Free (FTS5):  2 results — exact matches only
Pro  (HNSW):  7 results — includes "JWT rotation", "session hardening", "OAuth migration"
```

### What Pro adds

| | Free (SQLite) | Pro (InfinityDB) |
|--|:---:|:---:|
| **Recall** | Keyword match (FTS5) | Semantic similarity (HNSW) |
| **Speed at 1M neurons** | ~500ms | **<5ms** |
| **Scale tested** | ~50K neurons | 2M+ neurons |
| **Compression** | Text-level trimming | 5-tier vector compression (97% savings) |
| **Consolidation** | O(N²) brute-force | O(N×k) HNSW clustering |
| **Storage per 1M** | ~5 GB | **~1 GB** |
| **Cloud sync** | Manual push/pull | Merkle delta (auto, diffs only) |

### Pro-exclusive features

- **Cone Queries** — adjustable semantic recall. Narrow the cone for precision, widen for exploration
- **Smart Merge** — consolidation that scales to 1M+ neurons using HNSW neighbor clustering
- **Directional Compression** — compress along multiple semantic axes while preserving meaning
- **5-Tier Auto Lifecycle** — memories flow from float32 → float16 → int8 → binary → metadata. Auto-promote on access

### Get Pro

```bash
pip install neural-memory                 # Pro features included
nmem pro activate YOUR_LICENSE_KEY       # activate license
nmem pro status                          # verify: Pro: Active
```

**[$9/mo](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)** — 30-day money-back guarantee. All free tools keep working. Downgrade anytime, keep your data.

> [Pro quickstart →](https://nhadaututtheky.github.io/neural-memory/guides/pro-quickstart/) · [Full comparison →](https://nhadaututtheky.github.io/neural-memory/landing/pro/) · [Pricing →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)

---

## Setup by Tool

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
<summary><b>Upgrade to Pro</b></summary>

Already using Neural Memory? Just activate your key:

```bash
nmem pro activate YOUR_LICENSE_KEY    # activate license
```

Then enable InfinityDB (semantic search engine):

```toml
# ~/.neuralmemory/config.toml
storage_backend = "infinitydb"
```

Restart your MCP server. Existing memories are auto-migrated from SQLite to InfinityDB on first startup.

> [Get a license →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/) · [Pro quickstart →](https://nhadaututtheky.github.io/neural-memory/guides/pro-quickstart/)

</details>

<details>
<summary><b>Installation extras</b></summary>

```bash
pip install neural-memory[server]              # FastAPI server + dashboard
pip install neural-memory[extract]             # PDF/DOCX/PPTX/HTML/XLSX extraction
pip install neural-memory[nlp-vi]              # Vietnamese NLP
pip install neural-memory[embeddings]          # Local embedding models
pip install neural-memory[embeddings-openai]   # OpenAI embeddings
pip install neural-memory[all]                 # Everything
```

</details>

<details>
<summary><b>Benchmarks vs alternatives</b></summary>

| Metric | NeuralMemory | Mem0 | Cognee |
|--------|:---:|:---:|:---:|
| **Write 50 memories** | 1.2s | 148.2s (121x slower) | 290.6s (80x slower) |
| **Read 20 queries** | 1.8s | 2.9s | 34.6s |
| **API calls** | **0** | 70 | 149 |

Zero LLM calls, zero API cost. [Full benchmarks →](docs/benchmarks.md)

</details>

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart Guide](https://nhadaututtheky.github.io/neural-memory/guides/quickstart-guide/) | Interactive guide with animated demos |
| [Pro Quickstart](https://nhadaututtheky.github.io/neural-memory/guides/pro-quickstart/) | Get started with Pro features |
| [CLI Reference](https://nhadaututtheky.github.io/neural-memory/getting-started/cli-reference/) | All 66 CLI commands |
| [MCP Tools Reference](https://nhadaututtheky.github.io/neural-memory/api/mcp-tools/) | All 55 MCP tools with parameters |
| [Cloud Sync](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/) | Multi-device sync setup |
| [Brain Health Guide](https://nhadaututtheky.github.io/neural-memory/guides/brain-health/) | Understanding and improving brain health |
| [Embedding Setup](https://nhadaututtheky.github.io/neural-memory/guides/embedding-setup/) | Configure embedding providers |
| [Architecture](https://nhadaututtheky.github.io/neural-memory/architecture/overview/) | Technical design deep-dive |

## Development

```bash
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory && pip install -e ".[dev]"
pytest tests/ -v          # 5900+ tests
ruff check src/ tests/    # Lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

If Neural Memory helps your AI agent remember, please consider giving it a star — it helps others discover the project and keeps development going.

<a href="https://github.com/nhadaututtheky/neural-memory/stargazers">
  <img src="https://img.shields.io/github/stars/nhadaututtheky/neural-memory?style=social" alt="Star on GitHub"/>
</a>

You can also [sponsor](https://github.com/sponsors/nhadaututtheky) the project.

## License

MIT — see [LICENSE](LICENSE).
