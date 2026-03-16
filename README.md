# NeuralMemory

[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![CI](https://github.com/nhadaututtheky/neural-memory/workflows/CI/badge.svg)](https://github.com/nhadaututtheky/neural-memory/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/neuralmem.neuralmemory?label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
[![OpenClaw Plugin](https://img.shields.io/npm/v/neuralmemory?label=OpenClaw)](https://www.npmjs.com/package/neuralmemory)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Reflex-based memory system for AI agents** — retrieval through activation, not search.

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories surface through associative recall — activating related concepts until the relevant memory emerges.

**45 MCP tools** · **14 memory types** · **24 synapse types** · **Schema v27** · **3976+ tests** · **Cognitive reasoning layer**

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Model** | Search engine | Human brain |
| **LLM/Embedding** | Required (embedding API calls) | **Optional** — core recall is pure algorithmic graph traversal |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, etc. |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **Lifecycle** | Static | Decay, reinforcement, consolidation |
| **API Cost** | ~$0.02/1K queries | **$0.00** — fully offline (optional embeddings available) |

**Example: "Why did Tuesday's outage happen?"**

- **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
- **NeuralMemory**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

### Benchmarks

Real-world benchmarks on 50 AI agent memories (decisions, errors, workflows, preferences):

**NeuralMemory vs Mem0 v1.0.3**

| Metric | NeuralMemory | Mem0 | |
|--------|:---:|:---:|---|
| **Write 50 memories** | 1.2s | 148.2s | **121x faster** |
| **Read 20 queries** | 1.8s | 2.9s | **1.6x faster** |
| **Multi-hop reasoning** | 0.42 | 0.38 | NM wins |
| **Accuracy** | 0.14 | 0.14 | Tie |
| **API calls** | **0** | **70** | **$0 cost** |

**NeuralMemory vs Cognee v0.5.5**

| Metric | NeuralMemory | Cognee | |
|--------|:---:|:---:|---|
| **Write 50 memories** | 3.6s | 290.6s | **80x faster** |
| **Read 20 queries** | 1.9s | 34.6s | **18x faster** |
| **Multi-hop reasoning** | 0.42 | 0.63 | Cognee wins |
| **Accuracy** | 0.14 | 0.18 | Cognee wins |
| **API calls** | **0** | **149** | **$0 cost** |

> NeuralMemory: zero LLM calls, zero API cost, up to 121x faster writes. Cognee trades speed for LLM-powered semantic richness. [Full benchmarks →](docs/benchmarks.md)

---

## Installation

```bash
pip install neural-memory
```

With optional features:
```bash
pip install neural-memory[server]       # FastAPI server + dashboard
pip install neural-memory[extract]      # PDF/DOCX/PPTX/HTML/XLSX extraction
pip install neural-memory[nlp-vi]       # Vietnamese NLP
pip install neural-memory[embeddings]   # Local embedding (cross-language recall)
pip install neural-memory[all]          # All features
```

### Optional: Embedding for Cross-Language Recall

Core recall works without embeddings. Enable embeddings to recall memories across languages (e.g., search in Vietnamese, find English memories):

```toml
# ~/.neuralmemory/config.toml
[embedding]
enabled = true
provider = "auto"    # Auto-detects: Ollama → sentence-transformers → Gemini → OpenAI
```

Or pick a specific provider: **sentence_transformer** (free/local), **ollama** (local via Ollama API), **gemini** (Google free tier), **openai** (paid). See the [Embedding Setup Guide](docs/guides/embedding-setup.md) for details.

## Quick Setup

### Claude Code (Plugin — Recommended)

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

That's it. MCP server, skills, commands, and agent are all configured automatically via `uvx`.

### OpenClaw (Plugin)

```bash
pip install neural-memory
npm install -g neuralmemory
```

Then set the memory slot in `~/.openclaw/openclaw.json`:

```json
{ "plugins": { "slots": { "memory": "neuralmemory" } } }
```

Restart the gateway. See the [full setup guide](docs/guides/openclaw-plugin.md).

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
```

Then add to your editor's MCP config (Cursor: `.cursor/mcp.json`, Windsurf: `~/.codeium/windsurf/mcp_config.json`):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "nmem-mcp"
    }
  }
}
```

The editor spawns `nmem-mcp` automatically via stdio — no manual server start needed. No `nmem init` needed — auto-initializes on first use.

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
nmem serve                    # Start FastAPI server
# Then open http://localhost:8000/dashboard

# Telegram backup
nmem telegram status          # Show Telegram config status
nmem telegram test            # Send test message
nmem telegram backup          # Send brain .db to Telegram
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

**3 tools you need. 42 the agent handles automatically.**

Once configured, 45 tools are available — but you only need to know three:

| Tool | What You Do |
|------|-------------|
| `nmem_remember` | Tell the agent to remember something — it auto-detects type, tags, and connections |
| `nmem_recall` | Ask the agent to recall — spreading activation surfaces related memories |
| `nmem_health` | Check brain health — purity score, grade (A-F), actionable fix suggestions |

Everything else works transparently. The agent auto-manages sessions, context loading, habit tracking, and memory maintenance through the remaining 42 tools.

<details>
<summary><b>All 45 tools</b> (click to expand)</summary>

**Core Memory** — Remember, recall, edit, forget

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory (auto-detects type: fact, decision, insight, error, etc.) |
| `nmem_recall` | Query with spreading activation (4 depth levels: instant → deep) |
| `nmem_context` | Get recent memories as session context |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_auto` | Auto-capture memories from conversation text |
| `nmem_suggest` | Autocomplete suggestions from brain neurons |
| `nmem_edit` | Edit memory type, content, or priority (preserves connections) |
| `nmem_forget` | Soft delete (set expiry) or hard delete (permanent removal) |

**Agent Workflow** — Session management, habits, context loading

| Tool | Description |
|------|-------------|
| `nmem_session` | Track session state: task, feature, progress |
| `nmem_eternal` | Save project context, decisions, instructions |
| `nmem_recap` | Load saved context at session start |
| `nmem_stats` | Brain statistics and health metrics |
| `nmem_habits` | Workflow habit suggestions from usage patterns |
| `nmem_tool_stats` | Tool usage analytics: summary + daily trends |

**Knowledge Base** — Train from docs, index codebases

| Tool | Description |
|------|-------------|
| `nmem_train` | Train brain from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD) |
| `nmem_train_db` | Train brain from database schema |
| `nmem_index` | Index codebase for code-aware recall |
| `nmem_pin` | Pin/unpin memories (pinned = permanent, skip decay/prune) |

**Brain Management** — Health, maintenance, version control

| Tool | Description |
|------|-------------|
| `nmem_health` | Brain health: purity score, grade (A-F), top penalties with fix actions |
| `nmem_review` | Spaced repetition reviews (Leitner box system) |
| `nmem_conflicts` | Memory conflicts: list, resolve, or pre-check |
| `nmem_narrative` | Generate narratives: timeline, topic, or causal chain |
| `nmem_alerts` | Brain health alerts: list or acknowledge |
| `nmem_version` | Brain version control: snapshot, list, rollback, diff |
| `nmem_transplant` | Transplant memories between brains by tags/types |
| `nmem_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `nmem_drift` | Detect and manage semantic drift in tags/terminology |
| `nmem_explain` | Trace shortest path between two concepts |

**Cognitive Reasoning** — Hypotheses, predictions, knowledge gaps

| Tool | Description |
|------|-------------|
| `nmem_hypothesize` | Create hypotheses with Bayesian confidence tracking |
| `nmem_evidence` | Submit evidence for/against hypotheses |
| `nmem_predict` | Make falsifiable predictions with deadlines |
| `nmem_verify` | Verify predictions as correct/wrong |
| `nmem_cognitive` | Hot index: ranked active hypotheses + predictions |
| `nmem_gaps` | Knowledge gaps: detect, track, and resolve |
| `nmem_schema` | Schema evolution: evolve hypotheses with SUPERSEDES links |

**Cloud Sync & Backup**

| Tool | Description |
|------|-------------|
| `nmem_sync` | Cloud sync: push, pull, full, or seed |
| `nmem_sync_status` | Sync status, devices, and cloud tier |
| `nmem_sync_config` | Configure sync — setup, get, or set |
| `nmem_telegram_backup` | Send brain backup to Telegram |

</details>

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
| **Synapse** | A weighted, typed connection between neurons (`CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, etc.) |
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

### Knowledge Base Training

```bash
# Train from documents (permanent knowledge)
nmem_train(action="train", path="/docs/", domain_tag="project-docs")

# Supported formats: PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD, TXT, RST
# Trained memories are pinned — they never decay, prune, or compress

# Pin/unpin specific memories
nmem_pin(fiber_ids=["abc123"], pinned=True)
```

Install extraction dependencies:
```bash
pip install neural-memory[extract]
```

### Brain Health & Diagnostics

```bash
nmem_health()                       # Purity score, grade (A-F), top penalties
nmem_alerts(action="list")          # Active health alerts
nmem_review(action="queue")         # Spaced repetition review queue
```

Health reports include **`top_penalties`** — a ranked list of what's hurting your score most, with exact fix actions. Always fix the highest penalty first.

7 components: Connectivity (25%), Diversity (20%), Freshness (15%), Consolidation (15%), Orphan Rate (10%), Activation (10%), Recall Confidence (5%).

See the [Brain Health Guide](docs/guides/brain-health.md) for detailed explanations and improvement roadmap.

### Connection Tracing

Trace the shortest path between two concepts in your neural graph:

```bash
# CLI
nmem explain "Redis" "auth outage"

# MCP tool
nmem_explain(entity_a="Redis", entity_b="auth outage")
```

Returns the path with evidence: `Redis → USED_BY → session-store → CAUSED_BY → auth outage`. Use this to debug recall, verify brain connections, or discover unexpected relationships between concepts.

### Cognitive Reasoning

Hypothesize, predict, verify, and evolve beliefs — the brain reasons about what it knows:

```bash
# Create a hypothesis with initial confidence
nmem_hypothesize(action="create", content="Redis is causing the latency spike", confidence=0.6)

# Submit evidence
nmem_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency at 200ms")
nmem_evidence(hypothesis_id="h-1", evidence_type="against", content="Network latency was 500ms")

# Make a falsifiable prediction
nmem_predict(action="create", content="Switching to Valkey will fix latency",
             hypothesis_id="h-1", deadline="2026-04-01")

# Verify prediction outcome — propagates to linked hypothesis
nmem_verify(prediction_id="p-1", outcome="correct")

# Evolve hypothesis when understanding changes (creates SUPERSEDES chain)
nmem_schema(action="evolve", hypothesis_id="h-1",
            content="Network config was root cause, not Redis",
            reason="New evidence from network team")

# Track what the brain doesn't know
nmem_gaps(action="detect", topic="Why does latency spike at 3am?", source="recall_miss")

# View cognitive dashboard
nmem_cognitive(action="summary")    # Hot index of active hypotheses + predictions
nmem_schema(action="history", hypothesis_id="h-2")  # Version evolution chain
```

Auto-resolution: hypotheses with confidence ≥0.9 + 3 supporting evidence → auto-confirmed. Confidence ≤0.1 + 3 against → auto-refuted. Calibration score tracks prediction accuracy.

### Brain Versioning

```bash
nmem_version(action="create", name="v1-stable")  # Snapshot
nmem_version(action="list")                       # List versions
nmem_version(action="rollback", version_id="...")  # Restore
nmem_version(action="diff", from_version="...", to_version="...")
```

### Web Dashboard

```bash
nmem serve                         # Start server on localhost:8000
# Open http://localhost:8000/dashboard  # React dashboard (7 pages)
# Open http://localhost:8000/docs       # API docs (Swagger)
```

Pages:
- **Overview** — KPI cards (neurons, synapses, fibers, brains) + brain table with click-to-switch and delete
- **Health** — Radar chart + health warnings + recommendations
- **Graph** — Sigma.js WebGL neural graph with ForceAtlas2 layout, color-coded by type, node inspector
- **Timeline** — Chronological memory feed with type badges
- **Evolution** — Brain maturity, plasticity, stage distribution charts
- **Mindmap** — ReactFlow interactive fiber mindmap (dagre tree, zoom/pan, MiniMap)
- **Settings** — Brain files, Telegram backup config

Light/Dark/System theme toggle with warm cream light mode.

### Telegram Backup

Send brain `.db` files to Telegram for offsite backup:

```bash
# Setup: set env var + config
export NMEM_TELEGRAM_BOT_TOKEN="your-bot-token"
# Add to config.toml:
# [telegram]
# enabled = true
# chat_ids = ["123456789"]

# CLI
nmem telegram status              # Check config
nmem telegram test                # Send test message
nmem telegram backup              # Send brain backup
nmem telegram backup --brain work # Specific brain

# MCP tool
nmem_telegram_backup(brain_name="work")
```

### Cloud Sync (Multi-Device)

Sync memories across devices using your own Cloudflare Worker (free tier). Your data stays on **your** account:

```bash
# 1. Deploy your sync hub (one-time)
cd sync-hub && npx wrangler deploy    # Requires free CF account
```

```python
# 2. Register & connect
nmem_sync_config(action="set",
    hub_url="https://YOUR-WORKER.workers.dev",
    api_key="nmk_YOUR_KEY")

# 3. Sync
nmem_sync(action="seed")              # Prepare existing memories
nmem_sync(action="push")              # Push to cloud
nmem_sync(action="pull")              # Pull on another device
nmem_sync(action="full")              # Bidirectional sync
nmem_sync_status()                    # Check sync status & devices
```

See the full [Cloud Sync Guide](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/) for key management, conflict resolution, and troubleshooting.

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
WS   /sync/ws           - Real-time sync (local server)
POST /v1/hub/sync       - Cloud sync (push/pull/full)
POST /v1/hub/register   - Register device for sync
GET  /v1/hub/status     - Hub sync status
GET  /dashboard         - Web dashboard
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

# Run tests (3500+ tests)
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

If you find NeuralMemory useful, consider supporting development:

**Solana:** `5XVY6dZDeyuZJy6Co9KeLDxY5RZ6EwCpjsUVkacMz7HF`

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
