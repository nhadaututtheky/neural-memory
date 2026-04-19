# NeuralMemory FAQ

## Installation

### Q: `pip install neural-memory` installs what?

- **Core**: pydantic, networkx, python-dateutil, typer, aiohttp, aiosqlite, rich
- **CLI tools**: `nmem`, `neural-memory`, `nmem-mcp`
- **Optional extras**:
  - `[server]` — FastAPI + Uvicorn
  - `[nlp-en]` — English NLP (spaCy)
  - `[nlp-vi]` — Vietnamese NLP (underthesea, pyvi)
  - `[all]` — All of the above
  - `[dev]` — Development tools (pytest, ruff, mypy, etc.)

### Q: `pip` not working on Windows?

Use `python -m pip` instead:

```powershell
python -m pip install neural-memory[all]
```

### Q: How to install from source?

```bash
git clone https://github.com/nhadaututtheky/neural-memory.git
cd neural-memory
pip install -e ".[all,dev]"
```

The `-e` flag enables editable mode — code changes take effect immediately without reinstalling.

### Q: Too many commands — is there a simpler UI/UX approach?

Yes. If you use VS Code, the **NeuralMemory extension** provides a full GUI — no terminal commands needed:

- **Encode memory**: Select text → `Ctrl+Shift+M E`
- **Query memory**: `Ctrl+Shift+M Q` → type your question
- **Start/stop server**: `Ctrl+Shift+P` → NeuralMemory: Start Server
- **Switch brain**: `Ctrl+Shift+P` → NeuralMemory: Switch Brain
- **View graph**: `Ctrl+Shift+P` → NeuralMemory: Open Graph View

The sidebar panel also shows neurons, fibers, and brain stats at a glance.

### Q: How to update to the latest version?

```bash
pip install --upgrade neural-memory[all]
```

If installed from source:

```bash
git pull
pip install -e ".[all,dev]"
```

## Troubleshooting

### Q: Something's not working — where do I start?

Follow this decision tree:

1. **MCP server not connecting?**
    - Check: `nmem-mcp --version` → works? If not, `pip install neural-memory`
    - Check your MCP config file (`.mcp.json` or `~/.claude.json`) has the correct command
    - Restart your AI tool after config changes

2. **Recall returns nothing?**
    - Check: `nmem stats` → `neuron_count > 0`? If zero, store something first with `nmem remember`
    - Try broader queries — spreading activation needs connected neurons

3. **Memories from wrong project?**
    - Check: `nmem brain list` → which brain is active?
    - Switch: `nmem brain switch <name>` or let the MCP auto-detect via git context

4. **Slow performance?**
    - Check: `nmem health` → look for "consolidation needed" warnings
    - Run: `nmem consolidate` to compact the brain
    - Past 10K memories? Consider [Neural Memory Pro](guides/pro-quickstart.md) for HNSW-accelerated recall

5. **Import/export issues?**
    - Export: `nmem export --format json > backup.json`
    - Import: `nmem import backup.json`

## VS Code Extension

### Q: How to use NeuralMemory without writing Python code?

Install the VS Code extension:

1. Open VS Code
2. Go to Extensions (`Ctrl+Shift+X`)
3. Search **NeuralMemory**
4. Click **Install**

> **Note**: The extension still requires Python + `neural-memory` package installed on the machine as its backend.

### Q: Extension not showing data?

1. Start the server: `Ctrl+Shift+P` → **NeuralMemory: Start Server**
2. Switch brain if needed: `Ctrl+Shift+P` → **NeuralMemory: Switch Brain** → select your brain
3. Click refresh

### Q: Server running on a different port than the extension expects?

The extension defaults to port `8000`. If your server runs on a different port, update the setting:

1. Open VS Code Settings (`Ctrl+,`)
2. Search `neuralmemory.serverUrl`
3. Set it to your server's URL (e.g. `http://127.0.0.1:8080`)

### Q: Server is running, correct port set, but extension still shows nothing?

A new brain starts empty — there is no data to display yet. You need to encode at least one memory first:

1. Select any text in your editor
2. Press `Ctrl+Shift+M E` to encode it
3. Click refresh in the NeuralMemory sidebar

After encoding, neurons and fibers will appear in the sidebar.

### Q: I opened the server URL in the browser but only see JSON info, not my data?

The root URL (`/`) only shows basic API info. To view your data:

- **Graph visualization (UI)**: go to `/ui` (e.g. `http://127.0.0.1:8000/ui`)
- **API documentation (Swagger)**: go to `/docs` (e.g. `http://127.0.0.1:8000/docs`)
- **Neurons list (API)**: go to `/memory/neurons` (requires `X-Brain-ID` header)

The VS Code sidebar is the main way to browse your neurons and fibers.

## Per-Project Configuration

### Q: How do I keep memories from different projects separate?

**You don't need to do anything.** NeuralMemory auto-detects which git repo and branch you're in, then tags your memories accordingly. When you recall, it prioritizes memories from your current context.

Just use NeuralMemory normally — the right memories surface for the right project.

### Q: What if I want complete separation between projects?

Think of a **brain** as a separate notebook. By default, everything goes into one notebook called `default`. If you want a completely separate notebook for a different project, just tell the AI:

> "Create a new brain called work-api and switch to it"

To switch back later:

> "Switch to my default brain"

Memories in `work-api` and `default` never mix.

### Q: Can I make Claude Code auto-pick the right brain per project?

Yes. Drop a `.mcp.json` file in each project root:

```jsonc
// ~/projects/work-api/.mcp.json
{
  "mcpServers": {
    "neuralmemory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"],
      "env": { "NEURALMEMORY_BRAIN": "work-api" }
    }
  }
}
```

Now when you open that project in Claude Code, it automatically uses the `work-api` brain.

### Q: Can I save project rules that the AI always follows?

Yes. Just tell the AI in plain language:

> "Remember: this project uses PostgreSQL, never SQLite"
>
> "Remember: all API responses must use the {data, error, meta} format"

These are saved in your current brain and automatically surface when relevant — you don't need to know any commands.

For bigger project context, just say:

> "Save this project context: name is work-api, tech stack is Python, FastAPI, PostgreSQL"

Next session, say "recap" or "what were we working on?" and everything reloads instantly.

## Database Training

### Q: Can NeuralMemory learn from my database?

Yes. The `nmem_train_db` tool teaches your brain to understand database structure:

```
nmem_train_db(action="train", connection_string="sqlite:///myapp.db", domain_tag="myapp")
```

This extracts schema knowledge (tables, relationships, patterns) — NOT raw data rows.

### Q: Which databases are supported?

SQLite only in v1. The architecture supports adding PostgreSQL and MySQL dialects in the future.

### Q: Will it read my actual data?

No. DB-to-Brain opens the database in **read-only mode** and only reads PRAGMA metadata (table definitions, foreign keys, indexes). No `SELECT` queries are run on your data tables (except `COUNT(*)` for row count estimates).

### Q: How do I check what schema knowledge is stored?

```
nmem_train_db(action="status")
```

This shows how many schema entities have been trained into the current brain.

## AI Agent Skills

### Q: What are NeuralMemory skills?

Composable AI agent workflows that follow the [ship-faster](https://github.com/Heyvhuang/ship-faster) SKILL.md pattern. They give Claude Code structured methods for memory management:

| Skill | What it does |
|-------|-------------|
| `memory-intake` | Converts messy notes into well-typed, tagged memories via 1-question-at-a-time clarification |
| `memory-audit` | 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure) with A-F grading |
| `memory-evolution` | Evidence-based optimization — consolidation, enrichment, pruning, tag normalization |

### Q: How do I install the skills?

Copy from the NeuralMemory repo to your Claude Code skills directory:

```bash
# Linux/macOS
cp -r skills/memory-* ~/.claude/skills/

# Windows
xcopy /E /I skills\memory-intake %USERPROFILE%\.claude\skills\memory-intake
xcopy /E /I skills\memory-audit %USERPROFILE%\.claude\skills\memory-audit
xcopy /E /I skills\memory-evolution %USERPROFILE%\.claude\skills\memory-evolution
```

### Q: How do I use them?

In Claude Code, invoke by name:

```
/memory-intake "Meeting notes: decided on Redis caching, Bob handles migration, deadline Friday"
/memory-audit
/memory-evolution "Focus on the auth topic"
```

### Q: Do they require anything besides NeuralMemory?

Just the NeuralMemory MCP server configured in Claude Code. The skills use existing MCP tools (`nmem_remember`, `nmem_recall`, `nmem_stats`, `nmem_health`, etc.) — no new dependencies.

## Data & Multi-tool Sharing

### Q: Do I need to install NeuralMemory per project?

No. Install once globally and it works for the entire machine:

```bash
python -m pip install neural-memory[all]
```

Data is stored in `~/.neuralmemory/` — not tied to any specific project. All tools (CLI, AntiGravity, Claude Code, VS Code extension) read from the same location.

To separate data per project, use different brains:

```bash
nmem brain create my-project
nmem brain switch my-project
```

### Q: How to share brain data between AntiGravity and Claude Code?

They already share the same brain automatically. Both read/write to the same files:

- **Config**: `~/.neuralmemory/config.toml`
- **Data**: `~/.neuralmemory/brains/<name>.db`

As long as both tools point to the same `current_brain`, all memories are synced. Verify with:

```bash
nmem config show
```

### Q: How do I let Claude Code query my brain using natural language?

Add the NeuralMemory MCP server to `~/.claude.json`:

```json
{
  "mcpServers": {
    "neuralmemory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"]
    }
  }
}
```

After restarting Claude Code, just ask naturally:

> "What did we decide about the volume spike feature?"

The agent will automatically call the `recall` tool to search your brain.

### Q: How do I sync memories across multiple devices?

Use cloud sync — register once, then sync from any device:

```python
# One-time setup
nmem_sync_config(action="setup")  # Shows registration steps

# Connect (after getting your API key)
nmem_sync_config(action="set",
    hub_url="https://YOUR-WORKER.workers.dev",
    api_key="nmk_YOUR_KEY")

# Sync
nmem_sync(action="seed")   # Prepare existing memories (first time)
nmem_sync(action="push")   # Push to cloud
nmem_sync(action="pull")   # Pull on another device
```

See the full [Cloud Sync Guide](guides/cloud-sync.md) for details.

### Q: Is cloud sync free?

Yes, the free tier is available for personal use. Your data is transmitted over HTTPS and only accessible with your API key.

### Q: Can I self-host the sync hub?

Yes. Run `nmem serve` locally and point your devices to it:

```python
nmem_sync_config(action="set", hub_url="http://localhost:8000")
```

Local hubs don't require API keys or HTTPS.

## NeuralMemory vs RAG

### Q: I have 100 document files (law, specs, etc.) — should I use NeuralMemory or RAG?

**RAG/Vector Search is better for document lookup.** Here's why:

| Aspect | RAG | NeuralMemory |
|--------|-----|-------------|
| **Best for** | "Find the paragraph about X" | "Why did we decide X?" |
| **Query style** | Semantic similarity search | Associative recall through graph traversal |
| **100 law docs** | Chunks + embeddings → sub-10ms exact match | SimHash + keyword anchors → weaker semantic matching |
| **Embedding quality** | LLM understands synonyms, paraphrasing | No embeddings — pure algorithmic, may miss paraphrases |
| **Returns** | Exact text chunks from source docs | Fiber summaries, not original verbatim text |
| **Cost** | ~$0.02/1K queries (embedding API) | $0.00 — fully offline |

**When to use which:**

| Scenario | Winner |
|----------|--------|
| "What does Article 42 say?" (document lookup) | **RAG** |
| "Which law covers remote work?" (search) | **RAG** |
| "Why did the team choose Article 42 over 43?" (decision chain) | **NeuralMemory** |
| "What pattern do we see in deploy failures?" (habit detection) | **NeuralMemory** |
| "What did Alice suggest in yesterday's meeting?" (episodic recall) | **NeuralMemory** |

**Can I use both?** Yes — and that's often the best approach:
- **RAG** for document corpus search (laws, specs, API docs)
- **NeuralMemory** for agent memory (decisions, errors, workflows, context across sessions)

They complement each other. RAG answers "what does the document say?", NeuralMemory answers "what did we learn and decide?"

### Q: What about `nmem_train` for documents — doesn't that compete with RAG?

`nmem_train` ingests documents into the neural graph as permanent (pinned) knowledge. It works well for:
- Project docs you reference occasionally
- Onboarding guides where relationships matter
- Small doc sets (<20 files) where graph traversal adds value

For **large document corpus search** (100+ files, legal compliance, exact text retrieval), RAG with a proper vector database (Chroma, Pinecone, Weaviate) will outperform NeuralMemory significantly. NeuralMemory's strength is *associative recall* and *relationship tracing*, not *semantic text search*.

### Q: Is NeuralMemory basically a Knowledge Graph database?

**Close, but no.** They share graph structure, but serve fundamentally different purposes:

| Aspect | Knowledge Graph DB (Neo4j, FalkorDB) | NeuralMemory |
|--------|--------------------------------------|-------------|
| **Purpose** | Static knowledge map — store entities & relationships | Living agent memory — learn from experience |
| **Schema** | Explicit: entities, relationships defined upfront | Emergent: neurons form connections organically |
| **Query** | Cypher/GQL — precise graph queries | Spreading activation — signal propagates like a reflex |
| **Lifecycle** | Data persists unchanged until explicitly modified | Memories decay, strengthen with use (Hebbian learning), consolidate during "sleep" |
| **Scale target** | Millions of nodes, enterprise workloads | Agent-scale (thousands of neurons), sub-ms latency |
| **Behavior** | Database — stores what you tell it | Brain — forgets what you don't use, strengthens what you repeat |

**In short:**
- Knowledge Graph = **encyclopedia** (static facts, explicit relationships)
- NeuralMemory = **biological memory** (learns, forgets, associates, consolidates)

NeuralMemory *can use* a knowledge graph DB as its backend (FalkorDB opt-in is shipped), but the memory system itself adds biological behaviors on top: decay, Hebbian learning, sleep consolidation, spreading activation, and memory stages (episodic → semantic).

---

## Technical & Architecture

### Q: Does NeuralMemory use LLMs or embeddings?

**No.** NeuralMemory does **not** call any LLM or embedding API for encoding or retrieval. This is a common misconception.

| Operation | How it works |
|-----------|-------------|
| **Encoding** | Rule-based decomposition: keyword extraction, regex patterns, entity detection, relation extraction |
| **Retrieval** | Spreading activation on a neural graph — signal propagates through weighted synapses |
| **Ranking** | Multi-factor scoring: activation level, freshness, frequency, conductivity, memory stage |

This means:

- **Zero API cost** — no OpenAI/Anthropic calls for memory operations
- **Zero hallucination risk** at the retrieval layer — what you stored is what you get back
- **Fully offline** — works without internet after install
- **Deterministic** — same query on same brain = same results (no model randomness)

The only AI involvement is the *agent itself* (Claude, GPT, etc.) deciding when to call NeuralMemory tools. The memory system is pure algorithmic graph traversal.

### Q: What are the performance benchmarks?

NeuralMemory includes reproducible benchmarks. Key results (median, 10 runs each):

**Activation Engine** (synthetic graphs):

| Graph Size | Hybrid Latency | Speedup vs Classic |
|-----------|---------------|-------------------|
| 100 neurons | 0.34ms | 4.6x faster |
| 1,000 neurons | 0.40ms | 5.5x faster |
| 5,000 neurons | 0.45ms | 5.1x faster |

**Full Pipeline** (15 memories, 5 queries):

| Query | Latency | Confidence |
|-------|---------|-----------|
| "What did Alice suggest?" | 1.6ms | 1.0 |
| "Why did we choose PostgreSQL?" | 0.9ms | 1.0 |

Sub-millisecond retrieval even at 5,000 neurons. See [full benchmarks](benchmarks.md) for methodology.

To regenerate on your hardware:

```bash
python benchmarks/run_benchmarks.py
```

### Q: Do I need to manually migrate my database when upgrading?

**No.** NeuralMemory auto-migrates your database schema on startup. When you upgrade to a new version, the first time the brain is opened, `run_migrations()` detects the current schema version and applies all necessary migrations sequentially.

For example, upgrading from v4.0 (schema v22) to v4.2 (schema v26) runs migrations v22→v23→v24→v25→v26 automatically. Each step is logged:

```
INFO: Migrating schema v22 → v23 (added cognitive_states table)
INFO: Migrating schema v23 → v24 (added tool_events table)
...
```

**Verify after upgrade:**

```bash
nmem doctor  # Shows current schema version + any issues
```

**Important notes:**

- Migrations are **forward-only** — downgrading to an older version after migration is not supported
- Your existing memories, brains, and configurations are preserved
- If a migration fails mid-way, it halts safely (no partial schema changes)
- Back up your brain DB before major upgrades: `cp ~/.neuralmemory/brains/default.db ~/.neuralmemory/brains/default.db.bak`

### Q: Why is my consolidation 0%?

This is **completely normal for new brains**. Consolidation measures how many memories have matured from episodic (raw) to semantic (pattern) stage. A new brain has no matured memories yet.

**When to worry**: If your brain is older than 1 week with 50+ memories and consolidation is still 0%, run:

```bash
nmem consolidate --strategy mature
```

Or in Claude Code, just say: "consolidate my memories".

**Expected progression**: After 1-2 weeks of active use → 20-40%. After a month → 50%+. A brand new brain at 0% is not a bug — it just means no memories have been recalled enough times to qualify for maturation.

### Q: How does consolidation work? When should I run it?

Consolidation maintains brain health through three strategies:

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `mature` | Advances episodic memories to semantic stage | Weekly — promotes stable patterns |
| `merge` | Combines overlapping fibers, prunes weak synapses | When brain grows large (1000+ fibers) |
| `full` | All strategies + topic summarization | Monthly deep cleanup |

```bash
# Recommended schedule
nmem consolidate --strategy mature     # Weekly
nmem consolidate --strategy full       # Monthly

# Check what would be affected first
nmem brain health
```

**Important**: Consolidation never deletes memories outright. It:

- **Prunes weak synapses** (connections with very low weight) — the neurons remain
- **Merges overlapping fibers** — combines near-duplicate memory traces
- **Summarizes topic clusters** — creates summary neurons linking related memories

If you're unsure, run `nmem brain health` first to see diagnostics and recommendations.

### Q: How does multi-device sync handle conflicts?

NeuralMemory uses a **local-first** architecture with hub-and-spoke sync:

```
Device A (SQLite) ──┐
                    ├──→ Hub Server ──→ Device B (SQLite)
Device C (SQLite) ──┘
```

**Conflict resolution strategies:**

| Strategy | Behavior |
|----------|---------|
| `prefer_recent` | Most recently modified memory wins (default) |
| `prefer_local` | Local device always wins |
| `prefer_remote` | Hub version always wins |
| `prefer_stronger` | Memory with higher activation/confidence wins |

**Offline support**: Full. Each device has a complete SQLite brain. Sync is incremental — only changed fibers/neurons are transmitted. Changes queue locally and sync when connectivity resumes.

**Concurrent writes**: Each device tracks a vector clock. When two devices modify the same neuron, the configured strategy resolves the conflict automatically. No manual merge required.

```bash
# Configure sync
nmem sync config set --hub-url https://your-hub:8000 --strategy prefer_recent

# Manual sync
nmem sync push    # Send local changes
nmem sync pull    # Get remote changes
nmem sync full    # Bidirectional
```

### Q: Is NeuralMemory production-ready?

NeuralMemory is designed for **AI agent memory** — not as a general-purpose database. Here's an honest assessment:

| Aspect | Status |
|--------|--------|
| **Test suite** | 7100+ tests, 67%+ coverage enforced by CI |
| **Security** | Input validation, ReDoS protection, activation queue caps, sensitive content detection |
| **Stability** | 51+ releases, used daily by the maintainers in production AI workflows |
| **Scalability** | Tested up to 5,000 neurons with sub-ms latency; designed for agent-scale data, not big data |
| **Third-party audit** | Not yet — contributions welcome |
| **SLA** | None — this is open-source MIT software |
| **Contributors** | Small team — bus factor risk mitigated by clean architecture + MIT license |

**Good fit for:**

- AI coding assistants (Claude Code, Cursor, Windsurf)
- Personal knowledge management
- Small team agent workflows
- Research and experimentation

**Not designed for:**

- Enterprise databases with SLA requirements
- High-throughput write-heavy workloads (100K+ writes/sec)
- Multi-tenant SaaS platforms
- Replacing production search infrastructure

### Q: How does NeuralMemory handle concurrent access from multiple agents?

SQLite with WAL (Write-Ahead Logging) mode enables:

- **Multiple concurrent readers** — agents can query simultaneously
- **Single writer at a time** — writes are serialized by SQLite's locking
- **No corruption risk** — WAL mode prevents reader-writer conflicts

For multi-agent write scenarios, NeuralMemory uses a **deferred write queue** — non-critical writes (Hebbian weight updates, conductivity changes) are batched and flushed after the response, reducing lock contention.

If you need true multi-writer concurrency across processes, use the [FastAPI server](api/server.md) as a central write coordinator.
