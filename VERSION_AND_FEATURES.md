# NeuralMemory — Version Bump & Feature Reference

> Complete checklist and feature catalog. Update this file when adding new features.

---

## Version Bump Checklist (7 touch points)

| # | File | Line | What | When |
|---|------|------|------|------|
| 1 | `pyproject.toml` | 7 | `version = "0.8.0"` | Every release |
| 2 | `src/neural_memory/__init__.py` | 17 | `__version__ = "0.8.0"` | Every release (must match #1) |
| 3 | `CHANGELOG.md` | 8 | Move `[Unreleased]` to `[X.Y.Z]` | Every release |
| 4 | `vscode-extension/package.json` | 5 | `"version": "0.1.2"` | Every extension release |
| 5 | `vscode-extension/CHANGELOG.md` | 3 | Add `[X.Y.Z] - date` section | Every extension release |
| 6 | `storage/sqlite_schema.py` | 14 | `SCHEMA_VERSION = 3` | Only on DB schema changes |
| 7 | `storage/sqlite_brain_ops.py` + `memory_brain_ops.py` | 108/138 | `version="0.1.0"` | Only on snapshot format changes |

---

## Files That Auto-Read Version (verify after bump)

| File | How |
|------|-----|
| `server/app.py` | `from neural_memory import __version__` — FastAPI metadata + `/health` |
| `mcp/server.py` | `from neural_memory import __version__` — MCP server |
| `cli/commands/info.py` | `version()` command displays `__version__` |
| `cli/update_check.py` | Compares `__version__` vs latest on PyPI |
| `vscode-extension/src/utils/updateChecker.ts` | Reads from `packageJSON.version` |

---

## API Versioning

- `/api/v1/` prefix — only bump on **breaking** API changes
- Legacy unversioned routes kept at `/` for backward compat
- Both defined in `server/app.py`

---

## Release Triggers

| Target | Trigger | Workflow |
|--------|---------|----------|
| Python package (PyPI) | Git tag `v*` | `.github/workflows/release.yml` |
| VS Code extension | Manual `vsce publish` | N/A |
| Docker | Manual build | `Dockerfile` (Python 3.11-slim) |

---

## All CLI Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `nmem remember` | Store memory | --tag, --type, --priority, --expires, --project, --shared, --force, --redact, --json |
| `nmem recall` | Query memories | --depth, --max-tokens, --min-confidence, --show-age, --show-routing, --json |
| `nmem context` | Get recent context | --limit, --fresh-only, --json |
| `nmem todo` | Quick TODO | --priority, --project, --expires, --tag, --json |
| `nmem q` | Quick recall shortcut | -d (depth) |
| `nmem a` | Quick add shortcut | -p (priority) |
| `nmem last` | Show last N memories | -n (count) |
| `nmem today` | Show today's memories | — |
| `nmem stats` | Enhanced brain stats | --json |
| `nmem status` | Quick status + suggestions | --json |
| `nmem check` | Sensitive content detection | --json |
| `nmem version` | Show version | — |
| `nmem list` | List memories | --type, --min-priority, --project, --expired, --include-expired, --limit, --json |
| `nmem cleanup` | Remove expired memories | --expired, --type, --dry-run, --force, --json |
| `nmem consolidate` | Prune/merge/summarize | --brain, --strategy, --dry-run, --prune-threshold, --merge-overlap, --min-inactive-days |
| `nmem decay` | Run memory decay | --brain |
| `nmem init` | Initialize NeuralMemory | — |
| `nmem serve` | Run FastAPI server | --host, --port |
| `nmem mcp` | Run MCP server | — |
| `nmem mcp-config` | Generate MCP config JSON | --with-prompt, --compact |
| `nmem prompt` | Show system prompt | --compact, --copy |
| `nmem dashboard` | Rich dashboard | — |
| `nmem ui` | Interactive browser | — |
| `nmem graph` | Graph explorer | — |
| `nmem hooks` | Configure hooks | — |
| `nmem export` | Export brain to JSON | --brain |
| `nmem import` | Import brain from JSON | --brain, --merge, --strategy |
| `nmem brain list` | List brains | — |
| `nmem brain use` | Switch brain | — |
| `nmem brain create` | Create brain | — |
| `nmem brain delete` | Delete brain | — |
| `nmem brain export` | Export brain | --output, --name, --exclude-sensitive |
| `nmem brain import` | Import brain | --name, --use, --scan |
| `nmem brain health` | Brain health check | --name, --json |
| `nmem project create` | Create project | --description, --duration, --tag, --priority, --json |
| `nmem project list` | List projects | --json |
| `nmem project show` | Show project details | --json |
| `nmem project delete` | Delete project | — |
| `nmem project extend` | Extend project duration | — |
| `nmem shared enable` | Enable shared mode | --api-key, --timeout |
| `nmem shared disable` | Disable shared mode | — |
| `nmem shared status` | Show shared mode status | — |
| `nmem shared test` | Test shared connection | — |
| `nmem shared sync` | Sync local with remote | — |

---

## All API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + version |
| GET | `/` | API info |
| POST | `/memory/encode` | Encode memory |
| POST | `/memory/query` | Query memories |
| GET | `/memory/fiber/{id}` | Get fiber details |
| GET | `/memory/neurons` | List neurons |
| GET | `/memory/suggest` | Prefix-based neuron suggestions |
| POST | `/brain/create` | Create brain |
| GET | `/brain/{id}` | Get brain details |
| GET | `/brain/{id}/stats` | Enhanced statistics |
| GET | `/brain/{id}/export` | Export brain snapshot |
| POST | `/brain/{id}/import` | Import brain snapshot |
| POST | `/brain/{id}/merge` | Merge brain with conflict resolution |
| POST | `/brain/{id}/consolidate` | Sleep & consolidate |
| DELETE | `/brain/{id}` | Delete brain |
| WS | `/sync/ws` | WebSocket real-time sync |
| GET | `/sync/stats` | Sync statistics |

> All REST endpoints also available at `/api/v1/` prefix.

---

## All MCP Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `nmem_remember` | content, type?, priority?, tags?, expires_days? | Store a memory |
| `nmem_recall` | query, depth?, max_tokens?, min_confidence? | Query memories |
| `nmem_context` | limit?, fresh_only? | Get recent context |
| `nmem_todo` | task, priority? | Quick TODO (30-day expiry) |
| `nmem_stats` | — | Brain statistics |
| `nmem_auto` | action (status/enable/disable/analyze/process), text?, save? | Auto-capture from text (with passive recall learning) |
| `nmem_suggest` | prefix, limit?, type_filter? | Prefix-based autocomplete suggestions ranked by relevance + frequency |

**MCP Resources:**
- `neuralmemory://prompt/system` — Full system prompt
- `neuralmemory://prompt/compact` — Compact system prompt

---

## VS Code Extension Features

### Commands (10)

| Command ID | Label | Keybinding |
|------------|-------|------------|
| `neuralmemory.encode` | Encode Selection as Memory | `Ctrl+Shift+M E` |
| `neuralmemory.encodeInput` | Encode Text as Memory | — |
| `neuralmemory.recall` | Recall Memory | `Ctrl+Shift+M R` |
| `neuralmemory.openGraph` | Open Graph Explorer | `Ctrl+Shift+M G` |
| `neuralmemory.switchBrain` | Switch Brain | — |
| `neuralmemory.createBrain` | Create Brain | — |
| `neuralmemory.refreshMemories` | Refresh Memory Tree | — |
| `neuralmemory.startServer` | Start Server | — |
| `neuralmemory.connectServer` | Connect to Server | — |
| `neuralmemory.recallFromTree` | Recall Related Memories | — |

### Settings (6)

| Setting | Default | Description |
|---------|---------|-------------|
| `neuralmemory.pythonPath` | `"python"` | Python interpreter path |
| `neuralmemory.autoStart` | `false` | Auto-start server on activate |
| `neuralmemory.serverUrl` | `"http://127.0.0.1:8000"` | Server URL |
| `neuralmemory.graphNodeLimit` | `1000` | Max nodes in graph (50-10000) |
| `neuralmemory.codeLensEnabled` | `true` | Show CodeLens hints |
| `neuralmemory.commentTriggers` | `["remember:", "note:", "decision:", "todo:"]` | Comment patterns |

### UI

- **Activity Bar**: Brain icon sidebar
- **Tree View**: "Memories" panel showing neurons and fibers
- **CodeLens**: Inline hints on matching comment patterns
- **Graph Explorer**: Interactive neural graph webview

---

## Auto-Capture System

Brain tự động tích lũy memories qua MCP usage, aligned với "The Key: Associative Reflex" vision.

### How It Works

1. **Passive capture on `nmem_recall`**: Queries >=50 chars are analyzed for capturable patterns (fire-and-forget, higher confidence threshold 0.8)
2. **Explicit capture via `nmem_auto process`**: Analyze text and save detected memories (respects `enabled` flag)
3. **Pattern analysis via `nmem_auto analyze`**: Preview detected patterns without saving

### Detection Patterns (5 categories)

| Category | Confidence | Priority | Example Triggers |
|----------|-----------|----------|-----------------|
| `decision` | 0.8 | 6 | "decided to", "chose X over Y", "quyết định" |
| `error` | 0.85 | 7 | "error:", "bug:", "fixed by", "lỗi do" |
| `todo` | 0.75 | 5 | "TODO:", "need to", "cần phải" |
| `fact` | 0.7 | 5 | "answer is", "works because", "giải pháp là" |
| `insight` | 0.8 | 6 | "turns out", "root cause was", "hóa ra", "TIL" |

### Configuration (`~/.neuralmemory/config.toml`)

```toml
[auto]
enabled = true              # Master switch (default: true)
capture_decisions = true
capture_errors = true
capture_todos = true
capture_facts = true
capture_insights = true     # NEW: "aha moment" detection
min_confidence = 0.7        # Threshold for explicit process
                            # Passive capture uses max(0.8, min_confidence)
```

### Safety Guards

- Minimum text length: 20 chars (avoids false positives on tiny inputs)
- Passive capture: >=50 char queries only, confidence >=0.8
- Fire-and-forget: passive capture errors never break `nmem_recall`
- `nmem_auto process` enforces `enabled` flag (returns early if disabled)
- Deduplication with type-prefix stripping

### Supported Languages

- English (all 5 categories)
- Vietnamese (decision, error, todo, fact, insight patterns)

---

## Memory Types (10)

| Type | Description |
|------|-------------|
| `fact` | Factual information |
| `decision` | Decisions made |
| `preference` | User preferences |
| `todo` | Tasks to do |
| `insight` | Insights and learnings |
| `context` | Contextual information |
| `instruction` | Instructions and rules |
| `error` | Error patterns |
| `workflow` | Workflow patterns |
| `reference` | Reference material |

---

## Conflict Strategies (4)

| Strategy | Description |
|----------|-------------|
| `prefer_local` | Keep local version on conflict |
| `prefer_remote` | Keep incoming version on conflict |
| `prefer_recent` | Keep whichever was created/updated more recently |
| `prefer_stronger` | Keep higher weight synapses, higher frequency neurons |

---

## Consolidation Strategies (4)

| Strategy | Description |
|----------|-------------|
| `prune` | Remove weak synapses + orphan neurons |
| `merge` | Combine overlapping fibers (Jaccard similarity) |
| `summarize` | Create concept neurons for tag clusters |
| `all` | Run all strategies in order |

---

## Entry Points (pyproject.toml)

```
neural-memory = neural_memory.cli:main
nmem = neural_memory.cli:main
nmem-mcp = neural_memory.mcp:main
```

---

## Optional Extras

| Extra | Packages |
|-------|----------|
| `server` | fastapi, uvicorn |
| `neo4j` | neo4j driver |
| `nlp-en` | spacy |
| `nlp-vi` | underthesea, pyvi |
| `nlp` | nlp-en + nlp-vi |
| `all` | server + neo4j + nlp |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff, mypy, pre-commit, httpx |

---

## Test Files (24)

### Unit Tests (16)
- test_neuron, test_synapse, test_fiber, test_brain_mode, test_project
- test_memory_types, test_activation, test_consolidation, test_hebbian
- test_mcp (52 tests: schemas, tool calls, protocol, resources, storage, auto-capture, passive capture)
- test_router, test_safety, test_sqlite_storage
- test_sync, test_temporal, test_typed_memory_storage

### Integration Tests (2)
- test_encoding_flow, test_query_flow

### E2E Tests (1)
- test_api

### Config
- conftest.py (fixtures)

---

## Documentation Files (20)

| File | Topic |
|------|-------|
| `docs/index.md` | Home |
| `docs/installation.md` | Installation guide |
| `docs/quickstart.md` | Quick start |
| `docs/cli.md` | CLI reference |
| `docs/integration.md` | AI assistant integration |
| `docs/mcp-server.md` | MCP server setup |
| `docs/safety.md` | Security best practices |
| `docs/brain-sharing.md` | Brain export/import |
| `docs/memory-types.md` | Memory types |
| `docs/neurons-synapses.md` | Neural architecture |
| `docs/spreading-activation.md` | Retrieval mechanism |
| `docs/how-it-works.md` | System overview |
| `docs/architecture/overview.md` | System design |
| `docs/architecture/scalability.md` | Scalability |
| `docs/api/server.md` | REST API reference |
| `docs/api/python.md` | Python API reference |
| `docs/benchmarks.md` | Performance benchmarks |
| `docs/FAQ.md` | FAQ |
| `docs/changelog.md` | Version history |
| `docs/contributing.md` | Contributing |

---

## Current Versions (as of 2026-02-06)

| Component | Version |
|-----------|---------|
| Python Package | 0.8.0 |
| VS Code Extension | 0.1.2 |
| Database Schema | 3 |
| Brain Snapshot Format | 0.1.0 |
| API Prefix | /api/v1 |
| Python Requirement | >=3.11 |
| Docker Base | python:3.11-slim |
