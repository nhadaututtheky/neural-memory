# NeuralMemory - Complete Guide

> **Reflex-based memory system for AI agents** - retrieval through activation, not search.

---

## Table of Contents

1. [What is NeuralMemory?](#what-is-neuralmemory)
2. [NeuralMemory vs RAG & Vector Search](#neuralmemory-vs-rag--vector-search)
3. [The Problem](#the-problem)
4. [The Solution](#the-solution)
5. [Features](#features)
6. [Installation](#installation)
7. [CLI Commands Reference](#cli-commands-reference)
8. [Token Economy & Optimization](#token-economy--optimization)
9. [Real-Time Brain Sharing](#real-time-brain-sharing)
10. [Python API](#python-api)
11. [Server API](#server-api)
12. [Architecture](#architecture)
13. [Safety & Security](#safety--security)
14. [Best Practices](#best-practices)

---

## What is NeuralMemory?

NeuralMemory is a **reflex-based memory system** for AI agents. Unlike traditional databases or vector stores that use search queries, NeuralMemory stores experiences as interconnected neurons and recalls them through **spreading activation** - mimicking how the human brain works.

### Key Differentiators

| Traditional Search | NeuralMemory |
|-------------------|--------------|
| Query → Search → Results | Query → Activation → Emergence |
| Keyword matching | Associative recall |
| Flat document storage | Neural graph structure |
| No context understanding | Temporal, causal, semantic links |
| Manual relevance ranking | Activation-based relevance |

### How It Works

1. **Encode**: Experience → Neurons (entities, time, concepts) + Synapses (relationships)
2. **Query**: Question → Signals (time hints, entities, intent)
3. **Activate**: Signals → Anchor neurons → Spread through synapses
4. **Emerge**: High-activation subgraph → Reconstructed context

---

## NeuralMemory vs RAG & Vector Search

There are many RAG and vector search tools on GitHub. Here's why NeuralMemory takes a fundamentally different approach.

### Core Philosophy

| Aspect | RAG / Vector Search | NeuralMemory |
|--------|---------------------|--------------|
| **Mental Model** | Search Engine | Human Brain |
| **Retrieval** | Similarity matching | Associative recall |
| **Core Question** | "What's similar?" | "What's connected?" |
| **Data Model** | Flat chunks + embeddings | Neural graph + synapses |

### Retrieval Mechanism

**RAG/Vector Search:**
```
Query → Embed → Cosine Similarity → Top-K Results
```
Returns documents that are *textually similar* to your query.

**NeuralMemory:**
```
Query → Decompose Signals → Activate Anchors → Spread Through Synapses → Emerge
```
Returns memories that are *semantically connected* through explicit relationships.

### Data Structure Comparison

**RAG - Flat, Independent Chunks:**
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Chunk 1 │  │ Chunk 2 │  │ Chunk 3 │
│ [embed] │  │ [embed] │  │ [embed] │
└─────────┘  └─────────┘  └─────────┘
     ↑            ↑            ↑
     └────────────┴────────────┘
            Similarity Search
```

**NeuralMemory - Connected Neural Graph:**
```
    [Alice]───DISCUSSED───[API design]
       │                       │
   MET_AT                 LEADS_TO
       │                       │
  [Coffee shop]          [FastAPI decision]
       │                       │
   HAPPENED_AT            CAUSED_BY
       │                       │
    [Tuesday]            [Scalability need]
```

### Relationship Understanding

**RAG:** "Alice" and "API" appear in same chunk → considered similar
- No understanding of *how* they're related
- No distinction between "Alice discussed API" vs "Alice hates API"

**NeuralMemory:** Explicit typed relationships (synapses)
- `Alice` → `DISCUSSED` → `API design`
- `API design` → `LEADS_TO` → `FastAPI decision`
- `FastAPI decision` → `CAUSED_BY` → `Scalability need`

### Practical Example: Causal Query

**Storing memories:**
```bash
nmem remember "Met Alice at cafe, she suggested using JWT for auth"
nmem remember "Later that day, implemented JWT based on Alice's advice"
nmem remember "JWT implementation caused the Tuesday outage"
```

**Query: "Why did Tuesday's outage happen?"**

**RAG approach:**
- Embeds query, searches for similar chunks
- Returns: "JWT implementation caused the Tuesday outage" ✓
- **Missing**: *why* we used JWT, *who* suggested it
- Would need multiple queries to trace the causal chain

**NeuralMemory approach:**
- Activates: "Tuesday", "outage" neurons
- Spreads activation through synapses:
  ```
  [outage] ← CAUSED_BY ← [JWT impl] ← BASED_ON ← [Alice's suggestion]
  ```
- Returns complete causal chain:
  > "Tuesday outage was caused by JWT implementation, which was based on Alice's suggestion at the cafe meeting"

### Temporal Intelligence

**RAG:** Time is just metadata, filtered like any field
```sql
WHERE timestamp > '2024-01-01' AND content SIMILAR TO query
```

**NeuralMemory:** Time is a first-class neuron with relationships
```
Query: "What happened before the API decision?"

[API decision]───AFTER───[Alice meeting]───AFTER───[Requirements doc]
                              │
                          HAPPENED_AT
                              │
                          [Tuesday 3pm]
```

Temporal queries work through graph traversal, not timestamp filtering.

### Multi-hop Reasoning

**Query: "What did the person who suggested rate limiting also work on?"**

**RAG:** Cannot do this natively. Would need to:
1. Search for "rate limiting"
2. Parse results to extract person's name
3. Search again for that person's work
4. Hope the chunks contain enough context

**NeuralMemory:** Natural multi-hop graph traversal
```
[rate limiting] ← SUGGESTED_BY ← [Alice] → WORKED_ON → [Auth module]
                                        → WORKED_ON → [Database schema]
```
Single query, follows relationship edges.

### Memory Lifecycle

**RAG:** Static
- Documents don't change unless manually updated
- No concept of memory aging or relevance decay
- Stale information persists indefinitely

**NeuralMemory:** Dynamic like human memory
- **Decay**: Unused memories weaken over time (Ebbinghaus forgetting curve)
- **Reinforcement**: Frequently accessed paths strengthen (Hebbian learning)
- **Compression**: Old memories get summarized, details archived
- **Expiry**: Temporary memories (todos, context) auto-expire

### Feature Comparison Table

| Feature | RAG/Vector | NeuralMemory |
|---------|------------|--------------|
| Text similarity search | ✅ Excellent | ⚠️ Not primary |
| Relationship tracking | ❌ None | ✅ Explicit synapses |
| Causal reasoning | ❌ Manual | ✅ CAUSED_BY links |
| Temporal queries | ⚠️ Metadata filter | ✅ Time neurons |
| Multi-hop reasoning | ❌ Multiple queries | ✅ Graph traversal |
| Memory decay | ❌ Static | ✅ Ebbinghaus curve |
| Memory reinforcement | ❌ None | ✅ Hebbian learning |
| Knowledge sharing | ⚠️ Export embeddings | ✅ Export brain graph |
| Contradiction detection | ❌ None | ⚠️ Planned |
| Human-like recall | ❌ Search model | ✅ Association model |

### When to Use What

| Use Case | Best Tool | Why |
|----------|-----------|-----|
| Document Q&A | RAG | Need text similarity |
| Semantic code search | Vector DB | Embedding similarity works well |
| Knowledge base search | RAG | Large corpus, keyword-ish queries |
| **Agent memory** | **NeuralMemory** | Need persistence + relationships |
| **Decision tracking** | **NeuralMemory** | Need causal chains |
| **"Why" questions** | **NeuralMemory** | Need relationship traversal |
| **Temporal queries** | **NeuralMemory** | Need time-aware recall |
| **Multi-session continuity** | **NeuralMemory** | Need persistent graph |
| **Team knowledge sharing** | **NeuralMemory** | Need exportable brains |

### The Key Insight

**Human memory doesn't work like search.**

You don't query your brain with:
```sql
SELECT * FROM memories WHERE content LIKE '%Alice%' ORDER BY similarity DESC LIMIT 10
```

Instead, thinking of "Alice" *activates* related memories - her face, your last conversation, the project you worked on together, the coffee shop where you met. These emerge through **association**, not **search**.

**RAG/Vector Search** = "Find documents similar to this query"
- Great for: knowledge bases, documentation, search engines

**NeuralMemory** = "Remember experiences and recall through association"
- Great for: agent memory, decision history, causal reasoning, temporal understanding

---

## The Problem

AI agents (Claude, GPT, etc.) face fundamental memory limitations:

### 1. Limited Context Windows
```
┌─────────────────────────────────────────┐
│ Context Window (~200k tokens)           │
│ ┌─────────────────────────────────────┐ │
│ │ System prompt + instructions        │ │
│ │ Previous conversation               │ │
│ │ Current file contents               │ │
│ │ Tool outputs                        │ │
│ │ ❌ Project history (lost!)          │ │
│ │ ❌ Past decisions (lost!)           │ │
│ │ ❌ Learned patterns (lost!)         │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Impact**: Cannot complete large projects that span multiple sessions.

### 2. Session Amnesia
Each conversation starts fresh - no memory of:
- Previous debugging sessions
- Architectural decisions made
- User preferences learned
- Patterns discovered

### 3. No Knowledge Sharing
- Agents cannot share learned skills
- Each agent starts from scratch
- No "training" transfer between instances

### 4. Context Overflow
As conversation grows:
- Early important context gets pushed out
- System forgets initial requirements
- Decisions made early are forgotten

---

## The Solution

NeuralMemory provides:

### 1. Persistent Memory
```bash
# Store during session
nmem remember "Fixed auth bug with null check in login.py:42"

# Recall in future sessions
nmem recall "auth bug fix"
# → "Fixed auth bug with null check in login.py:42"
```

### 2. Efficient Retrieval
Only inject relevant context, not everything:
```bash
# Get only what's needed for current task
nmem recall "authentication issues" --max-tokens 500
```

### 3. Shareable Brains
Export and share learned patterns:
```bash
nmem brain export -o auth-expert.json
# Share with team or other agents
nmem brain import auth-expert.json
```

### 4. Project-Bounded Memory
Focus on active project timeframe:
```bash
nmem project create "Q1 Sprint" --duration 14
nmem remember "Sprint task completed" --project "Q1 Sprint"
```

---

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Spreading Activation** | Neural graph-based retrieval |
| **Multi-language** | English + Vietnamese support |
| **Typed Memories** | fact, decision, todo, insight, etc. |
| **Priority System** | 0-10 priority levels |
| **Expiry/TTL** | Auto-expire temporary memories |
| **Project Scoping** | Organize memories by project |
| **Sensitive Content Detection** | Auto-detect secrets, PII |
| **Freshness Tracking** | Age warnings for old memories |

### Storage Options

| Mode | Description | Use Case |
|------|-------------|----------|
| **Local (JSON)** | PersistentStorage | Development, single user |
| **SQLite** | SQLiteStorage | Production, single instance |
| **Shared** | SharedStorage (HTTP) | Multi-agent, team sharing |
| **Hybrid** | Local + sync | Offline-first with sync |

### Brain Sharing

| Feature | Description |
|---------|-------------|
| **Export/Import** | JSON snapshot transfer |
| **Real-time Sync** | WebSocket-based live sync |
| **Toggle On/Off** | Enable/disable sharing per command |

---

## Installation

### Basic Installation
```bash
pip install neural-memory
```

### With Optional Features
```bash
# FastAPI server
pip install neural-memory[server]

# Vietnamese NLP
pip install neural-memory[nlp-vi]

# All features
pip install neural-memory[all]
```

### Development Installation
```bash
git clone https://github.com/neural-memory/neural-memory
cd neural-memory
pip install -e ".[dev]"
```

---

## CLI Commands Reference

### Core Commands

#### `nmem remember` - Store Memory
```bash
nmem remember "content" [OPTIONS]

Options:
  -t, --tag TEXT          Tags for the memory (repeatable)
  -T, --type TEXT         Memory type (auto-detected if not specified)
  -p, --priority INTEGER  Priority 0-10 (0=lowest, 5=normal, 10=critical)
  -e, --expires INTEGER   Days until expiry
  -P, --project TEXT      Associate with project
  -S, --shared            Use shared/remote storage
  -f, --force             Store even if sensitive content detected
  -r, --redact            Auto-redact sensitive content
  -j, --json              Output as JSON

Examples:
  nmem remember "Fixed auth bug with null check"
  nmem remember "We decided to use PostgreSQL" --type decision
  nmem remember "Refactor auth module" --type todo --priority 7
  nmem remember "Meeting notes" --expires 7 --tag meeting
  nmem remember "Team knowledge" --shared
```

**Memory Types:**
| Type | Description | Default Expiry |
|------|-------------|----------------|
| `fact` | Objective information | Never |
| `decision` | Choices made | Never |
| `preference` | User preferences | Never |
| `todo` | Action items | 30 days |
| `insight` | Learned patterns | Never |
| `context` | Situational info | 7 days |
| `instruction` | User guidelines | Never |
| `error` | Error patterns | Never |
| `workflow` | Process patterns | Never |
| `reference` | External references | Never |

#### `nmem recall` - Query Memories
```bash
nmem recall "query" [OPTIONS]

Options:
  -d, --depth INTEGER      Search depth (0=instant, 1=context, 2=habit, 3=deep)
  -m, --max-tokens INTEGER Max tokens in response (default: 500)
  -c, --min-confidence FLOAT Minimum confidence threshold
  -S, --shared             Use shared/remote storage
  -a, --show-age           Show memory ages (default: true)
  -R, --show-routing       Show query routing info
  -j, --json               Output as JSON

Examples:
  nmem recall "auth bug fix"
  nmem recall "meetings with Alice" --depth 2
  nmem recall "Why did the build fail?" --show-routing
  nmem recall "team decisions" --shared --min-confidence 0.5
```

**Depth Levels:**
| Level | Name | Description |
|-------|------|-------------|
| 0 | Instant | Direct recall (who, what, where) |
| 1 | Context | Before/after context (2-3 hops) |
| 2 | Habit | Cross-time patterns |
| 3 | Deep | Full causal/emotional analysis |

#### `nmem todo` - Quick TODO Shortcut
```bash
nmem todo "task" [OPTIONS]

Options:
  -p, --priority INTEGER  Priority 0-10 (default: 5)
  -P, --project TEXT      Associate with project
  -e, --expires INTEGER   Days until expiry (default: 30)
  -t, --tag TEXT          Tags (repeatable)
  -j, --json              Output as JSON

Examples:
  nmem todo "Fix the login bug"
  nmem todo "Review PR #123" --priority 7
  nmem todo "Deploy to prod" -p 10 --project "Q1 Sprint"
```

#### `nmem context` - Get Recent Context
```bash
nmem context [OPTIONS]

Options:
  -l, --limit INTEGER   Number of recent memories (default: 10)
  --fresh-only          Only include memories < 30 days old
  -j, --json            Output as JSON

Examples:
  nmem context
  nmem context --limit 5 --json
  nmem context --fresh-only
```

#### `nmem list` - List Memories
```bash
nmem list [OPTIONS]

Options:
  -T, --type TEXT         Filter by memory type
  -p, --min-priority INT  Minimum priority
  -P, --project TEXT      Filter by project
  -e, --expired           Show only expired memories
  --include-expired       Include expired in results
  -l, --limit INTEGER     Maximum results (default: 20)
  -j, --json              Output as JSON

Examples:
  nmem list --type todo
  nmem list --type decision -p 7
  nmem list --expired
  nmem list --project "Q1 Sprint"
```

#### `nmem cleanup` - Clean Expired Memories
```bash
nmem cleanup [OPTIONS]

Options:
  -e, --expired           Only clean expired (default: true)
  -T, --type TEXT         Only clean specific type
  -n, --dry-run           Preview without deleting
  -f, --force             Skip confirmation
  -j, --json              Output as JSON

Examples:
  nmem cleanup --expired --dry-run
  nmem cleanup --type context
```

#### `nmem stats` - Brain Statistics
```bash
nmem stats [OPTIONS]

Options:
  -j, --json    Output as JSON

Output includes:
  - Neuron/synapse/fiber counts
  - Memory type distribution
  - Priority distribution
  - Freshness breakdown
  - Expired memory count
```

#### `nmem check` - Check for Sensitive Content
```bash
nmem check "content" [OPTIONS]

Options:
  -j, --json    Output as JSON

Examples:
  nmem check "My API_KEY=sk-xxx123"
  # ⚠️ SENSITIVE CONTENT DETECTED
```

---

### Brain Commands

#### `nmem brain list` - List Brains
```bash
nmem brain list [--json]
```

#### `nmem brain create` - Create Brain
```bash
nmem brain create NAME [OPTIONS]

Options:
  -u, --use / --no-use    Switch to new brain (default: true)

Examples:
  nmem brain create work
  nmem brain create personal --no-use
```

#### `nmem brain use` - Switch Brain
```bash
nmem brain use NAME

Examples:
  nmem brain use work
```

#### `nmem brain export` - Export Brain
```bash
nmem brain export [OPTIONS]

Options:
  -o, --output TEXT           Output file path
  -n, --name TEXT             Brain name (default: current)
  -s, --exclude-sensitive     Exclude sensitive content

Examples:
  nmem brain export -o backup.json
  nmem brain export --exclude-sensitive -o safe.json
```

#### `nmem brain import` - Import Brain
```bash
nmem brain import FILE [OPTIONS]

Options:
  -n, --name TEXT    Name for imported brain
  -u, --use          Switch to imported brain (default: true)
  --scan             Scan for sensitive content before import

Examples:
  nmem brain import backup.json
  nmem brain import shared.json --name shared --scan
```

#### `nmem brain delete` - Delete Brain
```bash
nmem brain delete NAME [OPTIONS]

Options:
  -f, --force    Skip confirmation
```

#### `nmem brain health` - Check Brain Health
```bash
nmem brain health [OPTIONS]

Options:
  -n, --name TEXT    Brain name (default: current)
  -j, --json         Output as JSON

Output includes:
  - Health score (0-100)
  - Issues found
  - Sensitive content count
  - Freshness breakdown
```

---

### Project Commands

#### `nmem project create` - Create Project
```bash
nmem project create NAME [OPTIONS]

Options:
  -d, --description TEXT   Project description
  -D, --duration INTEGER   Duration in days
  -t, --tag TEXT           Tags (repeatable)
  -p, --priority FLOAT     Priority (default: 1.0)
  -j, --json               Output as JSON

Examples:
  nmem project create "Q1 Sprint"
  nmem project create "Auth Refactor" --duration 14 -t backend
```

#### `nmem project list` - List Projects
```bash
nmem project list [OPTIONS]

Options:
  -a, --active    Show only active projects
  -j, --json      Output as JSON
```

#### `nmem project show` - Show Project Details
```bash
nmem project show NAME [--json]
```

#### `nmem project delete` - Delete Project
```bash
nmem project delete NAME [--force]
```

#### `nmem project extend` - Extend Deadline
```bash
nmem project extend NAME DAYS [--json]

Examples:
  nmem project extend "Q1 Sprint" 7
```

---

### Shared Mode Commands

#### `nmem shared enable` - Enable Shared Mode
```bash
nmem shared enable URL [OPTIONS]

Options:
  -k, --api-key TEXT     API key for authentication
  -t, --timeout FLOAT    Request timeout in seconds (default: 30)

Examples:
  nmem shared enable http://localhost:8000
  nmem shared enable https://memory.example.com --api-key mykey
```

#### `nmem shared disable` - Disable Shared Mode
```bash
nmem shared disable
```

#### `nmem shared status` - Show Status
```bash
nmem shared status [--json]
```

#### `nmem shared test` - Test Connection
```bash
nmem shared test
```

#### `nmem shared sync` - Sync with Server
```bash
nmem shared sync [OPTIONS]

Options:
  -d, --direction TEXT   push, pull, or both (default: both)
  -j, --json             Output as JSON

Examples:
  nmem shared sync
  nmem shared sync --direction push
  nmem shared sync --direction pull
```

---

## Token Economy & Optimization

NeuralMemory helps optimize your AI agent's token usage.

### The Token Problem

```
Without NeuralMemory:
┌──────────────────────────────────────────┐
│ 200k token context window                 │
│ ┌──────────────────────────────────────┐ │
│ │ Everything from previous sessions    │ │
│ │ All project files                    │ │
│ │ All conversation history             │ │
│ │ ═══════════════════════════════════  │ │
│ │ 💥 OVERFLOW - context truncated      │ │
│ └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### The Solution

```
With NeuralMemory:
┌──────────────────────────────────────────┐
│ 200k token context window                 │
│ ┌──────────────────────────────────────┐ │
│ │ Current task context only            │ │
│ │ Relevant memories (500 tokens)       │ │
│ │ ════════════════════════════════════ │ │
│ │ ✅ Plenty of space remaining         │ │
│ └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### Optimization Strategies

#### 1. Project Scoping
```bash
# Create project with timeframe
nmem project create "Sprint 5" --duration 14

# Memories auto-prioritized by project
nmem remember "Sprint task" --project "Sprint 5"

# Retrieve only project-relevant context
nmem recall "sprint tasks" --project "Sprint 5"
```

#### 2. Memory Types with Auto-Expiry
```bash
# Context expires in 7 days
nmem remember "Meeting notes" --type context

# TODOs expire in 30 days
nmem todo "Review PR"

# Facts never expire
nmem remember "API endpoint: /v2/users" --type fact
```

#### 3. Priority-Based Retrieval
```bash
# Store with priority
nmem remember "Critical decision" --priority 10
nmem remember "Minor note" --priority 2

# Retrieve high-priority only
nmem recall "decisions" --min-priority 7
```

#### 4. Token-Limited Context
```bash
# Get exactly 500 tokens of context
nmem recall "auth implementation" --max-tokens 500

# Get minimal context for injection
nmem context --limit 5 --json
```

#### 5. Fresh-Only Queries
```bash
# Skip stale memories
nmem context --fresh-only

# Cleanup expired regularly
nmem cleanup --expired
```

### Token Budget Planning

| Use Case | Token Budget | Strategy |
|----------|--------------|----------|
| Quick check | 200 tokens | `--max-tokens 200 --depth 0` |
| Context injection | 500 tokens | `--max-tokens 500 --fresh-only` |
| Deep research | 2000 tokens | `--max-tokens 2000 --depth 3` |
| Full recall | 5000 tokens | `--max-tokens 5000` |

---

## Real-Time Brain Sharing

NeuralMemory supports real-time brain sharing between multiple agents.

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Local** | Store locally only | Single user, offline |
| **Shared** | Remote server only | Team sharing, real-time sync |
| **Hybrid** | Local + sync to server | Offline-first with team sync |

### Enable Shared Mode

```bash
# Enable globally
nmem shared enable http://localhost:8000

# Test connection
nmem shared test

# All commands now use remote server
nmem remember "Shared knowledge"
nmem recall "team decisions"

# Disable when done
nmem shared disable
```

### Per-Command Sharing

```bash
# Use --shared flag for single command
nmem remember "Team insight" --shared
nmem recall "project status" --shared

# Without --shared, uses local storage
nmem remember "Local note"
```

### Sync Local with Remote

```bash
# Full sync
nmem shared sync

# Push local to server
nmem shared sync --direction push

# Pull from server to local
nmem shared sync --direction pull
```

### WebSocket Real-Time Updates

When connected to a server, you can receive real-time updates:

```python
from neural_memory.sync import SyncClient

async with SyncClient("http://localhost:8000") as client:
    await client.subscribe("brain-123")

    # Register handlers
    client.on("neuron_created", handle_new_neuron)
    client.on("memory_encoded", handle_new_memory)

    # Run forever to receive updates
    await client.run_forever()
```

---

## Python API

### Basic Usage

```python
import asyncio
from neural_memory import Brain, BrainConfig
from neural_memory.storage import InMemoryStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline, DepthLevel

async def main():
    # Create storage and brain
    storage = InMemoryStorage()
    brain = Brain.create("my-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # Encode memories
    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice at coffee shop to discuss API design")
    await encoder.encode("Decided to use FastAPI for the backend")

    # Query memories
    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What was decided about the backend?")

    print(f"Answer: {result.context}")
    print(f"Confidence: {result.confidence}")
    print(f"Neurons activated: {result.neurons_activated}")

asyncio.run(main())
```

### Using SQLite Storage

```python
from neural_memory.storage import SQLiteStorage

async def main():
    storage = SQLiteStorage("./brain.db")
    await storage.initialize()

    # Use same as InMemoryStorage
    brain = Brain.create("persistent-brain")
    await storage.save_brain(brain)
    # ...
```

### Using Shared Storage

```python
from neural_memory.storage import SharedStorage

async def main():
    async with SharedStorage("http://localhost:8000", "brain-id") as storage:
        # All operations go to remote server
        neurons = await storage.find_neurons(type=NeuronType.CONCEPT)
        # ...
```

### Typed Memories

```python
from neural_memory.core import TypedMemory, MemoryType, Priority

# Create typed memory
memory = TypedMemory.create(
    fiber_id="fiber-123",
    memory_type=MemoryType.DECISION,
    priority=Priority.HIGH,
    source="user_input",
    expires_in_days=30,
    tags={"project", "auth"},
)

await storage.add_typed_memory(memory)

# Query by type
decisions = await storage.find_typed_memories(
    memory_type=MemoryType.DECISION,
    min_priority=Priority.NORMAL,
)
```

### Project Scoping

```python
from neural_memory.core import Project, MemoryScope

# Create project
project = Project.create(
    name="Q1 Sprint",
    description="Sprint 1 of Q1",
    duration_days=14,
    tags={"sprint", "q1"},
)
await storage.add_project(project)

# Associate memories with project
memory = TypedMemory.create(
    fiber_id="fiber-123",
    memory_type=MemoryType.TODO,
    project_id=project.id,
    # ...
)

# Query project memories
memories = await storage.get_project_memories(project.id)
```

---

## Server API

### Start Server

```bash
pip install neural-memory[server]
uvicorn neural_memory.server:app --reload --port 8000
```

### Endpoints

#### Health Check
```
GET /health
Response: {"status": "healthy", "version": "0.1.0"}
```

#### Memory Operations
```
POST /memory/encode
Headers: X-Brain-ID: brain-123
Body: {"content": "Memory content", "tags": ["tag1"]}

POST /memory/query
Headers: X-Brain-ID: brain-123
Body: {"query": "What happened?", "depth": 1, "max_tokens": 500}

GET /memory/neurons?type=concept&limit=50
GET /memory/fiber/{fiber_id}
```

#### Brain Operations
```
POST /brain/create
Body: {"name": "my-brain", "is_public": false}

GET /brain/{brain_id}
GET /brain/{brain_id}/stats
GET /brain/{brain_id}/export
POST /brain/{brain_id}/import
DELETE /brain/{brain_id}
```

#### Sync (WebSocket)
```
WS /sync/ws

Messages:
  → {"action": "connect", "client_id": "client-1"}
  ← {"type": "connected", ...}

  → {"action": "subscribe", "brain_id": "brain-123"}
  ← {"type": "subscribed", ...}

  ← {"type": "neuron_created", "brain_id": "brain-123", "data": {...}}
  ← {"type": "memory_encoded", ...}
```

---

## Architecture

### Memory Layers

```
┌─────────────────────────────────────────────────────────┐
│                    User / CLI / API                      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Engine Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ MemoryEncoder│  │ReflexPipeline│  │SpreadingActivation│
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Extraction Layer                        │
│  ┌───────────┐  ┌────────────┐  ┌──────────────────┐    │
│  │QueryParser│  │QueryRouter │  │TemporalExtractor │    │
│  └───────────┘  └────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Storage Layer                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │InMemory   │  │SQLite     │  │SharedStorage(HTTP)│    │
│  └───────────┘  └───────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Core Layer                            │
│  ┌──────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌───────────┐  │
│  │Neuron│  │Synapse│  │Fiber│  │Brain│  │TypedMemory│  │
│  └──────┘  └───────┘  └─────┘  └─────┘  └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Performance

| Operation | Latency |
|-----------|---------|
| Encode (simple) | 10-30ms |
| Encode (complex) | 30-50ms |
| Query (depth 0) | 5-20ms |
| Query (depth 1) | 20-50ms |
| Query (depth 2) | 50-100ms |
| Query (depth 3) | 100-150ms |

---

## Safety & Security

### Sensitive Content Detection

NeuralMemory automatically detects:
- API keys and secrets
- Passwords
- AWS/Azure/GCP credentials
- Private keys (PEM)
- JWT tokens
- Credit card numbers
- Social Security Numbers

```bash
# Check before storing
nmem check "API_KEY=sk-xxx123"

# Auto-redact
nmem remember "Config: API_KEY=sk-xxx" --redact

# Safe export
nmem brain export --exclude-sensitive -o safe.json
```

### Memory Freshness

| Level | Age | Risk |
|-------|-----|------|
| 🟢 Fresh | < 7 days | Safe |
| 🟢 Recent | 7-30 days | Generally safe |
| 🟡 Aging | 30-90 days | Consider verifying |
| 🟠 Stale | 90-365 days | Verify before using |
| 🔴 Ancient | > 365 days | Likely outdated |

```bash
# Check brain health
nmem brain health

# Fresh-only context
nmem context --fresh-only

# Set confidence threshold
nmem recall "critical info" --min-confidence 0.7
```

### Best Practices

1. **Never store secrets** - Use `nmem check` first
2. **Use brain isolation** - Separate brains for different security contexts
3. **Regular cleanup** - `nmem cleanup --expired`
4. **Export safely** - Always use `--exclude-sensitive`
5. **Verify old memories** - Check freshness before trusting

---

## Best Practices

### Memory Quality

```bash
# BAD - No context
nmem remember "fixed it"

# GOOD - Rich context
nmem remember "Fixed auth bug: null email in validateUser(). Added null check at login.py:42. Took 2 hours." -t auth -t bugfix
```

### Entity Naming

```bash
# BAD - Inconsistent
nmem remember "talked to bob"
nmem remember "Bob mentioned..."
nmem remember "Robert said..."

# GOOD - Consistent
nmem remember "Meeting with Bob about AuthService"
nmem remember "Bob suggested using JWT"
nmem remember "Bob approved PR #123"
```

### Time References

```bash
# GOOD - Clear temporal context
nmem remember "Sprint 5 planning: prioritize auth refactor"
nmem remember "Yesterday's standup: blocked on API limits"
```

### Decision Records

```bash
# GOOD - Decision + rationale
nmem remember "DECISION: Using JWT instead of sessions. REASON: Stateless, scales better. ALTERNATIVE: Redis sessions" --type decision
```

### Error-Solution Pairs

```bash
# GOOD - Problem + solution together
nmem remember "ERROR: 'Cannot read id of undefined' in UserService. SOLUTION: Add null check before user.id access. Fixed in PR #456" --type error
```

---

## Quick Reference

```bash
# Store memory
nmem remember "content" [-t tag] [-T type] [-p priority] [-e expires] [-P project] [-S shared]

# Quick TODO
nmem todo "task" [-p priority] [-P project]

# Query
nmem recall "query" [-d depth] [-m max-tokens] [-c min-confidence] [-S shared]

# List
nmem list [-T type] [-p min-priority] [-P project] [--expired]

# Stats
nmem stats

# Brain management
nmem brain list | create | use | export | import | delete | health

# Project management
nmem project create | list | show | delete | extend

# Shared mode
nmem shared enable | disable | status | test | sync
```

---

## Support

- **Documentation**: https://neural-memory.github.io/neural-memory
- **Issues**: https://github.com/neural-memory/neural-memory/issues
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*NeuralMemory - Reflex-based memory for AI agents*
