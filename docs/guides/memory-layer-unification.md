# Memory Layer Unification

NeuralMemory acts as a **unification layer** that imports, normalizes, and connects memories from multiple external memory systems into a single neural graph. Instead of managing fragmented knowledge across Mem0, Cognee, ChromaDB, Graphiti, LlamaIndex, and AWF separately, NeuralMemory brings them together — preserving relationships, embeddings, and provenance while enabling cross-system recall via spreading activation.

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   ChromaDB   │  │     Mem0     │  │    Cognee    │
│  (vectors)   │  │  (memories)  │  │   (graphs)   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ChromaDB     │  │  Mem0        │  │  Cognee      │
│ Adapter      │  │  Adapter     │  │  Adapter     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────────┬────┴────┬────────────┘
                    ▼         ▼
              ┌─────────────────────┐
              │   ExternalRecord    │  ← Universal intermediate format
              │  (normalized data)  │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │    RecordMapper     │  ← Type resolution + encoding
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │    SyncEngine       │  ← Batch import + dedup + relationships
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │   NeuralMemory      │  ← Neurons, synapses, fibers
              │   (unified brain)   │
              └─────────────────────┘
```

### Pipeline stages

1. **SourceAdapter** fetches raw records from the external system
2. **ExternalRecord** normalizes content, metadata, embeddings, and relationships into a universal format
3. **RecordMapper** runs each record through the NeuralMemory encoder (creating neurons, synapses, fibers) and assigns TypedMemory with provenance
4. **SyncEngine** orchestrates the full pipeline — batched commits, deduplication, cross-record relationship synapses, and sync state tracking

## Supported Systems

| System | Type | Adapter | Install Extra | Capabilities |
|--------|------|---------|---------------|-------------|
| **ChromaDB** | Vector DB | `ChromaDBAdapter` | `pip install neural-memory[chromadb]` | fetch_all, embeddings, metadata, health |
| **Mem0** | Memory Layer | `Mem0Adapter` | `pip install neural-memory[mem0]` | fetch_all, fetch_since, metadata, health |
| **Cognee** | Graph Store | `CogneeAdapter` | `pip install neural-memory[cognee]` | fetch_all, relationships, metadata, health |
| **Graphiti** | Graph Store | `GraphitiAdapter` | `pip install neural-memory[graphiti]` | fetch_all, fetch_since, relationships, metadata, health |
| **LlamaIndex** | Index Store | `LlamaIndexAdapter` | `pip install neural-memory[llamaindex]` | fetch_all, embeddings, metadata, health |
| **AWF** | File Store | `AWFAdapter` | *(built-in)* | fetch_all, metadata, health |

## Quick Start

### Import from ChromaDB

```python
from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.integration.adapters import get_adapter
from neural_memory.integration.sync_engine import SyncEngine
from neural_memory.storage.sqlite_store import SQLiteStorage

# Setup storage
storage = SQLiteStorage("unified.db")
brain = Brain.create(name="unified")
await storage.save_brain(brain)
storage.set_brain(brain.id)

# Import from ChromaDB
adapter = get_adapter("chromadb", path="/path/to/chroma")
engine = SyncEngine(storage, brain.config)
result, state = await engine.sync(adapter, collection="my_docs")

print(f"Imported {result.records_imported} records from ChromaDB")
```

### Import from Mem0

```python
adapter = get_adapter("mem0", api_key="your-api-key", user_id="user-123")
result, state = await engine.sync(adapter)

print(f"Imported {result.records_imported} memories from Mem0")
```

### Import from Cognee

```python
adapter = get_adapter("cognee", api_key="your-api-key")
result, state = await engine.sync(adapter, collection="my_dataset")

# Cognee graph edges become synapses in NeuralMemory
print(f"Imported {result.records_imported} knowledge nodes")
```

### Import from Graphiti

```python
adapter = get_adapter("graphiti", uri="bolt://localhost:7687")
result, state = await engine.sync(adapter)

# Graphiti's bi-temporal edges get temporal weight scaling
print(f"Imported {result.records_imported} graph episodes")
```

### Import from LlamaIndex

```python
adapter = get_adapter("llamaindex", persist_dir="/path/to/index")
result, state = await engine.sync(adapter)

# Or with a live index object:
# adapter = get_adapter("llamaindex", index=my_live_index)

print(f"Imported {result.records_imported} index nodes")
```

### Import from AWF (.brain/)

```python
adapter = get_adapter("awf", brain_dir="/path/to/.brain")
result, state = await engine.sync(adapter)

# AWF 3-tier context (project info, session state, snapshots)
print(f"Imported {result.records_imported} context entries")
```

## Multi-System Unification

Import from multiple systems into one brain:

```python
storage = SQLiteStorage("unified.db")
brain = Brain.create(name="all-memory")
await storage.save_brain(brain)
storage.set_brain(brain.id)

engine = SyncEngine(storage, brain.config)

# Import from all sources
sources = [
    get_adapter("chromadb", path="/data/chroma"),
    get_adapter("mem0", api_key="...", user_id="user-1"),
    get_adapter("cognee", api_key="..."),
]

for adapter in sources:
    result, state = await engine.sync(adapter)
    print(f"{adapter.system_name}: {result.records_imported} imported")
```

After import, all memories are queryable through a single recall interface. Spreading activation finds connections across systems — a Mem0 memory about "PostgreSQL" will link to a ChromaDB document about database migrations and a Cognee knowledge node about SQL optimization.

## How Data Flows

### ExternalRecord — The Universal Format

Every adapter produces `ExternalRecord` instances with a common shape:

```python
@dataclass(frozen=True)
class ExternalRecord:
    id: str                      # Unique ID in the source system
    source_system: str           # "chromadb", "mem0", "cognee", etc.
    source_collection: str       # Collection/namespace within the source
    content: str                 # The memory text
    created_at: datetime         # When it was created
    source_type: str | None      # Type hint ("fact", "document", "entity")
    metadata: dict[str, Any]     # Original metadata preserved
    embedding: list[float] | None  # Vector embedding (from vector DBs)
    tags: frozenset[str]         # Source tags + "import:{system}"
    relationships: tuple[...]    # Graph edges (from graph stores)
```

### Type Resolution

External type strings map automatically to NeuralMemory types:

| Source Type | NeuralMemory Type | Source Systems |
|-------------|-------------------|----------------|
| `fact`, `memory`, `note` | `FACT` | Mem0, generic |
| `document`, `code`, `text_node` | `REFERENCE` | ChromaDB, LlamaIndex |
| `knowledge`, `concept` | `INSIGHT` | Cognee |
| `entity`, `relationship` | `FACT` | Graphiti |
| `episode` | `CONTEXT` | Graphiti |
| `decision` | `DECISION` | AWF, generic |
| `error` | `ERROR` | AWF, generic |

If no explicit type is provided, NeuralMemory infers one from content using keyword heuristics.

### Relationship Preservation

Graph systems (Cognee, Graphiti, LlamaIndex) export relationships that become synapses:

```
External: (nodeA) --[caused_by]--> (nodeB)
    ↓
NeuralMemory: (neuronA) --[CAUSED_BY, weight=0.5]--> (neuronB)
```

Supported relationship mappings: `related_to`, `similar_to`, `caused_by`, `leads_to`, `is_a`, `has_property`, `involves`, `before`, `after`, `co_occurs`, `at_location`, `contains`, `enables`, `prevents`.

### Embedding Preservation

Vector DBs (ChromaDB, LlamaIndex) export embeddings that are stored as metadata on anchor neurons. This preserves the original vector representation for potential hybrid search.

### Provenance Tracking

Every imported record carries metadata tracing back to its origin:

```python
{
    "import_source": "chromadb",
    "import_collection": "my_docs",
    "import_record_id": "chroma-uuid-123",
    # Original metadata prefixed with src_
    "src_author": "alice",
    "src_category": "tutorial",
}
```

Tags also track provenance: `import:chromadb`, `collection:my_docs`.

## Incremental Sync

Systems that support temporal queries (Mem0, Graphiti) enable incremental sync:

```python
# First sync — full import
result, state = await engine.sync(adapter)

# Later — only new records since last sync
result, state = await engine.sync(adapter, sync_state=state)

print(f"Incremental: {result.records_imported} new records")
```

`SyncState` tracks `last_sync_at`, `records_imported`, and `last_record_id` per source/collection pair. Systems without `FETCH_SINCE` capability always do a full fetch with deduplication.

## Health Checks

Verify connectivity before syncing:

```python
health = await engine.health_check(adapter)
if health["healthy"]:
    result, state = await engine.sync(adapter)
else:
    print(f"Source unhealthy: {health['message']}")
```

## Progress Tracking

Monitor long-running imports:

```python
def on_progress(processed: int, total: int, record_id: str) -> None:
    pct = (processed / total) * 100
    print(f"[{pct:.0f}%] Imported {record_id}")

result, state = await engine.sync(adapter, progress_callback=on_progress)
```

## Writing a Custom Adapter

Implement the `SourceAdapter` protocol to connect any memory system:

```python
from neural_memory.integration.adapter import SourceAdapter
from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)

class MyCustomAdapter:
    """Adapter for MyMemorySystem."""

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.MEMORY_LAYER

    @property
    def system_name(self) -> str:
        return "my-system"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset({
            SourceCapability.FETCH_ALL,
            SourceCapability.HEALTH_CHECK,
        })

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        # Fetch from your system and normalize
        raw_items = await my_system.get_all()
        return [
            ExternalRecord.create(
                id=item.id,
                source_system="my-system",
                content=item.text,
                source_type="fact",
                tags={"import:my-system"},
            )
            for item in raw_items
        ]

    async def fetch_since(self, since, collection=None, limit=None):
        raise NotImplementedError

    async def health_check(self) -> dict:
        return {"healthy": True, "message": "OK", "system": "my-system"}
```

Register and use it:

```python
from neural_memory.integration.adapters import register_adapter

register_adapter("my-system", MyCustomAdapter)
adapter = get_adapter("my-system")
result, state = await engine.sync(adapter)
```

## Adapter Details

### ChromaDB

Connects to a ChromaDB vector database (local or remote). Preserves embeddings, metadata, and document content.

- **Local**: `ChromaDBAdapter(path="/path/to/chroma")`
- **Remote**: `ChromaDBAdapter(host="localhost", port=8000)`
- Iterates all collections (or a specific one via `collection` parameter)
- Embeddings stored on anchor neuron metadata
- Tags: `import:chromadb`, `collection:{name}`

### Mem0

Connects to Mem0's memory API. Supports incremental sync via `updated_at` timestamps.

- `Mem0Adapter(api_key="...", user_id="user-123")`
- Fetches all memories for a user via `mem0.get_all()`
- Supports `fetch_since()` for incremental updates
- Maps Mem0 categories to NeuralMemory types
- Tags: `import:mem0`, `user:{user_id}`

### Cognee

Connects to Cognee's knowledge graph API. Extracts both nodes and edges.

- `CogneeAdapter(api_key="...")`
- Fetches cognified datasets as knowledge nodes
- Graph edges become `ExternalRelationship` instances → NeuralMemory synapses
- Tags: `import:cognee`, `collection:{dataset}`

### Graphiti

Connects to Graphiti's temporal knowledge graph (Neo4j-backed). Bi-temporal edges get temporal weight scaling.

- `GraphitiAdapter(uri="bolt://localhost:7687")`
- Supports `fetch_since()` for incremental episode sync
- Temporal weight computation: recent edges weighted higher
- Entity nodes + episode nodes imported with their relationships
- Tags: `import:graphiti`, `episode:{id}`, `entity:{name}`

### LlamaIndex

Imports nodes from a persisted or live LlamaIndex index. Preserves parent/child/next/previous node relationships.

- `LlamaIndexAdapter(persist_dir="/path/to/index")` or `LlamaIndexAdapter(index=live_index)`
- Extracts text nodes, embeddings, and node relationships
- Parent/child/next/previous relationships → synapses
- Does **not** support incremental sync
- Tags: `import:llamaindex`, `collection:{name}`

### AWF (Antigravity Workflow Framework)

Reads context from AWF's `.brain/` directory structure (JSON files on disk).

- `AWFAdapter(brain_dir="/path/to/.brain")`
- **Tier 1**: `brain.json` — project info, tech stack, key decisions
- **Tier 2**: `session.json` — working state, errors, conversation summaries
- **Tier 3**: `snapshots/` — historical snapshots
- No external dependencies (built-in)
- Tags: `import:awf`, `tier:1`/`tier:2`/`tier:3`

## Why Unify?

| Problem | Without Unification | With NeuralMemory |
|---------|--------------------|--------------------|
| **Fragmented recall** | Query each system separately | Single `nmem recall` searches all |
| **No cross-system links** | ChromaDB docs don't connect to Mem0 memories | Spreading activation finds connections |
| **Duplicate knowledge** | Same fact stored in 3 places | Deduplication at import time |
| **No decay/consolidation** | Stale memories persist forever | Lifecycle engine prunes and consolidates |
| **No associative inference** | Static storage | Co-activation → inferred synapses |
| **Type inconsistency** | Each system has its own schema | Unified TypedMemory with provenance |
