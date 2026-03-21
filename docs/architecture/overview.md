# Architecture Overview

NeuralMemory's layered architecture for memory management.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / MCP Server / REST API               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Encoder    │  │  Retrieval   │  │   Lifecycle  │       │
│  │              │  │   Pipeline   │  │   Manager    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  DocTrainer  │  │  DBTrainer   │  │Consolidation │       │
│  │              │  │              │  │   Engine     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Context     │  │  Query       │  │  Alert       │       │
│  │  Optimizer   │  │  Patterns    │  │  Handler     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    Extraction Layer                          │
│  ┌───────────┐  ┌────────────┐  ┌──────────────────┐        │
│  │QueryParser│  │QueryRouter │  │TemporalExtractor │        │
│  └───────────┘  └────────────┘  └──────────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Storage Interface                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐        │
│  │InMemory   │  │SQLite     │  │SharedStorage(HTTP)│        │
│  └───────────┘  └───────────┘  └───────────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Core Layer                                │
│  ┌──────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌───────────┐      │
│  │Neuron│  │Synapse│  │Fiber│  │Brain│  │TypedMemory│      │
│  └──────┘  └───────┘  └─────┘  └─────┘  └───────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Layers

### Interface Layer

Entry points for users and applications:

- **CLI** - Command-line interface (`nmem` commands)
- **MCP Server** - Model Context Protocol for Claude integration
- **REST API** - FastAPI-based HTTP server

### Engine Layer

Core processing components:

- **MemoryEncoder** - Converts text to neural structures
- **ReflexPipeline** - Query processing with spreading activation
- **ReflexActivation** - Trail-based activation through fiber pathways (v0.6.0+)
- **SpreadingActivation** - Classic distance-based activation
- **LifecycleManager** - Decay, reinforcement, compression

### Training Layer

Specialized pipelines for training brains from external sources:

- **DocTrainer** - Train from documents (chunking, encoding, consolidation)
- **DBTrainer** - Train from database schemas (introspection → knowledge extraction → encoding)
- **SchemaIntrospector** - Database schema metadata extraction (SQLite dialect)
- **KnowledgeExtractor** - Transform schema snapshots into confidence-scored teachable knowledge
- **ConsolidationEngine** - ENRICH, DREAM, LEARN_HABITS, MATURE, INFER, PRUNE strategies; auto-runs MATURE+INFER+ENRICH on MCP shutdown (B1)
- **ContextOptimizer** - Composite scoring, SimHash dedup, token budget allocation for `nmem_context`
- **QueryPatternMining** - Topic co-occurrence mining from recall events, pattern materialization
- **AlertHandler** - Persistent alert lifecycle (active → seen → acknowledged → resolved)

### Extraction Layer

NLP and parsing utilities:

- **QueryParser** - Decomposes queries into signals
- **QueryRouter** - Determines query intent and depth
- **TemporalExtractor** - Extracts time references

### Storage Layer

Pluggable storage backends:

- **InMemoryStorage** - NetworkX-based, for testing
- **SQLiteStorage** - Persistent, single-user
- **SharedStorage** - HTTP client for remote server

### Core Layer

Fundamental data structures:

- **Neuron** - Atomic information unit
- **Synapse** - Typed connection between neurons
- **Fiber** - Signal pathway with conductivity
- **Brain** - Container with configuration
- **TypedMemory** - Metadata layer for memories

## Data Flow

### Encoding Flow

```
Input Text
    │
    ▼
┌─────────────────┐
│  QueryParser    │  Extract entities, time, concepts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MemoryEncoder  │  Create neurons and synapses
│                 │  · CreateSynapsesStep: IDF-weighted keywords (B4)
│                 │  · CrossMemoryLinkStep: RELATED_TO via shared entities (B3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Storage        │  Persist to graph (schema v35)
└─────────────────┘
```

### Retrieval Flow

```
Query
    │
    ▼
┌─────────────────┐
│  QueryParser    │  Decompose query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QueryRouter    │  Determine depth, intent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Find Anchors   │  Time-first anchor selection
│  (Time-First)   │  Time(1.0) → Entity(0.8) → Action(0.6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Find Fibers    │  Get fiber pathways for anchors
└────────┬────────┘
         │
         ├─── reflex mode ──┐
         │                   ▼
         │          ┌─────────────────┐
         │          │  Trail          │  Activate along fiber
         │          │  Activation     │  pathways with decay
         │          └────────┬────────┘
         │                   │
         │                   ▼
         │          ┌─────────────────┐
         │          │  Co-Activation  │  Hebbian binding
         │          └────────┬────────┘
         │                   │
         ├─── classic mode ──┤
         │                   │
         │          ┌─────────────────┐
         │          │  Spreading      │  BFS with decay
         │          │  Activation     │
         │          └────────┬────────┘
         │                   │
         ├───────────────────┘
         │
         ▼
┌─────────────────┐
│  Score Fibers   │  Fiber-level recall scoring (B5)
│                 │  base_quality × activation_signal × stage_multiplier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract        │  Build response context
│  Subgraph       │  · Contextual compression by age (B6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reinforce      │  Update fiber conductivity
│  Fibers         │  · Adaptive synapse decay (B8)
└─────────────────┘
```

## Storage Interface

All storage backends implement `NeuralStorage`:

```python
class NeuralStorage(ABC):
    # Neuron operations
    async def add_neuron(self, neuron: Neuron) -> str
    async def get_neuron(self, neuron_id: str) -> Neuron | None
    async def find_neurons(self, **filters) -> list[Neuron]

    # Synapse operations
    async def add_synapse(self, synapse: Synapse) -> str
    async def get_synapses(self, **filters) -> list[Synapse]

    # Graph traversal
    async def get_neighbors(self, neuron_id: str, ...) -> list[tuple]

    # Fiber operations
    async def add_fiber(self, fiber: Fiber) -> str
    async def get_fiber(self, fiber_id: str) -> Fiber | None

    # Brain operations
    async def export_brain(self, brain_id: str) -> BrainSnapshot
    async def import_brain(self, snapshot: BrainSnapshot, brain_id: str)
```

Benefits:

- Swap backends without changing application code
- Test with InMemory, deploy with SQLite/Neo4j
- Future-proof for scaling needs

## Configuration

### Brain Configuration

```python
@dataclass
class BrainConfig:
    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
```

### CLI Configuration

Stored in `~/.neural-memory/config.toml`:

```toml
[brain]
default = "default"
decay_rate = 0.1

[auto]
min_confidence = 0.7
detect_decisions = true
detect_errors = true

[shared]
enabled = false
url = ""
api_key = ""
```

## File Structure

```
~/.neural-memory/
├── config.toml           # User configuration
├── brains/
│   ├── default.db        # SQLite brain database
│   ├── work.db
│   └── personal.db
└── cache/
    └── ...
```

## Module Organization

```
src/neural_memory/
├── __init__.py                # Public API exports
├── py.typed                   # PEP 561 marker
├── core/
│   ├── brain.py               # Brain, BrainConfig
│   ├── brain_mode.py          # BrainMode, SharedConfig
│   ├── neuron.py              # Neuron, NeuronType, NeuronState
│   ├── synapse.py             # Synapse, SynapseType, Direction
│   ├── fiber.py               # Fiber (with pathway, conductivity)
│   ├── memory_types.py        # TypedMemory, MemoryType, Priority
│   ├── alert.py               # Alert, AlertType, AlertStatus
│   └── project.py             # Project
├── engine/
│   ├── encoder.py             # MemoryEncoder
│   ├── retrieval.py           # ReflexPipeline
│   ├── retrieval_types.py     # DepthLevel, Subgraph, RetrievalResult
│   ├── retrieval_context.py   # reconstitute_answer, format_context
│   ├── activation.py          # ActivationResult, SpreadingActivation
│   ├── reflex_activation.py   # ReflexActivation, CoActivation
│   ├── lifecycle.py              # DecayManager, ReinforcementManager
│   ├── db_introspector.py        # Database schema introspection
│   ├── db_knowledge.py           # Schema → teachable knowledge extraction
│   ├── db_trainer.py             # DB-to-Brain training orchestrator
│   ├── doc_chunker.py            # Document chunking
│   ├── doc_trainer.py            # Doc-to-Brain training pipeline
│   ├── context_optimizer.py      # Smart context scoring + dedup + budgeting
│   └── query_pattern_mining.py   # Recall topic co-occurrence mining
├── extraction/
│   ├── parser.py              # QueryParser, Stimulus
│   ├── router.py              # QueryRouter
│   ├── entities.py            # EntityExtractor
│   ├── keywords.py            # extract_keywords, STOP_WORDS
│   └── temporal.py            # TemporalExtractor
├── storage/
│   ├── base.py                # NeuralStorage ABC
│   ├── sqlite_store.py        # SQLiteStorage (core)
│   ├── sqlite_schema.py       # Schema DDL and migrations
│   ├── sqlite_row_mappers.py  # Row-to-object converters
│   ├── sqlite_neurons.py      # Neuron CRUD mixin
│   ├── sqlite_synapses.py     # Synapse CRUD mixin
│   ├── sqlite_fibers.py       # Fiber CRUD mixin
│   ├── sqlite_typed.py        # TypedMemory/Project mixin
│   ├── sqlite_projects.py     # Project CRUD mixin
│   ├── sqlite_brain_ops.py    # Brain import/export mixin
│   ├── sqlite_alerts.py       # Proactive alerts CRUD mixin
│   ├── memory_store.py        # InMemoryStorage (core)
│   ├── memory_brain_ops.py    # InMemory brain operations
│   ├── memory_collections.py  # InMemory neuron/synapse/fiber ops
│   ├── shared_store.py        # SharedStorage HTTP client (core)
│   ├── shared_store_mappers.py    # dict_to_* converters
│   └── shared_store_collections.py # Fiber/brain HTTP mixin
├── server/
│   ├── app.py                 # FastAPI application
│   ├── routes/                # API route handlers
│   └── models.py              # Pydantic models
├── mcp/
│   ├── server.py              # MCP server implementation
│   ├── tool_schemas.py        # MCP tool definitions
│   ├── auto_capture.py        # Pattern-based memory detection
│   ├── db_train_handler.py       # Database schema training handler
│   ├── alert_handler.py          # Proactive alerts lifecycle
│   └── prompt.py              # System prompts
├── cli/
│   ├── main.py                # Entry point, app registration
│   └── commands/              # One file per command group
├── safety/                    # Freshness, sensitive content
├── sync/                      # Shared brain sync client
└── utils/                     # Shared utilities

vscode-extension/
├── src/
│   ├── extension.ts           # Entry point
│   ├── commands/              # Command handlers
│   ├── editors/               # CodeLens, decorations
│   ├── server/                # HTTP client, WebSocket, lifecycle
│   └── views/
│       ├── tree/              # MemoryTreeProvider
│       └── graph/
│           ├── GraphPanel.ts      # Panel controller
│           └── graphTemplate.ts   # Cytoscape.js HTML template
└── test/                      # Unit, integration, perf tests
```
