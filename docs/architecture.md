# Architecture & Scalability

This document covers NeuralMemory's architecture and future scalability paths.

---

## Current Architecture (v0.1.x)

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI / MCP Server                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Encoder    │  │  Retrieval   │  │   Lifecycle  │       │
│  │              │  │   Pipeline   │  │   Manager    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    Storage Interface                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  In-Memory   │  │   SQLite     │  │   Shared     │       │
│  │  (NetworkX)  │  │  (Default)   │  │   (HTTP)     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Storage Backends

| Backend | Status | Use Case |
|---------|--------|----------|
| `InMemoryStorage` | ✅ Ready | Testing, small datasets |
| `SQLiteStorage` | ✅ Ready | **Default** - Personal use, single machine |
| `PersistentStorage` | ✅ Ready | CLI wrapper for SQLite |
| `SharedStorage` | ✅ Ready | Remote server connection |
| `Neo4jStorage` | 🔲 Interface only | Production, large scale |

### Performance Characteristics (Current)

| Metric | SQLite Backend | Notes |
|--------|----------------|-------|
| Neurons | Up to ~100,000 | Comfortable |
| Query latency | 10-50ms | Typical |
| Memory usage | ~100MB per 10k neurons | Estimate |
| Concurrent users | 1 | SQLite limitation |

---

## Scalability Paths

### Path 1: Neo4j Backend (Interface Ready)

For scaling beyond 100k neurons or needing concurrent access.

**Interface exists at:** `src/neural_memory/storage/neo4j_store.py`

```python
# Interface defined, implementation needed
class Neo4jStorage(NeuralStorage):
    async def add_neuron(self, neuron: Neuron) -> str: ...
    async def get_neighbors(self, neuron_id: str, ...) -> list: ...
    # ... full NeuralStorage interface
```

**When to implement:**
- Need >100k neurons
- Need concurrent multi-user access
- Need complex graph queries (Cypher)

**Effort estimate:** 2-3 weeks

### Path 2: Rust Extensions (Not Started)

For CPU-intensive operations at massive scale.

**Candidates for Rust optimization:**
1. Spreading activation algorithm
2. Graph traversal
3. Similarity computation

**When to consider:**
- Need >1M neurons
- Need <5ms query latency
- Batch processing without LLM calls

**Approach:** Use PyO3 for Python bindings

```rust
// Future: src/neural_memory_core/src/activation.rs
#[pyfunction]
fn spread_activation(
    graph: &PyGraph,
    anchors: Vec<String>,
    max_hops: usize,
) -> PyResult<HashMap<String, f64>> {
    // Rust implementation
}
```

**Effort estimate:** 4-6 weeks

### Path 3: Distributed Architecture (Future)

For multi-region, high-availability deployments.

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                           │
├──────────────────┬──────────────────┬───────────────────────┤
│   API Server 1   │   API Server 2   │    API Server N       │
├──────────────────┴──────────────────┴───────────────────────┤
│                    Message Queue (Redis)                     │
├──────────────────┬──────────────────┬───────────────────────┤
│  Neo4j Primary   │  Neo4j Replica   │   Neo4j Replica       │
└──────────────────┴──────────────────┴───────────────────────┘
```

**When to consider:**
- SaaS offering
- Enterprise deployment
- 99.9% uptime requirement

**Effort estimate:** 2-3 months

---

## Interface Contracts

### NeuralStorage (Abstract Base)

All storage backends must implement this interface:

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
    async def import_brain(self, snapshot: BrainSnapshot, brain_id: str) -> None
```

**Location:** `src/neural_memory/storage/base.py`

### Why Interface-First Design?

1. **Swap backends** without changing application code
2. **Test with InMemory**, deploy with SQLite/Neo4j
3. **Future-proof** for scaling needs

---

## Roadmap

### v0.1.x (Current)
- [x] Core data structures
- [x] SQLite storage
- [x] CLI with full features
- [x] MCP server for Claude
- [x] Real-time sharing (WebSocket)
- [x] Typed memories with expiry

### v0.2.x (Planned)
- [ ] Neo4j storage backend
- [ ] Vector embeddings (optional semantic layer)
- [ ] Memory compression for old fibers
- [ ] Decay/reinforcement system

### v0.3.x (Future)
- [ ] Multi-tenancy
- [ ] Admin dashboard
- [ ] Brain marketplace (share/import)

### v1.0.x (Long-term)
- [ ] Distributed architecture
- [ ] Rust core (if needed)
- [ ] Enterprise features

---

## Contributing to Scalability

If you want to help implement scalability features:

1. **Neo4j Backend** - Good first contribution
   - Implement `Neo4jStorage` class
   - Follow existing `SQLiteStorage` patterns
   - Add integration tests

2. **Benchmarking** - Help needed
   - Create benchmark suite
   - Test with various dataset sizes
   - Identify actual bottlenecks

3. **Rust Extensions** - Advanced
   - Requires PyO3 experience
   - Start with spreading activation
   - Must maintain Python API compatibility

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
