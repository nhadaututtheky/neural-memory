# Python API

NeuralMemory provides a comprehensive async Python API.

## Quick Start

```python
import asyncio
from neural_memory import Brain, BrainConfig
from neural_memory.storage import InMemoryStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def main():
    # Create storage and brain
    storage = InMemoryStorage()
    brain = Brain.create("my-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # Encode memories
    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Met Alice to discuss API design")
    await encoder.encode("Decided to use FastAPI for backend")

    # Query memories
    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What was decided?")

    print(f"Answer: {result.context}")
    print(f"Confidence: {result.confidence}")

asyncio.run(main())
```

## Core Classes

### Brain

The main container for a memory system.

```python
from neural_memory import Brain, BrainConfig

# Create with default config
brain = Brain.create("my-brain")

# Create with custom config
config = BrainConfig(
    decay_rate=0.1,
    reinforcement_delta=0.05,
    activation_threshold=0.2,
    max_spread_hops=4,
    max_context_tokens=1500
)
brain = Brain.create("my-brain", config=config)

# Access properties
print(brain.id)
print(brain.name)
print(brain.config)
```

### BrainConfig

Configuration for brain behavior.

```python
from neural_memory import BrainConfig

config = BrainConfig(
    decay_rate=0.1,           # How fast memories fade
    reinforcement_delta=0.05,  # Strength gain on access
    activation_threshold=0.2,  # Minimum activation
    max_spread_hops=4,         # Max traversal depth
    max_context_tokens=1500    # Max response tokens
)
```

### Neuron

Atomic unit of information.

```python
from neural_memory import Neuron, NeuronType

neuron = Neuron(
    id="neuron-123",
    type=NeuronType.ENTITY,
    content="Alice",
    metadata={"role": "developer"}
)
```

**Neuron Types:**

- `NeuronType.TIME` - Temporal references
- `NeuronType.SPATIAL` - Locations
- `NeuronType.ENTITY` - People, things
- `NeuronType.ACTION` - Activities
- `NeuronType.STATE` - Conditions
- `NeuronType.CONCEPT` - Ideas, topics
- `NeuronType.SENSORY` - Observations
- `NeuronType.INTENT` - Goals

### Synapse

Connection between neurons.

```python
from neural_memory import Synapse, SynapseType, Direction

synapse = Synapse(
    id="synapse-456",
    source_id="neuron-a",
    target_id="neuron-b",
    type=SynapseType.CAUSED_BY,
    weight=0.8,
    direction=Direction.UNIDIRECTIONAL
)
```

**Synapse Types:**

Temporal: `HAPPENED_AT`, `BEFORE`, `AFTER`, `DURING`
Causal: `CAUSED_BY`, `LEADS_TO`, `ENABLES`, `PREVENTS`
Associative: `CO_OCCURS`, `RELATED_TO`, `SIMILAR_TO`
Semantic: `IS_A`, `HAS_PROPERTY`, `INVOLVES`

### Fiber

A signal pathway through the neural graph (a memory).

```python
from neural_memory import Fiber

# Create with factory method
fiber = Fiber.create(
    neuron_ids={"n1", "n2", "n3"},
    synapse_ids={"s1", "s2"},
    anchor_neuron_id="n1",
    pathway=["n1", "n2", "n3"],  # Ordered signal pathway
)

# Fiber properties
fiber.conductivity      # 0.0-1.0, signal transmission quality
fiber.pathway           # Ordered neuron sequence
fiber.pathway_length    # Number of neurons in pathway
fiber.last_conducted    # When last activated (datetime or None)

# Immutable operations (return new Fiber)
fiber = fiber.with_conductivity(0.8)
fiber = fiber.conduct(reinforce=True)       # +0.02 conductivity
fiber = fiber.conduct(conducted_at=now)     # Set activation time
fiber = fiber.with_salience(0.9)
fiber = fiber.add_tags("important", "reviewed")

# Pathway queries
fiber.pathway_position("n2")   # Returns 1 (index in pathway)
fiber.is_in_pathway("n2")      # Returns True
```

## Storage Backends

### InMemoryStorage

For testing and development.

```python
from neural_memory.storage import InMemoryStorage

storage = InMemoryStorage()

# Operations
await storage.add_neuron(neuron)
await storage.get_neuron("neuron-id")
await storage.find_neurons(type=NeuronType.ENTITY)
await storage.add_synapse(synapse)
await storage.get_synapses(source_id="neuron-a")
await storage.add_fiber(fiber)
await storage.get_fiber("fiber-id")
```

### SQLiteStorage

For persistent single-user storage.

```python
from neural_memory.storage import SQLiteStorage

storage = SQLiteStorage("./brain.db")
await storage.initialize()

brain = Brain.create("my-brain")
await storage.save_brain(brain)
storage.set_brain(brain.id)

# Use same API as InMemoryStorage
await storage.add_neuron(neuron)
```

### SharedStorage

For remote server connection.

```python
from neural_memory.storage import SharedStorage

async with SharedStorage("http://localhost:8000", "brain-id") as storage:
    neurons = await storage.find_neurons()
    await storage.add_neuron(neuron)
```

## Engine Components

### MemoryEncoder

Encodes text into neurons and synapses.

```python
from neural_memory.engine.encoder import MemoryEncoder

encoder = MemoryEncoder(storage, brain.config)

# Encode text
result = await encoder.encode("Met Alice at coffee shop")

print(f"Fiber: {result.fiber.id}")
print(f"Neurons created: {len(result.neurons_created)}")
print(f"Synapses created: {len(result.synapses_created)}")
```

### ReflexPipeline

Query memories using spreading activation.

```python
from neural_memory.engine.retrieval import ReflexPipeline, DepthLevel

# Hybrid mode (default) - reflex trail + classic BFS discovery
pipeline = ReflexPipeline(storage, brain.config, use_reflex=True)

# Classic mode - distance-based activation only (pre-v0.6.0 behavior)
pipeline = ReflexPipeline(storage, brain.config, use_reflex=False)

# Basic query
result = await pipeline.query("What did Alice say?")

# Query with depth
result = await pipeline.query(
    "Why did we choose PostgreSQL?",
    depth=DepthLevel.DEEP,
    max_tokens=1000
)

print(f"Context: {result.context}")
print(f"Confidence: {result.confidence}")
print(f"Neurons activated: {result.neurons_activated}")
print(f"Depth used: {result.depth_used}")
print(f"Co-activations: {len(result.co_activations)}")
```

**Depth Levels:**

- `DepthLevel.INSTANT` - 1 hop, direct answers
- `DepthLevel.CONTEXT` - 2-3 hops, surrounding context
- `DepthLevel.HABIT` - 4+ hops, patterns
- `DepthLevel.DEEP` - Full traversal, causal chains

### ReflexActivation :material-new-box:{ .new }

Trail-based activation engine (v0.6.0+).

```python
from neural_memory.engine.reflex_activation import ReflexActivation

reflex = ReflexActivation(storage, brain.config)

# Activate along fiber pathways
results = await reflex.activate_trail(
    anchor_neurons=["time_neuron_1", "entity_neuron_2"],
    fibers=fibers,
    reference_time=datetime.utcnow(),
)

# results: dict[str, ActivationResult]
for neuron_id, result in results.items():
    print(f"{neuron_id}: level={result.activation_level}, hops={result.hop_distance}")

# Combined activation with co-activation detection
activations, co_activations = await reflex.activate_with_co_binding(
    anchor_sets=[["time_1"], ["entity_1"]],
    fibers=fibers,
)
```

### CoActivation :material-new-box:{ .new }

Represents neurons that co-fired from multiple anchor sets (Hebbian binding).

```python
from neural_memory.engine.reflex_activation import CoActivation

# Created by ReflexActivation.find_co_activated()
co = CoActivation(
    neuron_ids=frozenset(["n1", "n2"]),
    temporal_window_ms=500,
    co_fire_count=3,           # Activated by 3 anchor sets
    binding_strength=1.0,      # 3/3 = perfect co-activation
    source_anchors=["time_1", "entity_1", "concept_1"],
)
```

### DecayManager

Apply memory decay (Ebbinghaus curve).

```python
from neural_memory.engine.lifecycle import DecayManager

manager = DecayManager(
    decay_rate=0.1,
    prune_threshold=0.01,
    min_age_days=1.0
)

# Apply decay
report = await manager.apply_decay(storage)

print(f"Neurons decayed: {report.neurons_decayed}")
print(f"Synapses decayed: {report.synapses_decayed}")
print(f"Neurons pruned: {report.neurons_pruned}")

# Dry run
report = await manager.apply_decay(storage, dry_run=True)
```

### ReinforcementManager

Strengthen frequently accessed paths.

```python
from neural_memory.engine.lifecycle import ReinforcementManager

manager = ReinforcementManager(reinforcement_delta=0.05)

await manager.reinforce(
    storage,
    activated_neurons=["n1", "n2"],
    activated_synapses=["s1"]
)
```

### DBTrainer

Train a brain from database schema knowledge (v1.6.0+).

```python
from neural_memory.engine.db_trainer import DBTrainer, DBTrainingConfig

# Configure training
config = DBTrainingConfig(
    connection_string="sqlite:///data/ecommerce.db",
    domain_tag="ecommerce",
    max_tables=100,
    consolidate=True,
)

# Train
trainer = DBTrainer(storage, brain.config)
result = await trainer.train(config)

print(f"Tables: {result.tables_processed}")
print(f"Relationships: {result.relationships_mapped}")
print(f"Patterns: {result.patterns_detected}")
print(f"Fingerprint: {result.schema_fingerprint}")
```

**DBTrainingConfig options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_string` | str | required | Database URI (e.g., `sqlite:///path/to/db`) |
| `domain_tag` | str | `""` | Tag applied to all schema knowledge |
| `brain_name` | str | `""` | Target brain (empty = current) |
| `consolidate` | bool | `True` | Run ENRICH consolidation after |
| `salience_ceiling` | float | `0.5` | Cap initial fiber salience |
| `initial_stage` | str | `"episodic"` | Maturation stage |
| `include_patterns` | bool | `True` | Detect schema patterns |
| `include_relationships` | bool | `True` | Create FK-based synapses |
| `max_tables` | int | `100` | Maximum tables to process |

**Schema introspection (advanced):**

```python
from neural_memory.engine.db_introspector import SchemaIntrospector

introspector = SchemaIntrospector()
snapshot = await introspector.introspect("sqlite:///data/app.db")

for table in snapshot.tables:
    print(f"{table.name}: {len(table.columns)} cols, {len(table.foreign_keys)} FKs")
print(f"Fingerprint: {snapshot.schema_fingerprint}")
```

## Typed Memories

```python
from neural_memory.core import TypedMemory, MemoryType, Priority

# Create typed memory
memory = TypedMemory.create(
    fiber_id="fiber-123",
    memory_type=MemoryType.DECISION,
    priority=Priority.HIGH,
    source="user_input",
    expires_in_days=30,
    tags={"project", "auth"}
)

await storage.add_typed_memory(memory)

# Query by type
decisions = await storage.find_typed_memories(
    memory_type=MemoryType.DECISION,
    min_priority=Priority.NORMAL
)
```

**Memory Types:**

- `MemoryType.FACT`
- `MemoryType.DECISION`
- `MemoryType.PREFERENCE`
- `MemoryType.TODO`
- `MemoryType.INSIGHT`
- `MemoryType.CONTEXT`
- `MemoryType.INSTRUCTION`
- `MemoryType.ERROR`
- `MemoryType.WORKFLOW`
- `MemoryType.REFERENCE`

## Projects

```python
from neural_memory.core import Project

# Create project
project = Project.create(
    name="Q1 Sprint",
    description="First quarter sprint",
    duration_days=14,
    tags={"sprint", "q1"}
)
await storage.add_project(project)

# Associate memories
memory = TypedMemory.create(
    fiber_id="fiber-123",
    memory_type=MemoryType.TODO,
    project_id=project.id
)

# Query project memories
memories = await storage.get_project_memories(project.id)
```

## Export & Import

```python
from neural_memory.sharing import BrainExporter

exporter = BrainExporter()

# Export
snapshot = await exporter.export(storage, brain.id)
json_data = exporter.to_json(snapshot)

# Save to file
with open("brain.json", "w") as f:
    f.write(json_data)

# Import
from neural_memory.sharing import BrainImporter

importer = BrainImporter()
await importer.import_brain(storage, snapshot, "new-brain-id")
```

## Complete Example

```python
import asyncio
from neural_memory import Brain, BrainConfig
from neural_memory.storage import SQLiteStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline, DepthLevel
from neural_memory.engine.lifecycle import DecayManager

async def main():
    # Setup
    storage = SQLiteStorage("./memories.db")
    await storage.initialize()

    brain = Brain.create("work", config=BrainConfig(
        max_context_tokens=2000
    ))
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage, brain.config)
    pipeline = ReflexPipeline(storage, brain.config)

    # Store memories
    await encoder.encode("Met with team to discuss architecture")
    await encoder.encode("DECISION: Use microservices. REASON: Better scaling")
    await encoder.encode("TODO: Set up CI/CD pipeline")

    # Query
    result = await pipeline.query(
        "What architecture decision was made?",
        depth=DepthLevel.CONTEXT
    )
    print(f"Answer: {result.context}")

    # Apply decay (for old memories)
    decay_manager = DecayManager()
    report = await decay_manager.apply_decay(storage, dry_run=True)
    print(f"Would decay {report.neurons_decayed} neurons")

asyncio.run(main())
```
