# How NeuralMemory Works

NeuralMemory uses a fundamentally different approach to memory retrieval than traditional search or RAG systems.

## The Core Idea

**Human memory doesn't work like search.**

You don't query your brain with:
```sql
SELECT * FROM memories WHERE content LIKE '%Alice%' ORDER BY similarity DESC
```

Instead, thinking of "Alice" *activates* related memories - her face, your last conversation, the project you worked on together. These emerge through **association**, not **search**.

NeuralMemory replicates this process:

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

## Key Components

### Neurons

Neurons are atomic units of information:

- **Entity neurons** - People, places, things ("Alice", "coffee shop")
- **Time neurons** - Temporal references ("Tuesday 3pm", "last week")
- **Concept neurons** - Ideas, topics ("authentication", "rate limiting")
- **Action neurons** - What happened ("discussed", "decided", "fixed")
- **State neurons** - Conditions ("blocked", "completed", "urgent")

### Synapses

Synapses are typed connections between neurons:

- **Temporal** - `HAPPENED_AT`, `BEFORE`, `AFTER`
- **Causal** - `CAUSED_BY`, `LEADS_TO`, `ENABLES`
- **Associative** - `RELATED_TO`, `CO_OCCURS`
- **Semantic** - `IS_A`, `HAS_PROPERTY`, `INVOLVES`

### Fibers

Fibers are **signal pathways** through the neural graph - ordered sequences of neurons that form a coherent memory. Each fiber has a **conductivity** (0.0-1.0) that determines how well signals travel through it:

```
Fiber: "Meeting with Alice" (conductivity: 0.95)
Pathway: [Tuesday 3pm] → [Alice] → [Meeting] → [API design] → [Rate limiting]
├── [Alice] ←DISCUSSED→ [API design]
├── [Coffee shop] ←AT_LOCATION→ [Meeting]
├── [Tuesday 3pm] ←HAPPENED_AT→ [Meeting]
└── [Rate limiting] ←SUGGESTED_BY→ [Alice]
```

Frequently-accessed fibers develop higher conductivity, making their memories easier to recall - similar to how neural pathways strengthen with use in the biological brain.

## Encoding Process

When you store a memory:

```bash
nmem remember "Met Alice at coffee shop to discuss API design, she suggested rate limiting"
```

NeuralMemory:

1. **Extracts entities** - Alice, coffee shop, API design, rate limiting
2. **Extracts temporal context** - (uses current time if not specified)
3. **Identifies relationships** - Alice DISCUSSED API design, Alice SUGGESTED rate limiting
4. **Creates neurons** - One for each entity/concept
5. **Creates synapses** - Typed connections between neurons
6. **Bundles into fiber** - Groups everything into a coherent memory

## Retrieval Process

When you query:

```bash
nmem recall "What did Alice suggest last Tuesday?"
```

NeuralMemory (reflex mode, default in v0.6.0+):

1. **Parses query** - Identifies "last Tuesday" as time hint, "Alice" as entity, "suggest" as action hint
2. **Finds anchors (time-first)** - Locates time neurons first (weight 1.0), then entities (0.8), then actions (0.6)
3. **Finds fibers** - Gets fiber pathways containing anchor neurons
4. **Trail activation** - Spreads signals along fiber pathways with conductivity and time decay
5. **Co-activation detection** - Neurons reached by multiple anchor sets get binding strength boost
6. **Extracts subgraph** - Gets highest-scoring connected cluster
7. **Reinforces fibers** - Accessed fibers get conductivity boost (+0.02)
8. **Reconstructs answer** - "Alice suggested rate limiting"

## Activation Dynamics

### Reflex Mode (v0.6.0+, default)

Activation spreads along **fiber pathways** with trail decay:

```
activation(next) = current * (1 - decay) * synapse_weight * conductivity * time_factor
```

Neurons co-activated by multiple anchor sets receive Hebbian binding boost:

```
[Tuesday] ──fiber──► [Meeting] ◄──fiber── [Alice]
                         │
              co-activated (binding=1.0)
                         │
                    [BEST RESULT]
```

### Classic Mode

Distance-based decay through BFS:

```
activation(hop) = initial * decay_factor^hop
```

## Depth Levels

Different queries need different exploration depths:

| Level | Name | Hops | Use Case |
|-------|------|------|----------|
| 0 | Instant | 1 | Who, what, where |
| 1 | Context | 2-3 | Before/after context |
| 2 | Habit | 4+ | Cross-time patterns |
| 3 | Deep | Full | Causal chains, emotions |

## Memory Lifecycle

Memories evolve over time:

### Decay

Unused memories weaken following the Ebbinghaus forgetting curve:

```
activation = initial * e^(-decay_rate * days)
```

### Reinforcement

Frequently accessed memories strengthen (Hebbian learning):

```
When recalled: synapse.weight += reinforcement_delta
When fiber activated: fiber.conductivity += 0.02  (capped at 1.0)
```

### Compression

Old memories can be summarized:

```
Original: [20 detailed neurons about Tuesday meeting]
Compressed: [1 summary neuron: "API design meeting with Alice"]
```

## Comparison with RAG

| Aspect | RAG | NeuralMemory |
|--------|-----|--------------|
| Data model | Flat chunks | Neural graph |
| Retrieval | Similarity search | Spreading activation |
| Relationships | Implicit | Explicit typed synapses |
| Temporal | Metadata filter | First-class neurons |
| Multi-hop | Multiple queries | Single traversal |
| Memory lifecycle | Static | Dynamic decay/reinforce |

## Training from External Sources

Beyond encoding individual memories, NeuralMemory can learn domain knowledge from structured sources.

### Database Schema Training (v1.6.0+)

NeuralMemory can learn to understand database structure by training from schema metadata:

```
SQLite Database
    │
    ▼
┌─────────────────────┐
│ SchemaIntrospector   │  Extract tables, columns, FKs, indexes
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ KnowledgeExtractor   │  Map FKs → synapse types, detect patterns
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ DBTrainer            │  Batch encode into brain neurons + synapses
└─────────────────────┘
```

**What it learns:**

- Table entities as CONCEPT neurons with semantic descriptions
- FK relationships as typed synapses (IS_A, INVOLVES, AT_LOCATION, RELATED_TO)
- Schema patterns: audit trails, soft deletes, tree hierarchies, polymorphic types, enum tables
- Join tables become direct CO_OCCURS synapses (no separate entity node)

**What it does NOT import:** Raw data rows. Only structural knowledge.

This enables queries like:
- "How are orders related to customers?" → Traces FK relationships
- "Which tables have audit trails?" → Recalls detected patterns
