# NeuralMemory Roadmap

> From associative reflex engine to universal memory platform.
> Every feature passes the VISION.md 4-question test + brain test.
> ZERO LLM dependency — pure algorithmic, regex, graph-based.

**Current state**: v1.0.2 shipped (schema v11). v1.1.0 in progress.
**Next milestone**: v1.2.0 — Dashboard Foundation (Alpine.js SPA + Cytoscape.js).

---

# Part I: v0.14.0 → v1.0.0 (COMPLETE)

**Status**: All versions shipped.
v0.14.0 shipped: relation extraction, tag origin, confirmatory boost.
v0.15.0 shipped: associative inference, co-activation persistence, tag normalization.
v0.16.0 shipped: emotional valence, sentiment extraction, FELT synapses, emotional decay.
v0.17.0 shipped: brain diagnostics, purity score, nmem_health MCP tool + CLI.
v0.19.0 shipped: temporal reasoning, causal chain traversal, event sequence tracing.
v0.20.0 shipped: habitual recall — ENRICH, DREAM, habit learning, workflow suggestions, nmem update.

---

## Table of Contents

- [Expert Feedback Summary](#expert-feedback-summary)
- [v0.14.0 — Relation Extraction Engine](#v0140--relation-extraction-engine)
- [v0.15.0 — Associative Inference](#v0150--associative-inference)
- [v0.16.0 — Emotional Valence](#v0160--emotional-valence)
- [v0.17.0 — Brain Diagnostics](#v0170--brain-diagnostics)
- [v0.18.0 — Advanced Consolidation](#v0180--advanced-consolidation)
- [v0.19.0 — Temporal Reasoning](#v0190--temporal-reasoning)
- [v0.20.0 — Habitual Recall](#v0200--habitual-recall)
- [v1.0.0 — Portable Consciousness v2](#v100--portable-consciousness-v2)
- [Dependency Graph](#dependency-graph)
- [Gap Coverage Matrix](#gap-coverage-matrix)
- [Expert Feedback Coverage](#expert-feedback-coverage)
- [VISION.md Checklist Per Phase](#visionmd-checklist-per-phase)
- [Implementation Priority](#implementation-priority)

---

## Expert Feedback Summary

Four expert reviewers analyzed NeuralMemory v0.13.0 and identified critical architectural gaps:

| Expert | Role | Core Insight | Key Contribution |
|--------|------|-------------|-----------------|
| E1 | Architecture | Memory ingestion must be agent-agnostic. 3-layer tag model: structural / associative / semantic | "Agent = narrator, not architect" principle |
| E2 | Philosophy | Reflexive vs Cognitive memory. Cognitive enrichment needed | Synapse diversity matters more than tag quality |
| E3 | Pragmatic | Auto-synapses > auto-tags. Brain diversity is evolutionary | Dynamic purity score, accept + mitigate |
| E4 | Quality | Descriptive vs Functional tags. Semantic drift. Confirmation weighting | Tag origin tracking, ontology alignment, Hebbian tag confirmation |

### Expert 4's Unique Gaps (not covered by E1–E3)

1. **Tag origin tracking** — Tags should carry `origin` metadata (`auto` vs `agent`). Auto-tags for accuracy in recall, agent-tags for creativity in deep reasoning.
2. **Semantic drift / ontology alignment** — Multiple agents create "UI" vs "Frontend" vs "Client-side" → brain fragmentation. NM needs tag normalization.
3. **Confirmatory weight boost** — When agent tags overlap with auto-tags → Hebbian confirmation signal → boost synapse weights. Divergent agent tags → new association, needs validation.

---

## v0.14.0 — Relation Extraction Engine

> Auto-synapses from content: the brain wires itself.

**Release target**: Next after auto-tags merge.

### The Gap

`CAUSED_BY`, `LEADS_TO`, `BEFORE`, `AFTER`, `ENABLES`, `PREVENTS` synapse types are defined in `core/synapse.py` (29 types total) but are **never auto-created**. The only way to create causal/temporal synapses is manual agent input. A brain that can't wire its own causal relationships is a brain that can't reason about "why."

### Solution

#### 1. Relation extraction module (`extraction/relations.py`)

Regex-based pattern extraction for three relation families:

| Family | Patterns | Synapse Types |
|--------|----------|---------------|
| **Causal** | "because", "caused by", "due to", "as a result", "therefore", "so that", "vì", "nên", "do đó" | `CAUSED_BY`, `LEADS_TO` |
| **Comparative** | "better than", "worse than", "similar to", "unlike", "compared to", "tốt hơn", "giống như" | `SIMILAR_TO`, `CONTRADICTS` |
| **Sequential** | "then", "after", "before", "first...then", "followed by", "trước khi", "sau khi" | `BEFORE`, `AFTER` |

Each extracted relation produces a `RelationCandidate` with: source span, target span, relation type, confidence score (0.0–1.0).

#### 2. Integrate `suggest_memory_type()` into encoder

`suggest_memory_type()` in `core/memory_types.py` (lines 294–363) exists but is not called during encoding. Integrate it as a fallback when no explicit `memory_type` is provided, enabling auto type inference for every memory.

#### 3. Tag origin tracking (E4)

Transform `Fiber.tags: set[str]` into a richer structure that preserves origin metadata:

```python
# Current: tags = {"python", "api", "auth"}
# New: tags carry origin
TagEntry = namedtuple("TagEntry", ["tag", "origin"])  # origin: "auto" | "agent"

# Fiber gains:
#   - auto_tags: set[str]   (from _generate_auto_tags)
#   - agent_tags: set[str]  (from agent input)
#   - tags property: union of both (backward compatible)
```

- Auto-tags used for **accuracy** in recall scoring
- Agent-tags used for **creativity** in deep reasoning (depth 2–3)
- Storage: `typed_memories.tags` JSON gains `{"auto": [...], "agent": [...]}` format

#### 4. Confirmatory weight boost (E4)

When an agent-provided tag matches an auto-generated tag → Hebbian confirmation signal:
- Boost anchor synapse weight by **+0.1** (capped at 1.0)
- Log confirmation event for diagnostics
- Divergent agent tags (no auto-tag match) → create new `RELATED_TO` synapse with weight 0.3 (needs validation through use)

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `extraction/relations.py` | Relation extraction engine (~300 lines) |
| **New** | `tests/unit/test_relation_extraction.py` | Comprehensive pattern tests (~250 lines) |
| **Modified** | `engine/encoder.py` | Add relation extraction step, auto type inference, tag origin, confirmation boost (~80 lines) |
| **Modified** | `core/fiber.py` | Tag origin fields (`auto_tags`, `agent_tags`, backward-compatible `tags` property) |
| **Modified** | `storage/sqlite_store.py` | Tag origin storage format |

### Scope

~600 new lines + ~80 modified lines + ~250 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — auto-synapses create richer graph for spreading activation |
| Spreading activation still central? | Yes — more synapse types = more activation pathways |
| Works without embeddings? | Yes — pure regex pattern matching |
| More detailed query = faster? | Yes — causal queries activate precise chains instead of broad clusters |
| Brain test? | Yes — human brains auto-wire causal associations during encoding |

---

## v0.15.0 — Associative Inference

> Co-activation becomes persistent structure: neurons that fire together wire together.

**Depends on**: v0.14.0 (needs relation extraction for richer co-activation data).

### The Gap

`CoActivation` data is collected during retrieval (spreading activation records which neurons fire together) but is **never synthesized** into persistent synapses. The brain observes patterns but never learns from them. This is like a brain that notices associations but never forms memories of those associations.

### Solution

#### 1. Associative inference engine (`engine/associative_inference.py`)

Accumulate co-activation events across retrievals. When a neuron pair co-activates above a threshold:
- Create persistent `CO_OCCURS` synapse with weight proportional to co-activation frequency
- Track co-activation history for confidence scoring
- Prune inferred synapses that stop being reinforced (natural forgetting)

**Threshold**: 3 co-activations within 7 days → create synapse (configurable).

#### 2. New `INFER` consolidation strategy

Add to `ConsolidationStrategy` enum in `engine/consolidation.py`:
```python
INFER = "infer"  # Create synapses from co-activation patterns
```

Run during consolidation cycle alongside PRUNE/MERGE/SUMMARIZE.

#### 3. Associative tag generation (E1's Layer 2)

E1's 3-layer tag model:
- **Layer 1 (Structural)**: Entity/keyword tags — already implemented in auto-tags
- **Layer 2 (Associative)**: Tags inferred from co-activation clusters — **this phase**
- **Layer 3 (Semantic)**: Abstract concept tags from pattern extraction — future (v0.18.0)

Generate associative tags from frequently co-activated neuron groups. Example: if "Redis", "cache", "performance" neurons co-activate 5+ times → infer associative tag "caching-infrastructure".

#### 4. Tag normalization / ontology alignment (E4)

New module `utils/tag_normalizer.py`:

```python
class TagNormalizer:
    # Static synonym map
    SYNONYMS = {
        "frontend": ["ui", "client-side", "client side", "front-end"],
        "backend": ["server-side", "server side", "back-end"],
        "database": ["db", "datastore", "data store"],
        ...
    }

    def normalize(self, tag: str) -> str:
        """Map tag to canonical form via synonyms + SimHash near-match."""

    def detect_drift(self, tags: list[str]) -> list[DriftReport]:
        """Flag tags that are likely synonyms but stored separately."""
```

- **Synonym map**: Curated set of common software/general synonyms
- **SimHash near-match**: Use existing SimHash infrastructure (from v0.13.0 dedup) to detect similar tags that aren't in the synonym map
- Applied during encoding (normalize new tags) and consolidation (detect drift in existing tags)

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `engine/associative_inference.py` | Co-activation → synapse inference (~300 lines) |
| **New** | `utils/tag_normalizer.py` | Synonym map + SimHash tag normalization (~200 lines) |
| **New** | `tests/unit/test_associative_inference.py` | Inference threshold + edge cases (~200 lines) |
| **New** | `tests/unit/test_tag_normalizer.py` | Normalization + drift detection (~100 lines) |
| **Modified** | `engine/retrieval.py` | Record co-activation events for inference (~40 lines) |
| **Modified** | `engine/consolidation.py` | Add INFER strategy (~50 lines) |
| **Modified** | `storage/sqlite_store.py` | Co-activation event storage (~30 lines) |

### Scope

~500 new lines + ~120 modified lines + ~300 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — inferred synapses create new activation pathways |
| Spreading activation still central? | Yes — more connections = richer spreading |
| Works without embeddings? | Yes — pure co-activation counting |
| More detailed query = faster? | Yes — inferred links provide shortcuts |
| Brain test? | Yes — Hebbian learning: "neurons that fire together wire together" |

---

## v0.16.0 — Emotional Valence

> Memories gain emotional color: the brain feels, not just knows.

**Independent**: Can be built in parallel with v0.15.0 or v0.17.0.

### The Gap

`FELT` and `EVOKES` synapse types exist in `core/synapse.py` but **nothing creates them**. Emotional context is a fundamental dimension of biological memory — traumatic memories persist longer, positive associations strengthen recall. Without valence, the brain is purely logical.

### Solution

#### 1. Sentiment extraction (`extraction/sentiment.py`)

Regex/lexicon-based sentiment analysis — NO LLM dependency:

```python
class Valence(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass(frozen=True)
class SentimentResult:
    valence: Valence
    intensity: float      # 0.0 – 1.0
    emotion_tags: set[str]  # {"frustration", "satisfaction", ...}
```

**Approach**:
- Curated lexicon: ~200 positive + ~200 negative words (English + Vietnamese)
- Negation handling: "not good" → negative
- Intensifier handling: "very frustrated" → higher intensity
- Emotion tag mapping: word clusters → emotion categories (frustration, satisfaction, confusion, excitement, etc.)

#### 2. Emotional synapses at encode time

During encoding in `engine/encoder.py`:
- Run sentiment extraction on content
- If non-neutral: create `FELT` synapse from anchor neuron to emotion concept neuron
- Emotion concept neurons are shared across fibers (reused, not duplicated)

#### 3. Valence-aware retrieval scoring

In `engine/retrieval.py`, add emotional resonance to score breakdown:
- Queries with emotional content (e.g., "frustrated about the bug") get a boost for matching-valence fibers
- Score component: `emotional_resonance` (0.0–0.1 range)

#### 4. Emotional decay modulation

Extend type-aware decay (v0.13.0):
- High-intensity negative memories decay **slower** (trauma persistence)
- High-intensity positive memories decay **slightly slower** (reward reinforcement)
- Neutral memories follow standard decay curves

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `extraction/sentiment.py` | Lexicon-based sentiment analysis (~250 lines) |
| **New** | `tests/unit/test_sentiment.py` | Sentiment accuracy + edge cases (~200 lines) |
| **Modified** | `engine/encoder.py` | Sentiment extraction step + FELT synapse creation (~40 lines) |
| **Modified** | `engine/retrieval.py` | Valence-aware scoring component (~40 lines) |

### Scope

~250 new lines + ~80 modified lines + ~200 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — emotional synapses are new activation pathways |
| Spreading activation still central? | Yes — emotion nodes become high-connectivity hubs |
| Works without embeddings? | Yes — pure lexicon matching |
| More detailed query = faster? | Yes — emotional context narrows activation |
| Brain test? | Yes — emotional valence is fundamental to biological memory |

---

## v0.17.0 — Brain Diagnostics

> Know thy brain: quality metrics, health reports, actionable insights.

**Independent**: Can be built in parallel with v0.15.0 or v0.16.0.

### The Gap

No purity score, no activation efficiency metrics, no diagnostic tools. Users can't assess brain quality, detect fragmentation, or identify structural problems. Flying blind.

### Solution

#### 1. Diagnostics engine (`engine/diagnostics.py`)

```python
@dataclass(frozen=True)
class BrainHealthReport:
    # Overall health
    purity_score: float        # 0–100, weighted composite
    grade: str                 # A/B/C/D/F

    # Component scores
    connectivity: float        # Avg synapses per neuron (target: 3–8)
    diversity: float           # Synapse type distribution entropy
    freshness: float           # % of fibers accessed in last 7 days
    consolidation_ratio: float # Semantic fibers / total fibers
    orphan_rate: float         # Neurons with 0 synapses / total

    # Activation metrics
    avg_activation_efficiency: float  # Queries reaching depth-0 / total
    avg_recall_confidence: float      # Mean reconstruction confidence

    # Structural warnings
    warnings: list[DiagnosticWarning]
    recommendations: list[str]
```

**Purity score formula** (E3's dynamic purity):
```
purity = (
    connectivity_score * 0.25 +
    diversity_score * 0.20 +
    freshness_score * 0.15 +
    consolidation_score * 0.15 +
    (1 - orphan_rate) * 0.10 +
    activation_efficiency * 0.10 +
    recall_confidence * 0.05
) * 100
```

#### 2. Semantic drift detection (E4)

Part of diagnostics: scan all tags across fibers and flag likely synonyms:
- Use `TagNormalizer.detect_drift()` from v0.15.0 (or include it here if v0.15.0 not yet shipped)
- Report synonym clusters: `{"UI", "Frontend", "Client-side"}` → recommend normalization
- Include in `BrainHealthReport.warnings`

#### 3. MCP tool: `nmem_health`

New MCP tool exposing diagnostics:
```
nmem_health → BrainHealthReport (JSON)
```

#### 4. CLI command: `nmem health`

```
$ nmem health
Brain: default
Grade: B (78/100)

Connectivity:     ████████░░  8.2 synapses/neuron (good)
Diversity:        ██████░░░░  6 of 29 synapse types used (moderate)
Freshness:        █████████░  91% accessed this week (excellent)
Consolidation:    ████░░░░░░  12% semantic (low — run consolidation)
Orphan rate:      █░░░░░░░░░  3% orphaned (excellent)

Warnings:
  ⚠ Tag drift detected: {"UI", "Frontend"} — consider normalization
  ⚠ Low synapse diversity — only RELATED_TO and CO_OCCURS used

Recommendations:
  → Run `nmem consolidate --strategy mature` to advance episodic memories
  → Causal patterns detected but no CAUSED_BY synapses — upgrade to v0.14.0+
```

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `engine/diagnostics.py` | BrainHealthReport + scoring (~350 lines) |
| **New** | `tests/unit/test_diagnostics.py` | Score calculation + edge cases (~200 lines) |
| **Modified** | `mcp/tool_schemas.py` | `nmem_health` tool schema (~20 lines) |
| **Modified** | `mcp/server.py` | `nmem_health` handler (~30 lines) |
| **Modified** | `cli/commands/info.py` | `health` subcommand (~60 lines) |

### Scope

~450 new lines + ~60 modified lines (MCP/CLI) + ~200 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Meta — diagnostics improve the activation network itself |
| Spreading activation still central? | Yes — diagnostics measure activation quality |
| Works without embeddings? | Yes — pure graph metrics |
| More detailed query = faster? | N/A (diagnostic tool, not query feature) |
| Brain test? | Yes — self-awareness / metacognition is a brain function |

---

## v0.18.0 — Habitual Recall ✅

> The brain sleeps, dreams, learns habits, and wakes up smarter.
> "A workflow in NeuralMemory is not stored. It is a stabilized activation path
> that may optionally be reified as a WORKFLOW fiber for interaction."

**Status**: ✅ Shipped as v0.20.0 (2026-02-09). 86 new tests, schema v10, 1105 total tests.
**Depends on**: v0.14.0 ✅ (relation extraction), v0.15.0 ✅ (associative inference + co-activation data).

### Design Philosophy (refined through 3 rounds of expert feedback)

1. **Same substrate** — workflows are neurons + synapses + fibers in the SAME graph, not a separate layer
2. **Hebbian sequence learning** — repeated use strengthens BEFORE/AFTER synapses (online + batch)
3. **Success bias** — only promote completed, non-interrupted sequences
4. **Cognitive patterns** — not just CLI commands, but activation patterns (debug = root_cause → fix → test)
5. **Spreading activation = suggestion engine** — no separate matcher needed
6. **Detect + Store + Suggest ONLY** — NM does not execute or orchestrate workflows
7. **Activation energy decreases with repetition** — existing `synapse.weight` + `fiber.conductivity` already model this

### The Gaps

**Gap 1 — No knowledge creation**: No `ENRICH` strategy, no transitive inference, no dream-like consolidation. Current consolidation (PRUNE/MERGE/SUMMARIZE/MATURE/INFER) handles cleanup, compression, and co-activation inference but doesn't create **new knowledge from existing knowledge**. A brain that can't dream can't make novel connections.

**Gap 2 — No habit learning**: The brain tracks frequency (fiber.frequency, neuron.access_frequency) and co-activation (pairs) but cannot detect **ordered sequences of repeated actions**. A brain that can't recognize habits can't suggest workflows.

### Solution — Part A: Advanced Consolidation

#### 1. `ENRICH` consolidation strategy

Add to `ConsolidationStrategy`:
```python
ENRICH = "enrich"  # Create new synapses via transitive inference
```

**Transitive closure** ("myelination" — shortcut synapses for well-worn paths):
If A→CAUSED_BY→B and B→CAUSED_BY→C, infer A→CAUSED_BY→C with reduced weight (0.5 × min(w_AB, w_BC)).

**Cross-cluster links**: Find fibers in different tag clusters that share entity neurons → create `RELATED_TO` synapses between their anchors.

#### 2. `DREAM` consolidation strategy

```python
DREAM = "dream"  # Random activation for hidden connections
```

**Algorithm**:
1. Select N random neurons (configurable, default 5)
2. Run spreading activation from each
3. Record unexpected co-activations (neurons that wouldn't normally co-activate)
4. If unexpected co-activation count > threshold → create weak `RELATED_TO` synapse (weight 0.1)
5. These "dream synapses" must be reinforced through actual use or they decay quickly (10× normal decay rate)

This is E1's Layer 3 (semantic) tag generation via emergent concept discovery.

#### 3. Importance-based retention

During PRUNE strategy:
- High-salience fibers (salience > 0.8) resist pruning even if inactive
- Fibers with many inbound synapses (hub neurons) get decay protection
- Emotional fibers (from v0.16.0) decay slower

### Solution — Part B: Habit Formation

> "Actions that sequence together template together."
> Workflows emerge as **stable activation attractors** — not stored templates.

Zero new synapse types. Zero new neuron types. Zero LLM.
Everything through existing neurons, synapses, fibers, spreading activation.

#### 4. Action event log — hippocampal buffer (`storage/action_log.py`)

Lightweight, temporary log of user actions (NOT neurons — raw events are too numerous
and low-value for permanent graph storage). Analogous to the hippocampal buffer that holds
recent events before consolidation promotes significant patterns to long-term memory.

```python
@dataclass(frozen=True)
class ActionEvent:
    id: str
    brain_id: str
    session_id: str          # Groups actions within a session
    action_type: str         # "remember", "recall", "encode", "consolidate", etc.
    action_context: str      # Content summary / query text (truncated)
    tags: frozenset[str]     # Tags involved in this action
    fiber_id: str | None     # Fiber created/accessed (if applicable)
    created_at: datetime
```

- Schema migration v9→v10 adds `action_events` table
- Storage interface: `record_action()`, `get_action_sequences()`, `prune_action_events()`
- 30-day retention, auto-prune during consolidation
- Hooks into CLI + MCP tools to record every action

#### 5. Sequential Hebbian — online strengthening

Real-time Hebbian learning for action sequences during normal usage:

```python
# BrainConfig extension
sequential_window_seconds: float = 30.0  # Configurable (30s default for coding workflows)
```

When action B follows action A within the temporal window:
- Find or create BEFORE synapse between ACTION neurons for A and B
- Strengthen weight (Hebbian: `weight += reinforcement_delta`, capped at 1.0)
- Increment `synapse.metadata.sequential_count`
- Combined with existing `fiber.conductivity` increase via `conduct()`: activation energy
  for the path naturally decreases with repetition (STDP-like mechanism)

No batch processing needed — happens incrementally during usage.

#### 6. `LEARN_HABITS` consolidation strategy — batch mining

```python
LEARN_HABITS = "learn_habits"  # Extract workflow patterns from action logs
```

Run during consolidation alongside ENRICH/DREAM:
1. Query action sequences from last N days (configurable window, default 30)
2. N-gram frequency counting across sessions → find repeated subsequences
3. **Success bias**: only promote sequences that completed without interruption (freq ≥ 3 sessions)
4. Strengthen BEFORE/AFTER synapses between existing ACTION concept neurons
5. Create WORKFLOW-typed fibers for recognized patterns (minimal reification of activation attractors)
6. **User confirmation for naming**: heuristic default (join action types: "recall-edit-test") + prompt user to override with semantic name ("dev-cycle")
7. Prune old action events outside the retention window

No new synapse types (uses existing BEFORE/AFTER). No new neuron types (uses existing ACTION/CONCEPT).

#### 7. Proactive suggestion via spreading activation

No separate matcher engine. Spreading activation IS the suggestion engine:

1. When user performs action A → ACTION neuron for A activates
2. Spreading activation flows through strong BEFORE synapses to B
3. **Dual threshold**: only suggest when `synapse.weight > 0.8 AND sequential_count > 5`
4. Surface B as the predicted next step

```python
@dataclass(frozen=True)
class WorkflowSuggestion:
    workflow_fiber_id: str | None  # WORKFLOW fiber if reified, None if pure attractor
    workflow_name: str             # User-named or heuristic
    next_steps: tuple[str, ...]    # Remaining steps in the detected pattern
    confidence: float              # Synapse weight of the next transition
    message: str                   # "You usually do X next. Continue?"
```

Exposed via:
- **MCP tool**: `nmem_suggest` → returns active workflow suggestions
- **CLI**: `nmem suggest` → show current workflow suggestions
- **Retrieval metadata**: `RetrievalResult.workflow_suggestions: list[WorkflowSuggestion]`

#### 8. Privacy controls

User control over habit data:
- `nmem habits list` — show learned workflow patterns with confidence scores
- `nmem habits clear` — wipe action_events table + WORKFLOW fibers
- `nmem habits show <name>` — detail of a specific habit pattern

### Files

| Action | File | Changes |
|--------|------|---------|
| **Modified** | `engine/consolidation.py` | ENRICH + DREAM + LEARN_HABITS strategies (~350 lines) |
| **Modified** | `engine/pattern_extraction.py` | Transitive closure helper (~100 lines) |
| **New** | `storage/action_log.py` | ActionEvent storage mixin (~80 lines) |
| **New** | `engine/sequence_mining.py` | Sequence mining + habit formation (~250 lines) |
| **New** | `engine/workflow_suggest.py` | Proactive suggestion via activation (~100 lines) |
| **Modified** | `engine/retrieval.py` | Attach workflow suggestions to results (~30 lines) |
| **Modified** | `storage/sqlite_schema.py` | action_events table migration v9→v10 (~20 lines) |
| **Modified** | `storage/base.py` | Action log abstract methods (~20 lines) |
| **Modified** | `core/brain.py` | BrainConfig: `sequential_window_seconds` (~5 lines) |
| **Modified** | `mcp/tool_schemas.py` | `nmem_suggest` tool schema (~15 lines) |
| **Modified** | `mcp/server.py` | `nmem_suggest` handler (~20 lines) |
| **Modified** | `cli/commands/` | `habits` subcommand group (~60 lines) |
| **New** | `tests/unit/test_enrichment.py` | Transitive inference + dream tests (~200 lines) |
| **New** | `tests/unit/test_sequence_mining.py` | Sequence mining + habit formation tests (~200 lines) |
| **New** | `tests/unit/test_workflow_suggest.py` | Proactive suggestion tests (~150 lines) |

### Scope

~950 new lines + ~555 modified lines + ~550 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — DREAM uses spreading activation; habits emerge as stable activation attractors |
| Spreading activation still central? | Yes — DREAM uses it; suggestions flow through it; no separate matcher engine |
| Works without embeddings? | Yes — graph traversal + frequency-based n-gram mining + Hebbian strengthening |
| More detailed query = faster? | Yes — enrichment creates shortcuts; strong BEFORE synapses predict next action |
| Brain test? | Yes — dreaming, transitive inference, habit formation, and myelination are core brain functions |

---

## v0.19.0 — Temporal Reasoning

> "Why did this happen?" — trace the causal chain. "When?" — query time ranges.

**Depends on**: v0.14.0 (needs auto-created causal/temporal synapses to traverse).

### The Gap

"Why?" queries can't trace `CAUSED_BY` chains — the router identifies causal intent (in `extraction/router.py`) but retrieval has no causal traversal algorithm. "When?" queries can't do temporal range filtering beyond basic time bounds. Event sequences aren't first-class query results.

### Solution

#### 1. Causal traversal engine (`engine/causal_traversal.py`)

```python
def trace_causal_chain(
    store: SQLiteStore,
    brain_id: str,
    fiber_id: str,
    direction: Literal["causes", "effects"],
    max_depth: int = 5,
) -> CausalChain:
    """Follow CAUSED_BY/LEADS_TO synapses to build a causal chain."""

def query_temporal_range(
    store: SQLiteStore,
    brain_id: str,
    start: datetime,
    end: datetime,
    memory_types: set[MemoryType] | None = None,
) -> list[Fiber]:
    """Retrieve fibers within a temporal range, ordered chronologically."""

def trace_event_sequence(
    store: SQLiteStore,
    brain_id: str,
    seed_fiber_id: str,
    direction: Literal["forward", "backward"],
    max_steps: int = 10,
) -> EventSequence:
    """Follow BEFORE/AFTER synapses to reconstruct event sequences."""
```

#### 2. New synthesis methods

Add to `SynthesisMethod` in `engine/reconstruction.py`:
```python
CAUSAL_CHAIN = "causal_chain"        # "A because B because C"
TEMPORAL_SEQUENCE = "temporal_sequence"  # "First A, then B, then C"
```

Reconstruction formats the chain/sequence into natural language output.

#### 3. Router integration

Enhance `extraction/router.py` to route:
- "Why?" queries → causal traversal → `CAUSAL_CHAIN` synthesis
- "When?" queries → temporal range → `TEMPORAL_SEQUENCE` synthesis
- "What happened after X?" → event sequence → `TEMPORAL_SEQUENCE` synthesis

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `engine/causal_traversal.py` | Causal chain + temporal range + event sequence (~300 lines) |
| **New** | `tests/unit/test_causal_traversal.py` | Chain traversal + edge cases (~200 lines) |
| **Modified** | `engine/reconstruction.py` | CAUSAL_CHAIN + TEMPORAL_SEQUENCE synthesis (~60 lines) |
| **Modified** | `extraction/router.py` | Route causal/temporal queries to traversal (~40 lines) |

### Scope

~300 new lines + ~100 modified lines + ~200 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — causal traversal IS directed activation along causal synapses |
| Spreading activation still central? | Yes — causal traversal is constrained spreading activation |
| Works without embeddings? | Yes — pure graph traversal |
| More detailed query = faster? | Yes — "Why did X fail?" traverses a specific causal chain |
| Brain test? | Yes — causal reasoning is fundamental to human cognition |

---

## v0.20.0 — Habitual Recall (shipped)

> Implements the v0.18.0 plan. Knowledge creation + habit learning + self-update.

**Shipped**: 2026-02-09. 86 new tests (1105 total), schema v9 → v10.

### What shipped

| Feature | Files |
|---------|-------|
| **ENRICH consolidation** — transitive closure on CAUSED_BY chains + cross-cluster linking | `engine/enrichment.py` |
| **DREAM consolidation** — random exploration via spreading activation, 10x decay | `engine/dream.py` |
| **Action event log** — hippocampal buffer (schema v10), session-grouped action tracking | `core/action_event.py`, `storage/sqlite_action_log.py` |
| **Sequence mining** — detect repeated action patterns, create WORKFLOW fibers + BEFORE synapses | `engine/sequence_mining.py` |
| **Workflow suggestions** — proactive next-action hints via dual-threshold activation | `engine/workflow_suggest.py` |
| **nmem_habits MCP tool** — suggest/list/clear learned habits | `mcp/server.py`, `mcp/tool_schemas.py` |
| **nmem habits CLI** — list, show, clear subcommands | `cli/commands/habits.py` |
| **nmem update CLI** — self-update with auto-detect pip/git source | `cli/commands/update.py` |
| **Salience-based prune protection** — high-salience fibers resist pruning | `engine/consolidation.py` |
| **Action recording** — MCP server records remember/recall/context actions | `mcp/server.py` |

### BrainConfig additions

`sequential_window_seconds`, `dream_neuron_count`, `dream_decay_multiplier`, `habit_min_frequency`, `habit_suggestion_min_weight`, `habit_suggestion_min_count`

### Design decisions

- Zero new synapse/neuron types — reuses BEFORE, CAUSED_BY, RELATED_TO, CO_OCCURS + ACTION, CONCEPT
- Hippocampal buffer — action events are lightweight DB rows (not neurons) to avoid graph bloat
- Zero LLM dependency — pure frequency-based pattern detection
- Dream synapses decay 10x faster unless reinforced

---

## v1.0.0 — Portable Consciousness v2 ✅

> Marketplace foundations: brains become products.

**Status**: ✅ Shipped (2026-02-09). Schema v11, 16 MCP tools.
**Depends on**: v0.17.0 (diagnostics for brain quality rating), all prior versions for stable API surface.

### Features

#### 1. Brain versioning
- Snapshot history: save named versions of brain state
- Rollback: restore any previous version
- Diff: compare two versions (neurons/synapses added/removed)
- Storage: version metadata in `brain_versions` table

#### 2. Partial brain transplant
- Topic-filtered merge: extract fibers matching tag/type filters from one brain
- Import into target brain with conflict resolution
- Preserve synapse structure within the transplanted subgraph
- Example: "transplant all Python knowledge from expert-brain to my-brain"

#### 3. Brain quality rating
- Grade A–F derived from `BrainHealthReport` (v0.17.0)
- Quality badge for marketplace display
- Minimum grade requirements for marketplace listing (B or above)
- Auto-computed, not self-reported

#### 4. Stable API guarantee
- Semantic versioning from v1.0.0 onward
- Public API surface documented and frozen
- Deprecation policy: 2 minor versions before removal
- Migration guides for breaking changes

#### 5. Documentation
- API reference (auto-generated from schemas)
- Architecture guide
- Brain marketplace specification
- Migration guide from v0.x → v1.0

### Files

| Action | File | Changes |
|--------|------|---------|
| **New** | `engine/brain_versioning.py` | Snapshot + rollback + diff (~300 lines) |
| **New** | `engine/brain_transplant.py` | Topic-filtered merge (~200 lines) |
| **Modified** | `engine/diagnostics.py` | Quality grade computation (~50 lines) |
| **Modified** | `storage/sqlite_store.py` | Version table + transplant queries (~100 lines) |
| **Modified** | `storage/sqlite_schema.py` | `brain_versions` table migration (~30 lines) |
| **New** | `tests/unit/test_brain_versioning.py` | Snapshot + rollback tests (~200 lines) |
| **New** | `tests/unit/test_brain_transplant.py` | Filtered merge tests (~150 lines) |

### Scope

~500 new lines + ~150 modified lines + ~350 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Activation — transplanted subgraphs preserve activation structure |
| Spreading activation still central? | Yes — versioning/transplant don't change the core algorithm |
| Works without embeddings? | Yes — pure graph operations |
| More detailed query = faster? | N/A (infrastructure, not query feature) |
| Brain test? | Yes — brain transplants are real (well, almost). Versioning = memory snapshots |

---

## Dependency Graph

```
v0.14.0 ✅ (Relation Extraction)
  ├──→ v0.15.0 ✅ (Associative Inference)
  │       └──→ v0.20.0 ✅ (Habitual Recall — shipped as v0.18.0 plan)
  └──→ v0.19.0 ✅ (Temporal Reasoning)

v0.16.0 ✅ (Emotional Valence)     ← shipped
v0.17.0 ✅ (Brain Diagnostics)     ← shipped
  └──→ v1.0.0 ✅ (Portable Consciousness v2)
```

**All versions shipped.** Roadmap complete.

**Critical path**: v0.14.0 ✅ → v0.15.0 ✅ → v0.20.0 ✅ → v1.0.0 ✅

---

## Gap Coverage Matrix

### 7 Critical Architectural Gaps

| # | Gap | Status Before | Resolved In |
|---|-----|---------------|-------------|
| G1 | Causal/temporal synapses never auto-created | 29 synapse types defined, 0 auto-created | **v0.14.0** ✅ |
| G2 | Co-activation never synthesized into synapses | Data collected, never used | **v0.15.0** ✅ |
| G3 | Emotional synapses (`FELT`/`EVOKES`) never created | Types exist, unused | **v0.16.0** ✅ |
| G4 | No brain health metrics or diagnostics | Flying blind | **v0.17.0** ✅ |
| G5 | No enrichment or dream consolidation | Only PRUNE/MERGE/SUMMARIZE/MATURE/INFER | **v0.20.0** ✅ |
| G6 | "Why?" and "When?" queries can't trace chains | Router detects intent, no traversal | **v0.19.0** ✅ |
| G7 | No brain versioning or partial transplant | Export/import only (all-or-nothing) | **v1.0.0** ✅ |
| G8 | No habit/workflow detection from repeated actions | Frequency tracked but sequences ignored | **v0.20.0** ✅ |

### Expert 4's 3 New Gaps

| # | Gap | Resolved In |
|---|-----|-------------|
| E4-1 | Tag origin tracking (auto vs agent) | **v0.14.0** |
| E4-2 | Semantic drift / ontology alignment | **v0.15.0** (normalizer) + **v0.17.0** (detection) |
| E4-3 | Confirmatory weight boost (Hebbian tag confirmation) | **v0.14.0** |

---

## Expert Feedback Coverage

| Expert | Key Point | Phase |
|--------|-----------|-------|
| **E1** | Agent-agnostic ingestion | v0.14.0 (auto type inference, auto relations) |
| **E1** | Layer 1 tags (structural) | v0.13.0 ✓ (auto-tags, already implemented) |
| **E1** | Layer 2 tags (associative) | v0.15.0 (co-activation → tags) |
| **E1** | Layer 3 tags (semantic) | v0.18.0 (DREAM consolidation → emergent concepts) |
| **E2** | Reflexive vs Cognitive memory | v0.14.0+ (auto-synapses = reflexive wiring) |
| **E2** | Cognitive enrichment | v0.18.0 (ENRICH strategy) |
| **E2** | Synapse diversity > tag quality | v0.14.0–v0.18.0 (each phase adds new synapse creation paths) |
| **E3** | Auto-synapses > auto-tags | v0.14.0 (relation extraction = auto-synapses) |
| **E3** | Dynamic purity score | v0.17.0 (BrainHealthReport.purity_score) |
| **E3** | Accept + mitigate (brain diversity is evolutionary) | v0.15.0 (normalize) + v0.18.0 (DREAM validates via use) |
| **E4** | Tag origin tracking | v0.14.0 (auto_tags/agent_tags split) |
| **E4** | Semantic drift / ontology alignment | v0.15.0 (TagNormalizer) + v0.17.0 (drift detection) |
| **E4** | Confirmatory weight boost (Hebbian) | v0.14.0 (agent tag ∩ auto tag → +0.1 weight) |
| **E4** | Descriptive vs Functional tags | v0.14.0 (origin tracking enables differential use) |

---

## VISION.md Checklist Per Phase

Each phase must pass all 4 questions + brain test before implementation begins.

| Phase | Q1: Activation? | Q2: Spreading central? | Q3: No embeddings? | Q4: Detail = fast? | Brain test? |
|-------|-----------------|----------------------|--------------------|--------------------|-------------|
| v0.14.0 | ✓ Auto-synapses | ✓ More pathways | ✓ Regex | ✓ Precise chains | ✓ Causal wiring |
| v0.15.0 | ✓ Inferred links | ✓ Richer graph | ✓ Counting | ✓ Shortcuts | ✓ Hebbian learning |
| v0.16.0 | ✓ Emotion paths | ✓ Emotion hubs | ✓ Lexicon | ✓ Emotional focus | ✓ Emotional memory |
| v0.17.0 | ✓ Meta-quality | ✓ Measures it | ✓ Graph metrics | N/A | ✓ Metacognition |
| v0.18.0 | ✓ Dream links | ✓ DREAM uses it | ✓ Graph ops | ✓ Transitive shortcuts | ✓ Dream consolidation |
| v0.19.0 | ✓ Causal activation | ✓ Directed spreading | ✓ Graph traversal | ✓ Precise chains | ✓ Causal reasoning |
| v1.0.0 | ✓ Preserved structure | ✓ Unchanged | ✓ Graph ops | N/A | ✓ Memory snapshots |

---

## Implementation Priority

Ranked by impact × feasibility:

| Rank | Phase | Impact | Feasibility | Rationale |
|------|-------|--------|-------------|-----------|
| 1 | **v0.14.0** ✅ | Critical | High | Shipped. Relation extraction, tag origin, confirmatory boost. |
| 2 | **v0.15.0** ✅ | High | Medium | Shipped. Associative inference, co-activation, tag normalization. |
| 3 | **v0.17.0** ✅ | High | High | Shipped. Brain diagnostics, purity score, nmem_health MCP + CLI. |
| 4 | **v0.16.0** ✅ | Medium | High | Shipped. Emotional valence, sentiment extraction, FELT synapses, emotional decay. |
| 5 | **v0.19.0** ✅ | High | Medium | Shipped. Temporal reasoning, causal/event traversal, pipeline integration. |
| 6 | **v0.20.0** ✅ | High | Medium | Shipped. DREAM + ENRICH + habit learning + workflow suggestions + nmem update. |
| 7 | **v1.0.0** ✅ | Critical | Low | Shipped. Brain versioning, transplant, quality badge, embedding layer, LLM extraction. |

### Recommended execution order

```
v0.14.0 ✅ → v0.15.0 ✅ → v0.16.0 ✅ → v0.17.0 ✅ → v0.19.0 ✅ → v0.20.0 ✅ → v1.0.0
```

All versions shipped. Roadmap complete.

---

## Cumulative Scope Estimate

| Phase | New Lines | Modified Lines | Test Lines | Cumulative Tests (est.) |
|-------|-----------|---------------|------------|------------------------|
| v0.14.0 | ~600 | ~80 | ~250 | ~1,041 |
| v0.15.0 | ~500 | ~120 | ~300 | ~1,341 |
| v0.16.0 | ~250 | ~80 | ~200 | ~1,541 |
| v0.17.0 | ~450 | ~60 | ~200 | ~1,741 |
| v0.18.0/v0.20.0 | ~990 | ~290 | ~900 | 1,105 (actual) |
| v0.19.0 | ~300 | ~100 | ~200 | ~2,208 |
| v1.0.0 | ~500 | ~150 | ~350 | ~2,558 |
| **Total** | **~3,550** | **~1,145** | **~2,050** | **~2,558** |

Starting from 1105 tests (v0.20.0) → targeting ~1,455+ tests at v1.0.0.

---

*See [VISION.md](VISION.md) for the north star guiding all decisions.*

---
---

# Post-v1.0 Roadmap: v1.1.0 → v2.0.0

> From library to platform. From CLI to visual. From single-user to ecosystem.
>
> **Three pillars**: Dashboard + Integrations + Community
>
> **Current state**: v1.0.2 shipped. 1,340 tests. 16 MCP tools. Nanobot integration done. ClawHub SKILL published.

## Strategic Context

### Market Position (Feb 2026)

| Project | Stars | Language | Memory Approach | LLM Required |
|---------|-------|----------|----------------|--------------|
| **OpenClaw** | 178k | TypeScript | SQLite + FTS5 + sqlite-vec flat files | Yes (embeddings) |
| **Mem0** | 47k | Python | Vector DB + LLM extraction | Yes (2+ calls/write) |
| **LlamaIndex** | 47k | Python | Document chunking + vector | Yes |
| **Graphiti** | 23k | Python | Bi-temporal knowledge graph | Yes (LLM extraction) |
| **claude-mem** | 26k | TypeScript | sqlite-vec + FTS5 | Yes (embeddings) |
| **Cognee** | 12k | Python | ECL pipeline + KuzuDB | Yes (LLM ingestion) |
| **NeuralMemory** | <1k | Python | Neural graph + spreading activation | **No** |

### NM's Unique Differentiators

1. **Zero LLM dependency** — pure algorithmic (regex, graph, Hebbian)
2. **Spreading activation** — associative recall, not search
3. **Self-improving** — Hebbian learning strengthens used paths
4. **Contradiction detection** — auto-detects conflicting memories
5. **Memory lifecycle** — STM → Working → Episodic → Semantic with decay
6. **Temporal reasoning** — causal chains, event sequences
7. **Brain versioning + transplant** — no competitor has this

### OpenClaw Memory Weaknesses (documented in issues)

| Issue | Problem | NM Solves |
|-------|---------|-----------|
| [#9143](https://github.com/openclaw/openclaw/issues/9143) | Embedding API failures permanently disable memory search | No embedding required |
| [#5696](https://github.com/openclaw/openclaw/issues/5696) | Token limit exceeded, no truncation | Neuron-level chunking |
| [#7776](https://github.com/openclaw/openclaw/issues/7776) | Cross-project noise in search | Brain isolation per project |
| [#8921](https://github.com/openclaw/openclaw/issues/8921) | Third-party memory plugins not detected by status | Contribution opportunity |
| No issue | No semantic relationships | 20 typed synapses |
| No issue | No decay/retention | Ebbinghaus curve + type-aware decay |
| No issue | Returns contradictory results | Auto-contradiction detection |

### Competitors Already Inside OpenClaw

- **Mem0**: Published [blog post + integration guide](https://mem0.ai/blog/mem0-memory-for-openclaw)
- **Cognee**: Published [blog post + integration](https://www.cognee.ai/blog/integrations/what-is-openclaw-ai-and-how-we-give-it-memory-with-cognee)

Both replace the exclusive memory slot. NM needs to enter this space with a stronger value proposition.

---

## Phase 1: v1.1.0 — Community Foundations

> Get noticed. Minimal code, maximum visibility.

**Status**: In progress. SKILL published, blog written, PR research done.

### 1.1 ClawHub SKILL.md ✅

Published `neural-memory@1.0.0` to [ClawHub](https://clawhub.ai/skills/neural-memory) — OpenClaw's official skill registry (2,999+ curated skills, 60k Discord users browse it).

**What it does**: Instructs OpenClaw's agent to use NM via the existing MCP server.
**Shipped**: Commit `8d661cb`, verified live via `clawhub inspect neural-memory`.

### 1.2 OpenClaw Issue #7273 Fix PR ✅

Issue #8921 was **closed as duplicate** of [#7273](https://github.com/openclaw/openclaw/issues/7273) — `openclaw status` reports memory as unavailable for third-party plugins.

**Submitted**: [PR #12596](https://github.com/openclaw/openclaw/pull/12596) — minimal 2-file fix (~50 lines). Type-check clean, 5/5 tests pass. The competing PR #7289 had 32 files and was flagged 2/5 confidence by Greptile.

**Fix**: `status.scan.ts` now probes non-core plugins via gateway RPC `memory.status`. `status.command.ts` renders "active" or "N entries" instead of "unavailable".

### 1.3 Blog Post: "Neural Memory for OpenClaw" ✅

Written in Vietnamese at `docs/blog/neural-memory-openclaw.md`.
Updated to v1.0.2: 1,340 tests, 16 MCP tools, 10 feature sections, full comparison table.

**Pending**: Publish to Dev.to / Viblo.asia / Medium.

### 1.4 Community Launch

| Action | Channel | Status |
|--------|---------|--------|
| Publish SKILL | ClawHub | ✅ Done |
| Post blog | Dev.to + Viblo.asia | Pending |
| Post SKILL + demo | OpenClaw Discord #showcase | Pending |
| Tag @openclaw on X | Twitter | Pending |
| Submit to awesome-openclaw-skills | GitHub PR | Pending |
| Submit clean PR for #7273 | OpenClaw repo | ✅ [PR #12596](https://github.com/openclaw/openclaw/pull/12596) |

### Files

| Action | File | Status |
|--------|------|--------|
| ✅ | `integrations/neural-memory/SKILL.md` | Published to ClawHub |
| ✅ | `docs/blog/neural-memory-openclaw.md` | Written, needs publishing |
| ✅ | `docs/ARCHITECTURE_V1_EXTENDED.md` | Committed |

### Scope

~500 lines docs shipped. Remaining: external publishing + OpenClaw PR.

---

## Phase 2: v1.2.0 — Dashboard Foundation

> Replace the vis.js prototype with a production dashboard.

**Target**: 2-3 weeks after v1.1.0

### The Problem

Current UI is a single `index.html` with vis.js — no navigation, no filtering, no management, no analytics. The VS Code extension is richer but locked inside VS Code. Users need a standalone browser dashboard.

### Architecture

```
Browser (SPA)
  ├── Dashboard    — Overview, stats, health grade
  ├── Explorer     — Interactive graph (Cytoscape.js)
  ├── Timeline     — Memory creation over time
  ├── Search       — Query + recall with preview
  ├── Brain Mgmt   — Create, switch, export, import, transplant
  ├── Health       — Diagnostics, warnings, recommendations
  ├── Integrations — MCP status, Nanobot/OpenClaw config
  └── Settings     — Config, language (EN/VI)
        │
  FastAPI Backend (existing)
  + WebSocket (real-time updates)
  + New endpoints for dashboard data
```

### Tech Stack Decision

| Option | Pros | Cons |
|--------|------|------|
| **Vanilla HTML/JS + Alpine.js** | Zero build, ships with NM, lightweight | Limited component reuse |
| **React + Vite** | Rich ecosystem, component library | Separate build, npm dependency |
| **Vue 3 + Vite** | Lighter than React, good DX | Still separate build |
| **Svelte** | Smallest bundle, fast | Smaller ecosystem |

**Recommendation**: **Vanilla HTML/JS + Alpine.js + Tailwind CSS (CDN)**

Rationale:
- Zero build step — dashboard ships as static files inside NM package
- No Node.js/npm dependency for users
- `pip install neural-memory` includes dashboard automatically
- CDN-loaded libraries (Alpine.js, Tailwind, Cytoscape.js, Chart.js)
- Same pattern as existing `index.html` but much richer

### 2.1 Dashboard Overview Page

```
┌─────────────────────────────────────────────────┐
│  🧠 NeuralMemory Dashboard    [brain: default]  │
├──────────┬──────────┬──────────┬────────────────┤
│ Neurons  │ Synapses │ Fibers   │ Health: A (92) │
│   847    │  2,341   │   156    │ ████████████░  │
├──────────┴──────────┴──────────┴────────────────┤
│                                                  │
│  Memory Timeline (30 days)     Type Distribution │
│  ▁▂▃▅▇█▇▅▃▂▁▂▃▅▇            ██ fact    42%    │
│                                ██ decision 18%   │
│  Recent Activity               ██ insight  15%   │
│  • [14:23] Remembered: ...     ██ todo     12%   │
│  • [14:20] Recalled: ...       ██ other    13%   │
│  • [14:15] Consolidated: ...                     │
│                                                  │
│  Warnings                      Quick Actions     │
│  ⚠ Tag drift: UI/Frontend     [Remember] [Recall]│
│  ⚠ Low diversity              [Health] [Export]  │
└──────────────────────────────────────────────────┘
```

### 2.2 Graph Explorer (Cytoscape.js upgrade)

Replace vis.js with Cytoscape.js (already used in VS Code extension):
- Force-directed + hierarchical + radial layouts
- Filter by neuron type, date range, tags
- Fiber pathway highlighting
- Synapse type color coding (20 types)
- Sub-graph navigation (click neuron → show neighborhood)
- Export as PNG/SVG

### 2.3 Brain Management UI

- Brain list with health grades
- Create / delete / switch brains
- Export / import (JSON)
- Version history with rollback
- Transplant wizard (source brain → filter → target brain)

### 2.4 Health Diagnostics Page

- Radar chart of 7 component scores
- Warning list with severity badges
- Recommendation cards with action buttons
- Historical health trend (if data available)

### 2.5 Vietnamese Localization (i18n)

```javascript
// locales/vi.json
{
  "dashboard": "Bảng Điều Khiển",
  "neurons": "Nơ-ron",
  "synapses": "Khớp Thần Kinh",
  "fibers": "Sợi Ký Ức",
  "health": "Sức Khỏe Não",
  "remember": "Ghi Nhớ",
  "recall": "Hồi Tưởng",
  "brain": "Não",
  "settings": "Cài Đặt",
  ...
}
```

- Language toggle in settings (EN/VI)
- Auto-detect from browser locale
- All UI labels, tooltips, error messages localized
- Vietnamese-first approach for targeting Vietnamese developer community

### Files

| Action | File | Description |
|--------|------|-------------|
| **New** | `server/static/dashboard/index.html` | Main SPA entry point |
| **New** | `server/static/dashboard/app.js` | Alpine.js app logic |
| **New** | `server/static/dashboard/graph.js` | Cytoscape.js explorer |
| **New** | `server/static/dashboard/charts.js` | Chart.js analytics |
| **New** | `server/static/dashboard/style.css` | Tailwind overrides |
| **New** | `server/static/dashboard/locales/en.json` | English strings |
| **New** | `server/static/dashboard/locales/vi.json` | Vietnamese strings |
| **Modified** | `server/app.py` | Mount dashboard route, new API endpoints |
| **New** | `server/api/dashboard.py` | Dashboard-specific endpoints |
| **New** | `tests/e2e/test_dashboard_api.py` | Dashboard API tests |

### Scope

~2,500 lines (HTML/JS/CSS) + ~300 lines (Python API) + ~200 test lines

### VISION.md Check

| Question | Answer |
|----------|--------|
| Activation or Search? | Visualization of activation — graph explorer shows spreading paths |
| Spreading activation still central? | Yes — explorer visualizes it |
| Works without embeddings? | Yes — pure frontend rendering |
| Brain test? | Yes — visual cortex processes spatial information |

---

## Phase 3: v1.3.0 — Integration Dashboard

> Dashboard becomes the control center for all integrations.

**Target**: 2 weeks after v1.2.0

### 3.1 Integration Management Panel

```
┌─────────────────────────────────────────────────┐
│  Integrations                                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  ✅ MCP Server          Running on stdio         │
│     Tools: 16 active    Clients: 1 connected     │
│     Last call: 2 min ago                         │
│                                                  │
│  ✅ Nanobot             Connected                │
│     Brain: nanobot      Tools: 4 registered      │
│     Memories today: 23  Recalls today: 47        │
│                                                  │
│  ⬚ OpenClaw            Not configured            │
│     [Setup Guide] [Install SKILL]                │
│                                                  │
│  ⬚ THOR NEXUS          Not configured            │
│     [Setup Guide]                                │
│                                                  │
│  Import Sources                                  │
│  ├── ChromaDB    [Import]                        │
│  ├── Mem0        [Import]                        │
│  ├── Cognee      [Import]                        │
│  ├── Graphiti    [Import]                        │
│  └── LlamaIndex  [Import]                        │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3.2 Integration Activity Log

- Real-time feed of tool calls from all integrations
- Source attribution (MCP, Nanobot, OpenClaw)
- Error tracking and alerts
- Usage statistics per integration

### 3.3 One-Click Setup Wizards

- **Nanobot**: Generate `setup_neural_memory()` code snippet
- **OpenClaw**: Generate MCP config + SKILL.md installation
- **Claude Code**: Generate `mcp_servers.json` entry
- **Cursor**: Generate `mcp.json` entry

### Files

| Action | File | Description |
|--------|------|-------------|
| **New** | `server/static/dashboard/integrations.js` | Integration panel logic |
| **New** | `server/api/integrations.py` | Integration status endpoints |
| **Modified** | `server/static/dashboard/index.html` | Add integration page |
| **New** | `tests/e2e/test_integration_api.py` | Integration API tests |

### Scope

~800 lines (HTML/JS) + ~200 lines (Python API) + ~100 test lines

---

## Phase 4: v1.4.0 — OpenClaw Memory Plugin

> Replace OpenClaw's default memory with NeuralMemory.

**Target**: 3-4 weeks after v1.3.0

### Architecture

```
OpenClaw Gateway (TypeScript)
  │
  ├── Memory Slot (exclusive)
  │   └── @neuralmemory/openclaw-plugin (TypeScript)
  │       ├── registerTool("nmem_remember")
  │       ├── registerTool("nmem_recall")
  │       ├── memory.search() → MCP recall
  │       ├── memory.index() → MCP remember
  │       └── MCP Client (stdio) → python -m neural_memory.mcp
  │
  └── Agent uses NM memory transparently
```

### 4.1 TypeScript MCP Client

Lightweight TypeScript client that spawns NM's MCP server as a subprocess and communicates via JSON-RPC 2.0 over stdio.

```typescript
// @neuralmemory/openclaw-plugin
export default {
  name: "neural-memory",
  version: "1.0.0",
  openclaw: {
    extensions: {
      memory: {
        slot: "memory",
        type: "memory"
      }
    }
  }
}
```

### 4.2 Memory Slot Implementation

Map OpenClaw's memory interface to NM's MCP tools:

| OpenClaw Method | NM MCP Tool |
|----------------|-------------|
| `memory.search(query)` | `nmem_recall` (depth=1) |
| `memory.index(content)` | `nmem_remember` |
| `memory.getContext()` | `nmem_context` |
| `memory.getRecentMemories()` | `nmem_context` (fresh_only=true) |

### 4.3 Contribute Fix for Issue #8921

Submit PR to OpenClaw repo fixing third-party memory plugin detection in `status` command. This directly benefits NM and all other memory plugins.

### Files

| Action | File | Description |
|--------|------|-------------|
| **New** | `integrations/openclaw/plugin/` | TypeScript plugin package |
| **New** | `integrations/openclaw/plugin/package.json` | npm manifest |
| **New** | `integrations/openclaw/plugin/src/index.ts` | Plugin entry point |
| **New** | `integrations/openclaw/plugin/src/mcp-client.ts` | MCP stdio client |
| **New** | `integrations/openclaw/plugin/src/memory-provider.ts` | Memory slot adapter |
| **New** | `integrations/openclaw/plugin/README.md` | Installation guide |

### Scope

~600 lines TypeScript + ~200 lines docs

### External

- PR to openclaw/openclaw fixing Issue #8921
- Publish to npm as `@neuralmemory/openclaw-plugin`
- Update ClawHub SKILL.md with plugin instructions

---

## Phase 5: v1.5.0 — Ecosystem Expansion

> Widen the net. More integrations, more languages, marketplace.

**Target**: 1-2 months after v1.4.0

### 5.1 THOR NEXUS Integration (D:\Antigravity\OmniAI)

Integrate NM into the local THOR NEXUS project for cross-session Solana forensics memory:

| THOR Agent | NM Value |
|------------|----------|
| Zeus (orchestrator) | Persistent decision history |
| Athena (analysis) | Pattern memory across sessions |
| Hermes (trading) | Trade outcome learning |
| Apollo (sentiment) | Sentiment correlation memory |
| Hephaestus (infra) | Error pattern detection |
| Artemis (risk) | Risk assessment history |

### 5.2 Brain Marketplace (Preview)

- Public brain gallery on NM website
- Upload/download brains with quality badges
- Categories: programming, devops, security, data-science, vietnamese-dev
- Search by tags, language, grade

### 5.3 Neo4j Storage Backend

For users with large-scale graph requirements:
- `Neo4jStorage` implementing `BaseStorage` ABC
- Native graph queries (Cypher) for complex traversals
- Horizontal scaling for enterprise use
- Optional — SQLite remains default

### 5.4 Multi-Language Expansion

Beyond EN/VI:
- Japanese (large AI dev community)
- Korean
- Chinese (simplified)
- Extraction patterns + lexicons per language

---

## Phase 6: v2.0.0 — Platform

> NeuralMemory becomes the universal memory layer for AI agents.

### Vision

```
Any AI Agent → NM Protocol (MCP/REST/SDK) → Neural Graph → Intelligent Recall
                                                 ↑
                                          Dashboard (visual management)
                                                 ↑
                                          Brain Marketplace (shared knowledge)
```

### 2.0 Features

- **NM Protocol**: Standardized memory protocol beyond MCP
- **Multi-brain reasoning**: Query across multiple brains simultaneously
- **Federated memory**: Distributed brains across machines
- **Real-time collaboration**: Multiple agents sharing one brain with conflict resolution
- **Memory compression**: Lossy summarization for very old fibers (save storage)
- **Adaptive recall**: Learn which depth level works best per query pattern

---

## Dependency Graph (Post-v1.0)

```
v1.1.0 (Community Foundations)
  ├──→ v1.2.0 (Dashboard Foundation)
  │       └──→ v1.3.0 (Integration Dashboard)
  │               └──→ v1.5.0 (Ecosystem Expansion)
  └──→ v1.4.0 (OpenClaw Memory Plugin)
              └──→ v1.5.0 (Ecosystem Expansion)
                        └──→ v2.0.0 (Platform)
```

**Critical path**: v1.1.0 → v1.2.0 → v1.3.0 → v1.5.0

**Parallel track**: v1.4.0 (OpenClaw plugin) can run independently after v1.1.0

---

## Priority Matrix

| Phase | Impact | Effort | Priority | Rationale |
|-------|--------|--------|----------|-----------|
| **v1.1.0** | High | Low | **P0** | Community visibility, zero code |
| **v1.2.0** | High | High | **P1** | Dashboard is the face of NM |
| **v1.3.0** | Medium | Medium | **P2** | Builds on dashboard, integration UX |
| **v1.4.0** | High | Medium | **P1** | 178k-star ecosystem access |
| **v1.5.0** | Medium | High | **P3** | Expansion, can be incremental |
| **v2.0.0** | Critical | Very High | **P4** | Long-term platform vision |

### Recommended Execution Order

```
v1.1.0 (1 week) → v1.2.0 (3 weeks) → v1.4.0 (2 weeks) → v1.3.0 (2 weeks) → v1.5.0 (ongoing)
```

Do v1.4.0 before v1.3.0 because OpenClaw plugin provides the real-world integration data that makes the Integration Dashboard useful.

---

## Vietnamese Community Strategy

### Why Vietnamese-First

- Creator is Vietnamese
- NM already supports Vietnamese (extraction, sentiment, temporal parsing)
- Vietnamese AI dev community is growing fast but underserved
- First-mover advantage for Vietnamese-localized AI memory tool
- Cultural alignment: Vietnamese developers appreciate tools built by Vietnamese

### Channels

| Channel | Action |
|---------|--------|
| **Facebook Groups** | Post in Vietnamese AI/Dev groups with demo |
| **Vietnamese Tech Blogs** | Viblo.asia, TopDev, ITviec blog posts |
| **YouTube/TikTok** | Short demo videos in Vietnamese |
| **GitHub Vietnamese** | Vietnamese README, Vietnamese docs |
| **Local Meetups** | Present at Vietnamese AI meetups (online/offline) |

### Content Strategy

1. **Vietnamese README** for GitHub (alongside English)
2. **Vietnamese Getting Started guide** in docs
3. **Vietnamese blog posts** on Viblo.asia
4. **Dashboard default language** detects Vietnamese locale
5. **Vietnamese-specific brain templates** (common Vietnamese dev patterns)

---

*See [VISION.md](VISION.md) for the north star guiding all decisions.*
*Last updated: 2026-02-09 (v1.1.0 nearly complete: SKILL published, blog written, #7273 PR submitted)*
