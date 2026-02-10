# NeuralMemory Roadmap

> From associative reflex engine to universal memory platform.
> Every feature passes the VISION.md 4-question test + brain test.
> ZERO LLM dependency — pure algorithmic, regex, graph-based.

**Current state**: v1.6.0 shipped. DB-to-Brain schema training pipeline. 3 composable AI skills. 18 MCP tools, 1648 tests.
**Next milestone**: v1.7.0 — Ecosystem Expansion (Marketplace, Neo4j, multi-language).

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
> **Current state**: v1.6.0 shipped. 1,648 tests. 18 MCP tools. DB-to-Brain schema training pipeline.

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

## Phase 1: v1.1.0 — Community Foundations ✅

> Get noticed. Minimal code, maximum visibility.

**Status**: ✅ Complete. SKILL published, blog written, PR submitted, Issue #1 fixed.

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

~500 lines docs shipped. PR #12596 submitted. Remaining: external publishing (blog to Dev.to/Viblo).

---

## Phase 2: v1.2.0 — Dashboard Foundation ✅

> Full-featured dashboard with OAuth, OpenClaw config, channel setup, neural graph.

**Status**: ✅ Shipped to PyPI (2026-02-09). ~2,800 new lines across 15 files. UX overhaul post-expert review. CI green. Tag `v1.2.0` → PyPI published.

### What shipped

| Feature | Description |
|---------|-------------|
| **Alpine.js + Tailwind SPA** | Zero-build dashboard at `/dashboard` (CDN-loaded, ships with pip install) |
| **5 tab sections** | Overview, Neural Graph, Integrations (status-only), Health, Settings |
| **Cytoscape.js graph** | COSE force-directed layout, 8 neuron type colors, node click → detail panel |
| **Graph toolbar** | Search nodes, filter by type, zoom in/out, fit to view, reload |
| **Integration status** | Status cards for MCP, Nanobot, CLIProxyAPI, OpenClaw, Telegram, Discord with deep links |
| **Health radar chart** | Chart.js radar with 7 diagnostics metrics, warnings, recommendations |
| **Brain management** | Switch brains, export/import JSON, health grade sidebar badge |
| **EN/VI i18n** | Auto-detect browser locale, toggle in settings, full Vietnamese translation (68 keys) |
| **Design system** | Dark mode (#0F172A), Fira Code/Sans fonts, #22C55E CTA green, Lucide icons |
| **Toast notifications** | Global `nmToast()` system, 4s auto-dismiss, success/error/warning/info types |
| **Loading states** | Skeleton shimmer for stats, spinner for graph/health, proper empty states |
| **ARIA accessibility** | `aria-label` on icon buttons, `role="tabpanel"/"tablist"`, `aria-live="polite"` on toasts |
| **Mobile touch targets** | 44px minimum touch targets, responsive nav buttons |

### UX overhaul (post-expert review)

4 UI/UX experts reviewed the initial dashboard. Actionable feedback implemented:

| Fix | Before | After |
|-----|--------|-------|
| **Tab restructure** | 7 separate tabs | 5 tabs — Integrations is status-only with deep links to service dashboards |
| **Loading states** | No loading feedback | Skeleton shimmer for stats, spinner for graph/health |
| **Toast system** | `alert()` for errors | Global `nmToast()` via CustomEvent, auto-dismiss, 4 severity types |
| **Graph toolbar** | No search/filter/zoom | Search + type filter + zoom in/out + fit + reload buttons |
| **Quick Actions** | None | Health Check, Export Brain, View Warnings on Overview |
| **Health summary** | Only in Health tab | Grade + purity score card on Overview |
| **Empty states** | Blank when no data | Placeholder UI for 0 brains, empty graph, no warnings |
| **ARIA labels** | Missing | All icon buttons, tabs, toast container |
| **Mobile** | Small touch targets | 44px min height, proper font size on mobile nav |

### Architecture

```
Browser (SPA — Alpine.js + Tailwind CDN)
  ├── /dashboard ─── dashboard.html (entry point)
  │     ├── Overview       — stats, quick actions, health summary, brain list
  │     ├── Graph          — Cytoscape.js explorer + search/filter/zoom toolbar
  │     ├── Integrations   — Status cards + deep links (config via each service's own dashboard)
  │     ├── Health         — Radar chart, warnings, recommendations
  │     └── Settings       — Language (EN/VI), brain export/import
  │
  FastAPI Backend (existing server at :8000)
  ├── /api/dashboard/*    — Stats, brain list, health (dashboard_api.py)
  ├── /api/oauth/*        — Proxy to CLIProxyAPI :8317 (oauth.py)
  ├── /api/openclaw/*     — Config CRUD from ~/.neuralmemory/openclaw.json (openclaw_api.py)
  └── /api/graph          — Existing graph endpoint
```

### Bugfixes

- **Issue #1**: `ModuleNotFoundError: typing_extensions` on fresh Python 3.12 installs. Root cause: `aiohttp` internal import — environment issue, not NM bug. Defensive fix: added `typing_extensions>=4.0` to dependencies.
- **CI lint**: Fixed `ruff` F821/I001 errors in `test_nanobot_integration.py` (missing `Any` import, unsorted imports).

### Files shipped

| Action | File |
|--------|------|
| NEW | `server/routes/oauth.py` — OAuth proxy to CLIProxyAPI |
| NEW | `server/routes/dashboard_api.py` — Dashboard stats, brain list, health |
| NEW | `server/routes/openclaw_api.py` — OpenClaw config CRUD |
| NEW | `integrations/openclaw_config.py` — Pydantic models + JSON persistence |
| NEW | `server/static/dashboard.html` — Main SPA (Alpine.js tabs + CDN imports) |
| NEW | `server/static/js/app.js` — Core Alpine component (toast, tabs, loading, integrations status, quick actions) |
| NEW | `server/static/js/graph.js` — Cytoscape.js graph (search, filter, zoom) |
| NEW | `server/static/js/oauth.js` — Stub (OAuth managed by CLIProxyAPI) |
| NEW | `server/static/js/openclaw.js` — Stub (config managed by OpenClaw) |
| NEW | `server/static/js/i18n.js` — Locale detection + translation |
| NEW | `server/static/css/dashboard.css` — Custom styles (toast, skeleton, spinner, mobile) |
| NEW | `server/static/locales/en.json` — English strings (68 keys) |
| NEW | `server/static/locales/vi.json` — Vietnamese strings (68 keys) |
| MOD | `server/app.py` — Mount new routers + `/dashboard` endpoint |
| MOD | `server/routes/__init__.py` — Export new routers |
| MOD | `pyproject.toml` — Add httpx + typing_extensions deps, version → 1.2.0 |

---

## Architectural Decision: Dashboard Roles (Option B)

> **Decided 2026-02-09**: NM Dashboard is a specialist tool, not a hub.

### Problem

Three dashboards exist:
1. **CLIProxyAPI** (:8317) — OAuth session management, token management
2. **NeuralMemory** (:8000/dashboard) — Brain management, graph, health
3. **OpenClaw** (CLI) — Agent config, memory settings, extensions

The initial v1.2.0 dashboard tried to be a hub: managing OAuth, OpenClaw config, and channel setup inside NM. This creates duplicate UIs and confusing responsibility boundaries.

### Decision

**NM Dashboard = specialist tool for what ONLY NM can do:**
- Neural graph visualization
- Brain health / diagnostics / radar chart
- Brain switching, versioning, transplant, export/import
- Integration **status** (read-only) with deep links to the right dashboard

**NM Dashboard does NOT manage:**
- OAuth sessions → CLIProxyAPI's dashboard
- OpenClaw config → OpenClaw's own settings
- Channel setup (Telegram/Discord) → CLIProxyAPI or bot platform
- API keys → environment variables or CLIProxyAPI

**Integrations tab = status dashboard**, not config editor:
```
Integrations (read-only status + deep links)
├── MCP Server    ✅ Running — v1.2.0, 16 tools
├── Nanobot       ✅ Available — 4 tools
├── CLIProxyAPI   ✅ 3/6 providers    [Open Dashboard →]
├── OpenClaw      ⬚ Not configured
├── Telegram      ⬚ Not configured    [Configure →]
└── Discord       ⬚ Not configured    [Configure →]
```

### Rationale

- OpenClaw (178k stars) is the user's primary tool — NM is a component inside it
- CLIProxyAPI handles OAuth complexity — NM shouldn't duplicate it
- Specialist dashboards > monolith dashboards
- Deep links to the right place > duplicated config forms

---

## Phase 3: v1.3.0 — Deep Integration Status ✅

> Richer integration status with activity logs and setup wizards.

**Status**: ✅ Shipped (2026-02-09). 12 new tests (1352 total).

### What shipped

| Feature | Description |
|---------|-------------|
| **Enhanced status cards** | Live metrics: memories/recalls today, last call timestamp, error badges |
| **Activity log** | Collapsible feed of recent tool calls with source attribution (MCP/OpenClaw/Nanobot) |
| **Setup wizards** | Accordion config snippets for Claude Code, Cursor, OpenClaw, generic MCP with copy-to-clipboard |
| **Import sources** | Detection panel for ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex (placeholder for v1.5.0) |
| **Source attribution** | `NEURALMEMORY_SOURCE` env var → session_id prefix for integration tracking |
| **i18n** | 25 new keys in EN + VI (87 total) |

### Files shipped

| Action | File |
|--------|------|
| NEW | `server/routes/integration_status.py` — Activity metrics + log endpoint |
| NEW | `tests/unit/test_integration_status.py` — 12 tests for metrics + attribution |
| MOD | `server/routes/__init__.py` — Export new router |
| MOD | `server/app.py` — Mount integration_status_router |
| MOD | `mcp/server.py` — Source attribution via NEURALMEMORY_SOURCE env var |
| MOD | `server/static/dashboard.html` — Enhanced cards, activity log, wizards, import panel |
| MOD | `server/static/js/app.js` — loadActivity, formatTime, setupWizards, importSources, copySnippet |
| MOD | `server/static/css/dashboard.css` — Code snippet styling |
| MOD | `server/static/locales/en.json` — 25 new keys |
| MOD | `server/static/locales/vi.json` — 25 new keys |

---

## Phase 4: v1.4.0 — OpenClaw Memory Plugin ✅

> NM becomes the memory layer inside OpenClaw (the hub).

**Status**: ✅ Shipped. `@neuralmemory/openclaw-plugin@1.4.1` published to npm. 6 tools, 2 hooks, 1 service. Verified in real OpenClaw (loaded among 35 plugins).

### Why This Is P1

Option B says OpenClaw is the hub. For that to work, NM must plug into OpenClaw seamlessly. This is the highest-impact integration: 178k-star ecosystem, exclusive memory slot, every OpenClaw user becomes a potential NM user.

### Architecture

```
OpenClaw (the hub — user's primary interface)
  │
  ├── Memory Slot (exclusive — kind: "memory")
  │   └── @neuralmemory/openclaw-plugin (TypeScript, zero deps)
  │       ├── openclaw.plugin.json  — manifest with configSchema + uiHints
  │       ├── src/index.ts          — plugin entry, registers tools/hooks/service
  │       ├── src/mcp-client.ts     — JSON-RPC 2.0 over stdio (Content-Length framing)
  │       ├── src/tools.ts          — 6 core tools with zod schemas
  │       └── src/types.ts          — minimal OpenClaw type stubs
  │
  ├── Service: neuralmemory-mcp
  │   └── Spawns `python -m neural_memory.mcp` as subprocess
  │
  ├── Hooks (configurable):
  │   ├── before_agent_start → nmem_recall(prompt) → prependContext
  │   └── agent_end → nmem_auto(process, text) → auto-capture
  │
  └── Config via uiHints:
      pythonPath, brain, autoContext, autoCapture, contextDepth, maxContextTokens
```

### 4.1 MCP stdio Client ✅

Zero-dependency JSON-RPC 2.0 client implementing MCP protocol with Content-Length framing. Handles initialize handshake, tool calls, timeouts, process lifecycle.

Key design:
- Manual protocol implementation (no `@modelcontextprotocol/sdk` dependency)
- `connect()` → spawns Python process + MCP initialize handshake
- `callTool(name, args)` → JSON-RPC request with timeout
- `close()` → SIGTERM + pending request cleanup
- Buffer-based response parsing with Content-Length header extraction

### 4.2 Tools Registered ✅

6 core tools registered via `api.registerTool()` with zod parameter schemas:

| Tool | Description | Parameters |
|------|-------------|------------|
| `nmem_remember` | Store a memory | content, type?, priority?, tags?, expires_days? |
| `nmem_recall` | Query/search memories | query, depth?, max_tokens?, min_confidence? |
| `nmem_context` | Get recent context | limit?, fresh_only? |
| `nmem_todo` | Quick TODO shortcut | task, priority? |
| `nmem_stats` | Brain statistics | (none) |
| `nmem_health` | Brain health diagnostics | (none) |

Each tool proxies to MCP via `callTool()` with JSON parse of response.

### 4.3 OpenClaw Status Fix ✅

[PR #12596](https://github.com/openclaw/openclaw/pull/12596) submitted — fixes `openclaw status` to probe non-core memory plugins via gateway RPC. 2-file fix, 5/5 tests pass. Competing PR #7289 had 32 files.

### 4.4 NM Dashboard Integration Card ✅

When OpenClaw plugin is active, NM dashboard's Integrations status card shows:
- "OpenClaw: ✅ Connected — N API keys configured"
- No config UI (Option B — config lives in OpenClaw's settings via uiHints)

### Files

| Status | File | Lines | Description |
|--------|------|-------|-------------|
| ✅ | `integrations/openclaw-plugin/openclaw.plugin.json` | 57 | Plugin manifest: id, kind, configSchema, uiHints |
| ✅ | `integrations/openclaw-plugin/package.json` | 33 | npm package (zero runtime deps, zod as peer) |
| ✅ | `integrations/openclaw-plugin/tsconfig.json` | 15 | TypeScript config (ES2022, bundler resolution) |
| ✅ | `integrations/openclaw-plugin/src/index.ts` | 156 | Plugin entry: register tools, service, hooks |
| ✅ | `integrations/openclaw-plugin/src/mcp-client.ts` | 198 | MCP stdio client: JSON-RPC 2.0, Content-Length framing |
| ✅ | `integrations/openclaw-plugin/src/tools.ts` | 152 | 6 core tools with zod schemas |
| ✅ | `integrations/openclaw-plugin/src/types.ts` | 73 | Minimal OpenClaw plugin type stubs |

### Remaining

- [ ] Integration test: install plugin in OpenClaw, verify tools work end-to-end
- [ ] Publish to npm as `@neuralmemory/openclaw-plugin`
- [ ] Update ClawHub SKILL.md with plugin installation instructions
- [ ] Add to `~/.openclaw/extensions/` discovery path documentation

---

## Phase 5: v1.5.0 — Conflict Detection + Quality Hardening ✅

> Surface silent conflict detection as user-facing tool. Harden quality via deep audit.

**Status**: ✅ Shipped (2026-02-10). 1372 tests. Published to PyPI.

### What shipped

| Feature | Description |
|---------|-------------|
| **`nmem_conflicts` MCP tool** | List, resolve, and pre-check memory conflicts — surfaces previously silent detection |
| **ConflictHandler mixin** | `list` (view active CONTRADICTS), `resolve` (keep_existing/keep_new/keep_both), `check` (pre-check content) |
| **Recall conflict surfacing** | `has_conflicts` flag + `conflict_count` in default recall response; full details opt-in via `include_conflicts=true` |
| **Remember conflict reporting** | `conflicts_detected` count in `nmem_remember` response when conflicts found |
| **Stats conflict count** | `conflicts_active` in `nmem_stats` response |
| **Provenance tracking** | `NEURALMEMORY_SOURCE` env var → `mcp:{source}` provenance with `mcp_tool` fallback |
| **Purity score penalty** | Unresolved CONTRADICTS reduce health score (max -10 points), `HIGH_CONFLICT_COUNT` warning |
| **Pre-dispute activation** | `_pre_dispute_activation` saved before anti-Hebbian update, restored on `keep_existing` resolve |
| **`_conflict_resolved` guard** | Prevents re-flagging after manual resolution |

### Deep audit fixes (included in v1.5.0)

| Category | Fixes |
|----------|-------|
| **Performance** | 3× `get_all_synapses()` → filtered `get_synapses(type=CONTRADICTS)`, diagnostics double fiber fetch eliminated |
| **Logic** | `_resolve_keep_new` missing `_disputed=False`, recall field name alignment |
| **Security** | Error messages no longer leak exceptions, tag validation (max 50/100), `NEURALMEMORY_SOURCE` truncated to 256 |
| **Quality** | UUID regex case-insensitive, deprecated `datetime.utcnow()` replaced, ruff B905 lint fix |
| **Tests** | 9 new audit-driven tests for edge cases (deleted neurons, missing synapses, invalid inputs, error sanitization) |

### Files

| Action | File |
|--------|------|
| NEW | `mcp/conflict_handler.py` — ConflictHandler mixin (list/resolve/check) |
| NEW | `tests/unit/test_conflict_handler.py` — 20 tests |
| MOD | `mcp/server.py` — Conflict surfacing in recall/remember/stats, provenance enrichment |
| MOD | `mcp/tool_schemas.py` — `nmem_conflicts` schema + `include_conflicts` on recall |
| MOD | `engine/conflict_detection.py` — `_conflict_resolved` guard, `_pre_dispute_activation` save |
| MOD | `engine/encoder.py` — `conflicts_detected` in `EncodingResult` |
| MOD | `engine/retrieval.py` — `disputed_ids` in retrieval metadata |
| MOD | `engine/diagnostics.py` — Conflict penalty + warning, single fiber fetch |
| MOD | `mcp/auto_handler.py` — ruff B905 lint fix |

---

## Phase 6: v1.6.0 — DB-to-Brain Schema Training ✅

> Teach your brain to understand database structure — zero LLM, pure schema introspection.

**Status**: ✅ Shipped (2026-02-10). 1,648 tests. 18 MCP tools. Published to PyPI.

### What shipped

| Feature | Description |
|---------|-------------|
| **SchemaIntrospector** | SQLite dialect using PRAGMA statements, frozen dataclasses, SHA256 fingerprinting |
| **KnowledgeExtractor** | FK-to-SynapseType mapping (IS_A, INVOLVES, AT_LOCATION, RELATED_TO) with confidence scoring |
| **Join table detection** | Structure-based (not name-based) — composite PK from FKs, no meaningful text cols |
| **5 pattern detectors** | Audit trail, soft delete, tree hierarchy, polymorphic, enum table — all with confidence |
| **DBTrainer** | Batch encode via MemoryEncoder, per-table error isolation, optional ENRICH consolidation |
| **`nmem_train_db` MCP tool** | `train` and `status` actions, validates connection strings, max_tables guard |
| **Security hardening** | Read-only SQLite (`?mode=ro`), absolute path rejection, SQL identifier sanitization |

### Architecture

```
Connection String (sqlite:///path)
    │
SchemaIntrospector  (SQLite dialect, PRAGMA-based)
    │
SchemaSnapshot  (frozen: tables, columns, FKs, indexes, fingerprint)
    │
KnowledgeExtractor  (confidence-scored FK mapping + pattern detection)
    │
SchemaKnowledge  (entities, relationships, patterns — all with confidence)
    │
DBTrainer  (batch encode via MemoryEncoder + direct relationship synapses)
    │
Brain  (neurons = tables/patterns, synapses = relationships)
```

### Files

| Action | File |
|--------|------|
| NEW | `engine/db_introspector.py` — Schema introspection (SQLite dialect) |
| NEW | `engine/db_knowledge.py` — Schema → teachable knowledge extraction |
| NEW | `engine/db_trainer.py` — DB-to-Brain training orchestrator |
| NEW | `mcp/db_train_handler.py` — MCP tool handler (train/status) |
| MOD | `mcp/server.py` — Register DBTrainHandler mixin |
| MOD | `mcp/tool_schemas.py` — `nmem_train_db` schema |

### Composable AI Agent Skills (post-release)

3 skills following the [ship-faster](https://github.com/Heyvhuang/ship-faster) SKILL.md pattern, installable to `~/.claude/skills/`:

| Skill | Stage | Description |
|-------|-------|-------------|
| **memory-intake** | workflow | Messy notes → structured memories. 1-question-at-a-time clarification, 6-phase pipeline (triage → clarify → enrich → dedup → batch store → report) |
| **memory-audit** | review | 6-dimension quality review (purity 25%, freshness 20%, coverage 20%, clarity 15%, relevance 10%, structure 10%). A-F grading, prioritized findings |
| **memory-evolution** | workflow | Evidence-based optimization from usage patterns. Consolidation, enrichment, pruning, tag normalization, priority rebalancing. Checkpoint Q&A between cycles |

Invoke via `/memory-intake`, `/memory-audit`, `/memory-evolution` in Claude Code.

---

## Phase 7: v1.7.0 — Ecosystem Expansion

> Marketplace, storage backends, more languages.

**Target**: After v1.6.0

### 7.1 Brain Marketplace (Preview)

NM dashboard gains a "Marketplace" tab (6th tab — the exception where NM IS the hub):
- Public brain gallery
- Upload/download brains with quality badges (grade A/B required)
- Categories: programming, devops, security, data-science, vietnamese-dev
- Search by tags, language, grade
- Brain transplant from marketplace → local brain

### 7.2 Lab Brains — Build-and-Ship Knowledge Pipelines

> Clone → build workflow → train brain → publish → delete source. Labs are temporary.

#### Why Labs exist

Some platforms (e.g. AntiGravity Google) **do not provide public APIs**. The only way to obtain auth credentials is:

1. Clone a proxy tool (e.g. [Clipproxy](https://github.com/AltimateAI/clipproxy)) locally
2. Run its OAuth flow to authenticate a Google account
3. Extract the resulting credentials (tokens, session data)
4. Build a custom API workflow from those credentials

This is a **high-friction onboarding wall** for OpenClaw users — many give up before getting a working setup. Labs solve this by packaging the entire workflow into a reproducible, trainable pipeline.

#### Lifecycle

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ 1. Clone     │────→│ 2. Build     │────→│ 3. Train      │────→│ 4. Publish │
│ git repo     │     │ workflow     │     │ nmem_train    │     │ to Market  │
│ into lab/    │     │ (setup,test) │     │ → brain JSON  │     │ brain JSON │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                                                                      │
                                          ┌───────────────┐           │
                                          │ 5. Cleanup    │←──────────┘
                                          │ rm -rf lab/   │
                                          │ (source gone, │
                                          │  brain lives) │
                                          └───────────────┘
```

**Key principle**: The cloned repo is scaffolding. Once the knowledge is encoded into a brain, the source is deleted. The brain is the deliverable, not the repo.

#### Lab structure (draft)

```
labs/                                    # .gitignored — ephemeral workspace
├── clipproxy-oauth/                     # Lab 1: OAuth via Clipproxy
│   ├── README.md                        #   Step-by-step setup guide
│   ├── scripts/
│   │   ├── setup.sh                     #   Clone Clipproxy, install deps
│   │   ├── run-oauth.sh                 #   Start OAuth flow
│   │   └── extract-credentials.sh       #   Export tokens
│   └── docs/                            #   Trainable docs for nmem_train
│       ├── oauth-flow.md                #   How the OAuth flow works
│       ├── troubleshooting.md           #   Common errors + fixes
│       └── credential-management.md     #   Token refresh, expiry, rotation
│
├── openclaw-quickstart/                 # Lab 2: OpenClaw first-run
│   ├── README.md
│   ├── scripts/setup.sh
│   └── docs/
│       ├── install-guide.md
│       ├── first-agent.md
│       └── mcp-config.md
│
└── antigravity-api-builder/             # Lab 3: Custom API from OAuth creds
    ├── README.md
    ├── scripts/build-api.sh
    └── docs/
        ├── api-construction.md
        └── rate-limits-and-quotas.md
```

#### CLI workflow (target)

```bash
# 1. Init lab from template
nmem lab init clipproxy-oauth

# 2. User follows setup (interactive or scripted)
cd labs/clipproxy-oauth && ./scripts/setup.sh

# 3. Train brain from lab docs
nmem train --path labs/clipproxy-oauth/docs/ \
           --domain-tag clipproxy-oauth \
           --brain-name clipproxy-setup

# 4. Verify brain quality
nmem health                              # Must be grade B+ for marketplace

# 5. Publish to marketplace
nmem brain publish clipproxy-setup       # Upload brain JSON to marketplace

# 6. Cleanup — source is scaffolding
rm -rf labs/clipproxy-oauth/             # Brain lives on, repo is gone
```

#### End-user experience (consumer side)

```bash
# New OpenClaw user, day 1:
nmem brain install clipproxy-setup       # Download from marketplace

nmem recall "how to activate Google OAuth via Clipproxy"
# → Step-by-step: clone repo, run setup.sh, authorize, extract tokens

nmem recall "Clipproxy OAuth token expired what do I do"
# → Token refresh procedure, common errors, rotation strategy
```

#### Initial Lab Brains (launch inventory for marketplace)

| Brain | Source | Pain point solved |
|-------|--------|-------------------|
| `clipproxy-oauth` | Clipproxy repo + custom docs | Google OAuth setup via proxy |
| `openclaw-quickstart` | OpenClaw docs + setup scripts | First-run onboarding |
| `antigravity-api-builder` | Custom workflow docs | Building API without official endpoints |
| `nm-best-practices` | NeuralMemory docs + guides | NM power-user patterns |
| `mcp-server-setup` | MCP spec + NM integration docs | MCP configuration for any agent |

#### Relationship to 7.1 Marketplace

Labs are the **content pipeline** for marketplace. Marketplace is the shelf — Labs produce the inventory. Without Labs, marketplace launches empty. Without marketplace, lab brains stay local.

```
Labs (produce) ──→ Brains (train) ──→ Marketplace (distribute) ──→ Users (consume)
```

### 7.3 Neo4j Storage Backend

For users with large-scale graph requirements:
- `Neo4jStorage` implementing `BaseStorage` ABC
- Native graph queries (Cypher) for complex traversals
- Horizontal scaling for enterprise use
- Optional — SQLite remains default
- Dashboard graph tab auto-detects Neo4j and uses native traversal

### 7.4 Multi-Language Expansion

Beyond EN/VI:
- Japanese (large AI dev community)
- Korean
- Chinese (simplified)
- Extraction patterns + lexicons per language
- Dashboard i18n keys for JA/KO/ZH

---

## Phase 8: v2.0.0 — Platform

> NeuralMemory becomes the universal memory layer for AI agents.

### Vision

```
Any AI Agent (OpenClaw, Claude Code, Cursor, THOR, Nanobot, custom)
  │
  └── NM Protocol (MCP/REST/SDK) → Neural Graph → Intelligent Recall
                                        │
                                   NM Dashboard (specialist tool)
                                   ├── Brain management + health
                                   ├── Neural graph explorer
                                   ├── Integration status
                                   └── Brain Marketplace
```

**Dashboard role at v2.0**: Still a specialist tool (Option B). But now with marketplace, it becomes the place where brains are browsed, traded, transplanted. Agents configure their own memory via MCP — dashboard is for humans observing and managing the neural graphs.

### 2.0 Features

- **NM Protocol**: Standardized memory protocol beyond MCP (REST + WebSocket + SDK)
- **Multi-brain reasoning**: Query across multiple brains simultaneously
- **Federated memory**: Distributed brains across machines
- **Real-time collaboration**: Multiple agents sharing one brain with conflict resolution
- **Memory compression**: Lossy summarization for very old fibers (save storage)
- **Adaptive recall**: Learn which depth level works best per query pattern
- **Dashboard real-time**: WebSocket-powered live graph updates as agents remember/recall

---

## Dependency Graph (Post-v1.0)

```
v1.1.0 ✅ (Community Foundations)
  ├──→ v1.2.0 ✅ (Dashboard — specialist tool + status-only integrations)
  │       └──→ v1.3.0 ✅ (Deep Integration Status + activity logs)
  └──→ v1.4.0 ✅ (OpenClaw Memory Plugin — NM inside the hub)
              └──→ v1.5.0 ✅ (Conflict Detection + Quality Hardening)
                        └──→ v1.6.0 ✅ (DB-to-Brain Schema Training)
                                  └──→ v1.7.0 (Ecosystem — Marketplace, Neo4j, i18n)
                                            └──→ v2.0.0 (Platform — protocol, federation, real-time)
```

**Critical path**: v1.1.0 ✅ → v1.2.0 ✅ → v1.3.0 ✅ → v1.4.0 ✅ → v1.5.0 ✅ → v1.6.0 ✅ → v1.7.0

**Next**: v1.7.0 — Ecosystem Expansion (Marketplace, Neo4j, multi-language)

---

## Priority Matrix

| Phase | Impact | Effort | Priority | Rationale |
|-------|--------|--------|----------|-----------|
| **v1.1.0** ✅ | High | Low | **P0** | Community visibility, zero code |
| **v1.2.0** ✅ | High | High | **P1** | Dashboard is the face of NM |
| **v1.3.0** ✅ | Medium | Low | **P2** | Richer status + activity logs, shipped |
| **v1.4.0** ✅ | Critical | Medium | **P1** | 178k-star ecosystem — NM inside the hub, shipped |
| **v1.5.0** ✅ | High | Medium | **P1** | Conflict detection surfaced, quality hardening, shipped |
| **v1.6.0** ✅ | High | Medium | **P2** | DB-to-Brain schema training, 18 MCP tools, shipped |
| **v1.7.0** | High | High | **P3** | Marketplace + Neo4j + i18n, can be incremental |
| **v2.0.0** | Critical | Very High | **P4** | Protocol + federation + real-time, long-term vision |

### Recommended Execution Order

```
v1.1.0 ✅ → v1.2.0 ✅ → v1.4.0 ✅ → v1.3.0 ✅ → v1.5.0 ✅ → v1.6.0 ✅ → v1.7.0
```

All post-v1.0 milestones through v1.6.0 complete. Next: v1.7.0 Ecosystem Expansion.

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
*Last updated: 2026-02-10 (v1.6.0 shipped: DB-to-Brain schema training pipeline + 3 composable AI skills. 18 MCP tools, 1648 tests.)*
