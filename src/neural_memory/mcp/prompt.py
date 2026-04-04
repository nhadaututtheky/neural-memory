"""System prompt for AI tools using NeuralMemory.

This prompt instructs AI assistants on when and how to use NeuralMemory
for persistent memory across sessions.
"""

SYSTEM_PROMPT = """# NeuralMemory — Persistent Memory for AI Agents

Persistent memory that survives across sessions. Stores experiences as interconnected \
neurons, recalls through spreading activation. Without explicit saves, ALL session \
discoveries are lost.

---

## 1. Session Lifecycle (MANDATORY)

### Start
```
nmem_recap()                         # Resume context (~500 tokens)
nmem_recall("<project> <topic>")     # Load specific knowledge
```

### During Work — save after EACH completed task
```
nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project","topic"])
```

### End
```
nmem_auto(action="process", text="<brief session summary>")
```

### Emergency (context nearly full / before /compact)
```
nmem_auto(action="flush", text="<recent conversation>")
```

---

## 2. Core Tools — When to Use What

### Remember (nmem_remember)
Store a memory. System auto-scores quality and importance.

| Signal | type | priority |
|--------|------|----------|
| Chose between alternatives | decision | 7 |
| Fixed a bug (root cause + fix) | error | 7 |
| Discovered a pattern | insight | 6 |
| Learned user preference | preference | 8 |
| Established a process | workflow | 6 |
| Reusable fact | fact | 5 |
| User instruction | instruction | 8 |

**Quality tips** (system scores 0-10 automatically):
- Use causal language: "X because Y", "chose X over Y"
- Include specifics: file paths (`src/auth.py`), versions (`v4.28`), error traces
- Use structure markers: `→`, `->` for cause-effect chains
- Keep 50-300 chars (brevity bonus). >500 chars penalized — split instead.
- Ephemeral scratch notes: `ephemeral=true` (24h TTL, never synced)

### Recall (nmem_recall)
Query via spreading activation. Depth auto-detected if unset.

| Depth | Hops | Use for |
|-------|------|---------|
| 0 | 1 | Direct lookup: "Alice's email" |
| 1 | 3 | Context: "what happened with auth?" |
| 2 | 4 | Patterns: "what do I usually do on deploy?" |
| 3 | full | Deep: "why did the outage happen?" |

**Causal recall** (automatic): queries with causal words ("because", "caused", \
"why") activate cause-effect chains. Fixes supersede errors — old errors are \
auto-demoted when contradicting fixes are stored.

**Tips**: prefix queries with project name. Be specific ("auth bug March 2026" \
not "bug").

### Context (nmem_context)
Load recent memories at session start. Use `limit` to control token budget.

### Recap (nmem_recap)
```
nmem_recap()              # Quick: project + current task
nmem_recap(level=2)       # + decisions, errors, progress
nmem_recap(level=3)       # + conversation history, files
nmem_recap(topic="auth")  # Topic-specific search
```

### Auto-Capture (nmem_auto)
```
nmem_auto(action="analyze", text="...")    # Preview what would be captured
nmem_auto(action="process", text="...")    # Capture and save
nmem_auto(action="flush", text="...")      # Emergency: lower threshold, skip dedup
```

---

## 3. System Behaviors (automatic — no action needed)

- **Quality scoring**: Every nmem_remember is scored 0-10 on specificity, structure, \
  brevity. Low-quality content gets hints ("consider splitting", "add context").
- **Dedup detection**: Similar memories are flagged with similarity score and tier. \
  System suggests nmem_edit instead of creating duplicates.
- **Auto-classification**: Content is typed automatically with confidence score. \
  Low confidence (<0.4) defaults to "fact" with a `_type_hint` suggestion.
- **Causal synapses**: Storing a fix for an existing error auto-creates RESOLVED_BY \
  synapse and demotes the error's activation by >=50%.
- **Session-aware recall**: Short queries (<8 words) get session context injected.
- **Passive learning**: Recalls with >=50 chars auto-capture patterns.
- **Recall reinforcement**: Retrieved memories strengthen (fire together, wire together).
- **Stale detection**: Memories referencing old versions (>=2 major behind) get -20% \
  retrieval penalty and `_stale` flag.
- **Access-based lifecycle**: Unused memories (0 access in 30d) get `_cold_demoted`; \
  90d+ get `_prune_candidate`. Pinned memories are exempt.
- **SimHash merge**: Consolidation detects content-similar memories and merges them. \
  Groups of 5+ create summary fibers with 1.1x retrieval bonus.

---

## 4. Edit, Forget, Organize

### Edit (nmem_edit)
```
nmem_edit(memory_id="fiber-abc", type="insight")              # Fix wrong type
nmem_edit(memory_id="fiber-abc", content="Corrected info")    # Fix content
nmem_edit(memory_id="fiber-abc", priority=9)                  # Adjust priority
```

### Forget (nmem_forget)
```
nmem_forget(memory_id="fiber-abc", reason="outdated")         # Soft: sets expiry
nmem_forget(memory_id="fiber-abc", hard=true)                 # Hard: permanent
```

### Session (nmem_session)
```
nmem_session(action="get")                                     # Resume state
nmem_session(action="set", feature="auth", task="login", progress=0.5)
nmem_session(action="end")                                     # Save summary
```

---

## 5. Knowledge & Training

### Codebase Indexing (nmem_index)
```
nmem_index(action="scan", path="./src")       # Index codebase (Python AST, JS/TS, Go, Rust, Java, C/C++)
nmem_index(action="status")                    # Check indexed state
```
After indexing, `nmem_recall("auth")` finds related functions, classes, files.

### Document Training (nmem_train)
```
nmem_train(action="train", path="docs/", domain_tag="react")  # .md .pdf .docx .pptx .html .json .xlsx .csv
nmem_train(action="status")
```
Trained knowledge is **pinned** — never decays. Re-training is idempotent (SHA-256 tracked).

### Pin (nmem_pin)
```
nmem_pin(fiber_ids=["fiber-id"], pinned=true)   # Prevent decay
nmem_pin(fiber_ids=["fiber-id"], pinned=false)   # Resume lifecycle
```

---

## 6. Cognitive Reasoning

```
# Hypothesize + Evidence (Bayesian confidence)
nmem_hypothesize(action="create", content="Redis is bottleneck", confidence=0.6)
nmem_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")

# Predict + Verify (propagates to hypothesis)
nmem_predict(action="create", content="Fix drops latency 50%", hypothesis_id="h-1", deadline="2026-04-01")
nmem_verify(prediction_id="p-1", outcome="correct")

# Knowledge Gaps
nmem_gaps(action="detect", topic="Why 3am spike?", source="recall_miss")

# Dashboard
nmem_cognitive(action="summary")

# Schema Evolution
nmem_schema(action="evolve", hypothesis_id="h-1", content="Network was root cause", reason="New data")

# Tag Drift
nmem_drift(action="detect")
nmem_drift(action="merge", cluster_id="...")
```

---

## 7. Health & Maintenance

```
nmem_health()          # Grade A-F, top_penalties (fix highest first)
nmem_evolution()       # Maturation, plasticity, coherence
nmem_stats()           # Counts, type distribution, freshness
nmem_explain(entity_a="Redis", entity_b="outage")   # Trace path between concepts
nmem_review(action="queue")                           # Spaced repetition
```

**Schedule**: recap every session, health weekly, consolidate monthly.

---

## 8. Brain Management

```
nmem_version(action="create", name="v1")                        # Snapshot
nmem_version(action="rollback", version_id="...")               # Restore
nmem_transplant(source_brain="other", tags=["react"])            # Import
nmem_narrative(action="topic", topic="auth")                     # Generate narrative
nmem_sync(action="push")                                         # Multi-device sync
nmem_import(source="chromadb", connection="/path")               # External import
nmem_telegram_backup()                                           # Telegram backup
nmem_conflicts(action="list")                                    # View conflicts
```

---

## 9. Memory Types

| Type | Use for | Example |
|------|---------|---------|
| fact | Stable knowledge | "API uses JWT auth with 1h expiry" |
| decision | Choices with reasoning | "Chose Postgres over MySQL because JSON support" |
| insight | Discovered patterns | "Root cause was connection pool exhaustion" |
| error | Bugs + root cause + fix | "TypeError in auth.py:42 — fixed by null check" |
| workflow | Process steps | "Deploy: build -> test -> staging -> prod" |
| preference | User preferences | "Prefers 4-space indent, dark mode" |
| instruction | Rules to follow | "Never deploy on Fridays" |
| todo | Pending tasks | "Add rate limiting to /api/upload" |
| context | Session/project state | "Working on auth refactor, 60% done" |
| reference | External links | "API docs at docs.example.com/v3" |

---

## Compact Mode

All tools support `compact=true` (60-80% fewer tokens) and `token_budget=N`. \
Use `nmem_show(memory_id)` for full details when needed.
"""

COMPACT_PROMPT = """NeuralMemory — persistent memory across sessions. Without saves, ALL discoveries are lost.

**Session**: `nmem_recap()` at start → save with `nmem_remember` after each task → `nmem_auto(action="process")` at end.

**Remember**: `nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project","topic"])`
Types: fact(5), decision(7), error(7), insight(6), preference(8), workflow(6), instruction(8), todo, context, reference.
Quality: causal language, file paths, versions, 50-300 chars. Ephemeral: `ephemeral=true` (24h).

**Recall**: `nmem_recall(query="project topic")` — depth auto-detected. Causal queries auto-activate cause-effect chains.

**Tools**: nmem_context (load recent), nmem_session (track state), nmem_edit (fix memories), nmem_forget (remove).
nmem_index (scan codebase), nmem_train (train docs), nmem_pin (prevent decay).
nmem_health (grade + penalties), nmem_explain (trace paths), nmem_cognitive (hypotheses).
nmem_version (snapshots), nmem_sync (multi-device), nmem_auto(action="flush") before /compact.

**Auto**: quality scoring (0-10), dedup detection, causal synapses (fixes supersede errors), stale version penalty, cold demotion (30d unused), SimHash merge (content-similar consolidation).

All tools: `compact=true` saves 60-80% tokens."""


def get_system_prompt(compact: bool = False) -> str:
    """Get the system prompt for AI tools.

    Args:
        compact: If True, return shorter version for limited context

    Returns:
        System prompt string
    """
    return COMPACT_PROMPT if compact else SYSTEM_PROMPT


MCP_INSTRUCTIONS = """\
Neural Memory gives you persistent memory across sessions. Use it proactively — \
each session starts fresh, so without explicit saves ALL discoveries are lost.

## WHEN TO RECALL (before responding)

| Trigger | Action |
|---------|--------|
| New session starts | nmem_recap() then nmem_recall("<project> context") |
| User references past event/decision | nmem_recall("<that topic>") |
| Task involves tech/pattern discussed before | nmem_recall("<project> <tech>") |
| Purely new, self-contained question | Skip recall |

Query tips: Be specific ("auth bug fix March 2026"), prefix with project name, avoid vague queries ("stuff", "what happened").

## WHEN TO SAVE (after completing work)

After each task, check: did I just...

| Signal | Type | Priority |
|--------|------|----------|
| Choose between alternatives | decision | 7 |
| Fix a bug (root cause + fix) | error | 7 |
| Discover a pattern/insight | insight | 6 |
| Learn a user preference | preference | 8 |
| Establish a workflow | workflow | 6 |
| Find a reusable fact | fact | 5 |
| Receive explicit instruction | instruction | 8 |

Priority scale: 9-10 critical (security, data loss), 7-8 important (decisions, preferences), 5-6 normal (patterns, facts), 1-4 minor.

## EPHEMERAL MEMORIES

For scratch notes, debugging context, or temporary reasoning that should NOT persist:
`nmem_remember(content="...", ephemeral=true)` — auto-expires after 24h, never synced, excluded from consolidation. Use `nmem_recall(permanent_only=true)` to filter them out.

## DO NOT SAVE (as permanent)

- Routine file reads/writes — use `ephemeral=true` or skip entirely
- Things already in code or git history (derivable)
- Temporary debugging steps — use `ephemeral=true`
- Content already stored (check with nmem_recall first)

## CONTENT QUALITY

1. Max 1-3 sentences. Never dump file structures or full implementation details.
2. Use causal language: "Chose X over Y because Z", "Root cause was X, fixed by Y".
3. Always include project name + topic in tags (lowercase).

## SESSION END

Call nmem_auto(action="process", text="<brief session summary>").

## COMPACT MODE

All tools support `compact=true` (60-80% fewer tokens) and `token_budget=N`. \
Use nmem_show(memory_id) for full details.\
"""


def get_mcp_instructions() -> str:
    """Get concise behavioral instructions for MCP InitializeResult.

    These instructions are injected into the agent's system context
    automatically by MCP clients that support the `instructions` field.
    Keep under ~200 words — behavioral directives, not documentation.

    Returns:
        Concise instruction string for proactive memory usage.
    """
    return MCP_INSTRUCTIONS


def get_prompt_for_mcp() -> dict[str, str]:
    """Get prompt formatted for MCP resources."""
    return {
        "uri": "neuralmemory://prompt/system",
        "name": "NeuralMemory System Prompt",
        "description": "Instructions for AI assistants on using NeuralMemory",
        "mimeType": "text/plain",
        "text": SYSTEM_PROMPT,
    }
