# NeuralMemory — Instructions for Claude Code

> Copy this section into your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` (global).

## Memory System

This workspace uses **NeuralMemory** for persistent memory across sessions.
You have access to `nmem_*` MCP tools. Use them **proactively** — do not wait for the user to ask.

### Session Start (ALWAYS do this)

```
nmem_recap()                          # Resume context from last session
nmem_context(limit=20)                # Load recent memories
nmem_session(action="get")            # Check current task/feature/progress
```

If `gap_detected: true`, run `nmem_auto(action="flush", text="<recent context>")` to recover lost content.

### During Work — REMEMBER automatically

| Event | Action |
|-------|--------|
| Decision made | `nmem_remember(content="...", type="decision", priority=7)` |
| Bug fixed | `nmem_remember(content="...", type="error", priority=7)` |
| User preference stated | `nmem_remember(content="...", type="preference", priority=6)` |
| Important fact learned | `nmem_remember(content="...", type="fact", priority=5)` |
| TODO identified | `nmem_todo(task="...", priority=6)` |
| Workflow discovered | `nmem_remember(content="...", type="workflow", priority=6)` |

### During Work — RECALL before asking

Before asking the user a question, check memory first:

```
nmem_recall(query="<topic>", depth=1)
```

Depth guide: 0=instant lookup, 1=context (default), 2=patterns, 3=deep graph traversal.

### Session End / Before Compaction

```
nmem_auto(action="process", text="<summary of session>")
nmem_session(action="set", feature="...", task="...", progress=0.8)
```

Before `/compact` or `/new`:
```
nmem_auto(action="flush", text="<recent conversation>")
```

### Project Context

```
nmem_eternal(action="save", project_name="MyProject", tech_stack=["React", "Node.js"])
nmem_eternal(action="save", decision="Use PostgreSQL", reason="Team expertise")
```

### Codebase Indexing

First time on a project:
```
nmem_index(action="scan", path="./src")
```

Then `nmem_recall(query="authentication")` finds related code through the neural graph.

### Knowledge Base Training

Train permanent knowledge from documentation files:
```
# Train from docs directory (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD, TXT, RST)
nmem_train(action="train", path="docs/", domain_tag="react")

# Train a single file
nmem_train(action="train", path="api-spec.pdf")

# Check training status
nmem_train(action="status")
```

Trained knowledge is **pinned** — permanent, no decay, no pruning. Re-training same file is skipped (SHA-256 dedup).

For non-text formats: `pip install neural-memory[extract]`

### Pin/Unpin Memories

```
nmem_pin(fiber_ids=["id1", "id2"], pinned=true)   # Make permanent
nmem_pin(fiber_ids=["id1"], pinned=false)           # Resume lifecycle
```

### Health & Diagnostics

```
nmem_health()                              # Brain health score + warnings
nmem_stats()                               # Memory counts and freshness
nmem_alerts(action="list")                 # Active health alerts
nmem_conflicts(action="list")              # Conflicting memories
nmem_evolution()                           # Brain maturation + plasticity
```

### Spaced Repetition

```
nmem_review(action="queue")                # Get memories due for review
nmem_review(action="mark", fiber_id="...", success=true)  # Record result
```

### Brain Versioning & Transplant

```
nmem_version(action="create", name="pre-refactor")  # Snapshot
nmem_version(action="rollback", version_id="...")    # Restore
nmem_transplant(source_brain="other", tags=["react"])  # Import from another brain
```

### Multi-Device Sync

```
nmem_sync(action="full")                   # Bi-directional sync with hub
nmem_sync_status()                         # Check sync status
nmem_sync_config(action="set", hub_url="https://hub:8080", enabled=true)
```

### Import External Data

```
nmem_import(source="chromadb", connection="/path/to/chroma")
nmem_import(source="mem0", user_id="user123")
```

### Ephemeral Memories

For scratch notes, debugging context, or temporary reasoning:
```
nmem_remember(content="Debugging: token expires at step 3", ephemeral=true)
```

Auto-expires after 24h, never synced to cloud, excluded from consolidation.
Filter them out: `nmem_recall(query="...", permanent_only=true)`

### Compact Mode

All tools support `compact=true` to reduce response tokens by 60-80%.
Use `token_budget=N` to cap response size.
```
nmem_recall(query="auth decisions", compact=true)
nmem_stats(token_budget=200)
```

### Edit & Forget — Correct Mistakes

```
# Fix wrong type (auto-detector got it wrong)
nmem_edit(memory_id="fiber-abc", type="insight")

# Fix wrong content
nmem_edit(memory_id="fiber-abc", content="Corrected: the bug was in auth.py, not login.py")

# Adjust priority
nmem_edit(memory_id="fiber-abc", priority=9)

# Soft delete — memory decays naturally (recommended for outdated info)
nmem_forget(memory_id="fiber-abc", reason="outdated")

# Hard delete — permanent removal (for sensitive data, test garbage)
nmem_forget(memory_id="fiber-abc", hard=true)
```

### Cognitive Reasoning

```
nmem_hypothesize(action="create", content="Redis is the bottleneck", confidence=0.6)
nmem_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")
nmem_predict(action="create", content="Fix will drop latency 50%", hypothesis_id="h-1", deadline="2026-04-01")
nmem_verify(prediction_id="p-1", outcome="correct")
nmem_cognitive(action="summary")           # Hot index
nmem_gaps(action="detect", topic="...")    # Track unknowns
nmem_schema(action="evolve", hypothesis_id="h-1", content="...", reason="...")
```

### Connection Tracing

```
nmem_explain(entity_a="Redis", entity_b="auth outage")
```

Traces shortest path with evidence. Use to debug recall or verify connections.

### Rules

1. **Be proactive** — remember important info without being asked
2. **Store 3-5 memories per task** — a bug fix has: root cause, fix, insight, prevention
3. **Use rich language** — "Chose X over Y because Z" not just "X". Mix causal, temporal, relational, comparative
4. **Check memory first** — recall before asking questions the user may have answered before
5. **Use diverse types** — fact, decision, error, preference, todo, workflow, insight, instruction, context
6. **Set priority** — critical=7-10, normal=5, trivial=1-3
7. **Add tags** — always include project name + topic for better retrieval
8. **Recap on start** — always call `nmem_recap()` at session beginning
9. **Train KB first** — if project has docs/, train them into memory for permanent context
10. **Fix mistakes** — use `nmem_edit` for wrong types/content, `nmem_forget` for outdated info
11. **Health weekly** — `nmem_health()` and fix the highest penalty first
