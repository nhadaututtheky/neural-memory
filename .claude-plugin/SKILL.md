# Neural Memory

Persistent memory for AI agents — stores experiences as interconnected neurons, recalls through spreading activation.

## Session Lifecycle

1. **Start**: `nmem_recap()` → `nmem_recall("<project> <topic>")`
2. **During**: `nmem_remember(...)` after each completed task
3. **End**: `nmem_auto(action="process", text="brief summary")`
4. **Emergency**: `nmem_auto(action="flush", text="...")` before /compact

## Save — When & How

```
nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project", "topic"])
```

| Signal | type | priority |
|--------|------|----------|
| Chose between alternatives | decision | 7 |
| Fixed a bug (root cause + fix) | error | 7 |
| Discovered a pattern | insight | 6 |
| Learned user preference | preference | 8 |
| Established a process | workflow | 6 |
| Reusable fact | fact | 5 |
| User instruction | instruction | 8 |

**Quality** (system scores 0-10 automatically):
- Causal language: "X because Y", "chose X over Y", "root cause was X"
- Include specifics: file paths, versions, error traces
- 50-300 chars optimal. >500 chars penalized — split instead.
- Scratch notes: `ephemeral=true` (24h, never synced)
- Do NOT save: routine file reads, things in git, duplicates

## Recall

```
nmem_recall(query="project auth bug")   # depth auto-detected
```

- Causal queries ("why", "because", "caused") activate cause-effect chains
- Fixes auto-supersede errors (old errors demoted when fix is stored)
- Prefix queries with project name for accuracy

## Smart Behaviors (automatic)

- **Quality scoring**: specificity, structure, brevity scored per memory
- **Dedup**: similar memories flagged — use nmem_edit instead of duplicating
- **Causal synapses**: fixes create RESOLVED_BY links, demote old errors
- **Stale detection**: old-version references get -20% retrieval penalty
- **Cold demotion**: 0 access in 30d → cold; 90d → prune candidate (pinned exempt)
- **SimHash merge**: content-similar memories consolidated; 5+ group → summary fiber

## Key Tools

| Tool | When |
|------|------|
| `nmem_recall` | Query memories (depth auto-detected) |
| `nmem_context` | Load recent memories at session start |
| `nmem_recap` | Resume session (level=1-3, topic="X") |
| `nmem_session` | Track session state (get/set/end) |
| `nmem_edit` | Fix wrong type/content/priority |
| `nmem_forget` | Remove outdated memories (soft/hard) |
| `nmem_index` | Scan codebase into memory graph |
| `nmem_train` | Train docs into permanent knowledge |
| `nmem_health` | Brain grade + top penalties to fix |
| `nmem_explain` | Trace path between two concepts |
| `nmem_cognitive` | Hypotheses + predictions dashboard |

All tools: `compact=true` saves 60-80% tokens.

## Links

- [GitHub](https://github.com/nhadaututtheky/neural-memory)
- [Documentation](https://nhadaututtheky.github.io/neural-memory)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
