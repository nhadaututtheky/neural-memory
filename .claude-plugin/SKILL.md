# Neural Memory

Persistent memory for AI agents — stores experiences as interconnected neurons, recalls through spreading activation.

## Usage

**RECALL** — before responding to tasks that reference past work:
- New session → `nmem_recall("current project context")`
- Past decision/event → `nmem_recall("<project> <topic>")`
- Skip for purely new, self-contained questions

**SAVE** — after completing each task, if you made a decision, fixed a bug, learned a preference, or discovered a pattern:
- `nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project", "topic"])`
- Use causal language (not flat facts). Max 1-3 sentences.
- Do NOT save routine file reads, things in git history, or duplicates.

**EPHEMERAL** — for scratch notes, debugging context, temporary reasoning:
- `nmem_remember(content="...", ephemeral=true)` — auto-expires after 24h, never synced.

**FLUSH** — at session end:
- `nmem_auto(action="process", text="brief summary")`

**COMPACT** — all tools support `compact=true` (saves 60-80% tokens) and `token_budget=N`.

## Memory Types

| Type | Use For |
|------|---------|
| fact | Stable knowledge |
| decision | "Chose X over Y because Z" |
| insight | Patterns discovered |
| error | Bugs and root causes |
| workflow | Process steps |
| preference | User preferences |
| instruction | Rules to follow |

## Links

- [GitHub](https://github.com/nhadaututtheky/neural-memory)
- [Documentation](https://nhadaututtheky.github.io/neural-memory)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
