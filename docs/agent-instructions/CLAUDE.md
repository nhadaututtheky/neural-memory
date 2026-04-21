# Neural Memory — Quickstart for Claude Code

> Copy this into your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` (global).
> MCP-aware clients auto-receive behavioral instructions on connect — this file is a concise reference. Full prompt: `nmem_show_prompt` or `src/neural_memory/mcp/prompt.py`.

## Session Lifecycle

```
nmem_recap()                          # Start of session — resume context
nmem_recall("<project> <topic>")      # Before answering from memory
nmem_remember(content="...", type="...", priority=7, tags=["project","topic"])  # After each task
nmem_auto(action="process", text="<summary>")   # End of session
nmem_auto(action="flush", text="<recent>")       # Before /compact or /new
```

## When to Remember

| Signal | type | priority |
|--------|------|----------|
| Chose between alternatives | decision | 7 |
| Fixed a bug (root cause + fix) | error | 7 |
| Discovered a pattern | insight | 6 |
| User preference stated | preference | 8 |
| Established workflow | workflow | 6 |
| Reusable fact | fact | 5 |
| User instruction | instruction | 8 |

Priorities: 9-10 critical · 7-8 important · 5-6 normal · 1-4 minor.

## Content Quality (system scores 0-10 automatically)

- Causal language: `"Chose X over Y because Z"`, `"Root cause was X, fixed by Y"`
- Include specifics: file paths, versions, error messages
- 50-300 chars sweet spot. >500 chars penalized — split instead.
- Prefix tags + queries with project name to avoid cross-project noise.
- Scratch / debug notes: `ephemeral=true` (24h TTL, never synced, never consolidated).

## Recall Depth

| Depth | Hops | Use for |
|-------|------|---------|
| 0 | 1 | Direct lookup ("Alice's email") |
| 1 | 3 | Context ("what happened with auth?") |
| 2 | 4 | Patterns ("what do I usually do on deploy?") |
| 3 | full | Deep tracing ("why did the outage happen?") |

## Core Tools

- `nmem_remember` · `nmem_recall` · `nmem_context` · `nmem_recap` — daily use
- `nmem_edit` · `nmem_forget` · `nmem_pin` — correction
- `nmem_index` · `nmem_train` — codebase + docs ingestion
- `nmem_hypothesize` · `nmem_evidence` · `nmem_predict` · `nmem_verify` — reasoning chain
- `nmem_health` · `nmem_stats` · `nmem_evolution` — diagnostics
- `nmem_version` · `nmem_sync` · `nmem_import` · `nmem_transplant` — brain management

All tools accept `compact=true` (60-80% fewer tokens) and `token_budget=N`.

## DO NOT Save

- Routine file reads/writes — use `ephemeral=true` or skip
- Things derivable from code or git history
- Content already stored (check with `nmem_recall` first)
- Wall-of-text memories — split into focused pieces

## Rules

1. Recall before asking.
2. Save 2-5 memories per completed task — don't batch.
3. Prefix queries with project name.
4. Use causal / temporal / relational language, not flat facts.
5. Recap on session start, process on session end.
