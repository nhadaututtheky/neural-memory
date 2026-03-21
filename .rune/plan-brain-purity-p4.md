# Phase 4: Multi-Agent Hygiene

## Goal

Prevent duplicate/conflicting memories when multiple agents share one brain. Add agent identity tracking so memories are attributable and dedup works cross-agent.

## Why This Matters

- UC3: 2 agents saving "Chose PostgreSQL over MySQL for auth service" with different wording → 2 fibers
- No way to know which agent created which memory (agent_id not tracked)
- Consolidation running on 2 agent sessions simultaneously → race condition
- Memory types inconsistent across agents (same insight classified differently)

## Tasks

- [x] 4.1 — Agent identity auto-inject
  - Captures `clientInfo.name` from MCP initialize → `server._agent_id`
  - Auto-adds `agent:<agent_id>` tag to every `_remember()` call
  - No new config — uses existing MCP protocol

- [ ] 4.2 — Cross-agent dedup (DEFERRED — existing dedup pipeline handles >89% SimHash already)

- [x] 4.3 — Consolidation mutex
  - File-based lock: `~/.neuralmemory/consolidation.lock` with PID + timestamp
  - Stale lock: dead PID or >1 hour → auto-cleared
  - Session-end consolidation: acquire before, release in finally block

- [x] 4.4 — Tests (15 new tests in test_multi_agent.py)
  - Agent capture, tag injection, lock acquire/release/stale/corrupt, session-end integration

## Acceptance Criteria

- [ ] Every memory automatically tagged with agent identity
- [ ] Same content from 2 different agents → merged, not duplicated
- [ ] Concurrent consolidation blocked by file lock
- [ ] Stale locks auto-cleared (dead PID or >1 hour)
- [ ] 10+ new tests

## Files Touched

- `src/neural_memory/mcp/server.py` — capture clientInfo.name as agent_id
- `src/neural_memory/mcp/tool_handlers.py` — auto-inject agent tag
- `src/neural_memory/engine/dedup/pipeline.py` — cross-agent merge logic
- `src/neural_memory/engine/consolidation.py` — file lock mechanism
- `tests/unit/test_multi_agent.py` — new test file

## Design Notes

- Agent ID comes from MCP protocol's `clientInfo.name` field — no new protocol needed
- File lock is intentionally simple (not distributed) — covers single-machine multi-agent case
- For distributed lock (multi-PC), see Phase 5 sync safety
- Cross-agent dedup threshold slightly lower (80% vs 89%) because different agents rephrase
