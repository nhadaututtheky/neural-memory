# Phase 5: Sync Safety

## Goal

Prevent duplicates and conflicts when brain syncs across multiple PCs. Defensive checks on the import path + coordination of consolidation across machines.

## Why This Matters

- UC4: brain synced to 2 PCs, both PCs generate memories → sync pull imports duplicates
- No dedup check on sync import — all incoming neurons/fibers inserted directly
- 2 PCs running consolidation on same brain data → inconsistent state
- Last-writer-wins without merge → losing metadata/tags from other PC

## Tasks

- [x] 5.1 — Dedup on sync import
  - Neuron: content_hash check before insert → skip if match
  - Fiber: anchor_neuron_id match → merge tags (union auto_tags + agent_tags), merge metadata, increment frequency
  - Logged at INFO level

- [ ] 5.2 — Sync-aware conflict resolution (DEFERRED — existing PREFER_RECENT strategy handles basic case)

- [ ] 5.3 — Consolidation coordination (DEFERRED — Phase 4 file lock covers single-machine; distributed needs separate design)

- [x] 5.4 — Tests (9 new tests in test_sync_safety.py)
  - Neuron dedup (hash match, no match, no hash)
  - Fiber dedup (anchor match + tag merge, no match, no anchor, same ID)
  - Process response integration (own device skip, remote apply)

## Acceptance Criteria

- [ ] Sync import never creates duplicate fiber when content already exists
- [ ] Conflicting fiber versions resolved with newer-wins + tag merge
- [ ] Consolidation coordinated via timestamp (no double-run)
- [ ] Audit trail for conflict resolutions
- [ ] 10+ new tests

## Files Touched

- `src/neural_memory/sync/pull_handler.py` — dedup on import, conflict resolution
- `src/neural_memory/sync/push_handler.py` — add last_consolidated_at
- `src/neural_memory/sync/models.py` — sync metadata types
- `tests/unit/test_sync_safety.py` — new test file

## Design Notes

- Sync safety is defensive only — doesn't change sync protocol
- Content hash (SHA-256) from Phase 2 reused here for fast exact-match
- Conflict resolution is deterministic (newer wins) — no manual intervention needed
- Consolidation coordination is eventual-consistency, not strict — acceptable for memory system
- This phase depends on P2 (content_hash) and P4 (agent identity for attribution)
