# Phase 3: Ghost Recall + Recall Key

## Goal
Agent sees ghost memories with enough info to decide recall. Includes a deterministic
recall key (fiber ID) so recall is exact — not fuzzy search.

## Ghost Format
```
[~] decision | 14d ago | auth, PostgreSQL | 3 links | recall:fiber:abc123
```
- Type + age + top 3 tags + connection count + recall key
- Recall key = `fiber:{fiber_id}` — deterministic, exact match

## Tasks
- [x] Add `include_ghosts: bool = True` param to `nmem_context` tool schema
- [x] Add `render_ghost(fiber) -> str` in `engine/fidelity.py`:
  - Format: `[~] {type} | {age_human} | {tags_csv} | {link_count} links | recall:fiber:{id}`
  - Age: "2d ago", "3w ago", "2mo ago" — human-readable
  - Tags: top 3 by frequency, comma-separated
  - Link count: synapse count for anchor neuron
- [x] Ghost section in context output:
  - Separator: `\n--- faded memories (use recall key to restore) ---\n`
  - Ghost entries after separator
  - Include `include_ghosts=false` to suppress
- [x] Add `nmem_recall` enhancement: accept `fiber:{id}` recall key
  - If query starts with `fiber:`, do exact fiber lookup → return full content
  - This is the "hard recall primitive" (Expert 3 feedback)
- [x] Add `last_ghost_shown_at: datetime | None` to fiber metadata
  - Updated when fiber is rendered as ghost in context
  - Schema migration v35: `ALTER TABLE fibers ADD COLUMN last_ghost_shown_at TEXT`
- [x] Ghost visibility boost: if `last_ghost_shown_at` < 24h, boost recall score by 0.1
  - Rationale: agent recently saw this ghost, it's contextually relevant
- [x] Tests: ghost format, recall key exact match, visibility boost, include_ghosts=false

## Acceptance Criteria
- [x] Ghost entries show at bottom of context with separator
- [x] `nmem_recall "fiber:abc123"` returns exact full content
- [x] Ghost includes enough info for agent to decide recall
- [x] include_ghosts=false suppresses ghost section entirely
- [x] Visibility boost works: recently-shown ghosts rank higher in recall

## Files Touched
- `src/neural_memory/engine/fidelity.py` — modify (ghost render)
- `src/neural_memory/engine/context_optimizer.py` — modify (ghost section)
- `src/neural_memory/mcp/tool_handlers.py` — modify (recall key, ghost param)
- `src/neural_memory/mcp/tool_schemas.py` — modify (include_ghosts param)
- `src/neural_memory/storage/sqlite_schema.py` — modify (migration v35)
- `src/neural_memory/storage/sqlite_fibers.py` — modify (last_ghost_shown_at)
- `tests/unit/test_fidelity.py` — extend
- `tests/unit/test_ghost_recall.py` — NEW

## Dependencies
- Phase 2 (fidelity selection must work to generate ghosts)
