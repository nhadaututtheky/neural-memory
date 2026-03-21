# Phase 1: Write Gate

## Goal

Convert quality_scorer from soft-hint (always stores) to hard-gate (can reject). Add configurable thresholds so low-quality memories never enter the brain. Directly addresses Issue #95.

## Why This Matters

- Brain purity 56.7% → primary cause is unchecked writes
- 3,589 duplicates found in production brain (12,700 neurons)
- Auto-capture (post_tool_capture) generates ~3 memories/min with no quality floor
- quality_scorer.py currently returns hints but NEVER rejects — everything gets stored
- Session stop hook dumps 150-line transcript walls into single memory

## Tasks

- [x] 1.1 — Add `WriteGateConfig` to BrainConfig (`unified_config.py`)
  - `enabled: bool = False` (opt-in, backward compat)
  - `min_length: int = 30`
  - `min_quality_score: float = 3.0` (out of 10)
  - `auto_capture_min_score: float = 5.0` (stricter for passive captures)
  - `max_fiber_content_length: int = 2000` (cap wall-of-text)
  - `reject_generic_filler: bool = True`

- [x] 1.2 — Upgrade `quality_scorer.py` to return reject/accept decision
  - Add `rejected: bool` and `rejection_reason: str` to QualityResult
  - Score < min_quality_score → rejected
  - Content length < min_length → rejected
  - Generic filler detection: ["done", "ok", "completed", "xong", "noted"] as sole content → rejected
  - Content > max_fiber_content_length → rejected with hint to split

- [x] 1.3 — Wire write gate into `_remember()` handler (tool_handlers.py)
  - After safety checks, before encoding
  - If write_gate.enabled AND quality_result.rejected → return error with reason + hints
  - Log rejected memories at DEBUG level (for diagnostics)
  - Pass `is_auto_capture: bool` flag to use stricter threshold

- [x] 1.4 — Wire write gate into `_remember_batch()` handler
  - Auto via `_remember()` delegation — batch handler already handles per-item errors

- [x] 1.5 — Wire write gate into auto-capture paths
  - `_post_tool_capture()` in auto_handler.py → pass `_auto_capture=True`
  - `_save_detected_memories_no_dedup()` → pass `_auto_capture=True`
  - Stop hook `capture_text()` → gate check before each encode + session summary

- [ ] 1.6 — Add `nmem_write_gate` diagnostic tool (optional, low priority)
  - Dry-run: test content against write gate without storing
  - Returns: score, pass/reject, hints
  - Useful for agents to pre-check before remembering

- [x] 1.7 — Tests (32 new tests in test_write_gate.py)
  - Quality scorer rejection logic (length, score, filler, wall-of-text)
  - Write gate integration in _remember (gate enabled, gate disabled)
  - Auto-capture stricter threshold
  - Batch remember with partial rejections
  - Config loading and defaults
  - Backward compat: gate disabled = current behavior exactly

## Acceptance Criteria

- [ ] `quality_scorer.py` can hard-reject based on configurable thresholds
- [ ] `_remember()` returns rejection error when gate triggers (not silent)
- [ ] Auto-captured memories use stricter threshold than explicit remembers
- [ ] Wall-of-text (>2000 chars) rejected with "split" hint
- [ ] Generic filler rejected (configurable word list)
- [ ] Gate disabled by default — existing users see zero behavior change
- [ ] All existing tests still pass (backward compat)
- [ ] 20+ new tests covering gate logic

## Files Touched

- `src/neural_memory/unified_config.py` — add WriteGateConfig
- `src/neural_memory/engine/quality_scorer.py` — upgrade to hard gate
- `src/neural_memory/mcp/tool_handlers.py` — wire into _remember, _remember_batch
- `src/neural_memory/mcp/auto_handler.py` — wire into _post_tool_capture
- `src/neural_memory/hooks/stop_hook.py` — wire into session summary
- `tests/unit/test_write_gate.py` — new test file

## Design Notes

- Write gate runs AFTER safety checks (sensitive content) but BEFORE encoding
- Order: validate length → check filler → score quality → accept/reject
- No LLM calls — pure regex + heuristic, must complete in <10ms
- Rejection response includes hints so agents can improve and retry
- Config lives at brain level so different brains can have different standards
