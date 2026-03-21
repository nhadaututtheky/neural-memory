# Phase 4: LLM-Enhanced Essence (Optional)

## Goal
When LLM is available, generate higher-quality abstractive essence. Hybrid approach:
write-time extractive (cheap) → consolidation-time LLM upgrade (quality).

## Tasks
- [x] Define `EssenceGenerator` ABC in `engine/fidelity.py`
- [x] `ExtractiveEssenceGenerator` — wraps existing Phase 1 logic
- [x] `LLMEssenceGenerator` — async LLM call with cost guard
  - Prompt: "Distill into one sentence (max 30 words): {content}"
  - Cost guard: skip for priority < 3
  - Fallback to extractive on failure/empty
  - Input truncated to 2000 chars
- [x] Add `BrainConfig.essence_generator: str = "extractive"` (options: extractive, llm)
- [x] Hybrid consolidation path:
  - `_essence_backfill()` reads strategy from BrainConfig
  - Uses `get_essence_generator()` factory
  - Passes priority from typed_memory for cost guard
- [x] Fallback: if LLM call fails, keep extractive essence
- [x] Background consolidation: essence upgrade happens async via consolidation engine
- [x] Tests: LLM generator with mock, fallback, cost guard, factory (14 tests)

## Acceptance Criteria
- [x] LLM essence noticeably better for complex memories (via mock tests)
- [x] No LLM calls at write time (extractive placeholder only)
- [x] Graceful fallback when LLM unavailable
- [x] Priority < 3 memories never get LLM essence (cost guard)
- [x] Consolidation reads strategy from brain config

## Files Touched
- `src/neural_memory/engine/fidelity.py` — modify (LLM generator)
- `src/neural_memory/core/brain.py` — modify (essence_generator config)
- `src/neural_memory/engine/consolidation.py` — modify (hybrid upgrade)
- `tests/unit/test_fidelity.py` — extend

## Dependencies
- Phase 1 (essence field + extractive generator)
- Optional: only implement if extractive quality insufficient
