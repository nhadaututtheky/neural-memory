# Phase 2: Fidelity-Aware Context Assembly

## Goal
`optimize_context` selects fidelity level per fiber based on composite score and budget
pressure. High-score memories get full text, low-score get summary/essence/ghost.

## Score Formula
```
score = (importance + graph_boost) * e^(-lambda * hours_since_access)
```
- `importance` = fiber priority (1-10, normalized to 0-1)
- `graph_boost` = spreading activation level from retrieval
- `lambda` = decay rate from BrainConfig
- `hours_since_access` = hours since last access
- Floor: max(score, decay_floor) where decay_floor = 0.05

## Tasks
- [x] Define `FidelityLevel` enum in `engine/fidelity.py`: FULL, SUMMARY, ESSENCE, GHOST
- [x] Add `compute_fidelity_score()` function:
  - Inputs: activation_level, priority, last_accessed_at, decay_rate, decay_floor
  - Returns: float (0.05 to 1.0)
  - Formula: max(importance * exp(-lambda * t), decay_floor)
- [x] Add `select_fidelity(score, budget_pressure) -> FidelityLevel` function:
  - `score >= full_threshold` → FULL
  - `score >= summary_threshold` → SUMMARY (fallback to FULL if no summary)
  - `score >= essence_threshold` → ESSENCE (fallback chain: essence → summary → full)
  - `score < essence_threshold` → GHOST
- [x] Add `render_at_fidelity(fiber, level) -> str` function:
  - FULL: anchor neuron content
  - SUMMARY: fiber.summary
  - ESSENCE: fiber.essence
  - GHOST: `"[~] {type} | {age} | {tags} | {connection_count} links"`
- [x] Modify `optimize_context` in `context_optimizer.py`:
  - After scoring phase, before token budgeting
  - Compute budget_pressure = tokens_used / max_tokens
  - Use greedy priority queue: pop highest score, assign fidelity, deduct tokens
  - If remaining budget tight, downgrade next items
  - Re-estimate tokens after fidelity selection
- [x] Add BrainConfig fields:
  - `fidelity_enabled: bool = True`
  - `fidelity_full_threshold: float = 0.6`
  - `fidelity_summary_threshold: float = 0.3`
  - `fidelity_essence_threshold: float = 0.1`
- [x] Add `fidelity_stats` to context response: `{full: N, summary: N, essence: N, ghost: N}`
- [x] Update `_context` MCP handler to include fidelity stats
- [ ] Benchmark: context assembly with 100, 1000, 10000 fibers across 16K-2M budgets

## Acceptance Criteria
- [x] 20 memories + 500 token budget: top 2-3 FULL, next SUMMARY, rest ESSENCE/GHOST
- [x] 20 memories + 4000 token budget: most FULL, lowest-scoring SUMMARY
- [x] Graceful fallback: ESSENCE requested but null → serve SUMMARY → serve FULL
- [x] Ghost always visible (decay_floor = 0.05 prevents score = 0)
- [ ] Context assembly <50ms for 1000 fibers
- [ ] Priority queue algorithm, not full sort

## Files Touched
- `src/neural_memory/engine/fidelity.py` — modify (selection logic, render)
- `src/neural_memory/engine/context_optimizer.py` — modify (fidelity phase)
- `src/neural_memory/core/brain.py` — modify (BrainConfig fields)
- `src/neural_memory/mcp/tool_handlers.py` — modify (fidelity stats)
- `tests/unit/test_fidelity.py` — extend
- `tests/unit/test_context_optimizer.py` — extend

## Dependencies
- Phase 1 (essence field must exist)
