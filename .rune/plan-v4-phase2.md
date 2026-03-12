# Phase 2: Adaptive Depth v2

## Goal

Close the calibration → depth feedback loop. Brain learns optimal depth per query type AND session context, not just per entity.

## Problem

Bayesian depth priors exist (Beta(α,β) per entity) but:
1. Only entity-based — "auth" always gets same depth regardless of session context
2. Calibration EMA only downgrades gate confidence, never adjusts depth thresholds
3. No feedback from result quality to retriever weights (RRF weights are static)

## Tasks

- [ ] 2.1: Session-aware depth adjustment
  - If session has established topic (EMA > 0.5), bias toward CONTEXT depth (more exploration)
  - If query matches recent session topic closely, bias toward INSTANT (already primed)
  - New method: `AdaptiveDepthSelector.select_depth(stimulus, session_state)`
  - Session context = soft prior, not hard override

- [ ] 2.2: Calibration-driven gate threshold tuning
  - Use `get_gate_ema_stats()` to adjust sufficiency gate thresholds dynamically
  - If `intersection_convergence` gate has high accuracy (>0.8) → trust it more (lower confidence threshold)
  - If `default_pass` gate has low accuracy (<0.4) → raise its confidence threshold
  - Store adjusted thresholds in `BrainConfig` override (per-brain, not global)

- [ ] 2.3: Result quality feedback to depth priors
  - Currently: success = `confidence >= 0.3 AND fibers >= 1`
  - Enhance: also track if agent used the recall result (via `nmem_remember` following `nmem_recall`)
  - "Agent remembered something after recalling" = strong positive signal
  - "Agent recalled but never used result" = weak/negative signal (not immediately — wait 5 mins)

- [ ] 2.4: Dynamic RRF weight adjustment
  - Track which retriever type (entity/keyword/embedding/expansion) contributes most to successful recalls
  - Per-brain `retriever_weights` that evolve based on calibration data
  - New table: `retriever_calibration(brain_id, retriever_type, success_ema, sample_count)`
  - EMA with alpha=0.1 (slow adaptation — don't overfit to recent queries)

- [ ] 2.5: Activation strategy auto-selection
  - Currently `activation_strategy` is static config ("classic"/"ppr"/"hybrid")
  - Auto-select based on graph density:
    - Sparse graph (few synapses per neuron) → classic (BFS reaches more)
    - Dense graph (many synapses) → PPR (dampens hub noise)
    - Mixed → hybrid
  - Compute graph density metric during consolidation, store in brain metadata

- [ ] 2.6: Tests
  - Session-aware depth selection with various session states
  - Gate threshold tuning from calibration data
  - Result quality feedback loop (remember-after-recall signal)
  - RRF weight adaptation over multiple queries
  - Strategy auto-selection based on graph density
  - Regression: existing depth selection behavior preserved when session is empty

## Acceptance Criteria

- [ ] Depth selection considers session context (not just entity)
- [ ] Gate thresholds adapt from calibration history
- [ ] RRF weights evolve per-brain based on retriever success
- [ ] Activation strategy auto-selected from graph density
- [ ] All existing depth selection tests still pass (backward compat)
- [ ] 25+ tests

## Files Touched

- `src/neural_memory/engine/depth_prior.py` — session-aware select_depth
- `src/neural_memory/engine/retrieval.py` — wire session, dynamic RRF weights
- `src/neural_memory/engine/sufficiency.py` — dynamic gate thresholds
- `src/neural_memory/storage/sqlite_calibration.py` — retriever calibration table
- `src/neural_memory/storage/sqlite_schema.py` — new table migration
- `src/neural_memory/core/brain.py` — per-brain threshold overrides
- `tests/unit/test_adaptive_depth_v2.py` — new

## Dependencies

- Requires Phase 1 (session state for session-aware depth)
