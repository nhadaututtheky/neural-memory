# Phase 3: Recall Quality

## Goal

Improve recall precision by pruning dead neurons, making recency decay configurable, and boosting project-relevant results. Target: recall_confidence from 48.8% → 65%+.

## Why This Matters

- 89% of neurons (6,793/7,633) never accessed — dead weight that pollutes activation spreading
- Recency sigmoid hardcoded at 72h peak — bad for project memories that should live longer
- Cross-project noise: recall "auth" returns auth from all projects mixed together
- Fiber scoring doesn't factor in project/tag match with query context
- activation_efficiency at 10.2% — most of the graph is inert noise

## Tasks

- [x] 3.1 — Dead neuron pruning in PRUNE strategy
  - Enhanced orphan detection: also prunes neurons with access_frequency=0, age > 14 days, not pinned
  - Uses `get_neuron_states_batch()` for efficient batch lookup
  - Configurable via `BrainConfig.prune_dead_neuron_days` (default 14.0)
  - Logs pruned counts at INFO level

- [x] 3.2 — Configurable recency decay
  - Added `recency_halflife_hours: float = 168.0` to BrainConfig (was hardcoded 72h)
  - Sigmoid: `1 / (1 + exp((hours_ago - halflife) / (halflife / 2)))`
  - Per-brain tuning: fast project → 72h, long-term knowledge → 720h

- [x] 3.3 — Tag-aware fiber scoring boost
  - Added `tag_match_boost: float = 0.15` to BrainConfig
  - Matching tags: +boost (capped at 3 tags), zero overlap: -boost*0.5 penalty
  - Lightweight set intersection on existing fiber metadata tags

- [ ] 3.4 — Recall confidence tracking (DEFERRED — needs agent feedback loop, separate PR)

- [x] 3.5 — Tests (22 new tests in test_recall_quality.py)
  - BrainConfig fields, recency sigmoid, tag scoring, dead neuron pruning, backward compat

## Acceptance Criteria

- [ ] Dead neurons (never accessed, >14 days, no fiber) pruned automatically
- [ ] Recency decay configurable per brain (halflife in hours)
- [ ] Tag-matching fibers score higher in recall results
- [ ] Existing recall tests pass (backward compat)
- [ ] 15+ new tests

## Files Touched

- `src/neural_memory/engine/consolidation.py` — PRUNE_DEAD sub-strategy
- `src/neural_memory/engine/retrieval.py` — configurable recency, tag-aware scoring
- `src/neural_memory/unified_config.py` — recency_halflife_hours config
- `tests/unit/test_recall_quality.py` — new test file

## Design Notes

- PRUNE_DEAD is safe: only targets neurons with access_count=0 (never been useful)
- 14-day grace period ensures new neurons get a chance to be recalled
- Tag boost is additive (not multiplicative) to avoid amplifying bad results
- Recency halflife default 168h (7 days) is close enough to old 72h behavior for most use cases
  - Old sigmoid at 72h: memory at 7 days scores ~0.15
  - New sigmoid at 168h: memory at 7 days scores ~0.50 (more balanced)
