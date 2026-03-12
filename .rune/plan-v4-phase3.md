# Phase 3: Predictive Priming

## Goal

Pre-warm memories based on session context so recall is faster and more relevant. Brain anticipates what you'll ask next.

## Problem

Every recall starts cold — finds anchors, spreads activation, returns results. Real brains prime: when you think about "auth", "JWT" and "session" are already half-activated before you ask.

Graph expansion (1-hop) is implicit priming but:
1. Not session-aware — doesn't know what topics are active
2. Not predictive — only expands FROM current query, not from session history
3. No temporal priming — doesn't use habit patterns ("after auth, user usually asks about env vars")

## Tasks

- [x] 3.1: Session-based activation cache
  - `ActivationCache` in `engine/priming.py`
  - Stores activation levels from recent queries (last 5 queries in session)
  - On new query, seed neurons start with cached activation (decayed by age)
  - Decay: `cached_level * 0.7^(queries_since_cached)` — rapid decay, 3 queries = 0.34x
  - Max cache size: 200 neurons per session

- [x] 3.2: Topic-based pre-warming
  - When session topic is established (EMA > 0.5):
    - Find top neurons tagged with session topics
    - Give them small activation boost (0.1-0.2) before query starts
  - Implementation: inject pre-warm list into `_find_anchors_ranked()` as soft anchors
  - Pre-warm happens BEFORE query parsing (truly predictive)

- [x] 3.3: Habit-based priming
  - Use `query_pattern_mining` co-occurrence data
  - If session contains topic A and habit says "A → B" (above threshold):
    - Pre-warm topic B neurons
  - Threshold: co-occurrence count >= 3 AND confidence >= 0.6
  - Max 3 predicted topics (prevent over-priming)

- [x] 3.4: Co-activation priming
  - Use `co_activation_events` (Hebbian bindings)
  - If neuron X was in recent query result AND X has strong co-activation with Y:
    - Y gets small activation boost in next query
  - Only for bindings with `binding_strength >= 0.5` AND `count >= 3`

- [x] 3.5: Priming metrics
  - Track priming hit rate: "primed neuron appeared in final result" = hit
  - `priming_metrics` in session state: hits, misses, total primed
  - Expose in recall response: `priming_hit_rate: float`
  - Use hit rate to auto-adjust priming aggressiveness

- [x] 3.6: Tests
  - Activation cache with decay over multiple queries
  - Topic-based pre-warming from session EMA
  - Habit-based priming from co-occurrence data
  - Co-activation priming from Hebbian bindings
  - Priming metrics tracking and hit rate calculation
  - Over-priming prevention (max neurons, max topics)
  - Cache eviction and session cleanup

## Acceptance Criteria

- [ ] Recent query results carry forward as soft activation
- [ ] Session topics pre-warm related neurons
- [ ] Habit patterns predict next topic
- [ ] Co-activation data used for neuron-level priming
- [ ] Priming hit rate tracked and exposed
- [ ] No hot-path SQLite queries (all priming data from in-memory cache)
- [ ] 25+ tests

## Files Touched

- `src/neural_memory/engine/priming.py` — new (ActivationCache, TopicPrimer, HabitPrimer)
- `src/neural_memory/engine/retrieval.py` — inject priming into activation pipeline
- `src/neural_memory/engine/session_state.py` — priming metrics in SessionState
- `src/neural_memory/mcp/tool_handlers.py` — expose priming_hit_rate
- `tests/unit/test_predictive_priming.py` — new

## Dependencies

- Requires Phase 1 (session state for topic tracking)
- Uses existing: co_activation_events, query_pattern_mining, graph expansion
