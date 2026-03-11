# NM v4.0 — Brain Intelligence

Vision: Brain that learns from its own usage. Adaptive depth, session priming, drift detection.

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Session Intelligence | ✅ Done | plan-v4-phase1.md | Session state tracking across MCP calls |
| 2 | Adaptive Depth v2 | ✅ Done | plan-v4-phase2.md | Calibration → depth tuning, session-aware |
| 3 | Predictive Priming | ⬚ Pending | plan-v4-phase3.md | Pre-warm memories from session context |
| 4 | Semantic Drift Detection | ⬚ Pending | plan-v4-phase4.md | Tag co-occurrence, cluster merge suggestions |
| 5 | Diminishing Returns Gate | ⬚ Pending | plan-v4-phase5.md | Stop traversal when new hops add no signal |

## Key Decisions

- Build on existing foundation (calibration table, depth priors, RRF, co-activation)
- Primary goal: connect existing subsystems into feedback loops
- Session state = lightweight in-memory + periodic SQLite persist (not on hot path)
- Zero new dependencies, zero LLM dependency
- Each phase ships independently as minor version (v3.2, v3.3, ...)
- v4.0 tag when Phases 1-4 complete (Brain Intelligence = done)

## Dependency Graph

```
Phase 1 (session intelligence)
  ├→ Phase 2 (adaptive depth — uses session context)
  └→ Phase 3 (predictive priming — uses session topics)
       └→ Phase 4 (drift detection — uses tag co-occurrence from priming)

Phase 5 (diminishing returns) — independent, can run parallel
```

## What Exists (Foundation)

| Subsystem | Status | Gap |
|-----------|--------|-----|
| Calibration table + EMA | ✅ Full | Only downgrades gates, never adjusts depth thresholds |
| Bayesian depth priors | ✅ Full | Entity-based only, not session-aware |
| RRF score fusion | ✅ Full | No feedback loop from result quality |
| PPR activation | ✅ Full | Opt-in, no auto-selection based on query type |
| Graph expansion (1-hop) | ✅ Full | Not session-aware, no topic continuity |
| Co-activation tracking | ✅ Full | Hebbian binding exists but not used for priming |
| Query pattern mining | ⚠️ Partial | Mines patterns but no session-level topic model |
| Session context | ⚠️ Partial | tool_events has session_id, but no topic tracking |
| Tag clustering | ⚠️ Partial | Jaccard in consolidation, no co-occurrence matrix |
| Priming | ⚠️ Partial | Graph expansion = implicit priming, not predictive |

## Version Plan

| Phase | Version | Scope |
|-------|---------|-------|
| 1 | v3.2.0 | Session state, topic detection, context persistence |
| 2 | v3.3.0 | Adaptive depth v2, calibration feedback loops |
| 3 | v3.4.0 | Predictive priming, session-based pre-warming |
| 4 | v4.0.0 | Drift detection + auto-merge — Brain Intelligence complete |
| 5 | v4.1.0 | Diminishing returns gate |
