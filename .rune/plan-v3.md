# NM v3.0 — Source-Aware Memory

Vision Pillar 4: Brain that knows its sources. Verbatim + Navigational recall modes.

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 0 | Stability | ⬚ | plan-v3-phase0.md | Fix #53 SharedStorage parity, close bugs |
| 1 | Exact Recall | ⬚ | plan-v3-phase1.md | `nmem_show` (#35) + `mode="exact"` in recall |
| 2 | Source Registry | ⬚ | plan-v3-phase2.md | Schema v23, sources table, SOURCE_OF synapse |
| 3 | Structured Encoding | ⬚ | plan-v3-phase3.md | Schema-aware encoder, tabular data support |
| 4 | Citation + Audit | ⬚ | plan-v3-phase4.md | Citation output format, audit synapses |
| 5 | DX Sprint | ⬚ | plan-v3-phase5.md | One-line installer, first-run wizard, embedding UX |

## Key Decisions

- Each phase ships independently as a minor version (v2.30, v2.31, ...)
- v3.0 tag when Phases 1-4 complete (Source-Aware Memory = done)
- Phase 0 first — unblock hub users before adding features
- Phase 5 parallel — DX work doesn't depend on core features
- SharedStorage must have feature parity — abstract `brain_id` property on base class

## Dependency Graph

```
Phase 0 (stability)
  └→ Phase 1 (exact recall)
       └→ Phase 2 (source registry)
            ├→ Phase 3 (structured encoding)
            └→ Phase 4 (citation + audit)

Phase 5 (DX) — independent, can run parallel with any phase
```

## Version Plan

| Phase | Version | Scope |
|-------|---------|-------|
| 0 | v2.30.0 | Bug fixes, SharedStorage parity |
| 1 | v2.31.0 | nmem_show + exact recall mode |
| 2 | v2.32.0 | Source registry, schema v23 |
| 3 | v2.33.0 | Structured encoding |
| 4 | v3.0.0 | Citation engine + audit — Pillar 4 complete |
| 5 | v3.1.0 | DX improvements |
