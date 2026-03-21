# Feature: Fidelity Layers (λ-Memory Style)

## Overview
Memories fade through fidelity levels (FULL → SUMMARY → ESSENCE → GHOST) based on
decay score and context budget pressure. Never truly deleted — agent sees the "shape"
of faded memories and can recall them back to full detail.

## Key Decisions
1. **Fidelity on Fiber** — Fiber is the unit agents see. Not Neuron.
2. **Essence at consolidation** — No LLM at write time. Extractive in Phase 1, LLM optional Phase 4.
3. **Fidelity selection in context assembly** — Retrieval finds+scores, context assembly budgets fidelity.
4. **Ghost = tags+type+age+ID** — Interpretable, not opaque hash. Includes recall hint.
5. **Graph promotion is free** — Spreading activation already boosts linked memories.
6. **Fidelity is derived, not stored** — Computed at read time from score + budget. Schema stays simple.
7. **Decay floor = 0.05** — Ghost never reaches 0, always visible as hint.
8. **Score = activation × importance × e^(−λt)** — Explicit formula, deterministic.
9. **Budget algorithm: greedy priority queue** — Scales to >10k fibers.

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Schema + Essence | ✅ Done | plan-fidelity-p1.md | Essence field, extractive generator, consolidation hook, decay_floor |
| 2 | Fidelity Context | ✅ Done | plan-fidelity-p2.md | Budget-aware fidelity selection in optimize_context |
| 3 | Ghost Recall | ✅ Done | plan-fidelity-p3.md | Ghost view, recall hints, visibility boost |
| 4 | LLM Essence | ✅ Done | plan-fidelity-p4.md | Optional LLM abstractive essence |

## NOT Doing
- Pre-compute layers at write time (latency, waste)
- Separate fidelity table (1 column enough)
- Per-memory fidelity config (system auto-selects)
- New MCP tool for ghost recall (nmem_recall sufficient)
- Store fidelity level in DB (derived at read time)

## Expert Feedback Incorporated
- Decay floor 0.05 (Expert 1) — ghost always visible
- Priority queue budget algorithm (Expert 1) — scales >10k
- Benchmark gates: write <60ms, context assembly, recall accuracy (Expert 1)
- Ghost includes fiber ID + recall hint (Expert 2)
- Decay params stored per-memory for dynamic score (Expert 2)
- Background consolidation for essence (Expert 2)
- Fidelity = read-time derived, not stored (Expert 2)
