# Feature: Brain Quality — Track C: Vertical Intelligence

## Overview

Domain-specific capabilities for vertical use cases (accounting, legal, data viz). Track A (Proactive) and Track B (Graph Quality) are fully shipped — this plan covers only the remaining Track C.

## Track A & B Status (COMPLETED)
- A1 Smart Instructions ✅ | A2+A3 Knowledge Surface ✅ | A4 Background Processing ✅
- B1-B8 all ✅ (auto-consolidation, Hebbian, cross-memory, IDF, fiber scoring, compression, lazy entity, adaptive decay)

## Track C Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| C1+C2 | Domain Entities + Structured Data | ⬚ Pending | plan-brain-quality-c1c2.md | Entity subtypes + table-as-graph encoding | **Free** |
| C3 | Cross-Encoder Reranking | ⬚ Pending | plan-brain-quality-c3.md | Optional bge-reranker post-SA refinement | **Private (Pro)** |
| C4 | Agent Visualization | 🔄 Active | plan-brain-quality-c4.md | nmem_visualize → Vega-Lite/markdown/ASCII charts | **Free** |

## Dependency Graph
```
C1+C2 (Domain Entities + Structured Data)
  └→ C4 (Visualization — needs financial/structured data)
C3 (Reranking) ← independent, enhances retrieval precision
```

## Key Decisions
- Zero LLM calls — all improvements are algorithmic/rule-based
- Domain extraction via regex patterns (Vietnamese + English)
- C3 reranker is optional dep: `neural-memory[reranker]`
- C4 outputs Vega-Lite (primary), markdown table (fallback), ASCII (CLI)
- VISION.md brain-test mandatory for every phase
