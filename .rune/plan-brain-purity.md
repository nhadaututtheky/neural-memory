# Feature: Brain Purity — Input/Output Quality Optimization

## Overview

Brain health dropped to D grade (56.7% purity) with 7,633 neurons, 89% never accessed, 98% fibers still episodic. Root cause: no hard validation on write path + weak dedup + no cleanup of dead neurons. This plan fixes input quality (write gate), output quality (recall precision), and multi-agent/sync hygiene.

Addresses GitHub Issue #95 (write-gate request from production user with 12,700+ neurons, 63% purity).

## Phases

| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Write Gate | ✅ Done | plan-brain-purity-p1.md | Hard quality gate before storage, configurable thresholds |
| 2 | Dedup Improvement | ✅ Done | plan-brain-purity-p2.md | Tighter thresholds (simhash 7, candidates 30), DEDUP in session-end |
| 3 | Recall Quality | ✅ Done | plan-brain-purity-p3.md | Configurable recency halflife, tag-aware scoring, dead neuron prune |
| 4 | Multi-Agent Hygiene | ✅ Done | plan-brain-purity-p4.md | Agent tag auto-inject, consolidation file lock |
| 5 | Sync Safety | ✅ Done | plan-brain-purity-p5.md | Neuron/fiber dedup on sync import, tag merge |

## Dependency Graph

```
P1 (Write Gate) ← foundational, blocks nothing
P2 (Dedup) ← independent, but benefits from P1 quality scoring
P3 (Recall) ← independent, can run parallel with P1/P2
P4 (Multi-Agent) ← uses P1 write gate + P2 dedup improvements
P5 (Sync Safety) ← uses P2 dedup + P4 agent identity
```

## Use Cases Driving This Plan

| UC | Description | Pain Points |
|----|-------------|-------------|
| UC1 | 1 brain, 1 agent, 1 project | Noise from auto-capture, no min length, dead neurons |
| UC2 | 1 brain, 1 agent, N projects | Cross-project noise in recall, weak tag discipline |
| UC3 | 1 brain, N agents, N projects | Agent collision (same insight, different words = duplicate) |
| UC4 | 1 brain synced, N PCs, N agents | Concurrent write conflicts, dedup across sync, consolidation race |

## Key Decisions

- Write gate is configurable per-brain (not global) — different brains can have different thresholds
- Quality scorer upgraded from soft-hint to hard-reject (with `enabled` toggle for backward compat)
- No LLM calls in write gate — pure heuristic/algorithmic, must be <10ms
- Dedup improvement = tuning existing pipeline, not rebuilding it
- Multi-agent identity via MCP session metadata (already available), not new protocol
- Sync safety = defensive checks on import path, not redesigning sync protocol
- SHIP FIRST principle: each phase is independently shippable as a minor version bump
