---
name: audit
description: Brain health review — assess memory quality across 6 dimensions
arguments:
  - name: focus
    description: Optional dimension or topic to focus the audit on
    required: false
allowed-tools:
  - mcp__neuralmemory__nmem_stats
  - mcp__neuralmemory__nmem_health
  - mcp__neuralmemory__nmem_conflicts
  - mcp__neuralmemory__nmem_recall
  - mcp__neuralmemory__nmem_context
---

# /audit — Brain Health Review

## Instruction

Audit the current brain's memory quality: $ARGUMENTS

If no specific focus given, run full audit across all 6 dimensions.

## Method

### Step 1: Collect Baseline

```
nmem_stats          → neuron count, synapse count, memory types, age distribution
nmem_health         → purity score, component scores, warnings, recommendations
nmem_context        → recent memories, freshness indicators
nmem_conflicts(action="list") → active contradictions
```

### Step 2: Score 6 Dimensions

#### Purity (25%)
- Active contradictions (CRITICAL if >0)
- Near-duplicates (HIGH)
- Outdated facts (MEDIUM)

#### Freshness (20%)
- Stale ratio (% memories >90 days with no access)
- Expired TODOs still active
- Zombie memories (never recalled, >30 days)

#### Coverage (20%)
- Topic balance across project areas
- Decision coverage (reasoning stored?)
- Error pattern documentation

#### Clarity (15%)
- Vague memories (<20 chars, no specifics)
- Missing context (decisions without reasoning)
- Overstuffed memories (>500 chars, 3+ topics)

#### Relevance (10%)
- Orphaned project references
- Technology drift
- Context mismatch

#### Structure (10%)
- Connectivity (orphan neurons)
- Synapse diversity
- Fiber conductivity

### Step 3: Triage Findings

| Severity | Criteria |
|----------|----------|
| CRITICAL | Active contradictions, security-sensitive errors |
| HIGH | Significant gaps, widespread staleness |
| MEDIUM | Moderate quality issues, some duplicates |
| LOW | Cosmetic, minor optimization |
| INFO | Observations, no action needed |

### Step 4: Generate Report

```
Memory Audit Report
Brain: {name} | Date: {today}

Overall Grade: {letter} ({score}/100)

Dimension Scores:
  Purity:     {bar}  {score}/100
  Freshness:  {bar}  {score}/100
  Coverage:   {bar}  {score}/100
  Clarity:    {bar}  {score}/100
  Relevance:  {bar}  {score}/100
  Structure:  {bar}  {score}/100

Findings: {total}
  CRITICAL: {n}
  HIGH:     {n}
  MEDIUM:   {n}
  LOW:      {n}

Top Recommendations:
  1. [{severity}] {recommendation}
  2. [{severity}] {recommendation}
  3. [{severity}] {recommendation}
```

## Rules

- Evidence-based only — every finding must reference specific memories or metrics
- No guessing — if data is insufficient, report "insufficient data" for that dimension
- Prioritize by impact — present CRITICAL before LOW
- Actionable recommendations — every finding must have a concrete fix
- Audit is read-only — user decides what to fix
- Estimate effort for each recommendation
