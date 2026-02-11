---
name: intake
description: Guided memory creation — analyze input and store as structured memories
arguments:
  - name: text
    description: The text to process into memories
    required: true
allowed-tools:
  - mcp__neuralmemory__nmem_remember
  - mcp__neuralmemory__nmem_recall
  - mcp__neuralmemory__nmem_stats
  - mcp__neuralmemory__nmem_context
  - mcp__neuralmemory__nmem_auto
---

# /intake — Guided Memory Creation

## Instruction

Process the following input into structured memories: $ARGUMENTS

## Method

### Step 1: Triage

Scan the input and classify each information unit:

| Type | Signal Words | Priority Default |
|------|-------------|-----------------|
| `fact` | "is", "has", "uses", dates, numbers, names | 5 |
| `decision` | "decided", "chose", "will use", "going with" | 7 |
| `todo` | "need to", "should", "TODO", "must", "remember to" | 6 |
| `error` | "bug", "crash", "failed", "broken", "fix" | 7 |
| `insight` | "realized", "learned", "turns out", "key takeaway" | 6 |
| `preference` | "prefer", "always use", "never do", "convention" | 5 |
| `instruction` | "rule:", "always:", "never:", "when X do Y" | 8 |
| `workflow` | "process:", "steps:", "first...then...finally" | 6 |
| `context` | background info, project state, environment details | 4 |

### Step 2: Clarification (if needed)

For ambiguous items, ask ONE question at a time with 2-4 options:

```
I found: "We're using PostgreSQL now"

What type of memory is this?
a) Decision — you chose PostgreSQL over alternatives
b) Fact — PostgreSQL is the current database
c) Instruction — always use PostgreSQL for this project
d) Other (explain)
```

Rules:
- ONE question per round — never dump a checklist
- Always provide options — don't ask open-ended unless necessary
- Infer when confident (>80% sure) — don't ask
- Max 5 clarification rounds
- Group similar items when possible

### Step 3: Deduplication Check

Before storing, check for existing similar memories:

```
nmem_recall("{memory_content_summary}")
```

- **Identical**: Skip, report as duplicate
- **Updated version**: Store new, note supersedes old
- **Contradicts**: Alert user, let them decide
- **Complements**: Store, note connection

### Step 4: Preview Batch

Present the batch before storing:

```
Ready to store N memories:

  1. [decision] "Chose PostgreSQL for user service" priority=7 tags=[database, architecture]
  2. [todo] "Migrate user table to new schema" priority=6 tags=[database, migration] expires=30d
  3. [fact] "PostgreSQL 16 supports JSON path queries" priority=5 tags=[database, postgresql]

Store all? [yes / edit # / skip # / cancel]
```

### Step 5: Store

After confirmation, store each via `nmem_remember` with proper type, tags, and priority.

### Step 6: Report

```
Intake Complete
  Stored: N memories (breakdown by type)
  Skipped: N duplicates
  Conflicts: N flagged
  Follow-up: items needing clarification
```

## Rules

- Never auto-store without user seeing the preview first
- Never guess security-sensitive information — ask explicitly
- Prefer specific over vague — "PostgreSQL 16 on AWS RDS" over "using a database"
- Include reasoning in decisions — "Chose X because Y"
- One concept per memory — don't cram multiple facts into one
- Max 10 per batch — split larger inputs
