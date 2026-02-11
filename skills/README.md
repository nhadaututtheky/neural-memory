# NeuralMemory Skills

Composable AI agent skills for NeuralMemory, following the [ship-faster](https://github.com/Heyvhuang/ship-faster) pattern.

## Available Skills

| Skill | Stage | Description |
|-------|-------|-------------|
| **memory-intake** | workflow | Structured memory creation from messy notes — 1-question-at-a-time clarification, batch storage with preview |
| **memory-audit** | review | 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure) with graded findings |
| **memory-evolution** | workflow | Evidence-based optimization from usage patterns — consolidation, enrichment, pruning, tag normalization |

## Install

Skills are bundled with the pip package and can be installed with one command:

```bash
nmem install-skills
```

Or as part of full setup:

```bash
nmem init
```

Other options:

```bash
nmem install-skills --list     # Show available skills
nmem install-skills --force    # Overwrite with latest versions
nmem init --skip-skills        # Skip skills during init
```

## Usage

In Claude Code, invoke by name:

```
/memory-intake "Messy meeting notes: decided to use Redis for caching,
Bob will handle the migration, deadline is next Friday, also need to
update the API docs"

/memory-audit

/memory-evolution "Focus on the auth topic"
```

## Skill Format

Each skill follows the ship-faster SKILL.md format:

```yaml
---
name: skill-name
description: What it does
metadata:
  stage: workflow|review|tool
  tags: [relevant, tags]
agent: Agent Role Name
allowed-tools:
  - nmem_remember
  - nmem_recall
---
```

## Prerequisites

- NeuralMemory installed (`pip install neural-memory`)
- NeuralMemory MCP server configured in Claude Code
- At least one brain with some memories (for audit/evolution)
