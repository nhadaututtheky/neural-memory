# Integration Guide

This guide covers how to integrate NeuralMemory with AI assistants, IDEs, and development tools.

---

## Table of Contents

1. [Claude Code Integration](#claude-code-integration)
2. [Cursor Integration](#cursor-integration)
3. [Windsurf Integration](#windsurf-integration)
4. [Aider Integration](#aider-integration)
5. [Other AI Assistants](#other-ai-assistants)
6. [Shell Integration](#shell-integration)
7. [CI/CD Integration](#cicd-integration)
8. [Python API Integration](#python-api-integration)
9. [Coding Principles for Better Memory](#coding-principles-for-better-memory)

---

## Claude Code Integration

### Option A: MCP Server (Recommended)

NeuralMemory provides a native MCP (Model Context Protocol) server that exposes memory tools directly to Claude Code.

#### 1. Install NeuralMemory

```bash
pip install neural-memory
```

#### 2. Configure MCP Server

Add to `~/.claude/mcp_servers.json` (or create the file):

**Option 1: Using the CLI (recommended)**
```json
{
  "neural-memory": {
    "command": "nmem",
    "args": ["mcp"]
  }
}
```

**Option 2: Using the entry point**
```json
{
  "neural-memory": {
    "command": "nmem-mcp"
  }
}
```

**Option 3: Using Python module**
```json
{
  "neural-memory": {
    "command": "python",
    "args": ["-m", "neural_memory.mcp"]
  }
}
```

#### 3. Restart Claude Code

After restarting, Claude will have access to these tools:

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory with type, priority, tags |
| `nmem_recall` | Query memories with depth and confidence |
| `nmem_context` | Get recent context for injection |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_stats` | Get brain statistics |

#### 4. Usage in Claude Code

Claude will automatically use these tools when appropriate:

```
You: Remember that we decided to use PostgreSQL for the database
Claude: [uses nmem_remember tool]
       Stored the decision about PostgreSQL.

You: What database did we choose?
Claude: [uses nmem_recall tool]
       Based on my memory, you decided to use PostgreSQL for the database.
```

### Option B: CLAUDE.md Instructions

Add to your project's `CLAUDE.md`:

```markdown
## Memory Instructions

At the start of each session, get context:
```bash
nmem context --limit 20 --json
```

When learning something important, remember it:
```bash
nmem remember "Important information here" --type decision
```

When you need to recall past information:
```bash
nmem recall "query here"
```
```

### Option C: Manual Context Injection

Get context and inject at session start:

```bash
# Get recent context
CONTEXT=$(nmem context --json --limit 20)

# Include in your first message to Claude
echo "Recent project context: $CONTEXT"
```

---

## Cursor Integration

Cursor is an AI-powered IDE. Here's how to integrate NeuralMemory:

### Option A: Cursor Rules

Add to `.cursorrules` in your project:

```markdown
## Memory System

This project uses NeuralMemory for persistent context.

### Getting Context
Before starting work, run:
```bash
nmem context --limit 10
```

### Storing Important Information
When making decisions or learning patterns:
```bash
nmem remember "description" --type decision
nmem remember "error fix" --type error
nmem todo "task description"
```

### Recalling Information
When you need past context:
```bash
nmem recall "query"
```

### Guidelines
- Store all architectural decisions with --type decision
- Store error resolutions with --type error
- Store user preferences with --type preference
- Use tags for better organization: --tag feature --tag auth
```

### Option B: Cursor Commands

Create custom commands in Cursor settings:

```json
{
  "cursor.commands": [
    {
      "name": "Memory: Get Context",
      "command": "nmem context --limit 10"
    },
    {
      "name": "Memory: Remember Selection",
      "command": "nmem remember \"${selectedText}\""
    },
    {
      "name": "Memory: Recall",
      "command": "nmem recall \"${input:Query}\""
    }
  ]
}
```

### Option C: Cursor Composer Integration

In Cursor Composer, start your session with:

```
@terminal nmem context --json --limit 15

Use the above context. Remember important decisions with:
@terminal nmem remember "decision" --type decision
```

---

## Windsurf Integration

Windsurf (by Codeium) is an AI IDE. Here's how to integrate:

### Option A: Windsurf Rules

Create `.windsurfrules` in your project:

```markdown
## NeuralMemory Integration

### Session Start
Run this to get project context:
```bash
nmem context --fresh-only --limit 10
```

### During Development
Store important information:
- Decisions: `nmem remember "X" --type decision`
- Errors: `nmem remember "X" --type error`
- TODOs: `nmem todo "X" --priority 7`

### Querying
Recall past information:
```bash
nmem recall "your query" --depth 2
```

### Best Practices
1. Store decisions immediately after making them
2. Include rationale: "DECISION: X. REASON: Y"
3. Link related items with tags
```

### Option B: Windsurf Cascade Instructions

Add to Cascade system prompt:

```
You have access to NeuralMemory for persistent context.

Commands available in terminal:
- nmem context: Get recent memories
- nmem remember "X": Store memory
- nmem recall "X": Query memories
- nmem todo "X": Add TODO

Use these to maintain context across sessions.
```

### Option C: AI Flow Integration

In Windsurf's AI Flow:

```yaml
name: "With Memory Context"
steps:
  - run: "nmem context --json --limit 10"
    output: memory_context
  - prompt: |
      Recent project context:
      {{memory_context}}

      Now, {{user_request}}
```

---

## Aider Integration

Aider is a CLI AI coding assistant. Here's how to integrate:

### Option A: Aider Configuration

Create `.aider.conf.yml`:

```yaml
# Run before each session
auto-commits: false
edit-format: diff

# Custom commands
alias:
  /context: "!nmem context --limit 10"
  /remember: "!nmem remember"
  /recall: "!nmem recall"
```

### Option B: Shell Wrapper

Create `aider-with-memory.sh`:

```bash
#!/bin/bash
# aider-with-memory.sh

# Get context before starting
echo "Loading memory context..."
CONTEXT=$(nmem context --json --limit 15)

# Start aider with context
aider --message "Project context from memory:
$CONTEXT

Remember to use 'nmem remember' for important decisions." "$@"
```

Make executable and use:

```bash
chmod +x aider-with-memory.sh
./aider-with-memory.sh
```

### Option C: In-Session Commands

During Aider session, use shell commands:

```
> /run nmem context
> /run nmem remember "We decided to use FastAPI" --type decision
> /run nmem recall "API framework decision"
```

### Option D: Git Hook Integration

Since Aider auto-commits, capture decisions via git hooks.

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
# Auto-remember commits
MSG=$(git log -1 --pretty=%B)
nmem remember "Git commit: $MSG" --tag git --tag auto
```

---

## Other AI Assistants

### GitHub Copilot

Add to `.github/copilot-instructions.md`:

```markdown
## Memory Context

Get project context: `nmem context`
Store decisions: `nmem remember "X" --type decision`
Query past info: `nmem recall "X"`
```

### ChatGPT / Custom GPT

Create a Custom GPT with these instructions:

```
You are a coding assistant with access to NeuralMemory.

When the user shares terminal output from these commands, use it:
- `nmem context`: Recent project memories
- `nmem recall "X"`: Specific memory query
- `nmem stats`: Brain statistics

Suggest the user run these commands when:
- Starting a session: "Run `nmem context` to share recent context"
- Making decisions: "Run `nmem remember 'decision' --type decision`"
- Needing past info: "Run `nmem recall 'query'`"
```

### VS Code with Continue.dev

In `.continue/config.json`:

```json
{
  "customCommands": [
    {
      "name": "memory-context",
      "description": "Get NeuralMemory context",
      "prompt": "{{#if output}}Here's my project memory context:\n\n{{output}}\n\nUse this context.{{/if}}",
      "command": "nmem context --limit 10"
    }
  ]
}
```

---

## Shell Integration

### Auto-Remember Git Commits

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Auto-remember git commits
git() {
    command git "$@"
    if [[ "$1" == "commit" ]]; then
        local msg=$(command git log -1 --pretty=%B)
        nmem remember "Git commit: $msg" --tag git --type workflow &
    fi
}
```

### Remember Command Outputs

```bash
# Remember output of any command
remember-output() {
    local output=$("$@" 2>&1)
    echo "$output"
    nmem remember "Command: $* -> Output: ${output:0:500}" --tag shell
}

# Usage
remember-output npm test
```

### Session Start Hook

```bash
# Add to ~/.bashrc or ~/.zshrc
nmem-session() {
    echo "📧 Recent Memory Context:"
    nmem context --limit 5
    echo ""
}

# Auto-run when entering project directories
cd() {
    builtin cd "$@"
    if [[ -f ".neural-memory" ]]; then
        nmem-session
    fi
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: CI with Memory

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install NeuralMemory
        run: pip install neural-memory

      - name: Remember deployment
        if: github.ref == 'refs/heads/main'
        run: |
          nmem remember "Deployed ${{ github.sha }} to main" \
            --type workflow \
            --tag deploy \
            --tag ci

      - name: Remember test results
        if: always()
        run: |
          nmem remember "CI run: ${{ job.status }} for ${{ github.sha }}" \
            --type workflow \
            --tag ci
```

### GitLab CI

```yaml
remember-deployment:
  stage: deploy
  script:
    - pip install neural-memory
    - nmem remember "Deployed ${CI_COMMIT_SHA} to ${CI_ENVIRONMENT_NAME}" --type workflow --tag deploy
```

---

## Python API Integration

### Basic Usage

```python
import asyncio
from neural_memory.cli.config import CLIConfig
from neural_memory.cli.storage import PersistentStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def remember(content: str, memory_type: str = "fact") -> dict:
    """Store a memory programmatically."""
    config = CLIConfig.load()
    storage = await PersistentStorage.load(config.get_brain_path())
    brain = await storage.get_brain(storage._current_brain_id)

    encoder = MemoryEncoder(storage, brain.config)
    result = await encoder.encode(content)
    await storage.save()

    return {"fiber_id": result.fiber.id, "neurons": len(result.neurons_created)}

async def recall(query: str) -> dict:
    """Query memories programmatically."""
    config = CLIConfig.load()
    storage = await PersistentStorage.load(config.get_brain_path())
    brain = await storage.get_brain(storage._current_brain_id)

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query(query)

    return {
        "answer": result.context,
        "confidence": result.confidence,
    }

# Usage
asyncio.run(remember("Important decision about architecture"))
print(asyncio.run(recall("architecture decision")))
```

### Flask/FastAPI Integration

```python
from fastapi import FastAPI
from neural_memory.cli.config import CLIConfig
from neural_memory.cli.storage import PersistentStorage
from neural_memory.engine.retrieval import ReflexPipeline

app = FastAPI()

@app.get("/memory/context")
async def get_context(limit: int = 10):
    config = CLIConfig.load()
    storage = await PersistentStorage.load(config.get_brain_path())
    fibers = await storage.get_fibers(limit=limit)

    context = []
    for fiber in fibers:
        if fiber.summary:
            context.append(fiber.summary)

    return {"context": context}

@app.post("/memory/remember")
async def remember(content: str):
    # ... implementation
    pass
```

---

## Coding Principles for Better Memory

### 1. Semantic Commit Messages

```bash
# BAD - Hard to extract meaning
git commit -m "fix bug"

# GOOD - Rich semantic information
git commit -m "fix(auth): handle null email in validateUser

- Added null check at login.py:42
- Prevents crash when user submits empty form
- Related to issue #123"
```

### 2. Structured Memories

```bash
# BAD - Missing context
nmem remember "fixed it"

# GOOD - Rich context
nmem remember "Fixed auth bug: null email caused crash in validateUser(). Added null check at login.py:42. Took 2 hours to debug." --tag auth --tag bugfix
```

### 3. Decision Records

```bash
# GOOD - Decision with rationale
nmem remember "DECISION: Using JWT instead of sessions. REASON: Stateless, scales better for microservices. ALTERNATIVE_CONSIDERED: Redis sessions" --type decision
```

### 4. Error-Solution Pairs

```bash
# GOOD - Problem + solution together
nmem remember "ERROR: 'Cannot read property id of undefined' in UserService. SOLUTION: Add null check before accessing user.id. Fixed in PR #456" --type error
```

### 5. Consistent Entity Naming

```bash
# BAD - Inconsistent
nmem remember "talked to bob"
nmem remember "Bob mentioned..."
nmem remember "Robert said..."

# GOOD - Consistent
nmem remember "Meeting with Bob about AuthService"
nmem remember "Bob suggested using JWT for AuthService"
nmem remember "Bob approved AuthService PR #123"
```

---

## Quick Reference

### CLI Commands

```bash
# Remember
nmem remember "content" [--type TYPE] [--priority N] [--tag TAG]

# Recall
nmem recall "query" [--depth N] [--max-tokens N]

# Context
nmem context [--limit N] [--fresh-only] [--json]

# TODO
nmem todo "task" [--priority N]

# Stats
nmem stats

# Brain management
nmem brain list | create | use | export | import

# Shared mode
nmem shared enable URL | disable | status | test | sync
```

### Memory Types

| Type | Use For |
|------|---------|
| `fact` | Objective information |
| `decision` | Choices made |
| `preference` | User preferences |
| `todo` | Action items |
| `insight` | Learned patterns |
| `context` | Situational info |
| `instruction` | User guidelines |
| `error` | Error patterns |
| `workflow` | Process patterns |
| `reference` | External references |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_MEMORY_DIR` | Data directory | `~/.neural-memory` |
| `NEURAL_MEMORY_BRAIN` | Default brain name | `default` |
| `NEURAL_MEMORY_JSON` | Always output JSON | `false` |

---

## Troubleshooting

### Memory Not Found

1. Check if content was stored: `nmem stats`
2. Try broader query terms
3. Use `--depth 3` for deeper search

### MCP Server Not Working

1. Check Python path: `which python`
2. Test manually: `python -m neural_memory.mcp`
3. Check Claude Code logs for errors

### Slow Queries

1. Use specific queries instead of broad ones
2. Limit context: `nmem context --limit 5`
3. Create separate brains for different projects

### Data Location

- **Linux/Mac**: `~/.neural-memory/brains/`
- **Windows**: `C:\Users\<name>\.neural-memory\brains\`
