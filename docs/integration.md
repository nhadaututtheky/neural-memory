# Integration Guide

This guide covers how to integrate Neural Memory with AI assistants, development tools, and other applications.

## Quick Start

```bash
# Install
pip install neural-memory

# Store a memory
nmem remember "Fixed auth bug by adding null check in login.py:42"

# Query memories
nmem recall "auth bug"

# Get recent context (for AI injection)
nmem context --json
```

## Integration Patterns

### 1. Claude Code Integration

#### Option A: Manual Context Injection

At the start of each session, inject context:

```bash
# Get recent context as JSON
CONTEXT=$(nmem context --json --limit 20)

# Include in your prompt or system message
echo "Recent project context: $CONTEXT"
```

#### Option B: MCP Server (Coming Soon)

Configure in `~/.claude/mcp_servers.json`:

```json
{
  "neural-memory": {
    "command": "python",
    "args": ["-m", "neural_memory.mcp"]
  }
}
```

This will expose `remember` and `recall` tools directly to Claude.

### 2. Shell Integration

Add to your `.bashrc` or `.zshrc`:

```bash
# Auto-remember git commits
git() {
    command git "$@"
    if [[ "$1" == "commit" ]]; then
        local msg=$(command git log -1 --pretty=%B)
        nmem remember "Git commit: $msg" -t git -t commit
    fi
}

# Remember command outputs
remember-output() {
    local output=$("$@" 2>&1)
    echo "$output"
    nmem remember "Command: $* -> Output: ${output:0:500}" -t shell
}
```

### 3. VS Code Integration

Create a task in `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Remember Selection",
      "type": "shell",
      "command": "nmem remember \"${selectedText}\" -t vscode",
      "problemMatcher": []
    }
  ]
}
```

### 4. CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Remember deployment
  run: |
    pip install neural-memory
    nmem remember "Deployed ${{ github.sha }} to ${{ github.ref }}" -t deploy -t ci
```

### 5. API Integration

```python
import asyncio
from neural_memory.cli.storage import PersistentStorage
from neural_memory.cli.config import CLIConfig
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def remember(content: str) -> dict:
    config = CLIConfig.load()
    storage = await PersistentStorage.load(config.get_brain_path())
    brain = await storage.get_brain(storage._current_brain_id)

    encoder = MemoryEncoder(storage, brain.config)
    result = await encoder.encode(content)
    await storage.save()

    return {"fiber_id": result.fiber.id}

async def recall(query: str) -> dict:
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
asyncio.run(remember("Important meeting notes"))
print(asyncio.run(recall("meeting")))
```

---

## Coding Principles for Better Memory

These principles help Neural Memory understand and retrieve your information more effectively.

### 1. Semantic Commit Messages

Write commits that Neural Memory can understand:

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

When storing memories, include context:

```bash
# BAD - Missing context
nmem remember "fixed it"

# GOOD - Rich context
nmem remember "Fixed auth bug: null email caused crash in validateUser(). Added null check at login.py:42. Took 2 hours to debug." -t auth -t bugfix
```

### 3. Entity Naming

Use consistent names for people, projects, and concepts:

```bash
# BAD - Inconsistent naming
nmem remember "talked to bob about the thing"
nmem remember "Bob mentioned that project"
nmem remember "Robert said..."

# GOOD - Consistent naming
nmem remember "Meeting with Bob about AuthService refactor"
nmem remember "Bob suggested using JWT for AuthService"
nmem remember "Bob approved AuthService PR #123"
```

### 4. Time References

Include temporal context when relevant:

```bash
# GOOD - Clear time reference
nmem remember "Sprint 5 planning: decided to prioritize auth refactor"
nmem remember "Yesterday's standup: blocked on API rate limits"
nmem remember "Q1 2024 goal: improve login performance by 50%"
```

### 5. Causal Chains

Link causes and effects:

```bash
# GOOD - Shows causality
nmem remember "Login failures increased after deploying v2.3.0. Root cause: null email handling removed accidentally in commit abc123"
```

### 6. Error-Solution Pairs

Document problems and their solutions together:

```bash
# GOOD - Problem + Solution
nmem remember "ERROR: 'Cannot read property id of undefined' in UserService. SOLUTION: Add null check before accessing user.id. Fixed in PR #456"
```

### 7. Decision Records

Record why decisions were made:

```bash
# GOOD - Decision + Rationale
nmem remember "DECISION: Using JWT instead of sessions. REASON: Stateless, scales better for microservices. ALTERNATIVE_CONSIDERED: Redis sessions"
```

---

## Output Formats

### JSON Output (for programmatic use)

All commands support `--json` flag:

```bash
nmem recall "auth" --json
```

Output:
```json
{
  "answer": "...",
  "confidence": 0.85,
  "depth_used": 1,
  "neurons_activated": 15,
  "fibers_matched": ["abc123"],
  "latency_ms": 45.2
}
```

### Structured Context (for AI injection)

```bash
nmem context --json --limit 10
```

Output:
```json
{
  "context": "- Fixed auth bug...\n- Meeting with Bob...",
  "count": 10,
  "fibers": [...]
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_MEMORY_DIR` | Data directory | `~/.neural-memory` |
| `NEURAL_MEMORY_BRAIN` | Default brain name | `default` |
| `NEURAL_MEMORY_JSON` | Always output JSON | `false` |

---

## Brain Management

### Multiple Brains

Use different brains for different contexts:

```bash
# Create project-specific brains
nmem brain create work
nmem brain create personal
nmem brain create project-x

# Switch between brains
nmem brain use work
nmem remember "Work meeting notes..."

nmem brain use personal
nmem remember "Personal reminder..."

# List all brains
nmem brain list
```

### Export/Import

Share brains between machines or team members:

```bash
# Export
nmem brain export -o my-brain.json

# Import on another machine
nmem brain import my-brain.json --name shared-brain
```

### Brain Backup

Automated backup script:

```bash
#!/bin/bash
# backup-brain.sh
DATE=$(date +%Y%m%d)
nmem brain export -o "backup-${DATE}.json"
```

---

## Troubleshooting

### Memory Not Found

If recall returns no results:
1. Check if content was stored: `nmem stats`
2. Try broader query terms
3. Use `--depth 3` for deeper search

### Slow Queries

For large brains:
1. Use specific queries instead of broad ones
2. Limit context: `nmem context --limit 5`
3. Consider creating separate brains for different projects

### Data Location

Brain data is stored in:
- **Linux/Mac**: `~/.neural-memory/brains/`
- **Windows**: `C:\Users\<name>\.neural-memory\brains\`
