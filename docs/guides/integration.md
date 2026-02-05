# Integration Guide

Integrate NeuralMemory with AI assistants, IDEs, and development tools.

## Claude Code Integration

### Option A: MCP Server (Recommended)

NeuralMemory provides a native MCP (Model Context Protocol) server.

#### 1. Install NeuralMemory

```bash
pip install neural-memory
```

#### 2. Configure MCP Server

Add to `~/.claude/mcp_servers.json`:

=== "CLI (recommended)"
    ```json
    {
      "neural-memory": {
        "command": "nmem",
        "args": ["mcp"]
      }
    }
    ```

=== "Entry point"
    ```json
    {
      "neural-memory": {
        "command": "nmem-mcp"
      }
    }
    ```

=== "Python module"
    ```json
    {
      "neural-memory": {
        "command": "python",
        "args": ["-m", "neural_memory.mcp"]
      }
    }
    ```

#### 3. Restart Claude Code

After restarting, Claude has access to:

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory with type, priority, tags |
| `nmem_recall` | Query memories with depth and confidence |
| `nmem_context` | Get recent context for injection |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_stats` | Get brain statistics |
| `nmem_auto` | Auto-capture memories from text |

#### 4. Usage

Claude automatically uses these tools:

```
You: Remember that we decided to use PostgreSQL
Claude: [uses nmem_remember tool]
       Stored the decision about PostgreSQL.

You: What database did we choose?
Claude: [uses nmem_recall tool]
       Based on my memory, you decided to use PostgreSQL.
```

### Option B: CLAUDE.md Instructions

Add to your project's `CLAUDE.md`:

```markdown
## Memory Instructions

At session start, get context:
```bash
nmem context --limit 20 --json
```

When learning something important:
```bash
nmem remember "Important info" --type decision
```

When recalling past information:
```bash
nmem recall "query"
```
```

### Option C: Manual Context Injection

```bash
# Get context and inject at session start
CONTEXT=$(nmem context --json --limit 20)
echo "Recent project context: $CONTEXT"
```

---

## VS Code Extension

NeuralMemory has a dedicated VS Code extension with visual brain exploration and inline memory tools.

### Installation

```bash
cd vscode-extension
npm install && npm run build
```

Then install the generated `.vsix` file via **Extensions > Install from VSIX** or use Extension Developer Host (`F5`).

### Features

| Feature | Description |
|---------|-------------|
| **Memory Tree** | Activity bar sidebar with neurons grouped by type |
| **Graph Explorer** | Cytoscape.js force-directed graph with sub-graph navigation |
| **Encode** | Store selected text or typed input as memories |
| **Recall** | Query memories with depth selection |
| **CodeLens** | Memory counts on functions/classes, comment triggers |
| **Status Bar** | Live brain stats (neurons, synapses, fibers) |
| **WebSocket Sync** | Real-time updates across all views |

### Configuration

In VS Code settings (`neuralmemory.*`):

| Setting | Default | Description |
|---------|---------|-------------|
| `serverUrl` | `http://localhost:8000` | NeuralMemory server URL |
| `pythonPath` | `python` | Python executable path |
| `graphNodeLimit` | `200` | Max nodes shown in graph |
| `codeLensTriggers` | `remember,note,decision,todo` | Comment triggers |

### Usage

1. Start the NeuralMemory server: `nmem serve`
2. Open VS Code — the extension connects automatically
3. Use command palette (`Ctrl+Shift+P`) for:
   - `NeuralMemory: Encode Selection as Memory`
   - `NeuralMemory: Recall Memory`
   - `NeuralMemory: Open Graph Explorer`
   - `NeuralMemory: Switch Brain`

---

## Cursor Integration

### Cursor Rules

Add to `.cursorrules` in your project:

```markdown
## Memory System

This project uses NeuralMemory for persistent context.

### Getting Context
Before starting work:
```bash
nmem context --limit 10
```

### Storing Information
```bash
nmem remember "description" --type decision
nmem remember "error fix" --type error
nmem todo "task description"
```

### Recalling
```bash
nmem recall "query"
```
```

### Cursor Commands

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
    }
  ]
}
```

---

## Windsurf Integration

### Windsurf Rules

Create `.windsurfrules` in your project:

```markdown
## NeuralMemory Integration

### Session Start
```bash
nmem context --fresh-only --limit 10
```

### During Development
- Decisions: `nmem remember "X" --type decision`
- Errors: `nmem remember "X" --type error`
- TODOs: `nmem todo "X" --priority 7`

### Querying
```bash
nmem recall "your query" --depth 2
```
```

### AI Flow Integration

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

### Shell Wrapper

Create `aider-with-memory.sh`:

```bash
#!/bin/bash
echo "Loading memory context..."
CONTEXT=$(nmem context --json --limit 15)

aider --message "Project context from memory:
$CONTEXT

Remember to use 'nmem remember' for important decisions." "$@"
```

### In-Session Commands

```
> /run nmem context
> /run nmem remember "We decided to use FastAPI" --type decision
> /run nmem recall "API framework decision"
```

### Git Hook Integration

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
MSG=$(git log -1 --pretty=%B)
nmem remember "Git commit: $MSG" --tag git --tag auto
```

---

## GitHub Copilot

Add to `.github/copilot-instructions.md`:

```markdown
## Memory Context

Get project context: `nmem context`
Store decisions: `nmem remember "X" --type decision`
Query past info: `nmem recall "X"`
```

---

## VS Code with Continue.dev

In `.continue/config.json`:

```json
{
  "customCommands": [
    {
      "name": "memory-context",
      "description": "Get NeuralMemory context",
      "prompt": "{{#if output}}Project memory context:\n\n{{output}}{{/if}}",
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
git() {
    command git "$@"
    if [[ "$1" == "commit" ]]; then
        local msg=$(command git log -1 --pretty=%B)
        nmem remember "Git commit: $msg" --tag git --type workflow &
    fi
}
```

### Session Start Hook

```bash
nmem-session() {
    echo "Recent Memory Context:"
    nmem context --limit 5
}

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
            --tag deploy

      - name: Remember test results
        if: always()
        run: |
          nmem remember "CI: ${{ job.status }} for ${{ github.sha }}" \
            --type workflow \
            --tag ci
```

---

## Best Practices

### 1. Semantic Commit Messages

```bash
# BAD
git commit -m "fix bug"

# GOOD
git commit -m "fix(auth): handle null email in validateUser

- Added null check at login.py:42
- Prevents crash on empty form submission"
```

### 2. Structured Memories

```bash
# BAD
nmem remember "fixed it"

# GOOD
nmem remember "Fixed auth bug: null email in validateUser(). Added null check at login.py:42." --tag auth --tag bugfix
```

### 3. Decision Records

```bash
nmem remember "DECISION: JWT over sessions. REASON: Stateless scaling. ALTERNATIVE: Redis sessions" --type decision
```

### 4. Error-Solution Pairs

```bash
nmem remember "ERROR: 'Cannot read id of undefined'. SOLUTION: Add null check before user.id" --type error
```

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

1. Use specific queries
2. Limit context: `nmem context --limit 5`
3. Create separate brains per project
