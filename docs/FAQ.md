# NeuralMemory FAQ

## Installation

### Q: `pip install neural-memory` installs what?

- **Core**: pydantic, networkx, python-dateutil, typer, aiohttp, aiosqlite, rich
- **CLI tools**: `nmem`, `neural-memory`, `nmem-mcp`
- **Optional extras**:
  - `[server]` — FastAPI + Uvicorn
  - `[neo4j]` — Neo4j graph database
  - `[nlp-en]` — English NLP (spaCy)
  - `[nlp-vi]` — Vietnamese NLP (underthesea, pyvi)
  - `[all]` — All of the above
  - `[dev]` — Development tools (pytest, ruff, mypy, etc.)

### Q: `pip` not working on Windows?

Use `python -m pip` instead:

```powershell
python -m pip install neural-memory[all]
```

### Q: How to install from source?

```bash
git clone https://github.com/nhadaututtheky/neural-memory.git
cd neural-memory
pip install -e ".[all,dev]"
```

The `-e` flag enables editable mode — code changes take effect immediately without reinstalling.

### Q: Too many commands — is there a simpler UI/UX approach?

Yes. If you use VS Code, the **NeuralMemory extension** provides a full GUI — no terminal commands needed:

- **Encode memory**: Select text → `Ctrl+Shift+M E`
- **Query memory**: `Ctrl+Shift+M Q` → type your question
- **Start/stop server**: `Ctrl+Shift+P` → NeuralMemory: Start Server
- **Switch brain**: `Ctrl+Shift+P` → NeuralMemory: Switch Brain
- **View graph**: `Ctrl+Shift+P` → NeuralMemory: Open Graph View

The sidebar panel also shows neurons, fibers, and brain stats at a glance.

### Q: How to update to the latest version?

```bash
pip install --upgrade neural-memory[all]
```

If installed from source:

```bash
git pull
pip install -e ".[all,dev]"
```

## VS Code Extension

### Q: How to use NeuralMemory without writing Python code?

Install the VS Code extension:

1. Open VS Code
2. Go to Extensions (`Ctrl+Shift+X`)
3. Search **NeuralMemory**
4. Click **Install**

> **Note**: The extension still requires Python + `neural-memory` package installed on the machine as its backend.

### Q: Extension not showing data?

1. Start the server: `Ctrl+Shift+P` → **NeuralMemory: Start Server**
2. Switch brain if needed: `Ctrl+Shift+P` → **NeuralMemory: Switch Brain** → select your brain
3. Click refresh

### Q: Server running on a different port than the extension expects?

The extension defaults to port `8000`. If your server runs on a different port, update the setting:

1. Open VS Code Settings (`Ctrl+,`)
2. Search `neuralmemory.serverUrl`
3. Set it to your server's URL (e.g. `http://127.0.0.1:8080`)

### Q: Server is running, correct port set, but extension still shows nothing?

A new brain starts empty — there is no data to display yet. You need to encode at least one memory first:

1. Select any text in your editor
2. Press `Ctrl+Shift+M E` to encode it
3. Click refresh in the NeuralMemory sidebar

After encoding, neurons and fibers will appear in the sidebar.

### Q: I opened the server URL in the browser but only see JSON info, not my data?

The root URL (`/`) only shows basic API info. To view your data:

- **Graph visualization (UI)**: go to `/ui` (e.g. `http://127.0.0.1:8000/ui`)
- **API documentation (Swagger)**: go to `/docs` (e.g. `http://127.0.0.1:8000/docs`)
- **Neurons list (API)**: go to `/memory/neurons` (requires `X-Brain-ID` header)

The VS Code sidebar is the main way to browse your neurons and fibers.

## Per-Project Configuration

### Q: How do I keep AI instructions/memories separate between projects?

**Short answer: NeuralMemory already handles this automatically.**

When you start a session (`nmem_session`), NeuralMemory auto-detects your git branch, commit, and repo name. Memories are tagged with `branch:<name>`, and recall queries are enriched with your current session context (feature, task, branch). This means relevant memories naturally surface for the project/branch you're working in.

**No configuration needed** — just use NeuralMemory normally and it will prioritize context from your current branch and feature.

For **full isolation** between completely unrelated projects, use separate brains:

```bash
nmem brain create my-web-app
nmem brain create my-ml-pipeline
nmem brain switch my-web-app
```

Or configure per-project in `.mcp.json` (for Claude Code):

```jsonc
// my-web-app/.mcp.json
{
  "mcpServers": {
    "neuralmemory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"],
      "env": {
        "NEURALMEMORY_BRAIN": "my-web-app"
      }
    }
  }
}
```

| Approach | When to use |
|----------|------------|
| Auto branch tagging (default) | Same project, different features/branches |
| Separate brains | Completely unrelated projects |

### Q: Can I store project-specific instructions that persist across AI sessions?

Yes. Use `nmem_remember` with type `instruction`:

```
nmem_remember(content="Always use Redis for caching in this project", type="instruction", priority=9)
nmem_remember(content="API responses must follow the {success, data, error} format", type="instruction", priority=9)
```

Instructions are stored in the current brain and automatically recalled when relevant. Combined with per-project brains, each project gets its own set of persistent instructions.

You can also use the **Eternal Context** system to save critical project facts that survive across sessions:

```
nmem_eternal(action="save", project_name="my-web-app", tech_stack=["Python", "FastAPI", "Redis"])
```

On next session start, call `nmem_recap()` to reload all saved context instantly.

## Data & Multi-tool Sharing

### Q: Do I need to install NeuralMemory per project?

No. Install once globally and it works for the entire machine:

```bash
python -m pip install neural-memory[all]
```

Data is stored in `~/.neuralmemory/` — not tied to any specific project. All tools (CLI, AntiGravity, Claude Code, VS Code extension) read from the same location.

To separate data per project, use different brains:

```bash
nmem brain create my-project
nmem brain switch my-project
```

### Q: How to share brain data between AntiGravity and Claude Code?

They already share the same brain automatically. Both read/write to the same files:

- **Config**: `~/.neuralmemory/config.toml`
- **Data**: `~/.neuralmemory/brains/<name>.db`

As long as both tools point to the same `current_brain`, all memories are synced. Verify with:

```bash
nmem config show
```

### Q: How do I let Claude Code query my brain using natural language?

Add the NeuralMemory MCP server to `~/.claude.json`:

```json
{
  "mcpServers": {
    "neuralmemory": {
      "command": "python",
      "args": ["-m", "neural_memory.mcp"]
    }
  }
}
```

After restarting Claude Code, just ask naturally:

> "What did we decide about the volume spike feature?"

The agent will automatically call the `recall` tool to search your brain.
