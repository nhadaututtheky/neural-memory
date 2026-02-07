# NeuralMemory for VS Code

A visual brain explorer, inline recall, and memory management extension for [NeuralMemory](https://github.com/nhadaututtheky/neural-memory) — the reflex-based memory system for AI agents.

## Features

### Memory Tree View

Browse your brain's neurons directly in the sidebar. Memories are grouped by type (Concepts, Entities, Actions, Time, State, Spatial, Sensory, Intent) with counts and relative timestamps. Click any neuron to instantly recall related memories.

### Graph Explorer

Visualize your entire brain as an interactive force-directed graph. Neurons are color-coded by type, synapses show weighted connections. Double-click any node to zoom into its neighborhood.

- Drag to pan, scroll to zoom
- Click nodes for details and quick recall
- Respects VS Code dark/light themes

### Eternal Context

Project decisions, instructions, and tech stack are stored directly in the neural graph — fully discoverable via recall and spreading activation. No more JSON sidecar files.

- Save project context, decisions, and instructions that persist across sessions
- Recap at session start with configurable detail levels (quick / detailed / full)
- Topic-based recall for specific areas (e.g., "auth", "database")

### Codebase Indexing

Index your codebase into neural memory for code-aware recall. Scans Python files and creates neurons for functions, classes, and imports.

- "Where is X implemented?" queries work after indexing
- Keyboard shortcut: `Ctrl+Shift+M I`

### External Memory Import

Import memories from other systems into NeuralMemory:

- ChromaDB, Mem0, AWF, Cognee, Graphiti, LlamaIndex
- Full and incremental sync support

### Encode Memories from the Editor

Select any text and encode it as a memory, or use comment triggers (`remember:`, `note:`, `decision:`, `todo:`) to get inline suggestions via CodeLens.

### CodeLens Integration

Functions and classes show memory counts inline. Click to recall related memories or encode new ones. Works across Python, TypeScript, JavaScript, Go, Rust, Java, and C#.

### Real-Time Sync

WebSocket connection keeps your tree view, graph, and status bar updated in real time as memories are created or modified from any source (CLI, MCP, other editors).

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| Encode Selection as Memory | `Ctrl+Shift+M E` | Encode selected text with optional tags |
| Encode Text as Memory | — | Type and encode memory content |
| Recall Memory | `Ctrl+Shift+M R` | Query brain with selectable search depth |
| Open Graph Explorer | `Ctrl+Shift+M G` | Interactive neuron/synapse visualization |
| Index Codebase | `Ctrl+Shift+M I` | Scan and index code into neural memory |
| Recap Session Context | `Ctrl+Shift+M C` | Load saved context at session start |
| Recap by Topic | — | Search for specific topic in context |
| Save Eternal Context | — | Save project decisions/instructions |
| Eternal Context Status | — | View memory counts and session state |
| Import Memories | — | Import from ChromaDB, Mem0, AWF, etc. |
| Switch Brain | Click status bar | Switch between local brains |
| Create Brain | — | Create a new isolated brain |
| Refresh Memory Tree | Tree header icon | Force refresh from server |
| Start Server | — | Start local NeuralMemory server |
| Connect to Server | — | Connect to a remote server |

> On macOS, use `Cmd` instead of `Ctrl`.

## Recall Workflow

1. Trigger recall (`Ctrl+Shift+M R`) and type your query
2. Select search depth: Auto, Instant, Context, Habit, or Deep
3. Choose from matched memories:
   - **Paste** into the active editor
   - **Copy** to clipboard
   - **View details** with confidence score, latency, and matched fiber IDs

## Requirements

- [NeuralMemory](https://pypi.org/project/neural-memory/) v0.9.5+ (`pip install neural-memory`)
- Python 3.11+
- A configured brain (`nmem init` or `nmem brain create my-brain && nmem brain use my-brain`)

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `neuralmemory.pythonPath` | `"python"` | Python interpreter with neural-memory installed |
| `neuralmemory.autoStart` | `false` | Auto-start the server on activation |
| `neuralmemory.serverUrl` | `"http://127.0.0.1:8000"` | NeuralMemory server URL |
| `neuralmemory.graphNodeLimit` | `1000` | Max nodes in graph explorer (50-10000) |
| `neuralmemory.codeLensEnabled` | `true` | Show CodeLens hints for functions and comments |
| `neuralmemory.commentTriggers` | `["remember:", "note:", "decision:", "todo:"]` | Comment patterns that trigger encode suggestions |

## Getting Started

1. Install the extension
2. Install NeuralMemory: `pip install neural-memory`
3. Run setup: `nmem init` (creates config, brain, and auto-configures MCP)
4. Start the server: run **NeuralMemory: Start Server** from the command palette, or enable `neuralmemory.autoStart`
5. Open the NeuralMemory sidebar (brain icon in the activity bar)

## Status Bar

The status bar shows your active brain and live statistics:

```
$(brain) my-brain | N:512 S:1024 F:256
```

- **N** = Neurons, **S** = Synapses, **F** = Fibers
- Click to switch brains
- Updates every 30 seconds (or instantly via WebSocket)

## Support

If you find NeuralMemory useful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/vietnamit)

## License

MIT
