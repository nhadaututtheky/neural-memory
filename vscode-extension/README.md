# NeuralMemory for VS Code

A visual brain explorer, inline recall, and memory management extension for [NeuralMemory](https://github.com/nhadaututtheky/neural-memory) — the reflex-based memory system for AI agents.

## Features

### Memory Tree View

Browse your brain's neurons directly in the sidebar. Memories are grouped by type (Concepts, Entities, Actions, Time, State) with counts and relative timestamps. Click any neuron to instantly recall related memories.

![Memory Tree](https://raw.githubusercontent.com/nhadaututtheky/neural-memory/main/docs/img/tree-view.png)

### Graph Explorer

Visualize your entire brain as an interactive force-directed graph. Neurons are color-coded by type, synapses show weighted connections. Double-click any node to zoom into its neighborhood.

- Drag to pan, scroll to zoom
- Click nodes for details and quick recall
- Respects VS Code dark/light themes

![Graph Explorer](https://raw.githubusercontent.com/nhadaututtheky/neural-memory/main/docs/img/graph-explorer.png)

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

- [NeuralMemory](https://pypi.org/project/neural-memory/) Python package (`pip install neural-memory`)
- Python 3.10+
- A configured brain (`nmem brain create my-brain && nmem brain use my-brain`)

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
3. Create a brain: `nmem brain create my-brain`
4. Set it active: `nmem brain use my-brain`
5. Start the server: run **NeuralMemory: Start Server** from the command palette, or enable `neuralmemory.autoStart`
6. Open the NeuralMemory sidebar (brain icon in the activity bar)

## Status Bar

The status bar shows your active brain and live statistics:

```
$(brain) my-brain | N:512 S:1024 F:256
```

- **N** = Neurons, **S** = Synapses, **F** = Fibers
- Click to switch brains
- Updates every 30 seconds (or instantly via WebSocket)

## License

MIT
