# Installation

## Requirements

- Python 3.11 or higher
- pip or pipx

## Basic Installation

Install NeuralMemory from PyPI:

```bash
pip install neural-memory
```

This installs the **base profile** (≤4 direct runtime deps: `aiosqlite`, `typer`, `rich`, `networkx`) with CLI, MCP stdio, SQLite, and in-memory storage.

Fresh installs default MCP tool discovery to the **standard** tier (10 tools). Existing configs without `[tool_tier]` keep **full** discovery for compatibility.

## Optional Dependencies

NeuralMemory has optional features you can install as needed:

### Sync / shared HTTP client

```bash
pip install neural-memory[sync]
```

Includes:

- `aiohttp` for remote SharedStorage and WebSocket sync clients

### Server (FastAPI + Web UI)

```bash
pip install neural-memory[server]
```

Includes:

- FastAPI REST API + pydantic models
- Interactive Web UI for brain visualization
- WebSocket sync support (`aiohttp` included)

### Vietnamese NLP

```bash
pip install neural-memory[nlp-vi]
```

Includes:

- underthesea - Vietnamese NLP toolkit
- pyvi - Vietnamese word segmentation

### English NLP

```bash
pip install neural-memory[nlp-en]
```

Includes:

- spaCy - Industrial NLP
- en_core_web_sm model

### All Features

```bash
pip install neural-memory[all]
```

Installs everything above.

## Development Installation

For contributing or local development:

```bash
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory
pip install -e ".[dev]"
pre-commit install
```

## Verify Installation

```bash
# Check CLI
nmem --version

# Check help
nmem --help

# Check brain status
nmem brain list
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_MEMORY_DIR` | Data directory | `~/.neural-memory` |
| `NEURAL_MEMORY_BRAIN` | Default brain name | `default` |
| `NEURAL_MEMORY_JSON` | Always output JSON | `false` |

## Troubleshooting

### Command not found

If `nmem` is not found after installation:

```bash
# Check if scripts are in PATH
python -m neural_memory.cli --help

# Or use pipx for isolated install
pipx install neural-memory
```

### Permission errors on Windows

Run terminal as Administrator or install with `--user`:

```bash
pip install --user neural-memory
```

### Missing dependencies

If optional features fail:

```bash
# Reinstall with all dependencies
pip install --force-reinstall neural-memory[all]
```
