# NeuralMemory

[![CI](https://github.com/neural-memory/neural-memory/workflows/CI/badge.svg)](https://github.com/neural-memory/neural-memory/actions)
[![Coverage](https://codecov.io/gh/neural-memory/neural-memory/branch/main/graph/badge.svg)](https://codecov.io/gh/neural-memory/neural-memory)
[![PyPI](https://img.shields.io/pypi/v/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![Python](https://img.shields.io/pypi/pyversions/neural-memory.svg)](https://pypi.org/project/neural-memory/)
[![License](https://img.shields.io/github/license/neural-memory/neural-memory.svg)](https://github.com/neural-memory/neural-memory/blob/main/LICENSE)

**Reflex-based memory system for AI agents** - retrieval through activation, not search.

NeuralMemory stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories are retrieved through associative recall - activating related concepts until the relevant memory emerges.

## Why NeuralMemory?

AI agents (like Claude, GPT, etc.) face fundamental memory limitations:

- **Limited context windows** - Cannot complete large projects across sessions
- **Session amnesia** - Forget everything between conversations
- **No knowledge sharing** - Cannot share learned patterns with other agents
- **Context overflow** - Important early context gets lost as windows fill up

**NeuralMemory solves these problems:**

- **Persistent memory** that survives across sessions
- **Efficient retrieval** - inject only relevant context, not everything
- **Shareable brains** - export/import learned patterns like Git repos
- **Project-bounded** - optimize memory for active project timeframes

## Installation

```bash
pip install neural-memory
```

With optional dependencies:

```bash
# With FastAPI server
pip install neural-memory[server]

# With Vietnamese NLP support
pip install neural-memory[nlp-vi]

# With all features
pip install neural-memory[all]
```

## Quick Start

### CLI (Simplest)

```bash
# Store a memory
nmem remember "Fixed auth bug by adding null check in login.py"

# Query memories
nmem recall "auth bug"

# Get recent context (for AI injection)
nmem context

# Manage brains
nmem brain list
nmem brain create work
nmem brain use work
```

### Python API

```python
from neural_memory import Brain, MemoryEncoder, ReflexPipeline

# Create a brain
brain = Brain.create("my_agent_brain")

# Encode memories
encoder = MemoryEncoder(brain)
encoder.encode("Met with Alice at the coffee shop to discuss the API design")
encoder.encode("Decided to use FastAPI for the backend, Alice suggested adding rate limiting")
encoder.encode("Completed the authentication module, took 3 hours")

# Query memories through activation (not search!)
pipeline = ReflexPipeline(brain)

result = pipeline.query("What did Alice suggest?")
print(result.answer)  # "Alice suggested adding rate limiting"

result = pipeline.query("What was decided about the backend?")
print(result.answer)  # "Use FastAPI for the backend"
```

## Core Concepts

### Neurons

The basic unit of memory. Each neuron represents a distinct piece of information:

- **Time neurons**: "3pm", "yesterday", "last week"
- **Entity neurons**: "Alice", "coffee shop", "FastAPI"
- **Action neurons**: "discussed", "decided", "completed"
- **State neurons**: "happy", "frustrated", "confident"
- **Concept neurons**: "API design", "authentication", "rate limiting"

### Synapses

Connections between neurons with semantic meaning:

- **Temporal**: `happened_at`, `before`, `after`
- **Causal**: `caused_by`, `leads_to`, `enables`
- **Associative**: `co_occurs`, `related_to`, `similar_to`
- **Semantic**: `is_a`, `has_property`, `involves`

### Fibers

Memory clusters - subgraphs of related neurons representing a coherent experience or concept.

### Spreading Activation

When you query, NeuralMemory:

1. **Decomposes** your query into signals (time, entities, intent)
2. **Activates** matching anchor neurons
3. **Spreads** activation through synapses
4. **Finds intersections** where multiple signals converge
5. **Extracts** the relevant subgraph as context

This mimics how human memory works - you don't "search" for memories, they emerge through association.

## Multi-language Support

NeuralMemory supports both English and Vietnamese from the start:

```python
# Vietnamese
encoder.encode("Chiều nay 3h uống cafe ở Viva với Minh")
result = pipeline.query("Chiều nay làm gì?")

# English
encoder.encode("Had coffee at Viva with Minh at 3pm")
result = pipeline.query("What did I do this afternoon?")
```

## Server Mode

Run NeuralMemory as a service:

```bash
pip install neural-memory[server]
uvicorn neural_memory.server:app --reload
```

API endpoints:

```
POST /memory/encode    - Store new memory
POST /memory/query     - Query memories
POST /brain/create     - Create new brain
GET  /brain/{id}       - Get brain info
GET  /brain/{id}/export - Export brain snapshot
```

## Brain Sharing

Export and share learned patterns:

```python
from neural_memory.sharing import BrainExporter, BrainImporter

# Export
exporter = BrainExporter()
snapshot = exporter.export(brain, time_range=(start, end))
exporter.to_json(snapshot, "my_brain.json")

# Import into another agent
importer = BrainImporter()
importer.import_brain("my_brain.json", target_brain)
```

## Development

```bash
# Clone and setup
git clone https://github.com/neural-memory/neural-memory
cd neural-memory
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v --cov=neural_memory

# Type check
mypy src/

# Lint
ruff check src/ tests/
```

## Documentation

Full documentation: [https://neural-memory.github.io/neural-memory](https://neural-memory.github.io/neural-memory)

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE) for details.
