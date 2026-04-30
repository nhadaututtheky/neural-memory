# Contributing to NeuralMemory

Thank you for your interest in contributing to NeuralMemory! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Setup Steps

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/neural-memory
   cd neural-memory
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Verify setup**

   ```bash
   nmem doctor --dev
   pytest tests/ -v
   mypy src/
   ruff check src/ tests/
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch from `main`
2. Make your changes
3. Write/update tests
4. Ensure all checks pass
5. Submit a pull request

### Code Style

We use the following tools:

- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pytest** - Testing

Run all checks:

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/

# Tests
pytest tests/ -v --cov=neural_memory
```

### Type Hints

All functions must have type hints:

```python
# Good
def process_neurons(
    neurons: list[Neuron],
    threshold: float = 0.5
) -> list[ActivationResult]:
    ...

# Bad - missing types
def process_neurons(neurons, threshold=0.5):
    ...
```

### Immutability

Prefer immutable data structures. Never mutate input arguments:

```python
# Good - create new object
def update_neuron(neuron: Neuron, new_content: str) -> Neuron:
    return Neuron(
        id=neuron.id,
        type=neuron.type,
        content=new_content,
        metadata=neuron.metadata,
    )

# Bad - mutation
def update_neuron(neuron: Neuron, new_content: str) -> Neuron:
    neuron.content = new_content  # Mutation!
    return neuron
```

### Error Handling

Use specific exceptions with context:

```python
# Good
try:
    result = storage.get_neuron(neuron_id)
except NeuronNotFoundError as e:
    logger.error(f"Neuron {neuron_id} not found: {e}")
    raise RetrievalError(f"Cannot retrieve neuron: {e}") from e

# Bad
try:
    result = storage.get_neuron(neuron_id)
except:
    pass
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
└── e2e/           # End-to-end API tests
```

### Writing Tests

- Test files: `test_<module>.py`
- Test functions: `test_<description>`
- Use fixtures from `conftest.py`
- Aim for 80%+ coverage

Example:

```python
import pytest
from neural_memory.core.neuron import Neuron, NeuronType

class TestNeuron:
    def test_create_time_neuron(self) -> None:
        neuron = Neuron(
            id="test-1",
            type=NeuronType.TIME,
            content="3pm",
            metadata={"hour": 15},
        )
        assert neuron.type == NeuronType.TIME
        assert neuron.content == "3pm"

    def test_neuron_is_immutable(self) -> None:
        neuron = Neuron(...)
        with pytest.raises(AttributeError):
            neuron.content = "new content"
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=neural_memory --cov-report=term-missing

# Specific test file
pytest tests/unit/test_neuron.py -v

# Specific test
pytest tests/unit/test_neuron.py::TestNeuron::test_create_time_neuron -v
```

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure CI passes** - all checks must be green
5. **Request review** from maintainers

### PR Title Format

Use conventional commits format:

- `feat: add brain export functionality`
- `fix: correct activation decay calculation`
- `docs: update quickstart guide`
- `refactor: simplify synapse weight updates`
- `test: add integration tests for query flow`

### PR Description

Include:

- What changes were made
- Why the changes were needed
- How to test the changes
- Any breaking changes

## Reporting Issues

### Bug Reports

Include:

- Python version
- NeuralMemory version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternatives considered

## Questions?

- Open a [Discussion](https://github.com/neural-memory/neural-memory/discussions)
- Check existing [Issues](https://github.com/neural-memory/neural-memory/issues)

Thank you for contributing!
