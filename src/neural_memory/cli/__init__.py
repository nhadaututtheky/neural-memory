"""Neural Memory CLI.

Simple command-line interface for storing and retrieving memories.

Usage:
    nmem remember "content"     Store a memory
    nmem recall "query"         Query memories
    nmem context                Get recent context
    nmem brain list             List brains
    nmem brain use <name>       Switch brain
"""

from neural_memory.cli.main import app, main

__all__ = ["app", "main"]
