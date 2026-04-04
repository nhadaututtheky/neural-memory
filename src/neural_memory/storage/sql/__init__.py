"""Unified SQL storage layer for neural-memory.

Provides a single set of storage mixins that work with any SQL backend
via the Dialect abstraction. Configuration determines which database
engine to use — no code duplication.
"""

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.storage.sql.sql_storage import SQLStorage

__all__ = ["Dialect", "SQLStorage"]
