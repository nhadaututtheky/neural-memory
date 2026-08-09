"""Setup function for NeuralMemory-Nanobot integration.

One-line initialization that creates storage, registers tools, and
returns a drop-in MemoryStore replacement::

    nm_store = await setup_neural_memory(registry, workspace)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neural_memory.integrations.nanobot.context import NMContext
from neural_memory.integrations.nanobot.memory_store import NMMemoryStore

logger = logging.getLogger(__name__)


async def setup_neural_memory(
    registry: Any,
    workspace: Path,
    brain_id: str = "nanobot",
    db_filename: str = "neural.db",
) -> NMMemoryStore:
    """Initialize NeuralMemory and register tools with Nanobot.

    Args:
        registry: Nanobot's ToolRegistry (any object with ``register(tool)``).
        workspace: Nanobot workspace directory.
        brain_id: Brain identifier (default: ``"nanobot"``).
        db_filename: SQLite database filename (default: ``"neural.db"``).

    Returns:
        NMMemoryStore that can replace Nanobot's ``MemoryStore``.
    """
    from neural_memory.core.brain import Brain
    from neural_memory.storage.factory import open_sqlite_storage

    db_dir = workspace / "memory"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / db_filename

    # New Nanobot workspaces use unified adapter (fresh install policy)
    storage = await open_sqlite_storage(
        str(db_path),
        storage_adapter="unified",
        brain_id=brain_id,
    )

    brain = await storage.get_brain(brain_id)
    if brain is None:
        brain = await storage.find_brain_by_name(brain_id)
    if brain is None:
        brain = Brain.create(name=brain_id, brain_id=brain_id)
        await storage.save_brain(brain)

    storage.set_brain(brain.id)

    ctx = NMContext(storage=storage, brain=brain, config=brain.config)

    from neural_memory.integrations.nanobot.tools import (
        NMContextTool,
        NMHealthTool,
        NMRecallTool,
        NMRememberTool,
    )

    tools = [
        NMRememberTool(ctx),
        NMRecallTool(ctx),
        NMContextTool(ctx),
        NMHealthTool(ctx),
    ]
    for tool in tools:
        registry.register(tool)

    logger.info(
        "NeuralMemory: registered %d tools for brain '%s' at %s",
        len(tools),
        brain_id,
        db_path,
    )

    return NMMemoryStore(ctx, workspace)
