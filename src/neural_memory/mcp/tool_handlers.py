"""MCP tool handler orchestrator.

ToolHandler composes domain-specific handler mixins via multiple inheritance.
Each domain mixin lives in its own file for maintainability (<500 LOC each).

The ToolHandler class is inherited by MCPServer in server.py.
All methods access storage/config via self.get_storage() and self.config
from the MCPServer base class.

Domain handlers:
- StatsHandler: stats, health, todo, storage info
- EvolutionHandler: evolution, suggest, habits, version, transplant
- ProvenanceHandler: source registry, provenance, show
- LifecycleHandler: edit, forget, consolidate, tool_stats, lifecycle
- InstructionHandler: refine, report_outcome
- BudgetHandler: token budget analysis
- TierHandler: auto-tier management (Pro feature)
- RememberHandler: remember, remember_batch (inherited)
- RecallHandler: recall, context (inherited)
"""

from __future__ import annotations

# Re-export frequently imported symbols that other modules
# import from this file (e.g., recall_handler imports ReflexPipeline).
# Domain handler mixins
from neural_memory.mcp.budget_handler import BudgetHandler
from neural_memory.mcp.evolution_handler import EvolutionHandler
from neural_memory.mcp.instruction_handler import InstructionHandler
from neural_memory.mcp.lifecycle_handler import LifecycleHandler
from neural_memory.mcp.provenance_handler import ProvenanceHandler
from neural_memory.mcp.recall_handler import RecallHandler
from neural_memory.mcp.remember_handler import RememberHandler
from neural_memory.mcp.stats_handler import StatsHandler
from neural_memory.mcp.store_handler import StoreHandler
from neural_memory.mcp.tier_handler import TierHandler

# Re-export shared utilities for backward compatibility.
# Many handlers and tests import these from tool_handlers.


class ToolHandler(
    StatsHandler,
    EvolutionHandler,
    ProvenanceHandler,
    LifecycleHandler,
    InstructionHandler,
    BudgetHandler,
    TierHandler,
    StoreHandler,
    RememberHandler,
    RecallHandler,
):
    """Orchestrator mixin composing all MCP tool handler domains.

    Inherits from domain-specific handler mixins. Each mixin provides
    a subset of the nmem_* tool implementations. Protocol stubs for
    shared attributes (config, get_storage, hooks, etc.) are declared
    in each mixin individually.
    """
