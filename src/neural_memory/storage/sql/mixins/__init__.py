"""Unified storage mixins — dialect-agnostic SQL operations.

These mixins merge SQLite and PostgreSQL implementations into a single
dialect-agnostic layer, using ``self._dialect`` for all SQL differences.

Mixins (must be composed into a host class that provides ``self._dialect``):
    NeuronMixin      — neuron + neuron_state CRUD (all FTS methods from SQLite)
    SynapseMixin     — synapse CRUD + BFS graph traversal
    FiberMixin       — fiber CRUD + pinning + stats
    BrainOpsMixin    — brain save/get/export/import + stats
    TypedMemoryMixin — typed memory CRUD with expiry
    CognitiveMixin   — cognitive_state + hot_index + knowledge_gaps

Row mappers (dialect-agnostic, auto-detect SQLite vs PostgreSQL datetime format):
    row_to_neuron, row_to_neuron_state, row_to_synapse, row_to_fiber,
    row_to_brain, row_to_typed_memory, provenance_to_dict, _row_to_joined_synapse
"""

from neural_memory.storage.sql.mixins.brain_ops import BrainOpsMixin
from neural_memory.storage.sql.mixins.cognitive import CognitiveMixin
from neural_memory.storage.sql.mixins.fibers import FiberMixin
from neural_memory.storage.sql.mixins.neurons import NeuronMixin
from neural_memory.storage.sql.mixins.synapses import SynapseMixin
from neural_memory.storage.sql.mixins.typed_memory import TypedMemoryMixin
from neural_memory.storage.sql.row_mappers import (
    _row_to_joined_synapse,
    provenance_to_dict,
    row_to_brain,
    row_to_fiber,
    row_to_neuron,
    row_to_neuron_state,
    row_to_synapse,
    row_to_typed_memory,
)

__all__ = [
    "NeuronMixin",
    "SynapseMixin",
    "FiberMixin",
    "BrainOpsMixin",
    "TypedMemoryMixin",
    "CognitiveMixin",
    "row_to_neuron",
    "row_to_neuron_state",
    "row_to_synapse",
    "row_to_fiber",
    "row_to_brain",
    "row_to_typed_memory",
    "provenance_to_dict",
    "_row_to_joined_synapse",
]
