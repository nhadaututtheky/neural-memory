"""NeuralMemory - Reflex-based memory system for AI agents.

Symbols at package level (Brain, Neuron, MemoryEncoder, ...) are lazily
imported on first attribute access via PEP 562 ``__getattr__``. This keeps
``import neural_memory`` cheap so cold-start callers (hooks, CLI startup)
do not pay the ~200ms cost of loading the full engine + storage stack.

All public symbols remain accessible exactly as before:
    >>> from neural_memory import Brain        # lazy
    >>> import neural_memory; neural_memory.Brain  # lazy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "4.57.0"

# Map public name -> (module, attribute) for lazy resolution.
_LAZY: dict[str, tuple[str, str]] = {
    "Brain": ("neural_memory.core.brain", "Brain"),
    "BrainConfig": ("neural_memory.core.brain", "BrainConfig"),
    "BrainMode": ("neural_memory.core.brain_mode", "BrainMode"),
    "BrainModeConfig": ("neural_memory.core.brain_mode", "BrainModeConfig"),
    "SharedConfig": ("neural_memory.core.brain_mode", "SharedConfig"),
    "SyncStrategy": ("neural_memory.core.brain_mode", "SyncStrategy"),
    "Fiber": ("neural_memory.core.fiber", "Fiber"),
    "Neuron": ("neural_memory.core.neuron", "Neuron"),
    "NeuronState": ("neural_memory.core.neuron", "NeuronState"),
    "NeuronType": ("neural_memory.core.neuron", "NeuronType"),
    "Direction": ("neural_memory.core.synapse", "Direction"),
    "Synapse": ("neural_memory.core.synapse", "Synapse"),
    "SynapseType": ("neural_memory.core.synapse", "SynapseType"),
    "TransplantFilter": ("neural_memory.engine.brain_transplant", "TransplantFilter"),
    "TransplantResult": ("neural_memory.engine.brain_transplant", "TransplantResult"),
    "BrainVersion": ("neural_memory.engine.brain_versioning", "BrainVersion"),
    "VersionDiff": ("neural_memory.engine.brain_versioning", "VersionDiff"),
    "VersioningEngine": ("neural_memory.engine.brain_versioning", "VersioningEngine"),
    "EncodingResult": ("neural_memory.engine.encoder", "EncodingResult"),
    "MemoryEncoder": ("neural_memory.engine.encoder", "MemoryEncoder"),
    "CoActivation": ("neural_memory.engine.reflex_activation", "CoActivation"),
    "ReflexActivation": ("neural_memory.engine.reflex_activation", "ReflexActivation"),
    "DepthLevel": ("neural_memory.engine.retrieval", "DepthLevel"),
    "ReflexPipeline": ("neural_memory.engine.retrieval", "ReflexPipeline"),
    "RetrievalResult": ("neural_memory.engine.retrieval", "RetrievalResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_path, attr_name = _LAZY[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'neural_memory' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.core.brain_mode import (
        BrainMode,
        BrainModeConfig,
        SharedConfig,
        SyncStrategy,
    )
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.synapse import Direction, Synapse, SynapseType
    from neural_memory.engine.brain_transplant import TransplantFilter, TransplantResult
    from neural_memory.engine.brain_versioning import BrainVersion, VersionDiff, VersioningEngine
    from neural_memory.engine.encoder import EncodingResult, MemoryEncoder
    from neural_memory.engine.reflex_activation import CoActivation, ReflexActivation
    from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline, RetrievalResult


__all__ = [
    "__version__",
    "Brain",
    "BrainConfig",
    "BrainMode",
    "BrainModeConfig",
    "BrainVersion",
    "CoActivation",
    "DepthLevel",
    "Direction",
    "EncodingResult",
    "Fiber",
    "MemoryEncoder",
    "Neuron",
    "NeuronState",
    "NeuronType",
    "ReflexActivation",
    "ReflexPipeline",
    "RetrievalResult",
    "SharedConfig",
    "Synapse",
    "SynapseType",
    "SyncStrategy",
    "TransplantFilter",
    "TransplantResult",
    "VersionDiff",
    "VersioningEngine",
]
