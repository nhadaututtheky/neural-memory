"""NeuralMemory - Reflex-based memory system for AI agents."""

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

__version__ = "1.3.0"

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
