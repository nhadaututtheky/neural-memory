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
from neural_memory.engine.encoder import EncodingResult, MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline, RetrievalResult

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Brain",
    "BrainConfig",
    "Fiber",
    "Neuron",
    "NeuronState",
    "NeuronType",
    "Synapse",
    "SynapseType",
    "Direction",
    # Brain mode (local/shared toggle)
    "BrainMode",
    "BrainModeConfig",
    "SharedConfig",
    "SyncStrategy",
    # Engine
    "MemoryEncoder",
    "EncodingResult",
    "ReflexPipeline",
    "RetrievalResult",
    "DepthLevel",
    # Version
    "__version__",
]
