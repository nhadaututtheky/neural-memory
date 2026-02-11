"""Codebase encoder: converts extracted code symbols into neural graph structures.

Maps source code into neurons, synapses, and fibers using the
existing NeuralMemory types. No external dependencies.

Neuron type mapping:
    File path   → SPATIAL   (location in codebase)
    Function    → ACTION    (executable behavior)
    Class       → CONCEPT   (abstract structure)
    Method      → ACTION    (executable behavior, metadata.parent = class)
    Import      → ENTITY    (named reference)
    Constant    → ENTITY    (named value)

Synapse mapping:
    contains    → CONTAINS  (weight 1.0)
    is_a        → IS_A      (weight 0.9)
    imports     → RELATED_TO (weight 0.7)
    co_occurs   → CO_OCCURS  (weight 0.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.encoder import EncodingResult
from neural_memory.extraction.codebase import CodeSymbolType, get_extractor

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

_SYMBOL_TYPE_TO_NEURON: dict[CodeSymbolType, NeuronType] = {
    CodeSymbolType.FUNCTION: NeuronType.ACTION,
    CodeSymbolType.CLASS: NeuronType.CONCEPT,
    CodeSymbolType.METHOD: NeuronType.ACTION,
    CodeSymbolType.IMPORT: NeuronType.ENTITY,
    CodeSymbolType.CONSTANT: NeuronType.ENTITY,
}

_RELATION_TO_SYNAPSE: dict[str, tuple[SynapseType, float]] = {
    "contains": (SynapseType.CONTAINS, 1.0),
    "is_a": (SynapseType.IS_A, 0.9),
    "imports": (SynapseType.RELATED_TO, 0.7),
    "co_occurs": (SynapseType.CO_OCCURS, 0.5),
}

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
    }
)
_DEFAULT_EXCLUDE: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        "build",
        "dist",
        ".next",
        "vendor",
    }
)


class CodebaseEncoder:
    """Encodes source code into the neural memory graph."""

    def __init__(self, storage: NeuralStorage, config: BrainConfig) -> None:
        self._storage = storage
        self._config = config

    async def index_file(
        self,
        file_path: Path,
        tags: set[str] | None = None,
    ) -> EncodingResult:
        """Index a single source file into neural graph.

        Args:
            file_path: Path to the source file.
            tags: Optional tags for the fiber.

        Returns:
            EncodingResult with created neurons, synapses, and fiber.
        """
        extractor = get_extractor(file_path.suffix)
        symbols, relationships = extractor.extract_file(file_path)

        neurons_created: list[Neuron] = []
        synapses_created: list[Synapse] = []

        # 1. Create file neuron (SPATIAL)
        file_neuron = Neuron.create(
            type=NeuronType.SPATIAL,
            content=str(file_path),
            metadata={
                "indexed": True,
                "symbol_count": len(symbols),
            },
        )
        await self._storage.add_neuron(file_neuron)
        neurons_created.append(file_neuron)

        # 2. Create symbol neurons
        symbol_id_map: dict[str, str] = {str(file_path): file_neuron.id}

        for sym in symbols:
            neuron_type = _SYMBOL_TYPE_TO_NEURON.get(sym.symbol_type, NeuronType.ENTITY)
            metadata: dict[str, Any] = {
                "symbol_type": sym.symbol_type.value,
                "file_path": sym.file_path,
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "indexed": True,
            }
            if sym.signature:
                metadata["signature"] = sym.signature
            if sym.docstring:
                metadata["docstring"] = sym.docstring
            if sym.parent:
                metadata["parent"] = sym.parent

            # Build a unique key for this symbol
            sym_key = f"{sym.parent}.{sym.name}" if sym.parent else sym.name

            neuron = Neuron.create(
                type=neuron_type,
                content=sym_key,
                metadata=metadata,
            )
            await self._storage.add_neuron(neuron)
            neurons_created.append(neuron)
            symbol_id_map[sym_key] = neuron.id

        # 3. Create synapses from relationships
        for rel in relationships:
            source_id = symbol_id_map.get(rel.source)
            target_id = symbol_id_map.get(rel.target)

            if not source_id or not target_id:
                continue

            synapse_info = _RELATION_TO_SYNAPSE.get(rel.relation)
            if not synapse_info:
                continue

            synapse_type, weight = synapse_info
            synapse = Synapse.create(
                source_id=source_id,
                target_id=target_id,
                type=synapse_type,
                weight=weight,
            )
            await self._storage.add_synapse(synapse)
            synapses_created.append(synapse)

        # 4. Create co-occurrence synapses (capped to avoid O(n²) explosion)
        symbol_neurons = neurons_created[1:]  # Skip file neuron
        max_co_occurs = 5  # Max files: create all pairs; large files: skip
        if len(symbol_neurons) <= max_co_occurs:
            for i, neuron_a in enumerate(symbol_neurons):
                for neuron_b in symbol_neurons[i + 1 :]:
                    synapse = Synapse.create(
                        source_id=neuron_a.id,
                        target_id=neuron_b.id,
                        type=SynapseType.CO_OCCURS,
                        weight=0.5,
                    )
                    await self._storage.add_synapse(synapse)
                    synapses_created.append(synapse)

        # 5. Bundle into fiber
        neuron_ids = {n.id for n in neurons_created}
        synapse_ids = {s.id for s in synapses_created}

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=file_neuron.id,
            summary=f"Code index: {file_path.name}",
            tags=(tags or set()) | {"code_index"},
        )
        await self._storage.add_fiber(fiber)

        return EncodingResult(
            fiber=fiber,
            neurons_created=neurons_created,
            neurons_linked=[],
            synapses_created=synapses_created,
        )

    async def index_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        tags: set[str] | None = None,
    ) -> list[EncodingResult]:
        """Index all matching files in a directory recursively.

        Args:
            directory: Root directory to scan.
            extensions: File extensions to index. Defaults to common source extensions.
            exclude_patterns: Directory names to skip.
            tags: Optional tags for all created fibers.

        Returns:
            List of EncodingResult, one per indexed file.
        """
        exts = extensions if extensions is not None else set(_DEFAULT_EXTENSIONS)
        excludes = exclude_patterns if exclude_patterns is not None else set(_DEFAULT_EXCLUDE)

        results: list[EncodingResult] = []
        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix not in exts:
                continue
            if any(p in file_path.parts for p in excludes):
                continue

            try:
                result = await self.index_file(file_path, tags=tags)
                results.append(result)
            except (SyntaxError, UnicodeDecodeError):
                continue

        return results
