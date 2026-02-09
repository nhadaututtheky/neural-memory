"""Doc-to-brain training pipeline — train expert brains from documentation.

Processes markdown files into a neural memory brain by:
1. Discovering and chunking documentation files
2. Encoding each chunk through MemoryEncoder (full NLP pipeline)
3. Building heading hierarchy as CONTAINS synapses
4. Optionally running ENRICH consolidation for cross-linking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.doc_chunker import DocChunk, chunk_markdown, discover_files
from neural_memory.engine.encoder import MemoryEncoder

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for doc-to-brain training.

    Attributes:
        domain_tag: Domain tag applied to all chunks (e.g., "react", "k8s").
        brain_name: Target brain name (empty = use current brain).
        min_chunk_words: Skip chunks with fewer words.
        max_chunk_words: Split chunks exceeding this at paragraph boundaries.
        memory_type: Memory type override for all chunks.
        consolidate: Run ENRICH consolidation after encoding.
        extensions: File extensions to include.
    """

    domain_tag: str = ""
    brain_name: str = ""
    min_chunk_words: int = 20
    max_chunk_words: int = 500
    memory_type: str = "reference"
    consolidate: bool = True
    extensions: tuple[str, ...] = (".md",)


@dataclass(frozen=True)
class TrainingResult:
    """Result of a doc-to-brain training run.

    Attributes:
        files_processed: Number of files that were read and chunked.
        chunks_encoded: Number of chunks successfully encoded.
        chunks_skipped: Number of chunks skipped (below min_words).
        chunks_failed: Number of chunks that failed encoding.
        neurons_created: Total neurons created across all chunks.
        synapses_created: Total synapses from encoding (excluding hierarchy).
        hierarchy_synapses: CONTAINS synapses from heading tree.
        enrichment_synapses: Synapses created by ENRICH consolidation.
        brain_name: Name of the brain that was trained.
    """

    files_processed: int
    chunks_encoded: int
    chunks_skipped: int
    chunks_failed: int = 0
    neurons_created: int = 0
    synapses_created: int = 0
    hierarchy_synapses: int = 0
    enrichment_synapses: int = 0
    brain_name: str = "current"


class DocTrainer:
    """Trains a neural memory brain from documentation files.

    Mirrors CodebaseEncoder's architecture: file hierarchy maps to
    heading hierarchy with CONTAINS synapses, while MemoryEncoder
    handles the actual NLP encoding pipeline.

    Key optimizations over naive encoding:
    - skip_conflicts=True: Avoids false-positive conflict detection between doc chunks
    - skip_time_neurons=True: Avoids TIME neuron super-hub (2000 chunks at same time)
    - Per-chunk error isolation: One chunk failure doesn't abort the batch
    - Heading neuron deduplication: Checks storage before creating heading neurons
    """

    def __init__(self, storage: NeuralStorage, config: BrainConfig) -> None:
        self._storage = storage
        self._config = config
        self._encoder = MemoryEncoder(storage, config)

    async def train_directory(
        self,
        directory: Path,
        training_config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train a brain from all documentation in a directory.

        Args:
            directory: Root directory containing documentation files.
            training_config: Training configuration (uses defaults if None).

        Returns:
            TrainingResult with statistics about the training run.
        """
        tc = training_config or TrainingConfig()
        extensions = frozenset(tc.extensions)

        files = discover_files(directory, extensions=extensions)
        if not files:
            return TrainingResult(
                files_processed=0,
                chunks_encoded=0,
                chunks_skipped=0,
                brain_name=tc.brain_name or "current",
            )

        # Collect all chunks from all files
        all_chunks: list[DocChunk] = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)
                continue

            rel_path = str(file_path.relative_to(directory))
            chunks = chunk_markdown(
                text,
                source_file=rel_path,
                min_words=tc.min_chunk_words,
                max_words=tc.max_chunk_words,
            )
            all_chunks.extend(chunks)

        return await self._encode_chunks(
            chunks=all_chunks,
            files_processed=len(files),
            training_config=tc,
        )

    async def train_file(
        self,
        file_path: Path,
        training_config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train a brain from a single documentation file.

        Args:
            file_path: Path to the markdown file.
            training_config: Training configuration (uses defaults if None).

        Returns:
            TrainingResult with statistics about the training run.
        """
        tc = training_config or TrainingConfig()

        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            return TrainingResult(
                files_processed=0,
                chunks_encoded=0,
                chunks_skipped=0,
                brain_name=tc.brain_name or "current",
            )

        chunks = chunk_markdown(
            text,
            source_file=file_path.name,
            min_words=tc.min_chunk_words,
            max_words=tc.max_chunk_words,
        )

        return await self._encode_chunks(
            chunks=chunks,
            files_processed=1,
            training_config=tc,
        )

    async def _encode_chunks(
        self,
        *,
        chunks: list[DocChunk],
        files_processed: int,
        training_config: TrainingConfig,
    ) -> TrainingResult:
        """Encode chunks into neural structures and build heading hierarchy.

        This is the core pipeline:
        1. Encode each chunk via MemoryEncoder.encode() (skip conflicts + time neurons)
        2. Build heading hierarchy as CONCEPT neurons + CONTAINS synapses
        3. Include heading neurons in chunk fibers for retrieval traversal
        4. Optionally run ENRICH consolidation
        """
        tc = training_config
        total_neurons = 0
        total_synapses = 0
        chunks_encoded = 0
        chunks_failed = 0

        # Track heading → neuron ID for hierarchy building
        heading_neuron_ids: dict[tuple[str, ...], str] = {}
        # Track chunk anchor neuron IDs for linking to heading neurons
        chunk_anchors: list[tuple[tuple[str, ...], str]] = []

        for chunk in chunks:
            tags: set[str] = {"doc_train"}
            if tc.domain_tag:
                tags.add(tc.domain_tag)

            metadata: dict[str, object] = {
                "type": tc.memory_type,
                "source_file": chunk.source_file,
                "heading": chunk.heading,
                "heading_path": "|".join(chunk.heading_path),
                "doc_train": True,
            }

            # Per-chunk error isolation: one failure doesn't abort the batch
            try:
                result = await self._encoder.encode(
                    content=chunk.content,
                    tags=tags,
                    metadata=metadata,
                    skip_conflicts=True,
                    skip_time_neurons=True,
                )
            except Exception:
                logger.warning(
                    "Failed to encode chunk from %s heading=%s",
                    chunk.source_file,
                    chunk.heading,
                    exc_info=True,
                )
                chunks_failed += 1
                continue

            total_neurons += len(result.neurons_created)
            total_synapses += len(result.synapses_created)
            chunks_encoded += 1

            # Record anchor for hierarchy linking
            if chunk.heading_path:
                chunk_anchors.append(
                    (chunk.heading_path, result.fiber.anchor_neuron_id)
                )

        # Build heading hierarchy (with dedup against storage)
        hierarchy_synapses = await self._build_heading_hierarchy(
            chunks=chunks,
            heading_neuron_ids=heading_neuron_ids,
            chunk_anchors=chunk_anchors,
        )

        # Run ENRICH consolidation if requested
        enrichment_synapses = 0
        if tc.consolidate and chunks_encoded > 0:
            enrichment_synapses = await self._run_enrichment()

        return TrainingResult(
            files_processed=files_processed,
            chunks_encoded=chunks_encoded,
            chunks_skipped=max(0, len(chunks) - chunks_encoded - chunks_failed),
            chunks_failed=chunks_failed,
            neurons_created=total_neurons,
            synapses_created=total_synapses,
            hierarchy_synapses=hierarchy_synapses,
            enrichment_synapses=enrichment_synapses,
            brain_name=tc.brain_name or "current",
        )

    async def _build_heading_hierarchy(
        self,
        *,
        chunks: list[DocChunk],
        heading_neuron_ids: dict[tuple[str, ...], str],
        chunk_anchors: list[tuple[tuple[str, ...], str]],
    ) -> int:
        """Create CONCEPT neurons for headings and CONTAINS synapses.

        Deduplicates heading neurons against storage to avoid duplicates
        across separate training runs. Builds a tree:
        root heading → sub heading → chunk anchor.

        Returns the number of hierarchy synapses created.
        """
        synapse_count = 0

        # Collect all unique heading paths from chunks
        all_paths: set[tuple[str, ...]] = set()
        for chunk in chunks:
            for i in range(1, len(chunk.heading_path) + 1):
                all_paths.add(chunk.heading_path[:i])

        # Create or reuse CONCEPT neuron for each unique heading path
        for path in sorted(all_paths, key=len):
            heading_text = path[-1]
            heading_path_str = "|".join(path)

            # Dedup: check storage for existing heading neuron with same path
            existing = await self._storage.find_neurons(
                type=NeuronType.CONCEPT,
                content_exact=heading_text,
                limit=20,
            )
            found = False
            for n in existing:
                if n.metadata.get("heading_path") == heading_path_str and n.metadata.get("doc_heading"):
                    heading_neuron_ids[path] = n.id
                    found = True
                    break

            if not found:
                neuron = Neuron.create(
                    type=NeuronType.CONCEPT,
                    content=heading_text,
                    metadata={
                        "heading_path": heading_path_str,
                        "heading_level": len(path),
                        "doc_heading": True,
                    },
                )
                await self._storage.add_neuron(neuron)
                heading_neuron_ids[path] = neuron.id

        # Create CONTAINS synapses: parent heading → child heading
        for path in sorted(all_paths, key=len):
            if len(path) > 1:
                parent_path = path[:-1]
                parent_id = heading_neuron_ids.get(parent_path)
                child_id = heading_neuron_ids.get(path)
                if parent_id and child_id:
                    synapse = Synapse.create(
                        source_id=parent_id,
                        target_id=child_id,
                        type=SynapseType.CONTAINS,
                        weight=0.9,
                    )
                    await self._storage.add_synapse(synapse)
                    synapse_count += 1

        # Create CONTAINS synapses: leaf heading → chunk anchor
        for heading_path, anchor_id in chunk_anchors:
            heading_id = heading_neuron_ids.get(heading_path)
            if heading_id:
                synapse = Synapse.create(
                    source_id=heading_id,
                    target_id=anchor_id,
                    type=SynapseType.CONTAINS,
                    weight=0.8,
                )
                await self._storage.add_synapse(synapse)
                synapse_count += 1

        return synapse_count

    async def _run_enrichment(self) -> int:
        """Run ENRICH consolidation to create cross-cluster links."""
        from neural_memory.engine.consolidation import (
            ConsolidationEngine,
            ConsolidationStrategy,
        )

        engine = ConsolidationEngine(self._storage)
        report = await engine.run(strategies=[ConsolidationStrategy.ENRICH])
        return report.synapses_enriched
