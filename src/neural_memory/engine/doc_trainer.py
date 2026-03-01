"""Doc-to-brain training pipeline — train expert brains from documentation.

Processes markdown files into a neural memory brain by:
1. Discovering and chunking documentation files
2. Encoding each chunk through MemoryEncoder (full NLP pipeline)
3. Building heading hierarchy as CONTAINS synapses
4. Optionally running ENRICH consolidation for cross-linking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.doc_chunker import DocChunk, chunk_markdown, discover_files
from neural_memory.engine.doc_extractor import (
    ExtractionError,
    extract_to_markdown,
)
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.utils.timeutils import utcnow

# Extensions that are already markdown and don't need extraction
_TEXT_PASSTHROUGH: frozenset[str] = frozenset({".md", ".mdx", ".txt", ".rst"})

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
        initial_stage: Maturation stage for doc chunks ("episodic" = skip fragile
            STM/WORKING stages since book-knowledge is not real-time memory).
        salience_ceiling: Cap initial fiber salience so doc chunks start weaker
            than organic memories and must earn salience through retrieval.
    """

    domain_tag: str = ""
    brain_name: str = ""
    min_chunk_words: int = 20
    max_chunk_words: int = 500
    memory_type: str = "reference"
    consolidate: bool = True
    extensions: tuple[str, ...] = (".md",)
    initial_stage: str = "episodic"
    salience_ceiling: float = 0.5
    pinned: bool = True


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
        session_synapses: HAPPENED_AT + BEFORE synapses for temporal topology.
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
    session_synapses: int = 0
    enrichment_synapses: int = 0
    brain_name: str = "current"


class DocTrainer:
    """Trains a neural memory brain from documentation files.

    NOT a RAG pipeline. The differences:
    - RAG: chunks are static, ranking is frozen, no lifecycle.
    - NM: chunks start at EPISODIC stage with capped salience (0.5), decay
      without use, mature to SEMANTIC only through spaced retrieval (3+ distinct
      days), and get cross-linked by ENRICH consolidation. A used doc brain
      looks fundamentally different from a freshly-trained one.

    Mirrors CodebaseEncoder's architecture: file hierarchy maps to
    heading hierarchy with CONTAINS synapses, while MemoryEncoder
    handles the actual NLP encoding pipeline.

    Biological model:
    - Reading a textbook → episodic declarative memory (not yet semantic)
    - One temporal context per reading session (session TIME neuron)
    - Local document order preserved (sibling BEFORE synapses)
    - Unretrieved knowledge decays naturally at EPISODIC rate (1.0x)
    - Frequently retrieved chunks earn higher salience via Hebbian learning

    Key optimizations over naive encoding:
    - skip_conflicts=True: Avoids false-positive conflict detection between doc chunks
    - skip_time_neurons=True: Per-chunk TIME neurons skipped (session TIME used instead)
    - initial_stage="episodic": Skip fragile STM/WORKING stages for static knowledge
    - salience_ceiling=0.5: Doc chunks start weaker than organic memories
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

        # Collect all chunks from all files, with hash-based dedup
        all_chunks: list[DocChunk] = []
        files_skipped = 0
        for file_path in files:
            # Check if file already trained (hash dedup)
            if await self._is_file_already_trained(file_path, tc):
                files_skipped += 1
                continue

            text = self._read_or_extract(file_path)
            if text is None:
                continue

            rel_path = str(file_path.relative_to(directory))
            chunks = chunk_markdown(
                text,
                source_file=rel_path,
                min_words=tc.min_chunk_words,
                max_words=tc.max_chunk_words,
            )
            all_chunks.extend(chunks)

            # Record file as trained
            await self._record_trained_file(file_path, len(chunks), tc)

        result = await self._encode_chunks(
            chunks=all_chunks,
            files_processed=len(files) - files_skipped,
            training_config=tc,
        )
        if files_skipped > 0:
            logger.info("Skipped %d already-trained files", files_skipped)
        return result

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

        text = self._read_or_extract(file_path)
        if text is None:
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

    async def _is_file_already_trained(self, file_path: Path, tc: TrainingConfig) -> bool:
        """Check if a file has already been trained via content hash."""
        if not hasattr(self._storage, "get_training_file_by_hash"):
            return False

        try:
            from neural_memory.storage.sqlite_training_files import compute_file_hash

            file_hash = compute_file_hash(file_path)
            record = await self._storage.get_training_file_by_hash(file_hash)
            if record and record["status"] == "completed":
                logger.info("Skipping already-trained file: %s", file_path.name)
                return True
        except (OSError, ValueError) as exc:
            logger.debug("Cannot hash file %s: %s", file_path, exc)
        return False

    async def _record_trained_file(
        self, file_path: Path, chunks_total: int, tc: TrainingConfig
    ) -> None:
        """Record a file as trained for future dedup."""
        if not hasattr(self._storage, "upsert_training_file"):
            return

        try:
            from neural_memory.storage.sqlite_training_files import compute_file_hash

            file_hash = compute_file_hash(file_path)
            await self._storage.upsert_training_file(
                file_hash=file_hash,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                chunks_total=chunks_total,
                chunks_completed=chunks_total,
                status="completed",
                domain_tag=tc.domain_tag,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Cannot record trained file %s: %s", file_path, exc)

    @staticmethod
    def _read_or_extract(file_path: Path) -> str | None:
        """Read a file, extracting to markdown if needed.

        For .md/.mdx/.txt/.rst files, reads directly as text.
        For other formats (PDF, DOCX, etc.), uses doc_extractor.

        Returns:
            Markdown text, or None if the file could not be read.
        """
        suffix = file_path.suffix.lower()
        if suffix in _TEXT_PASSTHROUGH:
            try:
                return file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)
                return None

        # Rich format — extract to markdown
        try:
            return extract_to_markdown(file_path)
        except ExtractionError as exc:
            logger.warning("Extraction failed for %s: %s", file_path, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected extraction error for %s: %s", file_path, exc)
            return None

    async def _encode_chunks(
        self,
        *,
        chunks: list[DocChunk],
        files_processed: int,
        training_config: TrainingConfig,
    ) -> TrainingResult:
        """Encode chunks into neural structures and build heading hierarchy.

        This is the core pipeline:
        1. Create session-level TIME neuron (one per training run, not per chunk)
        2. Encode each chunk via MemoryEncoder.encode() (skip conflicts + time neurons)
        3. Build heading hierarchy as CONCEPT neurons + CONTAINS synapses
        4. Connect top-level headings to session TIME via HAPPENED_AT
        5. Create BEFORE synapses between sibling chunks for document order
        6. Optionally run ENRICH consolidation
        """
        tc = training_config
        total_neurons = 0
        total_synapses = 0
        chunks_encoded = 0
        chunks_failed = 0

        # Create ONE session-level TIME neuron (avoids per-chunk super-hub)
        session_time = utcnow()
        session_time_neuron = Neuron.create(
            type=NeuronType.TIME,
            content=f"doc_train:{session_time.strftime('%Y-%m-%d %H:%M')}",
            metadata={
                "absolute_start": session_time.isoformat(),
                "granularity": "session",
                "doc_train_session": True,
            },
        )
        await self._storage.add_neuron(session_time_neuron)
        total_neurons += 1

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
                    initial_stage=tc.initial_stage,
                    salience_ceiling=tc.salience_ceiling,
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

            # Pin KB fibers so they skip decay/prune/compress
            if tc.pinned:
                from dataclasses import replace as dc_replace

                pinned_fiber = dc_replace(result.fiber, pinned=True)
                await self._storage.update_fiber(pinned_fiber)

            # Record anchor for hierarchy linking
            if chunk.heading_path:
                chunk_anchors.append((chunk.heading_path, result.fiber.anchor_neuron_id))

        # Build heading hierarchy + temporal topology
        hierarchy_synapses = await self._build_heading_hierarchy(
            chunks=chunks,
            heading_neuron_ids=heading_neuron_ids,
            chunk_anchors=chunk_anchors,
        )
        session_synapses = await self._build_temporal_topology(
            session_time_neuron_id=session_time_neuron.id,
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
            session_synapses=session_synapses,
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
                if n.metadata.get("heading_path") == heading_path_str and n.metadata.get(
                    "doc_heading"
                ):
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

        # Collect all CONTAINS synapses, then add in parallel
        synapses_to_add: list[Synapse] = []

        # Parent heading → child heading
        for path in sorted(all_paths, key=len):
            if len(path) > 1:
                parent_path = path[:-1]
                parent_id = heading_neuron_ids.get(parent_path)
                child_id = heading_neuron_ids.get(path)
                if parent_id and child_id:
                    synapses_to_add.append(
                        Synapse.create(
                            source_id=parent_id,
                            target_id=child_id,
                            type=SynapseType.CONTAINS,
                            weight=0.9,
                        )
                    )

        # Leaf heading → chunk anchor
        for heading_path, anchor_id in chunk_anchors:
            heading_id = heading_neuron_ids.get(heading_path)
            if heading_id:
                synapses_to_add.append(
                    Synapse.create(
                        source_id=heading_id,
                        target_id=anchor_id,
                        type=SynapseType.CONTAINS,
                        weight=0.8,
                    )
                )

        if synapses_to_add:
            results = await asyncio.gather(
                *[self._storage.add_synapse(s) for s in synapses_to_add],
                return_exceptions=True,
            )
            for r in results:
                if not isinstance(r, BaseException):
                    synapse_count += 1

        return synapse_count

    async def _build_temporal_topology(
        self,
        *,
        session_time_neuron_id: str,
        heading_neuron_ids: dict[tuple[str, ...], str],
        chunk_anchors: list[tuple[tuple[str, ...], str]],
    ) -> int:
        """Create temporal topology: session TIME + sibling BEFORE synapses.

        Biological model: when you read a textbook, you remember ONE temporal
        context for the reading session, and a vague sense of document order
        within each section. This avoids per-chunk TIME neuron super-hubs
        while preserving VISION.md Pillar 2 (temporal-causal topology).

        Returns the number of temporal synapses created.
        """
        # Weight just above activation_threshold (0.2) to be traversable
        doc_sequence_weight = 0.25
        synapse_count = 0
        synapses_to_add: list[Synapse] = []

        # Connect top-level heading neurons to session TIME neuron
        for path, neuron_id in heading_neuron_ids.items():
            if len(path) == 1:
                synapses_to_add.append(
                    Synapse.create(
                        source_id=neuron_id,
                        target_id=session_time_neuron_id,
                        type=SynapseType.HAPPENED_AT,
                        weight=0.3,
                    )
                )

        # Create BEFORE synapses between sibling chunks under same heading
        # (preserves local document order without runaway activation chains)
        siblings: dict[tuple[str, ...], list[str]] = {}
        for heading_path, anchor_id in chunk_anchors:
            siblings.setdefault(heading_path, []).append(anchor_id)

        for anchor_ids in siblings.values():
            for i in range(len(anchor_ids) - 1):
                synapses_to_add.append(
                    Synapse.create(
                        source_id=anchor_ids[i],
                        target_id=anchor_ids[i + 1],
                        type=SynapseType.BEFORE,
                        weight=doc_sequence_weight,
                        metadata={"doc_sequence": True},
                    )
                )

        if synapses_to_add:
            results = await asyncio.gather(
                *[self._storage.add_synapse(s) for s in synapses_to_add],
                return_exceptions=True,
            )
            for r in results:
                if not isinstance(r, BaseException):
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
