"""Ingest LongMemEval sessions into a fresh Neural Memory brain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from scripts.benchmark.data_loader import LMEInstance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of ingesting one LMEInstance."""

    question_id: str
    brain_id: str
    # Maps session_id → fiber_id for all ingested turns
    session_fiber_map: dict[str, list[str]] = field(default_factory=dict)
    total_turns: int = 0
    total_fibers: int = 0
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


async def ingest_instance(
    instance: LMEInstance,
    db_path: Path,
    backend: str = "sqlite",
    max_turns_per_session: int | None = None,
) -> IngestResult:
    """Ingest all sessions of one LMEInstance into a fresh NM brain.

    Each session gets a dedicated set of fibers. We track session_id → list of
    fiber_ids so that the benchmark can map retrieved fibers back to sessions
    for recall@k evaluation.

    Args:
        instance: The LMEInstance to ingest.
        db_path: Path for the per-instance SQLite DB (created fresh).
        backend: "sqlite" or "infinitydb".

    Returns:
        IngestResult with session→fiber mapping and stats.
    """
    import warnings

    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder

    t0 = time.perf_counter()

    # Remove existing DB so each run is fresh
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            # On Windows, the file may still be locked from a previous run.
            # Fall back to using it as-is — data will be overwritten by the
            # new brain anyway since brains are isolated by ID.
            logger.warning(
                "Could not delete existing DB %s (still locked). "
                "Reusing file — old data may remain.",
                db_path,
            )

    # --- Create storage ---
    storage = await _create_storage(backend, db_path)

    # --- Create brain ---
    config = BrainConfig(
        decay_rate=0.05,
        reinforcement_delta=0.03,
        activation_threshold=0.1,
        max_spread_hops=3,
        max_context_tokens=4000,
        embedding_enabled=True,
        embedding_provider="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
        embedding_similarity_threshold=0.5,
    )
    brain = Brain.create(name=f"lme_{instance.question_id}", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # --- Encode turns (full pipeline) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        encoder = MemoryEncoder(storage=storage, config=brain.config)

    session_fiber_map: dict[str, list[str]] = {}
    total_turns = 0

    # Sort sessions by timestamp so temporal ordering is preserved
    sorted_sessions = sorted(instance.sessions, key=lambda s: s.timestamp)

    # Preference detection (only if config enables it)
    pref_enabled = getattr(brain.config, "preference_detection_enabled", True)

    for session in sorted_sessions:
        fiber_ids: list[str] = []

        session_turns = session.turns
        if max_turns_per_session is not None:
            session_turns = session.turns[:max_turns_per_session]

        for turn_index, turn in enumerate(session_turns):
            content = f"[{turn.role}]: {turn.content}"
            turn_timestamp = session.timestamp + timedelta(seconds=turn_index)

            tags: set[str] = {
                f"session:{session.session_id}",
                f"role:{turn.role}",
            }
            metadata: dict[str, object] = {
                "session_id": session.session_id,
                "turn_index": turn_index,
                "_session_date": session.timestamp.isoformat(),
            }

            # Detect preference signals and tag fibers accordingly
            if pref_enabled:
                from neural_memory.engine.preference_detector import (
                    detect_preference_signals,
                )

                signal = detect_preference_signals(turn.content, role=turn.role)
                if signal is not None:
                    tags.add("preference")
                    metadata["_preference_confidence"] = signal.confidence
                    if signal.domain_keywords:
                        metadata["_preference_domain"] = list(signal.domain_keywords)

            try:
                result = await encoder.encode(
                    content=content,
                    timestamp=turn_timestamp,
                    tags=tags,
                    metadata=metadata,
                    skip_conflicts=True,
                )
                fiber_ids.append(result.fiber.id)
            except Exception:
                logger.exception(
                    "Failed to encode turn %d of session %s in instance %s",
                    turn_index,
                    session.session_id,
                    instance.question_id,
                )

            total_turns += 1

        session_fiber_map[session.session_id] = fiber_ids

    # Close storage to flush WAL and release locks before retrieval opens a new connection
    try:
        await storage.close()
    except Exception:
        logger.debug("Storage close after ingest failed (non-critical)", exc_info=True)

    elapsed = time.perf_counter() - t0
    total_fibers = sum(len(fids) for fids in session_fiber_map.values())

    logger.debug(
        "Ingested instance %s: %d turns -> %d fibers in %.2fs",
        instance.question_id,
        total_turns,
        total_fibers,
        elapsed,
    )

    return IngestResult(
        question_id=instance.question_id,
        brain_id=brain.id,
        session_fiber_map=session_fiber_map,
        total_turns=total_turns,
        total_fibers=total_fibers,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Storage factory
# ---------------------------------------------------------------------------


async def _create_storage(backend: str, db_path: Path) -> object:
    """Create and initialize the storage backend."""
    import warnings

    if backend == "infinitydb":
        try:
            from neural_memory.pro.storage_adapter import InfinityDBStorage

            storage = InfinityDBStorage(base_dir=db_path.parent, brain_id=db_path.stem)
            await storage.initialize()
            return storage
        except ImportError:
            logger.warning(
                "InfinityDB not available, falling back to SQLite. "
                "Make sure the Pro package is installed."
            )

    from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
    from neural_memory.storage.sql.sql_storage import SQLStorage

    dialect = SQLiteDialect(str(db_path))
    storage = SQLStorage(dialect)
    await storage.initialize()
    return storage
