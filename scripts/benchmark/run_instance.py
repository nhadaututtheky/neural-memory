"""Run a single LongMemEval instance in an isolated process.

Usage:
    python scripts/benchmark/run_instance.py <variant> <question_id> <output_json> [--backend sqlite]

Writes a QuestionResult JSON to <output_json> on success.
Designed to be called from longmemeval.py to avoid memory leaks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path

# Ensure imports work
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

# Suppress noisy loggers
for _name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
    "neural_memory.core",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


def _safe_database_path(work_dir: Path, question_id: str) -> Path:
    """Derive an opaque database path contained by one run-specific root."""
    import hashlib

    root = work_dir.resolve()
    filename = f"{hashlib.sha256(question_id.encode('utf-8')).hexdigest()}.db"
    db_path = (root / filename).resolve()
    if not db_path.is_relative_to(root):
        raise ValueError("benchmark database path escaped the run directory")
    return db_path


async def _verify_pinned_embedding_revision() -> None:
    """Fail the benchmark child before ingest if its pinned model cannot load."""
    if not os.environ.get("NMEM_SENTENCE_TRANSFORMER_REVISION"):
        return

    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.semantic_discovery import _create_provider

    provider = _create_provider(
        BrainConfig(
            embedding_enabled=True,
            embedding_provider="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
        )
    )
    await provider.embed("nmem benchmark revision probe")


async def _verify_vector_index(db_path: Path, backend: str, brain_id: str) -> None:
    """Prove the benchmark wrote searchable vectors before recall starts."""
    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.semantic_discovery import _create_provider
    from scripts.benchmark.longmemeval import _open_storage

    provider = _create_provider(
        BrainConfig(
            embedding_enabled=True,
            embedding_provider="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
        )
    )
    probe_vector = await provider.embed("nmem benchmark vector index probe")
    storage = await _open_storage(backend, db_path)
    try:
        storage.set_brain(brain_id)
        if not hasattr(storage, "knn_search"):
            raise RuntimeError("benchmark backend has no vector search capability")
        probe_results = await storage.knn_search(probe_vector, k=1)
        if not probe_results:
            raise RuntimeError("benchmark ingest produced no searchable vectors")
    finally:
        await storage.close()


async def run(
    variant: str,
    question_id: str,
    output_path: Path,
    backend: str = "sqlite",
) -> None:
    seed = int(os.environ.get("NMEM_BENCHMARK_SEED", "42"))
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    from scripts.benchmark.data_loader import load_dataset
    from scripts.benchmark.ingest import ingest_instance
    from scripts.benchmark.longmemeval import _retrieve
    from scripts.benchmark.metrics import QuestionResult

    data_dir = Path(__file__).resolve().parent / "data"
    # Load just this instance
    instances = load_dataset(variant, data_dir)
    instance = next((i for i in instances if i.question_id == question_id), None)
    if instance is None:
        raise ValueError(f"Instance {question_id} not found in variant={variant}")

    await _verify_pinned_embedding_revision()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    configured_work_dir = os.environ.get("NMEM_BENCHMARK_WORK_DIR")
    temporary_work_dir: tempfile.TemporaryDirectory[str] | None = None
    if configured_work_dir is None:
        temporary_work_dir = tempfile.TemporaryDirectory(
            prefix="nmem-instance-",
            dir=output_path.parent,
        )
        work_dir = Path(temporary_work_dir.name)
    else:
        work_dir = Path(configured_work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.perf_counter()
        db_path = _safe_database_path(work_dir, question_id)

        # Ingest and retrieve against one run-local database.
        ingest_result = await ingest_instance(instance, db_path, backend)
        if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
            await _verify_vector_index(db_path, backend, ingest_result.brain_id)
        retrieved_session_ids = await _retrieve(
            instance,
            ingest_result,
            db_path,
            backend,
            top_k=10,
        )
    finally:
        if temporary_work_dir is not None:
            temporary_work_dir.cleanup()

    retrieval_hit = any(sid in retrieved_session_ids for sid in instance.answer_session_ids)

    elapsed = time.perf_counter() - t0

    result = QuestionResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        hypothesis="",
        correct=None,
        retrieved_session_ids=retrieved_session_ids,
        answer_session_ids=instance.answer_session_ids,
        retrieval_hit=retrieval_hit,
        elapsed_sec=elapsed,
    )

    output_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("question_id")
    parser.add_argument("output_json")
    parser.add_argument("--backend", default="sqlite")
    args = parser.parse_args()

    asyncio.run(run(args.variant, args.question_id, Path(args.output_json), args.backend))
