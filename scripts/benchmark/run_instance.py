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
import sys
import time
import warnings
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


async def run(
    variant: str,
    question_id: str,
    output_path: Path,
    backend: str = "sqlite",
) -> None:
    from scripts.benchmark.data_loader import load_dataset
    from scripts.benchmark.ingest import ingest_instance
    from scripts.benchmark.longmemeval import _retrieve
    from scripts.benchmark.metrics import QuestionResult

    data_dir = Path(__file__).resolve().parent / "data"
    results_dir = Path(__file__).resolve().parent / "results"
    brain_dir = results_dir / "brains"
    brain_dir.mkdir(parents=True, exist_ok=True)

    # Load just this instance
    instances = load_dataset(variant, data_dir)
    instance = next((i for i in instances if i.question_id == question_id), None)
    if instance is None:
        raise ValueError(f"Instance {question_id} not found in variant={variant}")

    t0 = time.perf_counter()
    db_path = brain_dir / f"{question_id}.db"

    # Clean stale files
    for suffix in ("", "-wal", "-shm"):
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # Ingest
    ingest_result = await ingest_instance(instance, db_path, backend)

    # Retrieve
    retrieved_session_ids = await _retrieve(
        instance, ingest_result, db_path, backend, top_k=10
    )

    retrieval_hit = any(
        sid in retrieved_session_ids for sid in instance.answer_session_ids
    )

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
