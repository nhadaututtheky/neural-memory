"""LongMemEval benchmark — main orchestrator.

Usage:
    python scripts/benchmark/longmemeval.py --variant oracle --limit 50
    python scripts/benchmark/longmemeval.py --variant s --reader claude --retrieval-only
    python scripts/benchmark/longmemeval.py --variant oracle --resume
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import time
import warnings
from pathlib import Path

# Ensure the NM source tree and repo root are importable when run directly
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

# Suppress noisy NM loggers
for _logger_name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
    "neural_memory.core",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("longmemeval")

from scripts.benchmark.config import BenchmarkConfig, parse_args
from scripts.benchmark.data_loader import LMEInstance, load_dataset
from scripts.benchmark.ingest import IngestResult, ingest_instance
from scripts.benchmark.judge import BaseJudge, create_judge
from scripts.benchmark.metrics import QuestionResult
from scripts.benchmark.reader import BaseReader, create_reader
from scripts.benchmark.report import print_report, save_report

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_FILE = "checkpoint.jsonl"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / _CHECKPOINT_FILE


def load_checkpoint(output_dir: Path) -> dict[str, QuestionResult]:
    """Load previously completed results from JSONL checkpoint file."""
    cp_path = _checkpoint_path(output_dir)
    if not cp_path.exists():
        return {}

    completed: dict[str, QuestionResult] = {}
    with cp_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                result = QuestionResult.from_dict(d)
                completed[result.question_id] = result
            except Exception:
                logger.warning("Could not parse checkpoint line: %s", line[:80])

    logger.info("Loaded %d completed results from checkpoint", len(completed))
    return completed


def save_checkpoint(result: QuestionResult, output_dir: Path) -> None:
    """Append one QuestionResult as a JSONL line to the checkpoint file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_path = _checkpoint_path(output_dir)
    with cp_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


async def _retrieve(
    instance: LMEInstance,
    ingest_result: IngestResult,
    db_path: Path,
    backend: str,
    top_k: int = 10,
) -> list[str]:
    """Run ReflexPipeline query and return top-k session IDs.

    Maps retrieved fiber_ids back to session_ids using ingest_result.session_fiber_map.
    Returns a deduplicated, ordered list of session_ids (most relevant first).
    """
    # Build reverse map: fiber_id → session_id
    fiber_to_session: dict[str, str] = {}
    for session_id, fiber_ids in ingest_result.session_fiber_map.items():
        for fid in fiber_ids:
            fiber_to_session[fid] = session_id

    storage = await _open_storage(backend, db_path)

    try:
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.retrieval import ReflexPipeline

        # Re-create a minimal config (same as ingest)
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

        # Set the brain context
        storage.set_brain(ingest_result.brain_id)

        pipeline = ReflexPipeline(storage=storage, config=config)
        retrieval = await pipeline.query(instance.question)

        # Build a metadata-based fallback map for consolidated fibers
        # (encode() creates cluster/consolidated fibers beyond the primary,
        # and those inherit session_id in metadata but aren't in session_fiber_map)
        _metadata_session_cache: dict[str, str | None] = {}

        async def _resolve_session(fid: str) -> str | None:
            """Resolve fiber_id → session_id via ingest map or fiber metadata."""
            sid = fiber_to_session.get(fid)
            if sid:
                return sid
            if fid in _metadata_session_cache:
                return _metadata_session_cache[fid]
            try:
                fiber = await storage.get_fiber(fid)
                if fiber and fiber.metadata:
                    sid = fiber.metadata.get("session_id")
                _metadata_session_cache[fid] = sid
            except Exception:
                _metadata_session_cache[fid] = None
            return _metadata_session_cache.get(fid)

        # Collect session IDs in the order fibers appear (preserving ranking)
        seen: set[str] = set()
        ordered_session_ids: list[str] = []
        for fiber_id in retrieval.fibers_matched:
            session_id = await _resolve_session(fiber_id)
            if session_id and session_id not in seen:
                seen.add(session_id)
                ordered_session_ids.append(session_id)
            if len(ordered_session_ids) >= top_k:
                break

        # If we haven't filled top_k yet, add sessions from contributing neurons
        if len(ordered_session_ids) < top_k and retrieval.contributing_neurons:
            try:
                for neuron_id in retrieval.contributing_neurons:
                    if len(ordered_session_ids) >= top_k:
                        break
                    fibers = await storage.find_fibers_by_neuron(neuron_id)
                    for fiber in fibers:
                        session_id = await _resolve_session(fiber.id)
                        if session_id and session_id not in seen:
                            seen.add(session_id)
                            ordered_session_ids.append(session_id)
                            if len(ordered_session_ids) >= top_k:
                                break
            except Exception:
                pass  # Not all storage backends implement find_fibers_by_neuron

        return ordered_session_ids

    finally:
        try:
            await storage.close()
        except Exception:
            pass


async def _open_storage(backend: str, db_path: Path) -> object:
    """Open (read-only re-use) storage for retrieval."""
    if backend == "infinitydb":
        try:
            from neural_memory.pro.storage_adapter import InfinityDBStorage

            storage = InfinityDBStorage(base_dir=db_path.parent, brain_id=db_path.stem)
            await storage.initialize()
            return storage
        except ImportError:
            pass

    from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
    from neural_memory.storage.sql.sql_storage import SQLStorage

    dialect = SQLiteDialect(str(db_path))
    storage = SQLStorage(dialect)
    await storage.initialize()
    return storage


# ---------------------------------------------------------------------------
# Per-instance pipeline
# ---------------------------------------------------------------------------


async def _process_instance(
    instance: LMEInstance,
    config: BenchmarkConfig,
    reader: BaseReader | None,
    judge: BaseJudge | None,
) -> QuestionResult:
    """Run the full ingest → retrieve → read → judge pipeline for one instance."""
    t0 = time.perf_counter()

    brain_dir = config.output_dir / "brains"
    db_path = brain_dir / f"{instance.question_id}.db"

    # Clean up stale WAL/SHM files from previous crashed runs
    for suffix in ("-wal", "-shm"):
        stale = db_path.parent / (db_path.name + suffix)
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass

    # Phase 1: Ingest
    ingest_result = await ingest_instance(instance, db_path, config.backend)

    # Phase 2: Retrieve
    retrieved_session_ids = await _retrieve(
        instance,
        ingest_result,
        db_path,
        config.backend,
        top_k=10,
    )

    retrieval_hit = any(sid in retrieved_session_ids for sid in instance.answer_session_ids)

    # Phase 3: Read (build context from retrieved session content)
    hypothesis = ""
    if reader is not None:
        context = _build_context(instance, retrieved_session_ids, max_turns=30)
        hypothesis = await reader.answer(instance.question, context, instance.question_date)

    # Phase 4: Judge
    correct: bool | None = None
    if judge is not None and hypothesis:
        correct = await judge.evaluate(instance.question, hypothesis, instance.answer)

    elapsed = time.perf_counter() - t0

    return QuestionResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        hypothesis=hypothesis,
        correct=correct,
        retrieved_session_ids=retrieved_session_ids,
        answer_session_ids=instance.answer_session_ids,
        retrieval_hit=retrieval_hit,
        elapsed_sec=elapsed,
    )


def _build_context(
    instance: LMEInstance,
    retrieved_session_ids: list[str],
    max_turns: int = 30,
) -> str:
    """Build plain-text context from retrieved sessions for the reader."""
    session_map = {s.session_id: s for s in instance.sessions}

    lines: list[str] = []
    turn_count = 0

    for session_id in retrieved_session_ids:
        session = session_map.get(session_id)
        if session is None:
            continue

        lines.append(f"--- Session {session_id} ({session.timestamp.strftime('%Y-%m-%d %H:%M')}) ---")

        for turn in session.turns:
            if turn_count >= max_turns:
                break
            lines.append(f"{turn.role.capitalize()}: {turn.content}")
            turn_count += 1

        if turn_count >= max_turns:
            break

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    config = parse_args()

    # Ensure output dirs exist
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    instances = load_dataset(config.variant, config.data_dir)
    print(f"Loaded {len(instances)} instances (variant={config.variant})")

    # Filter by instance IDs (from mini_bench.py)
    if config.instance_ids_file is not None:
        import json as _json

        with config.instance_ids_file.open(encoding="utf-8") as _f:
            _id_data = _json.load(_f)
        _allowed_ids = set(_id_data["instance_ids"])
        instances = [inst for inst in instances if inst.question_id in _allowed_ids]
        print(f"Filtered to {len(instances)} instances from {config.instance_ids_file.name}")

    # Apply limit
    if config.limit is not None:
        instances = instances[: config.limit]
        print(f"Limiting to {len(instances)} instances")

    # Always load checkpoint to avoid re-processing completed instances.
    # Use --no-resume to force re-processing (deletes existing checkpoint).
    completed: dict[str, QuestionResult] = {}
    if config.resume:
        completed = load_checkpoint(config.output_dir)
        skip_count = sum(1 for inst in instances if inst.question_id in completed)
        print(f"Resuming: {skip_count} instances already done, {len(instances) - skip_count} remaining")
    else:
        # Fresh run — clear old checkpoint
        cp = _checkpoint_path(config.output_dir)
        if cp.exists():
            cp.unlink()
            print("Cleared previous checkpoint")

    # Create reader and judge
    reader: BaseReader | None = None
    judge: BaseJudge | None = None

    if not config.retrieval_only:
        reader = create_reader(config)
        judge = create_judge(config)

    results: list[QuestionResult] = []
    total = len(instances)

    # Subprocess-based execution to prevent memory leaks (each instance
    # runs in a fresh process, OS reclaims all memory on exit).
    _run_instance_script = Path(__file__).resolve().parent / "run_instance.py"
    _tmp_dir = config.output_dir / "_tmp"
    _tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(instances):
        # Use checkpoint result if available
        if inst.question_id in completed:
            results.append(completed[inst.question_id])
            continue

        print(
            f"[{i + 1}/{total}] {inst.question_id} ({inst.question_type})",
            flush=True,
        )

        result_file = _tmp_dir / f"{inst.question_id}.json"
        if result_file.exists():
            result_file.unlink()

        if config.retrieval_only:
            # Subprocess isolation: avoids memory leak from encoder.
            # Use Popen + memory monitoring to kill before OOM thrashing.
            import subprocess

            _MEM_LIMIT_MB = 10_000  # Kill subprocess if RSS > 10GB
            _TIMEOUT_SEC = 600     # 10 min hard timeout (embedding adds overhead)

            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(_run_instance_script),
                    config.variant,
                    inst.question_id,
                    str(result_file),
                    "--backend",
                    config.backend,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                import psutil

                ps_proc = psutil.Process(proc.pid)
                _t_start = time.perf_counter()
                while proc.poll() is None:
                    elapsed_wait = time.perf_counter() - _t_start
                    if elapsed_wait > _TIMEOUT_SEC:
                        raise subprocess.TimeoutExpired(cmd="run_instance", timeout=_TIMEOUT_SEC)
                    try:
                        mem_mb = ps_proc.memory_info().rss / (1024 * 1024)
                        if mem_mb > _MEM_LIMIT_MB:
                            logger.warning(
                                "Killing subprocess for %s — RSS %.0fMB > %dMB limit",
                                inst.question_id, mem_mb, _MEM_LIMIT_MB,
                            )
                            proc.kill()
                            proc.wait(timeout=10)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    time.sleep(2)  # Check every 2 seconds
            except ImportError:
                # No psutil — fall back to simple timeout
                try:
                    proc.wait(timeout=_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

            if proc.returncode == 0 and result_file.exists():
                d = json.loads(result_file.read_text(encoding="utf-8"))
                result = QuestionResult.from_dict(d)
            else:
                stderr_tail = (proc.stderr.read() if proc.stderr else "")[-500:]
                logger.error(
                    "Subprocess failed for %s (rc=%s): %s",
                    inst.question_id,
                    proc.returncode,
                    stderr_tail,
                )
                result = QuestionResult(
                    question_id=inst.question_id,
                    question_type=inst.question_type,
                    hypothesis="",
                    correct=None,
                    retrieved_session_ids=[],
                    answer_session_ids=inst.answer_session_ids,
                    retrieval_hit=False,
                    elapsed_sec=0.0,
                )
        else:
            # In-process for reader/judge mode (needs LLM client)
            try:
                result = await _process_instance(inst, config, reader, judge)
            except Exception:
                logger.exception("Failed to process instance %s", inst.question_id)
                result = QuestionResult(
                    question_id=inst.question_id,
                    question_type=inst.question_type,
                    hypothesis="",
                    correct=None,
                    retrieved_session_ids=[],
                    answer_session_ids=inst.answer_session_ids,
                    retrieval_hit=False,
                    elapsed_sec=0.0,
                )

        results.append(result)
        save_checkpoint(result, config.output_dir)

        # Live status line
        hit_rate = sum(1 for r in results if r.retrieval_hit) / len(results)
        scored = [r for r in results if r.correct is not None]
        if scored:
            acc = sum(1 for r in scored if r.correct) / len(scored)
            print(
                f"  -> hit={result.retrieval_hit} correct={result.correct} "
                f"| running R@hit={hit_rate:.3f} acc={acc:.3f} "
                f"({result.elapsed_sec:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  -> hit={result.retrieval_hit} "
                f"| running R@hit={hit_rate:.3f} ({result.elapsed_sec:.1f}s)",
                flush=True,
            )

    # Final report
    print_report(results, config)
    save_report(results, config, config.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
