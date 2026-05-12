"""Run NM retrieval in-process on all mini_bench instances with checkpoint + RSS guard.

Known issue: NM ingest/retrieval has a native memory leak (~500-600MB/instance RSS
growth, mostly torch/sqlite C allocations not visible to tracemalloc). Past 35
instances in-process the Python process stalls/OOMs at ~22GB RAM.

Workaround: checkpoint after EACH instance to JSONL + self-kill when RSS exceeds
`--rss-limit-mb`. User relaunches the script; it auto-resumes from checkpoint.

Output: JSON + JSONL (checkpoint) compatible with compare_baselines.py.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

warnings.simplefilter("ignore")
for n in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
]:
    logging.getLogger(n).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("nm_inprocess")

from scripts.benchmark.data_loader import LMEInstance, load_dataset
from scripts.benchmark.ingest import ingest_instance
from scripts.benchmark.longmemeval import _retrieve
from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_retrieval_metrics,
)


_CHECKPOINT_NAME = "nm_inprocess_checkpoint.jsonl"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / _CHECKPOINT_NAME


def _load_checkpoint(output_dir: Path) -> dict[str, QuestionResult]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return {}
    done: dict[str, QuestionResult] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                r = QuestionResult.from_dict(d)
                done[r.question_id] = r
            except Exception:
                logger.warning("bad checkpoint line: %s", line[:80])
    return done


def _append_checkpoint(output_dir: Path, r: QuestionResult) -> None:
    path = _checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r.to_dict()) + "\n")


async def run_one(inst: LMEInstance, brain_dir: Path) -> QuestionResult:
    """Run full NM ingest+retrieve for one instance, no subprocess."""
    t0 = time.perf_counter()
    db = brain_dir / f"{inst.question_id}.db"
    for s in ("", "-wal", "-shm"):
        p = brain_dir / (db.name + s)
        try:
            if p.exists():
                p.unlink()
        except PermissionError:
            pass

    try:
        ir = await ingest_instance(inst, db, "sqlite")
        retrieved = await _retrieve(inst, ir, db, "sqlite", top_k=10)
        hit = any(sid in retrieved for sid in inst.answer_session_ids)
        elapsed = time.perf_counter() - t0
        return QuestionResult(
            question_id=inst.question_id,
            question_type=inst.question_type,
            hypothesis="",
            correct=None,
            retrieved_session_ids=retrieved,
            answer_session_ids=inst.answer_session_ids,
            retrieval_hit=hit,
            elapsed_sec=elapsed,
        )
    except Exception as e:
        logger.error("Instance %s failed: %s", inst.question_id, e)
        return QuestionResult(
            question_id=inst.question_id,
            question_type=inst.question_type,
            hypothesis=f"FAILED: {e}",
            correct=None,
            retrieved_session_ids=[],
            answer_session_ids=inst.answer_session_ids,
            retrieval_hit=False,
            elapsed_sec=time.perf_counter() - t0,
        )


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["oracle", "s", "m"], default="s")
    parser.add_argument(
        "--instance-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "mini_bench_ids.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument(
        "--rss-limit-mb",
        type=int,
        default=14000,
        help="Self-terminate when RSS exceeds this (default 14GB). Relaunch to resume.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    instances = load_dataset(args.variant, Path(__file__).resolve().parent / "data")

    if args.instance_ids and args.instance_ids.exists():
        with args.instance_ids.open(encoding="utf-8") as f:
            allowed = set(json.load(f)["instance_ids"])
        instances = [i for i in instances if i.question_id in allowed]

    if args.limit:
        instances = instances[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.no_resume:
        cp = _checkpoint_path(args.output_dir)
        if cp.exists():
            cp.unlink()
            print("[resume] Cleared previous checkpoint")
        done: dict[str, QuestionResult] = {}
    else:
        done = _load_checkpoint(args.output_dir)
        if done:
            print(f"[resume] Loaded {len(done)} completed instances from checkpoint")

    pending = [i for i in instances if i.question_id not in done]
    print(f"Running {len(pending)} pending / {len(instances)} total (in-process)")
    print(f"RSS limit: {args.rss_limit_mb} MB — self-kill & relaunch to continue")

    brain_dir = args.output_dir / "brains_inprocess"
    brain_dir.mkdir(parents=True, exist_ok=True)

    results: list[QuestionResult] = list(done.values())
    t_start = time.perf_counter()
    idx_done = len(done)

    for inst in pending:
        r = await run_one(inst, brain_dir)
        results.append(r)
        idx_done += 1
        _append_checkpoint(args.output_dir, r)
        gc.collect()

        hit_rate = sum(1 for x in results if x.retrieval_hit) / len(results)
        rss = _rss_mb()
        print(
            f"[{idx_done}/{len(instances)}] {inst.question_id} ({inst.question_type}) "
            f"-> hit={r.retrieval_hit} retrieved={len(r.retrieved_session_ids)} "
            f"running_R={hit_rate:.3f} ({r.elapsed_sec:.1f}s, RSS={rss:.0f}MB)",
            flush=True,
        )

        if rss > args.rss_limit_mb:
            print(
                f"\n[rss-guard] RSS {rss:.0f}MB > limit {args.rss_limit_mb}MB — "
                f"self-terminating cleanly. Relaunch the same command to resume.",
                flush=True,
            )
            os._exit(0)

    total_elapsed = time.perf_counter() - t_start
    m = compute_retrieval_metrics(results)
    by_type = compute_metrics_by_type(results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"nm_inprocess_{ts}.json"
    summary = {
        "config": {"variant": args.variant, "mode": "in_process"},
        "n_instances": len(results),
        "total_elapsed_sec": total_elapsed,
        "metrics": {
            "recall_at_1": m.recall_at_1,
            "recall_at_3": m.recall_at_3,
            "recall_at_5": m.recall_at_5,
            "recall_at_10": m.recall_at_10,
            "ndcg_at_5": m.ndcg_at_5,
            "ndcg_at_10": m.ndcg_at_10,
        },
        "by_type": by_type,
        "results": [r.to_dict() for r in results],
        "failures": [r.question_id for r in results if r.hypothesis.startswith("FAILED")],
    }
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] Saved: {out}")

    print("\n=== Summary ===")
    print(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed / len(results):.1f}s/inst)")
    print(
        f"R@1={m.recall_at_1:.3f} R@3={m.recall_at_3:.3f} R@5={m.recall_at_5:.3f} R@10={m.recall_at_10:.3f}"
    )
    print(f"NDCG@5={m.ndcg_at_5:.3f} NDCG@10={m.ndcg_at_10:.3f}")
    print(f"Failures: {summary['failures']}")


if __name__ == "__main__":
    asyncio.run(main())
