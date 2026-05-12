"""Trace memory growth across N LongMemEval instances to find leak source.

Runs ingest+retrieve for N instances, takes tracemalloc snapshot before and
after each, prints top allocations by file that GREW between snapshots.

The leak should be identifiable as a file/line that adds a fixed number of MB
per instance. Focus on non-GC'd allocations — things that persist across
fresh Brain/Storage/Encoder instances.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
import tracemalloc
import warnings
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

warnings.simplefilter("ignore")
for n in ["neural_memory.engine", "neural_memory.storage", "neural_memory.safety", "neural_memory.extraction"]:
    logging.getLogger(n).setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING)


async def run_once(inst, brain_dir: Path) -> None:
    from scripts.benchmark.ingest import ingest_instance
    from scripts.benchmark.longmemeval import _retrieve

    db = brain_dir / f"_mem_{inst.question_id}.db"
    for s in ("", "-wal", "-shm"):
        p = brain_dir / (db.name + s)
        if p.exists():
            try:
                p.unlink()
            except PermissionError:
                pass
    ir = await ingest_instance(inst, db, "sqlite")
    await _retrieve(inst, ir, db, "sqlite", top_k=10)


def snapshot_diff(old, new, limit: int = 15) -> None:
    stats = new.compare_to(old, "filename")
    print(f"{'File':<65}  {'Size MB':>10}  {'Count':>10}")
    print("-" * 90)
    for s in stats[:limit]:
        frame = s.traceback[0]
        fn = frame.filename
        # Shorten path
        if "\\neural_memory\\" in fn:
            fn = fn.split("\\neural_memory\\", 1)[1]
        elif "\\site-packages\\" in fn:
            fn = "(pkg) " + fn.split("\\site-packages\\", 1)[1]
        else:
            fn = fn[-60:]
        size_mb = s.size_diff / (1024 * 1024)
        print(f"{fn:<65}  {size_mb:>+10.2f}  {s.count_diff:>+10}")


async def main() -> None:
    import psutil
    proc = psutil.Process()

    from scripts.benchmark.data_loader import load_dataset

    N = 3  # Small for speed — leak should be obvious after 2-3 iterations
    instances = load_dataset("s", Path(__file__).resolve().parent / "data")
    import json as _json
    with (Path(__file__).resolve().parent / "mini_bench_ids.json").open(encoding="utf-8") as f:
        allowed = set(_json.load(f)["instance_ids"])
    instances = [i for i in instances if i.question_id in allowed][:N]
    print(f"Profiling {len(instances)} instances\n")

    brain_dir = Path(__file__).resolve().parent / "results" / "brains_mem"
    brain_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start(25)
    gc.collect()
    snap_init = tracemalloc.take_snapshot()
    rss_init = proc.memory_info().rss / 1024 / 1024

    prev_snap = snap_init
    prev_rss = rss_init

    for i, inst in enumerate(instances):
        await run_once(inst, brain_dir)
        gc.collect()
        rss = proc.memory_info().rss / 1024 / 1024
        snap = tracemalloc.take_snapshot()
        delta_mb = rss - prev_rss
        print(f"\n=== Instance {i+1}/{N} ({inst.question_id}) ===")
        print(f"RSS now: {rss:.0f} MB (delta {delta_mb:+.0f} MB from prev, {rss - rss_init:+.0f} MB total)")
        print(f"\nTop file growers (vs prev instance):")
        snapshot_diff(prev_snap, snap)
        prev_snap = snap
        prev_rss = rss

    print(f"\n\n=== Overall growers (init → final) ===")
    snap_final = tracemalloc.take_snapshot()
    snapshot_diff(snap_init, snap_final, limit=25)


if __name__ == "__main__":
    asyncio.run(main())
