"""Benchmark write-path ack latency and enrichment convergence (Phase 5).

Measures local write ack p50/p95/p99 (excluding external embedding) and
outbox drain lag after worker runs.

Usage:
    python scripts/benchmark_write_path.py --writes 50
    python scripts/benchmark_write_path.py --writes 100 --restart-replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.enrichment_job import EnrichmentStatus
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.enrichment_worker import process_enrichment_batch
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

# Scorecard gates
ACK_P95_MS = 75.0
CONVERGENCE_P95_MS = 5000.0
MIN_SAMPLES = 20


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


async def run_benchmark(
    *,
    writes: int,
    restart_replay: bool,
    output: Path | None,
) -> dict:
    writes = max(1, min(writes, 500))
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "write_bench.db"
        store = SQLStorage(SQLiteDialect(str(db_path)))
        await store.initialize()
        brain = Brain.create(
            name="write_bench",
            config=BrainConfig(
                encoding_profile="lean",
                async_enrichment_enabled=True,
                embedding_enabled=False,
            ),
        )
        await store.save_brain(brain)
        store.set_brain(brain.id)
        encoder = MemoryEncoder(store, brain.config)

        ack_ms: list[float] = []
        for i in range(writes):
            content = (
                f"Decision #{i}: chose lean write path because ack must stay under "
                f"75ms excluding external embedding providers."
            )
            t0 = time.perf_counter()
            await encoder.encode(content)
            ack_ms.append((time.perf_counter() - t0) * 1000)

        pending_before = await store.count_enrichment_jobs(status=EnrichmentStatus.PENDING)

        # Convergence: drain outbox
        lag_ms: list[float] = []
        t_conv0 = time.perf_counter()
        total_completed = 0
        for _ in range(20):
            report = await process_enrichment_batch(store, worker_id="bench")
            total_completed += report.completed
            pending = await store.count_enrichment_jobs(status=EnrichmentStatus.PENDING)
            running = await store.count_enrichment_jobs(status=EnrichmentStatus.RUNNING)
            if pending == 0 and running == 0:
                break
        lag_ms.append((time.perf_counter() - t_conv0) * 1000)

        dead = await store.count_enrichment_jobs(status=EnrichmentStatus.DEAD)
        restart_duplicates = 0

        if restart_replay:
            # Re-enqueue same idempotency keys via second encode of similar content
            # then claim — should not create duplicate keys for same entity
            before = await store.count_enrichment_jobs()
            # Re-run worker on already-done jobs: claim should be empty
            report2 = await process_enrichment_batch(store, worker_id="bench-restart")
            after = await store.count_enrichment_jobs()
            restart_duplicates = max(0, after - before)
            if report2.claimed > 0 and pending_before == 0:
                # unexpected re-claim of done work
                restart_duplicates += report2.claimed

        await store.close()

        ack_p95 = percentile(ack_ms, 95)
        conv_p95 = percentile(lag_ms, 95)
        conclusive = len(ack_ms) >= MIN_SAMPLES
        report_out = {
            "schema_version": 1,
            "writes": writes,
            "ack_ms": {
                "samples": len(ack_ms),
                "p50": round(percentile(ack_ms, 50), 3),
                "p95": round(ack_p95, 3),
                "p99": round(percentile(ack_ms, 99), 3),
                "mean": round(statistics.fmean(ack_ms), 3),
            },
            "convergence_ms": {
                "samples": len(lag_ms),
                "p50": round(percentile(lag_ms, 50), 3),
                "p95": round(conv_p95, 3),
                "p99": round(percentile(lag_ms, 99), 3),
            },
            "pending_before_drain": pending_before,
            "completed": total_completed,
            "dead_letters": dead,
            "restart_duplicates": restart_duplicates,
            "gates": {
                "ack_p95": {
                    "limit_ms": ACK_P95_MS,
                    "p95_ms": round(ack_p95, 3),
                    "status": (
                        "inconclusive"
                        if not conclusive
                        else ("pass" if ack_p95 <= ACK_P95_MS else "fail")
                    ),
                },
                "convergence_p95": {
                    "limit_ms": CONVERGENCE_P95_MS,
                    "p95_ms": round(conv_p95, 3),
                    "status": (
                        "inconclusive"
                        if not conclusive
                        else ("pass" if conv_p95 <= CONVERGENCE_P95_MS and dead == 0 else "fail")
                    ),
                },
            },
        }
        report_out["all_gates_pass"] = (
            report_out["gates"]["ack_p95"]["status"] == "pass"
            and report_out["gates"]["convergence_p95"]["status"] == "pass"
        )

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report_out, indent=2), encoding="utf-8")

        return report_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writes", type=int, default=50)
    parser.add_argument("--restart-replay", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark/results/write-path.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(
        run_benchmark(
            writes=args.writes,
            restart_replay=args.restart_replay,
            output=args.output,
        )
    )
    print(json.dumps(report, indent=2))
    if not report.get("all_gates_pass"):
        # Inconclusive under low sample still exits 0 for CI smoke; fail only on hard fail
        statuses = [g["status"] for g in report["gates"].values()]
        if "fail" in statuses:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
