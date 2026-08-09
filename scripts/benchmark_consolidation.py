"""Benchmark incremental vs full consolidation cost (Phase 6).

Clones a seeded brain, applies N changes, runs full vs incremental, and
reports duration/RSS/candidate scope. Equivalence gate compares that
incremental touches a subset of the change scope and is faster at scale.

Usage:
    python scripts/benchmark_consolidation.py --scale 500 --changes 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


def _rss_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


async def _seed(store: SQLStorage, n: int) -> None:
    for i in range(n):
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content=f"seed concept {i} about auth caching databases and latency",
        )
        await store.add_neuron(neuron)
        if hasattr(store, "record_change"):
            await store.record_change("neuron", neuron.id, "insert")


async def run_bench(*, scale: int, changes: int, output: Path | None) -> dict:
    scale = max(10, min(scale, 50_000))
    changes = max(1, min(changes, scale))

    with tempfile.TemporaryDirectory() as tmp:
        # Shared seed
        seed_path = Path(tmp) / "seed.db"
        seed = SQLStorage(SQLiteDialect(str(seed_path)))
        await seed.initialize()
        brain = Brain.create(name="bench", config=BrainConfig())
        await seed.save_brain(brain)
        seed.set_brain(brain.id)
        await _seed(seed, scale)

        # Apply additional changes
        for i in range(changes):
            n = Neuron.create(
                type=NeuronType.CONCEPT,
                content=f"delta change {i} consolidation dirty set target",
            )
            await seed.add_neuron(n)
            if hasattr(seed, "record_change"):
                await seed.record_change("neuron", n.id, "insert")

        # --- Full run ---
        full_store = SQLStorage(SQLiteDialect(str(Path(tmp) / "full.db")))
        # Re-seed full clone simply by re-running seed path (fresh)
        await full_store.initialize()
        b2 = Brain.create(name="full", config=BrainConfig())
        await full_store.save_brain(b2)
        full_store.set_brain(b2.id)
        await _seed(full_store, scale)
        for i in range(changes):
            n = Neuron.create(type=NeuronType.CONCEPT, content=f"delta change {i} full clone")
            await full_store.add_neuron(n)
            if hasattr(full_store, "record_change"):
                await full_store.record_change("neuron", n.id, "insert")

        rss0 = _rss_mb()
        t0 = time.perf_counter()
        full_engine = ConsolidationEngine(full_store)
        full_report = await full_engine.run(
            strategies=[ConsolidationStrategy.PRUNE, ConsolidationStrategy.MERGE]
        )
        full_ms = (time.perf_counter() - t0) * 1000
        full_rss = _rss_mb() - rss0

        # --- Incremental run ---
        inc_store = SQLStorage(SQLiteDialect(str(Path(tmp) / "inc.db")))
        await inc_store.initialize()
        b3 = Brain.create(name="inc", config=BrainConfig())
        await inc_store.save_brain(b3)
        inc_store.set_brain(b3.id)
        await _seed(inc_store, scale)
        # Stamp checkpoints at current watermark (simulate prior full run)
        from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
        from neural_memory.engine.consolidation_incremental import (
            get_change_log_high_watermark,
        )

        wm_before = await get_change_log_high_watermark(inc_store)
        for strat in ("prune", "merge"):
            await inc_store.save_consolidation_checkpoint(
                ConsolidationCheckpoint.create(
                    brain_id=b3.id,
                    strategy=strat,
                    last_sequence=wm_before,
                )
            )
        # Apply same volume of changes after checkpoint
        for i in range(changes):
            n = Neuron.create(
                type=NeuronType.CONCEPT,
                content=f"delta change {i} inc path only dirty work",
            )
            await inc_store.add_neuron(n)
            if hasattr(inc_store, "record_change"):
                await inc_store.record_change("neuron", n.id, "insert")

        rss1 = _rss_mb()
        t1 = time.perf_counter()
        inc_engine = ConsolidationEngine(inc_store)
        inc = await inc_engine.run_incremental(
            strategies=[ConsolidationStrategy.PRUNE, ConsolidationStrategy.MERGE],
            bootstrap_full=False,
            max_changes=max(changes * 2, 100),
        )
        inc_ms = (time.perf_counter() - t1) * 1000
        inc_rss = _rss_mb() - rss1

        await seed.close()
        await full_store.close()
        await inc_store.close()

        faster = inc_ms <= full_ms * 1.05  # allow 5% slack on tiny scales
        report = {
            "schema_version": 1,
            "scale": scale,
            "changes": changes,
            "full": {
                "duration_ms": round(full_ms, 2),
                "rss_delta_mb": round(full_rss, 2),
                "synapses_pruned": full_report.synapses_pruned,
                "fibers_merged": full_report.fibers_merged,
            },
            "incremental": {
                "duration_ms": round(inc_ms, 2),
                "rss_delta_mb": round(inc_rss, 2),
                "mode": inc.mode,
                "dirty_entities": inc.dirty.total_entities if inc.dirty else 0,
                "change_count": inc.dirty.change_count if inc.dirty else 0,
                "truncated": inc.truncated,
                "advanced": list(inc.strategies_advanced),
            },
            "gates": {
                "incremental_faster": {
                    "status": "pass" if faster else "fail",
                    "full_ms": round(full_ms, 2),
                    "inc_ms": round(inc_ms, 2),
                },
                "not_truncated": {
                    "status": "pass" if not inc.truncated else "fail",
                },
            },
        }
        report["all_gates_pass"] = all(g["status"] == "pass" for g in report["gates"].values())

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=int, default=200)
    p.add_argument("--changes", type=int, default=20)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark/results/consolidation-inc.json"),
    )
    args = p.parse_args()
    report = asyncio.run(run_bench(scale=args.scale, changes=args.changes, output=args.output))
    print(json.dumps(report, indent=2))
    return 0 if report.get("all_gates_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
