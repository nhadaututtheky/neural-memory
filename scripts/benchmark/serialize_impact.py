#!/usr/bin/env python3
"""Benchmark: SQLite serial vs Postgres parallel write throughput.

Run against both backends to measure impact of dialect-aware strategy.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from neural_memory.storage.factory import open_storage
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.encoder import MemoryEncoder


async def benchmark_serial_writes(storage, num_writes: int = 100) -> dict:
    """Sequential writes (current SQLite behavior)."""
    await storage.initialize()
    
    # Create anchor neuron
    anchor = Neuron.create(type=NeuronType.EPISODIC, content="Benchmark anchor")
    await storage.add_neuron(anchor)
    
    latencies = []
    for i in range(num_writes):
        synapse = Synapse.create(
            source_id=anchor.id,
            target_id=Neuron.create(type=NeuronType.CONCEPT, content=f"Target {i}").id,
            type=SynapseType.INVOLVES,
            weight=0.5,
        )
        start = time.monotonic()
        await storage.add_synapse(synapse)
        latencies.append(time.monotonic() - start)
    
    await storage.close()
    return {
        "mode": "serial",
        "num_writes": num_writes,
        "total_time": sum(latencies),
        "mean_latency": statistics.mean(latencies),
        "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
    }


async def benchmark_parallel_writes(storage, num_writes: int = 100, concurrency: int = 10) -> dict:
    """Parallel writes (Postgres pool behavior)."""
    await storage.initialize()
    
    # Create anchor neuron
    anchor = Neuron.create(type=NeuronType.EPISODIC, content="Benchmark anchor")
    await storage.add_neuron(anchor)
    
    # Pre-create target neurons
    targets = [Neuron.create(type=NeuronType.CONCEPT, content=f"Target {i}") for i in range(num_writes)]
    for t in targets:
        await storage.add_neuron(t)
    
    sem = asyncio.Semaphore(concurrency)
    
    async def write_one(i: int) -> float:
        async with sem:
            synapse = Synapse.create(
                source_id=anchor.id,
                target_id=targets[i].id,
                type=SynapseType.INVOLVES,
                weight=0.5,
            )
            start = time.monotonic()
            await storage.add_synapse(synapse)
            return time.monotonic() - start
    
    latencies = await asyncio.gather(*[write_one(i) for i in range(num_writes)])
    await storage.close()
    
    return {
        "mode": f"parallel_concurrency_{concurrency}",
        "num_writes": num_writes,
        "total_time": sum(latencies),
        "mean_latency": statistics.mean(latencies),
        "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
    }


async def main():
    print("=" * 60)
    print("  Write Throughput Benchmark")
    print("=" * 60)
    
    # SQLite benchmark
    print("\n📊 SQLite (serial)...")
    sqlite_path = ROOT / "benchmark_tmp.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    
    os.environ["NEURAL_MEMORY_BACKEND"] = "sqlite"
    os.environ["NEURAL_MEMORY_SQLITE_PATH"] = str(sqlite_path)
    
    storage = await open_storage()
    result = await benchmark_serial_writes(storage, num_writes=200)
    print(f"  Mode: {result['mode']}")
    print(f"  Writes: {result['num_writes']}")
    print(f"  Total: {result['total_time']:.3f}s")
    print(f"  Mean latency: {result['mean_latency']*1000:.2f}ms")
    print(f"  P99 latency: {result['p99_latency']*1000:.2f}ms")
    print(f"  Throughput: {result['num_writes']/result['total_time']:.1f} writes/sec")
    
    # Cleanup
    if sqlite_path.exists():
        sqlite_path.unlink()
    
    # Postgres benchmark (if available)
    if os.environ.get("POSTGRES_DSN"):
        print("\n📊 Postgres (serial - current)...")
        os.environ["NEURAL_MEMORY_BACKEND"] = "postgres"
        os.environ["NEURAL_MEMORY_POSTGRES_DSN"] = os.environ["POSTGRES_DSN"]
        
        storage = await open_storage()
        result = await benchmark_serial_writes(storage, num_writes=200)
        print(f"  Mode: {result['mode']}")
        print(f"  Throughput: {result['num_writes']/result['total_time']:.1f} writes/sec")
        
        print("\n📊 Postgres (parallel - target)...")
        storage = await open_storage()
        result = await benchmark_parallel_writes(storage, num_writes=200, concurrency=20)
        print(f"  Mode: {result['mode']}")
        print(f"  Throughput: {result['num_writes']/result['total_time']:.1f} writes/sec")
    else:
        print("\n⚠️  POSTGRES_DSN not set — skipping Postgres benchmark")
        print("   Set POSTGRES_DSN=postgresql://user:pass@host/db to run")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())