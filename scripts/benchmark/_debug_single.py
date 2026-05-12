"""One-shot debug: retrieve on a known-failing instance and print what comes out."""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

warnings.simplefilter("ignore")
for n in ["neural_memory.engine", "neural_memory.storage", "neural_memory.safety"]:
    logging.getLogger(n).setLevel(logging.ERROR)


async def main() -> None:
    from scripts.benchmark.data_loader import load_dataset
    from scripts.benchmark.ingest import ingest_instance
    from scripts.benchmark.longmemeval import _open_storage

    instances = load_dataset("s", Path(__file__).resolve().parent / "data")
    inst = next(i for i in instances if i.question_id == "4d6b87c8")

    import time as _t

    db = Path(__file__).resolve().parent / f"results/brains/_debug_{int(_t.time())}.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    print(f"Question: {inst.question}")
    print(f"Answer sessions: {inst.answer_session_ids}")
    print("Ingesting...", flush=True)
    ir = await ingest_instance(inst, db, "sqlite")
    print(f"  fibers: {ir.total_fibers} across {len(ir.session_fiber_map)} sessions")

    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.retrieval import ReflexPipeline

    storage = await _open_storage("sqlite", db)
    cfg = BrainConfig(
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
    storage.set_brain(ir.brain_id)
    pipe = ReflexPipeline(storage=storage, config=cfg)
    r = await pipe.query(inst.question)

    print(f"\nfibers_matched: {len(r.fibers_matched)}")
    print(f"contributing_neurons: {len(r.contributing_neurons)}")

    ingest_fibers = {f for fs in ir.session_fiber_map.values() for f in fs}
    print(f"ingest_fiber_count: {len(ingest_fibers)}\n")

    for fid in r.fibers_matched:
        fiber = await storage.get_fiber(fid)
        if fiber:
            in_ingest = fid in ingest_fibers
            sid = (fiber.metadata or {}).get("session_id", "<none>") if fiber.metadata else "<none>"
            summary = (fiber.summary or "")[:100].replace("\n", " ")
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            anchor_content = (anchor.content[:100].replace("\n", " ")) if anchor else "<missing>"
            print(
                f"  fiber {fid[:8]}... in_ingest={in_ingest} sid={sid} neurons={len(fiber.neuron_ids)}"
            )
            print(f"    summary: {summary!r}")
            print(f"    anchor:  {anchor_content!r}")

    print(f"\nAnswer session turns:")
    for aid in inst.answer_session_ids:
        sess = next((s for s in inst.sessions if s.session_id == aid), None)
        if sess:
            for t in sess.turns[:3]:
                print(f"  [{aid}] {t.role}: {t.content[:100]}")

    await storage.close()


asyncio.run(main())
