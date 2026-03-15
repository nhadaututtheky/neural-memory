#!/usr/bin/env python3
"""Train a NeuralMemory brain from project documentation.

v2: Adds source tracking (provenance) so the chatbot can cite where answers came from.

Usage:
    python chatbot/train_docs_brain.py                    # Train brain from docs/
    python chatbot/train_docs_brain.py --brain my-docs    # Custom brain name
    python chatbot/train_docs_brain.py --export brain/    # Export after training
    python chatbot/train_docs_brain.py --no-verify        # Skip verification
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from neural_memory import Brain, BrainConfig
from neural_memory.engine.doc_trainer import DocTrainer, TrainingConfig
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── Config ─────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRAIN_NAME = "neuralmemory-docs"
DOCS_DIR = ROOT / "docs"
DB_PATH = ROOT / "chatbot" / "brain" / "docs.db"

# Files to include (relative to ROOT)
EXTRA_FILES = [
    "FAQ.md",
    "CHANGELOG.md",
    "README.md",
]

# Directories to skip
SKIP_DIRS = {"plans", "promo", "agent-instructions"}


# ── Source Registration ────────────────────────────────────


async def _register_source(
    storage: SQLiteStorage, file_path: Path, rel_path: str
) -> str | None:
    """Register a documentation file as a source for provenance tracking.

    Returns source_id if registration succeeds, None otherwise.
    """
    try:
        # Use the storage's source registration if available
        if hasattr(storage, "register_source"):
            source_id = await storage.register_source(
                name=rel_path,
                source_type="documentation",
                metadata={
                    "file_path": str(file_path),
                    "relative_path": rel_path,
                    "format": file_path.suffix.lstrip("."),
                },
            )
            return source_id
    except Exception:
        pass
    return None


# ── Training ───────────────────────────────────────────────


async def train_brain(brain_name: str) -> tuple[SQLiteStorage, BrainConfig]:
    """Create and train a brain from documentation files."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old DB for clean rebuild
    if DB_PATH.exists():
        DB_PATH.unlink()

    storage = SQLiteStorage(str(DB_PATH))
    await storage.initialize()

    config = BrainConfig(
        decay_rate=0.05,
        activation_threshold=0.15,
        max_spread_hops=3,
        max_context_tokens=2000,
    )

    brain = Brain.create(
        name=brain_name,
        config=config,
        metadata={"type": "documentation", "source": "neuralmemory-docs"},
    )
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    trainer = DocTrainer(storage, config)
    training_config = TrainingConfig(
        domain_tag="neuralmemory",
        brain_name=brain_name,
        min_chunk_words=15,
        max_chunk_words=400,
        memory_type="reference",
        consolidate=True,
        initial_stage="episodic",
        salience_ceiling=0.6,
    )

    total_files = 0
    total_chunks = 0
    total_neurons = 0
    total_sources = 0

    # Collect all files to train
    all_files: list[tuple[Path, str]] = []

    # docs/ directory
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        rel = md_file.relative_to(DOCS_DIR)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        all_files.append((md_file, f"docs/{rel}"))

    # Extra root files
    for extra in EXTRA_FILES:
        extra_path = ROOT / extra
        if extra_path.exists():
            all_files.append((extra_path, extra))

    print(f"\n  Training from {len(all_files)} documentation files...")
    print()

    for file_path, rel_path in all_files:
        # Register source for provenance
        source_id = await _register_source(storage, file_path, rel_path)
        if source_id:
            total_sources += 1

        result = await trainer.train_file(file_path, training_config=training_config)
        total_files += 1
        total_chunks += result.chunks_encoded
        total_neurons += result.neurons_created
        print(
            f"    {rel_path}: {result.chunks_encoded} chunks, "
            f"{result.neurons_created} neurons"
            + (f", source={source_id[:8]}" if source_id else "")
        )

    print(f"\n  Total: {total_files} files, {total_chunks} chunks, "
          f"{total_neurons} neurons, {total_sources} sources registered")

    return storage, config


async def verify_brain(storage: SQLiteStorage, config: BrainConfig) -> None:
    """Run test queries to verify the brain works."""
    pipeline = ReflexPipeline(storage, config)

    test_queries = [
        "how to install neural memory",
        "what is spreading activation",
        "list all MCP tools",
        "how to configure embeddings",
        "what is a neuron in neural memory",
        "how does cognitive reasoning work",
        "what is the cloud sync feature",
    ]

    print("\n  Verification queries:")
    for q in test_queries:
        result = await pipeline.query(q, depth=DepthLevel.CONTEXT)
        confidence = f"{result.confidence:.0%}" if result.confidence else "N/A"
        neurons = result.neurons_activated or 0
        latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
        print(f"    Q: {q}")
        print(f"      Confidence: {confidence} | Neurons: {neurons} | Latency: {latency}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train docs brain for chatbot")
    parser.add_argument("--brain", default=DEFAULT_BRAIN_NAME, help="Brain name")
    parser.add_argument("--export", type=str, help="Export brain DB to directory")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification queries"
    )
    args = parser.parse_args()

    print(f"Training brain '{args.brain}' from documentation...")
    storage, config = await train_brain(args.brain)

    if not args.no_verify:
        await verify_brain(storage, config)

    if args.export:
        export_dir = Path(args.export)
        export_dir.mkdir(parents=True, exist_ok=True)
        dest = export_dir / "docs.db"
        shutil.copy2(DB_PATH, dest)
        print(f"\n  Exported to {dest}")

    await storage.close()
    print(f"\n  Brain saved to {DB_PATH}")
    print("  Done!")


if __name__ == "__main__":
    asyncio.run(main())
