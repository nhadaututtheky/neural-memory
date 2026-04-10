"""Stratified mini-benchmark sampler for fast iteration.

Creates a deterministic sample of 50 instances from the full LongMemEval-S
dataset, stratified by question type to preserve category distribution.

Usage:
    # Generate sample IDs
    python scripts/benchmark/mini_bench.py --generate

    # Run mini-benchmark (uses saved IDs)
    python scripts/benchmark/longmemeval.py --variant s --retrieval-only --instance-ids scripts/benchmark/mini_bench_ids.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure imports work when run directly
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

from scripts.benchmark.data_loader import load_dataset

# Stratified sample sizes (total = 50)
# Proportional to full dataset but with minimum representation for small categories
SAMPLE_SIZES: dict[str, int] = {
    "knowledge-update": 7,  # 78 → 7
    "multi-session": 12,  # 133 → 12
    "single-session-assistant": 8,  # 56 → 8
    "single-session-preference": 5,  # 30 → 5 (oversample — weakest category)
    "single-session-user": 8,  # 70 → 8
    "temporal-reasoning": 10,  # 133 → 10
}

TOTAL_SAMPLE = sum(SAMPLE_SIZES.values())  # 50

# Deterministic seed for reproducibility
SEED = 42

OUTPUT_FILE = Path(__file__).resolve().parent / "mini_bench_ids.json"


def generate_sample(data_dir: Path | None = None) -> dict[str, list[str]]:
    """Generate stratified sample IDs from the full S dataset.

    Returns:
        Dict with 'instance_ids' (flat list) and 'by_type' (grouped).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"

    instances = load_dataset("s", data_dir)
    print(f"Loaded {len(instances)} instances from variant=s")

    # Group by question type
    by_type: dict[str, list[str]] = {}
    for inst in instances:
        by_type.setdefault(inst.question_type, []).append(inst.question_id)

    # Print distribution
    print("\nFull dataset distribution:")
    for qtype, ids in sorted(by_type.items()):
        target = SAMPLE_SIZES.get(qtype, 0)
        print(f"  {qtype}: {len(ids)} -> sample {target}")

    # Stratified sampling
    rng = random.Random(SEED)
    sampled: dict[str, list[str]] = {}
    all_ids: list[str] = []

    for qtype, target_n in SAMPLE_SIZES.items():
        pool = by_type.get(qtype, [])
        if not pool:
            print(f"  WARNING: no instances for type {qtype}")
            continue

        n = min(target_n, len(pool))
        selected = rng.sample(pool, n)
        sampled[qtype] = selected
        all_ids.extend(selected)

    print(f"\nSampled {len(all_ids)} instances ({TOTAL_SAMPLE} target)")

    return {
        "seed": SEED,
        "total": len(all_ids),
        "by_type": sampled,
        "instance_ids": all_ids,
    }


def save_sample(sample: dict[str, object], output_path: Path | None = None) -> Path:
    """Save sample IDs to JSON file."""
    path = output_path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)

    print(f"\nSaved to {path}")
    return path


def load_sample(path: Path | None = None) -> list[str]:
    """Load instance IDs from a saved sample file."""
    path = path or OUTPUT_FILE
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["instance_ids"]


if __name__ == "__main__":
    sample = generate_sample()
    save_sample(sample)

    print(f"\nTo run mini-benchmark:")
    print(
        f"  python scripts/benchmark/longmemeval.py "
        f"--variant s --retrieval-only "
        f"--instance-ids {OUTPUT_FILE}"
    )
