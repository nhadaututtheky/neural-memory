"""Driver: relaunch compare_inprocess.py until all instances done.

NM has a native memory leak, so compare_inprocess.py self-terminates at RSS limit.
This driver relaunches it automatically; checkpoint ensures no work is lost.

Usage: python scripts/benchmark/run_inprocess_with_restart.py --variant s
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _count_done(checkpoint: Path) -> int:
    if not checkpoint.exists():
        return 0
    return sum(1 for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip())


def _target_count(variant: str, instance_ids: Path) -> int:
    with instance_ids.open(encoding="utf-8") as f:
        return len(json.load(f)["instance_ids"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["oracle", "s", "m"], default="s")
    parser.add_argument(
        "--instance-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "mini_bench_ids.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--rss-limit-mb", type=int, default=14000)
    parser.add_argument("--max-relaunches", type=int, default=6)
    args = parser.parse_args()

    checkpoint = args.output_dir / "nm_inprocess_checkpoint.jsonl"
    target = _target_count(args.variant, args.instance_ids)
    print(f"Target: {target} instances")

    for attempt in range(1, args.max_relaunches + 1):
        done_before = _count_done(checkpoint)
        if done_before >= target:
            print(f"[driver] All {target} instances complete.")
            return 0

        print(
            f"\n[driver] Attempt {attempt}/{args.max_relaunches}: "
            f"{done_before}/{target} done — launching subprocess"
        )
        t0 = time.perf_counter()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "compare_inprocess.py"),
            "--variant", args.variant,
            "--instance-ids", str(args.instance_ids),
            "--output-dir", str(args.output_dir),
            "--rss-limit-mb", str(args.rss_limit_mb),
        ]
        rc = subprocess.call(cmd)
        elapsed = time.perf_counter() - t0
        done_after = _count_done(checkpoint)
        print(
            f"[driver] Subprocess exit={rc}, elapsed={elapsed:.0f}s, "
            f"added {done_after - done_before} instances "
            f"({done_after}/{target} done)"
        )
        if done_after == done_before:
            print("[driver] No progress — aborting to avoid infinite loop.")
            return 1

    done = _count_done(checkpoint)
    print(f"[driver] Max relaunches reached. {done}/{target} complete.")
    return 0 if done >= target else 1


if __name__ == "__main__":
    sys.exit(main())
