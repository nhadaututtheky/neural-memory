"""Benchmark hook subprocess spawn latency on the host platform.

Measures cold spawn cost for 4 hook invocation modes:
  1. .exe entry point (e.g. nmem-hook-post-tool-use.exe on Windows)
  2. python -m neural_memory.hooks.post_tool_use
  3. python -c "pass"          (Python interpreter cold start baseline)
  4. cmd /c rem                 (OS process spawn baseline, Windows)

Reports p50/p95/p99 and warm vs cold split.

Usage:
    python scripts/benchmark/hook_spawn_latency.py [--runs 30]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


SAMPLE_INPUT = json.dumps(
    {
        "tool_name": "Read",
        "tool_input": {"file_path": "C:/foo/bar.py"},
        "tool_error": None,
        "duration_ms": 12,
    }
).encode("utf-8")


def time_spawn(args: list[str], stdin: bytes | None = None) -> float:
    """Spawn process once, return wall-clock ms."""
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        timeout=10,
        shell=False,
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        sys.stderr.write(f"[warn] {args[0]} exit={proc.returncode} stderr={proc.stderr[:200]!r}\n")
    return elapsed


def measure(label: str, args: list[str], stdin: bytes | None, runs: int) -> dict:
    """Run N spawns, return stats (cold = first run, warm = rest)."""
    print(f"\n>>> {label}")
    samples: list[float] = []
    for i in range(runs):
        ms = time_spawn(args, stdin)
        samples.append(ms)
        if i < 3 or i == runs - 1:
            print(f"  run {i + 1:>2}: {ms:7.1f} ms")
    cold = samples[0]
    warm = samples[1:]
    return {
        "label": label,
        "args": " ".join(args),
        "runs": runs,
        "cold_ms": round(cold, 1),
        "warm_p50_ms": round(statistics.median(warm), 1) if warm else None,
        "warm_p95_ms": round(statistics.quantiles(warm, n=20)[18], 1) if len(warm) >= 20 else None,
        "warm_p99_ms": round(max(warm), 1) if warm else None,  # approx with low N
        "warm_mean_ms": round(statistics.mean(warm), 1) if warm else None,
        "all_samples": [round(s, 1) for s in samples],
    }


def find_exe(name: str) -> str | None:
    """Locate entry-point .exe (Windows) or symlink (POSIX)."""
    found = shutil.which(name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30, help="Spawns per scenario")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report")
    args = parser.parse_args()

    scenarios = []

    # Baseline 1: OS process spawn floor
    if sys.platform == "win32":
        scenarios.append(("OS spawn floor (cmd /c rem)", ["cmd", "/c", "rem"], None))
    else:
        scenarios.append(("OS spawn floor (/bin/true)", ["/bin/true"], None))

    # Baseline 2: Python interpreter cold start
    scenarios.append(
        (
            "Python interpreter cold start",
            [sys.executable, "-c", "pass"],
            None,
        )
    )

    # Hook 1: python -m neural_memory.hooks.post_tool_use
    scenarios.append(
        (
            "Hook via `python -m`",
            [sys.executable, "-m", "neural_memory.hooks.post_tool_use"],
            SAMPLE_INPUT,
        )
    )

    # Hook 2: compiled entry point .exe
    exe = find_exe("nmem-hook-post-tool-use")
    if exe:
        scenarios.append(
            (
                "Hook via `.exe` entry point",
                [exe],
                SAMPLE_INPUT,
            )
        )
    else:
        print("[skip] nmem-hook-post-tool-use not on PATH")

    # Hook 3: nmem CLI subcommand
    nmem = find_exe("nmem")
    if nmem:
        scenarios.append(
            (
                "Hook via `nmem post-tool-use-hook`",
                [nmem, "post-tool-use-hook"],
                SAMPLE_INPUT,
            )
        )

    results = []
    for label, cmd, stdin in scenarios:
        try:
            r = measure(label, cmd, stdin, args.runs)
            results.append(r)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}")
        except subprocess.TimeoutExpired:
            print(f"[fail] {label}: timeout")

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Scenario':<45} {'cold':>7} {'p50':>7} {'p95':>7} {'p99':>7}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<45} "
            f"{r['cold_ms']:>7.1f} "
            f"{(r['warm_p50_ms'] or 0):>7.1f} "
            f"{(r['warm_p95_ms'] or 0):>7.1f} "
            f"{(r['warm_p99_ms'] or 0):>7.1f}"
        )
    print("=" * 72)
    print(f"Platform: {sys.platform}  Python: {sys.version.split()[0]}  Runs: {args.runs}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nReport written: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
