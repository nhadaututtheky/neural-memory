"""Spike: benchmark light hook variants vs current heavy hook.

Variants tested:
  v0: current `nmem_memory.hooks.post_tool_use` (heavy, full imports)
  v1: minimal stdlib (no pathlib, no NM imports)
  v2: + file lock (msvcrt/fcntl)
  v3: stdlib + pathlib (measure pathlib cost)
  v4: empty fast-path (absolute floor)

Also measures `python -X importtime` for v0 to identify slow imports.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


SAMPLE_INPUT = json.dumps(
    {
        "tool_name": "Edit",
        "tool_input": {"file_path": "C:/foo/bar.py", "old_string": "x", "new_string": "y"},
        "tool_error": None,
        "duration_ms": 12,
    }
).encode("utf-8")


def time_spawn(args: list[str], stdin: bytes, env: dict[str, str] | None = None) -> float:
    start = time.perf_counter()
    proc = subprocess.run(args, input=stdin, capture_output=True, timeout=10, shell=False, env=env)
    elapsed = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        sys.stderr.write(f"[warn] exit={proc.returncode} stderr={proc.stderr[:200]!r}\n")
    return elapsed


def measure(label: str, args: list[str], runs: int, env: dict[str, str] | None = None) -> dict:
    print(f"\n>>> {label}")
    samples = [time_spawn(args, SAMPLE_INPUT, env=env) for _ in range(runs)]
    cold = samples[0]
    warm = samples[1:]
    p50 = statistics.median(warm) if warm else cold
    p95 = statistics.quantiles(warm, n=20)[18] if len(warm) >= 20 else max(warm)
    p99 = max(warm) if warm else cold
    print(f"  cold={cold:6.1f}  p50={p50:6.1f}  p95={p95:6.1f}  p99={p99:6.1f}")
    return {
        "label": label,
        "cold": round(cold, 1),
        "p50": round(p50, 1),
        "p95": round(p95, 1),
        "p99": round(p99, 1),
        "samples": [round(s, 1) for s in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    bench_dir = Path(__file__).parent

    scenarios = [
        (
            "v0: heavy (current post_tool_use)",
            [sys.executable, "-m", "neural_memory.hooks.post_tool_use"],
        ),
        ("v1: minimal stdlib", [sys.executable, str(bench_dir / "_light_hook_v1_minimal.py")]),
        ("v2: + file lock", [sys.executable, str(bench_dir / "_light_hook_v2_locked.py")]),
        ("v3: stdlib + pathlib", [sys.executable, str(bench_dir / "_light_hook_v3_pathlib.py")]),
        (
            "v4: empty fast-path (floor)",
            [sys.executable, str(bench_dir / "_light_hook_v4_disabled.py")],
        ),
    ]

    # Sandbox bench writes to a scratch dir so they don't pollute the project root.
    import tempfile

    scratch = tempfile.mkdtemp(prefix="nmem-bench-")
    bench_env = {**os.environ, "NEURALMEMORY_DIR": scratch}
    bench_env.pop("NEURALMEMORY_DISABLE_HOOKS", None)
    results = [measure(label, cmd, args.runs, env=bench_env) for label, cmd in scenarios]

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Variant':<40} {'cold':>7} {'p50':>7} {'p95':>7} {'p99':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['label']:<40} {r['cold']:>7.1f} {r['p50']:>7.1f} {r['p95']:>7.1f} {r['p99']:>7.1f}"
        )
    print("=" * 70)

    # Delta vs v0 (current heavy)
    v0 = results[0]["p50"]
    print(f"\nSpeedup vs v0 (current heavy, p50={v0:.0f}ms):")
    for r in results[1:]:
        speedup = v0 / r["p50"] if r["p50"] > 0 else float("inf")
        savings = v0 - r["p50"]
        print(f"  {r['label']:<40} {speedup:>5.1f}x  ({savings:+.0f}ms saved)")

    # importtime analysis on v0
    print("\n>>> python -X importtime analysis on heavy hook (top 10 slowest)")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                "from neural_memory.hooks import post_tool_use",
            ],
            capture_output=True,
            timeout=30,
        )
        lines = proc.stderr.decode("utf-8", errors="replace").splitlines()
        # Lines look like: "import time:   123 |    456 |     module_name"
        parsed = []
        for line in lines:
            if not line.startswith("import time:"):
                continue
            try:
                parts = line.split("|")
                self_us = int(parts[0].split(":")[1].strip())
                cum_us = int(parts[1].strip())
                name = parts[2].strip()
                parsed.append((cum_us, self_us, name))
            except (ValueError, IndexError):
                continue
        parsed.sort(key=lambda x: x[0], reverse=True)
        print(f"  {'cum_ms':>8}  {'self_ms':>8}  module")
        for cum, self_us, name in parsed[:15]:
            print(f"  {cum / 1000:>8.1f}  {self_us / 1000:>8.1f}  {name}")
    except Exception as e:
        print(f"  [skip importtime: {e}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
