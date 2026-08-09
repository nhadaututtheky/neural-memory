"""Measure cold-import and MCP tool-schema footprint (Phase 7).

Usage:
    python scripts/measure_runtime_footprint.py --tiers standard,full
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def cold_import_ms() -> float:
    """Spawn a clean process to measure import neural_memory time."""
    code = (
        "import time; t=time.perf_counter(); import neural_memory; "
        "print((time.perf_counter()-t)*1000)"
    )
    env_python = sys.executable
    proc = subprocess.run(
        [env_python, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    if proc.returncode != 0:
        return -1.0
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1.0


def schema_stats(tier: str) -> dict:
    from neural_memory.mcp.tool_schemas import get_tool_schemas_for_tier

    tools = get_tool_schemas_for_tier(tier)
    raw = json.dumps(tools)
    # Rough token estimate: ~4 chars/token
    return {
        "tier": tier,
        "tool_count": len(tools),
        "schema_bytes": len(raw.encode("utf-8")),
        "approx_tokens": len(raw) // 4,
    }


def base_dep_count() -> int:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.M | re.S)
    if not m:
        return -1
    return len(re.findall(r'"([^"]+)"', m.group(1)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="standard,full")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts/benchmark/results/runtime-footprint.json",
    )
    args = p.parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    report = {
        "schema_version": 1,
        "base_dependency_count": base_dep_count(),
        "cold_import_ms": round(cold_import_ms(), 2),
        "tiers": [schema_stats(t) for t in tiers],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    # Gate: base deps <= 4
    if report["base_dependency_count"] > 4:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
