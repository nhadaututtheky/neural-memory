"""Measure where tokens actually go in recall context.

The context has 2 sections:
1. '## Relevant Memories' — fiber summaries (max 5)
2. '## Related Information' — individual neurons with [type] tags (max 20)

Plus headers, overhead, age-compression effects.

This breaks down the actual token distribution so we know WHERE to optimize.

Usage: python scripts/measure_token_breakdown.py [--queries N]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
from pathlib import Path

QUERIES = [
    "Neural Memory architecture decisions",
    "Python async patterns and error handling",
    "config migration upgrade path",
    "InfinityDB performance optimization",
    "MCP tool handler implementation",
    "authentication and license activation",
    "consolidation quality tuning",
    "spreading activation lateral inhibition",
    "brain store publishing workflow",
    "error handling and logging patterns",
    "SQLite vector search embedding",
    "dashboard React components",
    "CI/CD pipeline GitHub Actions",
    "context compiler dedup merge",
    "fidelity decay ghost neurons",
    "sync hub Cloudflare Workers",
    "OpenClaw plugin development",
    "session reflection significance",
    "layered consciousness global brain",
    "predictive priming session topics",
]


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def split_sections(context: str) -> dict[str, str]:
    """Split context into Fibers / Related / Other sections."""
    if not context:
        return {"fibers": "", "related": "", "other": ""}
    parts = re.split(r"^##\s+(Relevant Memories|Related Information)\s*$", context, flags=re.MULTILINE)
    result = {"fibers": "", "related": "", "other": parts[0] if parts else ""}
    i = 1
    while i < len(parts) - 1:
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if header == "Relevant Memories":
            result["fibers"] = body
        elif header == "Related Information":
            result["related"] = body
        i += 2
    return result


def count_bullets(section: str) -> int:
    return sum(1 for line in section.split("\n") if line.strip().startswith("- "))


def measure_type_tag_overhead(related_section: str) -> tuple[int, int]:
    """[type] prefix on each bullet in Related Information section."""
    tag_re = re.compile(r"^\s*-\s+\[([^\]]+)\]\s+", re.MULTILINE)
    matches = tag_re.findall(related_section)
    total_tag_chars = sum(len(f"[{t}] ") for t in matches)
    total_tag_tokens = approx_tokens(" ".join(f"[{t}]" for t in matches))
    return total_tag_tokens, len(matches)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--output", type=str, default="scripts/_breakdown_output.txt")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = output_path.open("w", encoding="utf-8")

    def log(msg: str = "") -> None:
        out.write(msg + "\n")
        out.flush()
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    from neural_memory.engine.retrieval import ReflexPipeline
    from neural_memory.storage import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

    config = UnifiedConfig.load()
    brain_name = config.current_brain or "default"

    candidates = [
        config.data_dir / "brains" / f"{brain_name}.db",
        config.data_dir / "brains" / brain_name / "brain.db",
        config.data_dir / brain_name / "brain.db",
        config.data_dir / f"{brain_name}.db",
    ]
    db_path = next((c for c in candidates if c.exists()), None)
    if db_path is None:
        log("Brain DB not found")
        return

    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()
    brain = await storage.get_brain(brain_name)
    if brain is None:
        log("Brain not found")
        return
    storage.set_brain(brain_name)
    pipeline = ReflexPipeline(storage, brain.config)

    log(f"Brain: {brain_name}")
    log(f"Running {args.queries} queries...")
    log("")

    queries = (QUERIES * ((args.queries // len(QUERIES)) + 1))[: args.queries]

    rows: list[dict[str, float]] = []
    sample_context_shown = False

    for i, q in enumerate(queries):
        result = await pipeline.query(q)
        ctx = result.context or ""
        sections = split_sections(ctx)

        fibers_section = sections["fibers"]
        related_section = sections["related"]
        other_section = sections["other"]

        total_tokens = approx_tokens(ctx)
        fiber_tokens = approx_tokens(fibers_section)
        related_tokens = approx_tokens(related_section)
        other_tokens = approx_tokens(other_section)

        fiber_bullets = count_bullets(fibers_section)
        related_bullets = count_bullets(related_section)
        tag_tokens, tag_count = measure_type_tag_overhead(related_section)

        # Show the first sample's raw context once (to file only)
        if not sample_context_shown and ctx:
            out.write(f"\n===== SAMPLE CONTEXT: {q} =====\n")
            out.write(ctx)
            out.write("\n===== END SAMPLE =====\n\n")
            out.flush()
            sample_context_shown = True

        rows.append(
            {
                "query": q,
                "total": total_tokens,
                "fiber_toks": fiber_tokens,
                "fiber_n": fiber_bullets,
                "related_toks": related_tokens,
                "related_n": related_bullets,
                "tag_toks": tag_tokens,
                "other_toks": other_tokens,
            }
        )

        log(
            f"  [{i + 1}/{args.queries}] {q[:40]:<40}  "
            f"total={total_tokens:>5}  fib={fiber_tokens:>4}({fiber_bullets})  "
            f"rel={related_tokens:>4}({related_bullets})  "
            f"tag={tag_tokens:>3}  other={other_tokens:>3}"
        )

    log("")
    log("=" * 100)
    log(f"{'Section':<30} {'mean':>8} {'p50':>8} {'p95':>8} {'% of total':>12}")
    log("-" * 100)

    totals = [r["total"] for r in rows]
    mean_total = statistics.mean(totals)

    def stat_row(label: str, vals: list[float]) -> None:
        m = statistics.mean(vals) if vals else 0
        vs = sorted(vals)
        p50 = vs[len(vs) // 2] if vs else 0
        p95 = vs[int(len(vs) * 0.95)] if vs else 0
        pct = (m / mean_total * 100) if mean_total > 0 else 0
        log(f"{label:<30} {m:>8.0f} {p50:>8.0f} {p95:>8.0f} {pct:>11.1f}%")

    stat_row("total context", [r["total"] for r in rows])
    stat_row("  fibers section", [r["fiber_toks"] for r in rows])
    stat_row("  related section", [r["related_toks"] for r in rows])
    stat_row("  tag overhead in related", [r["tag_toks"] for r in rows])
    stat_row("  other (headers/blank)", [r["other_toks"] for r in rows])
    log("")
    stat_row("fiber bullets (count)", [r["fiber_n"] for r in rows])
    stat_row("related bullets (count)", [r["related_n"] for r in rows])

    log("=" * 100)

    # Opportunity estimate
    total_all = sum(r["total"] for r in rows)
    tag_all = sum(r["tag_toks"] for r in rows)
    related_all = sum(r["related_toks"] for r in rows)
    log(f"\nTotal tokens across {len(rows)} queries: {total_all:,}")
    log(f"Total 'related' section tokens: {related_all:,} ({related_all / max(total_all, 1) * 100:.1f}%)")
    log(f"Total '[type]' tag overhead: {tag_all:,} ({tag_all / max(total_all, 1) * 100:.1f}%)")
    log("")
    log("Optimization opportunities:")
    log(f"  - Drop [type] tags: save ~{tag_all / len(rows):.0f} tok/query ({tag_all / max(total_all, 1) * 100:.1f}% of context)")
    log(f"  - Dedup Related vs Fibers: unknown (need overlap check)")

    out.close()
    print(f"\n--> Full output (with sample context) written to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
