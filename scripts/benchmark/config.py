"""CLI configuration for LongMemEval benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """All benchmark configuration in one place."""

    variant: str = "oracle"
    reader: str = "claude"
    judge: str = "claude"
    limit: int | None = None
    backend: str = "sqlite"
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results")
    claude_model: str = "claude-sonnet-4-20250514"
    gemma_model: str = "gemma3:12b"
    ollama_url: str = "http://localhost:11434"
    resume: bool = False
    retrieval_only: bool = False
    instance_ids_file: Path | None = None


def parse_args() -> BenchmarkConfig:
    """Parse CLI arguments into BenchmarkConfig."""
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark for Neural Memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--variant",
        choices=["oracle", "s", "m"],
        default="oracle",
        help="Dataset variant: oracle (~3 sessions), s (~40 sessions), m (~500 sessions)",
    )
    parser.add_argument(
        "--reader",
        choices=["claude", "gemma4", "ollama"],
        default="claude",
        help="LLM reader to generate answer hypotheses",
    )
    parser.add_argument(
        "--judge",
        choices=["claude", "gpt4o"],
        default="claude",
        help="LLM judge to evaluate hypothesis correctness",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of instances to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--backend",
        choices=["sqlite", "infinitydb"],
        default="sqlite",
        help="Neural Memory storage backend",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing LongMemEval JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for results, checkpoints, and brain DBs",
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-20250514",
        help="Claude model ID for reader/judge",
    )
    parser.add_argument(
        "--gemma-model",
        default="gemma3:12b",
        help="Ollama model name for Gemma reader",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-completed instances)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only measure retrieval (R@k, NDCG) — skip reader and judge",
    )
    parser.add_argument(
        "--instance-ids",
        type=Path,
        default=None,
        help="JSON file with instance IDs to run (from mini_bench.py). Overrides --limit.",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        variant=args.variant,
        reader=args.reader,
        judge=args.judge,
        limit=args.limit,
        backend=args.backend,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        claude_model=args.claude_model,
        gemma_model=args.gemma_model,
        ollama_url=args.ollama_url,
        resume=args.resume,
        retrieval_only=args.retrieval_only,
        instance_ids_file=args.instance_ids,
    )
