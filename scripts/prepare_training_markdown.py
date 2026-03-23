"""Batch-convert markdown files into a trainer-friendly structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural_memory.engine.doc_chunker import discover_files
from neural_memory.engine.training_markdown import (
    MarkdownPreparationConfig,
    prepare_markdown_for_training,
)


def main() -> None:
    """Run markdown preparation across a file or directory."""
    parser = argparse.ArgumentParser(
        description="Prepare markdown files for NeuralMemory doc training."
    )
    parser.add_argument("source", help="Source markdown file or directory")
    parser.add_argument(
        "--output-dir",
        default="prepared-training-docs",
        help="Directory to write prepared markdown into",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".md", ".mdx"],
        help="Extensions to include when source is a directory",
    )
    parser.add_argument(
        "--min-section-words",
        type=int,
        default=20,
        help="Minimum words for a standalone section before notes are merged",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source.is_relative_to(cwd):
        raise SystemExit("Source path must be within the current working directory")
    if not output_dir.is_relative_to(cwd):
        raise SystemExit("Output directory must be within the current working directory")
    if not source.exists():
        raise SystemExit("Source path not found")

    config = MarkdownPreparationConfig(min_section_words=args.min_section_words)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = (
        [source]
        if source.is_file()
        else discover_files(source, extensions=frozenset(args.ext))
    )
    if not files:
        raise SystemExit("No markdown files found")

    converted = 0
    synthesized_total = 0
    for file_path in files:
        relative_path = file_path.name if source.is_file() else str(file_path.relative_to(source))
        result = prepare_markdown_for_training(
            file_path.read_text(encoding="utf-8"),
            source_name=relative_path,
            config=config,
        )

        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.output_text, encoding="utf-8")
        converted += 1
        synthesized_total += len(result.synthesized_sections)
        sys.stdout.write(
            f"prepared {relative_path} -> {destination.relative_to(cwd)} "
            f"(synthetic={', '.join(result.synthesized_sections) or 'none'}; "
            f"tags={', '.join(result.topic_tags) or 'none'}; "
            f"labels={', '.join(result.labels) or 'none'})\n"
        )

    sys.stdout.write(
        f"converted {converted} file(s) into {output_dir.relative_to(cwd)}; "
        f"added {synthesized_total} synthetic section(s)\n"
    )


if __name__ == "__main__":
    main()
