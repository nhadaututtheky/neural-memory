"""Markdown document chunker for doc-to-brain training pipeline.

Parses markdown files into semantic chunks based on heading structure.
Each chunk preserves its heading hierarchy path, enabling neural graph
construction with CONTAINS synapses mirroring the document structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ATX headings: optional 0-3 leading spaces, #{1,6}, mandatory space, content
# Captures: group(1)=hashes, group(2)=heading text (closing hashes stripped later)
_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")

# Setext headings: text line followed by === or --- underline
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")

# Fenced code blocks: backtick or tilde, at least 3, at line start (0-3 spaces)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# YAML frontmatter: --- delimiters at document start (greedy to handle inner ---)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*\n)---\s*\n", re.DOTALL)

# Inline markdown cleanup for heading text
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_INLINE_ITALIC_RE = re.compile(r"\*(.+?)\*|_(.+?)_")
_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".md", ".mdx"})

# All extensions supported when doc_extractor is used (extract → markdown → chunk)
EXTENDED_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".mdx", ".txt", ".rst",
    ".pdf", ".docx", ".pptx",
    ".html", ".htm",
    ".json", ".xlsx", ".csv",
})
_DEFAULT_EXCLUDE: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "dist",
        "build",
    }
)

MAX_DISCOVER_FILES = 500


@dataclass(frozen=True)
class DocChunk:
    """A semantic chunk extracted from a markdown document.

    Attributes:
        content: The chunk text (without the heading line itself).
        heading: Nearest heading text, cleaned of inline formatting.
        heading_level: 1-6 for headings, 0 if no heading context.
        heading_path: Full hierarchy tuple (e.g., ("Guide", "Installation")).
        source_file: Relative path of the source file.
        line_start: 1-based line number where the chunk starts in the original file.
        line_end: 1-based line number where the chunk ends in the original file.
        word_count: Number of whitespace-delimited words.
    """

    content: str
    heading: str
    heading_level: int
    heading_path: tuple[str, ...]
    source_file: str
    line_start: int
    line_end: int
    word_count: int


def chunk_markdown(
    text: str,
    source_file: str = "",
    *,
    min_words: int = 20,
    max_words: int = 500,
) -> list[DocChunk]:
    """Parse markdown text into semantic chunks split by headings.

    Args:
        text: Raw markdown content.
        source_file: Relative path for provenance tracking.
        min_words: Skip chunks with fewer words (noise filter).
        max_words: Split chunks exceeding this at paragraph boundaries.

    Returns:
        List of DocChunk instances ordered by document position.
    """
    if not text or not text.strip():
        return []

    # Normalize line endings (Windows CRLF → LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip YAML frontmatter, track offset for line numbers
    frontmatter_lines = 0
    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        frontmatter_lines = fm_match.group(0).count("\n")
        text = text[fm_match.end() :]

    # Split into lines and parse sections with fence-aware heading detection
    lines = text.split("\n")
    sections = _split_by_headings(lines)

    # Build chunks with heading hierarchy tracking
    heading_stack: list[tuple[int, str]] = []
    chunks: list[DocChunk] = []

    for section in sections:
        level = section["level"]
        heading = section["heading"]
        body = section["body"]
        # Adjust line numbers for frontmatter offset (1-based)
        line_start = section["line_start"] + frontmatter_lines
        line_end = section["line_end"] + frontmatter_lines

        # Update heading stack
        if level > 0:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))

        heading_path = tuple(h[1] for h in heading_stack)

        if not body.strip():
            continue

        word_count = len(body.split())

        if word_count < min_words:
            continue

        if word_count <= max_words:
            chunks.append(
                DocChunk(
                    content=body.strip(),
                    heading=heading,
                    heading_level=level,
                    heading_path=heading_path,
                    source_file=source_file,
                    line_start=line_start,
                    line_end=line_end,
                    word_count=word_count,
                )
            )
        else:
            sub_chunks = _split_large_section(
                body=body,
                heading=heading,
                heading_level=level,
                heading_path=heading_path,
                source_file=source_file,
                line_start=line_start,
                max_words=max_words,
                min_words=min_words,
            )
            chunks.extend(sub_chunks)

    return chunks


def discover_files(
    directory: Path,
    extensions: frozenset[str] | set[str] | None = None,
    exclude: frozenset[str] | set[str] | None = None,
    *,
    max_files: int = MAX_DISCOVER_FILES,
) -> list[Path]:
    """Recursively discover documentation files.

    Args:
        directory: Root directory to search.
        extensions: File extensions to include (default: .md, .mdx).
        exclude: Directory names to skip.
        max_files: Maximum number of files to return (safety limit).

    Returns:
        Sorted list of matching file paths (capped at max_files).
    """
    exts = extensions if extensions is not None else _DEFAULT_EXTENSIONS
    excl = exclude if exclude is not None else _DEFAULT_EXCLUDE

    files: list[Path] = []
    if not directory.is_dir():
        return files

    resolved_dir = directory.resolve()

    for item in sorted(directory.rglob("*")):
        if item.is_file() and item.suffix in exts:
            # Security: validate resolved path is within directory (prevent symlink escape)
            try:
                resolved_item = item.resolve()
                if not resolved_item.is_relative_to(resolved_dir):
                    continue
            except (OSError, ValueError):
                continue

            # Check no excluded directory in the path
            if not any(part in excl for part in item.relative_to(directory).parts):
                files.append(item)

            if len(files) >= max_files:
                break

    return files


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_heading_text(text: str) -> str:
    """Strip inline markdown formatting from heading text."""
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _INLINE_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _INLINE_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _INLINE_LINK_RE.sub(r"\1", text)
    return text.strip()


def _split_by_headings(lines: list[str]) -> list[dict[str, Any]]:
    """Split lines into sections by ATX and setext headings.

    Handles fenced code blocks (``` and ~~~) to avoid false heading matches.
    Returns list of dicts: {level, heading, body, line_start, line_end}.
    """
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_fence = False
    fence_marker = ""

    def _finalize_section(section: dict[str, Any], end_line: int) -> None:
        section["line_end"] = end_line
        section["body"] = "\n".join(lines[section["body_start"] : end_line])
        sections.append(section)

    def _start_section(
        level: int, heading: str, body_start: int, line_start: int
    ) -> dict[str, Any]:
        return {
            "level": level,
            "heading": _clean_heading_text(heading),
            "body_start": body_start,
            "line_start": line_start,
            "line_end": 0,
            "body": "",
        }

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track fenced code blocks (``` or ~~~)
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            marker_char = fence_match.group(1)[0]  # ` or ~
            marker_len = len(fence_match.group(1))
            if in_fence:
                # Check if this closes the current fence
                stripped = line.strip()
                if (
                    stripped == marker_char * len(stripped)
                    and len(stripped) >= marker_len
                    and marker_char == fence_marker
                ):
                    in_fence = False
                    fence_marker = ""
            else:
                in_fence = True
                fence_marker = marker_char
            if current is None:
                current = _start_section(0, "", 0, 1)
            i += 1
            continue

        if in_fence:
            if current is None:
                current = _start_section(0, "", 0, 1)
            i += 1
            continue

        # Check ATX heading
        atx_match = _ATX_HEADING_RE.match(line)
        if atx_match:
            if current is not None:
                _finalize_section(current, i)

            level = len(atx_match.group(1))
            heading = atx_match.group(2)
            current = _start_section(level, heading, i + 1, i + 1)
            i += 1
            continue

        # Check setext heading (line i is underline, line i-1 is text)
        if i > 0 and current is not None:
            if _SETEXT_H1_RE.match(line) and lines[i - 1].strip():
                # Previous line is heading text, this line is ===
                heading_text = lines[i - 1].strip()
                # Remove the heading-text line from current section's body
                _finalize_section(current, i - 1)
                current = _start_section(1, heading_text, i + 1, i + 1)
                i += 1
                continue
            if _SETEXT_H2_RE.match(line) and lines[i - 1].strip():
                heading_text = lines[i - 1].strip()
                _finalize_section(current, i - 1)
                current = _start_section(2, heading_text, i + 1, i + 1)
                i += 1
                continue

        # Regular content line
        if current is None:
            current = _start_section(0, "", 0, 1)

        i += 1

    # Finalize last section
    if current is not None:
        _finalize_section(current, len(lines))

    return sections


def _split_large_section(
    *,
    body: str,
    heading: str,
    heading_level: int,
    heading_path: tuple[str, ...],
    source_file: str,
    line_start: int,
    max_words: int,
    min_words: int,
) -> list[DocChunk]:
    """Split a large section at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", body.strip())
    chunks: list[DocChunk] = []
    current_parts: list[str] = []
    current_words = 0
    current_line = line_start

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > max_words and current_parts:
            # Flush current accumulation
            content = "\n\n".join(current_parts)
            if current_words >= min_words:
                chunks.append(
                    DocChunk(
                        content=content.strip(),
                        heading=heading,
                        heading_level=heading_level,
                        heading_path=heading_path,
                        source_file=source_file,
                        line_start=current_line,
                        line_end=current_line + content.count("\n"),
                        word_count=current_words,
                    )
                )
            current_line = current_line + content.count("\n") + 2
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    # Flush remainder
    if current_parts and current_words >= min_words:
        content = "\n\n".join(current_parts)
        chunks.append(
            DocChunk(
                content=content.strip(),
                heading=heading,
                heading_level=heading_level,
                heading_path=heading_path,
                source_file=source_file,
                line_start=current_line,
                line_end=current_line + content.count("\n"),
                word_count=current_words,
            )
        )

    return chunks
