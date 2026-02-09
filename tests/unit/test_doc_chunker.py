"""Tests for doc_chunker: markdown parsing, chunking, and file discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest  # noqa: F401 — needed for tmp_path fixture

from neural_memory.engine.doc_chunker import chunk_markdown, discover_files


class TestChunkMarkdown:
    """Tests for chunk_markdown()."""

    def test_chunk_basic_markdown(self) -> None:
        """Headings + paragraphs produce correct chunks with heading_path."""
        md = textwrap.dedent("""\
            # Getting Started

            Welcome to the project. This comprehensive guide walks you through the complete
            setup process and basic usage of the framework for new and experienced users alike
            who want to get productive quickly.

            ## Installation

            Install via pip using the terminal. Make sure you have Python version 3.10 or later
            installed on your system before proceeding with the full setup and configuration
            process that follows in the next steps.

            ## Usage

            Import the module and call the main function to get started with basic operations
            in your application code. The framework provides many utilities and helper functions
            that simplify common development tasks.
        """)

        chunks = chunk_markdown(md, "README.md")
        assert len(chunks) >= 2

        # First chunk should be from "Getting Started" section
        assert chunks[0].heading == "Getting Started"
        assert chunks[0].heading_path == ("Getting Started",)
        assert chunks[0].source_file == "README.md"

        # Second chunk should have nested heading path
        install_chunk = next(c for c in chunks if c.heading == "Installation")
        assert install_chunk.heading_path == ("Getting Started", "Installation")
        assert install_chunk.heading_level == 2

    def test_chunk_no_headings(self) -> None:
        """Plain text without headings produces a single chunk."""
        text = " ".join(["word"] * 30)
        chunks = chunk_markdown(text, "notes.md")
        assert len(chunks) == 1
        assert chunks[0].heading == ""
        assert chunks[0].heading_level == 0
        assert chunks[0].heading_path == ()

    def test_chunk_min_words_filter(self) -> None:
        """Short sections below min_words are skipped."""
        md = textwrap.dedent("""\
            # Title

            Short.

            ## Details

            This section has enough words to pass the minimum word count
            filter and should be included in the output chunks list.
        """)

        chunks = chunk_markdown(md, "test.md", min_words=10)
        assert all(c.word_count >= 10 for c in chunks)

    def test_chunk_max_words_split(self) -> None:
        """Long sections are split at paragraph boundaries."""
        para1 = " ".join(["alpha"] * 30)
        para2 = " ".join(["beta"] * 30)
        md = f"# Big Section\n\n{para1}\n\n{para2}"

        chunks = chunk_markdown(md, "big.md", max_words=35, min_words=10)
        assert len(chunks) == 2
        assert "alpha" in chunks[0].content
        assert "beta" in chunks[1].content

    def test_chunk_code_block_preserved(self) -> None:
        """Fenced code blocks are not split even if they contain heading-like patterns."""
        md = textwrap.dedent("""\
            # Guide

            Here is some example code that demonstrates usage patterns
            and common configurations for the library setup.

            ```python
            # This is a comment, not a heading
            def hello():
                print("world")
            ```

            More text after the code block to ensure proper parsing
            of content that follows fenced code sections.
        """)

        chunks = chunk_markdown(md, "guide.md")
        assert len(chunks) >= 1
        assert "```python" in chunks[0].content
        assert "def hello():" in chunks[0].content

    def test_chunk_tilde_fence_preserved(self) -> None:
        """Tilde-fenced code blocks (~) are handled correctly."""
        md = textwrap.dedent("""\
            # Guide

            Here is some example code that demonstrates usage patterns
            and common configurations for the library setup here.

            ~~~bash
            # Not a heading
            echo "hello world"
            ~~~

            More text after the tilde fence block to ensure proper
            parsing of content that follows tilde fenced sections.
        """)

        chunks = chunk_markdown(md, "guide.md")
        assert len(chunks) >= 1
        assert "~~~bash" in chunks[0].content

    def test_chunk_frontmatter_skipped(self) -> None:
        """YAML frontmatter is stripped before parsing."""
        md = textwrap.dedent("""\
            ---
            title: My Doc
            author: Test
            ---

            # Content

            The actual content starts here after the frontmatter has been stripped
            away by the preprocessing step. This paragraph needs enough words to pass
            the minimum word count threshold that filters out very short chunks.
        """)

        chunks = chunk_markdown(md, "doc.md")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "title: My Doc" not in chunk.content
            assert "author: Test" not in chunk.content

    def test_chunk_heading_hierarchy(self) -> None:
        """Nested headings build correct heading_path tuples."""
        md = textwrap.dedent("""\
            # Root

            Root content with enough words to pass the minimum filter threshold for chunk
            creation in the output. Adding more words here to ensure we exceed twenty words total.

            ## Level Two

            Level two content with enough words to pass the minimum filter threshold for chunk
            creation in the output. Adding more words here to ensure we exceed twenty words total.

            ### Level Three

            Level three content with enough words to pass the minimum filter threshold for chunk
            creation in the output. Adding more words here to ensure we exceed twenty words total.

            ## Another Level Two

            Another section at level two that should reset the heading stack appropriately when
            building the hierarchy. Adding more words here to ensure we exceed twenty words total.
        """)

        chunks = chunk_markdown(md, "nested.md")
        paths = [c.heading_path for c in chunks]

        assert ("Root",) in paths
        assert ("Root", "Level Two") in paths
        assert ("Root", "Level Two", "Level Three") in paths
        assert ("Root", "Another Level Two") in paths

    def test_chunk_empty_input(self) -> None:
        """Empty or whitespace-only input returns empty list."""
        assert chunk_markdown("", "empty.md") == []
        assert chunk_markdown("   \n\n  ", "blank.md") == []

    def test_chunk_word_count_accurate(self) -> None:
        """word_count matches actual word count of content."""
        md = "# Test\n\n" + " ".join(["word"] * 50)
        chunks = chunk_markdown(md, "count.md")
        assert len(chunks) == 1
        assert chunks[0].word_count == 50

    def test_chunk_setext_h1(self) -> None:
        """Setext-style H1 headings (===) are detected."""
        md = textwrap.dedent("""\
            My Title
            ========

            This is the content under a setext style heading level one that uses
            equals signs as the underline indicator for first-level headings.
        """)

        chunks = chunk_markdown(md, "setext.md")
        assert len(chunks) >= 1
        assert chunks[0].heading == "My Title"
        assert chunks[0].heading_level == 1

    def test_chunk_setext_h2(self) -> None:
        """Setext-style H2 headings (---) are detected."""
        md = textwrap.dedent("""\
            Subtitle
            --------

            This is the content under a setext style heading level two that uses
            dashes as the underline indicator for second-level headings here.
        """)

        chunks = chunk_markdown(md, "setext.md")
        assert len(chunks) >= 1
        assert chunks[0].heading == "Subtitle"
        assert chunks[0].heading_level == 2

    def test_chunk_atx_closing_hashes_stripped(self) -> None:
        """ATX closing hashes are stripped from heading text."""
        md = textwrap.dedent("""\
            ## Installation ##

            Install the package using pip or your preferred package manager
            to get started with the framework setup and configuration steps.
        """)

        chunks = chunk_markdown(md, "test.md")
        assert len(chunks) >= 1
        assert chunks[0].heading == "Installation"

    def test_chunk_heading_inline_formatting_cleaned(self) -> None:
        """Inline formatting is stripped from heading text."""
        md = textwrap.dedent("""\
            # Installing `numpy` on **Windows**

            This section covers the complete installation process for the numpy
            package on Windows operating systems with detailed step-by-step instructions
            that guide users through each phase of the setup and configuration workflow.
        """)

        chunks = chunk_markdown(md, "test.md")
        assert len(chunks) >= 1
        assert chunks[0].heading == "Installing numpy on Windows"

    def test_chunk_windows_crlf(self) -> None:
        """Windows CRLF line endings are handled correctly."""
        md = "# Title\r\n\r\nContent with CRLF line endings that should be normalized\r\nand parsed correctly by the chunker without any issues at all.\r\n"
        chunks = chunk_markdown(md, "win.md")
        assert len(chunks) >= 1
        assert "\r" not in chunks[0].content

    def test_chunk_frontmatter_line_offset(self) -> None:
        """Line numbers account for stripped frontmatter."""
        md = "---\ntitle: Test\n---\n\n# Content\n\n" + " ".join(["word"] * 25)
        chunks = chunk_markdown(md, "test.md")
        assert len(chunks) >= 1
        # The # Content heading is on line 5 (after 3 lines of frontmatter + blank)
        # Content starts on line 7, so line_start should be >= 5
        assert chunks[0].line_start >= 5


class TestDiscoverFiles:
    """Tests for discover_files()."""

    def test_discovers_md_files(self, tmp_path: Path) -> None:
        """Finds .md files recursively."""
        (tmp_path / "readme.md").write_text("# Hello", encoding="utf-8")
        sub = tmp_path / "docs"
        sub.mkdir()
        (sub / "guide.md").write_text("# Guide", encoding="utf-8")
        (sub / "notes.txt").write_text("not markdown", encoding="utf-8")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "readme.md" in names
        assert "guide.md" in names
        assert "notes.txt" not in names

    def test_excludes_directories(self, tmp_path: Path) -> None:
        """Excluded directories like node_modules are skipped."""
        (tmp_path / "good.md").write_text("# Good", encoding="utf-8")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "bad.md").write_text("# Bad", encoding="utf-8")
        git = tmp_path / ".git"
        git.mkdir()
        (git / "also_bad.md").write_text("# Also Bad", encoding="utf-8")

        files = discover_files(tmp_path)
        names = {f.name for f in files}
        assert "good.md" in names
        assert "bad.md" not in names
        assert "also_bad.md" not in names

    def test_custom_extensions(self, tmp_path: Path) -> None:
        """Custom extensions filter correctly."""
        (tmp_path / "doc.md").write_text("# MD", encoding="utf-8")
        (tmp_path / "doc.rst").write_text("RST", encoding="utf-8")

        files = discover_files(tmp_path, extensions={".rst"})
        names = {f.name for f in files}
        assert "doc.rst" in names
        assert "doc.md" not in names

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        files = discover_files(tmp_path)
        assert files == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory returns empty list."""
        files = discover_files(tmp_path / "nope")
        assert files == []

    def test_max_files_limit(self, tmp_path: Path) -> None:
        """File discovery respects max_files limit."""
        for i in range(10):
            (tmp_path / f"doc_{i:02d}.md").write_text(f"# Doc {i}", encoding="utf-8")

        files = discover_files(tmp_path, max_files=3)
        assert len(files) == 3
