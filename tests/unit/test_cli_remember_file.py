"""Tests for `nmem remember --file`, the Windows-safe input path.

Reading from a file bypasses the shell pipe entirely; these tests pin the
failure modes so a bad path fails loudly instead of storing truncated content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from neural_memory.cli.commands.memory import _read_content_file


class TestReadContentFile:
    def test_reads_utf8_verbatim(self, tmp_path: Path) -> None:
        path = tmp_path / "note.txt"
        text = "中文内容 with emoji 🐋 and | pipe and \\u4e2d literal"
        path.write_text(text, encoding="utf-8")
        assert _read_content_file(str(path)) == text

    def test_strips_utf8_bom(self, tmp_path: Path) -> None:
        """A BOM-prefixed file must not leak \\ufeff into the stored content."""
        path = tmp_path / "bom.txt"
        path.write_text("中文内容", encoding="utf-8-sig")
        assert _read_content_file(str(path)) == "中文内容"

    def test_surrounding_whitespace_is_trimmed(self, tmp_path: Path) -> None:
        path = tmp_path / "pad.txt"
        path.write_text("  content  \n", encoding="utf-8")
        assert _read_content_file(str(path)) == "content"

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(typer.Exit):
            _read_content_file(str(tmp_path / "does-not-exist.txt"))

    def test_invalid_utf8_exits(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_bytes(b"\xff\xfe\x00\x00 not valid utf-8")
        with pytest.raises(typer.Exit):
            _read_content_file(str(path))
