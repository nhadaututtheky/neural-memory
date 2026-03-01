"""Tests for multi-format document extractor."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neural_memory.engine.doc_extractor import (
    ExtractionError,
    SUPPORTED_EXTENSIONS,
    extract_to_markdown,
    get_missing_dependencies,
)


class TestExtractText:
    """Test markdown/text passthrough."""

    def test_extract_md(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld", encoding="utf-8")
        result = extract_to_markdown(md_file)
        assert "# Hello" in result
        assert "World" in result

    def test_extract_txt(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some plain text content", encoding="utf-8")
        result = extract_to_markdown(txt_file)
        assert "Some plain text content" in result


class TestExtractJson:
    """Test JSON extraction."""

    def test_extract_json_object(self, tmp_path: Path) -> None:
        json_file = tmp_path / "config.json"
        data = {"name": "Neural Memory", "version": "2.0", "features": {"pinned": True}}
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = extract_to_markdown(json_file)
        assert "# config" in result
        assert "Neural Memory" in result
        assert "features" in result

    def test_extract_json_array(self, tmp_path: Path) -> None:
        json_file = tmp_path / "items.json"
        data = [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}]
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = extract_to_markdown(json_file)
        assert "Alpha" in result
        assert "Beta" in result
        # Should create a table for array of objects
        assert "|" in result

    def test_extract_json_invalid(self, tmp_path: Path) -> None:
        json_file = tmp_path / "bad.json"
        json_file.write_text("{invalid json}", encoding="utf-8")

        with pytest.raises(ExtractionError, match="JSON parse error"):
            extract_to_markdown(json_file)


class TestExtractCsv:
    """Test CSV extraction."""

    def test_extract_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Age", "City"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])

        result = extract_to_markdown(csv_file)
        assert "# data" in result
        assert "Name" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "|" in result  # Markdown table

    def test_extract_csv_empty(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        result = extract_to_markdown(csv_file)
        assert "# empty" in result


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_unsupported_format(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("data", encoding="utf-8")

        with pytest.raises(ExtractionError, match="Unsupported format"):
            extract_to_markdown(bad_file)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_to_markdown(tmp_path / "nonexistent.md")

    def test_directory_not_file(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError, match="Not a file"):
            extract_to_markdown(tmp_path)

    def test_file_too_large(self, tmp_path: Path) -> None:
        """File size check (we mock it since we don't want to create a 50MB file)."""
        large_file = tmp_path / "large.md"
        large_file.write_text("x" * 100, encoding="utf-8")

        from unittest.mock import patch

        with patch("neural_memory.engine.doc_extractor._MAX_FILE_SIZE", 10):
            with pytest.raises(ExtractionError, match="File too large"):
                extract_to_markdown(large_file)

    def test_supported_extensions_complete(self) -> None:
        """All expected extensions are in SUPPORTED_EXTENSIONS."""
        expected = {
            ".md", ".mdx", ".txt", ".rst",
            ".pdf", ".docx", ".pptx",
            ".html", ".htm",
            ".json", ".xlsx", ".csv",
        }
        assert expected == SUPPORTED_EXTENSIONS


class TestMissingDependencies:
    """Test dependency checking."""

    def test_no_deps_needed_for_text(self) -> None:
        missing = get_missing_dependencies({".md", ".txt"})
        assert missing == []

    def test_json_csv_no_deps(self) -> None:
        missing = get_missing_dependencies({".json", ".csv"})
        assert missing == []
