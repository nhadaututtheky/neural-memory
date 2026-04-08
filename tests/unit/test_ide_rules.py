"""Tests for IDE rules generator."""

from __future__ import annotations

from pathlib import Path

from neural_memory.cli.ide_rules import (
    IDE_TARGETS,
    _get_rules_content,
    generate_rules_file,
)


class TestGetRulesContent:
    """Tests for rules content generation."""

    def test_content_not_empty(self) -> None:
        content = _get_rules_content()
        assert len(content) > 100

    def test_content_has_recall_instruction(self) -> None:
        content = _get_rules_content()
        assert "nmem_recall" in content

    def test_content_has_remember_instruction(self) -> None:
        content = _get_rules_content()
        assert "nmem_remember" in content

    def test_content_has_quality_tips(self) -> None:
        content = _get_rules_content()
        assert "quality" in content.lower()

    def test_content_has_context_dict(self) -> None:
        content = _get_rules_content()
        assert "context" in content

    def test_content_has_memory_types(self) -> None:
        content = _get_rules_content()
        for mem_type in ["decision", "error", "insight", "workflow"]:
            assert mem_type in content


class TestIdeTargets:
    """Tests for IDE target configuration."""

    def test_all_targets_have_required_keys(self) -> None:
        for ide, info in IDE_TARGETS.items():
            assert "file" in info, f"{ide} missing 'file'"
            assert "name" in info, f"{ide} missing 'name'"
            assert "description" in info, f"{ide} missing 'description'"

    def test_known_ides_present(self) -> None:
        assert "cursor" in IDE_TARGETS
        assert "windsurf" in IDE_TARGETS
        assert "cline" in IDE_TARGETS
        assert "gemini" in IDE_TARGETS
        assert "codex" in IDE_TARGETS
        assert "agents" in IDE_TARGETS

    def test_file_names_correct(self) -> None:
        assert IDE_TARGETS["cursor"]["file"] == ".cursorrules"
        assert IDE_TARGETS["windsurf"]["file"] == ".windsurfrules"
        assert IDE_TARGETS["cline"]["file"] == ".clinerules"
        assert IDE_TARGETS["gemini"]["file"] == "GEMINI.md"
        assert IDE_TARGETS["agents"]["file"] == "AGENTS.md"


class TestGenerateRulesFile:
    """Tests for generate_rules_file()."""

    def test_creates_file(self, tmp_path: Path) -> None:
        status = generate_rules_file(tmp_path, "cursor")
        assert status == "created"
        assert (tmp_path / ".cursorrules").exists()

    def test_file_has_content(self, tmp_path: Path) -> None:
        generate_rules_file(tmp_path, "cursor")
        content = (tmp_path / ".cursorrules").read_text(encoding="utf-8")
        assert "nmem_recall" in content
        assert "nmem_remember" in content

    def test_exists_without_force(self, tmp_path: Path) -> None:
        (tmp_path / ".cursorrules").write_text("existing", encoding="utf-8")
        status = generate_rules_file(tmp_path, "cursor")
        assert status == "exists"
        # Original content preserved
        assert (tmp_path / ".cursorrules").read_text(encoding="utf-8") == "existing"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        (tmp_path / ".cursorrules").write_text("old", encoding="utf-8")
        status = generate_rules_file(tmp_path, "cursor", force=True)
        assert status == "created"
        assert (tmp_path / ".cursorrules").read_text(encoding="utf-8") != "old"

    def test_unknown_ide(self, tmp_path: Path) -> None:
        status = generate_rules_file(tmp_path, "vscode")
        assert status == "unknown"

    def test_all_ides_generate(self, tmp_path: Path) -> None:
        for ide, info in IDE_TARGETS.items():
            status = generate_rules_file(tmp_path, ide)
            assert status == "created", f"Failed for {ide}"
            assert (tmp_path / info["file"]).exists(), f"File missing for {ide}"

    def test_all_generic_files_have_same_content(self, tmp_path: Path) -> None:
        """All non-Codex IDEs get the same rules content."""
        contents: list[str] = []
        for ide in IDE_TARGETS:
            if ide == "codex":
                continue
            generate_rules_file(tmp_path, ide)
            file_path = tmp_path / IDE_TARGETS[ide]["file"]
            contents.append(file_path.read_text(encoding="utf-8"))
        # All generic targets should be identical
        assert len(set(contents)) == 1

    def test_codex_has_distinct_content(self, tmp_path: Path) -> None:
        """Codex target produces different content from generic targets."""
        generate_rules_file(tmp_path, "codex")
        generate_rules_file(tmp_path, "cursor")
        codex = (tmp_path / "codex.md").read_text(encoding="utf-8")
        cursor = (tmp_path / ".cursorrules").read_text(encoding="utf-8")
        assert codex != cursor
        assert "Codex CLI" in codex
        assert "Hook 1: Session Start" in codex

    def test_immutability_of_targets(self) -> None:
        """IDE_TARGETS should not be modified by generate."""
        import copy

        original = copy.deepcopy(IDE_TARGETS)
        generate_rules_file(Path("/tmp"), "cursor")
        assert original == IDE_TARGETS
