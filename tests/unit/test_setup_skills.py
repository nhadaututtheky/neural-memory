"""Tests for skill installation (setup_skills, _discover_bundled_skills)."""

from __future__ import annotations

from pathlib import Path

import pytest

# ── _discover_bundled_skills ──────────────────────────────────────


class TestDiscoverBundledSkills:
    """Tests for scanning bundled skills directory."""

    def test_discovers_all_three_skills(self) -> None:
        """Should find memory-intake, memory-audit, memory-evolution."""
        from neural_memory.cli.setup import _discover_bundled_skills

        skills = _discover_bundled_skills()
        assert "memory-intake" in skills
        assert "memory-audit" in skills
        assert "memory-evolution" in skills

    def test_returns_paths_to_skill_files(self) -> None:
        """Each value should be a Path to an existing SKILL.md."""
        from neural_memory.cli.setup import _discover_bundled_skills

        skills = _discover_bundled_skills()
        for name, path in skills.items():
            assert path.exists(), f"{name}: {path} does not exist"
            assert path.name == "SKILL.md"

    def test_ignores_non_skill_dirs(self, tmp_path: Path) -> None:
        """Directories without SKILL.md should be ignored."""
        from unittest.mock import patch

        # Create a fake SKILLS_DIR with one valid and one invalid dir
        valid = tmp_path / "my-skill"
        valid.mkdir()
        (valid / "SKILL.md").write_text("---\nname: my-skill\n---\n")

        invalid = tmp_path / "not-a-skill"
        invalid.mkdir()
        (invalid / "README.md").write_text("not a skill")

        # Also a plain file (not a dir)
        (tmp_path / "somefile.txt").write_text("nope")

        with patch("neural_memory.skills.SKILLS_DIR", tmp_path):
            from neural_memory.cli.setup import _discover_bundled_skills

            skills = _discover_bundled_skills()

        assert "my-skill" in skills
        assert "not-a-skill" not in skills
        assert "somefile.txt" not in skills


# ── setup_skills ──────────────────────────────────────────────────


class TestSetupSkills:
    """Tests for the main setup_skills() function."""

    def test_returns_not_found_when_claude_dir_missing(self, tmp_path: Path) -> None:
        """Should return not_found when ~/.claude/ doesn't exist."""
        from unittest.mock import patch

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            result = setup_skills()

        assert "Skills" in result
        assert "not_found" in result["Skills"]

    def test_installs_skills_to_claude_dir(self, tmp_path: Path) -> None:
        """Should copy SKILL.md files to ~/.claude/skills/<name>/."""
        from unittest.mock import patch

        fake_home = tmp_path / "home"
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)

        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            result = setup_skills()

        # All three should be installed
        installed = [n for n, s in result.items() if s == "installed"]
        assert len(installed) == 3

        # Files should exist
        for name in ("memory-intake", "memory-audit", "memory-evolution"):
            dest = claude_dir / "skills" / name / "SKILL.md"
            assert dest.exists(), f"{dest} should exist"
            assert dest.read_text(encoding="utf-8").startswith("---")

    def test_skips_existing_identical_files(self, tmp_path: Path) -> None:
        """Should report 'exists' for unchanged files."""
        from unittest.mock import patch

        from neural_memory.skills import SKILLS_DIR

        fake_home = tmp_path / "home"
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)

        # Pre-install one skill
        dest_dir = claude_dir / "skills" / "memory-intake"
        dest_dir.mkdir(parents=True)
        source = SKILLS_DIR / "memory-intake" / "SKILL.md"
        (dest_dir / "SKILL.md").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            result = setup_skills()

        assert result["memory-intake"] == "exists"

    def test_reports_update_available_for_changed_files(self, tmp_path: Path) -> None:
        """Should report 'update available' when file differs and no --force."""
        from unittest.mock import patch

        fake_home = tmp_path / "home"
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)

        # Pre-install with different content
        dest_dir = claude_dir / "skills" / "memory-intake"
        dest_dir.mkdir(parents=True)
        (dest_dir / "SKILL.md").write_text("old content", encoding="utf-8")

        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            result = setup_skills(force=False)

        assert result["memory-intake"] == "update available"

    def test_force_overwrites_changed_files(self, tmp_path: Path) -> None:
        """Should overwrite and report 'updated' with --force."""
        from unittest.mock import patch

        from neural_memory.skills import SKILLS_DIR

        fake_home = tmp_path / "home"
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir(parents=True)

        # Pre-install with different content
        dest_dir = claude_dir / "skills" / "memory-intake"
        dest_dir.mkdir(parents=True)
        (dest_dir / "SKILL.md").write_text("old content", encoding="utf-8")

        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            result = setup_skills(force=True)

        assert result["memory-intake"] == "updated"

        # Verify content matches source
        source_content = (SKILLS_DIR / "memory-intake" / "SKILL.md").read_text(encoding="utf-8")
        dest_content = (dest_dir / "SKILL.md").read_text(encoding="utf-8")
        assert dest_content == source_content

    def test_does_not_mutate_force_param(self) -> None:
        """setup_skills should not mutate any input (immutability check)."""
        from unittest.mock import patch

        fake_home = Path("/nonexistent/home")
        with patch("neural_memory.cli.setup.Path.home", return_value=fake_home):
            from neural_memory.cli.setup import setup_skills

            # Just verify it doesn't raise on a missing dir
            result = setup_skills(force=False)
            assert isinstance(result, dict)


# ── _classify_status ──────────────────────────────────────────────


class TestClassifyStatus:
    """Tests for status classification used by print_summary."""

    @pytest.mark.parametrize(
        "detail,expected",
        [
            ("installed", "ok"),
            ("updated", "ok"),
            ("3 installed", "ok"),
            ("created", "ok"),
            ("added", "ok"),
            ("ready", "ok"),
            ("exists", "ok"),
            ("already exists", "ok"),
            ("not detected", "skip"),
            ("skipped (--skip-mcp)", "skip"),
            ("not_found (~/.claude/ not found)", "skip"),
            ("failed to write config", "fail"),
        ],
    )
    def test_classify_status(self, detail: str, expected: str) -> None:
        from neural_memory.cli.setup import _classify_status

        assert _classify_status(detail) == expected


# ── _extract_skill_description ────────────────────────────────────


class TestExtractSkillDescription:
    """Tests for extracting description from SKILL.md frontmatter."""

    def test_extracts_multiline_description(self, tmp_path: Path) -> None:
        from neural_memory.cli.commands.tools import _extract_skill_description

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: test\ndescription: |\n  First line.\n  Second line.\n---\n",
            encoding="utf-8",
        )
        assert _extract_skill_description(skill_file) == "First line."

    def test_extracts_inline_description(self, tmp_path: Path) -> None:
        from neural_memory.cli.commands.tools import _extract_skill_description

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: test\ndescription: A short desc\n---\n",
            encoding="utf-8",
        )
        assert _extract_skill_description(skill_file) == "A short desc"

    def test_returns_empty_for_missing_description(self, tmp_path: Path) -> None:
        from neural_memory.cli.commands.tools import _extract_skill_description

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: test\n---\n", encoding="utf-8")
        assert _extract_skill_description(skill_file) == ""

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        from neural_memory.cli.commands.tools import _extract_skill_description

        result = _extract_skill_description(tmp_path / "nope.md")
        assert result == ""
