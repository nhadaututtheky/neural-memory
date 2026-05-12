"""Tests for the SessionStart hook.

Covers: surface file lookup priority (project > global), source=resume skip,
empty surface skip, budget truncation, config disable, and the env-var
disable fast-path.
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from neural_memory.hooks.session_start import (
    _DEFAULT_BUDGET_CHARS,
    _emit_system_message,
    _find_global_surface,
    _find_surface_for_cwd,
    _is_enabled,
    _load_surface_text,
    _read_stdin,
    main,
)


@patch.dict(os.environ, {}, clear=False)
def _clear_disable_flag() -> None:
    os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)


class TestIsEnabled:
    def test_default_true(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True

    def test_explicit_false(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[hooks]\nsession_start = false\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is False

    def test_explicit_true(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[hooks]\nsession_start = true\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True


class TestFindSurfaceForCwd:
    def test_finds_project_surface(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        nm_dir = proj / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        surface = nm_dir / "surface.nm"
        surface.write_text("# project surface")

        found = _find_surface_for_cwd(str(proj))
        assert Path(found or "").resolve() == surface.resolve()

    def test_walks_up_to_project_root(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        nested = proj / "src" / "nested" / "deep"
        nested.mkdir(parents=True)
        nm_dir = proj / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        surface = nm_dir / "surface.nm"
        surface.write_text("# project surface")

        found = _find_surface_for_cwd(str(nested))
        assert Path(found or "").resolve() == surface.resolve()

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        empty = tmp_path / "noproj"
        empty.mkdir()
        assert _find_surface_for_cwd(str(empty)) is None

    def test_empty_cwd(self) -> None:
        assert _find_surface_for_cwd("") is None


class TestFindGlobalSurface:
    def test_returns_path_when_exists(self, tmp_path: Path) -> None:
        surfaces = tmp_path / "surfaces"
        surfaces.mkdir()
        (surfaces / "default.nm").write_text("# global surface")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            found = _find_global_surface()
        assert found is not None
        assert "default.nm" in found

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _find_global_surface() is None


class TestLoadSurfaceText:
    def test_project_wins_over_global(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        nm_dir = proj / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        (nm_dir / "surface.nm").write_text("PROJECT")

        nm_global = tmp_path / "global"
        (nm_global / "surfaces").mkdir(parents=True)
        (nm_global / "surfaces" / "default.nm").write_text("GLOBAL")

        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(nm_global)}):
            text = _load_surface_text(str(proj))
        assert text == "PROJECT"

    def test_falls_back_to_global(self, tmp_path: Path) -> None:
        nm_global = tmp_path / "global"
        (nm_global / "surfaces").mkdir(parents=True)
        (nm_global / "surfaces" / "default.nm").write_text("GLOBAL")
        nowhere = tmp_path / "no-project"
        nowhere.mkdir()

        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(nm_global)}):
            text = _load_surface_text(str(nowhere))
        assert text == "GLOBAL"

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        nm_dir = proj / ".neuralmemory"
        nm_dir.mkdir(parents=True)
        (nm_dir / "surface.nm").write_text("   \n  \n")

        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path / "g")}):
            assert _load_surface_text(str(proj)) is None


class TestEmitSystemMessage:
    def test_emits_full_message(self) -> None:
        stdout = StringIO()
        with patch.object(sys, "stdout", stdout):
            _emit_system_message("hello world", budget_chars=1000)
        out = json.loads(stdout.getvalue().strip())
        assert out["systemMessage"] == "hello world"

    def test_truncates_to_budget(self) -> None:
        stdout = StringIO()
        with patch.object(sys, "stdout", stdout):
            _emit_system_message("a" * 100, budget_chars=20)
        out = json.loads(stdout.getvalue().strip())
        assert len(out["systemMessage"]) <= 60  # 20 + truncation note
        assert "truncated" in out["systemMessage"]


class TestReadStdin:
    def test_valid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO('{"source": "startup", "cwd": "/x"}')):
            data = _read_stdin()
        assert data["source"] == "startup"

    def test_empty(self) -> None:
        with patch.object(sys, "stdin", StringIO("")):
            assert _read_stdin() == {}

    def test_invalid(self) -> None:
        with patch.object(sys, "stdin", StringIO("not json")):
            assert _read_stdin() == {}


class TestMain:
    def test_disabled_via_env(self, tmp_path: Path) -> None:
        stdout = StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "NEURALMEMORY_DIR": str(tmp_path),
                    "NEURALMEMORY_DISABLE_HOOKS": "1",
                },
            ),
            patch.object(sys, "stdin", StringIO('{"source": "startup"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_disabled_via_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[hooks]\nsession_start = false\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"source": "startup"}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_resume_source_skipped(self, tmp_path: Path) -> None:
        proj = tmp_path / "p"
        (proj / ".neuralmemory").mkdir(parents=True)
        (proj / ".neuralmemory" / "surface.nm").write_text("SHOULD_NOT_LOAD")

        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(
                sys,
                "stdin",
                StringIO(json.dumps({"source": "resume", "cwd": str(proj)})),
            ),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_no_surface_skipped(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path / "nm")}, clear=False),
            patch.object(
                sys,
                "stdin",
                StringIO(json.dumps({"source": "startup", "cwd": str(empty)})),
            ),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_injects_surface(self, tmp_path: Path) -> None:
        proj = tmp_path / "p"
        (proj / ".neuralmemory").mkdir(parents=True)
        (proj / ".neuralmemory" / "surface.nm").write_text(
            "# GRAPH\n[e1] my-concept (entity)\n"
        )

        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path / "nm")}, clear=False),
            patch.object(
                sys,
                "stdin",
                StringIO(json.dumps({"source": "startup", "cwd": str(proj)})),
            ),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        out = json.loads(stdout.getvalue().strip())
        assert "systemMessage" in out
        assert "my-concept" in out["systemMessage"]
        assert "Neural Memory surface" in out["systemMessage"]

    def test_default_budget_in_use(self) -> None:
        assert _DEFAULT_BUDGET_CHARS > 0
