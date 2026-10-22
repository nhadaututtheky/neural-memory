"""Windows pipe decoding for `nmem remember`.

The shell may write UTF-8, GBK, or cp949. The decoder must recover the
original text from those bytes. A code page that already replaced CJK
with "?" cannot be repaired.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_memory.cli.commands import memory as memory_mod
from neural_memory.cli.pipe_text import decode_pipe_bytes, read_stdin_text

CJK = "决定：用 Redis"
KOREAN = "결정: Redis를 사용한다"


class TestDecodePipeBytes:
    def test_utf8_cjk_ignores_legacy_code_page(self) -> None:
        assert decode_pipe_bytes(CJK.encode("utf-8"), code_pages=(1252,)) == CJK

    def test_utf8_bom_is_stripped(self) -> None:
        assert decode_pipe_bytes(CJK.encode("utf-8-sig"), code_pages=(936,)) == CJK

    def test_gbk_oem_page(self) -> None:
        raw = CJK.encode("gbk")
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")
        assert decode_pipe_bytes(raw, code_pages=(936,)) == CJK

    def test_cp949_oem_page(self) -> None:
        raw = KOREAN.encode("cp949")
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")
        assert decode_pipe_bytes(raw, code_pages=(949,)) == KOREAN

    def test_cp1252_western_text(self) -> None:
        text = "café — naïve"
        raw = text.encode("cp1252")
        assert decode_pipe_bytes(raw, code_pages=(1252,)) == text

    def test_cp1252_cannot_restore_cjk(self) -> None:
        raw = CJK.encode("cp1252", errors="replace")
        decoded = decode_pipe_bytes(raw, code_pages=(1252,))
        assert "决定" not in decoded
        assert "?" in decoded

    def test_undecodable_bytes_become_replacement(self) -> None:
        assert decode_pipe_bytes(b"\xff", code_pages=()) == "\ufffd"


class TestReadStdinText:
    def test_reads_buffer_with_oem_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdin = SimpleNamespace(buffer=SimpleNamespace(read=lambda: CJK.encode("gbk")))
        monkeypatch.setattr(sys, "stdin", stdin)
        monkeypatch.setattr(
            "neural_memory.cli.pipe_text.windows_code_pages",
            lambda: (936,),
        )
        assert read_stdin_text() == CJK


class TestResolveContent:
    def test_bare_command_reads_a_pipe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(memory_mod, "_stdin_is_pipe", lambda: True)
        monkeypatch.setattr(memory_mod, "read_stdin_text", lambda: CJK)
        assert memory_mod._resolve_content(content="", stdin=False, file_path=None) == CJK

    def test_tty_does_not_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(memory_mod, "_stdin_is_pipe", lambda: False)

        def _boom() -> str:
            raise AssertionError("stdin must stay unread on a console")

        monkeypatch.setattr(memory_mod, "read_stdin_text", _boom)
        assert memory_mod._resolve_content(content="", stdin=False, file_path=None) == ""

    def test_argument_wins_over_a_pipe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(memory_mod, "_stdin_is_pipe", lambda: True)

        def _boom() -> str:
            raise AssertionError("an argument must win over an unread pipe")

        monkeypatch.setattr(memory_mod, "read_stdin_text", _boom)
        assert memory_mod._resolve_content(content="kept", stdin=False, file_path=None) == "kept"

    def test_file_still_reads_utf8(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "note.txt"
        path.write_text(CJK, encoding="utf-8")
        monkeypatch.setattr(memory_mod, "_stdin_is_pipe", lambda: True)

        def _boom() -> str:
            raise AssertionError("--file must not read the pipe")

        monkeypatch.setattr(memory_mod, "read_stdin_text", _boom)
        assert memory_mod._resolve_content(content="", stdin=False, file_path=str(path)) == CJK
