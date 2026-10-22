"""Decode a shell pipe from the raw bytes.

``sys.stdin.read()`` uses the process code page. On Windows that page is
often cp1252 while the pipe is UTF-8, or UTF-8 while ``cmd.exe`` wrote
GBK. The bytes are still intact. Valid UTF-8 wins, including a BOM.
Anything else is decoded with the OEM code page (``echo``) and then the
ANSI code page.

A shell that cannot represent the characters has already written ``?``.
CP1252 has no CJK code points, and Windows PowerShell 5.1 pipes ASCII.
Those bytes cannot be repaired. ``nmem remember --file`` reads UTF-8
from disk instead.
"""

from __future__ import annotations

import codecs
import sys


def windows_code_pages() -> tuple[int, ...]:
    """Return the OEM code page, then the ANSI page when it differs."""
    if sys.platform != "win32":
        return ()
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_oem = kernel.GetOEMCP
    get_acp = kernel.GetACP
    get_oem.restype = ctypes.c_uint
    get_acp.restype = ctypes.c_uint
    pages: list[int] = []
    for code_page in (int(get_oem()), int(get_acp())):
        if code_page and code_page not in pages:
            pages.append(code_page)
    return tuple(pages)


def _codec_name(code_page: int) -> str | None:
    """Return a Python codec for a Windows code page, if one exists."""
    if code_page == 65001:
        return "utf-8"
    if code_page == 54936:
        name = "gb18030"
    else:
        name = f"cp{code_page}"
    try:
        codecs.lookup(name)
    except LookupError:
        return None
    return name


def decode_pipe_bytes(data: bytes, *, code_pages: tuple[int, ...] = ()) -> str:
    """Decode ``data`` to text.

    ``code_pages`` is tried only when the buffer is not UTF-8. The final
    fallback inserts U+FFFD so the encoding guard can refuse the write.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    for code_page in code_pages:
        name = _codec_name(code_page)
        if name is None or name == "utf-8":
            continue
        try:
            return data.decode(name)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_stdin_text() -> str:
    """Read stdin as bytes and decode the pipe. Strip surrounding space."""
    return decode_pipe_bytes(sys.stdin.buffer.read(), code_pages=windows_code_pages()).strip()
