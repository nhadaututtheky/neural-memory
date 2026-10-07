---
name: Windows Native CLI
about: Fix pipe encoding by shipping native binary (Rust/Go)
title: "[Windows] Native CLI binary for proper UTF-8 pipe handling"
labels: windows, cli, rust, go
assignees: ''
---

## Problem
Windows `cmd.exe` / PowerShell (non-UTF-8 codepage) corrupts CJK in shell pipes:
```
echo "决定：用 Redis" | nmem remember
# Becomes: "??：用 Redis" → stored as garbage
```

PR #209 workaround: `nmem remember --file <path>` reads UTF-8 directly.
But this puts burden on users and doesn't fix pipe usage.

## Root Cause
Python's `sys.stdin` inherits console codepage (CP936, CP949, CP1252, etc.).
`sys.stdin.buffer.read()` gets raw bytes but decoding uses wrong codec.

## Solutions (Pick One)

### Option A: Rust Binary (Recommended)
- Use `clap` for CLI, `tokio` + `sqlite`/`libsql`
- Compile to `nmem.exe` — single binary, no Python needed
- Proper UTF-8 handling via `std::io::stdin()`
- Can embed Python via `pyo3` for complex logic, or reimplement core

### Option B: Go Binary
- Similar to Rust, easier cross-compile
- Good SQLite support via `modernc.org/sqlite` (pure Go)
- Simpler deployment

### Option C: Python Fix (Partial)
```python
# In cli/_helpers.py - detect and warn
def ensure_utf8_pipe():
    if sys.platform == "win32":
        import locale, sys
        cp = locale.getpreferredencoding(False).lower()
        if "utf" not in cp:
            # Try to reconfigure stdin
            try:
                sys.stdin.reconfigure(encoding="utf-8")
            except Exception:
                logger.warning(f"Non-UTF-8 codepage {cp} — pipe input may corrupt CJK. Use --file.")
```

**Option C doesn't fix already-corrupted input** — only prevents future reads.

## Recommendation: Option A (Rust)
- Aligns with "Rust/PyO3 remains deferred" in CHANGELOG but for CLI only
- Can start with just `nmem remember --file` + `nmem forget` + `nmem list`
- Gradually migrate more commands
- Python package installs binary via `pip install neural-memory[cli-native]`

## Acceptance Criteria
- [ ] `echo "决定：用 Redis" | nmem remember` works on Windows (CP936, CP949, CP1252)
- [ ] `nmem remember --file` still works (backward compat)
- [ ] Binary size < 10MB
- [ ] CI builds Windows binary on every commit
- [ ] `pip install neural-memory` optionally installs binary

## Related
- PR #209: Chinese auto-capture + `--file` workaround
- `detect_encoding_damage()` in `safety/capture_hygiene.py`