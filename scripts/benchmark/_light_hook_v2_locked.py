"""Light hook variant 2: + file lock for concurrent safety.

Adds msvcrt (Win) / fcntl (Unix) import to handle parallel hook fires.
"""

import json
import os
import sys


def _lock(fileobj) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(fileobj.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(fileobj, fcntl.LOCK_EX)


def _unlock(fileobj) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            fileobj.seek(0)
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl

        fcntl.flock(fileobj, fcntl.LOCK_UN)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.stdout.write("{}\n")
        return

    tool = data.get("tool_name", "")
    if tool in {"TodoRead", "TodoWrite", "TaskList", "Read", "Glob", "Grep"}:
        sys.stdout.write("{}\n")
        return

    nm_dir = os.environ.get("NEURALMEMORY_DIR") or os.path.expanduser("~/.neuralmemory")
    buf_path = os.path.join(nm_dir, "tool_events.jsonl")

    event = {
        "tool_name": tool,
        "args_summary": json.dumps(data.get("tool_input", {}), default=str)[:200],
        "success": data.get("tool_error") is None,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", ""),
    }

    try:
        os.makedirs(nm_dir, exist_ok=True)
        with open(buf_path, "a", encoding="utf-8") as f:
            _lock(f)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                _unlock(f)
    except OSError:
        pass

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
