"""Light hook variant 1: absolute minimal stdlib imports.

Goal: < 80ms cold spawn.
Strategy: read stdin, parse JSON, append line, return. Zero file path validation.
"""

import json
import os
import sys


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
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
