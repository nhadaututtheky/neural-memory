"""Light hook variant 3: same as v1 but uses pathlib (heavier import).

Measures pathlib import cost in cold-start path.
"""

import json
import os
import sys
from pathlib import Path


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

    nm_dir = Path(os.environ.get("NEURALMEMORY_DIR") or "") or (Path.home() / ".neuralmemory")
    buf_path = nm_dir / "tool_events.jsonl"

    event = {
        "tool_name": tool,
        "args_summary": json.dumps(data.get("tool_input", {}), default=str)[:200],
        "success": data.get("tool_error") is None,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", ""),
    }

    try:
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(buf_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
