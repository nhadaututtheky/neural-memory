#!/usr/bin/env python3
"""Stop hook — reminds the user to capture session learnings before exit.

This hook fires when a Claude Code session ends. It outputs a reminder
suggesting the user run nmem_auto to capture any important decisions,
errors, or insights from the session.

No dependencies beyond Python stdlib. Reads from stdin (stop hook input),
writes reminder to stdout.
"""

import json
import sys


def main() -> None:
    # Read stop hook input (JSON on stdin)
    try:
        raw = sys.stdin.read()
        _hook_input = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, OSError):
        _hook_input = {}

    # Output reminder message
    reminder = (
        "\n"
        "NeuralMemory: Session ending.\n"
        "Consider running nmem_auto(action='process') to capture session learnings.\n"
        "This saves decisions, errors, and insights from this session automatically.\n"
    )
    print(reminder)


if __name__ == "__main__":
    main()
