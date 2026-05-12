"""Light hook variant 4: empty fast-path (early exit).

Measures bare floor: parse stdin, exit. No file I/O.
Tells us the absolute minimum we can ever achieve.
"""

import json
import sys


def main() -> None:
    try:
        json.load(sys.stdin)
    except Exception:
        pass
    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
