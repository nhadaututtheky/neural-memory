"""NeuralMemory demo script — run this to generate a terminal recording.

Usage:
    # Record with asciinema:
    asciinema rec demo.cast -c "python docs/demo/demo_script.py"

    # Or record with terminalizer:
    terminalizer record demo -c "python docs/demo/demo_script.py"

    # Or just run it to see the demo:
    python docs/demo/demo_script.py
"""

from __future__ import annotations

import asyncio
import sys
import time

# ── Terminal colors ──────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
PURPLE = "\033[35m"
GRAY = "\033[90m"


def typed(text: str, delay: float = 0.03) -> None:
    """Simulate typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def prompt(cmd: str, delay: float = 0.03) -> None:
    """Print a terminal prompt with typing effect."""
    sys.stdout.write(f"{GREEN}{BOLD}${RESET} ")
    sys.stdout.flush()
    time.sleep(0.3)
    typed(cmd, delay)


def output(text: str) -> None:
    """Print command output."""
    print(text)


def section(title: str) -> None:
    """Print section header."""
    print()
    print(f"{PURPLE}{BOLD}{'─' * 50}{RESET}")
    print(f"{PURPLE}{BOLD}  {title}{RESET}")
    print(f"{PURPLE}{BOLD}{'─' * 50}{RESET}")
    print()
    time.sleep(0.5)


def pause(seconds: float = 1.0) -> None:
    """Pause for effect."""
    time.sleep(seconds)


# ── Demo flow ────────────────────────────────────────────


async def main() -> None:
    """Run the NeuralMemory demo."""
    print()
    print(f"{BOLD}  NeuralMemory Demo{RESET}")
    print(f"{DIM}  Reflex-based memory for AI agents{RESET}")
    pause(1.5)

    # ── 1. Install ──
    section("1. Install")
    prompt("pip install neural-memory", delay=0.02)
    output(f"{DIM}Successfully installed neural-memory-1.6.0{RESET}")
    pause(1)

    # ── 2. Store memories ──
    section("2. Store Memories")

    prompt('nmem remember "Fixed auth bug with null check in login.py:42"')
    pause(0.3)
    output(f"  {GREEN}Encoded{RESET} | neurons: 4 | synapses: 3 | fiber: f-a1b2c3")
    pause(0.8)

    prompt('nmem remember "We decided to use PostgreSQL over MongoDB" --type decision')
    pause(0.3)
    output(f"  {GREEN}Encoded{RESET} | neurons: 3 | synapses: 2 | fiber: f-d4e5f6")
    pause(0.8)

    prompt('nmem remember "Alice suggested JWT for auth, caused Tuesday outage"')
    pause(0.3)
    output(f"  {GREEN}Encoded{RESET} | neurons: 5 | synapses: 6 | fiber: f-g7h8i9")
    pause(0.8)

    prompt('nmem todo "Review PR #123" --priority 7')
    pause(0.3)
    output(f"  {GREEN}TODO{RESET} | priority: 7 | expires: 30 days")
    pause(1)

    # ── 3. Recall ──
    section("3. Recall Through Association")

    prompt('nmem recall "auth bug fix"')
    pause(0.5)
    output(f"""
  {CYAN}Query:{RESET} auth bug fix
  {CYAN}Depth:{RESET} 1 (context)
  {CYAN}Neurons activated:{RESET} 6
  {CYAN}Confidence:{RESET} 0.87

  {BOLD}Result:{RESET}
  Fixed auth bug with null check in login.py:42
  {GRAY}(fact, priority 5, created 2 min ago){RESET}

  {DIM}Related:{RESET}
  {GRAY}  ← CAUSED_BY ← JWT auth config{RESET}
  {GRAY}  ← INVOLVES  ← Alice{RESET}
""")
    pause(1.5)

    # ── 4. Causal chain ──
    section("4. Multi-Hop Causal Chain")

    prompt('nmem recall "why did the outage happen" --depth 3')
    pause(0.5)
    output(f"""
  {CYAN}Query:{RESET} why did the outage happen
  {CYAN}Depth:{RESET} 3 (deep)
  {CYAN}Neurons activated:{RESET} 12
  {CYAN}Confidence:{RESET} 0.82

  {BOLD}Causal chain:{RESET}
  {YELLOW}outage{RESET} ← {PURPLE}CAUSED_BY{RESET} ← {YELLOW}JWT config{RESET} ← {PURPLE}SUGGESTED_BY{RESET} ← {YELLOW}Alice{RESET}
                                       ↑
                              {PURPLE}HAPPENED_AT{RESET} → {CYAN}Tuesday{RESET}

  {BOLD}Context:{RESET}
  Alice suggested JWT for auth, caused Tuesday outage.
  JWT configuration issue led to service disruption.
  {GRAY}(traced 3 hops through causal synapses){RESET}
""")
    pause(2)

    # ── 5. Brain stats ──
    section("5. Brain Health")

    prompt("nmem stats")
    pause(0.3)
    output(f"""
  {BOLD}Brain:{RESET} default
  {BOLD}Neurons:{RESET}     15
  {BOLD}Synapses:{RESET}    23
  {BOLD}Fibers:{RESET}      4
  {BOLD}Health:{RESET}       {GREEN}A- (91/100){RESET}
  {BOLD}Purity:{RESET}      {GREEN}96%{RESET}
  {BOLD}Freshness:{RESET}   {GREEN}100%{RESET} (all < 30 days)
""")
    pause(1)

    # ── 6. DB-to-Brain ──
    section("6. DB-to-Brain Training")

    prompt('nmem train-db sqlite:///myapp.db --domain ecommerce')
    pause(0.5)
    output(f"""
  {CYAN}Introspecting schema...{RESET}
  {GREEN}Tables found:{RESET}     12
  {GREEN}FK relationships:{RESET} 8
  {GREEN}Patterns:{RESET}        5 (audit_trail, soft_delete, tree_hierarchy, ...)

  {BOLD}Training result:{RESET}
    Tables processed:      12
    Neurons created:       28
    Synapses created:      35
    Schema fingerprint:    a1b2c3d4e5f6g7h8

  {GREEN}Brain now understands your database structure.{RESET}
""")
    pause(1.5)

    prompt('nmem recall "how are orders and customers related"')
    pause(0.5)
    output(f"""
  {BOLD}Result:{RESET}
  Table 'orders' stores order records. Links to: customers, products.
  {GRAY}  orders.customer_id → customers.id  (INVOLVES, confidence: 0.75){RESET}
  {GRAY}  orders.product_id  → products.id   (RELATED_TO, confidence: 0.60){RESET}

  {DIM}Pattern: orders uses audit trail (created_at + updated_at){RESET}
""")
    pause(2)

    # ── Done ──
    print()
    print(f"{GREEN}{BOLD}{'─' * 50}{RESET}")
    print(f"{GREEN}{BOLD}  That's NeuralMemory.{RESET}")
    print(f"{DIM}  pip install neural-memory{RESET}")
    print(f"{DIM}  github.com/nhadaututtheky/neural-memory{RESET}")
    print(f"{GREEN}{BOLD}{'─' * 50}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
