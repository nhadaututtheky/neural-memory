"""NeuralMemory vs Vanilla Vector Search -- Qualitative Comparison.

Shows WHAT each system returns for the same queries, not just scores.
The difference: vector search returns the single best match.
NeuralMemory returns context -- related memories, connections, chains.

Usage:
    python benchmarks/nm_vs_rag.py
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
from collections import Counter

from neural_memory import Brain
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage


# -- Memories ---------------------------------------------

MEMORIES = [
    "Alice proposed using JWT for authentication during the Monday standup meeting.",
    "The team chose JWT over session cookies because we need stateless auth for microservices.",
    "Bob configured the JWT signing key with RS256 algorithm.",
    "The signing key expired on Tuesday -- nobody noticed until services failed.",
    "Production outage lasted 2 hours on Tuesday afternoon.",
    "Root cause: the expired signing key in the auth service.",
    "Alice deployed a hotfix -- rotated the key and restarted auth service.",
    "New policy: automated alerts 7 days before any credential expires.",
    "Users reported slow page loads on the orders dashboard.",
    "Profiling revealed a full table scan on the orders table for each request.",
    "Added a composite index on (customer_id, created_at) to fix the slow query.",
    "Page load time dropped from 3 seconds to 50 milliseconds.",
    "Alice is the tech lead -- owns security and authentication decisions.",
    "Bob is a backend engineer on Alice's platform team.",
    "The orders table references customers via customer_id foreign key.",
]

QUERIES = [
    ("What authentication method do we use?", "direct_fact"),
    ("Why did production go down on Tuesday?", "causal_chain"),
    ("How was the outage fixed?", "multi_hop"),
    ("How did we fix the slow dashboard?", "causal_chain"),
    ("What does Alice do?", "associative"),
]


# -- Vanilla vector search --------------------------------

def _tokenize(text: str) -> Counter[str]:
    words = re.findall(r"[a-zA-Z_]+", text.lower())
    return Counter(words)


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def vanilla_search(query: str, memories: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    query_vec = _tokenize(query)
    scored = [
        (mem, _cosine_similarity(query_vec, _tokenize(mem)))
        for mem in memories
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(mem, score) for mem, score in scored[:top_k] if score > 0]


# -- Colors -----------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
PURPLE = "\033[35m"
RED = "\033[31m"
GRAY = "\033[90m"


# -- Main -------------------------------------------------

async def main() -> None:
    # Setup NeuralMemory
    storage = InMemoryStorage()
    brain = Brain.create("benchmark")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage, brain.config)
    pipeline = ReflexPipeline(storage, brain.config)

    print(f"\n{BOLD}Encoding {len(MEMORIES)} memories...{RESET}\n")
    for mem in MEMORIES:
        try:
            await encoder.encode(mem)
        except Exception as exc:
            print(f"  {RED}Warning: {exc}{RESET}", file=sys.stderr)
    print(f"  {GREEN}Done.{RESET}\n")

    # Run comparison
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  NeuralMemory vs Vector Search -- Side-by-Side Comparison{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    for query, qtype in QUERIES:
        print(f"{CYAN}{BOLD}Q: {query}{RESET}")
        print(f"{DIM}   Type: {qtype}{RESET}\n")

        # Vector search
        v_results = vanilla_search(query, MEMORIES)
        print(f"  {YELLOW}{BOLD}Vector Search (top-3 cosine similarity):{RESET}")
        if v_results:
            for mem, score in v_results:
                short = mem[:75] + "..." if len(mem) > 75 else mem
                print(f"    {DIM}[{score:.2f}]{RESET} {short}")
        else:
            print(f"    {RED}(no results){RESET}")

        # NeuralMemory
        result = await pipeline.query(query, depth=DepthLevel.DEEP)
        print(f"\n  {GREEN}{BOLD}NeuralMemory (spreading activation):{RESET}")
        print(f"    {DIM}Neurons activated: {result.neurons_activated} | Confidence: {result.confidence:.2f}{RESET}")

        # Parse relevant memories from context
        if result.context:
            in_memories = False
            in_related = False
            mem_count = 0
            rel_count = 0
            for line in result.context.split("\n"):
                stripped = line.strip()
                if "Relevant Memories" in stripped:
                    in_memories = True
                    in_related = False
                    continue
                if "Related Information" in stripped:
                    in_memories = False
                    in_related = True
                    continue
                if stripped.startswith("- ") and in_memories and mem_count < 3:
                    text = stripped[2:].strip()
                    if len(text) > 15:
                        short = text[:75] + "..." if len(text) > 75 else text
                        print(f"    {GREEN}>{RESET} {short}")
                        mem_count += 1
                if stripped.startswith("- ") and in_related and rel_count < 3:
                    text = stripped[2:].strip()
                    # Skip single-word concept neurons
                    if len(text) > 20 and "[concept]" not in text.lower():
                        short = text[:75] + "..." if len(text) > 75 else text
                        print(f"    {DIM}  related: {short}{RESET}")
                        rel_count += 1
        else:
            print(f"    {RED}(no results){RESET}")

        print(f"\n{DIM}{'-' * 80}{RESET}\n")

    # Summary
    print(f"{BOLD}Key Differences:{RESET}\n")
    print(f"  {YELLOW}Vector Search{RESET}:  Returns the single best-matching text chunk.")
    print(f"                  Great for: 'Find me the doc that mentions X'")
    print(f"                  Fails at:  'Why did X happen?' (needs chain traversal)")
    print()
    print(f"  {GREEN}NeuralMemory{RESET}:   Returns activated context -- memories + connections.")
    print(f"                  Great for: Causal chains, temporal queries, associations")
    print(f"                  Graph grows smarter with more memories (consolidation)")
    print()
    print(f"  {DIM}15 memories | 5 queries | Baseline: bag-of-words cosine similarity{RESET}")
    print(f"  {DIM}NeuralMemory: ReflexPipeline depth=DEEP (spreading activation + causal){RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
