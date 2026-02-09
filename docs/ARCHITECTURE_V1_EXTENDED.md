# Extended Neural Memory Architecture (v1.0 Vision)

> [!IMPORTANT]
> This architecture replaces the previous "3-layered brain" proposal (NM + Cognee + Mem0). Technical analysis shows that Neural Memory already covers the functional breadth of those layers with significantly less overhead and no forced LLM dependency.

## 1. The Core Philosophy: "One Strong Brain > Three Weak Ones"
Instead of distributing memory across three separate systems with redundant data and complex routing, we expand the existing Neural Memory system to cover the missing "semantic" and "complex relation" gaps.

## 2. Architectural Pillars

### A. Reflex Core (Existing)
*   **Mechanism**: Spreading Activation, Hebbian Learning, Reflex Pipelines.
*   **Infrastructure**: SQLite (Single Source of Truth).
*   **Cost**: Zero LLM, Zero API, Zero Latency (relative to LLM).

### B. Knowledge & personality (Existing)
*   **Structure**: 29+ Synapse Types (Causal, Temporal, Emotional).
*   **Features**: Habit Learning, Dream Consolidation, Memory Types (Preference, Instruction).

### C. Semantic Bridge (NEW/Optional)
*   **Goal**: Solve the "auth" ↔ "authentication" gap without prior co-activation history.
*   **Implementation**: Optional **Vector Embedding Layer** using `sentence-transformers` (Local).
*   **Advantage**: Zero API cost, integrates directly into the Spreading Activation seed selection.

### D. Cognitive Enrichment (NEW/Optional)
*   **Goal**: Extract complex relationships that Regex (v0.14) cannot catch.
*   **Implementation**: Optional **LLM Extraction Pipeline** (Ollama local or API).
*   **Advantage**: Use LLM as an *enhancement* step during encoding, not as a core storage dependency.

## 3. Why This Wins vs. Cognee/Mem0
1.  **Zero-LLM Priority**: Maintains local performance and privacy.
2.  **Infrastructure Simplicity**: 1 DB (SQLite) vs 6+ services.
3.  **Conflict Resolution**: Handled through Neural Memory's native `CONTRADICTS` synapses and weight reduction logic.
4.  **No Data Fragmentation**: Single source of truth for all memory types.

## 4. Implementation Roadmap
1.  **Phase 1**: Optimize v0.20 Habits & Workflows (Current).
2.  **Phase 2**: Add Local Vector Embedding support (Semantic Search).
3.  **Phase 3**: Optional LLM Integration for deep relation extraction.
