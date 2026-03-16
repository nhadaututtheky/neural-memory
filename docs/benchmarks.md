# Benchmarks


Last updated: **2026-03-16**


## NeuralMemory vs Mem0 — Competitive Benchmark

Head-to-head comparison on real-world AI agent memory tasks. 50 diverse memories (decisions, errors, workflows, preferences), 20 recall queries, multi-hop reasoning, and conversation context.

!!! info "Test Setup"
    - **NeuralMemory**: SQLiteStorage, spreading activation, zero external APIs
    - **Mem0 v1.0.3**: Qdrant local, HuggingFace embeddings (all-MiniLM-L6-v2), Qwen LLM via DashScope
    - **Platform**: Windows 11, Python 3.14, single-threaded async
    - **Script**: `scripts/benchmark_mem0_vs_nm.py`

### Speed

| Operation | NeuralMemory | Mem0 | Speedup |
|-----------|:------------:|:----:|:-------:|
| **Write 50 memories** | **1.22s** | 148.16s | **121x faster** |
| **Read 20 queries** | **1.80s** | 2.89s | **1.6x faster** |
| **Conversation (10 turns + 5 recalls)** | **1.25s** | 11.99s | **9.6x faster** |

Mem0's write bottleneck: every `add()` call triggers an LLM call to extract and summarize the memory. NeuralMemory encodes directly into neural structures — no LLM needed.

### Accuracy

| Metric | NeuralMemory | Mem0 | Winner |
|--------|:------------:|:----:|:------:|
| **Semantic accuracy** (Jaccard) | 0.141 | 0.141 | Tie |
| **Multi-hop reasoning** (keyword coverage) | **0.417** | 0.383 | NeuralMemory |
| **Conversation context** | 0.174 | 0.174 | Tie |

Equal accuracy on direct recall, but NeuralMemory outperforms on multi-hop queries thanks to graph-based spreading activation — connections between memories are first-class citizens, not afterthoughts.

### Cost

| Metric | NeuralMemory | Mem0 |
|--------|:------------:|:----:|
| **External API calls** | **0** | **70** |
| **LLM calls per add()** | 0 | 1 |
| **LLM calls per search()** | 0 | 1 |
| **Estimated cost at 10K ops/day** | **$0.00** | ~$2-5/day |

At scale, the cost difference is decisive. Mem0 requires an LLM call for every operation — NeuralMemory's retrieval is purely algorithmic.

### Verdict

```
NeuralMemory wins : 4 / 6 categories
Mem0 wins         : 0 / 6 categories
Ties              : 2 / 6 categories
```

**NeuralMemory is 121x faster on writes, equally accurate, and costs $0 in API calls.**
The tradeoff: Mem0 can leverage LLM intelligence for memory extraction (useful when input is unstructured), while NeuralMemory relies on its own extraction pipeline.

### Reproduce

```bash
pip install mem0ai sentence-transformers
DASHSCOPE_API_KEY=your-key python scripts/benchmark_mem0_vs_nm.py
```

---

## NeuralMemory vs Cognee — Competitive Benchmark

Head-to-head comparison using the same test suite: 50 diverse memories, 20 recall queries, multi-hop reasoning, and conversation context.

!!! info "Test Setup"
    - **NeuralMemory v4.7.0**: SQLiteStorage, spreading activation, zero external APIs
    - **Cognee v0.5.5**: KuzuDB graph + LanceDB vectors + fastembed embeddings, Qwen LLM via DashScope for cognify/search
    - **Platform**: Windows 11, Python 3.12, single-threaded async
    - **Script**: `scripts/benchmark_cognee_vs_nm.py`

### Speed

| Operation | NeuralMemory | Cognee | Speedup |
|-----------|:------------:|:------:|:-------:|
| **Write 50 memories** | **3.62s** | 290.63s | **80x faster** |
| **Read 20 queries** | **1.88s** | 34.56s | **18x faster** |
| **Conversation (10 turns + 5 recalls)** | **1.85s** | 88.78s | **48x faster** |

Cognee's write bottleneck: every `add()` + `cognify()` cycle triggers multiple LLM calls for entity extraction, relationship mining, and knowledge graph construction. NeuralMemory encodes directly into neural structures — no LLM needed.

### Accuracy

| Metric | NeuralMemory | Cognee | Winner |
|--------|:------------:|:------:|:------:|
| **Semantic accuracy** (Jaccard) | 0.141 | 0.180 | Cognee |
| **Multi-hop reasoning** (keyword coverage) | 0.417 | 0.633 | Cognee |
| **Conversation context** | 0.174 | 0.248 | Cognee |

Cognee's LLM-powered knowledge graph produces richer semantic connections — entity extraction creates explicit relationships that improve multi-hop reasoning. The accuracy gap comes at significant cost in speed and API usage.

### Cost

| Metric | NeuralMemory | Cognee |
|--------|:------------:|:------:|
| **External API calls** | **0** | **149** |
| **LLM calls per add()+cognify()** | 0 | ~2 |
| **LLM calls per search()** | 0 | 1 |
| **Estimated cost at 10K ops/day** | **$0.00** | ~$5-15/day |

Cognee requires LLM calls for both ingestion (entity extraction via `cognify()`) and retrieval (query parsing via `search()`). At scale, the cost compounds rapidly.

### Verdict

```
NeuralMemory wins : 3 / 6 categories (speed × 3)
Cognee wins       : 3 / 6 categories (accuracy × 3)
```

**The tradeoff is clear**: NeuralMemory dominates speed (80x write, 18x read, 48x conversation) and cost ($0 vs 149 API calls). Cognee wins on accuracy through LLM-powered knowledge graph construction — but at 80x the latency and significant per-operation cost.

For real-time AI agent workflows where sub-second response matters, NeuralMemory is the clear choice. For offline knowledge base construction where accuracy is paramount and cost/latency are acceptable, Cognee's approach has merit.

### Reproduce

```bash
pip install cognee neural-memory
DASHSCOPE_API_KEY=your-key python scripts/benchmark_cognee_vs_nm.py
```

---

## Internal Benchmarks

Generated by `benchmarks/run_benchmarks.py`.


## Activation Engine


Compares three activation modes on synthetic graphs with overlapping fiber pathways:


- **Classic**: BFS spreading activation with distance-based decay

- **Reflex**: Trail-based activation through fiber pathways only

- **Hybrid**: Reflex primary + limited classic BFS for discovery (default in v0.6.0+)


| Neurons | Fibers | Classic (ms) | Reflex (ms) | Hybrid (ms) | Classic # | Reflex # | Hybrid # | Reflex Recall | Hybrid Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 10 | 2.35 | 0.03 | 0.75 | 85 | 16 | 66 | 16.5% | 75.3% |
| 500 | 50 | 6.13 | 0.05 | 0.91 | 231 | 38 | 155 | 8.7% | 59.3% |
| 1000 | 100 | 4.57 | 0.03 | 0.74 | 190 | 29 | 126 | 3.7% | 54.7% |
| 3000 | 300 | 7.81 | 0.06 | 0.88 | 242 | 52 | 166 | 2.5% | 49.6% |
| 5000 | 500 | 4.53 | 0.13 | 0.67 | 171 | 151 | 232 | 3.5% | 50.9% |


### Speedup


| Graph Size | Classic vs Hybrid | Classic vs Reflex |
| --- | --- | --- |
| 100 | 3.1x | 78.3x |
| 500 | 6.7x | 122.6x |
| 1000 | 6.2x | 152.3x |
| 3000 | 8.9x | 130.2x |
| 5000 | 6.8x | 34.8x |


**Average recall** -- Reflex only: 7.0% | Hybrid: 58.0%


## Full Pipeline


End-to-end benchmark: 15 encoded memories, 5 queries, 10 runs each.


| Query | Depth | Classic (ms) | Hybrid (ms) | Speedup | C-Neurons | H-Neurons | C-Conf | H-Conf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| What did Alice suggest? | INSTANT | 1.3 | 5.09 | 0.3x | 16 | 13 | 1.0 | 1.0 |
| What was the auth bug fix? | INSTANT | 1.05 | 2.95 | 0.4x | 15 | 12 | 1.0 | 1.0 |
| What happened on Thursday? | CONTEXT | 1.33 | 1.7 | 0.8x | 8 | 8 | 1.0 | 1.0 |
| Why did we choose PostgreSQL? | DEEP | 2.24 | 3.18 | 0.7x | 10 | 10 | 1.0 | 1.0 |
| What is Bob working on? | CONTEXT | 2.1 | 3.45 | 0.6x | 10 | 10 | 1.0 | 1.0 |
| **Total** |  | **8.02** | **16.37** | **0.5x** |  |  |  |  |

## Ground-Truth Evaluation


30 curated memories, 25 queries, K=5.


### Overall (NeuralMemory vs Naive Baseline)


| Metric | NeuralMemory | Naive Baseline | Winner |
| --- | --- | --- | --- |
| Precision@5 | 0.168 | 0.248 | **Baseline** |
| Recall@5 | 0.380 | 0.466 | **Baseline** |
| MRR | 0.563 | 0.637 | **Baseline** |
| NDCG@5 | 0.350 | 0.464 | **Baseline** |


### Per-Category Recall


| Category | NeuralMemory | Baseline | Count |
| --- | --- | --- | --- |
| causal | 0.375 | 0.500 | 4 |
| coherence | 0.244 | 0.378 | 3 |
| factual | 0.556 | 0.819 | 8 |
| pattern | 0.237 | 0.304 | 4 |
| temporal | 0.312 | 0.125 | 6 |


## Methodology


- **Platform**: InMemoryStorage (NetworkX), single-threaded async
- **Runs**: 10 per measurement (median reported)
- **Warmup**: 1 warmup run excluded from timing
- **Hybrid strategy**: Reflex trail activation (primary) + classic BFS with `max_hops // 2` (discovery, dampened 0.6x)
- **Seed**: `random.seed(42)` for reproducibility

### Regenerate

```bash
python benchmarks/run_benchmarks.py
```

Results are written to `docs/benchmarks.md`.

## SQLite at Scale


Last updated: **2026-03-04 02:24**


Real SQLiteStorage benchmarks with diverse memory types on Windows 11.


### Encode Throughput


| Memories | Total (s) | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (mem/s) | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 26.5 | 26.52 | 22.59 | 51.33 | 66.75 | 37.7 | 0 |
| 5,000 | 190.8 | 38.16 | 34.65 | 76.36 | 99.64 | 26.2 | 0 |
| 10,000 | 536.1 | 53.61 | 47.9 | 102.27 | 131.85 | 18.7 | 0 |
| 50,000 | 10954.6 | 219.09 | 191.25 | 509.01 | 656.49 | 4.6 | 0 |


### Database Size


| Memories | After Encode (MB) | After Consolidation (MB) | Neurons | Synapses | Fibers |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 11.2 | 12.6 | 3,534 | 7,784 | 1,000 |
| 5,000 | 46.55 | 48.67 | 13,734 | 34,238 | 5,000 |
| 10,000 | 88.29 | 93.07 | 25,033 | 65,789 | 10,000 |
| 50,000 | 411.48 | 419.0 | 108,913 | 311,777 | 50,000 |


### Recall Latency (Post-Consolidation)


10 queries, 5 runs each (median reported).



#### 1,000 memories


| Query | Depth | Median (ms) | P95 (ms) | Neurons | Confidence | Found |
| --- | --- | --- | --- | --- | --- | --- |
| Python concurrency | INSTANT | 145.12 | 154.46 | 15 | 1.0 | yes |
| What database did we choose? | CONTEXT | 2.08 | 2.36 | 0 | 0.0 | no |
| connection error Redis | INSTANT | 109.15 | 121.76 | 21 | 1.0 | yes |
| deployment workflow | CONTEXT | 112.8 | 136.68 | 23 | 1.0 | yes |
| Why did we choose PostgreSQL? | DEEP | 38.02 | 40.82 | 7 | 1.0 | yes |
| authentication JWT | INSTANT | 70.4 | 94.68 | 15 | 1.0 | yes |
| What patterns were discovered? | CONTEXT | 18.83 | 21.54 | 8 | 1.0 | yes |
| machine learning integration | DEEP | 132.76 | 164.06 | 20 | 1.0 | yes |
| rate limiting implementation | INSTANT | 125.62 | 153.83 | 24 | 1.0 | yes |
| TODO before release | CONTEXT | 137.43 | 181.73 | 15 | 1.0 | yes |
| **Average** |  | **89.22** | **107.19** | **14.8** |  |  |


#### 5,000 memories


| Query | Depth | Median (ms) | P95 (ms) | Neurons | Confidence | Found |
| --- | --- | --- | --- | --- | --- | --- |
| Python concurrency | INSTANT | 117.43 | 160.7 | 19 | 1.0 | yes |
| What database did we choose? | CONTEXT | 1.73 | 2.15 | 0 | 0.0 | no |
| connection error Redis | INSTANT | 169.82 | 170.16 | 23 | 1.0 | yes |
| deployment workflow | CONTEXT | 169.55 | 198.36 | 23 | 1.0 | yes |
| Why did we choose PostgreSQL? | DEEP | 77.99 | 106.03 | 7 | 1.0 | yes |
| authentication JWT | INSTANT | 109.31 | 191.21 | 19 | 1.0 | yes |
| What patterns were discovered? | CONTEXT | 43.49 | 50.35 | 8 | 1.0 | yes |
| machine learning integration | DEEP | 83.03 | 124.42 | 22 | 1.0 | yes |
| rate limiting implementation | INSTANT | 126.62 | 166.48 | 26 | 1.0 | yes |
| TODO before release | CONTEXT | 199.36 | 211.71 | 19 | 1.0 | yes |
| **Average** |  | **109.83** | **138.16** | **16.6** |  |  |


#### 10,000 memories


| Query | Depth | Median (ms) | P95 (ms) | Neurons | Confidence | Found |
| --- | --- | --- | --- | --- | --- | --- |
| Python concurrency | INSTANT | 96.55 | 144.66 | 21 | 1.0 | yes |
| What database did we choose? | CONTEXT | 1.99 | 2.35 | 0 | 0.0 | no |
| connection error Redis | INSTANT | 156.88 | 174.88 | 26 | 1.0 | yes |
| deployment workflow | CONTEXT | 169.16 | 209.46 | 22 | 1.0 | yes |
| Why did we choose PostgreSQL? | DEEP | 75.2 | 89.14 | 7 | 1.0 | yes |
| authentication JWT | INSTANT | 116.5 | 143.92 | 19 | 1.0 | yes |
| What patterns were discovered? | CONTEXT | 49.92 | 58.23 | 8 | 1.0 | yes |
| machine learning integration | DEEP | 91.53 | 126.03 | 21 | 1.0 | yes |
| rate limiting implementation | INSTANT | 162.43 | 168.47 | 27 | 1.0 | yes |
| TODO before release | CONTEXT | 217.67 | 237.86 | 19 | 1.0 | yes |
| **Average** |  | **113.78** | **135.5** | **17** |  |  |


#### 50,000 memories


| Query | Depth | Median (ms) | P95 (ms) | Neurons | Confidence | Found |
| --- | --- | --- | --- | --- | --- | --- |
| Python concurrency | INSTANT | 190.35 | 207.35 | 21 | 1.0 | yes |
| What database did we choose? | CONTEXT | 2.36 | 3.44 | 0 | 0.0 | no |
| connection error Redis | INSTANT | 224.34 | 252.6 | 26 | 1.0 | yes |
| deployment workflow | CONTEXT | 207.73 | 235.62 | 23 | 1.0 | yes |
| Why did we choose PostgreSQL? | DEEP | 172.12 | 211.13 | 10 | 1.0 | yes |
| authentication JWT | INSTANT | 183.04 | 213.52 | 19 | 1.0 | yes |
| What patterns were discovered? | CONTEXT | 118.83 | 147.36 | 8 | 1.0 | yes |
| machine learning integration | DEEP | 168.41 | 174.37 | 21 | 1.0 | yes |
| rate limiting implementation | INSTANT | 227.81 | 286.0 | 27 | 1.0 | yes |
| TODO before release | CONTEXT | 297.5 | 331.74 | 19 | 1.0 | yes |
| **Average** |  | **179.25** | **206.31** | **17.4** |  |  |


### Consolidation Performance


| Memories | Duration (s) | Synapses Pruned | Neurons Pruned | Fibers Merged | Synapses Enriched |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 2.4 | 0 | 0 | 0 | 3 |
| 5,000 | 3.8 | 0 | 0 | 0 | 6 |
| 10,000 | 7.8 | 0 | 0 | 0 | 5 |
| 50,000 | 8.9 | 0 | 0 | 0 | 4 |


### Health Diagnostics


| Memories | Phase | Grade | Purity | Connectivity | Diversity | Freshness | Orphan Rate | Warnings | Diagnostics (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | Pre | D | 42.9 | 0.232 | 0.493 | 1.0 | 0.0 | 2 | 329.8 |
| 1,000 | Post | D | 44.6 | 0.268 | 0.531 | 1.0 | 0.0 | 2 | 261.5 |
| 5,000 | Pre | F | 36.6 | 0.319 | 0.409 | 1.0 | 0.672 | 3 | 449.7 |
| 5,000 | Post | F | 38.4 | 0.354 | 0.455 | 1.0 | 0.674 | 3 | 404.9 |
| 10,000 | Pre | F | 35.5 | 0.364 | 0.373 | 1.0 | 0.82 | 3 | 487.0 |
| 10,000 | Post | F | 38.4 | 0.434 | 0.437 | 1.0 | 0.821 | 3 | 488.4 |
| 50,000 | Pre | F | 34.8 | 0.449 | 0.305 | 1.0 | 0.959 | 3 | 650.9 |
| 50,000 | Post | F | 36.3 | 0.479 | 0.346 | 1.0 | 0.959 | 3 | 629.4 |


### Methodology


- **Storage**: Real SQLiteStorage (aiosqlite, WAL mode)
- **Platform**: Windows 11, single-threaded async
- **Memory types**: 7 types (fact, decision, error, insight, todo, workflow, context)
- **Content**: Diverse generated content from 50 topics × 16 actions × 26 features
- **Recall runs**: 5 per query (median reported)
- **Seed**: `random.seed(42)` for reproducibility
