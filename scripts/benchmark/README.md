# Benchmark Evidence

`run_evidence.py` is the canonical retrieval-evidence entry point. It selects one ordered LongMemEval instance set, runs every requested method against that set, and writes a versioned JSON artifact plus a Markdown view. The JSON retains the manifest, aggregate metrics, deterministic bootstrap intervals, per-type results, per-question rows, raw timing rows, and gate outcome.

This harness measures retrieval quality only. Do not describe its metrics as answer accuracy.

## Dependencies

Install the project and development tools before running the harness:

```bash
pip install -e ".[dev]"
```

The embedding baseline is optional unless `--require-embedding` is present:

```bash
pip install -e ".[dev,embeddings,pro]"
```

The `pro` extra supplies `hnswlib`, which canonical NM runs require to prove that ingest wrote searchable vectors. A model that loads successfully but cannot write/query its vector index fails the NM child instead of silently degrading to lexical-only retrieval.

LongMemEval data is cached under `scripts/benchmark/data/`. If a selected variant is absent, `huggingface_hub` is required so the existing data loader can download it.

## Smoke Run

The deterministic five-instance smoke validates offline baseline execution and artifact generation without an LLM reader or judge:

```bash
python scripts/benchmark/run_evidence.py --variant s --limit 5 --methods naive,fts5,recency --retrieval-only --output-dir scripts/benchmark/results/smoke
```

This smoke is intentionally non-canonical because it omits NM and embedding and does not use the pinned 50-instance set. It saves a failed gate result for inspection but exits successfully unless `--enforce-gate` is added.

## Canonical Run

Run the pinned LongMemEval-S sample with all required methods from a clean Git checkout:

```bash
python scripts/benchmark/run_evidence.py --variant s --instance-ids scripts/benchmark/mini_bench_ids.json --methods nm,naive,fts5,embedding --require-embedding --retrieval-only --enforce-gate --output-dir scripts/benchmark/results/evidence
```

A canonical artifact requires all of the following:

- Exact ordered IDs from `mini_bench_ids.json`.
- NM, naive, FTS5, and embedding in the same run.
- Available Git metadata and a clean working tree.
- The pinned corpus SHA-256 and evidence schema.
- NM Recall@5 and NDCG@5 above the absolute floors.
- No regression from `evidence_baseline.json` and a strict win over each same-run baseline on both gated metrics.

Use `--warmup-runs N` to execute `N` untimed one-instance warmups per method. The manifest records the value. Use `--top-k N` to apply one cutoff consistently across all methods.

## Outputs

Each run writes these immutable artifacts under `--output-dir`:

```text
evidence_v1_<timestamp>.json
evidence_v1_<timestamp>.md
```

`canonical: false` is accompanied by `non_canonical_reasons`. Common markers include `method_not_run:<method>`, `embedding_unavailable`, `git_worktree_dirty`, and `git_metadata_unknown`. `--enforce-gate` saves both artifacts first, then exits nonzero and prints every gate failure.

The pinned last-good contract lives in `scripts/benchmark/evidence_baseline.json`. Its bundled source artifact is a quality-only legacy regression anchor: it proves the pinned rows and metrics, but it predates runtime provenance capture, so the baseline explicitly lists every unverified field in `legacy_metadata_missing`. Current canonical runs must prove their own clean Git state, runtime config, corpus hash, ground truth, model revision, and same-run comparisons. Updating the anchor requires an explicit baseline migration with a new verified source artifact, corpus hash, ordered ID set, metrics, and tests; never edit only the threshold values to make a regression pass.
