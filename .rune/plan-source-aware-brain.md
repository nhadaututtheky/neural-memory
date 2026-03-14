# Feature: Source-Aware Brain

## Overview
Transform NM from "memory that stores text" to "smart index that knows what's where." Brain neurons act as semantic index entries pointing to source locations. When exact quotes are needed, system fetches from source documents — not from neuron content.

**Business driver**: SMB chatbots for law/accounting firms need exact citations (page, paragraph, file) to be trustworthy. Brain recalls the concept; source lookup provides the evidence.

## Phases
| # | Name | Status | Plan File | Summary |
|---|------|--------|-----------|---------|
| 1 | Source Locators in Training | ⬚ Pending | plan-source-aware-brain-phase1.md | Enrich DocChunk + neuron metadata with page/offset, auto-create Source + SOURCE_OF |
| 2 | Citation Tool | ⬚ Pending | plan-source-aware-brain-phase2.md | `nmem_cite` MCP tool, SourceResolver protocol, LocalResolver, staleness detection |
| 3 | Source Refresh | ⬚ Pending | plan-source-aware-brain-phase3.md | `nmem_train action="refresh"`, stale neuron marking, optional retrain |
| 4 | Cloud Resolvers | ⬚ Pending | plan-source-aware-brain-phase4.md | S3Resolver, GDriveResolver as optional extras |

## Architecture

```
Training:  file → extract (page markers) → chunk (with locators) → encode → neurons + Source + SOURCE_OF
Recall:    query → spreading activation → relevant neurons (fast, <200ms)
Citation:  neuron_ids → source_locator metadata → SourceResolver → exact text from file
Refresh:   scan sources → hash compare → mark stale → optional retrain
```

## Key Decisions
- `nmem_cite` is a SEPARATE tool from `nmem_recall` — keeps recall fast, citation is optional I/O
- No file watcher daemon — use `refresh` command + external cron/trigger
- Cloud resolvers are optional extras (`neural-memory[cloud-s3]`), not core dependencies
- Source locators go in neuron `metadata` JSON — no schema migration needed
- Only `LocalResolver` for MVP — add cloud when customer needs it
- Do NOT build: OCR, document editor, version diffing, real-time sync

## What Already Exists (70% done)
- `Source` dataclass with `SourceType`, `SourceStatus`, `file_hash`, `metadata`
- `SOURCE_OF` synapse type (defined, used in `nmem_remember`, but NOT in `nmem_train`)
- `nmem_source` + `nmem_provenance` MCP tools
- `training_files` table with file hash dedup
- `DocChunk` stores `source_file`, `line_start`, `line_end`, `heading_path`
- `citation.py` engine with INLINE/FOOTNOTE/FULL formats + domain templates (law, ledger)
- Citation building in recall/show/recap handlers (reads SOURCE_OF synapses)

## What's Missing (the 30%)
- DocChunk lacks `page_number`, `char_offset_start`, `char_offset_end`
- PDF extraction loses page boundaries (pymupdf4llm → markdown drops page info)
- `nmem_train` does NOT create Source records or SOURCE_OF synapses
- No SourceResolver protocol for fetching exact text from files
- No `nmem_cite` tool for on-demand source dereference
- No `refresh` action to detect source changes
