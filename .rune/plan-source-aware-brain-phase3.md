# Phase 3: Source Refresh

## Goal
Command to scan registered sources for changes, mark stale neurons, and optionally retrain affected files. No daemon — triggered on demand or via external cron.

## Tasks
- [ ] 3.1 Add `action="refresh"` to `nmem_train` handler
- [ ] 3.2 Implement source scan: iterate Sources → re-hash files → compare with stored hash
- [ ] 3.3 Mark stale neurons: add `source_stale=true` to metadata of affected neurons
- [ ] 3.4 Implement `--retrain` flag: re-chunk and re-encode changed files
- [ ] 3.5 Handle missing files: mark Source status as SUPERSEDED
- [ ] 3.6 Handle new files in directory: detect unregistered files, suggest training
- [ ] 3.7 Report: stale_neurons, retrained, missing_files, new_files counts
- [ ] 3.8 Write tests: changed file detection, stale marking, retrain flow, missing file

## nmem_train refresh Interface
```
Input:  nmem_train(action="refresh", path="/data/legal/", retrain=true)
Output:
{
  "sources_scanned": 15,
  "sources_changed": 2,
  "sources_missing": 1,
  "new_files_found": 3,
  "stale_neurons_marked": 24,
  "neurons_retrained": 18,
  "message": "2 sources changed, 1 missing. 24 stale neurons marked, 18 retrained."
}
```

## Refresh Flow
1. Get all Sources with `resolver: "local"` in current brain
2. For each Source, re-hash file at stored path
3. If hash differs → find linked neurons (via SOURCE_OF) → mark stale
4. If `retrain=true` → delete old neurons for changed file → re-chunk → re-encode
5. If file missing → mark Source as SUPERSEDED
6. Scan directory for files not yet registered → report as new_files_found

## Acceptance Criteria
- [ ] Changed files detected correctly via SHA-256 comparison
- [ ] Stale neurons flagged in metadata (not deleted)
- [ ] Retrain produces new neurons with updated source_hash
- [ ] Missing files don't crash — graceful SUPERSEDED status
- [ ] Unchanged files skipped entirely (no re-processing)

## Files Touched
- `src/neural_memory/mcp/train_handler.py` — add refresh action
- `src/neural_memory/engine/doc_trainer.py` — add refresh logic
- `src/neural_memory/storage/sqlite_sources.py` — add bulk stale marking helper
- `tests/unit/test_train_refresh.py` — NEW: refresh tests

## Dependencies
- Requires Phase 1 (source locators) and Phase 2 (SourceResolver for hash computation)
