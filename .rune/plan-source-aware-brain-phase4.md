# Phase 4: Cloud Resolvers (Post-MVP)

## Goal
Add S3 and GDrive resolvers as optional extras so source documents can live in cloud storage. Customer-driven — only build when a paying customer needs it.

## Tasks
- [ ] 4.1 Create `engine/resolvers/` package with `__init__.py` + resolver registry
- [ ] 4.2 Move `LocalResolver` into `engine/resolvers/local.py`
- [ ] 4.3 Implement `S3Resolver` in `engine/resolvers/s3.py` (requires boto3)
- [ ] 4.4 Implement `GDriveResolver` in `engine/resolvers/gdrive.py` (requires google-api-python-client)
- [ ] 4.5 Add optional extras: `neural-memory[cloud-s3]`, `neural-memory[cloud-gdrive]`
- [ ] 4.6 Add resolver config in Source metadata: `{"resolver": "s3", "bucket": "..."}`
- [ ] 4.7 Write tests with mocked cloud clients
- [ ] 4.8 Document cloud source setup in docs/guides/

## Resolver Registry
```python
_RESOLVERS: dict[str, type[SourceResolver]] = {
    "local": LocalResolver,
}

def register_resolver(name: str, cls: type[SourceResolver]) -> None:
    _RESOLVERS[name] = cls

def get_resolver(source: Source) -> SourceResolver:
    resolver_type = source.metadata.get("resolver", "local")
    return _RESOLVERS[resolver_type]()
```

Cloud extras auto-register on import:
```python
# neural_memory/resolvers/s3.py
from neural_memory.engine.source_resolver import register_resolver
register_resolver("s3", S3Resolver)
```

## What NOT to Build
- No fsspec/smart_open dependency — too heavy
- No file sync (download to local) — read on demand, cache briefly
- No SharePoint until customer asks
- No webhook-based real-time sync — poll via refresh command

## Acceptance Criteria
- [ ] S3Resolver reads files from S3 bucket using boto3
- [ ] GDriveResolver reads files from Google Drive using service account
- [ ] Core NM works without cloud extras installed (graceful ImportError)
- [ ] Resolver selection is automatic based on Source metadata

## Files Touched
- `src/neural_memory/engine/resolvers/__init__.py` — NEW: registry
- `src/neural_memory/engine/resolvers/local.py` — moved from source_resolver.py
- `src/neural_memory/engine/resolvers/s3.py` — NEW
- `src/neural_memory/engine/resolvers/gdrive.py` — NEW
- `pyproject.toml` — add optional extras
- `docs/guides/cloud-sources.md` — NEW: setup guide

## Dependencies
- Requires Phase 2 (SourceResolver protocol)
- Customer demand (do NOT build speculatively)
