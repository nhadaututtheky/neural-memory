"""Brain Store API routes — export, import, browse, and rate brain packages."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from neural_memory.engine.brain_package import (
    MAX_PACKAGE_BYTES,
    create_brain_package,
    preview_brain_package,
    validate_brain_package,
)
from neural_memory.engine.brain_registry import BrainRegistryClient
from neural_memory.engine.brain_transplant import TransplantFilter
from neural_memory.safety.brain_scanner import scan_brain_package
from neural_memory.server.dependencies import get_storage, require_local_request

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard/store",
    tags=["store"],
    dependencies=[Depends(require_local_request)],
)

# Singleton registry client (shared across requests)
_registry_client = BrainRegistryClient()


# ── Request / Response Models ────────────────────────────────────


class StoreExportRequest(BaseModel):
    """Request to export a brain as a .brain package."""

    display_name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=1000)
    author: str = Field(..., min_length=1, max_length=100)
    name: str | None = Field(None, max_length=64)
    version: str = Field("1.0.0", max_length=20)
    license_: str = Field("CC-BY-4.0", alias="license", max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=20)
    category: str = Field("general", max_length=30)
    filter_tags: list[str] | None = None
    filter_neuron_types: list[str] | None = None
    min_salience: float = Field(0.0, ge=0.0, le=1.0)


class StoreExportResponse(BaseModel):
    """Response from brain export."""

    manifest: dict[str, Any]
    scan_summary: dict[str, Any]
    package_size_bytes: int
    size_tier: str


class StoreImportResponse(BaseModel):
    """Response from brain import."""

    brain_id: str
    brain_name: str
    neurons_imported: int
    synapses_imported: int
    fibers_imported: int
    scan_result: dict[str, Any]
    warnings: list[str]


class BrainPreviewResponse(BaseModel):
    """Preview of a brain package."""

    manifest: dict[str, Any]
    sample_neurons: list[dict[str, Any]]
    neuron_type_breakdown: dict[str, int]
    top_tags: list[str]
    scan_result: dict[str, Any]


class BrainRatingRequest(BaseModel):
    """Submit a rating for a brain package."""

    brain_package_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field("", max_length=500)


class BrainRatingResponse(BaseModel):
    """Response after submitting a rating."""

    brain_package_id: str
    rating_avg: float
    rating_count: int


# ── Export ────────────────────────────────────────────────────────


@router.post("/export")
async def export_brain_package(
    req: StoreExportRequest,
    storage: Annotated[Any, Depends(get_storage)],
) -> JSONResponse:
    """Export the active brain as a .brain package.

    Runs security scan during export. Blocks if risk >= high.
    Returns the full .brain JSON as a downloadable response.
    """
    try:
        brain_id = storage.brain_id
        if not brain_id:
            raise HTTPException(status_code=400, detail="No active brain selected")
        snapshot = await storage.export_brain(brain_id)
    except Exception as e:
        logger.error("Failed to export brain: %s", e)
        raise HTTPException(status_code=500, detail="Failed to export brain") from e

    # Build transplant filter if selective export requested
    transplant_filter = None
    if req.filter_tags or req.filter_neuron_types or req.min_salience > 0:
        transplant_filter = TransplantFilter(
            tags=frozenset(req.filter_tags) if req.filter_tags else None,
            neuron_types=frozenset(req.filter_neuron_types) if req.filter_neuron_types else None,
            min_salience=req.min_salience,
        )

    try:
        import neural_memory

        package = create_brain_package(
            snapshot,
            display_name=req.display_name,
            description=req.description,
            author=req.author,
            name=req.name,
            version=req.version,
            license_=req.license_,
            tags=req.tags,
            category=req.category,
            nmem_version=getattr(neural_memory, "__version__", ""),
            transplant_filter=transplant_filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    manifest = package.get("manifest", {})
    package_json = json.dumps(package, default=str, ensure_ascii=False)

    return JSONResponse(
        content={
            "manifest": manifest,
            "scan_summary": manifest.get("scan_summary", {}),
            "package_size_bytes": manifest.get("size_bytes", len(package_json)),
            "size_tier": manifest.get("size_tier", "unknown"),
            "package": package,
        },
    )


# ── Import ────────────────────────────────────────────────────────


@router.post("/import", response_model=StoreImportResponse)
async def import_brain_package(
    file: UploadFile,
    storage: Annotated[Any, Depends(get_storage)],
) -> StoreImportResponse:
    """Import a .brain package file.

    Always scans before importing. Hard-blocks critical risk.
    """
    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_PACKAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes (max {MAX_PACKAGE_BYTES})",
        )

    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # Validate package format
    valid, errors = validate_brain_package(data)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    # Security scan — always run on import
    scan_result = scan_brain_package(data)
    warnings: list[str] = []

    if scan_result.risk_level in ("high", "critical"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Brain package contains dangerous content",
                "risk_level": scan_result.risk_level,
                "findings": [
                    {"severity": f.severity, "description": f.description}
                    for f in scan_result.findings[:10]
                ],
            },
        )

    if not scan_result.safe:
        warnings.extend(
            f"[{f.severity}] {f.description} at {f.location}" for f in scan_result.findings[:10]
        )

    # Import the brain
    snapshot_data = data.get("snapshot", {})
    manifest = data.get("manifest", {})
    brain_name = manifest.get("name", manifest.get("display_name", "imported"))

    try:
        import uuid

        from neural_memory.core.brain import BrainSnapshot
        from neural_memory.utils.timeutils import utcnow

        snapshot = BrainSnapshot(
            brain_id=str(uuid.uuid4()),
            brain_name=brain_name,
            exported_at=utcnow(),
            version=str(snapshot_data.get("version", "1")),
            neurons=snapshot_data.get("neurons", []),
            synapses=snapshot_data.get("synapses", []),
            fibers=snapshot_data.get("fibers", []),
            config=snapshot_data.get("config", {}),
            metadata=snapshot_data.get("metadata", {}),
        )

        brain_id = await storage.import_brain(snapshot)
    except Exception as e:
        logger.error("Failed to import brain package: %s", e)
        raise HTTPException(status_code=500, detail="Failed to import brain") from e

    scan_response = {
        "safe": scan_result.safe,
        "risk_level": scan_result.risk_level,
        "finding_count": len(scan_result.findings),
    }

    return StoreImportResponse(
        brain_id=brain_id,
        brain_name=brain_name,
        neurons_imported=len(snapshot_data.get("neurons", [])),
        synapses_imported=len(snapshot_data.get("synapses", [])),
        fibers_imported=len(snapshot_data.get("fibers", [])),
        scan_result=scan_response,
        warnings=warnings,
    )


# ── Preview (local file) ─────────────────────────────────────────


@router.post("/preview")
async def preview_brain_file(file: UploadFile) -> BrainPreviewResponse:
    """Preview a .brain file without importing.

    Returns manifest, sample neurons, type distribution, and scan results.
    """
    content = await file.read()
    if len(content) > MAX_PACKAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes (max {MAX_PACKAGE_BYTES})",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    preview = preview_brain_package(data)

    return BrainPreviewResponse(
        manifest=preview["manifest"],
        sample_neurons=preview["sample_neurons"],
        neuron_type_breakdown=preview["neuron_type_breakdown"],
        top_tags=preview["top_tags"],
        scan_result=preview["scan_result"],
    )


# ── Registry Browse ─────────────────────────────────────────────


class RegistryBrowseResponse(BaseModel):
    """Response from browsing the registry."""

    brains: list[dict[str, Any]]
    total: int
    cached: bool


class RemoteImportRequest(BaseModel):
    """Request to import a brain from a remote URL."""

    source_url: str = Field(..., min_length=1, max_length=2048, pattern=r"^https://")


@router.get("/registry")
async def browse_registry(
    category: str | None = Query(None, max_length=30),
    search: str | None = Query(None, max_length=100),
    tag: str | None = Query(None, max_length=50),
    sort_by: str = Query("created_at", pattern=r"^(created_at|rating_avg|download_count)$"),
    limit: int = Query(50, ge=1, le=100),
) -> RegistryBrowseResponse:
    """Browse the Brain Store registry.

    Fetches the catalog from GitHub (cached 5min) and applies filters.
    """
    cached_before = _registry_client.cache.get_index() is not None
    manifests = await _registry_client.fetch_index()
    cached_after = cached_before  # If we had cache before, response was from cache

    filtered = _registry_client.filter_index(
        manifests,
        category=category,
        search=search,
        tag=tag,
        sort_by=sort_by,
        limit=limit,
    )

    return RegistryBrowseResponse(
        brains=filtered,
        total=len(filtered),
        cached=cached_after,
    )


@router.get("/registry/preview/{brain_name}")
async def preview_registry_brain(
    brain_name: Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[\w-]+$")],
) -> BrainPreviewResponse:
    """Preview a brain from the registry without importing.

    Fetches the full brain package from GitHub and returns
    manifest, sample neurons, type distribution, and scan results.
    """
    data = await _registry_client.fetch_brain(brain_name)
    if data is None:
        raise HTTPException(status_code=404, detail="Brain not found in registry")

    preview = preview_brain_package(data)

    return BrainPreviewResponse(
        manifest=preview["manifest"],
        sample_neurons=preview["sample_neurons"],
        neuron_type_breakdown=preview["neuron_type_breakdown"],
        top_tags=preview["top_tags"],
        scan_result=preview["scan_result"],
    )


@router.post("/import-remote", response_model=StoreImportResponse)
async def import_remote_brain(
    req: RemoteImportRequest,
    storage: Annotated[Any, Depends(get_storage)],
) -> StoreImportResponse:
    """Import a brain from a remote URL (registry or direct link).

    Fetches the .brain package, validates, scans, and imports.
    """
    data = await _registry_client.fetch_brain_from_url(req.source_url)
    if data is None:
        raise HTTPException(status_code=400, detail="Failed to fetch brain from URL")

    # Validate size via serialized form
    serialized = json.dumps(data, default=str)
    if len(serialized.encode("utf-8")) > MAX_PACKAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Remote brain too large (max {MAX_PACKAGE_BYTES} bytes)",
        )

    # Validate package format
    valid, errors = validate_brain_package(data)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    # Security scan
    scan_result = scan_brain_package(data)
    warnings: list[str] = []

    if scan_result.risk_level in ("high", "critical"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Brain package contains dangerous content",
                "risk_level": scan_result.risk_level,
                "findings": [
                    {"severity": f.severity, "description": f.description}
                    for f in scan_result.findings[:10]
                ],
            },
        )

    if not scan_result.safe:
        warnings.extend(
            f"[{f.severity}] {f.description} at {f.location}" for f in scan_result.findings[:10]
        )

    # Import
    snapshot_data = data.get("snapshot", {})
    manifest = data.get("manifest", {})
    brain_name = manifest.get("name", manifest.get("display_name", "imported"))

    try:
        import uuid as _uuid

        from neural_memory.core.brain import BrainSnapshot
        from neural_memory.utils.timeutils import utcnow

        snapshot = BrainSnapshot(
            brain_id=str(_uuid.uuid4()),
            brain_name=brain_name,
            exported_at=utcnow(),
            version=str(snapshot_data.get("version", "1")),
            neurons=snapshot_data.get("neurons", []),
            synapses=snapshot_data.get("synapses", []),
            fibers=snapshot_data.get("fibers", []),
            config=snapshot_data.get("config", {}),
            metadata=snapshot_data.get("metadata", {}),
        )

        brain_id = await storage.import_brain(snapshot)
    except Exception as e:
        logger.error("Failed to import remote brain: %s", e)
        raise HTTPException(status_code=500, detail="Failed to import brain") from e

    return StoreImportResponse(
        brain_id=brain_id,
        brain_name=brain_name,
        neurons_imported=len(snapshot_data.get("neurons", [])),
        synapses_imported=len(snapshot_data.get("synapses", [])),
        fibers_imported=len(snapshot_data.get("fibers", [])),
        scan_result={
            "safe": scan_result.safe,
            "risk_level": scan_result.risk_level,
            "finding_count": len(scan_result.findings),
        },
        warnings=warnings,
    )


# ── Rating / Feedback ──────────────────────────────────────────


# In-memory ratings store — persists across requests within a server session
# Bounded to prevent unbounded memory growth
_MAX_RATED_PACKAGES = 1000
_MAX_RATINGS_PER_PACKAGE = 100
_ratings: dict[str, list[dict[str, Any]]] = {}


@router.post("/rate", response_model=BrainRatingResponse)
async def rate_brain_package(req: BrainRatingRequest) -> BrainRatingResponse:
    """Submit a rating for a brain package.

    Ratings are stored locally and will be synced to the
    registry in Phase 2 (GitHub-based index).
    """
    entry = {
        "rating": req.rating,
        "comment": req.comment,
    }

    if req.brain_package_id not in _ratings:
        if len(_ratings) >= _MAX_RATED_PACKAGES:
            raise HTTPException(status_code=429, detail="Rating storage full")
        _ratings[req.brain_package_id] = []

    if len(_ratings[req.brain_package_id]) >= _MAX_RATINGS_PER_PACKAGE:
        raise HTTPException(status_code=429, detail="Maximum ratings reached for this package")

    _ratings[req.brain_package_id].append(entry)

    all_ratings = _ratings[req.brain_package_id]
    avg = sum(r["rating"] for r in all_ratings) / len(all_ratings)

    return BrainRatingResponse(
        brain_package_id=req.brain_package_id,
        rating_avg=round(avg, 2),
        rating_count=len(all_ratings),
    )


@router.get("/ratings/{brain_package_id}")
async def get_brain_ratings(
    brain_package_id: str = Path(..., max_length=128),
) -> JSONResponse:
    """Get ratings for a specific brain package."""
    ratings = _ratings.get(brain_package_id, [])
    avg = sum(r["rating"] for r in ratings) / len(ratings) if ratings else 0.0

    return JSONResponse(
        content={
            "brain_package_id": brain_package_id,
            "rating_avg": round(avg, 2),
            "rating_count": len(ratings),
            "ratings": ratings,
        },
    )
