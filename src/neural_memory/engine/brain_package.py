"""Brain package format for the Brain Store.

A `.brain` file is a single JSON document containing:
- manifest: metadata for browsing/preview (name, author, stats, scan result)
- snapshot: full BrainSnapshot data (neurons, synapses, fibers, config)

The manifest is cheap to read without parsing the full snapshot,
enabling registry indexes to store only manifests (~500B each).
"""

from __future__ import annotations

import hashlib
import json
import re as _re
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from neural_memory.core.brain import BrainSnapshot
from neural_memory.engine.brain_transplant import TransplantFilter, extract_subgraph
from neural_memory.engine.brain_versioning import _snapshot_to_json
from neural_memory.safety.brain_scanner import scan_brain_package

# ── Package Format Version ───────────────────────────────────────

PACKAGE_FORMAT_VERSION = "1.0"

# ── Size Tiers ───────────────────────────────────────────────────

MAX_PACKAGE_BYTES = 10 * 1024 * 1024  # 10MB hard limit for registry

SIZE_TIERS = {
    "micro": 100 * 1024,  # < 100KB
    "small": 1024 * 1024,  # < 1MB
    "medium": MAX_PACKAGE_BYTES,  # < 10MB
}


def _classify_size(size_bytes: int) -> str:
    """Classify package into size tier."""
    if size_bytes < SIZE_TIERS["micro"]:
        return "micro"
    if size_bytes < SIZE_TIERS["small"]:
        return "small"
    if size_bytes <= SIZE_TIERS["medium"]:
        return "medium"
    return "oversized"


# ── Manifest ─────────────────────────────────────────────────────

CATEGORIES = frozenset(
    {
        "programming",
        "devops",
        "writing",
        "science",
        "personal",
        "security",
        "data",
        "design",
        "general",
    }
)


@dataclass(frozen=True)
class BrainPackageManifest:
    """Metadata header for a .brain package."""

    id: str
    name: str
    display_name: str
    description: str
    version: str
    author: str
    license: str
    tags: list[str]
    category: str
    created_at: str
    exported_at: str
    nmem_version: str
    stats: dict[str, int]
    size_bytes: int
    size_tier: str
    content_hash: str
    scan_summary: dict[str, Any] = field(default_factory=dict)
    # Rating fields — populated by registry, not during export
    rating_avg: float = 0.0
    rating_count: int = 0
    download_count: int = 0


# ── Core Functions ───────────────────────────────────────────────


def create_brain_package(
    snapshot: BrainSnapshot,
    *,
    display_name: str,
    description: str,
    author: str,
    name: str | None = None,
    version: str = "1.0.0",
    license_: str = "CC-BY-4.0",
    tags: list[str] | None = None,
    category: str = "general",
    nmem_version: str = "",
    transplant_filter: TransplantFilter | None = None,
) -> dict[str, Any]:
    """Create a .brain package from a BrainSnapshot.

    Args:
        snapshot: The brain data to package.
        display_name: Human-readable name for the brain.
        description: What this brain contains.
        author: Author identifier (e.g. GitHub username).
        name: Machine-readable slug. Defaults to sanitized display_name.
        version: Semantic version of the brain content.
        license_: SPDX license identifier.
        tags: Categorization tags.
        category: One of CATEGORIES.
        nmem_version: Neural Memory version that created this package.
        transplant_filter: Optional filter for selective export.

    Returns:
        Dict representing the .brain package (ready for json.dumps).

    Raises:
        ValueError: If the package fails security scan (risk >= high)
            or exceeds size limit.
    """
    # Apply selective filter if provided
    if transplant_filter is not None:
        snapshot = extract_subgraph(snapshot, transplant_filter)

    # Serialize snapshot
    snapshot_json = _snapshot_to_json(snapshot)
    snapshot_data = json.loads(snapshot_json)
    # Hash uses sort_keys for deterministic comparison during validation
    canonical_json = json.dumps(snapshot_data, sort_keys=True, default=str)
    content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # Calculate size
    package_name = name or _slugify(display_name)
    now = datetime.now(UTC).isoformat()

    # Build preliminary package for scanning
    stats = {
        "neuron_count": len(snapshot.neurons),
        "synapse_count": len(snapshot.synapses),
        "fiber_count": len(snapshot.fibers),
    }

    preliminary_package = {
        "nmem_brain_package": PACKAGE_FORMAT_VERSION,
        "manifest": {
            "name": package_name,
            "display_name": display_name,
            "description": description,
            "author": author,
            "tags": tags or [],
            "category": category,
        },
        "snapshot": snapshot_data,
    }

    # Security scan
    scan_result = scan_brain_package(preliminary_package)

    if scan_result.risk_level in ("high", "critical"):
        finding_summary = "; ".join(f.description for f in scan_result.findings[:5])
        raise ValueError(
            f"Brain package blocked: risk_level={scan_result.risk_level}. "
            f"Findings: {finding_summary}"
        )

    # Build final package
    full_json = json.dumps(preliminary_package, default=str)
    size_bytes = len(full_json.encode("utf-8"))
    size_tier = _classify_size(size_bytes)

    manifest = BrainPackageManifest(
        id=str(uuid.uuid4()),
        name=package_name,
        display_name=display_name,
        description=description,
        version=version,
        author=author,
        license=license_,
        tags=tags or [],
        category=category if category in CATEGORIES else "general",
        created_at=now,
        exported_at=now,
        nmem_version=nmem_version,
        stats=stats,
        size_bytes=size_bytes,
        size_tier=size_tier,
        content_hash=f"sha256:{content_hash}",
        scan_summary={
            "risk_level": scan_result.risk_level,
            "finding_count": len(scan_result.findings),
            "safe": scan_result.safe,
        },
    )

    return {
        "nmem_brain_package": PACKAGE_FORMAT_VERSION,
        "manifest": asdict(manifest),
        "snapshot": snapshot_data,
    }


def validate_brain_package(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a .brain package structure and integrity.

    Args:
        data: Parsed JSON dict of a .brain file.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # Format version check
    fmt_version = data.get("nmem_brain_package")
    if not fmt_version:
        errors.append("Missing 'nmem_brain_package' format version")
    elif fmt_version != PACKAGE_FORMAT_VERSION:
        errors.append(
            f"Unsupported format version: {fmt_version} (expected {PACKAGE_FORMAT_VERSION})"
        )

    # Manifest required fields
    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        errors.append("Missing or invalid 'manifest'")
    else:
        for required in ("id", "name", "display_name", "description", "author", "content_hash"):
            if not manifest.get(required):
                errors.append(f"Manifest missing required field: {required}")

    # Snapshot required fields
    snapshot = data.get("snapshot")
    if not isinstance(snapshot, dict):
        errors.append("Missing or invalid 'snapshot'")
    else:
        for required in ("neurons", "synapses", "fibers"):
            if not isinstance(snapshot.get(required), list):
                errors.append(f"Snapshot missing or invalid field: {required}")

    if errors:
        return False, errors

    assert isinstance(manifest, dict)
    assert isinstance(snapshot, dict)

    # Hash verification
    content_hash = manifest.get("content_hash", "")
    if content_hash.startswith("sha256:"):
        expected_hash = content_hash[7:]
        actual_json = json.dumps(snapshot, sort_keys=True, default=str)
        actual_hash = hashlib.sha256(actual_json.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            errors.append(
                f"Content hash mismatch: expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )

    # Size check
    size_bytes = manifest.get("size_bytes", 0)
    if size_bytes > MAX_PACKAGE_BYTES:
        errors.append(f"Package too large: {size_bytes} bytes (max {MAX_PACKAGE_BYTES})")

    return len(errors) == 0, errors


def extract_manifest(data: dict[str, Any]) -> BrainPackageManifest | None:
    """Extract manifest from a .brain package without parsing the snapshot.

    Args:
        data: Parsed JSON dict of a .brain file.

    Returns:
        BrainPackageManifest or None if invalid.
    """
    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        return None

    try:
        return BrainPackageManifest(
            id=manifest.get("id", ""),
            name=manifest.get("name", ""),
            display_name=manifest.get("display_name", ""),
            description=manifest.get("description", ""),
            version=manifest.get("version", "1.0.0"),
            author=manifest.get("author", ""),
            license=manifest.get("license", ""),
            tags=manifest.get("tags", []),
            category=manifest.get("category", "general"),
            created_at=manifest.get("created_at", ""),
            exported_at=manifest.get("exported_at", ""),
            nmem_version=manifest.get("nmem_version", ""),
            stats=manifest.get("stats", {}),
            size_bytes=manifest.get("size_bytes", 0),
            size_tier=manifest.get("size_tier", "unknown"),
            content_hash=manifest.get("content_hash", ""),
            scan_summary=manifest.get("scan_summary", {}),
            rating_avg=manifest.get("rating_avg", 0.0),
            rating_count=manifest.get("rating_count", 0),
            download_count=manifest.get("download_count", 0),
        )
    except (TypeError, KeyError):
        return None


def preview_brain_package(data: dict[str, Any]) -> dict[str, Any]:
    """Generate a preview of a .brain package for the Store UI.

    Returns manifest, sample neurons, type distribution, and scan results
    without importing the brain.

    Args:
        data: Parsed JSON dict of a .brain file.

    Returns:
        Preview dict with manifest, samples, stats, and scan results.
    """
    manifest = extract_manifest(data)
    snapshot = data.get("snapshot", {})
    neurons = snapshot.get("neurons", [])

    # Sample neurons (first 10, content truncated)
    sample_neurons = []
    for neuron in neurons[:10]:
        content = neuron.get("content", "")
        sample_neurons.append(
            {
                "id": neuron.get("id", ""),
                "type": neuron.get("type", ""),
                "content": content[:200] + ("..." if len(content) > 200 else ""),
                "created_at": neuron.get("created_at", ""),
            }
        )

    # Neuron type distribution
    type_breakdown: dict[str, int] = {}
    for neuron in neurons:
        ntype = neuron.get("type", "unknown")
        type_breakdown[ntype] = type_breakdown.get(ntype, 0) + 1

    # Top tags from fibers
    tag_counts: dict[str, int] = {}
    for fiber in snapshot.get("fibers", []):
        for tag_field in ("tags", "auto_tags", "agent_tags"):
            tags = fiber.get(tag_field)
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:20]  # type: ignore[arg-type]

    # Security scan
    scan_result = scan_brain_package(data)

    return {
        "manifest": asdict(manifest) if manifest else {},
        "sample_neurons": sample_neurons,
        "neuron_type_breakdown": type_breakdown,
        "top_tags": top_tags,
        "scan_result": {
            "safe": scan_result.safe,
            "risk_level": scan_result.risk_level,
            "finding_count": len(scan_result.findings),
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "description": f.description,
                    "location": f.location,
                }
                for f in scan_result.findings[:20]
            ],
        },
    }


# ── Helpers ──────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Convert display name to URL-safe slug.

    Falls back to a short UUID prefix for non-Latin names that
    would produce an empty slug after sanitization.
    """
    # Normalize Unicode to ASCII-compatible forms where possible
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    slug = ascii_text.lower().strip()
    slug = _re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = _re.sub(r"[\s-]+", "-", slug)
    slug = slug[:64].strip("-")

    if not slug:
        slug = f"brain-{uuid.uuid4().hex[:8]}"

    return slug
