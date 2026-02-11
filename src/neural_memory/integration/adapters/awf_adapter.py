"""AWF (Antigravity Workflow Framework) source adapter.

Imports memories from AWF's .brain/ directory structure:
- brain.json  (Tier 1: static knowledge — project info, key decisions)
- session.json (Tier 2: current state — working_on, errors, summaries)
- snapshots/   (Historical snapshots)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)

logger = logging.getLogger(__name__)


class AWFAdapter:
    """Adapter for importing context from AWF .brain/ directories.

    AWF stores AI agent context in a 3-tier system:
    - Tier 1 (CRITICAL): Project name, tech stack, key decisions
    - Tier 2 (IMPORTANT): Current working state, errors, progress
    - Tier 3 (CONTEXT): Conversation summaries, recent files

    Usage:
        adapter = AWFAdapter(brain_dir="/path/to/.brain")
        records = await adapter.fetch_all()
    """

    def __init__(self, brain_dir: str | Path) -> None:
        resolved = Path(brain_dir).resolve()
        # Validate path is not traversing outside expected locations
        if ".." in resolved.parts:
            raise ValueError("Invalid brain directory: path traversal detected")
        self._brain_dir = resolved

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.FILE_STORE

    @property
    def system_name(self) -> str:
        return "awf"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_METADATA,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch records from AWF .brain/ directory.

        Args:
            collection: Optional tier filter ("tier1", "tier2", or None for all)
            limit: Maximum records to return
        """
        records: list[ExternalRecord] = []

        if collection is None or collection == "tier1":
            records.extend(self._parse_brain_json())

        if collection is None or collection == "tier2":
            records.extend(self._parse_session_json())

        if collection is None:
            records.extend(self._parse_snapshots())

        if limit is not None:
            records = records[:limit]

        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """AWF does not support incremental sync."""
        raise NotImplementedError(
            "AWF adapter does not support incremental sync. Use fetch_all() instead."
        )

    async def health_check(self) -> dict[str, Any]:
        """Check if .brain/ directory is accessible."""
        if not self._brain_dir.exists():
            return {
                "healthy": False,
                "message": f"Brain directory not found: {self._brain_dir}",
                "system": "awf",
            }

        brain_json = self._brain_dir / "brain.json"
        if not brain_json.exists():
            return {
                "healthy": False,
                "message": f"brain.json not found in {self._brain_dir}",
                "system": "awf",
            }

        try:
            data = json.loads(brain_json.read_text(encoding="utf-8"))
            project_name = data.get("project", {}).get("name", "unknown")
            return {
                "healthy": True,
                "message": f"AWF brain accessible (project: {project_name})",
                "system": "awf",
            }
        except (json.JSONDecodeError, OSError) as e:
            return {
                "healthy": False,
                "message": f"Failed to read brain.json: {e}",
                "system": "awf",
            }

    def _parse_brain_json(self) -> list[ExternalRecord]:
        """Parse brain.json (Tier 1: static knowledge)."""
        brain_json = self._brain_dir / "brain.json"
        if not brain_json.exists():
            return []

        try:
            data = json.loads(brain_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse brain.json: %s", e)
            return []

        records: list[ExternalRecord] = []
        base_tags = {"import:awf", "tier:1"}

        # Project info
        project = data.get("project", {})
        if project:
            name = project.get("name", "")
            stack = project.get("tech_stack", [])
            if name:
                content = f"Project: {name}"
                if stack:
                    content += f". Stack: {', '.join(stack)}"
                records.append(
                    ExternalRecord.create(
                        id="awf:brain:project",
                        source_system="awf",
                        content=content,
                        source_collection="tier1",
                        source_type="fact",
                        tags=base_tags | {"project"},
                        metadata={"awf_key": "project", **project},
                    )
                )

        # Key decisions
        for i, decision in enumerate(data.get("key_decisions", [])):
            if isinstance(decision, dict):
                desc = decision.get("decision", "")
                reason = decision.get("reason", "")
                content = f"Decision: {desc}"
                if reason:
                    content += f". Reason: {reason}"
                meta = {"awf_key": "key_decisions", "index": i, **decision}
            else:
                content = f"Decision: {decision}"
                meta = {"awf_key": "key_decisions", "index": i}

            if content and content != "Decision: ":
                records.append(
                    ExternalRecord.create(
                        id=f"awf:brain:decision:{i}",
                        source_system="awf",
                        content=content,
                        source_collection="tier1",
                        source_type="decision",
                        tags=base_tags | {"decision"},
                        metadata=meta,
                    )
                )

        return records

    def _parse_session_json(self) -> list[ExternalRecord]:
        """Parse session.json (Tier 2: current state)."""
        session_json = self._brain_dir / "session.json"
        if not session_json.exists():
            return []

        try:
            data = json.loads(session_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse session.json: %s", e)
            return []

        records: list[ExternalRecord] = []
        base_tags = {"import:awf", "tier:2"}

        # Working on
        working_on = data.get("working_on", {})
        if working_on:
            feature = working_on.get("feature", "")
            task = working_on.get("task", "")
            progress = working_on.get("progress", "")
            parts = []
            if feature:
                parts.append(feature)
            if task:
                parts.append(task)
            content = "Working on: " + " — ".join(parts)
            if progress:
                content += f" ({progress}%)"

            if parts:
                records.append(
                    ExternalRecord.create(
                        id="awf:session:working_on",
                        source_system="awf",
                        content=content,
                        source_collection="tier2",
                        source_type="context",
                        tags=base_tags | {"working_on"},
                        metadata={"awf_key": "working_on", **working_on},
                    )
                )

        # Errors history
        for i, error in enumerate(data.get("errors_history", [])):
            if isinstance(error, dict):
                desc = error.get("error", "")
                fixed = error.get("fixed", False)
                content = f"Error: {desc}"
                if fixed:
                    content += " (fixed)"
                meta = {"awf_key": "errors_history", "index": i, **error}
            else:
                content = f"Error: {error}"
                meta = {"awf_key": "errors_history", "index": i}

            if content and content != "Error: ":
                records.append(
                    ExternalRecord.create(
                        id=f"awf:session:error:{i}",
                        source_system="awf",
                        content=content,
                        source_collection="tier2",
                        source_type="error",
                        tags=base_tags | {"error"},
                        metadata=meta,
                    )
                )

        # Conversation summary
        for i, summary in enumerate(data.get("conversation_summary", [])):
            if isinstance(summary, str) and summary.strip():
                records.append(
                    ExternalRecord.create(
                        id=f"awf:session:summary:{i}",
                        source_system="awf",
                        content=summary,
                        source_collection="tier2",
                        source_type="context",
                        tags=base_tags | {"summary"},
                        metadata={"awf_key": "conversation_summary", "index": i},
                    )
                )

        # Recent files
        for i, filepath in enumerate(data.get("recent_files", [])):
            if isinstance(filepath, str) and filepath.strip():
                records.append(
                    ExternalRecord.create(
                        id=f"awf:session:file:{i}",
                        source_system="awf",
                        content=filepath,
                        source_collection="tier2",
                        source_type="reference",
                        tags=base_tags | {"recent_file"},
                        metadata={"awf_key": "recent_files", "index": i},
                    )
                )

        return records

    def _parse_snapshots(self) -> list[ExternalRecord]:
        """Parse snapshot files from snapshots/ directory."""
        snapshots_dir = self._brain_dir / "snapshots"
        if not snapshots_dir.exists():
            return []

        records: list[ExternalRecord] = []
        base_tags = {"import:awf", "tier:3", "snapshot"}

        for snapshot_file in sorted(snapshots_dir.glob("*.json")):
            try:
                data = json.loads(snapshot_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse snapshot %s: %s", snapshot_file.name, e)
                continue

            snapshot_id = snapshot_file.stem

            # Extract any content from the snapshot
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and value.strip():
                        records.append(
                            ExternalRecord.create(
                                id=f"awf:snapshot:{snapshot_id}:{key}",
                                source_system="awf",
                                content=value,
                                source_collection="snapshots",
                                source_type="context",
                                tags=base_tags,
                                metadata={
                                    "awf_key": key,
                                    "snapshot": snapshot_id,
                                },
                            )
                        )

        return records
