"""Knowledge extraction from database schemas.

Transforms a SchemaSnapshot into teachable SchemaKnowledge:
- KnowledgeEntity: table-level descriptions (semantic, not technical)
- KnowledgeRelationship: FK-based relationships with SynapseType mapping
- KnowledgePattern: detected schema patterns (audit trail, soft delete, etc.)
- KnowledgeProperty: column-level constraints and purposes

All knowledge objects carry confidence scores (0.0-1.0).
Join tables are detected structurally (not by name) and become
direct CO_OCCURS synapses instead of entity nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.db_introspector import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaSnapshot,
    TableInfo,
)

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────


class SchemaPatternType(StrEnum):
    """Detectable schema patterns (v1: 5 patterns)."""

    AUDIT_TRAIL = "audit_trail"
    SOFT_DELETE = "soft_delete"
    TREE_HIERARCHY = "tree_hierarchy"
    POLYMORPHIC = "polymorphic"
    ENUM_TABLE = "enum_table"


# ── Knowledge dataclasses (all frozen, all with confidence) ─────


@dataclass(frozen=True)
class KnowledgeEntity:
    """Table-level knowledge with semantic description.

    Attributes:
        table_name: Database table name.
        description: Semantic description for MemoryEncoder.
        column_summary: Brief column listing.
        row_count_estimate: Approximate row count.
        business_purpose: Inferred purpose from naming conventions.
        confidence: How confident we are in the business_purpose (0.0-1.0).
    """

    table_name: str
    description: str
    column_summary: str
    row_count_estimate: int
    business_purpose: str
    confidence: float = 0.85


@dataclass(frozen=True)
class KnowledgeRelationship:
    """FK-based relationship between tables.

    Attributes:
        source_table: Table containing the FK column.
        source_column: FK column name.
        target_table: Referenced table.
        target_column: Referenced column.
        synapse_type: Mapped NeuralMemory SynapseType.
        confidence: How confident we are in the synapse_type mapping.
    """

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    synapse_type: SynapseType
    confidence: float


@dataclass(frozen=True)
class KnowledgePattern:
    """Detected schema pattern.

    Attributes:
        pattern_type: Which pattern was detected.
        table_name: Table exhibiting the pattern.
        evidence: Column names and values that triggered detection.
        description: Human-readable description for MemoryEncoder.
        confidence: Detection confidence (0.0-1.0).
    """

    pattern_type: SchemaPatternType
    table_name: str
    # NOTE: dict inside frozen dataclass — matches Neuron precedent.
    # Contents should not be mutated after creation.
    evidence: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 0.7


@dataclass(frozen=True)
class KnowledgeProperty:
    """Column-level knowledge.

    Attributes:
        table_name: Parent table name.
        column_name: Column name.
        data_type: SQL data type.
        constraints: Constraint descriptions (e.g., "NOT NULL", "UNIQUE").
        purpose: Inferred column purpose.
    """

    table_name: str
    column_name: str
    data_type: str
    constraints: tuple[str, ...] = ()
    purpose: str = ""


@dataclass(frozen=True)
class SchemaKnowledge:
    """Complete extracted knowledge from a database schema.

    Attributes:
        entities: Table-level knowledge (excludes join tables).
        relationships: FK-based relationships with SynapseType mappings.
        patterns: Detected schema patterns.
        properties: Column-level details.
    """

    entities: tuple[KnowledgeEntity, ...]
    relationships: tuple[KnowledgeRelationship, ...]
    patterns: tuple[KnowledgePattern, ...]
    properties: tuple[KnowledgeProperty, ...]


# ── FK-to-SynapseType mapper ───────────────────────────────────


# Column name patterns → SynapseType + confidence
_PARENT_PATTERNS = frozenset({"parent_id", "category_id", "type_id", "group_id"})
_OWNER_PATTERNS = frozenset(
    {"user_id", "owner_id", "author_id", "creator_id", "assignee_id"}
)
_LOCATION_PATTERNS = frozenset(
    {"location_id", "address_id", "place_id", "region_id", "country_id"}
)


class FKMapper:
    """Maps foreign key semantics to NeuralMemory SynapseType.

    Uses column naming conventions with confidence scores.
    Default is RELATED_TO (safe) rather than CONTAINS (assertive).
    """

    def map_fk_to_synapse(
        self, fk: ForeignKeyInfo, table: TableInfo
    ) -> tuple[SynapseType, float]:
        """Map a FK to (SynapseType, confidence).

        Returns:
            Tuple of (synapse_type, confidence).
        """
        col_lower = fk.column_name.lower()

        if col_lower in _PARENT_PATTERNS:
            return SynapseType.IS_A, 0.85

        if col_lower in _OWNER_PATTERNS:
            return SynapseType.INVOLVES, 0.75

        if col_lower in _LOCATION_PATTERNS:
            return SynapseType.AT_LOCATION, 0.80

        # Safe default — RELATED_TO is never wrong
        return SynapseType.RELATED_TO, 0.60


# ── Join table detector ─────────────────────────────────────────

# Columns that don't count as "business" columns in join table detection
_MAX_SUMMARY_COLUMNS = 8
_MAX_ENUM_TABLE_COLUMNS = 4

_AUDIT_COLUMNS = frozenset(
    {"id", "created_at", "updated_at", "created_on", "modified_at"}
)


class JoinTableDetector:
    """Detects join (many-to-many) tables by structure, not name.

    A join table has:
    - 2+ FK columns
    - At most 1 meaningful business column (beyond FKs + audit fields)
    """

    def is_join_table(self, table: TableInfo) -> bool:
        """Check if table is a join table based on structure."""
        fk_columns = {fk.column_name.lower() for fk in table.foreign_keys}
        if len(fk_columns) < 2:
            return False

        # Count meaningful business columns (not FK, not audit/id)
        business_cols = [
            c
            for c in table.columns
            if c.name.lower() not in _AUDIT_COLUMNS
            and c.name.lower() not in fk_columns
            and not c.primary_key
        ]

        return len(business_cols) <= 1


# ── Pattern detector ────────────────────────────────────────────


class PatternDetector:
    """Detects common schema patterns with confidence scores."""

    def detect_all(
        self, table: TableInfo
    ) -> list[tuple[SchemaPatternType, float, dict[str, Any]]]:
        """Detect all patterns in a table.

        Returns:
            List of (pattern_type, confidence, evidence) tuples.
        """
        col_names = {c.name.lower() for c in table.columns}
        results: list[tuple[SchemaPatternType, float, dict[str, Any]]] = []

        # AUDIT_TRAIL: timestamps + user tracking
        audit = self._detect_audit_trail(col_names)
        if audit:
            results.append(audit)

        # SOFT_DELETE: deleted_at, is_deleted, archived_at
        soft = self._detect_soft_delete(col_names)
        if soft:
            results.append(soft)

        # TREE_HIERARCHY: self-referencing FK
        tree = self._detect_tree_hierarchy(table)
        if tree:
            results.append(tree)

        # POLYMORPHIC: *_type + *_id pairs
        poly = self._detect_polymorphic(col_names)
        if poly:
            results.append(poly)

        # ENUM_TABLE: small lookup table
        enum = self._detect_enum_table(table, col_names)
        if enum:
            results.append(enum)

        return results

    def _detect_audit_trail(
        self, col_names: set[str]
    ) -> tuple[SchemaPatternType, float, dict[str, Any]] | None:
        """Detect audit trail pattern."""
        has_created = "created_at" in col_names or "created_on" in col_names
        has_updated = "updated_at" in col_names or "updated_on" in col_names
        has_user = any(
            c in col_names
            for c in ("created_by", "updated_by", "last_modified_by")
        )

        if has_created and has_updated and has_user:
            return (
                SchemaPatternType.AUDIT_TRAIL,
                0.90,
                {"has_timestamps": True, "has_user_tracking": True},
            )
        if has_created and has_updated:
            return (
                SchemaPatternType.AUDIT_TRAIL,
                0.70,
                {"has_timestamps": True, "has_user_tracking": False},
            )
        return None

    def _detect_soft_delete(
        self, col_names: set[str]
    ) -> tuple[SchemaPatternType, float, dict[str, Any]] | None:
        """Detect soft-delete pattern with graduated confidence."""
        if "deleted_at" in col_names:
            return (
                SchemaPatternType.SOFT_DELETE,
                0.90,
                {"column": "deleted_at"},
            )
        if "is_deleted" in col_names:
            return (
                SchemaPatternType.SOFT_DELETE,
                0.80,
                {"column": "is_deleted"},
            )
        if "archived_at" in col_names:
            return (
                SchemaPatternType.SOFT_DELETE,
                0.60,
                {"column": "archived_at"},
            )
        if "removed_at" in col_names:
            return (
                SchemaPatternType.SOFT_DELETE,
                0.70,
                {"column": "removed_at"},
            )
        return None

    def _detect_tree_hierarchy(
        self, table: TableInfo
    ) -> tuple[SchemaPatternType, float, dict[str, Any]] | None:
        """Detect self-referencing FK (tree/hierarchy)."""
        for fk in table.foreign_keys:
            if fk.referenced_table == table.name:
                return (
                    SchemaPatternType.TREE_HIERARCHY,
                    0.95,
                    {"self_fk_column": fk.column_name},
                )
        return None

    def _detect_polymorphic(
        self, col_names: set[str]
    ) -> tuple[SchemaPatternType, float, dict[str, Any]] | None:
        """Detect polymorphic pattern (*_type + *_id pairs)."""
        for col in col_names:
            if col.endswith("_type"):
                prefix = col[:-5]
                if f"{prefix}_id" in col_names:
                    return (
                        SchemaPatternType.POLYMORPHIC,
                        0.85,
                        {"type_column": col, "id_column": f"{prefix}_id"},
                    )
        return None

    def _detect_enum_table(
        self,
        table: TableInfo,
        col_names: set[str],
    ) -> tuple[SchemaPatternType, float, dict[str, Any]] | None:
        """Detect enum/lookup table (small, has name/label/code column)."""
        if len(table.columns) > _MAX_ENUM_TABLE_COLUMNS:
            return None

        name_cols = col_names & {"name", "label", "code", "value", "title"}
        if name_cols and len(table.foreign_keys) == 0:
            return (
                SchemaPatternType.ENUM_TABLE,
                0.70,
                {"name_column": sorted(name_cols)[0]},
            )
        return None


# ── Business purpose inference ──────────────────────────────────


def _infer_purpose(table_name: str) -> str:
    """Infer business purpose from table name using heuristics."""
    lower = table_name.lower()

    if lower.endswith("_log") or lower.endswith("_logs"):
        return "stores activity or audit logs"
    if lower.endswith("_history") or lower.endswith("_histories"):
        return "stores historical snapshots"
    if lower.endswith("_config") or lower.endswith("_settings"):
        return "stores configuration settings"
    if lower.startswith("dim_"):
        return "is a dimension table for analytics"
    if lower.startswith("fact_"):
        return "is a fact table for analytics"

    # Singularize for description
    if lower.endswith("ies"):
        singular = lower[:-3] + "y"
    elif lower.endswith("ses") or lower.endswith("xes"):
        singular = lower[:-2]
    elif lower.endswith("s") and not lower.endswith("ss"):
        singular = lower[:-1]
    else:
        singular = lower

    return f"stores {singular} records"


# ── Knowledge extractor ─────────────────────────────────────────


class KnowledgeExtractor:
    """Extracts teachable knowledge from a SchemaSnapshot.

    Produces confidence-scored entities, relationships, patterns,
    and properties. Join tables become CO_OCCURS synapses, not
    entity nodes.
    """

    def __init__(self) -> None:
        self._fk_mapper = FKMapper()
        self._join_detector = JoinTableDetector()
        self._pattern_detector = PatternDetector()

    def extract(self, snapshot: SchemaSnapshot) -> SchemaKnowledge:
        """Extract all knowledge from a schema snapshot.

        Args:
            snapshot: Frozen schema snapshot from SchemaIntrospector.

        Returns:
            SchemaKnowledge with entities, relationships, patterns, properties.
        """
        entities: list[KnowledgeEntity] = []
        relationships: list[KnowledgeRelationship] = []
        patterns: list[KnowledgePattern] = []
        properties: list[KnowledgeProperty] = []

        # First pass: identify join tables
        join_tables: set[str] = set()
        for table in snapshot.tables:
            if self._join_detector.is_join_table(table):
                join_tables.add(table.name)

        for table in snapshot.tables:
            is_join = table.name in join_tables

            # Join tables → CO_OCCURS relationships, not entity nodes
            if is_join:
                relationships.extend(
                    self._extract_join_relationships(table)
                )
                continue

            # Entity knowledge (semantic description)
            entity = self._create_entity(table, snapshot)
            entities.append(entity)

            # Column properties
            for col in table.columns:
                prop = self._create_property(table.name, col)
                properties.append(prop)

            # Detect patterns
            detected = self._pattern_detector.detect_all(table)
            for pattern_type, confidence, evidence in detected:
                pattern = self._create_pattern(
                    pattern_type, table.name, evidence, confidence
                )
                patterns.append(pattern)

            # FK relationships (non-join)
            for fk in table.foreign_keys:
                synapse_type, confidence = self._fk_mapper.map_fk_to_synapse(
                    fk, table
                )
                relationships.append(
                    KnowledgeRelationship(
                        source_table=table.name,
                        source_column=fk.column_name,
                        target_table=fk.referenced_table,
                        target_column=fk.referenced_column,
                        synapse_type=synapse_type,
                        confidence=confidence,
                    )
                )

        return SchemaKnowledge(
            entities=tuple(entities),
            relationships=tuple(relationships),
            patterns=tuple(patterns),
            properties=tuple(properties),
        )

    def _extract_join_relationships(
        self,
        table: TableInfo,
    ) -> list[KnowledgeRelationship]:
        """Convert join table FKs into direct CO_OCCURS relationships.

        For a join table with FKs to table A and B, creates a
        CO_OCCURS relationship between A and B (skipping the join
        table itself as an entity).
        """
        result: list[KnowledgeRelationship] = []
        fk_targets = [
            (fk.referenced_table, fk.referenced_column)
            for fk in table.foreign_keys
        ]

        # Create CO_OCCURS between all pairs of referenced tables
        for i in range(len(fk_targets)):
            for j in range(i + 1, len(fk_targets)):
                result.append(
                    KnowledgeRelationship(
                        source_table=fk_targets[i][0],
                        source_column=fk_targets[i][1],
                        target_table=fk_targets[j][0],
                        target_column=fk_targets[j][1],
                        synapse_type=SynapseType.CO_OCCURS,
                        confidence=0.85,
                    )
                )
        return result

    def _create_entity(
        self, table: TableInfo, snapshot: SchemaSnapshot
    ) -> KnowledgeEntity:
        """Create semantic entity description for a table."""
        purpose = _infer_purpose(table.name)

        # Build column summary (first N columns)
        col_parts: list[str] = []
        for col in table.columns[:_MAX_SUMMARY_COLUMNS]:
            constraints: list[str] = []
            if col.primary_key:
                constraints.append("PK")
            if not col.nullable and not col.primary_key:
                constraints.append("NOT NULL")
            suffix = f" [{', '.join(constraints)}]" if constraints else ""
            col_parts.append(f"{col.name} ({col.data_type}{suffix})")

        column_summary = ", ".join(col_parts)
        if len(table.columns) > _MAX_SUMMARY_COLUMNS:
            column_summary += f", ... (+{len(table.columns) - _MAX_SUMMARY_COLUMNS} more)"

        # Build FK context for richer descriptions
        fk_context = ""
        if table.foreign_keys:
            fk_refs = [
                f"{fk.referenced_table}" for fk in table.foreign_keys
            ]
            fk_context = f" Links to: {', '.join(fk_refs)}."

        # Use table comment if available (gold metadata)
        if table.comment:
            description = (
                f"Database table '{table.name}': {table.comment}. "
                f"Columns: {column_summary}.{fk_context}"
            )
        else:
            description = (
                f"Database table '{table.name}' {purpose}. "
                f"Columns: {column_summary}.{fk_context}"
            )

        return KnowledgeEntity(
            table_name=table.name,
            description=description,
            column_summary=column_summary,
            row_count_estimate=table.row_count_estimate,
            business_purpose=purpose,
            confidence=0.90 if table.comment else 0.75,
        )

    def _create_property(
        self, table_name: str, col: ColumnInfo
    ) -> KnowledgeProperty:
        """Create column-level property knowledge."""
        constraints: list[str] = []
        if col.primary_key:
            constraints.append("PRIMARY KEY")
        if not col.nullable:
            constraints.append("NOT NULL")
        if col.default_value is not None:
            constraints.append(f"DEFAULT {col.default_value}")

        purpose = self._infer_column_purpose(col.name)

        return KnowledgeProperty(
            table_name=table_name,
            column_name=col.name,
            data_type=col.data_type,
            constraints=tuple(constraints),
            purpose=purpose,
        )

    def _create_pattern(
        self,
        pattern_type: SchemaPatternType,
        table_name: str,
        evidence: dict[str, Any],
        confidence: float,
    ) -> KnowledgePattern:
        """Create pattern knowledge with human-readable description."""
        descriptions = {
            SchemaPatternType.AUDIT_TRAIL: (
                f"Table '{table_name}' uses audit trail pattern — "
                "tracks creation and modification timestamps"
                + (" with user attribution" if evidence.get("has_user_tracking") else "")
            ),
            SchemaPatternType.SOFT_DELETE: (
                f"Table '{table_name}' uses soft-delete pattern via "
                f"'{evidence.get('column', 'unknown')}' column — "
                "records are marked as deleted rather than removed"
            ),
            SchemaPatternType.TREE_HIERARCHY: (
                f"Table '{table_name}' implements tree/hierarchy structure "
                f"via self-referencing FK '{evidence.get('self_fk_column', 'unknown')}'"
            ),
            SchemaPatternType.POLYMORPHIC: (
                f"Table '{table_name}' uses polymorphic association via "
                f"'{evidence.get('type_column', 'unknown')}' + "
                f"'{evidence.get('id_column', 'unknown')}' pair"
            ),
            SchemaPatternType.ENUM_TABLE: (
                f"Table '{table_name}' appears to be a lookup/enum table "
                f"with '{evidence.get('name_column', 'unknown')}' as the label column"
            ),
        }

        return KnowledgePattern(
            pattern_type=pattern_type,
            table_name=table_name,
            evidence=evidence,
            description=descriptions.get(pattern_type, f"Pattern {pattern_type} on {table_name}"),
            confidence=confidence,
        )

    def _infer_column_purpose(self, col_name: str) -> str:
        """Infer column purpose from naming conventions."""
        lower = col_name.lower()

        if lower == "id":
            return "primary identifier"
        if lower.endswith("_id"):
            ref = lower[:-3].replace("_", " ")
            return f"references {ref}"
        if lower in ("created_at", "created_on"):
            return "creation timestamp"
        if lower in ("updated_at", "updated_on", "modified_at"):
            return "last modification timestamp"
        if lower in ("deleted_at", "removed_at"):
            return "soft-delete timestamp"
        if lower in ("is_deleted", "is_active", "is_enabled"):
            return "boolean status flag"
        if lower == "email":
            return "email address"
        if lower == "name":
            return "display name"
        if lower in ("description", "summary", "body", "content"):
            return "text content"
        if lower in ("status", "state"):
            return "lifecycle status"
        if lower == "version":
            return "record version number"
        if lower in ("price", "amount", "total", "cost"):
            return "monetary value"
        if lower in ("count", "quantity", "qty"):
            return "numeric count"

        return ""
