"""Tests for db_knowledge: schema knowledge extraction."""

from __future__ import annotations

import pytest

from neural_memory.core.synapse import SynapseType
from neural_memory.engine.db_introspector import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    SchemaSnapshot,
    TableInfo,
)
from neural_memory.engine.db_knowledge import (
    FKMapper,
    JoinTableDetector,
    KnowledgeEntity,
    KnowledgeExtractor,
    KnowledgePattern,
    KnowledgeProperty,
    KnowledgeRelationship,
    PatternDetector,
    SchemaKnowledge,
    SchemaPatternType,
    _infer_purpose,
)


# ── Frozen dataclass tests ──────────────────────────────────────


class TestFrozenDataclasses:
    """All knowledge dataclasses are immutable."""

    def test_knowledge_entity_frozen(self) -> None:
        e = KnowledgeEntity("t", "d", "c", 0, "p")
        with pytest.raises(AttributeError):
            e.table_name = "changed"  # type: ignore[misc]

    def test_knowledge_relationship_frozen(self) -> None:
        r = KnowledgeRelationship("a", "col", "b", "id", SynapseType.RELATED_TO, 0.6)
        with pytest.raises(AttributeError):
            r.source_table = "changed"  # type: ignore[misc]

    def test_knowledge_pattern_frozen(self) -> None:
        p = KnowledgePattern(SchemaPatternType.AUDIT_TRAIL, "t")
        with pytest.raises(AttributeError):
            p.table_name = "changed"  # type: ignore[misc]

    def test_knowledge_property_frozen(self) -> None:
        p = KnowledgeProperty("t", "col", "TEXT")
        with pytest.raises(AttributeError):
            p.column_name = "changed"  # type: ignore[misc]

    def test_schema_knowledge_frozen(self) -> None:
        sk = SchemaKnowledge((), (), (), ())
        with pytest.raises(AttributeError):
            sk.entities = ()  # type: ignore[misc]


# ── FKMapper tests ──────────────────────────────────────────────


def _make_table(name: str = "test", fk_col: str = "ref_id") -> TableInfo:
    """Helper: table with a single FK column."""
    return TableInfo(
        name=name,
        schema=None,
        columns=(ColumnInfo("id", "INTEGER", False, True),),
        foreign_keys=(
            ForeignKeyInfo(fk_col, "other_table", "id", None, None),
        ),
        indexes=(),
        row_count_estimate=0,
    )


class TestFKMapper:
    """FKMapper maps FK column names to (SynapseType, confidence)."""

    def setup_method(self) -> None:
        self.mapper = FKMapper()

    def test_parent_id_maps_to_is_a(self) -> None:
        table = _make_table(fk_col="parent_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.IS_A
        assert conf == 0.85

    def test_category_id_maps_to_is_a(self) -> None:
        table = _make_table(fk_col="category_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.IS_A
        assert conf == 0.85

    def test_type_id_maps_to_is_a(self) -> None:
        table = _make_table(fk_col="type_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.IS_A

    def test_group_id_maps_to_is_a(self) -> None:
        table = _make_table(fk_col="group_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.IS_A

    def test_user_id_maps_to_involves(self) -> None:
        table = _make_table(fk_col="user_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.INVOLVES
        assert conf == 0.75

    def test_owner_id_maps_to_involves(self) -> None:
        table = _make_table(fk_col="owner_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.INVOLVES

    def test_author_id_maps_to_involves(self) -> None:
        table = _make_table(fk_col="author_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.INVOLVES

    def test_creator_id_maps_to_involves(self) -> None:
        table = _make_table(fk_col="creator_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.INVOLVES

    def test_assignee_id_maps_to_involves(self) -> None:
        table = _make_table(fk_col="assignee_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.INVOLVES

    def test_location_id_maps_to_at_location(self) -> None:
        table = _make_table(fk_col="location_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.AT_LOCATION
        assert conf == 0.80

    def test_address_id_maps_to_at_location(self) -> None:
        table = _make_table(fk_col="address_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.AT_LOCATION

    def test_country_id_maps_to_at_location(self) -> None:
        table = _make_table(fk_col="country_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.AT_LOCATION

    def test_default_fk_maps_to_related_to(self) -> None:
        table = _make_table(fk_col="widget_id")
        stype, conf = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.RELATED_TO
        assert conf == 0.60

    def test_unknown_fk_is_safe_default(self) -> None:
        table = _make_table(fk_col="some_random_ref_id")
        stype, _ = self.mapper.map_fk_to_synapse(table.foreign_keys[0], table)
        assert stype == SynapseType.RELATED_TO


# ── JoinTableDetector tests ─────────────────────────────────────


class TestJoinTableDetector:
    """Structure-based join table detection (not name-based)."""

    def setup_method(self) -> None:
        self.detector = JoinTableDetector()

    def test_classic_join_table(self) -> None:
        """Two FKs + id only → join table."""
        table = TableInfo(
            name="user_roles",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("user_id", "INTEGER", False, False),
                ColumnInfo("role_id", "INTEGER", False, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("user_id", "users", "id", None, None),
                ForeignKeyInfo("role_id", "roles", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is True

    def test_join_table_with_audit_cols(self) -> None:
        """Two FKs + audit columns → still a join table."""
        table = TableInfo(
            name="enrollments",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("student_id", "INTEGER", False, False),
                ColumnInfo("course_id", "INTEGER", False, False),
                ColumnInfo("created_at", "TEXT", True, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("student_id", "students", "id", None, None),
                ForeignKeyInfo("course_id", "courses", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is True

    def test_join_table_with_one_business_col(self) -> None:
        """Two FKs + 1 business col (e.g., 'role') → still join table."""
        table = TableInfo(
            name="memberships",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("user_id", "INTEGER", False, False),
                ColumnInfo("group_id", "INTEGER", False, False),
                ColumnInfo("role", "TEXT", True, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("user_id", "users", "id", None, None),
                ForeignKeyInfo("group_id", "groups", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is True

    def test_not_join_table_with_business_cols(self) -> None:
        """Two FKs + 2+ business columns → NOT a join table."""
        table = TableInfo(
            name="orders",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("user_id", "INTEGER", False, False),
                ColumnInfo("product_id", "INTEGER", False, False),
                ColumnInfo("quantity", "INTEGER", False, False),
                ColumnInfo("price", "REAL", False, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("user_id", "users", "id", None, None),
                ForeignKeyInfo("product_id", "products", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is False

    def test_not_join_table_single_fk(self) -> None:
        """Only one FK → NOT a join table."""
        table = TableInfo(
            name="posts",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("user_id", "INTEGER", False, False),
                ColumnInfo("title", "TEXT", False, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("user_id", "users", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is False

    def test_not_join_table_no_fks(self) -> None:
        """No FKs → NOT a join table."""
        table = TableInfo(
            name="users",
            schema=None,
            columns=(ColumnInfo("id", "INTEGER", False, True),),
            foreign_keys=(),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is False

    def test_works_without_underscore_name(self) -> None:
        """Detection is structure-based — name doesn't matter."""
        table = TableInfo(
            name="Membership",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("user_id", "INTEGER", False, False),
                ColumnInfo("team_id", "INTEGER", False, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("user_id", "users", "id", None, None),
                ForeignKeyInfo("team_id", "teams", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        assert self.detector.is_join_table(table) is True


# ── PatternDetector tests ───────────────────────────────────────


class TestPatternDetector:
    """Detects schema patterns with graduated confidence."""

    def setup_method(self) -> None:
        self.detector = PatternDetector()

    def _make_table_with_cols(
        self,
        name: str,
        col_names: list[str],
        self_fk: bool = False,
    ) -> TableInfo:
        """Helper: build table with named columns."""
        columns = tuple(
            ColumnInfo(c, "TEXT", True, c == "id") for c in col_names
        )
        fks: tuple[ForeignKeyInfo, ...] = ()
        if self_fk:
            fks = (ForeignKeyInfo("parent_id", name, "id", None, None),)
        return TableInfo(
            name=name,
            schema=None,
            columns=columns,
            foreign_keys=fks,
            indexes=(),
            row_count_estimate=0,
        )

    # ── Audit trail ──

    def test_audit_trail_full(self) -> None:
        """created_at + updated_at + created_by → 0.90."""
        table = self._make_table_with_cols(
            "orders", ["id", "name", "created_at", "updated_at", "created_by"]
        )
        results = self.detector.detect_all(table)
        audit = [r for r in results if r[0] == SchemaPatternType.AUDIT_TRAIL]
        assert len(audit) == 1
        assert audit[0][1] == 0.90

    def test_audit_trail_timestamps_only(self) -> None:
        """created_at + updated_at without user tracking → 0.70."""
        table = self._make_table_with_cols(
            "orders", ["id", "name", "created_at", "updated_at"]
        )
        results = self.detector.detect_all(table)
        audit = [r for r in results if r[0] == SchemaPatternType.AUDIT_TRAIL]
        assert len(audit) == 1
        assert audit[0][1] == 0.70

    def test_no_audit_trail_just_created_at(self) -> None:
        """Just created_at alone → NOT audit trail."""
        table = self._make_table_with_cols(
            "events", ["id", "name", "created_at"]
        )
        results = self.detector.detect_all(table)
        audit = [r for r in results if r[0] == SchemaPatternType.AUDIT_TRAIL]
        assert len(audit) == 0

    # ── Soft delete ──

    def test_soft_delete_deleted_at(self) -> None:
        """deleted_at → 0.90."""
        table = self._make_table_with_cols(
            "users", ["id", "name", "deleted_at"]
        )
        results = self.detector.detect_all(table)
        soft = [r for r in results if r[0] == SchemaPatternType.SOFT_DELETE]
        assert len(soft) == 1
        assert soft[0][1] == 0.90

    def test_soft_delete_is_deleted(self) -> None:
        """is_deleted → 0.80."""
        table = self._make_table_with_cols(
            "users", ["id", "name", "is_deleted"]
        )
        results = self.detector.detect_all(table)
        soft = [r for r in results if r[0] == SchemaPatternType.SOFT_DELETE]
        assert len(soft) == 1
        assert soft[0][1] == 0.80

    def test_soft_delete_archived_at(self) -> None:
        """archived_at → 0.60."""
        table = self._make_table_with_cols(
            "posts", ["id", "body", "archived_at"]
        )
        results = self.detector.detect_all(table)
        soft = [r for r in results if r[0] == SchemaPatternType.SOFT_DELETE]
        assert len(soft) == 1
        assert soft[0][1] == 0.60

    def test_soft_delete_removed_at(self) -> None:
        """removed_at → 0.70."""
        table = self._make_table_with_cols(
            "items", ["id", "name", "removed_at"]
        )
        results = self.detector.detect_all(table)
        soft = [r for r in results if r[0] == SchemaPatternType.SOFT_DELETE]
        assert len(soft) == 1
        assert soft[0][1] == 0.70

    # ── Tree hierarchy ──

    def test_tree_hierarchy_self_fk(self) -> None:
        """Self-referencing FK → 0.95."""
        table = self._make_table_with_cols(
            "categories", ["id", "name", "parent_id"], self_fk=True
        )
        results = self.detector.detect_all(table)
        tree = [r for r in results if r[0] == SchemaPatternType.TREE_HIERARCHY]
        assert len(tree) == 1
        assert tree[0][1] == 0.95

    def test_no_tree_without_self_fk(self) -> None:
        """FK to another table → NOT tree hierarchy."""
        table = TableInfo(
            name="posts",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("parent_id", "INTEGER", True, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("parent_id", "users", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        results = self.detector.detect_all(table)
        tree = [r for r in results if r[0] == SchemaPatternType.TREE_HIERARCHY]
        assert len(tree) == 0

    # ── Polymorphic ──

    def test_polymorphic_type_id_pair(self) -> None:
        """commentable_type + commentable_id → 0.85."""
        table = self._make_table_with_cols(
            "comments",
            ["id", "body", "commentable_type", "commentable_id"],
        )
        results = self.detector.detect_all(table)
        poly = [r for r in results if r[0] == SchemaPatternType.POLYMORPHIC]
        assert len(poly) == 1
        assert poly[0][1] == 0.85

    def test_no_polymorphic_type_without_id(self) -> None:
        """Only _type column without matching _id → no detection."""
        table = self._make_table_with_cols(
            "things", ["id", "item_type", "name"]
        )
        results = self.detector.detect_all(table)
        poly = [r for r in results if r[0] == SchemaPatternType.POLYMORPHIC]
        assert len(poly) == 0

    # ── Enum table ──

    def test_enum_table_with_name_col(self) -> None:
        """Small table with 'name' column and no FKs → 0.70."""
        table = self._make_table_with_cols(
            "statuses", ["id", "name"]
        )
        results = self.detector.detect_all(table)
        enum_results = [r for r in results if r[0] == SchemaPatternType.ENUM_TABLE]
        assert len(enum_results) == 1
        assert enum_results[0][1] == 0.70

    def test_enum_table_with_label_col(self) -> None:
        """Small table with 'label' column → detected."""
        table = self._make_table_with_cols(
            "priorities", ["id", "label", "code"]
        )
        results = self.detector.detect_all(table)
        enum_results = [r for r in results if r[0] == SchemaPatternType.ENUM_TABLE]
        assert len(enum_results) == 1

    def test_no_enum_large_table(self) -> None:
        """Table with >4 columns → NOT enum table."""
        table = self._make_table_with_cols(
            "users", ["id", "name", "email", "phone", "address"]
        )
        results = self.detector.detect_all(table)
        enum_results = [r for r in results if r[0] == SchemaPatternType.ENUM_TABLE]
        assert len(enum_results) == 0

    def test_no_enum_with_fks(self) -> None:
        """Table with FKs → NOT enum table."""
        table = TableInfo(
            name="items",
            schema=None,
            columns=(
                ColumnInfo("id", "INTEGER", False, True),
                ColumnInfo("name", "TEXT", False, False),
            ),
            foreign_keys=(
                ForeignKeyInfo("category_id", "categories", "id", None, None),
            ),
            indexes=(),
            row_count_estimate=0,
        )
        results = self.detector.detect_all(table)
        enum_results = [r for r in results if r[0] == SchemaPatternType.ENUM_TABLE]
        assert len(enum_results) == 0

    # ── Multiple patterns ──

    def test_multiple_patterns_detected(self) -> None:
        """A table can have multiple patterns (e.g., audit + soft delete)."""
        table = self._make_table_with_cols(
            "posts",
            ["id", "body", "created_at", "updated_at", "deleted_at"],
        )
        results = self.detector.detect_all(table)
        pattern_types = {r[0] for r in results}
        assert SchemaPatternType.AUDIT_TRAIL in pattern_types
        assert SchemaPatternType.SOFT_DELETE in pattern_types


# ── Business purpose inference ──────────────────────────────────


class TestInferPurpose:
    """_infer_purpose derives purpose from table names."""

    def test_log_table(self) -> None:
        assert "log" in _infer_purpose("activity_log")

    def test_logs_table(self) -> None:
        assert "log" in _infer_purpose("access_logs")

    def test_history_table(self) -> None:
        assert "historical" in _infer_purpose("price_history")

    def test_config_table(self) -> None:
        assert "configuration" in _infer_purpose("app_config")

    def test_settings_table(self) -> None:
        assert "configuration" in _infer_purpose("user_settings")

    def test_dim_table(self) -> None:
        assert "dimension" in _infer_purpose("dim_customer")

    def test_fact_table(self) -> None:
        assert "fact" in _infer_purpose("fact_sales")

    def test_regular_table_singularizes(self) -> None:
        result = _infer_purpose("orders")
        assert "order" in result

    def test_ies_pluralization(self) -> None:
        result = _infer_purpose("categories")
        assert "category" in result


# ── KnowledgeExtractor integration ──────────────────────────────


def _build_ecommerce_snapshot() -> SchemaSnapshot:
    """Build a realistic e-commerce schema snapshot for testing."""
    users = TableInfo(
        name="users",
        schema=None,
        columns=(
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("name", "TEXT", False, False),
            ColumnInfo("email", "TEXT", False, False),
            ColumnInfo("created_at", "TEXT", True, False),
            ColumnInfo("updated_at", "TEXT", True, False),
        ),
        foreign_keys=(),
        indexes=(IndexInfo("idx_email", ("email",), True),),
        row_count_estimate=100,
    )

    categories = TableInfo(
        name="categories",
        schema=None,
        columns=(
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("name", "TEXT", False, False),
            ColumnInfo("parent_id", "INTEGER", True, False),
        ),
        foreign_keys=(
            ForeignKeyInfo("parent_id", "categories", "id", None, None),
        ),
        indexes=(),
        row_count_estimate=20,
    )

    products = TableInfo(
        name="products",
        schema=None,
        columns=(
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("name", "TEXT", False, False),
            ColumnInfo("price", "REAL", False, False),
            ColumnInfo("category_id", "INTEGER", True, False),
            ColumnInfo("created_at", "TEXT", True, False),
            ColumnInfo("updated_at", "TEXT", True, False),
            ColumnInfo("deleted_at", "TEXT", True, False),
        ),
        foreign_keys=(
            ForeignKeyInfo("category_id", "categories", "id", None, None),
        ),
        indexes=(),
        row_count_estimate=500,
    )

    # Join table: product_tags
    product_tags = TableInfo(
        name="product_tags",
        schema=None,
        columns=(
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("product_id", "INTEGER", False, False),
            ColumnInfo("tag_id", "INTEGER", False, False),
        ),
        foreign_keys=(
            ForeignKeyInfo("product_id", "products", "id", None, None),
            ForeignKeyInfo("tag_id", "tags", "id", None, None),
        ),
        indexes=(),
        row_count_estimate=1000,
    )

    tags = TableInfo(
        name="tags",
        schema=None,
        columns=(
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("name", "TEXT", False, False),
        ),
        foreign_keys=(),
        indexes=(),
        row_count_estimate=50,
    )

    return SchemaSnapshot(
        database_name="ecommerce.db",
        dialect="sqlite",
        tables=(users, categories, products, product_tags, tags),
    )


class TestKnowledgeExtractor:
    """Full extraction pipeline from SchemaSnapshot → SchemaKnowledge."""

    def setup_method(self) -> None:
        self.extractor = KnowledgeExtractor()
        self.snapshot = _build_ecommerce_snapshot()

    def test_extract_returns_schema_knowledge(self) -> None:
        result = self.extractor.extract(self.snapshot)
        assert isinstance(result, SchemaKnowledge)

    def test_join_table_excluded_from_entities(self) -> None:
        """product_tags is a join table — should not be an entity."""
        result = self.extractor.extract(self.snapshot)
        entity_names = {e.table_name for e in result.entities}
        assert "product_tags" not in entity_names

    def test_non_join_tables_are_entities(self) -> None:
        result = self.extractor.extract(self.snapshot)
        entity_names = {e.table_name for e in result.entities}
        assert "users" in entity_names
        assert "categories" in entity_names
        assert "products" in entity_names
        assert "tags" in entity_names

    def test_join_table_creates_co_occurs(self) -> None:
        """product_tags join table → CO_OCCURS between products and tags."""
        result = self.extractor.extract(self.snapshot)
        co_occurs = [
            r for r in result.relationships
            if r.synapse_type == SynapseType.CO_OCCURS
        ]
        assert len(co_occurs) >= 1
        sources_targets = {(r.source_table, r.target_table) for r in co_occurs}
        assert ("products", "tags") in sources_targets

    def test_fk_relationships_mapped(self) -> None:
        """category_id FK → IS_A relationship."""
        result = self.extractor.extract(self.snapshot)
        cat_rels = [
            r for r in result.relationships
            if r.source_table == "products" and r.target_table == "categories"
        ]
        assert len(cat_rels) == 1
        assert cat_rels[0].synapse_type == SynapseType.IS_A

    def test_tree_hierarchy_detected(self) -> None:
        """categories.parent_id self-FK → TREE_HIERARCHY pattern."""
        result = self.extractor.extract(self.snapshot)
        tree_patterns = [
            p for p in result.patterns
            if p.pattern_type == SchemaPatternType.TREE_HIERARCHY
        ]
        assert len(tree_patterns) == 1
        assert tree_patterns[0].table_name == "categories"

    def test_soft_delete_detected(self) -> None:
        """products.deleted_at → SOFT_DELETE pattern."""
        result = self.extractor.extract(self.snapshot)
        soft_del = [
            p for p in result.patterns
            if p.pattern_type == SchemaPatternType.SOFT_DELETE
        ]
        assert len(soft_del) == 1
        assert soft_del[0].table_name == "products"

    def test_audit_trail_detected(self) -> None:
        """products.created_at + updated_at → AUDIT_TRAIL pattern."""
        result = self.extractor.extract(self.snapshot)
        audit = [
            p for p in result.patterns
            if p.pattern_type == SchemaPatternType.AUDIT_TRAIL
        ]
        # products and users both have created_at + updated_at
        audit_tables = {p.table_name for p in audit}
        assert "products" in audit_tables
        assert "users" in audit_tables

    def test_enum_table_detected(self) -> None:
        """tags (small, has 'name', no FKs) → ENUM_TABLE."""
        result = self.extractor.extract(self.snapshot)
        enum_patterns = [
            p for p in result.patterns
            if p.pattern_type == SchemaPatternType.ENUM_TABLE
        ]
        enum_tables = {p.table_name for p in enum_patterns}
        assert "tags" in enum_tables

    def test_properties_extracted(self) -> None:
        """Column properties exist for all non-join tables."""
        result = self.extractor.extract(self.snapshot)
        prop_tables = {p.table_name for p in result.properties}
        assert "users" in prop_tables
        assert "products" in prop_tables
        # Join table excluded
        assert "product_tags" not in prop_tables

    def test_entity_description_is_semantic(self) -> None:
        """Entity descriptions contain table name and purpose."""
        result = self.extractor.extract(self.snapshot)
        users_entity = next(
            e for e in result.entities if e.table_name == "users"
        )
        assert "users" in users_entity.description
        assert "Columns:" in users_entity.description

    def test_entity_with_fk_context(self) -> None:
        """Entity description includes FK targets."""
        result = self.extractor.extract(self.snapshot)
        products_entity = next(
            e for e in result.entities if e.table_name == "products"
        )
        assert "Links to:" in products_entity.description
        assert "categories" in products_entity.description

    def test_entity_confidence_without_comment(self) -> None:
        """Tables without comments get 0.75 confidence."""
        result = self.extractor.extract(self.snapshot)
        users_entity = next(
            e for e in result.entities if e.table_name == "users"
        )
        assert users_entity.confidence == 0.75

    def test_entity_confidence_with_comment(self) -> None:
        """Tables with comments get 0.90 confidence."""
        table_with_comment = TableInfo(
            name="docs",
            schema=None,
            columns=(ColumnInfo("id", "INTEGER", False, True),),
            foreign_keys=(),
            indexes=(),
            row_count_estimate=0,
            comment="Documentation entries",
        )
        snapshot = SchemaSnapshot(
            database_name="test.db",
            dialect="sqlite",
            tables=(table_with_comment,),
        )
        result = self.extractor.extract(snapshot)
        assert result.entities[0].confidence == 0.90

    def test_pattern_descriptions_human_readable(self) -> None:
        """Pattern descriptions are human-readable strings."""
        result = self.extractor.extract(self.snapshot)
        for pattern in result.patterns:
            assert len(pattern.description) > 10
            assert pattern.table_name in pattern.description


# ── SchemaPatternType enum ──────────────────────────────────────


class TestSchemaPatternType:
    """StrEnum values match expected strings."""

    def test_values(self) -> None:
        assert SchemaPatternType.AUDIT_TRAIL == "audit_trail"
        assert SchemaPatternType.SOFT_DELETE == "soft_delete"
        assert SchemaPatternType.TREE_HIERARCHY == "tree_hierarchy"
        assert SchemaPatternType.POLYMORPHIC == "polymorphic"
        assert SchemaPatternType.ENUM_TABLE == "enum_table"
