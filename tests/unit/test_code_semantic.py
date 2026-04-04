"""Tests for Phase 2: Code-Semantic Encoding.

Covers:
- Code synapse types (IMPORTS, CALLS, DEPENDS_ON, INHERITS, IMPLEMENTS, DEFINED_IN, RAISES)
- Code entity subtypes (FUNCTION_NAME, CLASS_NAME, MODULE_NAME, PACKAGE_NAME, ERROR_TYPE)
- Compound identifier keyword preservation (PascalCase, camelCase, snake_case)
- Code relation patterns in RelationExtractor
- Stack trace structured extraction
"""

from __future__ import annotations

import pytest

from neural_memory.core.synapse import (
    BIDIRECTIONAL_TYPES,
    INVERSE_TYPES,
    SynapseType,
)
from neural_memory.extraction.entities import (
    EntityExtractor,
    EntitySubtype,
    EntityType,
)
from neural_memory.extraction.keywords import (
    extract_keywords,
    extract_weighted_keywords,
)
from neural_memory.extraction.relations import RelationExtractor

# ── Code Synapse Types ────────────────────────────────────────────


class TestCodeSynapseTypes:
    """Verify 7 new code-semantic synapse types exist and are configured."""

    def test_imports_type(self) -> None:
        assert SynapseType.IMPORTS == "imports"

    def test_calls_type(self) -> None:
        assert SynapseType.CALLS == "calls"

    def test_depends_on_type(self) -> None:
        assert SynapseType.DEPENDS_ON == "depends_on"

    def test_inherits_type(self) -> None:
        assert SynapseType.INHERITS == "inherits"

    def test_implements_type(self) -> None:
        assert SynapseType.IMPLEMENTS == "implements"

    def test_defined_in_type(self) -> None:
        assert SynapseType.DEFINED_IN == "defined_in"

    def test_raises_type(self) -> None:
        assert SynapseType.RAISES == "raises"

    def test_code_types_are_unidirectional(self) -> None:
        """Code synapses should be unidirectional (not in BIDIRECTIONAL_TYPES)."""
        code_types = {
            SynapseType.IMPORTS,
            SynapseType.CALLS,
            SynapseType.DEPENDS_ON,
            SynapseType.INHERITS,
            SynapseType.IMPLEMENTS,
            SynapseType.DEFINED_IN,
            SynapseType.RAISES,
        }
        for st in code_types:
            assert st not in BIDIRECTIONAL_TYPES, f"{st} should not be bidirectional"

    def test_imports_depends_on_inverse(self) -> None:
        """IMPORTS and DEPENDS_ON should be inverses."""
        assert INVERSE_TYPES[SynapseType.IMPORTS] == SynapseType.DEPENDS_ON
        assert INVERSE_TYPES[SynapseType.DEPENDS_ON] == SynapseType.IMPORTS

    def test_inherits_is_a_inverse(self) -> None:
        """INHERITS should map to IS_A as inverse."""
        assert INVERSE_TYPES[SynapseType.INHERITS] == SynapseType.IS_A


# ── Code Entity Subtypes ─────────────────────────────────────────


class TestCodeEntitySubtypes:
    """Verify 5 new code entity subtypes exist."""

    def test_function_name_subtype(self) -> None:
        assert EntitySubtype.FUNCTION_NAME == "function_name"

    def test_class_name_subtype(self) -> None:
        assert EntitySubtype.CLASS_NAME == "class_name"

    def test_module_name_subtype(self) -> None:
        assert EntitySubtype.MODULE_NAME == "module_name"

    def test_package_name_subtype(self) -> None:
        assert EntitySubtype.PACKAGE_NAME == "package_name"

    def test_error_type_subtype(self) -> None:
        assert EntitySubtype.ERROR_TYPE == "error_type"


class TestCodeEntityExtraction:
    """Entity extractor classifies code entities with correct subtypes."""

    @pytest.fixture()
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_pascal_case_as_class(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("ReflexPipeline processes data efficiently")
        code_entities = [e for e in entities if e.type == EntityType.CODE]
        class_ents = [e for e in code_entities if e.subtype == EntitySubtype.CLASS_NAME]
        assert any(e.text == "ReflexPipeline" for e in class_ents)

    def test_snake_case_function_call(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Called extract_keywords() on the input")
        fn_ents = [e for e in entities if e.subtype == EntitySubtype.FUNCTION_NAME]
        assert any(e.text == "extract_keywords" for e in fn_ents)

    def test_dotted_module_path(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Import from neural_memory.engine.retrieval")
        mod_ents = [e for e in entities if e.subtype == EntitySubtype.MODULE_NAME]
        assert any("neural_memory.engine" in e.text for e in mod_ents)

    def test_error_type_extraction(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Caught a ValueError in the handler")
        err_ents = [e for e in entities if e.subtype == EntitySubtype.ERROR_TYPE]
        assert any(e.text == "ValueError" for e in err_ents)

    def test_connection_refused_error(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Got ConnectionRefusedError when connecting")
        err_ents = [e for e in entities if e.subtype == EntitySubtype.ERROR_TYPE]
        assert any(e.text == "ConnectionRefusedError" for e in err_ents)

    def test_package_from_pip_install(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Run pip install aiohttp to add the dependency")
        pkg_ents = [e for e in entities if e.subtype == EntitySubtype.PACKAGE_NAME]
        assert any(e.text == "aiohttp" for e in pkg_ents)

    def test_package_from_import(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("Use import numpy for array operations")
        pkg_ents = [e for e in entities if e.subtype == EntitySubtype.PACKAGE_NAME]
        assert any(e.text == "numpy" for e in pkg_ents)

    def test_error_not_classified_as_class(self, extractor: EntityExtractor) -> None:
        """Error types should NOT also appear as CLASS_NAME."""
        entities = extractor.extract("KeyError was raised")
        class_ents = [e for e in entities if e.subtype == EntitySubtype.CLASS_NAME]
        assert not any(e.text == "KeyError" for e in class_ents)


# ── Compound Identifier Keywords ─────────────────────────────────


class TestCompoundIdentifierKeywords:
    """Keywords extractor preserves code identifiers as whole tokens."""

    def test_pascal_case_preserved(self) -> None:
        kws = extract_keywords("ReflexPipeline handles memory encoding")
        assert "ReflexPipeline" in kws

    def test_snake_case_preserved(self) -> None:
        kws = extract_keywords("The extract_keywords function parses text")
        assert "extract_keywords" in kws

    def test_camel_case_preserved(self) -> None:
        kws = extract_keywords("Call processData to transform input")
        assert "processData" in kws

    def test_pascal_case_sub_words_also_present(self) -> None:
        """PascalCase identifiers also add lowercase sub-words."""
        weighted = extract_weighted_keywords("ReflexPipeline is the core")
        texts = {kw.text for kw in weighted}
        assert "ReflexPipeline" in texts
        assert "reflex" in texts or "pipeline" in texts

    def test_snake_case_parts_also_present(self) -> None:
        """snake_case identifiers also add individual parts."""
        weighted = extract_weighted_keywords("The extract_keywords function runs")
        texts = {kw.text for kw in weighted}
        assert "extract_keywords" in texts
        assert "extract" in texts or "keywords" in texts

    def test_code_identifiers_have_high_weight(self) -> None:
        """Compound identifiers should have weight >= 1.0."""
        weighted = extract_weighted_keywords("ReflexPipeline processes data")
        pascal_kw = next((kw for kw in weighted if kw.text == "ReflexPipeline"), None)
        assert pascal_kw is not None
        assert pascal_kw.weight >= 1.0

    def test_error_type_keyword(self) -> None:
        kws = extract_keywords("Got ValueError from parsing input")
        assert "ValueError" in kws

    def test_dotted_module_keyword(self) -> None:
        kws = extract_keywords("Import neural_memory.engine for retrieval")
        assert "neural_memory.engine" in kws


# ── Code Relation Patterns ───────────────────────────────────────


class TestCodeRelationPatterns:
    """RelationExtractor detects code-semantic relationships."""

    @pytest.fixture()
    def extractor(self) -> RelationExtractor:
        return RelationExtractor()

    def test_imports_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("engine imports retrieval_context")
        imports_rels = [c for c in candidates if c.synapse_type == SynapseType.IMPORTS]
        assert len(imports_rels) >= 1

    def test_calls_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("consolidation_handler calls merge_neurons")
        calls_rels = [c for c in candidates if c.synapse_type == SynapseType.CALLS]
        assert len(calls_rels) >= 1

    def test_depends_on_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("ReflexPipeline depends on ActivationStore")
        dep_rels = [c for c in candidates if c.synapse_type == SynapseType.DEPENDS_ON]
        assert len(dep_rels) >= 1

    def test_inherits_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("InfinityDB inherits from SQLiteStorage")
        inh_rels = [c for c in candidates if c.synapse_type == SynapseType.INHERITS]
        assert len(inh_rels) >= 1

    def test_implements_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("PythonExtractor implements BaseExtractor")
        impl_rels = [c for c in candidates if c.synapse_type == SynapseType.IMPLEMENTS]
        assert len(impl_rels) >= 1

    def test_defined_in_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("SynapseType defined in core/synapse.py")
        def_rels = [c for c in candidates if c.synapse_type == SynapseType.DEFINED_IN]
        assert len(def_rels) >= 1

    def test_raises_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("validate_config raises ValueError")
        raise_rels = [c for c in candidates if c.synapse_type == SynapseType.RAISES]
        assert len(raise_rels) >= 1

    def test_extends_relation(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("CustomStore extends BaseStorage")
        inh_rels = [c for c in candidates if c.synapse_type == SynapseType.INHERITS]
        assert len(inh_rels) >= 1

    def test_requires_as_depends_on(self, extractor: RelationExtractor) -> None:
        candidates = extractor.extract("sync_module requires active_license")
        dep_rels = [c for c in candidates if c.synapse_type == SynapseType.DEPENDS_ON]
        assert len(dep_rels) >= 1


# ── Stack Trace Extraction ───────────────────────────────────────


class TestStackTraceExtraction:
    """Entity extractor handles Python stack traces."""

    @pytest.fixture()
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_traceback_frame_function(self, extractor: EntityExtractor) -> None:
        text = 'File "src/engine/retrieval.py", line 42, in search_neurons'
        entities = extractor.extract(text)
        fn_ents = [e for e in entities if e.subtype == EntitySubtype.FUNCTION_NAME]
        assert any(e.text == "search_neurons" for e in fn_ents)

    def test_traceback_frame_raw_value(self, extractor: EntityExtractor) -> None:
        text = 'File "src/engine/retrieval.py", line 42, in search_neurons'
        entities = extractor.extract(text)
        fn_ent = next(e for e in entities if e.text == "search_neurons")
        assert "retrieval.py" in fn_ent.raw_value
        assert "42" in fn_ent.raw_value

    def test_traceback_error_line(self, extractor: EntityExtractor) -> None:
        text = "ValueError: invalid literal for int() with base 10"
        entities = extractor.extract(text)
        err_ents = [e for e in entities if e.subtype == EntitySubtype.ERROR_TYPE]
        assert any(e.text == "ValueError" for e in err_ents)

    def test_traceback_error_raw_value(self, extractor: EntityExtractor) -> None:
        text = "KeyError: 'missing_key'"
        entities = extractor.extract(text)
        err_ent = next(e for e in entities if e.text == "KeyError")
        assert "missing_key" in err_ent.raw_value

    def test_module_not_found_error(self, extractor: EntityExtractor) -> None:
        text = "ModuleNotFoundError: No module named 'neural_memory.pro'"
        entities = extractor.extract(text)
        # Should detect as either error type from ERROR_TYPE_PATTERN or TRACEBACK_ERROR_PATTERN
        err_ents = [e for e in entities if e.subtype == EntitySubtype.ERROR_TYPE]
        # ModuleNotFoundError doesn't end with just "Error" — it has "Error" in it
        assert len(err_ents) >= 1

    def test_full_traceback(self, extractor: EntityExtractor) -> None:
        """Parse a realistic multi-line Python traceback."""
        text = (
            "Traceback (most recent call last):\n"
            '  File "src/mcp/server.py", line 120, in handle_request\n'
            "    result = await process_tool(params)\n"
            '  File "src/mcp/tools.py", line 45, in process_tool\n'
            "    data = validate(params)\n"
            "TypeError: validate() missing 1 required argument"
        )
        entities = extractor.extract(text)
        fn_names = {e.text for e in entities if e.subtype == EntitySubtype.FUNCTION_NAME}
        err_names = {e.text for e in entities if e.subtype == EntitySubtype.ERROR_TYPE}
        assert "handle_request" in fn_names
        assert "process_tool" in fn_names
        assert "TypeError" in err_names

    def test_traceback_skips_module_frame(self, extractor: EntityExtractor) -> None:
        """<module> frames should be skipped."""
        text = 'File "main.py", line 1, in <module>'
        entities = extractor.extract(text)
        fn_ents = [e for e in entities if e.subtype == EntitySubtype.FUNCTION_NAME]
        assert not any(e.text == "<module>" for e in fn_ents)
