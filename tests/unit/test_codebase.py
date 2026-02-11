"""Tests for codebase indexing: git context, AST extraction, and encoding."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.extraction.codebase import (
    CodeSymbolType,
    PythonExtractor,
    RegexExtractor,
    get_extractor,
)
from neural_memory.git_context import GitContext, detect_git_context


class TestGitContext:
    """Tests for git context detection."""

    def test_detect_git_context(self) -> None:
        """Returns GitContext in a git repo."""
        ctx = detect_git_context(Path("."))
        # This test runs inside the NeuralMemory repo itself
        assert ctx is not None
        assert isinstance(ctx, GitContext)

    def test_detect_git_context_not_repo(self, tmp_path: Path) -> None:
        """Returns None outside a git repo."""
        ctx = detect_git_context(tmp_path)
        assert ctx is None

    def test_git_context_fields(self) -> None:
        """branch, commit, repo_name are not empty."""
        ctx = detect_git_context(Path("."))
        assert ctx is not None
        assert ctx.branch
        assert ctx.commit
        assert ctx.repo_name
        assert ctx.repo_root


class TestPythonExtractor:
    """Tests for Python AST extraction."""

    @pytest.fixture
    def extractor(self) -> PythonExtractor:
        return PythonExtractor()

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        """Create a sample Python file for extraction."""
        source = textwrap.dedent('''\
            """Module docstring."""

            import os
            from pathlib import Path

            MAX_SIZE = 1024
            DEFAULT_NAME = "test"

            class Animal:
                """An animal."""

                def speak(self) -> str:
                    """Make a sound."""
                    return "..."

                def move(self, distance: int) -> None:
                    pass

            class Dog(Animal):
                """A dog."""

                def speak(self) -> str:
                    return "woof"

            def greet(name: str) -> str:
                """Greet someone."""
                return f"Hello, {name}"

            async def fetch_data(url: str) -> dict:
                """Fetch data from URL."""
                return {}
        ''')
        file_path = tmp_path / "sample.py"
        file_path.write_text(source, encoding="utf-8")
        return file_path

    def test_extract_function(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Finds FunctionDef with name, args, docstring."""
        symbols, _ = extractor.extract_file(sample_file)
        funcs = [s for s in symbols if s.name == "greet"]

        assert len(funcs) == 1
        func = funcs[0]
        assert func.symbol_type == CodeSymbolType.FUNCTION
        assert func.docstring == "Greet someone."
        assert func.signature is not None
        assert "name: str" in func.signature
        assert "-> str" in func.signature
        assert func.parent is None

    def test_extract_async_function(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Finds AsyncFunctionDef."""
        symbols, _ = extractor.extract_file(sample_file)
        async_funcs = [s for s in symbols if s.name == "fetch_data"]

        assert len(async_funcs) == 1
        func = async_funcs[0]
        assert func.symbol_type == CodeSymbolType.FUNCTION
        assert func.signature is not None
        assert "async def" in func.signature

    def test_extract_class(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Finds ClassDef with bases."""
        symbols, relationships = extractor.extract_file(sample_file)
        classes = [s for s in symbols if s.symbol_type == CodeSymbolType.CLASS]

        assert len(classes) == 2
        animal = next(c for c in classes if c.name == "Animal")
        assert animal.docstring == "An animal."

        dog = next(c for c in classes if c.name == "Dog")
        assert dog.docstring == "A dog."

        # Dog IS_A Animal
        is_a_rels = [r for r in relationships if r.relation == "is_a"]
        assert any(r.source == "Dog" and r.target == "Animal" for r in is_a_rels)

    def test_extract_method(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Method has parent=class_name."""
        symbols, _ = extractor.extract_file(sample_file)
        methods = [s for s in symbols if s.symbol_type == CodeSymbolType.METHOD]

        # Animal.speak, Animal.move, Dog.speak
        assert len(methods) == 3
        animal_speak = next(m for m in methods if m.name == "speak" and m.parent == "Animal")
        assert animal_speak.parent == "Animal"
        assert animal_speak.docstring == "Make a sound."

    def test_extract_imports(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Finds Import and ImportFrom."""
        symbols, relationships = extractor.extract_file(sample_file)
        imports = [s for s in symbols if s.symbol_type == CodeSymbolType.IMPORT]

        import_names = {i.name for i in imports}
        assert "os" in import_names
        assert "Path" in import_names

        # Check import relationships
        import_rels = [r for r in relationships if r.relation == "imports"]
        import_targets = {r.target for r in import_rels}
        assert "os" in import_targets
        assert "pathlib.Path" in import_targets

    def test_extract_constants(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """Finds SCREAMING_SNAKE top-level assigns."""
        symbols, _ = extractor.extract_file(sample_file)
        constants = [s for s in symbols if s.symbol_type == CodeSymbolType.CONSTANT]

        const_names = {c.name for c in constants}
        assert "MAX_SIZE" in const_names
        assert "DEFAULT_NAME" in const_names

    def test_extract_relationships(self, extractor: PythonExtractor, sample_file: Path) -> None:
        """CONTAINS, IS_A, imports relationships are created."""
        _, relationships = extractor.extract_file(sample_file)

        relation_types = {r.relation for r in relationships}
        assert "contains" in relation_types
        assert "is_a" in relation_types
        assert "imports" in relation_types

    def test_extract_empty_file(self, extractor: PythonExtractor, tmp_path: Path) -> None:
        """Empty file returns empty lists."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")

        symbols, relationships = extractor.extract_file(empty_file)
        assert symbols == []
        assert relationships == []


class TestCodebaseEncoder:
    """Tests for codebase encoder."""

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        """Create a sample Python file for encoding."""
        source = textwrap.dedent('''\
            """Module."""

            import os

            MAX_SIZE = 1024

            class Foo:
                """A foo."""

                def bar(self) -> str:
                    return "bar"

            def baz(x: int) -> int:
                """Compute baz."""
                return x + 1
        ''')
        file_path = tmp_path / "sample.py"
        file_path.write_text(source, encoding="utf-8")
        return file_path

    @pytest.mark.asyncio
    async def test_index_file(self, sample_file: Path) -> None:
        """Creates file neuron (SPATIAL) + symbol neurons."""
        from neural_memory.core.neuron import NeuronType
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        mock_storage = AsyncMock()
        mock_config = MagicMock()

        encoder = CodebaseEncoder(mock_storage, mock_config)
        result = await encoder.index_file(sample_file)

        assert len(result.neurons_created) >= 1
        # First neuron is the file (SPATIAL)
        file_neuron = result.neurons_created[0]
        assert file_neuron.type == NeuronType.SPATIAL
        assert file_neuron.metadata["indexed"] is True

        # Check that symbol neurons were created
        symbol_types = {n.type for n in result.neurons_created[1:]}
        assert NeuronType.ACTION in symbol_types  # function/method
        assert NeuronType.CONCEPT in symbol_types  # class

    @pytest.mark.asyncio
    async def test_index_file_synapses(self, sample_file: Path) -> None:
        """Correct synapse types and weights."""
        from neural_memory.core.synapse import SynapseType
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        mock_storage = AsyncMock()
        mock_config = MagicMock()

        encoder = CodebaseEncoder(mock_storage, mock_config)
        result = await encoder.index_file(sample_file)

        assert len(result.synapses_created) >= 1
        synapse_types = {s.type for s in result.synapses_created}
        assert SynapseType.CONTAINS in synapse_types

    @pytest.mark.asyncio
    async def test_index_directory(self, tmp_path: Path) -> None:
        """Scans recursively, skips __pycache__."""
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        # Create files
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.py").write_text("y = 2", encoding="utf-8")

        # Create __pycache__ (should be skipped)
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "c.cpython-312.py").write_text("z = 3", encoding="utf-8")

        mock_storage = AsyncMock()
        mock_config = MagicMock()

        encoder = CodebaseEncoder(mock_storage, mock_config)
        results = await encoder.index_directory(tmp_path)

        indexed_files = [r.fiber.summary for r in results]
        assert len(results) == 2
        assert any("a.py" in s for s in indexed_files)
        assert any("b.py" in s for s in indexed_files)
        # __pycache__ skipped
        assert not any("c.cpython" in s for s in indexed_files)

    @pytest.mark.asyncio
    async def test_index_file_metadata(self, sample_file: Path) -> None:
        """Neurons have file_path, line_start, signature metadata."""
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        mock_storage = AsyncMock()
        mock_config = MagicMock()

        encoder = CodebaseEncoder(mock_storage, mock_config)
        result = await encoder.index_file(sample_file)

        # Find the 'baz' function neuron
        baz_neurons = [n for n in result.neurons_created if n.content == "baz"]
        assert len(baz_neurons) == 1
        baz = baz_neurons[0]

        assert baz.metadata["file_path"] == str(sample_file)
        assert baz.metadata["line_start"] > 0
        assert baz.metadata["signature"] is not None
        assert "x: int" in baz.metadata["signature"]

    @pytest.mark.asyncio
    async def test_index_fiber_created(self, sample_file: Path) -> None:
        """Each file produces one fiber."""
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        mock_storage = AsyncMock()
        mock_config = MagicMock()

        encoder = CodebaseEncoder(mock_storage, mock_config)
        result = await encoder.index_file(sample_file)

        assert result.fiber is not None
        assert "code_index" in result.fiber.tags
        assert result.fiber.anchor_neuron_id == result.neurons_created[0].id
        # Fiber should reference all neurons and synapses
        assert len(result.fiber.neuron_ids) == len(result.neurons_created)
        assert len(result.fiber.synapse_ids) == len(result.synapses_created)


class TestRegexExtractor:
    """Tests for regex-based extraction of non-Python languages."""

    # -- JavaScript --

    def test_javascript_functions(self, tmp_path: Path) -> None:
        """Finds function declarations and arrow functions."""
        source = textwrap.dedent("""\
            export async function fetchUser(id) {
                return await fetch(`/users/${id}`);
            }

            const processData = async (items) => {
                return items.map(x => x * 2);
            }

            function helper() { return 1; }
        """)
        f = tmp_path / "app.js"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("javascript")
        symbols, rels = ext.extract_file(f)

        names = {s.name for s in symbols}
        assert "fetchUser" in names
        assert "processData" in names
        assert "helper" in names

    def test_javascript_classes(self, tmp_path: Path) -> None:
        """Finds class declarations with inheritance."""
        source = textwrap.dedent("""\
            export class Animal {
                speak() { return "..."; }
            }

            class Dog extends Animal {
                speak() { return "woof"; }
            }
        """)
        f = tmp_path / "models.js"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("javascript")
        symbols, rels = ext.extract_file(f)

        class_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CLASS}
        assert "Animal" in class_names
        assert "Dog" in class_names

        is_a = [r for r in rels if r.relation == "is_a"]
        assert any(r.source == "Dog" and r.target == "Animal" for r in is_a)

    def test_javascript_imports(self, tmp_path: Path) -> None:
        """Finds ES module imports and require()."""
        source = textwrap.dedent("""\
            import React from 'react';
            import { useState } from 'react';
            const fs = require('fs');
        """)
        f = tmp_path / "index.js"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("javascript")
        symbols, _ = ext.extract_file(f)

        import_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.IMPORT}
        assert "react" in import_names
        assert "fs" in import_names

    def test_javascript_constants(self, tmp_path: Path) -> None:
        """Finds SCREAMING_SNAKE constants."""
        source = "export const MAX_RETRIES = 3;\nconst API_URL = 'http://localhost';\n"
        f = tmp_path / "config.js"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("javascript")
        symbols, _ = ext.extract_file(f)

        const_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CONSTANT}
        assert "MAX_RETRIES" in const_names
        assert "API_URL" in const_names

    # -- TypeScript --

    def test_typescript_functions(self, tmp_path: Path) -> None:
        """TS uses same patterns as JS."""
        source = "export function greet(name: string): string { return `Hello ${name}`; }\n"
        f = tmp_path / "utils.ts"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("typescript")
        symbols, _ = ext.extract_file(f)

        names = {s.name for s in symbols}
        assert "greet" in names

    # -- Go --

    def test_go_functions(self, tmp_path: Path) -> None:
        """Finds Go func declarations."""
        source = textwrap.dedent("""\
            package main

            func main() {
                fmt.Println("hello")
            }

            func helper(x int) int {
                return x + 1
            }
        """)
        f = tmp_path / "main.go"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("go")
        symbols, _ = ext.extract_file(f)

        func_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.FUNCTION}
        assert "main" in func_names
        assert "helper" in func_names

    def test_go_structs(self, tmp_path: Path) -> None:
        """Finds Go struct and interface types."""
        source = textwrap.dedent("""\
            type User struct {
                Name string
                Age  int
            }

            type Reader interface {
                Read(p []byte) (n int, err error)
            }
        """)
        f = tmp_path / "types.go"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("go")
        symbols, _ = ext.extract_file(f)

        class_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CLASS}
        assert "User" in class_names
        assert "Reader" in class_names

    def test_go_methods(self, tmp_path: Path) -> None:
        """Finds Go methods with receiver."""
        source = textwrap.dedent("""\
            func (u *User) String() string {
                return u.Name
            }
        """)
        f = tmp_path / "methods.go"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("go")
        symbols, _ = ext.extract_file(f)

        methods = [s for s in symbols if s.symbol_type == CodeSymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "String"
        assert methods[0].parent == "User"

    # -- Rust --

    def test_rust_functions(self, tmp_path: Path) -> None:
        """Finds Rust fn declarations."""
        source = textwrap.dedent("""\
            pub fn process(data: &[u8]) -> Result<(), Error> {
                Ok(())
            }

            async fn fetch_data(url: &str) -> String {
                String::new()
            }
        """)
        f = tmp_path / "lib.rs"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("rust")
        symbols, _ = ext.extract_file(f)

        func_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.FUNCTION}
        assert "process" in func_names
        assert "fetch_data" in func_names

    def test_rust_structs_and_traits(self, tmp_path: Path) -> None:
        """Finds struct, enum, trait."""
        source = textwrap.dedent("""\
            pub struct Config {
                name: String,
            }

            pub enum Status {
                Active,
                Inactive,
            }

            pub trait Processor {
                fn process(&self);
            }
        """)
        f = tmp_path / "types.rs"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("rust")
        symbols, _ = ext.extract_file(f)

        class_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CLASS}
        assert "Config" in class_names
        assert "Status" in class_names
        assert "Processor" in class_names

    def test_rust_imports(self, tmp_path: Path) -> None:
        """Finds use statements."""
        source = "use std::collections::HashMap;\nuse crate::config::Settings;\n"
        f = tmp_path / "imports.rs"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("rust")
        symbols, _ = ext.extract_file(f)

        import_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.IMPORT}
        assert "std::collections::HashMap" in import_names
        assert "crate::config::Settings" in import_names

    # -- Java --

    def test_java_classes(self, tmp_path: Path) -> None:
        """Finds Java class declarations."""
        source = textwrap.dedent("""\
            public class UserService {
                public String getName(int id) {
                    return "user";
                }
            }

            public abstract class BaseService {
            }
        """)
        f = tmp_path / "UserService.java"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("java")
        symbols, _ = ext.extract_file(f)

        class_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CLASS}
        assert "UserService" in class_names
        assert "BaseService" in class_names

    def test_java_methods(self, tmp_path: Path) -> None:
        """Finds Java methods."""
        source = textwrap.dedent("""\
            public class Foo {
                public String bar() { return ""; }
                private int baz(String s) { return 0; }
            }
        """)
        f = tmp_path / "Foo.java"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("java")
        symbols, _ = ext.extract_file(f)

        method_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.METHOD}
        assert "bar" in method_names
        assert "baz" in method_names

    def test_java_imports(self, tmp_path: Path) -> None:
        """Finds Java import statements."""
        source = "import java.util.List;\nimport com.example.User;\n"
        f = tmp_path / "App.java"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("java")
        symbols, _ = ext.extract_file(f)

        import_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.IMPORT}
        assert "java.util.List" in import_names
        assert "com.example.User" in import_names

    # -- C/C++ --

    def test_c_functions(self, tmp_path: Path) -> None:
        """Finds C function definitions."""
        source = textwrap.dedent("""\
            #include <stdio.h>
            #include "utils.h"

            #define MAX_SIZE 1024

            int main(int argc, char *argv[]) {
                return 0;
            }

            void helper(int x) {
                printf("%d", x);
            }
        """)
        f = tmp_path / "main.c"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("c")
        symbols, _ = ext.extract_file(f)

        names = {s.name for s in symbols}
        assert "main" in names
        assert "helper" in names

    def test_c_includes(self, tmp_path: Path) -> None:
        """Finds #include directives."""
        source = '#include <stdio.h>\n#include "mylib.h"\n'
        f = tmp_path / "test.c"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("c")
        symbols, _ = ext.extract_file(f)

        import_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.IMPORT}
        assert "stdio.h" in import_names
        assert "mylib.h" in import_names

    def test_c_defines(self, tmp_path: Path) -> None:
        """Finds #define constants."""
        source = "#define MAX_SIZE 1024\n#define BUFFER_LEN 256\n"
        f = tmp_path / "defs.h"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("c")
        symbols, _ = ext.extract_file(f)

        const_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CONSTANT}
        assert "MAX_SIZE" in const_names
        assert "BUFFER_LEN" in const_names

    def test_cpp_classes(self, tmp_path: Path) -> None:
        """Finds C++ class/struct."""
        source = textwrap.dedent("""\
            class Widget {
            public:
                void draw();
            };

            struct Point {
                int x, y;
            };
        """)
        f = tmp_path / "widget.cpp"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("cpp")
        symbols, _ = ext.extract_file(f)

        class_names = {s.name for s in symbols if s.symbol_type == CodeSymbolType.CLASS}
        assert "Widget" in class_names
        assert "Point" in class_names

    # -- Generic fallback --

    def test_unsupported_extension_fallback(self, tmp_path: Path) -> None:
        """Unknown language uses generic patterns."""
        source = textwrap.dedent("""\
            function hello() {
                return 1
            }

            class MyClass {
            }
        """)
        f = tmp_path / "script.lua"
        f.write_text(source, encoding="utf-8")

        ext = RegexExtractor("generic")
        symbols, _ = ext.extract_file(f)

        names = {s.name for s in symbols}
        assert "hello" in names
        assert "MyClass" in names

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty lists."""
        f = tmp_path / "empty.js"
        f.write_text("", encoding="utf-8")

        ext = RegexExtractor("javascript")
        symbols, rels = ext.extract_file(f)

        assert symbols == []
        assert rels == []


class TestGetExtractor:
    """Tests for the get_extractor() dispatcher."""

    def test_python_returns_python_extractor(self) -> None:
        """Python extension returns PythonExtractor."""
        ext = get_extractor(".py")
        assert isinstance(ext, PythonExtractor)

    def test_js_returns_regex_extractor(self) -> None:
        """JS extension returns RegexExtractor."""
        ext = get_extractor(".js")
        assert isinstance(ext, RegexExtractor)

    def test_ts_returns_regex_extractor(self) -> None:
        """TS extension returns RegexExtractor."""
        ext = get_extractor(".ts")
        assert isinstance(ext, RegexExtractor)

    def test_go_returns_regex_extractor(self) -> None:
        """Go extension returns RegexExtractor."""
        ext = get_extractor(".go")
        assert isinstance(ext, RegexExtractor)

    def test_rust_returns_regex_extractor(self) -> None:
        """Rust extension returns RegexExtractor."""
        ext = get_extractor(".rs")
        assert isinstance(ext, RegexExtractor)

    def test_java_returns_regex_extractor(self) -> None:
        """Java extension returns RegexExtractor."""
        ext = get_extractor(".java")
        assert isinstance(ext, RegexExtractor)

    def test_c_returns_regex_extractor(self) -> None:
        """C extension returns RegexExtractor."""
        ext = get_extractor(".c")
        assert isinstance(ext, RegexExtractor)

    def test_unknown_returns_generic(self) -> None:
        """Unknown extension returns generic RegexExtractor."""
        ext = get_extractor(".xyz")
        assert isinstance(ext, RegexExtractor)

    def test_python_singleton(self) -> None:
        """Python extractor is reused (singleton)."""
        ext1 = get_extractor(".py")
        ext2 = get_extractor(".py")
        assert ext1 is ext2
