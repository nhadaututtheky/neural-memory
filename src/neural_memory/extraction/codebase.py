"""Code extractors for codebase indexing.

Parses source files and extracts symbols (functions, classes, methods,
imports, constants) and their relationships.

- PythonExtractor: stdlib ast.parse for .py files (full fidelity).
- RegexExtractor: regex-based extraction for JS/TS, Go, Rust, Java, C/C++.
- get_extractor(): dispatcher that selects the right extractor by extension.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class CodeSymbolType(StrEnum):
    """Types of code symbols extracted from source."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    CONSTANT = "constant"


@dataclass(frozen=True)
class CodeSymbol:
    """A symbol extracted from source code."""

    name: str
    symbol_type: CodeSymbolType
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None
    docstring: str | None = None
    parent: str | None = None


@dataclass(frozen=True)
class CodeRelationship:
    """A relationship between code symbols."""

    source: str
    target: str
    relation: str


_SCREAMING_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]+$")


class BaseExtractor(Protocol):
    """Protocol for source code extractors."""

    def extract_file(self, file_path: Path) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Extract symbols and relationships from a source file."""
        ...  # pragma: no cover


def _get_end_lineno(node: ast.AST) -> int:
    """Get the end line number of an AST node."""
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a function signature string from an AST node."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_parts: list[str] = []

    for arg in node.args.args:
        annotation = ""
        if arg.annotation:
            annotation = f": {ast.unparse(arg.annotation)}"
        args_parts.append(f"{arg.arg}{annotation}")

    returns = ""
    if node.returns:
        returns = f" -> {ast.unparse(node.returns)}"

    return f"{prefix} {node.name}({', '.join(args_parts)}){returns}"


class PythonExtractor:
    """Extracts code symbols and relationships from Python files."""

    def extract_file(self, file_path: Path) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Parse a Python file and extract symbols + relationships.

        Args:
            file_path: Path to the Python file to parse.

        Returns:
            Tuple of (symbols, relationships).
        """
        # Limit file size to 1MB to prevent memory issues
        if file_path.stat().st_size > 1_000_000:
            return [], []
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        file_str = str(file_path)
        symbols: list[CodeSymbol] = []
        relationships: list[CodeRelationship] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym, rels = self._extract_function(node, file_str, parent=None)
                symbols.append(sym)
                relationships.extend(rels)

            elif isinstance(node, ast.ClassDef):
                class_syms, class_rels = self._extract_class(node, file_str)
                symbols.extend(class_syms)
                relationships.extend(class_rels)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imp_syms, imp_rels = self._extract_import(node, file_str)
                symbols.extend(imp_syms)
                relationships.extend(imp_rels)

            elif isinstance(node, ast.Assign):
                const_syms, const_rels = self._extract_constant(node, file_str)
                symbols.extend(const_syms)
                relationships.extend(const_rels)

        return symbols, relationships

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        parent: str | None,
    ) -> tuple[CodeSymbol, list[CodeRelationship]]:
        """Extract a function or method symbol."""
        symbol_type = CodeSymbolType.METHOD if parent else CodeSymbolType.FUNCTION
        docstring = ast.get_docstring(node)
        signature = _build_signature(node)

        symbol = CodeSymbol(
            name=node.name,
            symbol_type=symbol_type,
            file_path=file_path,
            line_start=node.lineno,
            line_end=_get_end_lineno(node),
            signature=signature,
            docstring=docstring,
            parent=parent,
        )

        rel_target = f"{parent}.{node.name}" if parent else node.name
        relationships = [
            CodeRelationship(
                source=file_path,
                target=rel_target,
                relation="contains",
            )
        ]

        return symbol, relationships

    def _extract_class(
        self, node: ast.ClassDef, file_path: str
    ) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Extract a class and its methods."""
        symbols: list[CodeSymbol] = []
        relationships: list[CodeRelationship] = []

        docstring = ast.get_docstring(node)

        class_symbol = CodeSymbol(
            name=node.name,
            symbol_type=CodeSymbolType.CLASS,
            file_path=file_path,
            line_start=node.lineno,
            line_end=_get_end_lineno(node),
            docstring=docstring,
        )
        symbols.append(class_symbol)

        relationships.append(
            CodeRelationship(
                source=file_path,
                target=node.name,
                relation="contains",
            )
        )

        for base in node.bases:
            base_name = ast.unparse(base)
            relationships.append(
                CodeRelationship(
                    source=node.name,
                    target=base_name,
                    relation="is_a",
                )
            )

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym, rels = self._extract_function(child, file_path, parent=node.name)
                symbols.append(sym)
                relationships.extend(rels)

        return symbols, relationships

    def _extract_import(
        self, node: ast.Import | ast.ImportFrom, file_path: str
    ) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Extract import symbols."""
        symbols: list[CodeSymbol] = []
        relationships: list[CodeRelationship] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=CodeSymbolType.IMPORT,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=_get_end_lineno(node),
                    )
                )
                relationships.append(
                    CodeRelationship(
                        source=file_path,
                        target=alias.name,
                        relation="imports",
                    )
                )
        else:
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                full_target = f"{module}.{alias.name}" if module else alias.name
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=CodeSymbolType.IMPORT,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=_get_end_lineno(node),
                    )
                )
                relationships.append(
                    CodeRelationship(
                        source=file_path,
                        target=full_target,
                        relation="imports",
                    )
                )

        return symbols, relationships

    def _extract_constant(
        self, node: ast.Assign, file_path: str
    ) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Extract top-level constants (SCREAMING_SNAKE_CASE assignments)."""
        symbols: list[CodeSymbol] = []
        relationships: list[CodeRelationship] = []

        for target in node.targets:
            if isinstance(target, ast.Name) and _SCREAMING_SNAKE.match(target.id):
                symbols.append(
                    CodeSymbol(
                        name=target.id,
                        symbol_type=CodeSymbolType.CONSTANT,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=_get_end_lineno(node),
                    )
                )
                relationships.append(
                    CodeRelationship(
                        source=file_path,
                        target=target.id,
                        relation="contains",
                    )
                )

        return symbols, relationships


# ---------------------------------------------------------------------------
# Regex-based extractor for non-Python languages
# ---------------------------------------------------------------------------

_LANGUAGE_PATTERNS: dict[
    str, dict[str, list[tuple[re.Pattern[str], CodeSymbolType, str | None]]]
] = {}


def _compile_patterns() -> None:
    """Build regex pattern tables per language (called once at import time)."""
    # -- JavaScript / TypeScript --
    js_patterns: list[tuple[re.Pattern[str], CodeSymbolType, str | None]] = [
        (re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)"), CodeSymbolType.FUNCTION, None),
        (
            re.compile(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("),
            CodeSymbolType.FUNCTION,
            None,
        ),
        (
            re.compile(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"),
            CodeSymbolType.CLASS,
            None,
        ),
        (re.compile(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]"), CodeSymbolType.IMPORT, None),
        (re.compile(r"require\(['\"]([^'\"]+)['\"]\)"), CodeSymbolType.IMPORT, None),
        (re.compile(r"(?:export\s+)?const\s+([A-Z][A-Z0-9_]+)\s*="), CodeSymbolType.CONSTANT, None),
    ]
    for lang in ("javascript", "typescript"):
        _LANGUAGE_PATTERNS[lang] = {"patterns": js_patterns}

    # -- Go --
    _LANGUAGE_PATTERNS["go"] = {
        "patterns": [
            (re.compile(r"^func\s+(\w+)\s*\(", re.MULTILINE), CodeSymbolType.FUNCTION, None),
            (
                re.compile(r"^func\s+\(\w+\s+\*?(\w+)\)\s+(\w+)\s*\(", re.MULTILINE),
                CodeSymbolType.METHOD,
                "go_method",
            ),
            (re.compile(r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE), CodeSymbolType.CLASS, None),
            (
                re.compile(r"^type\s+(\w+)\s+interface\s*\{", re.MULTILINE),
                CodeSymbolType.CLASS,
                None,
            ),
            (re.compile(r"import\s+\"([^\"]+)\""), CodeSymbolType.IMPORT, None),
            (
                re.compile(r"^\s+\"([^\"]+)\"", re.MULTILINE),
                CodeSymbolType.IMPORT,
                "go_block_import",
            ),
            (
                re.compile(r"^const\s+([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE),
                CodeSymbolType.CONSTANT,
                None,
            ),
        ],
    }

    # -- Rust --
    _LANGUAGE_PATTERNS["rust"] = {
        "patterns": [
            (re.compile(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"), CodeSymbolType.FUNCTION, None),
            (re.compile(r"(?:pub\s+)?struct\s+(\w+)"), CodeSymbolType.CLASS, None),
            (re.compile(r"(?:pub\s+)?enum\s+(\w+)"), CodeSymbolType.CLASS, None),
            (re.compile(r"(?:pub\s+)?trait\s+(\w+)"), CodeSymbolType.CLASS, None),
            (re.compile(r"impl(?:<[^>]+>)?\s+(\w+)"), CodeSymbolType.CLASS, "impl"),
            (re.compile(r"use\s+([^;]+);"), CodeSymbolType.IMPORT, None),
            (re.compile(r"(?:pub\s+)?const\s+([A-Z][A-Z0-9_]+)"), CodeSymbolType.CONSTANT, None),
        ],
    }

    # -- Java / Kotlin --
    java_patterns: list[tuple[re.Pattern[str], CodeSymbolType, str | None]] = [
        (
            re.compile(
                r"(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"
            ),
            CodeSymbolType.CLASS,
            None,
        ),
        (
            re.compile(r"(?:public|private|protected)?\s*interface\s+(\w+)"),
            CodeSymbolType.CLASS,
            None,
        ),
        (
            re.compile(
                r"(?:public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\("
            ),
            CodeSymbolType.METHOD,
            None,
        ),
        (re.compile(r"import\s+([^;]+);"), CodeSymbolType.IMPORT, None),
        (
            re.compile(r"(?:static\s+)?final\s+\w+\s+([A-Z][A-Z0-9_]+)"),
            CodeSymbolType.CONSTANT,
            None,
        ),
    ]
    for lang in ("java", "kotlin"):
        _LANGUAGE_PATTERNS[lang] = {"patterns": java_patterns}

    # -- C / C++ --
    c_patterns: list[tuple[re.Pattern[str], CodeSymbolType, str | None]] = [
        (
            re.compile(r"^(?:[\w:*&\s]+?)\s+(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
            CodeSymbolType.FUNCTION,
            None,
        ),
        (re.compile(r"(?:class|struct)\s+(\w+)"), CodeSymbolType.CLASS, None),
        (re.compile(r"#include\s*[<\"]([^>\"]+)[>\"]"), CodeSymbolType.IMPORT, None),
        (re.compile(r"#define\s+([A-Z][A-Z0-9_]+)"), CodeSymbolType.CONSTANT, None),
    ]
    for lang in ("c", "cpp"):
        _LANGUAGE_PATTERNS[lang] = {"patterns": c_patterns}

    # -- Generic fallback --
    _LANGUAGE_PATTERNS["generic"] = {
        "patterns": [
            (
                re.compile(r"(?:export\s+)?(?:async\s+)?(?:def|function|fn|func)\s+(\w+)"),
                CodeSymbolType.FUNCTION,
                None,
            ),
            (
                re.compile(r"(?:class|struct|interface|trait|type)\s+(\w+)"),
                CodeSymbolType.CLASS,
                None,
            ),
            (
                re.compile(r"(?:import|use|require|include)\s+(.+?)(?:;|\s*$)", re.MULTILINE),
                CodeSymbolType.IMPORT,
                None,
            ),
        ],
    }


_compile_patterns()


class RegexExtractor:
    """Regex-based extractor for non-Python languages.

    Uses pre-compiled regex patterns per language to extract functions,
    classes, imports, and constants. Lower fidelity than AST but zero
    external dependencies.
    """

    def __init__(self, language: str) -> None:
        self._language = language
        self._patterns = _LANGUAGE_PATTERNS.get(language, _LANGUAGE_PATTERNS["generic"])

    def extract_file(self, file_path: Path) -> tuple[list[CodeSymbol], list[CodeRelationship]]:
        """Extract symbols and relationships from a source file using regex."""
        source = file_path.read_text(encoding="utf-8", errors="replace")
        if len(source) > 100_000:
            source = source[:100_000]
        file_str = str(file_path)

        symbols: list[CodeSymbol] = []
        relationships: list[CodeRelationship] = []
        seen: set[tuple[str, CodeSymbolType]] = set()

        for pattern, symbol_type, hint in self._patterns["patterns"]:
            for match in pattern.finditer(source):
                name = self._extract_name(match, hint)
                if not name:
                    continue

                actual_type = symbol_type
                parent = self._detect_parent(match, hint)
                if parent and symbol_type == CodeSymbolType.FUNCTION:
                    actual_type = CodeSymbolType.METHOD

                key = (name, actual_type)
                if key in seen:
                    continue
                seen.add(key)

                line_start = source[: match.start()].count("\n") + 1
                line_end = source[: match.end()].count("\n") + 1

                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=actual_type,
                        file_path=file_str,
                        line_start=line_start,
                        line_end=line_end,
                        parent=parent,
                    )
                )

                sym_key = f"{parent}.{name}" if parent else name
                relationships.append(
                    CodeRelationship(
                        source=file_str,
                        target=sym_key,
                        relation="contains" if symbol_type != CodeSymbolType.IMPORT else "imports",
                    )
                )

        # Detect class inheritance from matched class patterns
        relationships.extend(self._extract_inheritance(source))

        return symbols, relationships

    def _extract_name(self, match: re.Match[str], hint: str | None) -> str | None:
        """Extract the symbol name from a regex match."""
        if hint == "go_method":
            # Go method: group(1) = receiver type, group(2) = method name
            groups = match.groups()
            return str(groups[1]) if len(groups) >= 2 and groups[1] else None
        if hint == "go_block_import":
            raw = match.group(1).strip()
            return raw if raw and not raw.startswith("//") else None
        return str(match.group(1)) if match.group(1) else None

    def _detect_parent(self, match: re.Match[str], hint: str | None) -> str | None:
        """Detect parent class/struct for methods."""
        if hint == "go_method":
            return str(match.group(1))
        return None

    def _extract_inheritance(self, source: str) -> list[CodeRelationship]:
        """Extract is_a relationships from class definitions."""
        result: list[CodeRelationship] = []
        # JS/TS/Java class ... extends ...
        for m in re.finditer(r"class\s+(\w+)\s+extends\s+(\w+)", source):
            result.append(CodeRelationship(source=m.group(1), target=m.group(2), relation="is_a"))
        # Java implements
        for m in re.finditer(r"class\s+(\w+)\s+.*?implements\s+([\w,\s]+)", source):
            for iface in m.group(2).split(","):
                iface_name = iface.strip()
                if iface_name:
                    result.append(
                        CodeRelationship(source=m.group(1), target=iface_name, relation="is_a")
                    )
        return result


# ---------------------------------------------------------------------------
# Extractor dispatcher
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
}

_python_extractor = PythonExtractor()


def get_extractor(extension: str) -> BaseExtractor:
    """Return the appropriate extractor for a file extension.

    Args:
        extension: File extension including dot (e.g. ".py", ".ts").

    Returns:
        BaseExtractor implementation for the given language.
    """
    lang = _EXTENSION_MAP.get(extension)
    if lang == "python":
        return _python_extractor
    if lang:
        return RegexExtractor(lang)
    return RegexExtractor("generic")
