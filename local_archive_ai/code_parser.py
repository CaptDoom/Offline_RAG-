"""Code-aware parsing for repository indexing.

Provides language-specific parsers:
- Python: AST-based parsing for functions, classes, imports, docstrings
- JavaScript/TypeScript: Regex-based parsing for functions, classes, imports
- Java: Regex-based parsing for methods, classes, imports
- Rust: Regex-based parsing for functions, structs, impls

All parsers extract structured metadata for improved RAG retrieval.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CodeElement:
    """Represents a logical code element (function, class, module, etc.)."""
    type: str  # 'function', 'class', 'module', 'method', 'struct', 'interface', etc.
    name: str
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str] = None
    decorators: Optional[list[str]] = None
    modifiers: Optional[list[str]] = None  # public, private, static, etc.
    parameters: Optional[list[str]] = None
    return_type: Optional[str] = None
    dependencies: Optional[list[str]] = None


@dataclass
class ParsedCodeFile:
    """Result of parsing a code file."""
    file_path: Path
    language: str
    elements: list[CodeElement]
    imports: list[str]
    module_docstring: Optional[str] = None


class CodeParser(ABC):
    """Abstract base class for code parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedCodeFile:
        """Parse a code file and return structured representation."""
        pass

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        pass


class PythonCodeParser(CodeParser):
    """Python code parser using AST for structure-aware indexing."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".py"

    def parse(self, file_path: Path) -> ParsedCodeFile:
        """Parse Python file using AST."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}") from e

        elements: list[CodeElement] = []
        imports: list[str] = []

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Extract classes and functions with proper parent tracking
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                elements.append(self._parse_class(node, source, file_path.stem))
            elif isinstance(node, ast.FunctionDef):
                # Skip methods (they'll be handled by the class parser)
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    elements.append(self._parse_function(node, source, file_path.stem))
            elif isinstance(node, ast.AsyncFunctionDef):
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    elements.append(self._parse_function(node, source, file_path.stem, is_async=True))

        return ParsedCodeFile(
            file_path=file_path,
            language="python",
            elements=elements,
            imports=imports,
            module_docstring=module_docstring,
        )

    def _parse_class(self, node: ast.ClassDef, source: str, module_name: str) -> CodeElement:
        """Parse a class definition."""
        source_lines = source.splitlines()
        class_source = "\n".join(source_lines[node.lineno - 1 : node.end_lineno])
        docstring = ast.get_docstring(node)

        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Extract base classes
        bases = [ast.unparse(base) for base in node.bases]
        name = f"{module_name}.{node.name}" if bases else node.name

        return CodeElement(
            type="class",
            name=name,
            start_line=node.lineno,
            end_line=node.end_lineno or len(source_lines),
            source=class_source,
            docstring=docstring,
            decorators=decorators,
            modifiers=["extends " + ", ".join(bases)] if bases else None,
            parameters=None,
            return_type=None,
            dependencies=[],
        )

    def _parse_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, source: str, module_name: str, is_async: bool = False
    ) -> CodeElement:
        """Parse a function definition."""
        source_lines = source.splitlines()
        func_source = "\n".join(source_lines[node.lineno - 1 : node.end_lineno])
        docstring = ast.get_docstring(node)

        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Extract parameters
        params = []
        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)

        # Extract return type
        return_type = ast.unparse(node.returns) if node.returns else None

        return CodeElement(
            type="async_function" if is_async else "function",
            name=f"{module_name}.{node.name}",
            start_line=node.lineno,
            end_line=node.end_lineno or len(source_lines),
            source=func_source,
            docstring=docstring,
            decorators=decorators,
            modifiers=["async"] if is_async else None,
            parameters=params,
            return_type=return_type,
            dependencies=[],
        )


class JavaScriptCodeParser(CodeParser):
    """JavaScript/TypeScript code parser using regex fallbacks."""

    # Patterns for JavaScript/TypeScript extraction
    FUNC_PATTERN = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?\s*\{",
        re.MULTILINE,
    )
    ARROW_PATTERN = re.compile(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*(\w+))?\s*=>\s*\{",
        re.MULTILINE,
    )
    CLASS_PATTERN = re.compile(
        r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{", re.MULTILINE
    )
    METHOD_PATTERN = re.compile(
        r"(?:async\s+)?(?:\*|get\s+|set\s+)?(\w+)\s*\(([^)]*)\)\s*\{", re.MULTILINE
    )
    IMPORT_PATTERN = re.compile(
        r"import\s+(?:\{[^}]+\}|\*\s+as\s+\w+|\w+)\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE
    )
    REQUIRE_PATTERN = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.MULTILINE)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs"}

    def parse(self, file_path: Path) -> ParsedCodeFile:
        """Parse JavaScript/TypeScript file using regex patterns."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}") from e

        lines = source.splitlines()
        elements: list[CodeElement] = []
        imports: list[str] = []

        # Extract imports
        for match in self.IMPORT_PATTERN.finditer(source):
            imports.append(match.group(1))
        for match in self.REQUIRE_PATTERN.finditer(source):
            imports.append(match.group(1))

        # Extract functions
        for match in self.FUNC_PATTERN.finditer(source):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            return_type = match.group(3)

            start_line = source[: match.start()].count("\n") + 1
            source_text = self._extract_block(source, match.end() - 1, "{", "}")

            elements.append(
                CodeElement(
                    type="function",
                    name=name,
                    start_line=start_line,
                    end_line=start_line + source_text.count("\n"),
                    source=f"function {name}({match.group(2)}) {source_text}",
                    docstring=None,
                    parameters=params,
                    return_type=return_type,
                )
            )

        # Extract arrow functions (only at module level)
        for match in self.ARROW_PATTERN.finditer(source):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            return_type = match.group(3)

            start_line = source[: match.start()].count("\n") + 1
            source_text = self._extract_block(source, match.end() - 1, "{", "}")

            elements.append(
                CodeElement(
                    type="function",
                    name=name,
                    start_line=start_line,
                    end_line=start_line + source_text.count("\n"),
                    source=f"const {name} = ({match.group(2)}) => {source_text}",
                    docstring=None,
                    parameters=params,
                    return_type=return_type,
                )
            )

        # Extract classes
        for match in self.CLASS_PATTERN.finditer(source):
            name = match.group(1)
            extends = match.group(2)

            start_line = source[: match.start()].count("\n") + 1
            class_body = self._extract_block(source, match.end() - 1, "{", "}")

            modifiers = [f"extends {extends}"] if extends else None

            elements.append(
                CodeElement(
                    type="class",
                    name=name,
                    start_line=start_line,
                    end_line=start_line + class_body.count("\n"),
                    source=f"class {name} {class_body}",
                    docstring=None,
                    modifiers=modifiers,
                )
            )

        return ParsedCodeFile(
            file_path=file_path,
            language="javascript",
            elements=elements,
            imports=imports,
        )

    def _extract_block(self, source: str, start_pos: int, open_char: str, close_char: str) -> str:
        """Extract balanced block of code."""
        depth = 0
        start = None
        for i in range(start_pos, len(source)):
            if source[i] == open_char:
                depth += 1
                if start is None:
                    start = i
            elif source[i] == close_char:
                depth -= 1
                if depth == 0:
                    return source[start : i + 1]
        return source[start:] if start else ""


class JavaCodeParser(CodeParser):
    """Java code parser using regex fallbacks."""

    # Patterns for Java extraction
    CLASS_PATTERN = re.compile(
        r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{",
        re.MULTILINE,
    )
    METHOD_PATTERN = re.compile(
        r"(?:public|private|protected|static|final|abstract|synchronized|native)\s+(?:[<\w\s,?\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{",
        re.MULTILINE,
    )
    INTERFACE_PATTERN = re.compile(
        r"(?:public|private)?\s*interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?\s*\{", re.MULTILINE
    )
    IMPORT_PATTERN = re.compile(r"import\s+([\w.]+);", re.MULTILINE)
    PACKAGE_PATTERN = re.compile(r"package\s+([\w.]+);", re.MULTILINE)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".java"

    def parse(self, file_path: Path) -> ParsedCodeFile:
        """Parse Java file using regex patterns."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}") from e

        elements: list[CodeElement] = []
        imports: list[str] = []

        # Extract package and imports
        for match in self.PACKAGE_PATTERN.finditer(source):
            imports.append(f"package:{match.group(1)}")
        for match in self.IMPORT_PATTERN.finditer(source):
            imports.append(match.group(1))

        # Extract classes
        for match in self.CLASS_PATTERN.finditer(source):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)

            start_line = source[: match.start()].count("\n") + 1
            modifiers = []
            if extends:
                modifiers.append(f"extends {extends}")
            if implements:
                modifiers.append(f"implements {implements}")

            elements.append(
                CodeElement(
                    type="class",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,  # Approximate
                    source=f"class {name}",
                    modifiers=modifiers or None,
                )
            )

        # Extract interfaces
        for match in self.INTERFACE_PATTERN.finditer(source):
            name = match.group(1)
            extends = match.group(2)

            start_line = source[: match.start()].count("\n") + 1
            modifiers = [f"extends {extends}"] if extends else None

            elements.append(
                CodeElement(
                    type="interface",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    modifiers=modifiers,
                    source=f"interface {name}",
                )
            )

        # Extract methods
        for match in self.METHOD_PATTERN.finditer(source):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]

            start_line = source[: match.start()].count("\n") + 1

            elements.append(
                CodeElement(
                    type="method",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    parameters=params,
                    source=f"{name}({match.group(2)})",
                )
            )

        return ParsedCodeFile(
            file_path=file_path,
            language="java",
            elements=elements,
            imports=imports,
        )


class RustCodeParser(CodeParser):
    """Rust code parser using regex fallbacks."""

    # Patterns for Rust extraction
    FUNC_PATTERN = re.compile(
        r"(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*(?:\u003c[^\u003e]*\u003e)?\s*\(([^)]*)\)(?:\s*-\u003e\s*(\S+))?\s*\{",
        re.MULTILINE,
    )
    STRUCT_PATTERN = re.compile(
        r"(?:pub\s+)?struct\s+(\w+)(?:\s*\{[^}]*\}|\s*\([^)]*\))?", re.MULTILINE
    )
    ENUM_PATTERN = re.compile(r"(?:pub\s+)?enum\s+(\w+)\s*\{", re.MULTILINE)
    IMPL_PATTERN = re.compile(r"impl(?:\s*\u003c[^\u003e]*\u003e)?\s+(\w+)(?:\s+for\s+(\w+))?\s*\{", re.MULTILINE)
    TRAIT_PATTERN = re.compile(r"(?:pub\s+)?trait\s+(\w+)(?:\s*:\s*([\w+\s]+))?\s*\{", re.MULTILINE)
    USE_PATTERN = re.compile(r"use\s+([\w::{}\s,*]+);", re.MULTILINE)
    MOD_PATTERN = re.compile(r"mod\s+(\w+);", re.MULTILINE)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".rs"

    def parse(self, file_path: Path) -> ParsedCodeFile:
        """Parse Rust file using regex patterns."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}") from e

        elements: list[CodeElement] = []
        imports: list[str] = []

        # Extract imports
        for match in self.USE_PATTERN.finditer(source):
            imports.append(match.group(1))
        for match in self.MOD_PATTERN.finditer(source):
            imports.append(f"mod:{match.group(1)}")

        # Extract functions
        for match in self.FUNC_PATTERN.finditer(source):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(",") if p.strip()]
            return_type = match.group(3)

            start_line = source[: match.start()].count("\n") + 1

            elements.append(
                CodeElement(
                    type="function",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    parameters=params,
                    return_type=return_type,
                    source=f"fn {name}({match.group(2)})",
                )
            )

        # Extract structs
        for match in self.STRUCT_PATTERN.finditer(source):
            name = match.group(1)
            start_line = source[: match.start()].count("\n") + 1

            elements.append(
                CodeElement(
                    type="struct",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    source=f"struct {name}",
                )
            )

        # Extract enums
        for match in self.ENUM_PATTERN.finditer(source):
            name = match.group(1)
            start_line = source[: match.start()].count("\n") + 1

            elements.append(
                CodeElement(
                    type="enum",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    source=f"enum {name}",
                )
            )

        # Extract traits
        for match in self.TRAIT_PATTERN.finditer(source):
            name = match.group(1)
            bounds = match.group(2)
            start_line = source[: match.start()].count("\n") + 1

            modifiers = [f": {bounds}"] if bounds else None

            elements.append(
                CodeElement(
                    type="trait",
                    name=name,
                    start_line=start_line,
                    end_line=start_line,
                    modifiers=modifiers,
                    source=f"trait {name}",
                )
            )

        return ParsedCodeFile(
            file_path=file_path,
            language="rust",
            elements=elements,
            imports=imports,
        )


class CodeParsingService:
    """Service for parsing code files with appropriate parser."""

    _parsers: list[CodeParser] = [
        PythonCodeParser(),
        JavaScriptCodeParser(),
        JavaCodeParser(),
        RustCodeParser(),
    ]

    @classmethod
    def parse_file(cls, file_path: Path) -> Optional[ParsedCodeFile]:
        """Parse a code file using the appropriate parser.

        Args:
            file_path: Path to code file

        Returns:
            ParsedCodeFile or None if no parser available
        """
        for parser in cls._parsers:
            if parser.can_parse(file_path):
                try:
                    return parser.parse(file_path)
                except Exception:
                    return None
        return None

    @classmethod
    def get_supported_extensions(cls) -> set[str]:
        """Get set of supported file extensions."""
        return {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".rs"}

    @classmethod
    def element_to_metadata(cls, element: CodeElement, file_path: Path) -> dict[str, Any]:
        """Convert CodeElement to metadata dict for indexing.

        Args:
            element: Code element to convert
            file_path: Source file path

        Returns:
            Metadata dictionary for vector store
        """
        combined_text = f"{element.name}: {element.type}\n"
        if element.docstring:
            combined_text += f"Documentation: {element.docstring}\n"
        if element.parameters:
            combined_text += f"Parameters: {', '.join(element.parameters)}\n"
        if element.return_type:
            combined_text += f"Returns: {element.return_type}\n"
        combined_text += f"Source:\n{element.source}"

        metadata: dict[str, Any] = {
            "source_file": str(file_path),
            "file_path": str(file_path),
            "file_name": file_path.name,
            "code_type": element.type,
            "code_name": element.name,
            "start_line": element.start_line,
            "end_line": element.end_line,
            "source_page": None,
            "chunk_index": 0,  # Will be assigned later
            "text": combined_text,
        }

        if element.dependencies:
            metadata["dependencies"] = element.dependencies
        if element.modifiers:
            metadata["modifiers"] = element.modifiers
        if element.parameters:
            metadata["parameters"] = element.parameters
        if element.return_type:
            metadata["return_type"] = element.return_type

        return metadata
