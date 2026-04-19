"""
Code Repository Indexing Support

Provides intelligent indexing of code repositories with:
- AST parsing for Python files
- Structure-aware chunking (functions, classes, modules)
- Dependency extraction
- Code-specific metadata
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CodeElement:
    """Represents a logical code element (function, class, module)"""
    type: str  # 'function', 'class', 'module', 'async_function'
    name: str
    start_line: int
    end_line: int
    source: str
    docstring: str | None
    decorators: list[str]
    dependencies: list[str]  # Imports this element uses


@dataclass
class CodeChunk:
    """A chunk for code indexing"""
    file_path: Path
    file_name: str
    start_line: int
    end_line: int
    code_type: str  # Element type
    code_name: str  # Element name
    source_text: str
    docstring: str | None
    dependencies: list[str]


class PythonCodeParser:
    """Parse Python files using AST for structure-aware indexing"""

    @staticmethod
    def extract_elements(file_path: Path) -> list[CodeElement]:
        """Extract code elements from Python file using AST"""
        try:
            source = file_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}") from e

        elements: list[CodeElement] = []

        # Add module-level docstring if present
        module_doc = ast.get_docstring(tree) or ""
        if module_doc:
            elements.append(CodeElement(
                type='module',
                name=file_path.stem,
                start_line=1,
                end_line=1,
                source=module_doc,
                docstring=module_doc,
                decorators=[],
                dependencies=PythonCodeParser._extract_imports(tree),
            ))

        # Extract top-level classes and functions directly from tree.body
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                members = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members.append(f"{item.name}()")
                    elif isinstance(item, ast.ClassDef):
                        members.append(f"class {item.name}")

                source_lines = source.splitlines()[node.lineno - 1:node.end_lineno]
                elements.append(CodeElement(
                    type='class',
                    name=f"{file_path.stem}.{node.name}",
                    start_line=node.lineno,
                    end_line=node.end_lineno or len(source.splitlines()),
                    source='\n'.join(source_lines),
                    docstring=ast.get_docstring(node),
                    decorators=[ast.unparse(d) for d in node.decorator_list],
                    dependencies=[],
                ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only add top-level functions (those directly in tree.body)
                func_type = 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function'
                source_lines = source.splitlines()[node.lineno - 1:node.end_lineno]

                elements.append(CodeElement(
                    type=func_type,
                    name=f"{file_path.stem}.{node.name}",
                    start_line=node.lineno,
                    end_line=node.end_lineno or len(source.splitlines()),
                    source='\n'.join(source_lines),
                    docstring=ast.get_docstring(node),
                    decorators=[ast.unparse(d) for d in node.decorator_list],
                    dependencies=[],
                ))

        return elements if elements else [
            CodeElement(
                type='module',
                name=file_path.stem,
                start_line=1,
                end_line=len(source.splitlines()),
                source=source,
                docstring=None,
                decorators=[],
                dependencies=PythonCodeParser._extract_imports(tree),
            )
        ]

    @staticmethod
    def _extract_imports(tree: ast.AST) -> list[str]:
        """Extract all imports from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return list(set(imports))  # Remove duplicates


class CodeRepositoryIndexer:
    """Index code repositories with structure awareness"""

    SUPPORTED_CODE_LANGUAGES = {
        '.py': ('python', PythonCodeParser),
        # More languages can be added here
    }

    @staticmethod
    def is_code_file(file_path: Path) -> bool:
        """Check if file is a supported code file"""
        return file_path.suffix.lower() in CodeRepositoryIndexer.SUPPORTED_CODE_LANGUAGES

    @staticmethod
    def extract_code_chunks(file_path: Path, min_lines: int = 3) -> list[CodeChunk]:
        """Extract code chunks from file with structure awareness"""
        suffix = file_path.suffix.lower()
        if suffix not in CodeRepositoryIndexer.SUPPORTED_CODE_LANGUAGES:
            raise ValueError(f"Unsupported code file type: {suffix}")

        language, parser_class = CodeRepositoryIndexer.SUPPORTED_CODE_LANGUAGES[suffix]

        if language == 'python':
            elements = parser_class.extract_elements(file_path)
            chunks: list[CodeChunk] = []

            for elem in elements:
                line_count = elem.end_line - elem.start_line + 1
                if line_count < min_lines and elem.type != 'module':
                    continue  # Skip very short elements

                chunks.append(CodeChunk(
                    file_path=file_path,
                    file_name=file_path.name,
                    start_line=elem.start_line,
                    end_line=elem.end_line,
                    code_type=elem.type,
                    code_name=elem.name,
                    source_text=elem.source,
                    docstring=elem.docstring,
                    dependencies=elem.dependencies,
                ))

            return chunks

        return []

    @staticmethod
    def code_chunk_to_metadata(chunk: CodeChunk) -> dict[str, Any]:
        """Convert CodeChunk to metadata dict for indexing"""
        combined_text = f"{chunk.code_name}: {chunk.code_type}\n"
        if chunk.docstring:
            combined_text += f"Documentation: {chunk.docstring}\n"
        combined_text += f"Code:\n{chunk.source_text}"

        return {
            'source_file': str(chunk.file_path),
            'file_path': str(chunk.file_path),
            'file_name': chunk.file_name,
            'code_type': chunk.code_type,
            'code_name': chunk.code_name,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'dependencies': chunk.dependencies,
            'source_page': None,
            'chunk_index': 0,  # Will be assigned later
            'text': combined_text,
        }


def extract_repository_structure(repo_path: Path) -> dict[str, Any]:
    """Analyze repository structure"""
    structure = {
        'path': str(repo_path),
        'files_by_type': {},
        'total_files': 0,
        'languages': [],
    }

    for file_path in repo_path.rglob('*'):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        structure['files_by_type'][suffix] = structure['files_by_type'].get(suffix, 0) + 1
        structure['total_files'] += 1

        if CodeRepositoryIndexer.is_code_file(file_path):
            lang_info = CodeRepositoryIndexer.SUPPORTED_CODE_LANGUAGES.get(suffix)
            if lang_info and lang_info[0] not in structure['languages']:
                structure['languages'].append(lang_info[0])

    return structure
