"""Local Archive AI package for offline RAG operations."""

import importlib.util
import sys
from pathlib import Path

# Load the services.py module directly
services_path = Path(__file__).parent / 'services.py'
spec = importlib.util.spec_from_file_location('local_archive_ai.services', services_path)
services_module = importlib.util.module_from_spec(spec)
sys.modules['local_archive_ai.services'] = services_module
spec.loader.exec_module(services_module)

# Import specific classes and functions
EmbeddingService = services_module.EmbeddingService
HybridRetriever = services_module.HybridRetriever
HybridSearchResult = services_module.HybridSearchResult
IngestionReport = services_module.IngestionReport
IngestionStatus = services_module.IngestionStatus
FileIngestionResult = services_module.FileIngestionResult
IndexReport = services_module.IndexReport
answer_query = services_module.answer_query
check_ollama_status = services_module.check_ollama_status
check_ollama_model = services_module.check_ollama_model
collect_files = services_module.collect_files
extract_document_chunks = services_module.extract_document_chunks
extract_document_chunks_resilient = services_module.extract_document_chunks_resilient
extract_image_blocks = services_module.extract_image_blocks
get_autocomplete_suggestions = services_module.get_autocomplete_suggestions
get_index_status = services_module.get_index_status
get_ingestion_report = services_module.get_ingestion_report
get_ollama_status_message = services_module.get_ollama_status_message
index_documents = services_module.index_documents
index_documents_resilient = services_module.index_documents_resilient
index_image_documents = services_module.index_image_documents
load_index_metadata = services_module.load_index_metadata
load_reranker_model = services_module.load_reranker_model
load_text_from_file = services_module.load_text_from_file
prewarm_ollama = services_module.prewarm_ollama
runtime_mode = services_module.runtime_mode
search_image_chunks = services_module.search_image_chunks
search_index = services_module.search_index
summarize_indexable_content = services_module.summarize_indexable_content
system_checks = services_module.system_checks
vector_diagnostics = services_module.vector_diagnostics

from local_archive_ai.store import (
    LocalVectorStore,
    SearchHit,
)

from local_archive_ai.config import (
    AppConfig,
    load_config,
)

from local_archive_ai.logging_config import (
    log,
)

from local_archive_ai.code_indexing import (
    CodeChunk,
    CodeElement,
    CodeRepositoryIndexer,
    PythonCodeParser,
    extract_repository_structure,
)

from local_archive_ai.batch_processor import BatchProcessor
from local_archive_ai.chat_engine import ChatEngine
from local_archive_ai.multi_format_loader import MultiFormatLoader
from local_archive_ai.query_cache import QueryCache
from local_archive_ai.watcher import FolderWatcher

__version__ = "1.0.0"

__all__ = [
    # Services
    "EmbeddingService",
    "HybridRetriever",
    "HybridSearchResult",
    "IngestionReport",
    "IngestionStatus",
    "FileIngestionResult",
    "IndexReport",
    "answer_query",
    "check_ollama_status",
    "check_ollama_model",
    "collect_files",
    "extract_document_chunks",
    "extract_document_chunks_resilient",
    "extract_image_blocks",
    "get_autocomplete_suggestions",
    "get_index_status",
    "get_ingestion_report",
    "get_ollama_status_message",
    "index_documents",
    "index_documents_resilient",
    "index_image_documents",
    "load_index_metadata",
    "load_reranker_model",
    "load_text_from_file",
    "prewarm_ollama",
    "runtime_mode",
    "search_image_chunks",
    "search_index",
    "summarize_indexable_content",
    "system_checks",
    "vector_diagnostics",
    # Store
    "LocalVectorStore",
    "SearchHit",
    # Config
    "AppConfig",
    "load_config",
    # Logging
    "log",
    # Code indexing
    "CodeChunk",
    "CodeElement",
    "CodeRepositoryIndexer",
    "PythonCodeParser",
    "extract_repository_structure",
    # Core components
    "BatchProcessor",
    "ChatEngine",
    "MultiFormatLoader",
    "QueryCache",
    "FolderWatcher",
    # Constants
    "SUPPORTED_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "SKIP_DIRECTORIES",
]

# Re-export constants from services module
from local_archive_ai.services import (
    SUPPORTED_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SKIP_DIRECTORIES,
)
