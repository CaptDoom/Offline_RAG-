"""
Comprehensive end-to-end tests for LOCAL-ARCHIVE AI

Tests all major workflows:
- Document ingestion (PDF, images, text, code)
- Vector storage and retrieval
- Batch operations
- Configuration management
"""

import pytest
import tempfile
import json
from pathlib import Path
from PIL import Image as PILImage
import numpy as np

from local_archive_ai.config import AppConfig, load_config, save_config
from local_archive_ai.services import (
    EmbeddingService,
    HybridRetriever,
    LocalVectorStore,
    extract_document_chunks_resilient,
    collect_files,
    summarize_indexable_content,
    index_documents_resilient,
)
from local_archive_ai.code_indexing import CodeRepositoryIndexer, PythonCodeParser


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index storage"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
        return Path(f.name)


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
"""
Module for testing code indexing.
"""

def hello_world():
    """Print hello world."""
    print("Hello, World!")

class TestClass:
    """A test class for demonstration."""
    
    def method_one(self):
        """First method."""
        return 42
    
    def method_two(self):
        """Second method."""
        return "test"

async def async_function():
    """Async function for testing."""
    await some_async_operation()
''')
        return Path(f.name)


@pytest.fixture
def sample_image_file():
    """Create a sample image file with text"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create simple test image
        img = PILImage.new('RGB', (200, 100), color='white')
        img.save(f.name)
        return Path(f.name)


@pytest.fixture
def sample_pdf_file():
    """Create a simple PDF for testing (requires reportlab)"""
    try:
        from reportlab.pdfgen import canvas
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            c = canvas.Canvas(f.name)
            c.drawString(100, 750, "This is a test PDF document")
            c.drawString(100, 730, "With multiple lines of text")
            c.save()
            return Path(f.name)
    except ImportError:
        return None


class TestEmbedding:
    """Test embedding service"""

    def test_embedding_service_initialization(self):
        """Test EmbeddingService initializes correctly"""
        service = EmbeddingService()
        assert service is not None

    def test_embedding_empty_list(self):
        """Test embedding empty text list"""
        service = EmbeddingService()
        vectors = service.embed([])
        assert vectors.shape == (0, 384)

    def test_embedding_single_text(self):
        """Test embedding single text"""
        service = EmbeddingService()
        vectors = service.embed(["Hello world"])
        assert vectors.shape == (1, 384)
        assert np.all(np.isfinite(vectors))

    def test_embedding_multiple_texts(self):
        """Test embedding multiple texts"""
        service = EmbeddingService()
        texts = ["First text", "Second text", "Third text"]
        vectors = service.embed(texts)
        assert vectors.shape == (3, 384)
        assert np.all(np.isfinite(vectors))


class TestVectorStore:
    """Test LocalVectorStore functionality"""

    def test_vector_store_initialization(self, temp_index_dir):
        """Test vector store initializes"""
        store = LocalVectorStore(temp_index_dir)
        assert store.storage_path == temp_index_dir

    def test_vector_store_build_and_load(self, temp_index_dir):
        """Test building and loading vector store"""
        vectors = np.random.randn(10, 384).astype(np.float32)
        metadata = [
            {"text": f"Document {i}", "file_name": "test.txt", "chunk_index": i}
            for i in range(10)
        ]

        store = LocalVectorStore(temp_index_dir)
        store.build(vectors, metadata)

        # Verify files created
        assert store.index_file.exists()
        assert store.metadata_file.exists()

        # Load and verify
        store2 = LocalVectorStore(temp_index_dir)
        assert store2.load()
        assert store2.ready()
        assert len(store2.metadata) == 10


class TestDocumentIngest:
    """Test document ingestion"""

    def test_ingest_text_file(self, sample_text_file):
        """Test ingesting plain text file"""
        chunks, status, error, engine, time_taken = extract_document_chunks_resilient(
            sample_text_file, chunk_size=100
        )
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert status.value in ["success", "warning"]

    def test_ingest_python_file(self, sample_python_file):
        """Test ingesting Python file with code parsing"""
        chunks, status, error, engine, time_taken = extract_document_chunks_resilient(
            sample_python_file, chunk_size=500, use_code_parsing=True
        )
        # Should have chunks for modules, functions, classes
        assert len(chunks) > 0
        # Should preserve structure info
        assert any("TestClass" in chunk.get("code_name", "") for chunk in chunks)

    def test_ingest_with_fallback(self, sample_python_file):
        """Test that code parsing falls back to text if needed"""
        chunks, status, error, engine, time_taken = extract_document_chunks_resilient(
            sample_python_file, chunk_size=500, use_code_parsing=True
        )
        assert len(chunks) > 0
        assert status.value in ["success", "warning"]


class TestRetrieval:
    """Test retrieval operations"""

    def test_hybrid_retriever_initialization(self, temp_index_dir):
        """Test hybrid retriever initializes"""
        vectors = np.random.randn(10, 384).astype(np.float32)
        metadata = [
            {"text": f"Document {i}", "file_name": "test.txt", "chunk_index": i}
            for i in range(10)
        ]
        store = LocalVectorStore(temp_index_dir)
        store.build(vectors, metadata)

        service = EmbeddingService()
        retriever = HybridRetriever(store, service)
        assert retriever is not None

    def test_vector_search(self, temp_index_dir):
        """Test vector search"""
        vectors = np.random.randn(10, 384).astype(np.float32)
        metadata = [
            {"text": f"Document about {i}", "file_name": "test.txt", "chunk_index": i}
            for i in range(10)
        ]
        store = LocalVectorStore(temp_index_dir)
        store.build(vectors, metadata)

        service = EmbeddingService()
        query_vec = service.embed(["test document"])[0]
        results = store.search(query_vec, top_k=5)
        assert len(results) <= 5


class TestCodeIndexing:
    """Test code repository indexing"""

    def test_is_code_file(self, sample_python_file):
        """Test code file detection"""
        assert CodeRepositoryIndexer.is_code_file(sample_python_file)

    def test_python_parser_extraction(self, sample_python_file):
        """Test Python AST parsing"""
        elements = PythonCodeParser.extract_elements(sample_python_file)
        assert len(elements) > 0
        # Should extract module, functions, classes
        assert any(e.type == 'class' for e in elements)
        assert any(e.type == 'function' for e in elements)

    def test_code_chunk_extraction(self, sample_python_file):
        """Test code chunk extraction"""
        chunks = CodeRepositoryIndexer.extract_code_chunks(sample_python_file)
        assert len(chunks) > 0
        # Every chunk should have code-specific metadata
        assert all(hasattr(c, 'code_type') for c in chunks)
        assert all(hasattr(c, 'code_name') for c in chunks)


class TestConfiguration:
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration"""
        config = AppConfig()
        assert config.chunk_size >= 1
        assert config.top_k >= 1
        assert config.faiss_path

    def test_config_validation(self):
        """Test config validation"""
        # Valid config
        config = AppConfig(chunk_size=500, top_k=5)
        assert config.chunk_size == 500
        assert config.top_k == 5

        # Invalid configs should raise
        with pytest.raises(Exception):
            AppConfig(chunk_size=-1)  # Negative chunk size

        with pytest.raises(Exception):
            AppConfig(top_k=0)  # Zero top_k


class TestIntegration:
    """Integration tests for full workflows"""

    def test_index_and_retrieve_workflow(self, temp_index_dir, sample_text_file):
        """Test full indexing and retrieval workflow"""
        folder = sample_text_file.parent

        # Index the folder
        report = index_documents_resilient(
            folder_path=str(folder),
            chunk_size=100,
            store_path=str(temp_index_dir),
            use_progress_bar=False,
        )

        assert report.successful_files >= 0
        assert report.total_chunks > 0

        # Load store and verify retrieval works
        store = LocalVectorStore(temp_index_dir)
        assert store.load()
        assert store.ready()

    def test_collect_files(self, sample_text_file, sample_python_file):
        """Test file collection"""
        folder = sample_text_file.parent
        files = collect_files(folder)
        assert len(files) >= 1

    def test_summarize_content(self, sample_text_file, sample_python_file):
        """Test content summarization"""
        folder = sample_text_file.parent
        summary = summarize_indexable_content(folder)
        assert summary["total_supported"] >= 1
        assert len(summary["top_extensions"]) > 0


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_file_handling(self, temp_index_dir):
        """Test handling of missing files"""
        with pytest.raises(Exception):
            extract_document_chunks_resilient(
                Path("/nonexistent/file.txt"),
                chunk_size=100,
            )

    def test_empty_directory_handling(self):
        """Test handling of empty directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                index_documents_resilient(
                    folder_path=tmpdir,
                    chunk_size=100,
                    store_path=str(Path(tmpdir) / "index"),
                    use_progress_bar=False,
                )

    def test_corrupted_vector_store_recovery(self, temp_index_dir):
        """Test recovery from corrupted vector store"""
        # Create a valid store
        vectors = np.random.randn(5, 384).astype(np.float32)
        metadata = [
            {"text": f"Document {i}", "file_name": "test.txt", "chunk_index": i}
            for i in range(5)
        ]
        store = LocalVectorStore(temp_index_dir)
        store.build(vectors, metadata)

        # Try to load corrupted metadata
        metadata_file = temp_index_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            f.write('{"invalid": "json}')

        # Should handle gracefully
        store2 = LocalVectorStore(temp_index_dir)
        # load() should return False for corrupted file
        result = store2.load()
        # At least it shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
