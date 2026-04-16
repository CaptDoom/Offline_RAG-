"""Tests for the three new modules: chat_engine, batch_processor, multi_format_loader.

Verifies:
1. ChatEngine: initialization, memory management, query building, fallback
2. BatchProcessor: initialization, rate limiter, progress tracking, CSV processing
3. MultiFormatLoader: initialization, file type detection, chunk generation, repo info
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_archive_ai.chat_engine import (
    ChatEngine,
    ChatResponse,
    ConversationTurn,
    _FALLBACK_MESSAGE,
    _retry_with_backoff,
)
from local_archive_ai.batch_processor import (
    BatchProcessor,
    BatchProgress,
    FileResult,
    QueryResult,
    RateLimiter,
)
from local_archive_ai.multi_format_loader import (
    LoadResult,
    MultiFormatLoader,
    RepoInfo,
    _REPO_IGNORE_DIRS,
    _IMAGE_EXTENSIONS,
)


class TestRetryWithBackoff(unittest.TestCase):
    def test_succeeds_first_try(self) -> None:
        result = _retry_with_backoff(lambda: 42, max_retries=3, operation_name="test")
        self.assertEqual(result, 42)

    def test_succeeds_after_retries(self) -> None:
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("fail")
            return "ok"

        result = _retry_with_backoff(flaky, max_retries=3, operation_name="flaky_test")
        self.assertEqual(result, "ok")
        self.assertEqual(call_count["n"], 3)

    def test_raises_after_all_retries(self) -> None:
        def always_fail():
            raise ValueError("always fails")

        with self.assertRaises(ValueError):
            _retry_with_backoff(always_fail, max_retries=2, operation_name="fail_test")


class TestChatEngine(unittest.TestCase):
    def test_init_defaults(self) -> None:
        engine = ChatEngine(store_path="data/faiss_index")
        self.assertEqual(engine.store_path, "data/faiss_index")
        self.assertEqual(engine.model_name, "llama3.2:1b")
        self.assertEqual(engine.top_k, 4)
        self.assertEqual(len(engine.get_memory()), 0)

    def test_memory_management(self) -> None:
        engine = ChatEngine()
        # Add turns
        for i in range(7):
            engine.add_to_memory(f"q{i}", f"a{i}")

        # Should keep only last 5
        memory = engine.get_memory()
        self.assertEqual(len(memory), 5)
        self.assertEqual(memory[0].question, "q2")
        self.assertEqual(memory[-1].question, "q6")

    def test_clear_memory(self) -> None:
        engine = ChatEngine()
        engine.add_to_memory("q1", "a1")
        engine.add_to_memory("q2", "a2")
        self.assertEqual(len(engine.get_memory()), 2)

        engine.clear_memory()
        self.assertEqual(len(engine.get_memory()), 0)

    def test_last_turn(self) -> None:
        engine = ChatEngine()
        self.assertIsNone(engine.last_turn())

        engine.add_to_memory("q1", "a1")
        last = engine.last_turn()
        self.assertIsNotNone(last)
        self.assertEqual(last.question, "q1")
        self.assertEqual(last.answer, "a1")

    def test_build_memory_context(self) -> None:
        engine = ChatEngine()
        # Empty memory → empty context
        ctx = engine._build_memory_context()
        self.assertEqual(ctx, "")

        # With memory
        engine.add_to_memory("What is AI?", "AI is artificial intelligence.")
        ctx = engine._build_memory_context()
        self.assertIn("What is AI?", ctx)
        self.assertIn("AI is artificial intelligence.", ctx)
        self.assertIn("Previous conversation:", ctx)

    def test_query_empty_question(self) -> None:
        engine = ChatEngine()
        response = engine.query("")
        self.assertEqual(response.answer, "Please enter a question.")

    @patch.object(ChatEngine, '_load_store', return_value=False)
    def test_query_no_index(self, mock_load) -> None:
        engine = ChatEngine()
        response = engine.query("What is RAG?")
        self.assertIn("FAISS index is not loaded", response.answer)

    @patch.object(ChatEngine, '_load_store', return_value=True)
    @patch.object(ChatEngine, '_retrieve', return_value=[])
    def test_query_no_hits_returns_fallback(self, mock_retrieve, mock_load) -> None:
        engine = ChatEngine()
        response = engine.query("What is something completely unknown?")
        self.assertEqual(response.answer, _FALLBACK_MESSAGE)
        self.assertTrue(response.low_confidence)
        # Should also be in memory
        self.assertEqual(len(engine.get_memory()), 1)
        self.assertEqual(engine.get_memory()[0].answer, _FALLBACK_MESSAGE)

    def test_build_citations_empty(self) -> None:
        citations = ChatEngine._build_citations([])
        self.assertEqual(citations, [])


class TestRateLimiter(unittest.TestCase):
    def test_rate_limiting(self) -> None:
        import time
        limiter = RateLimiter(max_qps=100)  # High rate for fast test
        t0 = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        elapsed = time.monotonic() - t0
        # 5 calls at 100 qps should take about 0.04s minimum
        self.assertGreaterEqual(elapsed, 0.03)


class TestBatchProgress(unittest.TestCase):
    def test_progress_tracking(self) -> None:
        progress = BatchProgress(total=10)
        self.assertEqual(progress.processed, 0)
        self.assertAlmostEqual(progress.fraction, 0.0)

        progress.advance(success=True, current_file="test.pdf")
        self.assertEqual(progress.processed, 1)
        self.assertEqual(progress.succeeded, 1)
        self.assertEqual(progress.current_file, "test.pdf")

        progress.advance(success=False, current_file="bad.pdf")
        self.assertEqual(progress.processed, 2)
        self.assertEqual(progress.failed, 1)

        progress.advance(skipped=True, current_file="skip.pdf")
        self.assertEqual(progress.processed, 3)
        self.assertEqual(progress.skipped, 1)

    def test_bar_text(self) -> None:
        progress = BatchProgress(total=50, processed=24)
        bar = progress.bar_text()
        self.assertIn("24/50", bar)
        self.assertIn("files processed", bar)


class TestBatchProcessor(unittest.TestCase):
    def test_init_defaults(self) -> None:
        proc = BatchProcessor()
        self.assertEqual(proc.max_threads, 4)
        self.assertGreater(proc.max_qps, 0)
        self.assertEqual(proc.store_path, "data/faiss_index")

    def test_max_threads_clamped(self) -> None:
        proc = BatchProcessor(max_threads=100)
        self.assertLessEqual(proc.max_threads, 8)

        proc = BatchProcessor(max_threads=0)
        self.assertGreaterEqual(proc.max_threads, 1)

    def test_process_documents_missing_folder(self) -> None:
        proc = BatchProcessor()
        with self.assertRaises(FileNotFoundError):
            proc.process_documents("/nonexistent/path/to/nowhere")

    def test_results_to_csv(self) -> None:
        results = [
            QueryResult(query_id="B001", query="test?", status="COMPLETED", answer="yes", duration_ms=100),
            QueryResult(query_id="B002", query="fail?", status="FAILED", error_message="err", duration_ms=50),
        ]
        csv_str = BatchProcessor.results_to_csv(results)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "B001")
        self.assertEqual(rows[0]["status"], "COMPLETED")
        self.assertEqual(rows[1]["status"], "FAILED")

    def test_process_queries_csv_missing_column(self) -> None:
        proc = BatchProcessor()
        with self.assertRaises(ValueError):
            proc.process_queries_csv("name,value\nfoo,bar")

    def test_process_queries_csv_empty(self) -> None:
        proc = BatchProcessor()
        with self.assertRaises(ValueError):
            proc.process_queries_csv("query\n\n\n")


class TestMultiFormatLoader(unittest.TestCase):
    def test_init_defaults(self) -> None:
        loader = MultiFormatLoader()
        self.assertEqual(loader.chunk_size, 500)
        self.assertEqual(loader.max_retries, 3)

    def test_load_image_nonexistent(self) -> None:
        loader = MultiFormatLoader()
        result = loader.load_image(Path("/nonexistent/image.png"))
        self.assertEqual(result.status, "error")
        self.assertIn("not found", result.error_message.lower())

    def test_load_image_unsupported_format(self) -> None:
        loader = MultiFormatLoader()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()
            result = loader.load_image(Path(f.name))
        self.assertEqual(result.status, "error")
        self.assertIn("Unsupported", result.error_message)

    def test_load_pdf_nonexistent(self) -> None:
        loader = MultiFormatLoader()
        result = loader.load_pdf(Path("/nonexistent/doc.pdf"))
        self.assertEqual(result.status, "error")
        self.assertIn("not found", result.error_message.lower())

    def test_text_to_chunks(self) -> None:
        loader = MultiFormatLoader(chunk_size=500)
        text = "This is a test document. " * 100
        chunks = loader._text_to_chunks(text, "/test/file.txt", "file.txt")
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("file_path", chunk)
            self.assertEqual(chunk["file_name"], "file.txt")

    def test_text_to_chunks_empty(self) -> None:
        loader = MultiFormatLoader()
        chunks = loader._text_to_chunks("", "/test/file.txt", "file.txt")
        self.assertEqual(len(chunks), 0)

    def test_parse_jupyter_notebook(self) -> None:
        loader = MultiFormatLoader()
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Some text"]},
                {"cell_type": "code", "source": ["print('hello')\n", "x = 1"]},
                {"cell_type": "raw", "source": ["raw content"]},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".ipynb", mode="w", delete=False) as f:
            json.dump(notebook, f)
            f.flush()
            text = loader._parse_jupyter_notebook(Path(f.name))

        self.assertIn("[MARKDOWN]", text)
        self.assertIn("# Title", text)
        self.assertIn("[CODE]", text)
        self.assertIn("print('hello')", text)
        # Raw cells should be skipped
        self.assertNotIn("raw content", text)

    def test_repo_ignore_dirs(self) -> None:
        self.assertIn(".git", _REPO_IGNORE_DIRS)
        self.assertIn("__pycache__", _REPO_IGNORE_DIRS)
        self.assertIn("node_modules", _REPO_IGNORE_DIRS)
        self.assertIn(".env", _REPO_IGNORE_DIRS)

    def test_image_extensions(self) -> None:
        self.assertIn(".jpg", _IMAGE_EXTENSIONS)
        self.assertIn(".jpeg", _IMAGE_EXTENSIONS)
        self.assertIn(".png", _IMAGE_EXTENSIONS)
        self.assertIn(".tiff", _IMAGE_EXTENSIONS)

    def test_load_git_repo_invalid_path(self) -> None:
        loader = MultiFormatLoader()
        results = loader.load_git_repo("/nonexistent/repo/path")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].status, "error")

    def test_load_git_repo_local_directory(self) -> None:
        """Test loading a local directory as a 'repo'."""
        loader = MultiFormatLoader(chunk_size=500)
        with tempfile.TemporaryDirectory() as tmp:
            # Create a minimal repo-like structure
            (Path(tmp) / "README.md").write_text("# Test Repo\nA test repository for unit tests.")
            (Path(tmp) / "main.py").write_text("print('hello world')")
            (Path(tmp) / ".git").mkdir()  # Fake .git directory

            results = loader.load_git_repo(tmp)
            self.assertGreater(len(results), 0)
            # Should have summary + at least README and main.py
            summaries = [r for r in results if r.file_type == "repo_summary"]
            self.assertEqual(len(summaries), 1)
            self.assertIn("test repository for unit tests", summaries[0].text)

    def test_get_repo_info(self) -> None:
        loader = MultiFormatLoader()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text("# My Project\nThis is a test project.")
            (Path(tmp) / "main.py").write_text("print('hello')")
            (Path(tmp) / "utils.py").write_text("# utils")

            info = loader._get_repo_info(Path(tmp))
            self.assertEqual(info.name, Path(tmp).name)
            self.assertEqual(info.main_language, "Python")
            self.assertGreater(info.file_count, 0)

    def test_load_file_text(self) -> None:
        loader = MultiFormatLoader()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello world, this is a test document with some content.")
            f.flush()
            result = loader.load_file(Path(f.name))

        self.assertEqual(result.status, "success")
        self.assertIn("Hello world", result.text)
        self.assertEqual(result.file_type, "text")


class TestConversationTurn(unittest.TestCase):
    def test_creation(self) -> None:
        turn = ConversationTurn(question="q", answer="a")
        self.assertEqual(turn.question, "q")
        self.assertEqual(turn.answer, "a")
        self.assertEqual(turn.citations, [])
        self.assertFalse(turn.low_confidence)


class TestChatResponse(unittest.TestCase):
    def test_creation(self) -> None:
        resp = ChatResponse(answer="test answer")
        self.assertEqual(resp.answer, "test answer")
        self.assertEqual(resp.citations, [])
        self.assertAlmostEqual(resp.max_similarity, 0.0)
        self.assertFalse(resp.low_confidence)
        self.assertEqual(resp.duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
