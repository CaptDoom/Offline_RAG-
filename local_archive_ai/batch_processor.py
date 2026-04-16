"""Batch processing with multi-threading, progress tracking, rate limiting, and CSV I/O.

Implements:
- Multi-threading for parallel document ingestion (max 4 threads)
- Progress bar showing: [████░░] 24/50 files processed
- Batch query mode: Upload CSV with questions → output answers CSV
- Rate limiting to prevent Ollama overload (5 queries per second max)
- Timeout (120s per operation), retry logic (3 attempts with exponential backoff)
- Validation: Skip corrupted files with warning (don't crash)
- Logging to local_archive.log with timestamps
"""

from __future__ import annotations

import csv
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_archive_ai.logging_config import log
from local_archive_ai.services import (
    EmbeddingService,
    IngestionStatus,
    collect_files,
    extract_document_chunks_resilient,
)
from local_archive_ai.store import LocalVectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MAX_THREADS = 4
_DEFAULT_MAX_QPS = 5  # queries per second
_DEFAULT_TIMEOUT = 120  # seconds per operation
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Token-bucket rate limiter capped at *max_qps* calls per second."""

    def __init__(self, max_qps: float = _DEFAULT_MAX_QPS) -> None:
        self._interval = 1.0 / max(max_qps, 0.1)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        """Block until a call is allowed."""
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
@dataclass
class BatchProgress:
    """Mutable progress tracker shared across threads."""
    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    current_file: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def advance(self, *, success: bool = True, skipped: bool = False, current_file: str = "") -> None:
        with self._lock:
            self.processed += 1
            if skipped:
                self.skipped += 1
            elif success:
                self.succeeded += 1
            else:
                self.failed += 1
            if current_file:
                self.current_file = current_file

    @property
    def fraction(self) -> float:
        return self.processed / max(self.total, 1)

    def bar_text(self) -> str:
        """Return a text-based progress bar like [████░░] 24/50 files processed."""
        filled = int(20 * self.fraction)
        bar = "█" * filled + "░" * (20 - filled)
        return f"[{bar}] {self.processed}/{self.total} files processed"


# ---------------------------------------------------------------------------
# File ingestion result (per-file)
# ---------------------------------------------------------------------------
@dataclass
class FileResult:
    file_path: str
    file_name: str
    status: str  # 'success', 'warning', 'error', 'skipped'
    chunks_count: int = 0
    error_message: str = ""
    processing_time: float = 0.0


# ---------------------------------------------------------------------------
# Batch query result (per-query)
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    query_id: str
    query: str
    status: str  # 'COMPLETED', 'FAILED'
    answer: str = ""
    error_message: str = ""
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------
class BatchProcessor:
    """Batch document ingestion and query processing.

    Usage::

        processor = BatchProcessor(
            store_path="data/faiss_index",
            model_name="llama3.2:1b",
        )

        # Batch index 50 PDFs
        results = processor.process_documents(
            folder_path="/path/to/docs",
            chunk_size=500,
            on_progress=lambda p: print(p.bar_text()),
        )

        # Batch query from CSV
        output_csv = processor.process_queries_csv(input_csv_bytes)
    """

    def __init__(
        self,
        store_path: str = "data/faiss_index",
        model_name: str = "llama3.2:1b",
        ollama_endpoint: str = "http://127.0.0.1:11434",
        ollama_api_key: str = "",
        max_threads: int = _DEFAULT_MAX_THREADS,
        max_qps: float = _DEFAULT_MAX_QPS,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        top_k: int = 4,
        retrieval_mode: str = "vector",
        bm25_weight: float = 0.4,
        rerank_top_n: int = 5,
        confidence_threshold: float = 0.65,
        temperature: float = 0.7,
        max_context_tokens: int = 8192,
    ) -> None:
        self.store_path = store_path
        self.model_name = model_name
        self.ollama_endpoint = ollama_endpoint
        self.ollama_api_key = ollama_api_key
        self.max_threads = max(1, min(max_threads, 8))
        self.max_qps = max_qps
        self.timeout = timeout
        self.max_retries = max_retries
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.bm25_weight = bm25_weight
        self.rerank_top_n = rerank_top_n
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens

        self._rate_limiter = RateLimiter(max_qps)
        log.info(
            "BatchProcessor initialised (threads=%d, qps=%.1f, store=%s)",
            self.max_threads, self.max_qps, self.store_path,
        )

    # ------------------------------------------------------------------
    # Document ingestion (multi-threaded)
    # ------------------------------------------------------------------
    def _process_single_file(
        self,
        file_path: Path,
        chunk_size: int,
        existing_hashes: dict[str, str],
    ) -> tuple[FileResult, list[dict[str, Any]]]:
        """Process a single file with validation and error handling."""
        t0 = time.time()
        file_name = file_path.name

        # Validate file exists and is readable
        if not file_path.exists():
            return FileResult(
                file_path=str(file_path), file_name=file_name,
                status="error", error_message="File not found",
                processing_time=time.time() - t0,
            ), []

        if not file_path.is_file():
            return FileResult(
                file_path=str(file_path), file_name=file_name,
                status="error", error_message="Not a regular file",
                processing_time=time.time() - t0,
            ), []

        # Check file hash for skip logic
        try:
            file_hash = LocalVectorStore._hash_file(file_path)
        except Exception:
            file_hash = ""

        if file_hash and existing_hashes.get(str(file_path)) == file_hash:
            return FileResult(
                file_path=str(file_path), file_name=file_name,
                status="skipped", processing_time=time.time() - t0,
            ), []

        # Extract chunks with resilient processing
        try:
            chunks, status, error_msg, ocr_engine, proc_time = extract_document_chunks_resilient(
                file_path, chunk_size,
            )
        except Exception as exc:
            log.warning("File processing failed for %s: %s", file_name, exc)
            return FileResult(
                file_path=str(file_path), file_name=file_name,
                status="error", error_message=str(exc),
                processing_time=time.time() - t0,
            ), []

        result_status = "success" if status == IngestionStatus.SUCCESS else (
            "warning" if status == IngestionStatus.WARNING else "error"
        )

        return FileResult(
            file_path=str(file_path), file_name=file_name,
            status=result_status, chunks_count=len(chunks),
            error_message=error_msg, processing_time=time.time() - t0,
        ), chunks

    def process_documents(
        self,
        folder_path: str,
        chunk_size: int = 500,
        on_progress: Callable[[BatchProgress], None] | None = None,
    ) -> tuple[list[FileResult], int]:
        """Index documents from a folder using multi-threaded parallel ingestion.

        Args:
            folder_path: Path to folder containing documents.
            chunk_size: Chunk size in tokens.
            on_progress: Optional callback receiving BatchProgress updates.

        Returns:
            Tuple of (file_results, total_chunks_indexed).
        """
        t0 = time.time()
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = collect_files(folder)
        if not files:
            raise ValueError("No supported documents found in selected folder.")

        # Load existing store for hash comparison
        store = LocalVectorStore(Path(self.store_path))
        existing = store.load()
        existing_hashes = store.file_hashes if existing else {}

        progress = BatchProgress(total=len(files))
        file_results: list[FileResult] = []
        all_chunks: list[dict[str, Any]] = []
        file_hashes: dict[str, str] = {}

        log.info("Batch processing %d files with %d threads", len(files), self.max_threads)

        # Multi-threaded file processing
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_path = {
                executor.submit(self._process_single_file, fp, chunk_size, existing_hashes): fp
                for fp in files
            }

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result, chunks = future.result(timeout=self.timeout)
                except Exception as exc:
                    result = FileResult(
                        file_path=str(file_path), file_name=file_path.name,
                        status="error", error_message=f"Thread error: {exc}",
                    )
                    chunks = []

                file_results.append(result)
                if chunks:
                    all_chunks.extend(chunks)
                    try:
                        fh = LocalVectorStore._hash_file(file_path)
                        file_hashes[str(file_path)] = fh
                    except Exception:
                        pass

                progress.advance(
                    success=result.status in ("success", "warning"),
                    skipped=result.status == "skipped",
                    current_file=result.file_name,
                )

                if on_progress is not None:
                    try:
                        on_progress(progress)
                    except Exception:
                        pass

        # Build vector index from all collected chunks
        total_chunks = 0
        if all_chunks:
            try:
                texts = [item["text"] for item in all_chunks]
                embedder = EmbeddingService()
                vectors = embedder.embed(texts, batch_size=12)

                # Merge with existing if applicable
                if existing and store.metadata:
                    # Keep existing chunks from files that weren't re-processed
                    reprocessed_paths = {r.file_path for r in file_results if r.status != "skipped"}
                    preserved = [
                        chunk for chunk in store.metadata
                        if chunk.get("file_path", "") not in reprocessed_paths
                    ]
                    if preserved:
                        preserved_texts = [item["text"] for item in preserved]
                        preserved_vectors = embedder.embed(preserved_texts, batch_size=12)
                        import numpy as np
                        all_chunks = preserved + all_chunks
                        vectors = np.concatenate([preserved_vectors, vectors], axis=0)
                        file_hashes.update(store.file_hashes)

                new_store = LocalVectorStore(Path(self.store_path))
                new_store.build(vectors, all_chunks, file_hashes=file_hashes)
                total_chunks = len(all_chunks)
            except Exception as exc:
                log.error("Failed to build vector index: %s", exc)
                raise

        elapsed = time.time() - t0
        log.info(
            "Batch processing complete: %d files, %d chunks, %.1fs (success=%d, failed=%d, skipped=%d)",
            len(files), total_chunks, elapsed,
            progress.succeeded, progress.failed, progress.skipped,
        )
        return file_results, total_chunks

    # ------------------------------------------------------------------
    # Batch query processing
    # ------------------------------------------------------------------
    def _execute_single_query(self, query_id: str, query: str) -> QueryResult:
        """Execute a single query with rate limiting and retry logic."""
        self._rate_limiter.acquire()

        t0 = time.time()
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Import here to avoid circular imports at module level
                from local_archive_ai.chat_engine import ChatEngine

                engine = ChatEngine(
                    store_path=self.store_path,
                    model_name=self.model_name,
                    ollama_endpoint=self.ollama_endpoint,
                    ollama_api_key=self.ollama_api_key,
                    top_k=self.top_k,
                    retrieval_mode=self.retrieval_mode,
                    bm25_weight=self.bm25_weight,
                    rerank_top_n=self.rerank_top_n,
                    confidence_threshold=self.confidence_threshold,
                    temperature=self.temperature,
                    max_context_tokens=self.max_context_tokens,
                    timeout=self.timeout,
                    max_retries=1,  # Don't double-retry inside the engine
                )
                response = engine.query(query)
                duration_ms = int((time.time() - t0) * 1000)

                return QueryResult(
                    query_id=query_id,
                    query=query,
                    status="COMPLETED",
                    answer=response.answer,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    log.warning(
                        "Batch query %s failed (attempt %d/%d): %s – retrying in %ds",
                        query_id, attempt, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)

        duration_ms = int((time.time() - t0) * 1000)
        return QueryResult(
            query_id=query_id,
            query=query,
            status="FAILED",
            error_message=str(last_exc),
            duration_ms=duration_ms,
        )

    def process_queries(
        self,
        queries: list[str],
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[QueryResult]:
        """Process a list of queries sequentially with rate limiting.

        Args:
            queries: List of query strings.
            on_progress: Optional callback(processed, total, status_text).

        Returns:
            List of QueryResult objects.
        """
        results: list[QueryResult] = []
        total = len(queries)

        log.info("Batch query processing: %d queries (max %.1f qps)", total, self.max_qps)

        for idx, query in enumerate(queries, start=1):
            query_id = f"B{idx:03d}"
            if on_progress is not None:
                try:
                    on_progress(idx - 1, total, f"Processing {query_id}: {query[:60]}")
                except Exception:
                    pass

            result = self._execute_single_query(query_id, query.strip())
            results.append(result)

            if on_progress is not None:
                try:
                    on_progress(idx, total, f"{query_id}: {result.status}")
                except Exception:
                    pass

        succeeded = sum(1 for r in results if r.status == "COMPLETED")
        failed = sum(1 for r in results if r.status == "FAILED")
        log.info("Batch query complete: %d/%d succeeded, %d failed", succeeded, total, failed)

        return results

    def process_queries_csv(self, csv_content: str | bytes) -> str:
        """Process queries from CSV content and return results as CSV string.

        Input CSV must have a 'query' column. Output CSV has columns:
        id, query, status, answer, duration_ms.

        Args:
            csv_content: CSV file content (string or bytes).

        Returns:
            CSV string with results.
        """
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8", errors="ignore")

        reader = csv.DictReader(io.StringIO(csv_content))
        if reader.fieldnames is None or "query" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'query' column.")

        queries = [row["query"].strip() for row in reader if row.get("query", "").strip()]
        if not queries:
            raise ValueError("No valid queries found in CSV.")

        results = self.process_queries(queries)

        # Write output CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "query", "status", "answer", "duration_ms"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "id": result.query_id,
                "query": result.query,
                "status": result.status,
                "answer": result.answer,
                "duration_ms": result.duration_ms,
            })

        return output.getvalue()

    # ------------------------------------------------------------------
    # Convenience: export results to CSV
    # ------------------------------------------------------------------
    @staticmethod
    def results_to_csv(results: list[QueryResult]) -> str:
        """Convert a list of QueryResult to a CSV string."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "query", "status", "answer", "duration_ms"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r.query_id,
                "query": r.query,
                "status": r.status,
                "answer": r.answer,
                "duration_ms": r.duration_ms,
            })
        return output.getvalue()
