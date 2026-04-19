"""Retrieval service with retry, timeout, and batch query support.

Provides robust query handling with:
- Tenacity-based retry with exponential backoff
- Async timeout handling
- Concurrent batch query processing
- Ollama service validation
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests

from local_archive_ai.logging_config import log

# Optional imports with fallbacks
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from local_archive_ai.store import LocalVectorStore, SearchHit
from local_archive_ai.services import (
    EmbeddingService,
    HybridRetriever,
    HybridSearchResult,
)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class QueryTimeoutError(Exception):
    """Raised when query exceeds timeout limit."""
    pass


class QueryError(Exception):
    """Raised when query fails after retries."""
    pass


class OllamaError(Exception):
    """Raised when Ollama service is unavailable or model not found."""
    pass


@dataclass
class QueryResult:
    """Result of a single query."""
    query: str
    chunks: list[SearchHit]
    status: str  # 'success', 'timeout', 'error'
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class OllamaValidator:
    """Validates Ollama service availability and model status."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def health_check(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            session = self._get_session()
            response = session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if specific model is downloaded and available."""
        model = model_name or self.model
        if not model:
            return False

        try:
            session = self._get_session()
            response = session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                return False

            data = response.json()
            models = data.get("models", [])

            # Check if model name matches any available model
            model_base = model.split(":")[0].lower()
            for m in models:
                name = m.get("name", "").lower()
                if model_base in name or name in model.lower():
                    return True
            return False
        except Exception:
            return False

    def validate_before_request(self, model_name: Optional[str] = None) -> None:
        """Validate Ollama before making a request.

        Raises:
            OllamaError: If service unavailable or model not found
        """
        model = model_name or self.model

        if not self.health_check():
            raise OllamaError(
                f"Ollama service is not running at {self.base_url}. "
                "Please start Ollama with: ollama serve"
            )

        if model and not self.model_available(model):
            raise OllamaError(
                f"Model '{model}' is not available. "
                f"Please pull the model with: ollama pull {model}"
            )

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        try:
            session = self._get_session()
            response = session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                return []

            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models]
        except Exception:
            return []


class RetrievalService:
    """Service for document retrieval with retry and timeout handling."""

    def __init__(
        self,
        store_path: str | Path,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "",
        max_retries: int = 3,
        query_timeout: float = 30.0,
        enable_bm25: bool = True,
    ):
        """Initialize retrieval service.

        Args:
            store_path: Path to vector store
            ollama_url: Ollama service URL
            ollama_model: Default Ollama model name
            max_retries: Maximum retry attempts for queries
            query_timeout: Timeout per query in seconds
            enable_bm25: Enable BM25 hybrid search
        """
        self.store_path = Path(store_path)
        self.ollama_validator = OllamaValidator(ollama_url, ollama_model)
        self.max_retries = max_retries
        self.query_timeout = query_timeout
        self.enable_bm25 = enable_bm25 and BM25_AVAILABLE

        # Initialize store and services
        self.vector_store = LocalVectorStore(self.store_path)
        self.embedding_service = EmbeddingService()
        self.retriever: Optional[HybridRetriever] = None

        # Load index if available
        self._load_index()

    def _load_index(self) -> bool:
        """Load the vector index.

        Returns:
            True if index loaded successfully
        """
        if self.vector_store.load() and self.vector_store.ready():
            self.retriever = HybridRetriever(self.vector_store, self.embedding_service)
            if self.enable_bm25:
                self.retriever.build_bm25_index(self.vector_store.metadata)
            return True
        return False

    def ensure_index_loaded(self) -> None:
        """Ensure index is loaded, raising error if not."""
        if not self._load_index():
            raise RuntimeError(
                f"FAISS index not found at {self.store_path}. "
                "Please index documents first."
            )

    def _create_retry_decorator(self):
        """Create tenacity retry decorator."""
        if not TENACITY_AVAILABLE:
            # Return identity decorator if tenacity not available
            def no_retry(func):
                return func
            return no_retry

        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((QueryError, ConnectionError)),
            reraise=True,
        )

    async def retrieve_with_timeout(
        self,
        query: str,
        top_k: int = 10,
        retrieval_mode: str = "hybrid",
        bm25_weight: float = 0.4,
    ) -> list[HybridSearchResult]:
        """Retrieve chunks with timeout handling.

        Args:
            query: Search query
            top_k: Number of results
            retrieval_mode: 'vector', 'bm25', 'hybrid', or 'hybrid+rerank'
            bm25_weight: Weight for BM25 in hybrid search

        Returns:
            List of search results

        Raises:
            QueryTimeoutError: If query exceeds timeout
            QueryError: If retrieval fails
        """
        self.ensure_index_loaded()

        if self.retriever is None:
            raise QueryError("Retriever not initialized")

        async def _do_retrieve():
            if retrieval_mode == "vector":
                return self.retriever.search_vector(query, top_k)
            elif retrieval_mode == "bm25":
                return self.retriever.search_bm25(query, top_k)
            elif retrieval_mode == "hybrid":
                return self.retriever.search_hybrid(query, top_k, bm25_weight)
            elif retrieval_mode == "hybrid+rerank":
                candidates = self.retriever.search_hybrid(query, top_k * 2, bm25_weight)
                return self.retriever.rerank_with_cross_encoder(query, candidates, top_k)
            else:
                return self.retriever.search_vector(query, top_k)

        try:
            return await asyncio.wait_for(_do_retrieve(), timeout=self.query_timeout)
        except asyncio.TimeoutError:
            raise QueryTimeoutError(
                f"Query exceeded timeout of {self.query_timeout}s"
            )
        except Exception as e:
            raise QueryError(f"Retrieval failed: {str(e)}")

    async def batch_query(
        self,
        queries: list[str],
        top_k: int = 10,
        max_concurrent: int = 5,
        retrieval_mode: str = "hybrid",
        bm25_weight: float = 0.4,
    ) -> list[QueryResult]:
        """Process multiple queries concurrently with controlled concurrency.

        Args:
            queries: List of query strings
            top_k: Results per query
            max_concurrent: Maximum concurrent queries
            retrieval_mode: Search mode
            bm25_weight: BM25 weight for hybrid search

        Returns:
            List of QueryResult, one per query
        """
        self.ensure_index_loaded()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_retrieve(query: str) -> QueryResult:
            async with semaphore:
                start_time = time.time()
                try:
                    chunks = await self.retrieve_with_timeout(
                        query, top_k, retrieval_mode, bm25_weight
                    )
                    # Convert HybridSearchResult to SearchHit
                    search_hits = [
                        SearchHit(
                            text=c.text,
                            score=c.score,
                            metadata=c.metadata,
                        )
                        for c in chunks
                    ]
                    duration_ms = (time.time() - start_time) * 1000
                    return QueryResult(
                        query=query,
                        chunks=search_hits,
                        status="success",
                        duration_ms=duration_ms,
                    )
                except QueryTimeoutError as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return QueryResult(
                        query=query,
                        chunks=[],
                        status="timeout",
                        error_message=str(e),
                        duration_ms=duration_ms,
                    )
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    return QueryResult(
                        query=query,
                        chunks=[],
                        status="error",
                        error_message=str(e),
                        duration_ms=duration_ms,
                    )

        # Process all queries concurrently with semaphore control
        results = await asyncio.gather(*[_bounded_retrieve(q) for q in queries])
        return list(results)

    def validate_ollama(self, model_name: Optional[str] = None) -> dict[str, Any]:
        """Validate Ollama service status.

        Args:
            model_name: Optional model to check

        Returns:
            Status dictionary with health info
        """
        health = self.ollama_validator.health_check()
        model = model_name or self.ollama_validator.model

        result: dict[str, Any] = {
            "healthy": health,
            "url": self.ollama_validator.base_url,
            "model_requested": model,
        }

        if not health:
            result["error"] = (
                "Ollama service is not running. "
                "Start with: ollama serve"
            )
            result["status_code"] = 503
            return result

        if model:
            model_available = self.ollama_validator.model_available(model)
            result["model_available"] = model_available
            if not model_available:
                available = self.ollama_validator.get_available_models()
                result["available_models"] = available
                result["error"] = (
                    f"Model '{model}' not found. "
                    f"Available: {', '.join(available[:5])}... "
                    f"Pull with: ollama pull {model}"
                )
                result["status_code"] = 503
                return result

        result["status_code"] = 200
        return result


class BatchQueryProcessor:
    """Processor for batch queries with progress tracking."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.retrieval_service = retrieval_service
        self.progress_callback = progress_callback

    async def process_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        max_concurrent: int = 5,
    ) -> dict[str, Any]:
        """Process a batch of queries with detailed results.

        Args:
            queries: List of queries to process
            top_k: Results per query
            max_concurrent: Maximum concurrent queries

        Returns:
            Dictionary with results, statistics, and errors
        """
        total = len(queries)
        results = await self.retrieval_service.batch_query(
            queries, top_k, max_concurrent
        )

        # Aggregate statistics
        successful = sum(1 for r in results if r.status == "success")
        timeouts = sum(1 for r in results if r.status == "timeout")
        errors = sum(1 for r in results if r.status == "error")

        total_duration_ms = sum(r.duration_ms for r in results)
        avg_duration_ms = total_duration_ms / total if total > 0 else 0

        return {
            "total": total,
            "successful": successful,
            "timeouts": timeouts,
            "errors": errors,
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": avg_duration_ms,
            "results": [
                {
                    "query": r.query,
                    "status": r.status,
                    "chunks_count": len(r.chunks),
                    "error": r.error_message,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
            "successful_results": [
                {
                    "query": r.query,
                    "chunks": [
                        {
                            "text": c.text[:200],  # Truncate for response
                            "score": c.score,
                            "metadata": {
                                "file_name": c.metadata.get("file_name", "unknown"),
                                "chunk_index": c.metadata.get("chunk_index"),
                            },
                        }
                        for c in r.chunks[:5]  # Include top 5
                    ],
                }
                for r in results if r.status == "success"
            ],
        }
