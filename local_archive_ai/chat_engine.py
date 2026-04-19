"""Chat Engine with RetrievalQA chain, streaming responses, conversation memory, and fallback.

Implements a working chat engine connected to FAISS vector store + Ollama LLM with:
- LangChain-style RetrievalQA chain (FAISS retrieval → prompt composition → Ollama generation)
- Streaming response display compatible with Streamlit's st.write_stream
- Conversation memory (last 5 messages as context)
- Fallback: "I cannot find this information" when no relevant chunks found
- Timeout (120s per operation), retry logic (3 attempts with exponential backoff)
- Validation and error handling for corrupted/missing data
- Logging to local_archive.log with timestamps
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import requests

from local_archive_ai.logging_config import log
from local_archive_ai.services import (
    EmbeddingService,
    HybridRetriever,
    SearchHit,
    _compose_prompt,
    _get_ollama_session,
    _sanitize_text,
    check_ollama_status,
)
from local_archive_ai.store import LocalVectorStore

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT = 120  # seconds per operation
_MAX_RETRIES = 3
_FALLBACK_MESSAGE = "I cannot find this information in the indexed documents. Please try rephrasing your question or ensure the relevant documents have been indexed."
_MEMORY_SIZE = 5  # last N conversation turns to keep as context


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """A single conversation turn (question + answer)."""
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0
    low_confidence: bool = False
    duration_ms: int = 0


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    max_similarity: float = 0.0
    low_confidence: bool = False
    duration_ms: int = 0
    chunks_used: int = 0
    debug_payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _retry_with_backoff(
    fn,
    max_retries: int = _MAX_RETRIES,
    timeout: float = _DEFAULT_TIMEOUT,
    operation_name: str = "operation",
) -> Any:
    """Execute *fn* with exponential backoff retry logic.

    Args:
        fn: Callable to execute (no args).
        max_retries: Maximum number of retry attempts.
        timeout: Per-attempt timeout in seconds (not enforced here, caller must handle).
        operation_name: Label for log messages.

    Returns:
        Result of *fn* on success.

    Raises:
        Last exception encountered after all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            log.warning(
                "%s failed (attempt %d/%d): %s – retrying in %ds",
                operation_name, attempt, max_retries, exc, wait,
            )
            if attempt < max_retries:
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------
class ChatEngine:
    """Full-featured chat engine backed by FAISS retrieval and Ollama generation.

    Usage::

        engine = ChatEngine(
            store_path="data/faiss_index",
            model_name="llama3.2:1b",
            ollama_endpoint="http://127.0.0.1:11434",
        )
        response = engine.query("What is RAG?")
        print(response.answer)

    For streaming in Streamlit::

        for token in engine.query_stream("What is RAG?"):
            st.write(token, end="")
    """

    def __init__(
        self,
        store_path: str = "data/faiss_index",
        model_name: str = "llama3.2:1b",
        ollama_endpoint: str = "http://127.0.0.1:11434",
        ollama_api_key: str = "",
        top_k: int = 4,
        retrieval_mode: str = "vector",
        bm25_weight: float = 0.4,
        rerank_top_n: int = 5,
        confidence_threshold: float = 0.65,
        temperature: float = 0.7,
        max_context_tokens: int = 8192,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.store_path = store_path
        self.model_name = model_name
        self.ollama_endpoint = ollama_endpoint
        self.ollama_api_key = ollama_api_key
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.bm25_weight = bm25_weight
        self.rerank_top_n = rerank_top_n
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Conversation memory – fixed-size deque of recent turns
        self._memory: deque[ConversationTurn] = deque(maxlen=_MEMORY_SIZE)

        # Lazy-loaded heavy objects
        self._store: LocalVectorStore | None = None
        self._embedder: EmbeddingService | None = None
        self._retriever: HybridRetriever | None = None

        log.info(
            "ChatEngine initialised (model=%s, store=%s, top_k=%d, mode=%s)",
            model_name, store_path, top_k, retrieval_mode,
        )

    # ------------------------------------------------------------------
    # Properties / lazy loaders
    # ------------------------------------------------------------------
    @property
    def store(self) -> LocalVectorStore:
        if self._store is None:
            self._store = LocalVectorStore(Path(self.store_path))
        return self._store

    @property
    def embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = EmbeddingService()
        return self._embedder

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.store, self.embedder)
        return self._retriever

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def add_to_memory(self, question: str, answer: str, citations: list[dict[str, Any]] | None = None) -> None:
        """Append a conversation turn to memory."""
        self._memory.append(ConversationTurn(
            question=question,
            answer=answer,
            citations=citations or [],
            timestamp=time.time(),
        ))

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self._memory.clear()
        log.info("ChatEngine memory cleared")

    def get_memory(self) -> list[ConversationTurn]:
        """Return conversation history as a list (oldest first)."""
        return list(self._memory)

    def _build_memory_context(self) -> str:
        """Build a textual summary of recent conversation turns for the prompt."""
        if not self._memory:
            return ""
        lines: list[str] = ["Previous conversation:"]
        for turn in self._memory:
            lines.append(f"User: {turn.question}")
            # Truncate long answers in the memory context
            answer_preview = turn.answer[:500] + ("..." if len(turn.answer) > 500 else "")
            lines.append(f"Assistant: {answer_preview}")
        lines.append("")  # blank separator
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _load_store(self) -> bool:
        """Load the FAISS index with retry logic."""
        def _do_load() -> bool:
            if not self.store.load() or not self.store.ready():
                raise RuntimeError("FAISS index is not loaded or not ready.")
            return True

        try:
            return _retry_with_backoff(_do_load, max_retries=self.max_retries, operation_name="load_store")
        except Exception:
            log.error("Failed to load FAISS index after %d retries", self.max_retries)
            return False

    def _retrieve(self, query: str) -> list[SearchHit]:
        """Retrieve relevant chunks using the configured retrieval mode."""
        # Build BM25 index if available
        if BM25Okapi is not None:
            self.retriever.build_bm25_index(self.store.metadata)

        if self.retrieval_mode == "vector":
            return self.retriever.search_vector(query, self.top_k)
        elif self.retrieval_mode == "hybrid":
            return self.retriever.search_hybrid(query, self.top_k, self.bm25_weight)
        elif self.retrieval_mode == "hybrid+rerank":
            candidates = self.retriever.search_hybrid(query, self.top_k * 2, self.bm25_weight)
            return self.retriever.rerank_with_cross_encoder(query, candidates, self.rerank_top_n)
        else:
            return self.retriever.search_vector(query, self.top_k)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, hits: list[SearchHit]) -> tuple[str, int]:
        """Build prompt with conversation memory and retrieved context."""
        memory_context = self._build_memory_context()

        # Use the existing _compose_prompt for the retrieval part
        retrieval_prompt, chunks_used = _compose_prompt(
            question, hits, self.model_name, max_context_tokens=self.max_context_tokens,
        )

        if memory_context:
            # Extract the context part from retrieval prompt and combine with memory
            if "Context:" in retrieval_prompt and "\n\nQuestion:" in retrieval_prompt:
                # Split to get the context section
                before_context = retrieval_prompt.split("Context:")[0]
                after_context_marker = retrieval_prompt.split("Context:")[1]
                if "\n\nQuestion:" in after_context_marker:
                    context_part = after_context_marker.split("\n\nQuestion:")[0]
                    question_part = "\n\nQuestion:" + after_context_marker.split("\n\nQuestion:")[1]
                    
                    # Rebuild prompt with memory inserted before context
                    prompt = (
                        f"{before_context.rstrip()}\n\n"
                        f"{memory_context}\n\n"
                        f"Context:{context_part}{question_part}"
                    )
                else:
                    # Fallback: just prepend memory to the full prompt
                    prompt = f"{memory_context}\n\n{retrieval_prompt}"
            else:
                # Fallback: just prepend memory to the full prompt
                prompt = f"{memory_context}\n\n{retrieval_prompt}"
        else:
            prompt = retrieval_prompt

        return prompt, chunks_used

    def _generate(self, prompt: str) -> str:
        """Generate response from Ollama with retry and timeout."""
        session = _get_ollama_session()
        headers: dict[str, str] = {}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        def _do_generate() -> str:
            resp = session.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": self.temperature},
                },
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            response = str(data.get("response", "")).strip()
            if not response:
                return "Model generated an empty response."
            return response

        return _retry_with_backoff(_do_generate, max_retries=self.max_retries, operation_name="ollama_generate")

    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Streaming generator for Ollama responses with timeout."""
        session = _get_ollama_session()
        headers: dict[str, str] = {}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        try:
            resp = session.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": -1,
                    "options": {"temperature": self.temperature},
                },
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.error("Streaming generation error: %s", e)
            yield f"\n\n[Streaming error: {e}]"

    # ------------------------------------------------------------------
    # Build citations from hits
    # ------------------------------------------------------------------
    @staticmethod
    def _build_citations(hits: list[SearchHit]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for hit in hits:
            file_path = hit.metadata.get("file_path", "")
            citations.append({
                "file_path": file_path,
                "file_name": hit.metadata.get("file_name", "unknown"),
                "chunk_text": hit.text,
                "score": hit.score,
                "source_page": hit.metadata.get("source_page"),
                "chunk_index": hit.metadata.get("chunk_index"),
                "open_uri": Path(file_path).resolve().as_uri() if file_path else "",
            })
        return citations

    # ------------------------------------------------------------------
    # Main query methods
    # ------------------------------------------------------------------
    def query(self, question: str) -> ChatResponse:
        """Run a full retrieval-augmented query and return a ChatResponse.

        Steps:
        1. Load FAISS index
        2. Retrieve relevant chunks
        3. Check confidence – return fallback if no relevant content
        4. Build prompt with memory context
        5. Generate answer via Ollama (with retry)
        6. Add turn to memory
        7. Return structured response
        """
        t0 = time.time()
        question = question.strip()
        if not question:
            return ChatResponse(answer="Please enter a question.", duration_ms=0)

        # Step 1: Load index
        if not self._load_store():
            return ChatResponse(
                answer="FAISS index is not loaded. Please index files first.",
                duration_ms=int((time.time() - t0) * 1000),
            )

        # Step 2: Retrieve
        try:
            hits = self._retrieve(question)
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            return ChatResponse(
                answer=f"Retrieval error: {exc}",
                duration_ms=int((time.time() - t0) * 1000),
            )

        # Step 3: Check relevance – fallback if no hits
        if not hits:
            log.info("No relevant chunks found for query: %s", question[:80])
            self.add_to_memory(question, _FALLBACK_MESSAGE)
            return ChatResponse(
                answer=_FALLBACK_MESSAGE,
                low_confidence=True,
                duration_ms=int((time.time() - t0) * 1000),
            )

        max_similarity = max(h.score for h in hits)
        low_confidence = max_similarity < self.confidence_threshold
        citations = self._build_citations(hits)

        # Step 4: Build prompt with memory
        prompt, chunks_used = self._build_prompt(question, hits)

        # Step 5: Generate
        try:
            if not check_ollama_status(self.ollama_endpoint, self.ollama_api_key):
                answer = (
                    f"Ollama is not running on {self.ollama_endpoint}.\n\n"
                    "Retrieval succeeded, but the final response cannot be generated until Ollama is available."
                )
            else:
                answer = self._generate(prompt)
        except Exception as exc:
            log.error("Generation failed: %s", exc)
            answer = f"Generation error: {exc}"

        # Prepend low-confidence warning
        if low_confidence and answer and not answer.startswith("Ollama"):
            answer = (
                f"**Low Confidence Warning:** The documents may not clearly answer this question "
                f"(max similarity: {max_similarity:.3f}).\n\n{answer}"
            )

        # Step 6: Store in memory
        duration_ms = int((time.time() - t0) * 1000)
        self.add_to_memory(question, answer, citations)

        # Query embedding for debug
        query_vec = self.embedder.embed([question])[0]

        log.info(
            "ChatEngine query answered in %dms (mode=%s, chunks=%d, confidence=%.3f)",
            duration_ms, self.retrieval_mode, chunks_used, max_similarity,
            extra={"query": question, "duration_ms": duration_ms},
        )

        return ChatResponse(
            answer=answer,
            citations=citations,
            max_similarity=max_similarity,
            low_confidence=low_confidence,
            duration_ms=duration_ms,
            chunks_used=chunks_used,
            debug_payload={
                "embedding_preview": [float(x) for x in query_vec[:16]],
                "retrieved_chunks": [
                    {
                        "score": c["score"],
                        "file_name": c["file_name"],
                        "source_page": c["source_page"],
                        "chunk_index": c["chunk_index"],
                        "text_preview": c["chunk_text"][:240],
                    }
                    for c in citations
                ],
                "prompt_text": prompt,
                "memory_turns": len(self._memory),
                "retrieval_mode": self.retrieval_mode,
            },
        )

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Streaming version of query – yields tokens as they arrive.

        Also stores the full response in memory after streaming completes.

        Usage with Streamlit::

            engine = ChatEngine(...)
            response_container = st.empty()
            full_response = ""
            for token in engine.query_stream("What is RAG?"):
                full_response += token
                response_container.markdown(full_response)
        """
        question = question.strip()
        if not question:
            yield "Please enter a question."
            return

        # Load index
        if not self._load_store():
            yield "FAISS index is not loaded. Please index files first."
            return

        # Retrieve
        try:
            hits = self._retrieve(question)
        except Exception as exc:
            log.error("Retrieval failed during stream: %s", exc)
            yield f"Retrieval error: {exc}"
            return

        # Fallback if no hits
        if not hits:
            log.info("No relevant chunks found for streaming query: %s", question[:80])
            self.add_to_memory(question, _FALLBACK_MESSAGE)
            yield _FALLBACK_MESSAGE
            return

        max_similarity = max(h.score for h in hits)
        low_confidence = max_similarity < self.confidence_threshold
        citations = self._build_citations(hits)

        # Build prompt with memory
        prompt, chunks_used = self._build_prompt(question, hits)

        # Check Ollama
        if not check_ollama_status(self.ollama_endpoint, self.ollama_api_key):
            msg = (
                f"Ollama is not running on {self.ollama_endpoint}.\n\n"
                "Retrieval succeeded, but the final response cannot be generated until Ollama is available."
            )
            self.add_to_memory(question, msg, citations)
            yield msg
            return

        # Low confidence prefix
        if low_confidence:
            prefix = (
                f"**Low Confidence Warning:** The documents may not clearly answer this question "
                f"(max similarity: {max_similarity:.3f}).\n\n"
            )
            yield prefix
        else:
            prefix = ""

        # Stream tokens
        collected_tokens: list[str] = []
        for token in self._generate_stream(prompt):
            collected_tokens.append(token)
            yield token

        # Store full response in memory
        full_answer = prefix + "".join(collected_tokens)
        self.add_to_memory(question, full_answer, citations)

        log.info(
            "ChatEngine streaming query completed (mode=%s, chunks=%d)",
            self.retrieval_mode, chunks_used,
        )

    # ------------------------------------------------------------------
    # Convenience: get last response metadata (for debug panels)
    # ------------------------------------------------------------------
    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent conversation turn, or None."""
        return self._memory[-1] if self._memory else None
