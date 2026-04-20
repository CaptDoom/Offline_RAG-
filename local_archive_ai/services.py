from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import signal
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import re
import requests

from local_archive_ai.logging_config import log

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

from local_archive_ai.store import LocalVectorStore, SearchHit

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # pragma: no cover
    SentenceTransformer = None
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover
    Image = None
    ImageFilter = None

try:
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cv2 = None

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:
    import pytesseract  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _configure_tesseract_cmd() -> None:
    if pytesseract is None:
        return
    try:
        default_exe = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default_exe.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default_exe)
    except Exception:
        return


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".json",
    ".csv",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".go",
    ".rs",
    ".html",
    ".css",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
}
SKIP_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
_ROOT_DIR = Path(__file__).resolve().parent.parent
_MODEL_CACHE_DIR = _ROOT_DIR / "data" / "models"
_EMBED_MODEL_LOCK = threading.Lock()
_RERANK_MODEL_LOCK = threading.Lock()

# Ollama HTTP session pool (Problem: connection pooling)
_ollama_session: requests.Session | None = None
_ollama_session_lock = threading.Lock()


def _get_ollama_session() -> requests.Session:
    """Return a reusable requests.Session for Ollama calls."""
    global _ollama_session
    if _ollama_session is None:
        with _ollama_session_lock:
            if _ollama_session is None:
                _ollama_session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=4, pool_maxsize=8, max_retries=1
                )
                _ollama_session.mount("http://", adapter)
                _ollama_session.mount("https://", adapter)
    return _ollama_session


def _force_offline_transformers_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


class IngestionStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class FileIngestionResult:
    file_path: str
    file_name: str
    status: IngestionStatus
    chunks_count: int = 0
    error_message: str = ""
    ocr_engine_used: str = ""
    processing_time_seconds: float = 0.0
    file_hash: str = ""


@dataclass
class IngestionReport:
    total_files: int
    processed_files: int
    successful_files: int
    warning_files: int
    error_files: int
    skipped_files: int
    total_chunks: int
    completed_at: str
    processing_time_seconds: float
    file_results: list[FileIngestionResult]
    success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "successful_files": self.successful_files,
            "warning_files": self.warning_files,
            "error_files": self.error_files,
            "skipped_files": self.skipped_files,
            "total_chunks": self.total_chunks,
            "completed_at": self.completed_at,
            "processing_time_seconds": self.processing_time_seconds,
            "success_rate": self.success_rate,
            "file_results": [
                {
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "status": r.status.value,
                    "chunks_count": r.chunks_count,
                    "error_message": r.error_message,
                    "ocr_engine_used": r.ocr_engine_used,
                    "processing_time_seconds": r.processing_time_seconds,
                    "file_hash": r.file_hash,
                }
                for r in self.file_results
            ],
        }


@dataclass
class IndexReport:
    file_count: int
    chunk_count: int
    completed_at: str
    index_path: str
    faiss_enabled: bool


class EmbeddingService:
    """Lazy-loaded embedding service with batch processing (Problem W: OOM-safe)."""
    _model: Any = None
    _model_attempted = False

    def __init__(self) -> None:
        self._cache_dir = _MODEL_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        if self.__class__._model_attempted:
            return

        with _EMBED_MODEL_LOCK:
            if self.__class__._model_attempted:
                return
            self.__class__._model_attempted = True
            if SentenceTransformer is not None:
                try:
                    _force_offline_transformers_env()
                    self.__class__._model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        cache_folder=str(self._cache_dir),
                        device="cpu",
                        local_files_only=True,
                    )
                    log.info("Embedding model loaded: all-MiniLM-L6-v2")
                except Exception as exc:
                    self.__class__._model = None
                    log.warning("Embedding model unavailable offline, using fallback: %s", exc)

    @property
    def model(self) -> Any:
        self._load_model()
        return self.__class__._model

    def embed(self, texts: list[str], batch_size: int = 50) -> np.ndarray:
        """Embed texts in batches to prevent OOM (Problem W)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        self._load_model()
        if self.model is not None:
            batches: list[np.ndarray] = []
            for start in range(0, len(texts), max(1, batch_size)):
                chunk = texts[start : start + max(1, batch_size)]
                vectors = self.model.encode(
                    chunk,
                    batch_size=max(1, min(batch_size, len(chunk))),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                batches.append(np.asarray(vectors, dtype=np.float32))
            return np.concatenate(batches, axis=0)

        # Deterministic local fallback if sentence-transformers is unavailable.
        vectors = []
        dim = 384
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for token in text.split():
                idx = abs(hash(token)) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec) + 1e-12
            vectors.append(vec / norm)
        return np.stack(vectors, axis=0)


class ImageEmbeddingService:
    """Embedding service for Image Semantic Search using CLIP."""
    _model: Any = None
    _model_attempted = False

    def __init__(self) -> None:
        self._cache_dir = _MODEL_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        if self.__class__._model_attempted:
            return
        with _EMBED_MODEL_LOCK:
            if self.__class__._model_attempted:
                return
            self.__class__._model_attempted = True
            if SentenceTransformer is not None:
                try:
                    _force_offline_transformers_env()
                    self.__class__._model = SentenceTransformer(
                        "clip-ViT-B-32",
                        cache_folder=str(self._cache_dir),
                        device="cpu",
                        local_files_only=True,
                    )
                    log.info("Image Embedding model loaded: clip-ViT-B-32")
                except Exception as exc:
                    self.__class__._model = None
                    log.warning("Image embedding model unavailable offline: %s", exc)

    @property
    def model(self) -> Any:
        self._load_model()
        return self.__class__._model

    def embed_images(self, image_paths: list[str], batch_size: int = 12) -> np.ndarray:
        self._load_model()
        if self.model is not None:
            from PIL import Image
            batches = []
            for start in range(0, len(image_paths), max(1, batch_size)):
                chunk_paths = image_paths[start : start + max(1, batch_size)]
                images = []
                for p in chunk_paths:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        images.append(Image.new('RGB', (224, 224), color='black'))
                vectors = self.model.encode(
                    images,
                    batch_size=len(images),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                batches.append(np.asarray(vectors, dtype=np.float32))
            if batches:
                return np.concatenate(batches, axis=0)
            return np.empty((0, 512), dtype=np.float32)
        return np.empty((len(image_paths), 512), dtype=np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        if self.model is not None:
            vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        return np.empty((len(texts), 512), dtype=np.float32)


def load_reranker_model() -> Any:
    if CrossEncoder is None:
        return None

    if getattr(load_reranker_model, "_attempted", False):
        return getattr(load_reranker_model, "_model", None)

    with _RERANK_MODEL_LOCK:
        if getattr(load_reranker_model, "_attempted", False):
            return getattr(load_reranker_model, "_model", None)

        setattr(load_reranker_model, "_attempted", True)
        try:
            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                cache_folder=str(_MODEL_CACHE_DIR),
                local_files_only=True,
            )
        except Exception:
            model = None
        setattr(load_reranker_model, "_model", model)
        return model


@dataclass
class HybridSearchResult:
    text: str
    score: float
    metadata: dict[str, Any]
    source: str  # 'vector', 'bm25', or 'hybrid'


class HybridRetriever:
    def __init__(self, vector_store: LocalVectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []

    def build_bm25_index(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from document corpus."""
        if BM25Okapi is None:
            return

        self.bm25_corpus = []
        self.bm25_metadata = []

        for doc in documents:
            # Tokenize text for BM25
            text = doc.get('text', '')
            tokens = self._tokenize_for_bm25(text)
            self.bm25_corpus.append(tokens)
            self.bm25_metadata.append(doc)

        self.bm25_index = BM25Okapi(self.bm25_corpus)

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization: lowercase, remove punctuation, split on whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1] range."""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def search_vector(self, query: str, top_k: int) -> list[HybridSearchResult]:
        """Pure vector search, with BM25 fallback if embedding fails."""
        hits = []
        try:
            query_vec = self.embedding_service.embed([query])[0]
            hits = self.vector_store.search(query_vec, top_k=top_k)
        except Exception:
            hits = []

        if not hits and self.bm25_index is not None:
            return self.search_bm25(query, top_k)

        results = []
        for hit in hits:
            results.append(HybridSearchResult(
                text=hit.text,
                score=hit.score,
                metadata=hit.metadata,
                source='vector'
            ))
        return results

    def search_bm25(self, query: str, top_k: int) -> list[HybridSearchResult]:
        """Pure BM25 search."""
        if self.bm25_index is None or BM25Okapi is None:
            return []

        query_tokens = self._tokenize_for_bm25(query)
        bm25_scores = np.array(self.bm25_index.get_scores(query_tokens), dtype=np.float32)
        if bm25_scores.size == 0:
            return []

        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(HybridSearchResult(
                text=self.bm25_metadata[idx].get('text', ''),
                score=float(bm25_scores[idx]),
                metadata=self.bm25_metadata[idx],
                source='bm25'
            ))
        return results

    def search_hybrid(self, query: str, top_k: int, bm25_weight: float = 0.4) -> list[HybridSearchResult]:
        """Hybrid search combining vector and BM25 scores."""
        vector_results = self.search_vector(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        if not vector_results and not bm25_results:
            return []

        vector_scores = self._normalize_scores([r.score for r in vector_results]) if vector_results else []
        bm25_scores = self._normalize_scores([r.score for r in bm25_results]) if bm25_results else []

        combined: dict[str, dict[str, Any]] = {}

        for idx, result in enumerate(vector_results):
            key = f"{result.metadata.get('file_path','')}::{result.metadata.get('chunk_index','')}"
            combined[key] = {
                'result': result,
                'vector_score': vector_scores[idx],
                'bm25_score': 0.0,
            }

        for idx, result in enumerate(bm25_results):
            key = f"{result.metadata.get('file_path','')}::{result.metadata.get('chunk_index','')}"
            if key in combined:
                combined[key]['bm25_score'] = bm25_scores[idx]
            else:
                combined[key] = {
                    'result': result,
                    'vector_score': 0.0,
                    'bm25_score': bm25_scores[idx],
                }

        for item in combined.values():
            hybrid_score = (1.0 - bm25_weight) * item['vector_score'] + bm25_weight * item['bm25_score']
            item['hybrid_score'] = hybrid_score
            item['result'].score = hybrid_score
            item['result'].source = 'hybrid'

        sorted_results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)
        return [item['result'] for item in sorted_results[:top_k]]

    def rerank_with_cross_encoder(self, query: str, candidates: list[HybridSearchResult], top_n: int) -> list[HybridSearchResult]:
        """Rerank candidates using cross-encoder model."""
        if not candidates:
            return candidates[:top_n]

        model = load_reranker_model()
        if model is None:
            return candidates[:top_n]

        try:
            query_doc_pairs = [[query, result.text] for result in candidates]
            scores = model.predict(query_doc_pairs)
            for result, score in zip(candidates, scores):
                result.score = float(score)
                result.source = f"{result.source}+rerank"
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            return reranked[:top_n]
        except Exception:
            return candidates[:top_n]


_SPLITTER_CACHE = {}
def _build_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter | None:
    if RecursiveCharacterTextSplitter is None or tiktoken is None:
        return None
    if chunk_size in _SPLITTER_CACHE:
        return _SPLITTER_CACHE[chunk_size]
    # cl100k_base aligns with modern token budgeting and avoids word-count drift.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=max(60, int(chunk_size * 0.15)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    _SPLITTER_CACHE[chunk_size] = splitter
    return splitter


def _split_with_tokens(text: str, chunk_size: int) -> list[str]:
    if not text.strip():
        return []
    splitter = _build_splitter(chunk_size)
    if splitter is None:
        # Simple deterministic fallback by words when token splitter is unavailable.
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        overlap = max(30, int(chunk_size * 0.12))
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + chunk_size]).strip()
            if piece:
                chunks.append(piece)
        return chunks
    return splitter.split_text(text)


def _sanitize_text(text: str, max_length: int = 1500) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:max_length]


def _read_pdf(path: Path) -> str:
    """Extract text from PDF using multi-engine fallback chain."""
    errors = []

    # Try pypdf first
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = _sanitize_text("\n".join(pages))
            if text.strip():
                return text
        except Exception as e:
            errors.append(f"pypdf: {str(e)}")

    # Fallback to pdfplumber
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = _sanitize_text("\n".join(pages))
            if text.strip():
                return text
        except Exception as e:
            errors.append(f"pdfplumber: {str(e)}")

    # Final fallback: try OCR on PDF pages (convert to images first)
    if Image is not None:
        try:
            # Convert PDF pages to images and OCR them
            import tempfile
            import subprocess

            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images using pdftoppm (common on Linux/Mac) or similar
                try:
                    # Try pdftoppm first
                    result = subprocess.run([
                        "pdftoppm", "-png", "-r", "150", str(path), f"{temp_dir}/page"
                    ], capture_output=True, text=True, timeout=30)

                    if result.returncode == 0:
                        page_texts = []
                        for png_file in Path(temp_dir).glob("page-*.png"):
                            try:
                                page_text = _read_image_text_resilient(png_file)
                                if page_text.strip():
                                    page_texts.append(page_text)
                            except Exception:
                                continue
                        if page_texts:
                            return _sanitize_text("\n".join(page_texts))
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                # If pdftoppm not available, try OCR on first page only using pypdf to extract as image
                try:
                    reader = PdfReader(str(path))
                    if reader.pages:
                        # This is a simplified approach - in production you'd need proper PDF to image conversion
                        pass
                except Exception:
                    pass

        except Exception as e:
            errors.append(f"PDF-to-image OCR: {str(e)}")

    raise RuntimeError(f"Unable to extract text from PDF {path}. Errors: {'; '.join(errors)}")


def _preprocess_image_for_ocr(img: Any) -> Any:
    """Preprocess image for better OCR quality (Problem G).

    Steps: resize to ~300 DPI equivalent, adaptive threshold, denoise, deskew.
    """
    if img is None:
        return img
    try:
        # Convert to grayscale
        if img.mode != "L":
            img = img.convert("L")
        # Resize small images to ~300 DPI equivalent (assume 72 DPI input)
        width, height = img.size
        if width < 1500:
            scale = max(2, 3000 // max(width, 1))
            img = img.resize((width * scale, height * scale), Image.LANCZOS)
        # Sharpen
        if ImageFilter is not None:
            img = img.filter(ImageFilter.SHARPEN)
        # OpenCV-based denoising and adaptive threshold if available
        if cv2 is not None:
            arr = np.array(img)
            arr = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
            arr = cv2.adaptiveThreshold(
                arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            img = Image.fromarray(arr)
    except Exception:
        pass  # Return original image on any preprocessing failure
    return img


def _read_image_text_resilient(path: Path) -> str:
    """Extract text from image using multi-engine OCR chain: pytesseract -> easyocr -> fail."""
    errors = []

    # Try pytesseract first
    if Image is not None and pytesseract is not None:
        _configure_tesseract_cmd()
        try:
            with Image.open(path) as img:
                preprocessed = _preprocess_image_for_ocr(img.copy())
                text = pytesseract.image_to_string(preprocessed, timeout=15)
                if text.strip():
                    return _sanitize_text(text)
        except Exception as e:
            errors.append(f"pytesseract: {str(e)}")

    # Fallback to easyocr
    if easyocr is not None:
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            result = reader.readtext(str(path), detail=0)
            text = "\n".join(result)
            if text.strip():
                return _sanitize_text(text)
        except Exception as e:
            errors.append(f"easyocr: {str(e)}")

    if not errors:
        errors.append("No OCR engines available")
    log.debug("OCR failed for %s: %s", path, "; ".join(errors))
    return ""


def _read_image_text(path: Path, ocr_engine: str = "tesseract") -> str:
    """Read text from image with specified OCR engine.

    Args:
        path: Path to image file
        ocr_engine: OCR engine to use ('tesseract', 'easyocr', 'paddle')

    Returns:
        Extracted text string

    Raises:
        ValueError: If unsupported OCR engine is specified
    """
    supported_engines = {"tesseract", "easyocr", "paddle"}
    if ocr_engine not in supported_engines:
        raise ValueError(
            f"Unsupported OCR engine: {ocr_engine}. "
            f"Must be one of: {', '.join(supported_engines)}"
        )

    # Try the requested engine first, then fall back to resilient method
    if ocr_engine == "tesseract" and pytesseract is not None:
        _configure_tesseract_cmd()
        try:
            with Image.open(path) as img:
                preprocessed = _preprocess_image_for_ocr(img.copy())
                text = pytesseract.image_to_string(preprocessed, timeout=15)
                if text.strip():
                    return _sanitize_text(text)
        except Exception:
            pass  # Fall through to fallback

    if ocr_engine == "easyocr" and easyocr is not None:
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            result = reader.readtext(str(path), detail=0)
            text = "\n".join(result)
            if text.strip():
                return _sanitize_text(text)
        except Exception:
            pass  # Fall through to fallback

    # For 'paddle' or if primary engine failed, use resilient fallback
    return _read_image_text_resilient(path)


def _sort_image_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][2], item["bbox"][3]))


def _blocks_are_contiguous(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    prev_left, prev_top, prev_right, prev_bottom = previous["bbox"]
    curr_left, curr_top, curr_right, curr_bottom = current["bbox"]
    prev_height = prev_bottom - prev_top
    curr_height = curr_bottom - curr_top

    vertical_gap = curr_top - prev_bottom
    if vertical_gap <= max(prev_height, curr_height) * 1.5:
        return True

    horizontal_gap = curr_left - prev_right
    if horizontal_gap <= max(prev_right - prev_left, curr_right - curr_left, 1) * 0.5:
        return True

    return False


def _union_bbox(first: list[int], second: list[int]) -> list[int]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def _make_image_chunk(blocks: list[dict[str, Any]], chunk_index: int = 0) -> dict[str, Any]:
    """Create a chunk from image blocks with unique chunk_index.

    Args:
        blocks: List of OCR blocks to combine into a chunk
        chunk_index: Globally unique chunk index for this chunk

    Returns:
        Dict containing chunk data with unique chunk_index
    """
    chunk_text = " ".join(str(block["text"]).strip() for block in blocks if str(block["text"]).strip())
    chunk_text = _sanitize_text(chunk_text, max_length=1500)
    bbox = blocks[0]["bbox"]
    for block in blocks[1:]:
        bbox = _union_bbox(bbox, block["bbox"])

    return {
        "source_file": str(blocks[0]["file_path"]),
        "file_path": str(blocks[0]["file_path"]),
        "file_name": blocks[0]["file_name"],
        "source_page": None,
        "chunk_index": chunk_index,
        "text": chunk_text,
        "block_ids": [block["block_id"] for block in blocks],
        "bounding_boxes": [block["bbox"] for block in blocks],
        "bbox": bbox,
    }


def _extract_image_blocks_pytesseract(path: Path) -> list[dict[str, Any]]:
    _configure_tesseract_cmd()
    with Image.open(path) as img:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, timeout=10)

    blocks: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx in range(len(data["text"])):
        text = str(data["text"][idx]).strip()
        if not text:
            continue

        key = (int(data["block_num"][idx]), int(data["par_num"][idx]), int(data["line_num"][idx]))
        left = int(data["left"][idx])
        top = int(data["top"][idx])
        width = int(data["width"][idx])
        height = int(data["height"][idx])
        right = left + width
        bottom = top + height

        if key not in blocks:
            blocks[key] = {
                "block_id": f"{key[0]}-{key[1]}-{key[2]}",
                "file_path": str(path),
                "file_name": path.name,
                "bbox": [left, top, right, bottom],
                "text": text,
            }
            continue

        block = blocks[key]
        block["text"] = f"{block['text']} {text}"
        block["bbox"] = _union_bbox(block["bbox"], [left, top, right, bottom])

    if not blocks:
        # Return empty blocks instead of raising error
        return []

    return list(blocks.values())


def _extract_image_blocks_easyocr(path: Path) -> list[dict[str, Any]]:
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(str(path), detail=1)
    if not results:
        return []

    blocks: list[dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        bbox, text, _ = item
        blocks.append(
            {
                "block_id": f"easyocr-{idx}",
                "file_path": str(path),
                "file_name": path.name,
                "bbox": [
                    int(min(point[0] for point in bbox)),
                    int(min(point[1] for point in bbox)),
                    int(max(point[0] for point in bbox)),
                    int(max(point[1] for point in bbox)),
                ],
                "text": str(text).strip(),
            }
        )

    return blocks


def extract_image_blocks(path: Path) -> list[dict[str, Any]]:
    errors: list[str] = []
    if Image is None:
        raise RuntimeError("Pillow is required for image processing.")

    if pytesseract is not None:
        try:
            return _sort_image_blocks(_extract_image_blocks_pytesseract(path))
        except Exception as exc:
            errors.append(f"pytesseract: {exc}")

    if easyocr is not None:
        try:
            return _sort_image_blocks(_extract_image_blocks_easyocr(path))
        except Exception as exc:
            errors.append(f"easyocr: {exc}")

    # If no OCR engines work, create a single block with empty text
    # This allows images to still be processed even without OCR
    return [
        {
            "block_id": "fallback-1",
            "file_path": str(path),
            "file_name": path.name,
            "bbox": [0, 0, 100, 100],  # Default bbox
            "text": "",
        }
    ]


def _chunk_image_blocks(blocks: list[dict[str, Any]], chunk_size: int, chunk_index_offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Chunk image blocks with offset-based tracking for unique chunk indices.

    Args:
        blocks: List of OCR blocks to chunk
        chunk_size: Maximum chunk size in tokens
        chunk_index_offset: Starting index for chunk numbering (prevents duplicates)

    Returns:
        Tuple of (chunks list, final chunk index) for chaining multiple files
    """
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    chunk_counter = chunk_index_offset

    for block in blocks:
        if not block["text"].strip():
            continue

        tokens = max(1, _token_count(block["text"]))
        if current and (
            current_tokens + tokens > chunk_size
            or not _blocks_are_contiguous(current[-1], block)
        ):
            chunks.append(_make_image_chunk(current, chunk_index=chunk_counter))
            chunk_counter += 1
            current = []
            current_tokens = 0

        current.append(block)
        current_tokens += tokens

    if current:
        chunks.append(_make_image_chunk(current, chunk_index=chunk_counter))
        chunk_counter += 1

    return chunks, chunk_counter


def extract_document_chunks_resilient(
    path: Path,
    chunk_size: int,
    use_code_parsing: bool = True,
    ocr_engine: str = "tesseract",
    chunk_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], IngestionStatus, str, str, float, int]:
    """Extract document chunks with resilient processing and status tracking.

    Args:
        path: File path to process
        chunk_size: Chunk size in tokens
        use_code_parsing: Use AST-based parsing for code files (default: True)
        ocr_engine: OCR engine to use for images ('tesseract', 'easyocr', 'paddle')
        chunk_index_offset: Starting index for chunk numbering (prevents duplicates)

    Returns:
        chunks: List of document chunks
        status: Ingestion status
        error_message: Error message if any
        ocr_engine_used: Which extraction method was used
        processing_time: Time taken in seconds
        final_chunk_index: Final chunk index for chaining multiple files
    """
    import time
    start_time = time.time()

    records: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    ocr_engine_used = ""
    error_message = ""

    try:
        # Try code-aware parsing first if enabled
        if use_code_parsing and suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".go", ".rs"}:
            try:
                from local_archive_ai.code_indexing import CodeRepositoryIndexer
                if CodeRepositoryIndexer.is_code_file(path) and suffix == ".py":
                    chunks = CodeRepositoryIndexer.extract_code_chunks(path)
                    ocr_engine_used = "ast-parsing"
                    for idx, chunk in enumerate(chunks, start=chunk_index_offset):
                        metadata = CodeRepositoryIndexer.code_chunk_to_metadata(chunk)
                        metadata["chunk_index"] = idx
                        records.append(metadata)
                    processing_time = time.time() - start_time
                    final_chunk_index = chunk_index_offset + len(chunks)
                    if records:
                        return records, IngestionStatus.SUCCESS, "", ocr_engine_used, processing_time, final_chunk_index
            except Exception as e:
                # Fallback to text reading if code parsing fails
                pass

        if suffix == ".pdf":
            raw_text = _read_pdf(path)
            ocr_engine_used = "pdf-extraction"
        elif suffix in IMAGE_EXTENSIONS:
            blocks = extract_image_blocks(path)
            raw_chunks, final_chunk_index = _chunk_image_blocks(blocks, chunk_size, chunk_index_offset)
            ocr_engine_used = f"image-blocks-{ocr_engine}"
        else:
            # Text files
            try:
                raw_text = _sanitize_text(path.read_text(encoding="utf-8", errors="ignore"))
                ocr_engine_used = "text-read"
            except Exception as exc:
                raise RuntimeError(f"Unable to read file {path}: {exc}") from exc

        global_chunk_index = chunk_index_offset
        if suffix in IMAGE_EXTENSIONS:
            for chunk in raw_chunks:
                if not chunk["text"].strip():
                    continue
                records.append(chunk)
                global_chunk_index = max(global_chunk_index, chunk["chunk_index"] + 1)
        else:
            for chunk in _split_with_tokens(raw_text, chunk_size):
                chunk = _sanitize_text(chunk)
                if not chunk:
                    continue
                records.append(
                    {
                        "source_file": str(path),
                        "file_path": str(path),
                        "file_name": path.name,
                        "source_page": None,
                        "chunk_index": global_chunk_index,
                        "text": chunk,
                    }
                )
                global_chunk_index += 1

        processing_time = time.time() - start_time

        if not records:
            return records, IngestionStatus.WARNING, "No extractable text found", ocr_engine_used, processing_time, global_chunk_index

        return records, IngestionStatus.SUCCESS, "", ocr_engine_used, processing_time, global_chunk_index

    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        return records, IngestionStatus.ERROR, error_message, ocr_engine_used, processing_time, chunk_index_offset


def extract_document_chunks(path: Path, chunk_size: int, ocr_engine: str = "tesseract") -> list[dict[str, Any]]:
    """Legacy function for backward compatibility."""
    chunks, _, _, _, _, _ = extract_document_chunks_resilient(path, chunk_size, ocr_engine=ocr_engine)
    return chunks


def load_text_from_file(path: Path, ocr_engine: str = "tesseract") -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in IMAGE_EXTENSIONS:
        return _read_image_text(path, ocr_engine=ocr_engine)
    try:
        return _sanitize_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        raise RuntimeError(f"Unable to read file {path}: {exc}") from exc


def collect_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in __import__("os").walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES]
        root_path = Path(root)
        for name in filenames:
            path = root_path / name
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
    return sorted(files)


def summarize_indexable_content(folder: Path) -> dict[str, Any]:
    files = collect_files(folder)
    by_extension = Counter(p.suffix.lower() or "<none>" for p in files)
    top_ext = dict(sorted(by_extension.items(), key=lambda x: (-x[1], x[0]))[:12])
    return {"total_supported": len(files), "top_extensions": top_ext}


def collect_image_files(folder: Path) -> list[Path]:
    return [path for path in collect_files(folder) if path.suffix.lower() in IMAGE_EXTENSIONS]


def _normalize_metadata_payload(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("chunks", [])
    if not isinstance(raw, list):
        raise ValueError("Unsupported metadata format in metadata.json")
    return [item for item in raw if isinstance(item, dict)]


def load_index_metadata(store_path: str | Path) -> list[dict[str, Any]]:
    metadata_file = Path(store_path) / "metadata.json"
    if not metadata_file.exists():
        return []
    raw = json.loads(metadata_file.read_text(encoding="utf-8"))
    return _normalize_metadata_payload(raw)


def index_documents_resilient(
    folder_path: str,
    chunk_size: int,
    store_path: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    use_progress_bar: bool = True,
    file_filter: Callable[[Path], bool] | None = None,
    ocr_engine: str = "tesseract",
) -> IngestionReport:
    """Index documents with resilient ingestion, progress tracking, and detailed reporting."""
    import time
    start_time = time.time()

    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = collect_files(folder)
    if file_filter:
        files = [f for f in files if file_filter(f)]
    if not files:
        raise ValueError("No supported documents found in selected folder.")

    vector_store = LocalVectorStore(Path(store_path))
    existing = vector_store.load()
    existing_hashes = vector_store.file_hashes if existing else {}
    current_hashes: dict[str, str] = {}
    preserved_payload: list[dict[str, Any]] = []

    # Calculate file hashes concurrently to maximize IO throughput
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(16, len(files))) as executor:
        fs = {executor.submit(vector_store._hash_file, fp): fp for fp in files}
        for future in fs:
            fp = fs[future]
            try:
                current_hashes[str(fp)] = future.result()
            except Exception as e:
                log.warning(f"Failed to hash {fp}: {e}")
                current_hashes[str(fp)] = ""  # Force re-ingestion if hashing fails

    # Preserve unchanged files
    preserved_indices = []
    if existing and existing_hashes:
        for i, chunk in enumerate(vector_store.metadata):
            file_path = str(chunk.get("file_path", ""))
            if file_path in current_hashes and existing_hashes.get(file_path) == current_hashes[file_path]:
                preserved_payload.append(chunk)
                preserved_indices.append(i)

    # Process files with progress tracking
    new_payload: list[dict[str, Any]] = []
    file_results: list[FileIngestionResult] = []
    global_chunk_index = len(preserved_payload)  # Start after preserved chunks

    progress_iterator = tqdm(files, desc="Processing documents", unit="file") if use_progress_bar and tqdm else files

    for idx, file_path in enumerate(progress_iterator, start=1):
        file_hash = current_hashes[str(file_path)]

        # Check if file was already processed and unchanged
        if existing and existing_hashes.get(str(file_path)) == file_hash:
            file_results.append(FileIngestionResult(
                file_path=str(file_path),
                file_name=file_path.name,
                status=IngestionStatus.SKIPPED,
                file_hash=file_hash,
            ))
            if on_progress is not None:
                on_progress({
                    "current_file": file_path.name,
                    "processed_files": idx,
                    "total_files": len(files),
                    "chunk_count": len(preserved_payload) + len(new_payload),
                    "progress": idx / max(1, len(files)),
                    "status": "skipped",
                })
            continue

        # Process file with resilient extraction and offset-based chunk indexing
        chunks, status, error_msg, ocr_engine_used, processing_time, global_chunk_index = extract_document_chunks_resilient(
            file_path,
            chunk_size,
            ocr_engine=ocr_engine,
            chunk_index_offset=global_chunk_index,
        )

        file_result = FileIngestionResult(
            file_path=str(file_path),
            file_name=file_path.name,
            status=status,
            chunks_count=len(chunks),
            error_message=error_msg,
            ocr_engine_used=ocr_engine_used,
            processing_time_seconds=processing_time,
            file_hash=file_hash,
        )
        file_results.append(file_result)

        if status in [IngestionStatus.SUCCESS, IngestionStatus.WARNING]:
            new_payload.extend(chunks)

        if on_progress is not None:
            on_progress({
                "current_file": file_path.name,
                "processed_files": idx,
                "total_files": len(files),
                "chunk_count": len(preserved_payload) + len(new_payload),
                "progress": idx / max(1, len(files)),
                "status": status.value,
                "error": error_msg,
            })

    # Combine payloads
    payload = preserved_payload + new_payload
    if not payload:
        raise ValueError("No extractable text chunks found in supported documents.")

    # Embed new texts
    if new_payload:
        texts = [item["text"] for item in new_payload]
        new_vectors = EmbeddingService().embed(texts, batch_size=12)
    else:
        new_vectors = np.empty((0, EmbeddingService().model.get_sentence_embedding_dimension() if EmbeddingService().model else 384), dtype=np.float32)

    # Reconstruct preserved vectors
    if preserved_indices and vector_store.index is not None:
        kept_vectors = np.array(
            [vector_store.index.reconstruct(int(i)) for i in preserved_indices], dtype=np.float32
        )
        if new_vectors.size > 0:
            vectors = np.vstack([kept_vectors, new_vectors])
        else:
            vectors = kept_vectors
    else:
        vectors = new_vectors

    # Build vector store
    vector_store = LocalVectorStore(Path(store_path))
    vector_store.build(vectors, payload, file_hashes=current_hashes)

    # Calculate statistics
    total_processing_time = time.time() - start_time
    successful_files = sum(1 for r in file_results if r.status == IngestionStatus.SUCCESS)
    warning_files = sum(1 for r in file_results if r.status == IngestionStatus.WARNING)
    error_files = sum(1 for r in file_results if r.status == IngestionStatus.ERROR)
    skipped_files = sum(1 for r in file_results if r.status == IngestionStatus.SKIPPED)
    processed_files = successful_files + warning_files + error_files
    success_rate = (successful_files / processed_files * 100) if processed_files > 0 else 0.0

    # Create report
    report = IngestionReport(
        total_files=len(files),
        processed_files=processed_files,
        successful_files=successful_files,
        warning_files=warning_files,
        error_files=error_files,
        skipped_files=skipped_files,
        total_chunks=len(payload),
        completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        processing_time_seconds=total_processing_time,
        file_results=file_results,
        success_rate=success_rate,
    )

    # Save detailed report
    report_path = Path(store_path) / "ingestion_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    # Also save legacy index_status.json for compatibility
    legacy_report = {
        "file_count": len(files),
        "chunk_count": len(payload),
        "completed_at": report.completed_at,
        "index_path": str(Path(store_path).resolve()),
        "faiss_enabled": vector_store.faiss_enabled,
        "success_rate": success_rate,
        "error_files": error_files,
    }
    with (Path(store_path) / "index_status.json").open("w", encoding="utf-8") as fh:
        json.dump(legacy_report, fh, indent=2)

    return report


def index_documents(
    folder_path: str,
    chunk_size: int,
    store_path: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    file_filter: Callable[[Path], bool] | None = None,
    ocr_engine: str = "tesseract",
) -> IndexReport:
    """Legacy index function with resilient processing internally."""
    # Use resilient indexing but convert to legacy IndexReport format
    resilient_report = index_documents_resilient(
        folder_path=folder_path,
        chunk_size=chunk_size,
        store_path=store_path,
        on_progress=on_progress,
        use_progress_bar=False,  # Let the UI handle progress
        file_filter=file_filter,
        ocr_engine=ocr_engine,
    )

    # Convert IngestionReport to IndexReport for backward compatibility
    return IndexReport(
        file_count=resilient_report.total_files,
        chunk_count=resilient_report.total_chunks,
        completed_at=resilient_report.completed_at,
        index_path=str(Path(store_path).resolve()),
        faiss_enabled=True,  # Assume FAISS is enabled
    )


def get_ingestion_report(store_path: str) -> IngestionReport | None:
    """Load the detailed ingestion report if available."""
    report_path = Path(store_path) / "ingestion_report.json"
    if not report_path.exists():
        return None

    try:
        with report_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Convert dict back to IngestionReport
        file_results = []
        for fr_data in data["file_results"]:
            file_results.append(FileIngestionResult(
                file_path=fr_data["file_path"],
                file_name=fr_data["file_name"],
                status=IngestionStatus(fr_data["status"]),
                chunks_count=fr_data["chunks_count"],
                error_message=fr_data["error_message"],
                ocr_engine_used=fr_data["ocr_engine_used"],
                processing_time_seconds=fr_data["processing_time_seconds"],
                file_hash=fr_data["file_hash"],
            ))

        return IngestionReport(
            total_files=data["total_files"],
            processed_files=data["processed_files"],
            successful_files=data["successful_files"],
            warning_files=data["warning_files"],
            error_files=data["error_files"],
            skipped_files=data["skipped_files"],
            total_chunks=data["total_chunks"],
            completed_at=data["completed_at"],
            processing_time_seconds=data["processing_time_seconds"],
            file_results=file_results,
            success_rate=data["success_rate"],
        )
    except Exception:
        return None

def get_index_status(store_path: str) -> dict[str, Any]:
    store = LocalVectorStore(Path(store_path))
    loaded = store.load()
    status_file = Path(store_path) / "index_status.json"
    meta: dict[str, Any] = {}
    if status_file.exists():
        meta = json.loads(status_file.read_text(encoding="utf-8"))
    return {
        "exists": bool(loaded and store.ready()),
        "chunk_count": int(meta.get("chunk_count", len(store.metadata))),
        "file_count": int(meta.get("file_count", 0)),
        "completed_at": str(meta.get("completed_at", "N/A")),
        "faiss_enabled": store.faiss_enabled,
    }


def _detect_token_limit(model_name: str) -> int:
    if "8k" in model_name.lower() or "8192" in model_name:
        return 8192
    if "32k" in model_name.lower() or "32768" in model_name:
        return 32768
    return 4096


def _token_count(text: str) -> int:
    if tiktoken is None:
        return len(text.split())
    try:
        encoder = tiktoken.encoding_for_model("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(text))
        except Exception:
            return len(text.split())


def _compose_prompt(
    question: str,
    hits: list[SearchHit],
    model_name: str,
    max_context_tokens: int = 8192,
) -> tuple[str, int]:
    """Compose prompt with dynamic chunk selection (Problem N) and truncation (Problem R).

    Returns (prompt_text, chunks_used).
    """
    # Calculate tokens used by the fixed parts of the prompt template
    template_base = (
        "You are a local offline assistant for document analysis.\n"
        "Use only the provided context to answer the question. "
        "If the answer is not contained in the context, say so clearly.\n\n"
        "Context:\n\nQuestion: \n\nAnswer:"
    )
    template_tokens = _token_count(template_base) + _token_count(question)
    
    # Reserve budget for context, ensuring we stay within model limits
    context_budget = max(0, max_context_tokens - template_tokens - 100)  # 100 token buffer
    
    context_entries: list[str] = []
    current_tokens = 0
    truncated = False

    for idx, hit in enumerate(hits):
        chunk_text = _sanitize_text(hit.text, max_length=1500)
        entry = (
            f"[{idx + 1}] {hit.metadata.get('file_name', 'unknown')}"
            f" (page={hit.metadata.get('source_page')}, chunk={hit.metadata.get('chunk_index')})"
            f" :: {chunk_text}"
        )
        entry_tokens = _token_count(entry)
        if current_tokens + entry_tokens > context_budget:
            truncated = True
            break
        context_entries.append(entry)
        current_tokens += entry_tokens

    if truncated:
        log.warning("Context truncated at %d chunks (token budget %d)", len(context_entries), context_budget)

    context = "\n\n".join(context_entries)
    prompt = (
        "You are a local offline assistant for document analysis.\n"
        "Answer the question using only the provided context.\n"
        "If the context is insufficient or irrelevant, say that the indexed local documents "
        "do not contain enough information to answer reliably.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    # Final check: ensure total prompt doesn't exceed model limit
    total_tokens = _token_count(prompt)
    if total_tokens > max_context_tokens:
        log.error("Prompt exceeds token limit: %d > %d", total_tokens, max_context_tokens)
    
    return prompt, len(context_entries)


def prewarm_ollama(endpoint: str = "http://127.0.0.1:11434", model_name: str = "llama3.2:1b", api_key: str = "") -> bool:
    """Pre-warm Ollama by sending a dummy prompt to load the model into memory (Problem O)."""
    session = _get_ollama_session()
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = session.post(
            f"{endpoint}/api/generate",
            json={"model": model_name, "prompt": "hi", "stream": False, "keep_alive": -1},
            headers=headers,
            timeout=120,
        )
        ok = resp.status_code == 200
        if ok:
            log.info("Ollama pre-warmed with model %s", model_name)
        return ok
    except Exception:
        log.warning("Ollama pre-warm failed for model %s", model_name)
        return False


def _generate_with_ollama(
    prompt: str,
    model_name: str,
    endpoint: str = "http://127.0.0.1:11434",
    api_key: str = "",
    temperature: float = 0.7,
) -> str:
    """Generate response from Ollama with connection pooling."""
    session = _get_ollama_session()
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = session.post(
            f"{endpoint}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": temperature},
            },
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        response = str(data.get("response", "")).strip()
        if not response:
            return "Model generated an empty response. This may indicate the model is still loading or encountered an issue."
        return response
    except requests.exceptions.Timeout:
        return "Request timed out. The model may be taking too long to load or generate a response. Try again in a few moments."
    except requests.exceptions.ConnectionError:
        return f"Connection failed. Please ensure Ollama is running on {endpoint}."
    except requests.exceptions.HTTPError:
        try:
            status = resp.status_code
        except Exception:
            status = 0
        if status == 404:
            return f"Model '{model_name}' not found. Please pull the model first: ollama pull {model_name}"
        return f"HTTP error from Ollama (status {status})."
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_with_ollama_stream(
    prompt: str,
    model_name: str,
    endpoint: str = "http://127.0.0.1:11434",
    api_key: str = "",
    temperature: float = 0.7,
):
    """Streaming generator for Ollama responses (Problem O: streaming with st.write_stream)."""
    session = _get_ollama_session()
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = session.post(
            f"{endpoint}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "keep_alive": -1,
                "options": {"temperature": temperature},
            },
            headers=headers,
            timeout=300,
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
        yield f"\n\n[Streaming error: {e}]"


def answer_query(
    query: str,
    top_k: int,
    model_name: str,
    store_path: str,
    debug: bool = False,
    ollama_endpoint: str = "http://127.0.0.1:11434",
    ollama_api_key: str = "",
    retrieval_mode: str = "vector",
    bm25_weight: float = 0.4,
    rerank_top_n: int = 5,
    confidence_threshold: float = 0.65,
    temperature: float = 0.7,
    max_context_tokens: int = 8192,
) -> dict[str, Any]:
    t0 = time.time()
    store = LocalVectorStore(Path(store_path))
    if not store.load() or not store.ready():
        raise RuntimeError("FAISS index is not loaded. Please index files first.")

    embedder = EmbeddingService()
    retriever = HybridRetriever(store, embedder)

    # Always build BM25 index when rank_bm25 is available so vector-mode can fallback gracefully.
    if BM25Okapi is not None:
        retriever.build_bm25_index(store.metadata)

    # Generate query embedding for debug payload
    query_vec = embedder.embed([query])[0]

    if retrieval_mode == "vector":
        search_results = retriever.search_vector(query, top_k)
    elif retrieval_mode == "hybrid":
        search_results = retriever.search_hybrid(query, top_k, bm25_weight)
    elif retrieval_mode == "hybrid+rerank":
        candidates = retriever.search_hybrid(query, top_k * 2, bm25_weight)
        search_results = retriever.rerank_with_cross_encoder(query, candidates, rerank_top_n)
    else:
        search_results = retriever.search_vector(query, top_k)

    hits: list[SearchHit] = []
    for result in search_results:
        hits.append(SearchHit(
            text=result.text,
            score=result.score,
            metadata=result.metadata,
        ))

    if not hits:
        raise RuntimeError("No relevant content found in the index.")

    # Confidence scoring (Problem P)
    max_similarity = max(h.score for h in hits) if hits else 0.0
    low_confidence = max_similarity < confidence_threshold

    prompt, chunks_used = _compose_prompt(query, hits, model_name, max_context_tokens=max_context_tokens)
    answer = ""
    try:
        if not check_ollama_status(ollama_endpoint, ollama_api_key):
            answer = (
                f"Ollama is not running on {ollama_endpoint}.\n\n"
                "Your local retrieval succeeded, but the final response cannot be generated until Ollama is available."
            )
        else:
            answer = _generate_with_ollama(prompt, model_name, ollama_endpoint, ollama_api_key, temperature=temperature)
    except Exception as exc:
        answer = (
            "Local model call failed. Verify Ollama is running and model is available.\n\n"
            f"Error: {exc}"
        )

    # Prepend low-confidence warning (Problem P)
    if low_confidence and answer and not answer.startswith("Ollama") and not answer.startswith("Connection"):
        answer = (
            "*(Note: The retrieved documents had low relevance to this query. Max similarity: "
            f"{max_similarity:.3f})*\n\n{answer}"
        )

    citations = []
    for hit in hits:
        file_path = hit.metadata.get("file_path", "")
        citations.append(
            {
                "file_path": file_path,
                "file_name": hit.metadata.get("file_name", "unknown"),
                "chunk_text": hit.text,
                "score": hit.score,
                "source_page": hit.metadata.get("source_page"),
                "chunk_index": hit.metadata.get("chunk_index"),
                "open_uri": Path(file_path).resolve().as_uri() if file_path else "",
            }
        )

    duration_ms = int((time.time() - t0) * 1000)
    log.info("Query answered in %dms (mode=%s, chunks=%d, confidence=%.3f)",
             duration_ms, retrieval_mode, chunks_used, max_similarity,
             extra={"query": query, "duration_ms": duration_ms})

    payload: dict[str, Any] = {
        "answer_markdown": answer,
        "citations": citations,
        "max_similarity": max_similarity,
        "low_confidence": low_confidence,
        "duration_ms": duration_ms,
        "debug_payload": {
            "pipeline_checks": {
                "index_loaded": True,
                "retrieval_succeeded": bool(hits),
                "context_injected": bool(prompt and citations),
                "generation_attempted": True,
                "generation_succeeded": bool(answer and not answer.startswith("Error generating response")),
            },
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
            "hybrid_settings": {
                "retrieval_mode": retrieval_mode,
                "bm25_weight": bm25_weight,
                "rerank_top_n": rerank_top_n,
                "reranker_active": retrieval_mode == "hybrid+rerank" and load_reranker_model() is not None,
            },
        },
    }
    if not debug:
        payload["debug_payload"] = {}
    return payload


def search_index(
    query: str,
    top_k: int,
    store_path: str,
    retrieval_mode: str = "vector",
    bm25_weight: float = 0.4,
) -> list[SearchHit]:
    store = LocalVectorStore(Path(store_path))
    if not store.load() or not store.ready():
        raise RuntimeError("FAISS index is not loaded. Please index files first.")

    embedder = EmbeddingService()
    retriever = HybridRetriever(store, embedder)
    if BM25Okapi is not None:
        retriever.build_bm25_index(store.metadata)

    if retrieval_mode == "vector":
        return retriever.search_vector(query, top_k)
    if retrieval_mode == "hybrid":
        return retriever.search_hybrid(query, top_k, bm25_weight)
    if retrieval_mode == "hybrid+rerank":
        candidates = retriever.search_hybrid(query, top_k * 2, bm25_weight)
        return retriever.rerank_with_cross_encoder(query, candidates, top_k)
    return retriever.search_vector(query, top_k)


def index_image_documents(
    folder_path: str,
    chunk_size: int,
    store_path: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    ocr_engine: str = "tesseract",
) -> IndexReport:
    folder = Path(folder_path).expanduser().resolve()
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError("No supported image files found in folder.")

    file_results: list[FileIngestionResult] = []
    payload: list[dict[str, Any]] = []
    current_chunk_index = 0
    start_time = time.time()

    for idx, path in enumerate(files):
        chunks, status, error_msg, ocr_engine_used, processing_time, current_chunk_index = extract_document_chunks_resilient(
            path,
            chunk_size,
            use_code_parsing=False,
            ocr_engine=ocr_engine,
            chunk_index_offset=current_chunk_index,
        )
        file_results.append(
            FileIngestionResult(
                file_path=str(path),
                file_name=path.name,
                status=status,
                chunks_count=len(chunks),
                error_message=error_msg,
                ocr_engine_used=ocr_engine_used,
                processing_time_seconds=processing_time,
            )
        )
        if status in {IngestionStatus.SUCCESS, IngestionStatus.WARNING}:
            payload.extend(chunks)

        if on_progress is not None:
            on_progress({
                "current_file": path.name,
                "processed_files": idx + 1,
                "total_files": len(files),
                "chunk_count": len(payload),
                "status": status.value,
                "error": error_msg,
            })

    if not payload:
        raise ValueError("No OCR text could be extracted from the provided images.")

    embedder = EmbeddingService()
    texts = [item["text"] for item in payload]
    vectors = embedder.embed(texts, batch_size=12)

    vector_store = LocalVectorStore(Path(store_path))
    vector_store.build(vectors, payload)

    total_time = time.time() - start_time
    successful_files = sum(1 for result in file_results if result.status == IngestionStatus.SUCCESS)
    warning_files = sum(1 for result in file_results if result.status == IngestionStatus.WARNING)
    error_files = sum(1 for result in file_results if result.status == IngestionStatus.ERROR)
    report = IngestionReport(
        total_files=len(files),
        processed_files=successful_files + warning_files + error_files,
        successful_files=successful_files,
        warning_files=warning_files,
        error_files=error_files,
        skipped_files=0,
        total_chunks=len(payload),
        completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        processing_time_seconds=total_time,
        file_results=file_results,
        success_rate=((successful_files / max(successful_files + warning_files + error_files, 1)) * 100.0),
    )
    legacy_report = {
        "file_count": len(files),
        "chunk_count": len(payload),
        "completed_at": report.completed_at,
        "index_path": str(Path(store_path).resolve()),
        "faiss_enabled": vector_store.faiss_enabled,
        "success_rate": report.success_rate,
        "error_files": error_files,
    }
    with (Path(store_path) / "index_status.json").open("w", encoding="utf-8") as fh:
        json.dump(legacy_report, fh, indent=2)

    with (Path(store_path) / "ingestion_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    return IndexReport(
        file_count=len(files), chunk_count=len(payload), completed_at=report.completed_at, index_path=str(Path(store_path).resolve()), faiss_enabled=vector_store.faiss_enabled
    )


def search_image_chunks(
    query: str,
    top_k: int,
    store_path: str,
    retrieval_mode: str = "vector",
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    store = LocalVectorStore(Path(store_path))
    if not store.load() or not store.ready():
        raise RuntimeError("Image index is not loaded. Please index images first.")

    retriever = HybridRetriever(store, EmbeddingService())
    if BM25Okapi is not None:
        retriever.build_bm25_index(store.metadata)

    if retrieval_mode == "vector":
        hits = retriever.search_vector(query, top_k)
    elif retrieval_mode == "hybrid":
        hits = retriever.search_hybrid(query, top_k, bm25_weight)
    elif retrieval_mode == "hybrid+rerank":
        candidates = retriever.search_hybrid(query, top_k * 2, bm25_weight)
        hits = retriever.rerank_with_cross_encoder(query, candidates, top_k)
    else:
        hits = retriever.search_vector(query, top_k)

    results = []
    for hit in hits:
        m = hit.metadata
        results.append({
            "file_path": m.get("file_path", ""),
            "file_name": m.get("file_name", "unknown"),
            "score": hit.score,
            "chunk_index": m.get("chunk_index"),
            "text": m.get("text", ""),
            "bbox": m.get("bbox"),
            "bounding_boxes": m.get("bounding_boxes", []),
            "block_ids": m.get("block_ids", []),
        })
    return results


def check_ollama_status(endpoint: str = "http://127.0.0.1:11434", api_key: str = "") -> bool:
    """Check if Ollama is reachable (Problem V)."""
    try:
        session = _get_ollama_session()
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = session.get(f"{endpoint}/api/tags", headers=headers, timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def check_ollama_model(endpoint: str = "http://127.0.0.1:11434", api_key: str = "", model_name: str = "") -> bool:
    try:
        session = _get_ollama_session()
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = session.get(f"{endpoint}/api/tags", headers=headers, timeout=4)
        if r.status_code != 200:
            return False
        payload = r.json()
        if isinstance(payload, dict) and "models" in payload:
            return any(str(model_name).lower() in str(entry).lower() for entry in payload["models"])
        if isinstance(payload, list):
            return any(str(model_name).lower() in str(entry).lower() for entry in payload)
        return False
    except Exception:
        return False


def get_ollama_status_message(endpoint: str = "http://127.0.0.1:11434", api_key: str = "") -> dict[str, Any]:
    """Return a detailed Ollama status check with actionable messages (Problem V)."""
    result: dict[str, Any] = {"running": False, "message": "", "download_url": "https://ollama.com/download"}
    try:
        session = _get_ollama_session()
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = session.get(f"{endpoint}/api/tags", headers=headers, timeout=4)
        if r.status_code == 200:
            result["running"] = True
            result["message"] = "Ollama is running and reachable."
            models = r.json().get("models", [])
            result["models"] = [m.get("name", str(m)) for m in models] if isinstance(models, list) else []
        else:
            result["message"] = f"Ollama responded with status {r.status_code}. Ensure it is properly configured."
    except requests.exceptions.ConnectionError:
        result["message"] = (
            f"Ollama is not running on {endpoint}. "
            "Please start it with `ollama serve` in a terminal, or download from https://ollama.com/download"
        )
    except Exception as e:
        result["message"] = f"Cannot reach Ollama: {e}"
    return result


def get_autocomplete_suggestions(store_path: str, prefix: str, max_results: int = 8) -> list[str]:
    """Return autocomplete suggestions from indexed document titles and metadata (Problem BB)."""
    suggestions: list[str] = []
    metadata_file = Path(store_path) / "metadata.json"
    if not metadata_file.exists():
        return suggestions
    try:
        raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        if not isinstance(chunks, list):
            return suggestions
        seen: set[str] = set()
        needle = prefix.lower()
        for item in chunks:
            name = str(item.get("file_name", ""))
            if name and needle in name.lower() and name not in seen:
                seen.add(name)
                suggestions.append(name)
            if len(suggestions) >= max_results:
                break
    except Exception:
        pass
    return suggestions


def _run_command(command: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=4)
        return True, out.strip()
    except Exception:
        return False, ""


def _gpu_telemetry_from_nvidia_smi() -> dict[str, str] | None:
    ok, out = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not ok or not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None
    used, total, util, temp = parts[:4]
    return {
        "mode": "GPU Accelerated",
        "vram": f"{used} MiB / {total} MiB",
        "gpu_util": f"{util}%",
        "gpu_temp": f"{temp} C",
    }


def _gpu_telemetry_from_gpustat() -> dict[str, str] | None:
    ok, out = _run_command(["gpustat", "--json"])
    if not ok or not out:
        return None
    try:
        payload = json.loads(out)
        gpus = payload.get("gpus", [])
        if not gpus:
            return None
        gpu0 = gpus[0]
        used = gpu0.get("memory.used")
        total = gpu0.get("memory.total")
        util = gpu0.get("utilization.gpu", "N/A")
        temp = gpu0.get("temperature.gpu", "N/A")
        return {
            "mode": "GPU Accelerated",
            "vram": f"{used} MiB / {total} MiB",
            "gpu_util": f"{util}%",
            "gpu_temp": f"{temp} C",
        }
    except Exception:
        return None


def runtime_mode() -> dict[str, str]:
    nvidia = _gpu_telemetry_from_nvidia_smi()
    if nvidia is not None:
        return nvidia

    gpustat_payload = _gpu_telemetry_from_gpustat()
    if gpustat_payload is not None:
        return gpustat_payload

    return {
        "mode": "CPU Fallback",
        "vram": "N/A",
        "gpu_util": "N/A",
        "gpu_temp": "N/A",
    }


def system_checks(store_path: str, model_name: str = "", endpoint: str = "http://127.0.0.1:11434", api_key: str = "") -> dict[str, Any]:
    tesseract_installed = False
    if pytesseract is not None:
        try:
            _configure_tesseract_cmd()
            _ = pytesseract.get_tesseract_version()
            tesseract_installed = True
        except Exception:
            tesseract_installed = False

    plotly_available = False
    try:
        import plotly  # type: ignore # noqa: F401

        plotly_available = True
    except Exception:
        plotly_available = False

    faiss_loaded = False
    faiss_message = "Not loaded"
    try:
        store = LocalVectorStore(Path(store_path))
        faiss_loaded = store.load() and store.ready()
        faiss_message = "Loaded" if faiss_loaded else "Index missing/not ready"
    except Exception as exc:
        faiss_message = f"Error: {exc}"

    ollama_reachable = check_ollama_status(endpoint=endpoint, api_key=api_key)
    ollama_model_available = False
    if ollama_reachable and model_name:
        ollama_model_available = check_ollama_model(endpoint=endpoint, api_key=api_key, model_name=model_name)

    return {
        "tesseract_installed": tesseract_installed,
        "plotly_available": plotly_available,
        "faiss_loaded": faiss_loaded,
        "faiss_message": faiss_message,
        "ollama_reachable": ollama_reachable,
        "ollama_model_available": ollama_model_available,
    }

_DIAGNOSTICS_CACHE: dict[str, dict[str, Any]] = {}

def vector_diagnostics(store_path: str, sample_size: int = 500) -> dict[str, Any]:
    global _DIAGNOSTICS_CACHE
    store = LocalVectorStore(Path(store_path))
    if not store.load() or not store.ready():
        return {
            "ready": False,
            "message": "FAISS index is not ready.",
            "vector_count": 0,
            "dim": 0,
            "points": [],
        }

    current_count = int(store.index.ntotal if store.index is not None else len(store.metadata))
    cache_key = f"{store_path}_{current_count}_{sample_size}"
    if cache_key in _DIAGNOSTICS_CACHE:
        return _DIAGNOSTICS_CACHE[cache_key]

    vectors = store.sample_vectors(sample_size=sample_size)
    if vectors.size == 0:
        return {
            "ready": True,
            "message": "No vectors available in index.",
            "vector_count": 0,
            "dim": 0,
            "points": [],
        }

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # 2D PCA via SVD for lightweight local diagnostics.
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    proj = u[:, :2] * s[:2]

    points = []
    metadata = store.metadata
    step = max(1, len(metadata) // len(proj))
    for i, xy in enumerate(proj):
        meta_idx = min(i * step, len(metadata) - 1)
        m = metadata[meta_idx]
        points.append(
            {
                "x": float(xy[0]) if len(xy) > 0 else 0.0,
                "y": float(xy[1]) if len(xy) > 1 else 0.0,
                "file_name": str(m.get("file_name", "unknown")),
                "source_page": m.get("source_page"),
                "chunk_index": m.get("chunk_index"),
            }
        )

    result = {
        "ready": True,
        "message": "Vector diagnostics available.",
        "vector_count": current_count,
        "dim": int(vectors.shape[1]),
        "points": points,
    }
    
    if len(_DIAGNOSTICS_CACHE) > 5:
        _DIAGNOSTICS_CACHE.clear()
    _DIAGNOSTICS_CACHE[cache_key] = result
    
    return result
