from __future__ import annotations

import json
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import re
import requests
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
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_ROOT_DIR = Path(__file__).resolve().parent.parent
_MODEL_CACHE_DIR = _ROOT_DIR / "data" / "models"
_EMBED_MODEL_LOCK = threading.Lock()
_RERANK_MODEL_LOCK = threading.Lock()


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
                    self.__class__._model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        cache_folder=str(self._cache_dir),
                        local_files_only=True,
                        device="cpu",
                    )
                except Exception:
                    self.__class__._model = None

    @property
    def model(self) -> Any:
        self._load_model()
        return self.__class__._model

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
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


def _build_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter | None:
    if RecursiveCharacterTextSplitter is None or tiktoken is None:
        return None
    # cl100k_base aligns with modern token budgeting and avoids word-count drift.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=max(60, int(chunk_size * 0.15)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


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


def _read_image_text_resilient(path: Path) -> str:
    """Extract text from image using multi-engine OCR chain: pytesseract → easyocr → fail."""
    errors = []

    # Try pytesseract first
    if Image is not None and pytesseract is not None:
        _configure_tesseract_cmd()
        try:
            with Image.open(path) as img:
                text = pytesseract.image_to_string(img, timeout=10)  # Add timeout
                if text.strip():
                    return _sanitize_text(text)
        except Exception as e:
            errors.append(f"pytesseract: {str(e)}")

    # Fallback to easyocr
    if easyocr is not None:
        try:
            # Initialize reader with timeout protection
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            result = reader.readtext(str(path), detail=0)
            text = "\n".join(result)
            if text.strip():
                return _sanitize_text(text)
        except Exception as e:
            errors.append(f"easyocr: {str(e)}")

    # If no OCR available or all failed, return empty string with warning
    if not errors:
        errors.append("No OCR engines available")
    
    # For images without OCR, we still create chunks but with empty text
    # This allows the system to work even without OCR
    return ""


def _read_image_text(path: Path, ocr_engine: str = "tesseract") -> str:
    """Legacy function for backward compatibility."""
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


def _make_image_chunk(blocks: list[dict[str, Any]]) -> dict[str, Any]:
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
        "chunk_index": 0,
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


def _chunk_image_blocks(blocks: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for block in blocks:
        if not block["text"].strip():
            continue

        tokens = max(1, _token_count(block["text"]))
        if current and (
            current_tokens + tokens > chunk_size
            or not _blocks_are_contiguous(current[-1], block)
        ):
            chunks.append(_make_image_chunk(current))
            current = []
            current_tokens = 0

        current.append(block)
        current_tokens += tokens

    if current:
        chunks.append(_make_image_chunk(current))

    return chunks


def extract_document_chunks_resilient(path: Path, chunk_size: int, use_code_parsing: bool = True) -> tuple[list[dict[str, Any]], IngestionStatus, str, str, float]:
    """Extract document chunks with resilient processing and status tracking.
    
    Args:
        path: File path to process
        chunk_size: Chunk size in tokens
        use_code_parsing: Use AST-based parsing for code files (default: True)
    
    Returns:
        chunks: List of document chunks
        status: Ingestion status
        error_message: Error message if any
        ocr_engine_used: Which extraction method was used
        processing_time: Time taken in seconds
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
                    for idx, chunk in enumerate(chunks, start=1):
                        metadata = CodeRepositoryIndexer.code_chunk_to_metadata(chunk)
                        metadata["chunk_index"] = idx
                        records.append(metadata)
                    processing_time = time.time() - start_time
                    if records:
                        return records, IngestionStatus.SUCCESS, "", ocr_engine_used, processing_time
            except Exception as e:
                # Fallback to text reading if code parsing fails
                pass
        
        if suffix == ".pdf":
            raw_text = _read_pdf(path)
            ocr_engine_used = "pdf-extraction"
        elif suffix in IMAGE_EXTENSIONS:
            blocks = extract_image_blocks(path)
            raw_chunks = _chunk_image_blocks(blocks, chunk_size)
            ocr_engine_used = "image-blocks"
        else:
            # Text files
            try:
                raw_text = _sanitize_text(path.read_text(encoding="utf-8", errors="ignore"))
                ocr_engine_used = "text-read"
            except Exception as exc:
                raise RuntimeError(f"Unable to read file {path}: {exc}") from exc

        global_chunk_index = 0
        if suffix in IMAGE_EXTENSIONS:
            for chunk in raw_chunks:
                if not chunk["text"].strip():
                    continue
                global_chunk_index += 1
                chunk["chunk_index"] = global_chunk_index
                records.append(chunk)
        else:
            for chunk in _split_with_tokens(raw_text, chunk_size):
                chunk = _sanitize_text(chunk)
                if not chunk:
                    continue
                global_chunk_index += 1
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

        processing_time = time.time() - start_time

        if not records:
            return records, IngestionStatus.WARNING, "No extractable text found", ocr_engine_used, processing_time

        return records, IngestionStatus.SUCCESS, "", ocr_engine_used, processing_time

    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        return records, IngestionStatus.ERROR, error_message, ocr_engine_used, processing_time


def extract_document_chunks(path: Path, chunk_size: int, ocr_engine: str = "tesseract") -> list[dict[str, Any]]:
    """Legacy function for backward compatibility."""
    chunks, _, _, _, _ = extract_document_chunks_resilient(path, chunk_size)
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

    # Calculate file hashes
    for file_path in files:
        file_hash = vector_store._hash_file(file_path)
        current_hashes[str(file_path)] = file_hash

    # Preserve unchanged files
    if existing and existing_hashes:
        for chunk in vector_store.metadata:
            file_path = str(chunk.get("file_path", ""))
            if file_path in current_hashes and existing_hashes.get(file_path) == current_hashes[file_path]:
                preserved_payload.append(chunk)

    # Process files with progress tracking
    new_payload: list[dict[str, Any]] = []
    file_results: list[FileIngestionResult] = []

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

        # Process file with resilient extraction
        chunks, status, error_msg, ocr_engine, processing_time = extract_document_chunks_resilient(file_path, chunk_size)

        file_result = FileIngestionResult(
            file_path=str(file_path),
            file_name=file_path.name,
            status=status,
            chunks_count=len(chunks),
            error_message=error_msg,
            ocr_engine_used=ocr_engine,
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

    # Generate embeddings in small batches to keep memory stable on large indexes.
    texts = [item["text"] for item in payload]
    vectors = EmbeddingService().embed(texts, batch_size=12)

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


def _compose_prompt(question: str, hits: list[SearchHit], model_name: str) -> str:
    token_limit = _detect_token_limit(model_name)
    max_context_tokens = int(token_limit * 0.8)
    context_entries: list[str] = []
    current_tokens = 0

    for idx, hit in enumerate(hits):
        chunk_text = _sanitize_text(hit.text, max_length=1500)
        entry = (
            f"[{idx + 1}] {hit.metadata.get('file_name', 'unknown')}"
            f" (page={hit.metadata.get('source_page')}, chunk={hit.metadata.get('chunk_index')})"
            f" :: {chunk_text}"
        )
        entry_tokens = _token_count(entry)
        if current_tokens + entry_tokens > max_context_tokens:
            break
        context_entries.append(entry)
        current_tokens += entry_tokens

    context = "\n\n".join(context_entries)
    prompt = (
        "You are a local offline assistant for document analysis.\n"
        "Use only the provided context to answer the question. If the answer is not contained in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return prompt


def _generate_with_ollama(prompt: str, model_name: str, endpoint: str = "http://127.0.0.1:11434", api_key: str = "") -> str:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        resp = requests.post(
            f"{endpoint}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            headers=headers,
            timeout=300,  # Increased timeout to 5 minutes for model loading
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
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return f"Model '{model_name}' not found. Please pull the model first: ollama pull {model_name}"
        return f"HTTP error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Error generating response: {str(e)}"


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
) -> dict[str, Any]:
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

    prompt = _compose_prompt(query, hits, model_name)
    answer = ""
    try:
        if not check_ollama_status(ollama_endpoint, ollama_api_key):
            answer = (
                f"Ollama is not running on {ollama_endpoint}.\n\n"
                "Your local retrieval succeeded, but the final response cannot be generated until Ollama is available."
            )
        else:
            answer = _generate_with_ollama(prompt, model_name, ollama_endpoint, ollama_api_key)
    except Exception as exc:
        answer = (
            "Local model call failed. Verify Ollama is running and model is available.\n\n"
            f"Error: {exc}"
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

    payload = {
        "answer_markdown": answer,
        "citations": citations,
        "debug_payload": {
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
) -> IndexReport:
    return index_documents(
        folder_path=folder_path,
        chunk_size=chunk_size,
        store_path=store_path,
        on_progress=on_progress,
        file_filter=lambda path: path.suffix.lower() in IMAGE_EXTENSIONS,
    )


def search_image_chunks(
    query: str,
    top_k: int,
    store_path: str,
    retrieval_mode: str = "vector",
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    hits = search_index(
        query=query,
        top_k=top_k,
        store_path=store_path,
        retrieval_mode=retrieval_mode,
        bm25_weight=bm25_weight,
    )
    payload: list[dict[str, Any]] = []
    for hit in hits:
        payload.append(
            {
                "file_path": hit.metadata.get("file_path", ""),
                "file_name": hit.metadata.get("file_name", "unknown"),
                "score": float(hit.score),
                "chunk_index": hit.metadata.get("chunk_index"),
                "text": hit.text,
                "bbox": hit.metadata.get("bbox"),
                "bounding_boxes": hit.metadata.get("bounding_boxes", []),
                "block_ids": hit.metadata.get("block_ids", []),
            }
        )
    return payload


def check_ollama_status(endpoint: str = "http://127.0.0.1:11434", api_key: str = "") -> bool:
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.get(f"{endpoint}/api/tags", headers=headers, timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def check_ollama_model(endpoint: str = "http://127.0.0.1:11434", api_key: str = "", model_name: str = "") -> bool:
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.get(f"{endpoint}/api/tags", headers=headers, timeout=4)
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


def vector_diagnostics(store_path: str, sample_size: int = 500) -> dict[str, Any]:
    store = LocalVectorStore(Path(store_path))
    if not store.load() or not store.ready():
        return {
            "ready": False,
            "message": "FAISS index is not ready.",
            "vector_count": 0,
            "dim": 0,
            "points": [],
        }

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

    return {
        "ready": True,
        "message": "Vector diagnostics available.",
        "vector_count": int(store.index.ntotal if store.index is not None else len(vectors)),
        "dim": int(vectors.shape[1]),
        "points": points,
    }
