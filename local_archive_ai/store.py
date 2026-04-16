from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
    _faiss_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover
    faiss = None
    _faiss_import_error = exc

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


@dataclass
class SearchHit:
    score: float
    metadata: dict[str, Any]
    text: str


class LocalVectorStore:
    def __init__(self, storage_path: Path) -> None:
        if faiss is None:
            raise RuntimeError(
                "FAISS is required for production mode. Install dependency `faiss-cpu`."
            ) from _faiss_import_error
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.faiss"
        self.metadata_file = self.storage_path / "metadata.json"
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.bm25 = None
        self.bm25_tokenized_docs: list[list[str]] = []
        self.file_hashes: dict[str, str] = {}
        self.faiss_enabled = True
        self.load_error = ""

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        cleaned = str(text).lower()
        return [token for token in cleaned.split() if token]

    @staticmethod
    def _hash_file(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _prepare_bm25(self) -> None:
        self.bm25_tokenized_docs = [self._tokenize_text(item.get("text", "")) for item in self.metadata]
        if BM25Okapi is not None and self.bm25_tokenized_docs:
            self.bm25 = BM25Okapi(self.bm25_tokenized_docs)
        else:
            self.bm25 = None

    def _write_metadata(self) -> None:
        payload = {
            "chunks": self.metadata,
            "file_hashes": self.file_hashes,
        }
        with self.metadata_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _read_metadata(self) -> None:
        raw = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "chunks" in raw:
            self.metadata = raw.get("chunks", [])
            self.file_hashes = raw.get("file_hashes", {})
        elif isinstance(raw, list):
            self.metadata = raw
            self.file_hashes = {}
        else:
            raise ValueError("Unsupported metadata format in metadata.json")

    def build(
        self,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        file_hashes: dict[str, str] | None = None,
    ) -> None:
        vectors = vectors.astype("float32")
        vectors = self._normalize(vectors)
        self.metadata = metadata
        self.file_hashes = file_hashes or self.file_hashes
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_file))
        self._prepare_bm25()
        self._write_metadata()

    def load(self) -> bool:
        self.load_error = ""
        if not self.metadata_file.exists():
            return False
        try:
            self._read_metadata()
            self._prepare_bm25()
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                return True
            return False
        except Exception as exc:
            self.load_error = str(exc)
            return False

    def ready(self) -> bool:
        if not self.metadata:
            return False
        return self.index is not None

    def load_status(self) -> tuple[bool, str]:
        loaded = self.load()
        return loaded, self.load_error

    def _format_hits(self, scores: np.ndarray, idxs: np.ndarray) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[int(idx)]
            hits.append(
                SearchHit(
                    score=float(score),
                    metadata=item,
                    text=str(item.get("text", "")),
                )
            )
        return hits

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        query_vector = query_vector.astype("float32").reshape(1, -1)
        query_vector = self._normalize(query_vector)

        if self.index is not None:
            scores, idxs = self.index.search(query_vector, top_k)
            return self._format_hits(scores[0], idxs[0])
        return []

    def search_bm25(self, query: str, top_k: int) -> list[SearchHit]:
        if self.bm25 is None:
            return []
        tokens = self._tokenize_text(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        ranked = np.argsort(scores)[::-1][:top_k]
        hits: list[SearchHit] = []
        for idx in ranked:
            if idx < 0 or idx >= len(self.metadata):
                continue
            hits.append(
                SearchHit(
                    score=float(scores[idx]),
                    metadata=self.metadata[int(idx)],
                    text=str(self.metadata[int(idx)].get("text", "")),
                )
            )
        return hits

    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        semantic_weight: float = 0.6,
    ) -> list[SearchHit]:
        semantic_hits = self.search(query_vector, max(top_k * 2, 1))
        bm25_hits = self.search_bm25(query, max(top_k * 3, 1))
        candidate_scores: dict[str, dict[str, Any]] = {}

        max_sem = max((hit.score for hit in semantic_hits), default=0.0)
        max_bm = max((hit.score for hit in bm25_hits), default=0.0)

        def candidate_key(hit: SearchHit) -> str:
            return (
                f"{hit.metadata.get('file_path','')}::"
                f"{hit.metadata.get('source_page','')}::"
                f"{hit.metadata.get('chunk_index','')}"
            )

        for hit in semantic_hits:
            key = candidate_key(hit)
            candidate_scores.setdefault(key, {
                "semantic": 0.0,
                "bm25": 0.0,
                "metadata": hit.metadata,
                "text": hit.text,
            })
            candidate_scores[key]["semantic"] = float(hit.score)

        for hit in bm25_hits:
            key = candidate_key(hit)
            candidate_scores.setdefault(key, {
                "semantic": 0.0,
                "bm25": 0.0,
                "metadata": hit.metadata,
                "text": hit.text,
            })
            candidate_scores[key]["bm25"] = float(hit.score)

        hits: list[SearchHit] = []
        for values in candidate_scores.values():
            sem = values["semantic"]
            bm = values["bm25"]
            normalized_sem = sem / max_sem if max_sem > 0 else 0.0
            normalized_bm = bm / max_bm if max_bm > 0 else 0.0
            combined = semantic_weight * normalized_sem + (1.0 - semantic_weight) * normalized_bm
            hits.append(
                SearchHit(
                    score=combined,
                    metadata=values["metadata"],
                    text=values["text"],
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def sample_vectors(self, sample_size: int = 500) -> np.ndarray:
        if self.index is None:
            return np.empty((0, 0), dtype=np.float32)
        total = self.index.ntotal
        if total <= 0:
            return np.empty((0, 0), dtype=np.float32)
        take = min(sample_size, total)
        idxs = np.linspace(0, total - 1, num=take, dtype=int)
        vectors = [self.index.reconstruct(int(i)) for i in idxs]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

