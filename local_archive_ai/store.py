from __future__ import annotations

import hashlib
import json
import shutil
import time
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

from local_archive_ai.logging_config import log

_MAX_CHECKPOINTS = 3
_CHECKPOINT_INTERVAL = 1000  # chunks


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
        has_tokens = any(len(doc) > 0 for doc in self.bm25_tokenized_docs)
        if BM25Okapi is not None and self.bm25_tokenized_docs and has_tokens:
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
        log.info("Index built: %d vectors, dim=%d", self.index.ntotal, dim)

    # ------------------------------------------------------------------
    # Incremental indexing (Problem J)
    # ------------------------------------------------------------------
    def add_vectors(
        self,
        new_vectors: np.ndarray,
        new_metadata: list[dict[str, Any]],
        new_file_hashes: dict[str, str] | None = None,
    ) -> None:
        """Add vectors without rebuilding the entire index."""
        new_vectors = new_vectors.astype("float32")
        new_vectors = self._normalize(new_vectors)
        if self.index is None:
            dim = new_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(new_vectors)
        self.metadata.extend(new_metadata)
        if new_file_hashes:
            self.file_hashes.update(new_file_hashes)
        faiss.write_index(self.index, str(self.index_file))
        self._prepare_bm25()
        self._write_metadata()
        log.info("Incremental add: +%d vectors (total %d)", len(new_metadata), self.index.ntotal)

    def delete_document(self, file_path: str) -> int:
        """Remove all chunks belonging to *file_path* and rebuild the index."""
        if not self.metadata:
            return 0
        keep_indices = [i for i, m in enumerate(self.metadata) if m.get("file_path") != file_path]
        removed = len(self.metadata) - len(keep_indices)
        if removed == 0:
            return 0
        # Reconstruct vectors for kept items
        if self.index is not None and keep_indices:
            kept_vectors = np.array(
                [self.index.reconstruct(int(i)) for i in keep_indices], dtype=np.float32
            )
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.file_hashes.pop(file_path, None)
            dim = kept_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(kept_vectors)
            faiss.write_index(self.index, str(self.index_file))
        else:
            self.metadata = []
            self.file_hashes.pop(file_path, None)
            self.index = None
        self._prepare_bm25()
        self._write_metadata()
        log.info("Deleted %d chunks for %s", removed, file_path)
        return removed

    # ------------------------------------------------------------------
    # Checkpoint / backup (Problem T)
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> Path | None:
        """Save a timestamped checkpoint of the index + metadata."""
        cp_dir = self.storage_path / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        dest = cp_dir / f"checkpoint_{ts}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if self.index_file.exists():
                shutil.copy2(str(self.index_file), str(dest / "index.faiss"))
            if self.metadata_file.exists():
                shutil.copy2(str(self.metadata_file), str(dest / "metadata.json"))
            self._prune_old_checkpoints(cp_dir)
            log.info("Checkpoint saved: %s", dest)
            return dest
        except Exception:
            log.exception("Checkpoint save failed")
            return None

    def list_checkpoints(self) -> list[Path]:
        cp_dir = self.storage_path / "checkpoints"
        if not cp_dir.exists():
            return []
        return sorted(cp_dir.iterdir(), key=lambda p: p.name, reverse=True)

    def rollback_checkpoint(self, checkpoint_path: Path) -> bool:
        """Restore index from a checkpoint directory."""
        idx_src = checkpoint_path / "index.faiss"
        meta_src = checkpoint_path / "metadata.json"
        if not idx_src.exists() or not meta_src.exists():
            return False
        try:
            shutil.copy2(str(idx_src), str(self.index_file))
            shutil.copy2(str(meta_src), str(self.metadata_file))
            self.load()
            log.info("Rolled back to checkpoint %s", checkpoint_path.name)
            return True
        except Exception:
            log.exception("Rollback failed")
            return False

    @staticmethod
    def _prune_old_checkpoints(cp_dir: Path) -> None:
        existing = sorted(cp_dir.iterdir(), key=lambda p: p.name)
        while len(existing) > _MAX_CHECKPOINTS:
            oldest = existing.pop(0)
            shutil.rmtree(str(oldest), ignore_errors=True)

    # ------------------------------------------------------------------
    # Export / Import (Problem S)
    # ------------------------------------------------------------------
    def export_index(self, dest: Path) -> Path:
        """Export index + metadata as a zip."""
        import zipfile
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(dest), "w", zipfile.ZIP_DEFLATED) as zf:
            if self.index_file.exists():
                zf.write(str(self.index_file), "index.faiss")
            if self.metadata_file.exists():
                zf.write(str(self.metadata_file), "metadata.json")
        log.info("Index exported to %s", dest)
        return dest

    def import_index(self, src: Path) -> bool:
        """Import index from a zip file."""
        import zipfile
        if not src.exists():
            return False
        try:
            with zipfile.ZipFile(str(src), "r") as zf:
                zf.extractall(str(self.storage_path))
            return self.load()
        except Exception:
            log.exception("Index import failed")
            return False

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
        if len(scores) <= top_k:
            ranked = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -top_k)[-top_k:]
            ranked = idx[np.argsort(scores[idx])[::-1]]
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

    def chunk_count(self) -> int:
        return len(self.metadata)

    def is_file_unchanged(self, file_path: str, file_hash: str) -> bool:
        """Check if a file's hash matches the stored hash (Problem H)."""
        return self.file_hashes.get(file_path) == file_hash

