"""SQLite-backed query result cache with TTL for Local Archive AI."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "query_cache.db"
_DEFAULT_TTL = 3600  # 1 hour


class QueryCache:
    """Thread-safe SQLite cache for query results."""

    def __init__(self, db_path: Path | None = None, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self.db_path = db_path or _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()

    @staticmethod
    def _make_key(query: str, top_k: int, retrieval_mode: str, store_path: str) -> str:
        raw = f"{query}|{top_k}|{retrieval_mode}|{store_path}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, top_k: int, retrieval_mode: str, store_path: str) -> dict[str, Any] | None:
        key = self._make_key(query, top_k, retrieval_mode, store_path)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT result_json, created_at FROM query_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        result_json, created_at = row
        if time.time() - created_at > self.ttl:
            conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (key,))
            conn.commit()
            return None
        return json.loads(result_json)

    def put(self, query: str, top_k: int, retrieval_mode: str, store_path: str, result: dict[str, Any]) -> None:
        key = self._make_key(query, top_k, retrieval_mode, store_path)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO query_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(result, ensure_ascii=False, default=str), time.time()),
        )
        conn.commit()

    def invalidate_all(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM query_cache")
        conn.commit()

    def prune_expired(self) -> int:
        conn = self._get_conn()
        cutoff = time.time() - self.ttl
        cursor = conn.execute("DELETE FROM query_cache WHERE created_at < ?", (cutoff,))
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
