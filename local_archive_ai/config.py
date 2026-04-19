from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


SUPPORTED_PYTHON = (3, 12)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    chunk_size: int = Field(default=500, ge=1, description="Size of text chunks in tokens")
    top_k: int = Field(default=4, ge=1, description="Number of top results to retrieve")
    model_name: str = Field(default="llama3.2:1b", min_length=1, description="Name of the Ollama model")
    faiss_path: str = Field(default="data/faiss_index", min_length=1, description="Path to FAISS index")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    ollama_endpoint: str = Field(default="http://127.0.0.1:11434", min_length=1, description="Ollama API endpoint")
    ollama_api_key: str = Field(default="", description="Ollama API key (use env var)")
    ocr_engine: str = Field(default="tesseract", pattern=r"^(tesseract|easyocr)$", description="OCR engine to use")
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for BM25 in hybrid search")
    rerank_top_n: int = Field(default=5, ge=1, description="Number of top results to rerank")
    retrieval_mode: str = Field(default="vector", pattern=r"^(vector|hybrid|hybrid\+rerank)$", description="Retrieval mode")
    # --- Overhaul additions ---
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Min similarity for confident answer")
    enable_watcher: bool = Field(default=False, description="Enable folder watcher for auto-indexing")
    watcher_folder: str = Field(default="", description="Folder to watch for new documents")
    query_cache_ttl: int = Field(default=3600, ge=0, description="Query cache TTL in seconds (0 to disable)")
    embedding_batch_size: int = Field(default=50, ge=1, description="Batch size for embedding generation")
    ocr_workers: int = Field(default=0, ge=0, description="OCR parallel workers (0 = auto)")
    max_context_tokens: int = Field(default=8192, ge=512, description="Max context window tokens for LLM")


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def _get_env_value(key: str, default: Any) -> Any:
    value = None
    if key in os.environ:
        value = os.environ[key]
    return value if value is not None else default


def _check_duplicate_keys(yaml_content: str) -> None:
    import re
    keys = set()
    for line in yaml_content.splitlines():
        stripped = line.strip()
        if ':' in stripped and not stripped.startswith(' ') and not stripped.startswith('-'):
            key = stripped.split(':', 1)[0].strip()
            if key in keys:
                raise ValueError(f"Duplicate key '{key}' in config.yaml")
            keys.add(key)


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or _default_config_path()
    if not cfg_path.exists():
        config = AppConfig()
        env_api_key = _get_env_value("OLLAMA_API_KEY", config.ollama_api_key)
        config.ollama_api_key = str(env_api_key)
        config.ollama_endpoint = str(_get_env_value("OLLAMA_ENDPOINT", config.ollama_endpoint))
        return config

    with cfg_path.open("r", encoding="utf-8") as fh:
        content = fh.read()
        _check_duplicate_keys(content)
        payload: dict[str, Any] = yaml.safe_load(content) or {}

    try:
        config = AppConfig(**payload)
    except ValidationError as e:
        raise ValueError(f"Config validation error: {e}") from e

    config.ollama_api_key = str(_get_env_value("OLLAMA_API_KEY", config.ollama_api_key))
    config.ollama_endpoint = str(_get_env_value("OLLAMA_ENDPOINT", config.ollama_endpoint))
    return config


def python_runtime_status() -> dict[str, Any]:
    major = int(sys.version_info.major)
    minor = int(sys.version_info.minor)
    supported = (major, minor) == SUPPORTED_PYTHON
    version = f"{major}.{minor}.{sys.version_info.micro}"
    return {
        "version": version,
        "supported": supported,
        "supported_version": f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}",
        "message": (
            f"Python {version} detected. This project is validated on Python "
            f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}."
        ),
    }


def save_config(config: AppConfig, path: Path | None = None) -> None:
    cfg_path = path or _default_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    favorite = config.model_dump()
    # Never save API key to config file
    favorite["ollama_api_key"] = ""
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(favorite, fh, sort_keys=False)
