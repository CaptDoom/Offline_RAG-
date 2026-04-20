from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import api
from local_archive_ai.config import AppConfig
from local_archive_ai import services as local_services
from local_archive_ai.services import index_image_documents, load_index_metadata, search_image_chunks, search_index
from local_archive_ai.store import LocalVectorStore


def _test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        faiss_path=str(tmp_path / "faiss_index"),
        model_name="llama3.2:1b",
        ollama_endpoint="http://127.0.0.1:11434",
        retrieval_mode="vector",
        ocr_engine="tesseract",
    )


def test_api_status_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path))
    monkeypatch.setattr(api, "get_index_status", lambda _: {"exists": True, "file_count": 2, "chunk_count": 8})
    monkeypatch.setattr(api, "runtime_mode", lambda: {"mode": "CPU Fallback", "vram": "N/A"})
    monkeypatch.setattr(api, "system_checks", lambda *args, **kwargs: {"ollama_reachable": True})
    monkeypatch.setattr(api, "vector_diagnostics", lambda *args, **kwargs: {"ready": True, "vector_count": 8, "dim": 384, "points": []})
    monkeypatch.setattr(api, "check_ollama_status", lambda *args, **kwargs: True)

    client = TestClient(api.app)
    response = client.get("/api/status")
    payload = response.json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["index"]["chunk_count"] == 8
    assert "python_runtime" in payload


def test_api_chat_returns_stable_error_for_missing_index(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path))
    monkeypatch.setattr(api, "get_index_status", lambda _: {"exists": False})

    client = TestClient(api.app)
    response = client.post("/api/chat", json={"query": "hello"})
    payload = response.json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert payload["citations"] == []
    assert payload["debug_payload"] == {}


def test_api_chat_success_shape(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path))
    monkeypatch.setattr(api, "get_index_status", lambda _: {"exists": True})
    monkeypatch.setattr(api, "check_ollama_status", lambda *args, **kwargs: True)
    monkeypatch.setattr(api, "check_ollama_model", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        api,
        "answer_query",
        lambda **kwargs: {
            "answer_markdown": "Offline answer",
            "citations": [{"file_name": "notes.pdf"}],
            "debug_payload": {"prompt_text": "prompt"},
        },
    )

    client = TestClient(api.app)
    response = client.post("/api/chat", json={"query": "hello"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["answer_markdown"] == "Offline answer"
    assert payload["citations"][0]["file_name"] == "notes.pdf"


def test_api_vault_normalizes_dict_and_list_metadata(monkeypatch, tmp_path: Path) -> None:
    store_path = tmp_path / "vault_store"
    store_path.mkdir()
    (store_path / "metadata.json").write_text(
        json.dumps({"chunks": [{"file_name": "dict.pdf", "text": "A"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path).model_copy(update={"faiss_path": str(store_path)}))
    client = TestClient(api.app)
    payload = client.get("/api/vault").json()
    assert payload["success"] is True
    assert isinstance(payload["chunks"], list)
    assert payload["chunks"][0]["file_name"] == "dict.pdf"

    (store_path / "metadata.json").write_text(
        json.dumps([{"file_name": "list.pdf", "text": "B"}]),
        encoding="utf-8",
    )
    payload = client.get("/api/vault").json()
    assert payload["success"] is True
    assert payload["chunks"][0]["file_name"] == "list.pdf"


def test_search_index_hybrid_rerank_falls_back_without_local_reranker(monkeypatch, tmp_path: Path) -> None:
    store_path = tmp_path / "search_store"
    metadata = [
        {"file_name": "alpha.txt", "file_path": "alpha.txt", "chunk_index": 1, "text": "invoice total amount"},
        {"file_name": "beta.txt", "file_path": "beta.txt", "chunk_index": 1, "text": "security policy"},
    ]
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    LocalVectorStore(store_path).build(vectors, metadata)

    monkeypatch.setattr(
        local_services.EmbeddingService,
        "embed",
        lambda self, texts, batch_size=16: np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32),
    )
    monkeypatch.setattr(local_services, "load_reranker_model", lambda: None)

    hits = search_index(
        query="invoice",
        top_k=1,
        store_path=str(store_path),
        retrieval_mode="hybrid+rerank",
        bm25_weight=0.4,
    )

    assert len(hits) == 1
    assert hits[0].metadata["file_name"] == "alpha.txt"


def test_image_index_and_search_fixture(monkeypatch, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_file = image_dir / "scan.png"
    image_file.write_bytes(b"fake-image")

    monkeypatch.setattr(
        local_services.ImageEmbeddingService,
        "embed_images",
        lambda self, paths, batch_size=16: np.asarray(
            [[1.0, 0.0, 0.0] for _ in paths],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        local_services.ImageEmbeddingService,
        "embed_text",
        lambda self, texts: np.asarray(
            [
                [1.0, 0.0, 0.0] if "invoice" in text.lower() else [0.0, 1.0, 0.0]
                for text in texts
            ],
            dtype=np.float32,
        ),
    )

    store_path = tmp_path / "image_store"
    report = index_image_documents(
        folder_path=str(image_dir),
        chunk_size=32,
        store_path=str(store_path),
    )
    results = search_image_chunks(
        query="invoice",
        top_k=1,
        store_path=str(store_path),
        retrieval_mode="vector",
    )

    assert report.file_count == 1
    assert report.chunk_count == 1
    assert load_index_metadata(store_path)[0]["file_name"] == "scan.png"
    assert results[0]["file_name"] == "scan.png"


def test_image_search_module_imports() -> None:
    module = importlib.import_module("image_search")
    assert hasattr(module, "main")


def test_api_index_accepts_mixed_sources_and_preserves_folder_structure(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path))

    def fake_index_documents(folder_path: str, chunk_size: int, store_path: str, on_progress=None):
        root = Path(folder_path)
        files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
        seen["folder_path"] = folder_path
        seen["files"] = files
        return type("Report", (), {"file_count": len(files), "chunk_count": len(files), "index_path": store_path})()

    monkeypatch.setattr(api, "index_documents", fake_index_documents)

    client = TestClient(api.app)
    response = client.post(
        "/api/index",
        data={
            "sources": json.dumps([
                {"file_index": 0, "data_type": "pdf", "relative_path": "docs/specs/plan.pdf"},
                {"file_index": 1, "data_type": "image", "relative_path": "images/diagram.png"},
            ]),
        },
        files=[
            ("files", ("plan.pdf", b"pdf-bytes", "application/pdf")),
            ("files", ("diagram.png", b"image-bytes", "image/png")),
        ],
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert seen["files"] == ["docs/specs/plan.pdf", "images/diagram.png"]


def test_api_image_index_rejects_non_image_uploads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api, "_config", lambda: _test_config(tmp_path))
    client = TestClient(api.app)

    response = client.post(
        "/api/image/index",
        data={
            "sources": json.dumps([
                {"file_index": 0, "data_type": "pdf", "relative_path": "docs/spec.pdf"},
            ]),
        },
        files=[
            ("files", ("spec.pdf", b"pdf-bytes", "application/pdf")),
        ],
    )
    payload = response.json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert "only accepts image inputs" in payload["error"]
