from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np

import local_archive_ai.services as services_module
from local_archive_ai.chat_engine import ChatEngine
from local_archive_ai.services import (
    EmbeddingService,
    IngestionStatus,
    answer_query,
    index_image_documents,
    search_image_chunks,
)
from local_archive_ai.store import LocalVectorStore


def _workspace_tmp_dir() -> Path:
    root = Path("data/test_runs")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"pipeline_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_test_store(store_path: Path) -> None:
    metadata = [
        {
            "source_file": "alpha.txt",
            "file_path": "alpha.txt",
            "file_name": "alpha.txt",
            "source_page": None,
            "chunk_index": 0,
            "text": "Alpha project stores solar design specifications.",
        },
        {
            "source_file": "beta.txt",
            "file_path": "beta.txt",
            "file_name": "beta.txt",
            "source_page": None,
            "chunk_index": 1,
            "text": "Beta project stores network operations notes.",
        },
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    LocalVectorStore(store_path).build(vectors, metadata)


def test_embedding_service_stays_offline(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, name: str, **kwargs):
            captured["name"] = name
            captured.update(kwargs)

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(services_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    EmbeddingService._model = None
    EmbeddingService._model_attempted = False

    model = EmbeddingService().model

    assert model is not None
    assert captured["name"] == "all-MiniLM-L6-v2"
    assert captured["local_files_only"] is True


def test_answer_query_injects_retrieved_context(monkeypatch):
    temp_dir = _workspace_tmp_dir()
    try:
        store_path = temp_dir / "faiss"
        _build_test_store(store_path)

        def fake_embed(self, texts: list[str], batch_size: int = 50) -> np.ndarray:
            if "solar" in texts[0].lower():
                return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr(EmbeddingService, "embed", fake_embed)
        monkeypatch.setattr(services_module, "check_ollama_status", lambda *args, **kwargs: True)
        monkeypatch.setattr(services_module, "_generate_with_ollama", lambda prompt, *args, **kwargs: prompt)

        payload = answer_query(
            query="What does the solar project store?",
            top_k=1,
            model_name="llama3.2:1b",
            store_path=str(store_path),
            debug=True,
        )

        assert "Alpha project stores solar design specifications." in payload["answer_markdown"]
        assert payload["citations"][0]["file_name"] == "alpha.txt"
        assert payload["debug_payload"]["pipeline_checks"]["context_injected"] is True
        assert "use your general knowledge" not in payload["debug_payload"]["prompt_text"].lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chat_engine_keeps_memory_and_streams(monkeypatch):
    temp_dir = _workspace_tmp_dir()
    try:
        store_path = temp_dir / "faiss"
        _build_test_store(store_path)

        def fake_embed(self, texts: list[str], batch_size: int = 50) -> np.ndarray:
            if "first" in texts[0].lower() or "solar" in texts[0].lower():
                return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr(EmbeddingService, "embed", fake_embed)
        monkeypatch.setattr("local_archive_ai.chat_engine.check_ollama_status", lambda *args, **kwargs: True)

        captured_prompts: list[str] = []

        def fake_generate(self, prompt: str) -> str:
            captured_prompts.append(prompt)
            return "final answer"

        def fake_generate_stream(self, prompt: str):
            captured_prompts.append(prompt)
            yield "stream"
            yield "ed"

        monkeypatch.setattr(ChatEngine, "_generate", fake_generate)
        monkeypatch.setattr(ChatEngine, "_generate_stream", fake_generate_stream)

        engine = ChatEngine(store_path=str(store_path))
        first = engine.query("First solar question")
        streamed = "".join(engine.query_stream("Follow up"))

        assert first.answer == "final answer"
        assert streamed == "streamed"
        assert len(engine.get_memory()) == 2
        assert "Previous conversation:" in captured_prompts[-1]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_image_index_uses_ocr_text_for_retrieval(monkeypatch):
    temp_dir = _workspace_tmp_dir()
    try:
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        (image_dir / "diagram.png").write_bytes(b"png")

        def fake_extract(path: Path, chunk_size: int, use_code_parsing: bool = True, ocr_engine: str = "tesseract", chunk_index_offset: int = 0):
            return (
                [
                    {
                        "source_file": str(path),
                        "file_path": str(path),
                        "file_name": path.name,
                        "source_page": None,
                        "chunk_index": chunk_index_offset,
                        "text": "diagram contains inverter wiring labels",
                        "bbox": [0, 0, 50, 50],
                        "bounding_boxes": [[0, 0, 50, 50]],
                        "block_ids": ["b1"],
                    }
                ],
                IngestionStatus.SUCCESS,
                "",
                "tesseract",
                0.01,
                chunk_index_offset + 1,
            )

        def fake_embed(self, texts: list[str], batch_size: int = 50) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr(services_module, "extract_document_chunks_resilient", fake_extract)
        monkeypatch.setattr(EmbeddingService, "embed", fake_embed)

        report = index_image_documents(
            folder_path=str(image_dir),
            chunk_size=200,
            store_path=str(temp_dir / "image_index"),
        )
        results = search_image_chunks(
            query="wiring labels",
            top_k=1,
            store_path=str(temp_dir / "image_index"),
        )

        assert report.chunk_count == 1
        assert results[0]["text"] == "diagram contains inverter wiring labels"
        assert results[0]["bbox"] == [0, 0, 50, 50]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
