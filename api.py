from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from local_archive_ai.config import AppConfig, load_config, python_runtime_status, save_config
from local_archive_ai.services import (
    answer_query,
    check_ollama_model,
    check_ollama_status,
    get_index_status,
    index_documents,
    load_index_metadata,
    runtime_mode,
    system_checks,
    vector_diagnostics,
)

app = FastAPI(title="Local Archive AI Backend")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    query: str


class BatchQueryRequest(BaseModel):
    queries: list[str]


class ConfigUpdateRequest(BaseModel):
    chunk_size: int
    top_k: int
    model_name: str
    faiss_path: str
    bm25_weight: float
    rerank_top_n: int
    debug_mode: bool
    ollama_endpoint: str
    retrieval_mode: str
    ocr_engine: str


def _ok(**payload: object) -> dict[str, object]:
    return {"success": True, "error": None, **payload}


def _error(message: str, status_code: int = 400, **payload: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message, **payload},
    )


def _config() -> AppConfig:
    return load_config()


@app.get("/api/status")
def get_status() -> dict[str, object]:
    config = _config()
    idx = get_index_status(config.faiss_path)
    checks = system_checks(
        config.faiss_path,
        config.model_name,
        endpoint=config.ollama_endpoint,
        api_key=config.ollama_api_key,
    )
    return _ok(
        index=idx,
        mode=runtime_mode(),
        checks=checks,
        diagnostics=vector_diagnostics(config.faiss_path, sample_size=10),
        ollama_active=check_ollama_status(config.ollama_endpoint, config.ollama_api_key),
        config=config.model_dump(),
        python_runtime=python_runtime_status(),
    )


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest) -> dict[str, object]:
    updated = _config().model_copy(update=req.model_dump())
    save_config(updated)
    return _ok(status="success", config=updated.model_dump())


@app.post("/api/chat")
def chat(req: QueryRequest):
    config = _config()
    query = req.query.strip()
    if not query:
        return _error(
            "Query cannot be empty.",
            answer_markdown="",
            citations=[],
            debug_payload={},
        )

    idx = get_index_status(config.faiss_path)
    if not idx.get("exists"):
        return _error(
            "No local index is ready. Index documents before querying.",
            answer_markdown="",
            citations=[],
            debug_payload={},
        )

    if not check_ollama_status(config.ollama_endpoint, config.ollama_api_key):
        return _error(
            f"Ollama is not reachable at {config.ollama_endpoint}.",
            status_code=503,
            answer_markdown="",
            citations=[],
            debug_payload={},
        )

    if not check_ollama_model(config.ollama_endpoint, config.ollama_api_key, config.model_name):
        return _error(
            f"Configured model '{config.model_name}' is not available in local Ollama.",
            status_code=503,
            answer_markdown="",
            citations=[],
            debug_payload={},
        )

    try:
        payload = answer_query(
            query=query,
            top_k=config.top_k,
            model_name=config.model_name,
            store_path=config.faiss_path,
            debug=True,
            ollama_endpoint=config.ollama_endpoint,
            ollama_api_key=config.ollama_api_key,
            retrieval_mode=config.retrieval_mode,
            bm25_weight=config.bm25_weight,
            rerank_top_n=config.rerank_top_n,
        )
        return _ok(**payload)
    except Exception as exc:
        return _error(
            str(exc),
            answer_markdown="",
            citations=[],
            debug_payload={},
        )


@app.post("/api/batch")
def batch_queries(req: BatchQueryRequest) -> dict[str, object]:
    config = _config()
    queries = [query.strip() for query in req.queries if query.strip()]
    if not queries:
        return _ok(results=[], failed_count=0, completed_count=0)

    results: list[dict[str, object]] = []
    failed_count = 0
    for idx, query in enumerate(queries, start=1):
        row: dict[str, object] = {
            "id": f"B{idx:03d}",
            "query": query,
            "status": "PROCESSING",
            "answer": "",
            "target": "LOCAL_ARCHIVE",
        }
        try:
            payload = answer_query(
                query=query,
                top_k=config.top_k,
                model_name=config.model_name,
                store_path=config.faiss_path,
                debug=False,
                ollama_endpoint=config.ollama_endpoint,
                ollama_api_key=config.ollama_api_key,
                retrieval_mode=config.retrieval_mode,
                bm25_weight=config.bm25_weight,
                rerank_top_n=config.rerank_top_n,
            )
            row["answer"] = payload.get("answer_markdown", "")
            row["status"] = "COMPLETED"
        except Exception as exc:
            row["answer"] = str(exc)
            row["status"] = "FAILED"
            failed_count += 1
        results.append(row)

    return _ok(
        results=results,
        failed_count=failed_count,
        completed_count=len(results) - failed_count,
    )


@app.post("/api/index")
async def build_index(folder_path: str | None = Form(default=None), files: list[UploadFile] | None = File(default=None)):
    config = _config()
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    try:
        folder_to_index = folder_path.strip() if folder_path else ""
        if files:
            tmp_dir = tempfile.TemporaryDirectory(prefix="local_archive_import_")
            for upload in files:
                if not upload.filename:
                    continue
                file_path = Path(tmp_dir.name) / upload.filename
                file_path.write_bytes(await upload.read())
                if file_path.suffix.lower() == ".zip":
                    with zipfile.ZipFile(file_path) as archive:
                        archive.extractall(tmp_dir.name)
            folder_to_index = tmp_dir.name

        if not folder_to_index:
            return _error("No folder path or files provided.")

        folder = Path(folder_to_index).expanduser()
        if not folder.exists():
            return _error(f"Folder not found: {folder}")

        report = index_documents(
            folder_path=str(folder),
            chunk_size=config.chunk_size,
            store_path=config.faiss_path,
            on_progress=None,
        )
        return _ok(
            status="success",
            file_count=report.file_count,
            chunk_count=report.chunk_count,
            index_path=report.index_path,
        )
    except Exception as exc:
        return _error(str(exc))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


@app.get("/api/vault")
def get_vault_data() -> dict[str, object]:
    config = _config()
    return _ok(chunks=load_index_metadata(config.faiss_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
