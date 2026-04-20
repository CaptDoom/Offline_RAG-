from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import uuid
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from local_archive_ai.config import AppConfig, load_config, python_runtime_status, save_config
from local_archive_ai.services import (
    IMAGE_EXTENSIONS,
    answer_query,
    check_ollama_model,
    check_ollama_status,
    get_index_status,
    index_documents,
    index_image_documents,
    load_index_metadata,
    runtime_mode,
    search_image_chunks,
    system_checks,
    vector_diagnostics,
)
from local_archive_ai.chat_engine import ChatEngine

IMAGE_STORE_PATH = "data/image_faiss_index"
_CHAT_ENGINES: dict[str, ChatEngine] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preflight check on startup
    try:
        config = load_config()
        system_checks(config.faiss_path, config.model_name, endpoint=config.ollama_endpoint, api_key=config.ollama_api_key)
    except Exception as e:
        print(f"Warning: Startup system check failed: {e}")
    yield

app = FastAPI(title="Local Archive AI Backend", lifespan=lifespan)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class BatchQueryRequest(BaseModel):
    queries: list[str]


class ImageSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieval_mode: str = "vector"
    bm25_weight: float = 0.4


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
    
    model_config = {"protected_namespaces": ()}


def _ok(**payload: object) -> dict[str, object]:
    return {"success": True, "error": None, **payload}


def _error(message: str, status_code: int = 400, **payload: object) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message, **payload},
    )


def _config() -> AppConfig:
    return load_config()


def _build_chat_engine(config: AppConfig) -> ChatEngine:
    return ChatEngine(
        store_path=config.faiss_path,
        model_name=config.model_name,
        ollama_endpoint=config.ollama_endpoint,
        ollama_api_key=config.ollama_api_key,
        top_k=config.top_k,
        retrieval_mode=config.retrieval_mode,
        bm25_weight=config.bm25_weight,
        rerank_top_n=config.rerank_top_n,
        confidence_threshold=config.confidence_threshold,
        temperature=config.temperature,
        max_context_tokens=config.max_context_tokens,
    )


def _get_chat_engine(session_id: str | None, config: AppConfig) -> ChatEngine:
    if not session_id:
        return _build_chat_engine(config)
    engine = _CHAT_ENGINES.get(session_id)
    if engine is None:
        engine = _build_chat_engine(config)
        _CHAT_ENGINES[session_id] = engine
    return engine


def _load_sources_payload(raw_sources: str | None) -> list[dict[str, object]]:
    if not raw_sources:
        return []
    try:
        parsed = json.loads(raw_sources)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid sources payload: {exc.msg}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Sources payload must be a list.")
    return [item for item in parsed if isinstance(item, dict)]


def _safe_relative_parts(relative_path: str, fallback_name: str) -> list[str]:
    raw_path = (relative_path or fallback_name or "").replace("\\", "/").strip("/")
    parts = [part for part in raw_path.split("/") if part not in {"", ".", ".."}]
    return parts or [fallback_name or "upload.bin"]


async def _materialize_upload_sources(
    *,
    tmp_path: Path,
    uploads: list[UploadFile],
    sources: list[dict[str, object]],
    images_only: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_by_index = {
        int(item["file_index"]): item
        for item in sources
        if isinstance(item.get("file_index"), int)
    }

    for idx, upload in enumerate(uploads):
        if not upload.filename:
            continue

        source = source_by_index.get(idx, {})
        relative_path = str(source.get("relative_path") or upload.filename)
        file_parts = _safe_relative_parts(relative_path, upload.filename)
        data_type = str(source.get("data_type") or "").lower()
        target_path = tmp_path.joinpath(*file_parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(await upload.read())

        if images_only and data_type and data_type != "image":
            raise ValueError(f"Image indexing only accepts image inputs. Rejected: {upload.filename}")

        if images_only and target_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image input: {upload.filename}")

        if not images_only and target_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(target_path) as archive:
                    extract_dir = target_path.with_suffix("")
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    archive.extractall(extract_dir)
            except Exception as zip_e:
                print(f"Failed to extract {target_path}: {zip_e}")

    return tmp_path


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    err_msgs = [f"{err.get('loc')}: {err.get('msg')}" for err in exc.errors()]
    return _error(f"Validation error: {', '.join(err_msgs)}", status_code=422)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    traceback.print_exc()
    return _error(f"Internal server error: {str(exc)}", status_code=500)


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
        image_index=get_index_status(IMAGE_STORE_PATH),
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
        engine = _get_chat_engine(req.session_id, config)
        response = engine.query(query)
        return _ok(
            answer_markdown=response.answer,
            citations=response.citations,
            max_similarity=response.max_similarity,
            low_confidence=response.low_confidence,
            duration_ms=response.duration_ms,
            debug_payload=response.debug_payload,
            chunks_used=response.chunks_used,
            session_id=req.session_id,
        )
    except Exception as exc:
        return _error(
            str(exc),
            answer_markdown="",
            citations=[],
            debug_payload={},
        )


@app.post("/api/chat/stream")
def chat_stream(req: QueryRequest):
    config = _config()
    query = req.query.strip()
    if not query:
        return _error("Query cannot be empty.", status_code=400)

    idx = get_index_status(config.faiss_path)
    if not idx.get("exists"):
        return _error("No local index is ready. Index documents before querying.")

    if not check_ollama_status(config.ollama_endpoint, config.ollama_api_key):
        return _error(
            f"Ollama is not reachable at {config.ollama_endpoint}.",
            status_code=503,
        )

    if not check_ollama_model(config.ollama_endpoint, config.ollama_api_key, config.model_name):
        return _error(
            f"Configured model '{config.model_name}' is not available in local Ollama.",
            status_code=503,
        )

    engine = _get_chat_engine(req.session_id, config)
    return StreamingResponse(engine.query_stream(query), media_type="text/plain; charset=utf-8")


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
async def build_index(
    folder_path: str | None = Form(default=None),
    files: list[UploadFile] | None = File(default=None),
    sources: str | None = Form(default=None),
):
    config = _config()
    session_id = uuid.uuid4().hex[:8]
    tmp_path = Path("data/sessions") / f"import_{session_id}"

    try:
        folder_to_index = folder_path.strip() if folder_path else ""
        upload_list = files or []
        source_items = _load_sources_payload(sources)
        if upload_list:
            folder_to_index = str(
                await _materialize_upload_sources(
                    tmp_path=tmp_path,
                    uploads=upload_list,
                    sources=source_items,
                )
            )

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
            ocr_engine=config.ocr_engine,
        )
        return _ok(
            status="success",
            file_count=report.file_count,
            chunk_count=report.chunk_count,
            index_path=report.index_path,
            session_id=session_id
        )
    except Exception as exc:
        return _error(str(exc))
    # Removed automatic cleanup of temp files to prevent race conditions during preview and async reading.


@app.post("/api/image/index")
async def build_image_index(
    folder_path: str | None = Form(default=None),
    files: list[UploadFile] | None = File(default=None),
    sources: str | None = Form(default=None),
):
    config = _config()
    session_id = uuid.uuid4().hex[:8]
    tmp_path = Path("data/sessions") / f"image_{session_id}"

    try:
        folder_to_index = folder_path.strip() if folder_path else ""
        upload_list = files or []
        source_items = _load_sources_payload(sources)
        if upload_list:
            folder_to_index = str(
                await _materialize_upload_sources(
                    tmp_path=tmp_path,
                    uploads=upload_list,
                    sources=source_items,
                    images_only=True,
                )
            )

        if not folder_to_index:
            return _error("No folder path or image files provided.")

        folder = Path(folder_to_index).expanduser()
        if not folder.exists():
            return _error(f"Folder not found: {folder}")

        report = index_image_documents(
            folder_path=str(folder),
            chunk_size=config.chunk_size,
            store_path=IMAGE_STORE_PATH,
            on_progress=None,
            ocr_engine=config.ocr_engine,
        )
        return _ok(
            status="success",
            file_count=report.file_count,
            chunk_count=report.chunk_count,
            index_path=report.index_path,
            session_id=session_id
        )
    except Exception as exc:
        return _error(str(exc))
    # Temp directories are now persistent session paths


@app.post("/api/image/search")
def image_search(req: ImageSearchRequest) -> dict[str, object]:
    query = req.query.strip()
    if not query:
        return _error("Query cannot be empty.")

    idx = get_index_status(IMAGE_STORE_PATH)
    if not idx.get("exists"):
        return _error(
            "No image index is ready. Index images before querying.",
            status_code=404,
        )

    try:
        results = search_image_chunks(
            query=query,
            top_k=req.top_k,
            store_path=IMAGE_STORE_PATH,
            retrieval_mode=req.retrieval_mode,
            bm25_weight=req.bm25_weight,
        )
        return _ok(results=results)
    except Exception as exc:
        return _error(str(exc))


@app.get("/api/vault")
def get_vault_data() -> dict[str, object]:
    config = _config()
    return _ok(chunks=load_index_metadata(config.faiss_path))


@app.post("/api/sessions/clear")
def clear_sessions() -> dict[str, object]:
    session_dir = Path("data/sessions")
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
            return _ok(message="Sessions cleared successfully")
        except Exception as e:
            return _error(f"Failed to clear sessions: {e}")
    return _ok(message="No sessions to clear")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
