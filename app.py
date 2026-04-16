from __future__ import annotations

import argparse
import csv
import io
import html
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from local_archive_ai.config import AppConfig, load_config, python_runtime_status, save_config

# CLI argument parsing
parser = argparse.ArgumentParser(description="RAG Offline App")
parser.add_argument('--check-config', action='store_true', help='Validate config and exit')
args, unknown = parser.parse_known_args()

if args.check_config:
    try:
        config = load_config()
        print("Config validation successful.")
        exit(0)
    except Exception as e:
        print(f"Config validation failed: {e}")
        exit(1)
from local_archive_ai.services import (
    answer_query,
    check_ollama_status,
    collect_files,
    generate_with_ollama_stream,
    get_autocomplete_suggestions,
    get_index_status,
    get_ollama_status_message,
    index_documents,
    load_index_metadata,
    prewarm_ollama,
    runtime_mode,
    summarize_indexable_content,
    system_checks,
    vector_diagnostics,
)
from local_archive_ai.logging_config import log
from local_archive_ai.query_cache import QueryCache
from local_archive_ai.store import LocalVectorStore
from local_archive_ai.chat_engine import ChatEngine
from local_archive_ai.batch_processor import BatchProcessor
from local_archive_ai.multi_format_loader import MultiFormatLoader
from local_archive_ai.styles import CSS
from local_archive_ai.watcher import FolderWatcher


st.set_page_config(
    page_title="Local-Archive AI",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ----- st.cache_resource for heavy objects (Problem A) -----

@st.cache_resource(show_spinner=False)
def _get_query_cache() -> QueryCache:
    """Singleton query cache, persisted across reruns."""
    return QueryCache()


def _get_chat_engine(config: AppConfig) -> ChatEngine:
    """Get or create ChatEngine instance stored in session state."""
    if "chat_engine" not in st.session_state or st.session_state.chat_engine is None:
        st.session_state.chat_engine = ChatEngine(
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
    return st.session_state.chat_engine


@st.cache_resource(show_spinner=False)
def _get_folder_watcher() -> FolderWatcher:
    """Singleton folder watcher."""
    def _noop(paths: list) -> None:
        pass  # Will be replaced when watcher is started
    return FolderWatcher(callback=_noop)


@st.cache_data(show_spinner=False)
def _load_metadata_cached(metadata_path: str, mtime: float) -> list[dict[str, Any]]:
    del mtime
    return load_index_metadata(Path(metadata_path).parent)


def _section_panel(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='panel'><span class='metric-head'>{title}</span><br/>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def _render_runtime_warning() -> None:
    runtime = python_runtime_status()
    if runtime["supported"]:
        return
    st.warning(
        f"{runtime['message']} You can keep using the app here, but validation and production support target "
        f"Python {runtime['supported_version']}."
    )


def _render_retrieval_card(title: str, score: float | None, meta: str, body: str) -> None:
    score_html = f"<span class='score'>SIM: {score:.3f}</span>" if score is not None else ""
    st.markdown(
        (
            "<div class='retrieval-card'>"
            f"{score_html}"
            f"<div><strong>{title}</strong></div>"
            f"<div class='mono'>{meta}</div>"
            f"<div style='margin-top:0.4rem;'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "config": None,
        "chat_history": [],
        "last_index_report": None,
        "batch_results": [],
        "last_debug_payload": {},
        "selected_folder": "",
        "nav_page": "LANDING",
        "nav_pending": None,
        "pipeline_stage": "Document Intake",
        "ollama_prewarmed": False,
        "watcher_running": False,
        "chat_engine": None,
        "streaming_response": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state.config is None:
        st.session_state.config = load_config()

    # Pre-warm Ollama in background on first load (Problem O)
    if not st.session_state.ollama_prewarmed:
        st.session_state.ollama_prewarmed = True
        cfg = st.session_state.config
        thread = threading.Thread(
            target=prewarm_ollama,
            args=(cfg.ollama_endpoint, cfg.model_name, cfg.ollama_api_key),
            daemon=True,
        )
        thread.start()


def _goto(page: str) -> None:
    # Avoid mutating a widget-bound key after instantiation.
    # We set a pending nav request and apply it before the nav widget is created.
    st.session_state.nav_pending = page
    st.rerun()


def _pick_folder_native() -> str:
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select folder to index")
        root.destroy()
        return selected
    except Exception:
        return ""


def render_sidebar(config: AppConfig) -> AppConfig:
    st.sidebar.markdown("## LOCAL-ARCHIVE AI")
    st.sidebar.caption("Offline RAG Dashboard")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Indexing Engine")
    pick_col, clear_col = st.sidebar.columns([2, 1])
    if pick_col.button("BROWSE_FOLDER", use_container_width=True):
        selected = _pick_folder_native()
        if selected:
            st.session_state.selected_folder = selected
    if clear_col.button("CLEAR", use_container_width=True):
        st.session_state.selected_folder = ""

    st.sidebar.code(st.session_state.selected_folder or "No folder selected", language="text")

    archive_zip = st.sidebar.file_uploader(
        "Drag-drop ZIP archive",
        type=["zip"],
        accept_multiple_files=False,
        help="Upload a ZIP containing documents. It will be extracted locally for indexing.",
    )
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs for direct indexing.",
    )

    if st.session_state.selected_folder:
        selected_path = Path(st.session_state.selected_folder)
        if selected_path.exists():
            total_files = len([p for p in selected_path.rglob("*") if p.is_file()])
            supported_files = len(collect_files(selected_path))
            st.sidebar.caption(
                f"Preview: {supported_files} supported files / {total_files} total files"
            )
            summary = summarize_indexable_content(selected_path)
            st.sidebar.caption("Top file types")
            for ext, count in summary["top_extensions"].items():
                st.sidebar.caption(f"- {ext}: {count}")

    progress_slot = st.sidebar.empty()
    file_slot = st.sidebar.empty()
    chunk_slot = st.sidebar.empty()

    if st.sidebar.button("INDEX_FILES", use_container_width=True):
        if (
            archive_zip is None
            and not uploaded_pdfs
            and not st.session_state.selected_folder.strip()
        ):
            st.sidebar.error("Select a local folder, upload PDFs, or upload a ZIP archive.")
        else:
            progress_slot.progress(0.0, text="Preparing indexing...")

            def on_progress(update: dict) -> None:
                pct = float(update.get("progress", 0.0))
                progress_slot.progress(pct, text=f"Indexing... {int(pct * 100)}%")
                file_slot.caption(f"Current file: {update.get('current_file', 'N/A')}")
                chunk_slot.caption(f"Chunks indexed: {update.get('chunk_count', 0)}")

            try:
                folder_to_index = st.session_state.selected_folder
                tmp_dir: tempfile.TemporaryDirectory[str] | None = None
                if archive_zip is not None:
                    tmp_dir = tempfile.TemporaryDirectory(prefix="local_archive_zip_")
                    with zipfile.ZipFile(archive_zip) as zf:
                        zf.extractall(tmp_dir.name)
                    folder_to_index = tmp_dir.name
                elif uploaded_pdfs:
                    tmp_dir = tempfile.TemporaryDirectory(prefix="local_archive_pdf_")
                    for file in uploaded_pdfs:
                        file_path = Path(tmp_dir.name) / file.name
                        file_path.write_bytes(file.getbuffer())
                    folder_to_index = tmp_dir.name

                report = index_documents(
                    folder_path=folder_to_index,
                    chunk_size=config.chunk_size,
                    store_path=config.faiss_path,
                    on_progress=on_progress,
                )
                st.session_state.last_index_report = report
                st.sidebar.success(
                    f"Index complete: {report.file_count} files / {report.chunk_count} chunks."
                )
            except Exception as exc:
                st.sidebar.error(str(exc))
            finally:
                if "tmp_dir" in locals() and tmp_dir is not None:
                    tmp_dir.cleanup()

    left, right = st.sidebar.columns(2)
    if left.button("CLEAR CHAT", use_container_width=True):
        st.session_state.chat_history = []
    if right.button("RE-INDEX", use_container_width=True):
        st.rerun()

    with st.sidebar.expander("Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=500,
            max_value=1000,
            value=int(config.chunk_size),
            step=50,
        )
        top_k = st.selectbox("Top-k", options=[2, 3, 4, 5, 6, 8], index=[2, 3, 4, 5, 6, 8].index(config.top_k) if config.top_k in [2, 3, 4, 5, 6, 8] else 2)
        model_name = st.text_input("Model name", value=config.model_name)
        ollama_endpoint = st.text_input("Ollama endpoint", value=config.ollama_endpoint)
        faiss_path = st.text_input("FAISS path", value=config.faiss_path)
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=["vector", "hybrid", "hybrid+rerank"],
            index=["vector", "hybrid", "hybrid+rerank"].index(config.retrieval_mode)
            if config.retrieval_mode in {"vector", "hybrid", "hybrid+rerank"}
            else 0,
        )
        ocr_engine = st.selectbox(
            "OCR engine",
            options=["tesseract", "easyocr"],
            index=["tesseract", "easyocr"].index(config.ocr_engine)
            if config.ocr_engine in {"tesseract", "easyocr"}
            else 0,
        )
        bm25_weight = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=float(config.bm25_weight), step=0.05)
        rerank_top_n = st.slider("Rerank top N", min_value=1, max_value=20, value=int(config.rerank_top_n), step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=float(config.temperature), step=0.1)
        confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=float(config.confidence_threshold), step=0.05)
        debug_mode = st.checkbox("Debug mode", value=config.debug_mode)
        enable_watcher = st.checkbox("Enable folder watcher", value=config.enable_watcher)
        if st.button("Save settings", use_container_width=True):
            config = config.model_copy(update={
                "chunk_size": int(chunk_size),
                "top_k": int(top_k),
                "model_name": model_name.strip() or config.model_name,
                "ollama_endpoint": ollama_endpoint.strip() or config.ollama_endpoint,
                "faiss_path": faiss_path.strip() or config.faiss_path,
                "retrieval_mode": retrieval_mode,
                "ocr_engine": ocr_engine,
                "bm25_weight": float(bm25_weight),
                "rerank_top_n": int(rerank_top_n),
                "debug_mode": debug_mode,
                "temperature": float(temperature),
                "confidence_threshold": float(confidence_threshold),
                "enable_watcher": enable_watcher,
            })
            save_config(config)
            st.session_state.config = config
            st.success("Settings saved to config.yaml")
            log.info("Settings saved via UI")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Vitals")
    mode = runtime_mode()
    ollama_status = get_ollama_status_message(config.ollama_endpoint, config.ollama_api_key)
    idx = get_index_status(config.faiss_path)

    st.sidebar.caption(f"Mode: {mode['mode']}")
    st.sidebar.caption(f"VRAM: {mode['vram']}")
    st.sidebar.caption(f"GPU Utilization: {mode.get('gpu_util', 'N/A')}")
    st.sidebar.caption(f"GPU Temperature: {mode.get('gpu_temp', 'N/A')}")
    if ollama_status["running"]:
        st.sidebar.caption("Ollama: ACTIVE")
    else:
        st.sidebar.error(ollama_status["message"])
        st.sidebar.caption(f"Download: {ollama_status['download_url']}")
    st.sidebar.caption(
        f"FAISS: {'LOADED' if idx.get('exists') else 'NOT READY'}"
    )

    with st.sidebar.expander("System Check", expanded=False):
        checks = system_checks(
            config.faiss_path,
            config.model_name,
            endpoint=config.ollama_endpoint,
            api_key=config.ollama_api_key,
        )
        st.markdown(
            f"- Tesseract: {'OK' if checks['tesseract_installed'] else 'MISSING'}\n"
            f"- Plotly: {'OK' if checks['plotly_available'] else 'MISSING'}\n"
            f"- FAISS Loaded: {'YES' if checks['faiss_loaded'] else 'NO'}\n"
            f"- Ollama Reachable: {'YES' if checks['ollama_reachable'] else 'NO'}\n"
            f"- Ollama Model Available: {'YES' if checks['ollama_model_available'] else 'NO'}"
        )
        st.caption(f"FAISS detail: {checks['faiss_message']}")

    # Index management tools (Problem S, T)
    with st.sidebar.expander("Index Management", expanded=False):
        store = LocalVectorStore(Path(config.faiss_path))
        export_col, import_col = st.columns(2)
        if export_col.button("Export Index", use_container_width=True):
            try:
                dest = Path(config.faiss_path) / "export.zip"
                store.load()
                store.export_index(dest)
                st.success(f"Exported to {dest}")
            except Exception as exc:
                st.error(str(exc))
        uploaded_index = import_col.file_uploader("Import .zip", type=["zip"], key="import_index_zip")
        if uploaded_index is not None:
            import_path = Path(config.faiss_path) / "_import.zip"
            import_path.parent.mkdir(parents=True, exist_ok=True)
            import_path.write_bytes(uploaded_index.getbuffer())
            if store.import_index(import_path):
                st.success("Index imported!")
            else:
                st.error("Import failed.")
            import_path.unlink(missing_ok=True)

        # Checkpoints (Problem T)
        checkpoints = store.list_checkpoints()
        if checkpoints:
            st.caption(f"{len(checkpoints)} checkpoint(s) available")
            if st.button("Rollback to latest checkpoint", use_container_width=True):
                if store.rollback_checkpoint(checkpoints[0]):
                    st.success("Rolled back!")
                else:
                    st.error("Rollback failed.")

    return config


def render_status_card(config: AppConfig) -> dict[str, Any]:
    idx = get_index_status(config.faiss_path)
    if idx["exists"]:
        st.markdown(
            (
                "<div class='archive-banner'>"
                "<div style='display:flex;justify-content:space-between;gap:1rem;align-items:center;'>"
                "<div>"
                "<h3 style='margin:0;'>INDEX ACTIVE</h3>"
                f"<div class='mono'>Dataset: {idx['file_count']} documents | {idx['chunk_count']} chunks parsed</div>"
                "</div>"
                f"<div style='font-size:2rem;font-weight:700;'>{idx['chunk_count']}</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                "<div class='archive-banner warning'>"
                "<h3 style='margin:0;'>NO INDEX FOUND</h3>"
                "<div class='mono'>Use the sidebar indexing workflow to initialize local retrieval.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    return idx


def render_landing_page(config: AppConfig) -> None:
    idx = get_index_status(config.faiss_path)
    checks = system_checks(
        config.faiss_path,
        config.model_name,
        endpoint=config.ollama_endpoint,
        api_key=config.ollama_api_key,
    )
    mode = runtime_mode()
    st.markdown(
        (
            "<div class='landing-hero'>"
            "<div class='metric-head'>LOCAL-FIRST RAG PLATFORM</div>"
            "<h1>Build, Search, and Debug your Private Knowledge Pipeline</h1>"
            "<p>Index local documents, retrieve context with semantic search, and inspect the full RAG flow in real time.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    cta1, cta2, cta3 = st.columns(3)
    if cta1.button("OPEN CHAT ENGINE", use_container_width=True):
        _goto("CHAT_ENGINE")
    if cta2.button("OPEN PIPELINE VIEW", use_container_width=True):
        _goto("PIPELINE")
    if cta3.button("OPEN DEBUG LOGS", use_container_width=True):
        _goto("DEBUG_LOGS")

    st.markdown("#### Platform Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Docs", int(idx.get("file_count", 0)))
    m2.metric("Total Chunks", int(idx.get("chunk_count", 0)))
    m3.metric("Runtime Mode", mode.get("mode", "N/A"))
    m4.metric("Ollama", "Reachable" if checks.get("ollama_reachable") else "Offline")

    st.markdown("#### Core Features")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            (
                "<div class='feature-card'>"
                "<h4>Indexing Engine</h4>"
                "<p>Ingest folders, ZIP archives, or PDFs and build a local vector store for offline retrieval.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='feature-card'>"
                "<h4>Chat Engine</h4>"
                "<p>Ask questions against your indexed corpus with citation-backed answers from local context.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            (
                "<div class='feature-card'>"
                "<h4>Debug Visibility</h4>"
                "<p>Inspect embedding previews, retrieval scores, and the generated prompt payload.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='feature-card'>"
                "<h4>Batch Processing</h4>"
                "<p>Run multiple prompts in sequence and export structured outputs as CSV.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_pipeline_view(config: AppConfig) -> None:
    idx = get_index_status(config.faiss_path)
    checks = system_checks(
        config.faiss_path,
        config.model_name,
        endpoint=config.ollama_endpoint,
        api_key=config.ollama_api_key,
    )
    diagnostics = vector_diagnostics(config.faiss_path, sample_size=300)
    stages = [
        "Document Intake",
        "Parsing/Chunking",
        "Embedding",
        "Vector Store",
        "Retrieval",
        "LLM Response",
    ]

    st.markdown(
        "<div class='pipeline-wrap'><div class='metric-head'>INTERACTIVE RAG PIPELINE</div>"
        "<h3 style='margin-top:0.25rem;'>End-to-end flow from local documents to answer generation</h3></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        is_active = st.session_state.pipeline_stage == stage
        btn_label = f"{i+1}. {stage}"
        if cols[i].button(btn_label, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.pipeline_stage = stage
            st.rerun()

    st.markdown("<div class='pipeline-connector'></div>", unsafe_allow_html=True)

    active = st.session_state.pipeline_stage
    detail = {
        "Document Intake": "Collect files from folder selection, ZIP extraction, or direct PDF upload in the sidebar.",
        "Parsing/Chunking": f"Documents are split into context chunks using token-aware splitting. Current chunk size: {config.chunk_size}.",
        "Embedding": "Each chunk is embedded locally (SentenceTransformers) and normalized for cosine similarity search.",
        "Vector Store": f"Index stored at `{config.faiss_path}` with status: {'READY' if idx.get('exists') else 'NOT READY'}.",
        "Retrieval": f"Top-K retrieval uses semantic nearest-neighbor search. Current Top-K: {config.top_k}.",
        "LLM Response": f"Prompt is composed with citations and sent to model `{config.model_name}` via local Ollama.",
    }

    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            (
                "<div class='pipeline-detail'>"
                f"<h4 style='margin:0 0 .4rem 0;'>{active}</h4>"
                f"<p style='margin:0;'>{detail[active]}</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            (
                "<div class='pipeline-kpis'>"
                f"<div><strong>Index:</strong> {'Loaded' if idx.get('exists') else 'Missing'}</div>"
                f"<div><strong>Vectors:</strong> {diagnostics.get('vector_count', 0)}</div>"
                f"<div><strong>Dimension:</strong> {diagnostics.get('dim', 0)}</div>"
                f"<div><strong>Ollama:</strong> {'Reachable' if checks.get('ollama_reachable') else 'Offline'}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Pipeline stage guide", expanded=False):
        for stage in stages:
            st.markdown(f"- **{stage}**: {detail[stage]}")


def render_chat(config: AppConfig) -> None:
    if not st.session_state.chat_history:
        _section_panel(
            "CHAT ENGINE",
            "No queries yet. Ask a question to retrieve context from your local index.",
        )
        return

    max_messages = st.slider(
        "Messages to render",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Reduce render cost for large chat histories.",
    )
    visible_history = st.session_state.chat_history[-max_messages:]

    for item in visible_history:
        st.markdown(f"<div class='user-bubble'>{item['query']}</div>", unsafe_allow_html=True)
        safe_answer = html.escape(str(item["answer"])).replace("\n", "<br/>")
        st.markdown(
            f"<div class='assistant-bubble'>{safe_answer}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Retrieved Sources ({len(item.get('citations', []))})", expanded=False):
            for i, citation in enumerate(item.get("citations", []), start=1):
                _render_retrieval_card(
                    title=f"{i}. {citation.get('file_name', 'unknown')}",
                    score=float(citation.get("score", 0)),
                    meta=f"Page: {citation.get('source_page', 'N/A')} | Chunk: {citation.get('chunk_index', 'N/A')}",
                    body=str(citation.get("chunk_text", ""))[:420],
                )
                uri = citation.get("open_uri", "")
                if uri:
                    st.caption(f"[Open original file]({uri})")


def render_query_form(config: AppConfig) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval Mode", config.retrieval_mode.upper())
    c2.metric("Top-K", int(config.top_k))
    c3.metric("Model", config.model_name)
    c4.metric("Confidence", f"{config.confidence_threshold:.0%}")

    # Show conversation memory status
    engine = _get_chat_engine(config)
    memory = engine.get_memory()
    if memory:
        st.caption(f"Conversation memory: {len(memory)} turn(s) (last 5 kept as context)")

    # Problem D: use st.form to prevent premature queries on every keystroke
    with st.form("query_form", clear_on_submit=True, border=False):
        query = st.text_area(
            "Enter query for local archive",
            height=84,
            placeholder="What are the key findings about compiler optimization?",
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        submitted = c1.form_submit_button("SEND  >", use_container_width=True)
        stream_mode = c2.form_submit_button("STREAM  >>>", use_container_width=True)
        clear = c3.form_submit_button("CLEAR CHAT", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        engine.clear_memory()
        st.session_state.chat_engine = None  # Reset engine
        return

    if submitted or stream_mode:
        if not query.strip():
            st.warning("Please enter a query.")
            return
        ollama_ok = check_ollama_status(config.ollama_endpoint, config.ollama_api_key)
        if not ollama_ok:
            status_info = get_ollama_status_message(config.ollama_endpoint, config.ollama_api_key)
            st.error(status_info["message"])
            return

        if stream_mode:
            # Streaming mode using ChatEngine
            st.markdown(f"<div class='user-bubble'>{query.strip()}</div>", unsafe_allow_html=True)
            response_container = st.empty()
            full_response = ""
            try:
                for token in engine.query_stream(query.strip()):
                    full_response += token
                    response_container.markdown(full_response)
            except Exception as exc:
                st.error(str(exc))
                return

            # Get citations from last turn
            last = engine.last_turn()
            citations = last.citations if last else []

            st.session_state.chat_history.append(
                {
                    "query": query.strip(),
                    "answer": full_response,
                    "citations": citations,
                    "debug": {},
                    "low_confidence": last.low_confidence if last else False,
                    "duration_ms": 0,
                }
            )
        else:
            # Non-streaming mode using ChatEngine with cache support
            cache = _get_query_cache()
            cached = None
            if config.query_cache_ttl > 0:
                cached = cache.get(query.strip(), config.top_k, config.retrieval_mode, config.faiss_path)

            if cached is not None:
                st.caption("(cached result)")
                payload = cached
                st.session_state.chat_history.append(
                    {
                        "query": query.strip(),
                        "answer": payload["answer_markdown"],
                        "citations": payload["citations"],
                        "debug": payload.get("debug_payload", {}),
                        "low_confidence": payload.get("low_confidence", False),
                        "duration_ms": payload.get("duration_ms", 0),
                    }
                )
            else:
                # Use ChatEngine for query (includes memory context)
                with st.status("Processing query...", expanded=True) as status:
                    st.write("Retrieving context from index...")
                    try:
                        response = engine.query(query.strip())
                        st.write("Generating response...")
                        status.update(label=f"Done! ({response.duration_ms}ms)", state="complete")
                    except Exception as exc:
                        status.update(label="Error", state="error")
                        st.error(str(exc))
                        return

                # Build payload for cache and history
                payload = {
                    "answer_markdown": response.answer,
                    "citations": response.citations,
                    "debug_payload": response.debug_payload,
                    "low_confidence": response.low_confidence,
                    "duration_ms": response.duration_ms,
                    "max_similarity": response.max_similarity,
                }

                # Cache the result
                if config.query_cache_ttl > 0:
                    cache.put(query.strip(), config.top_k, config.retrieval_mode, config.faiss_path, payload)

                st.session_state.chat_history.append(
                    {
                        "query": query.strip(),
                        "answer": response.answer,
                        "citations": response.citations,
                        "debug": response.debug_payload,
                        "low_confidence": response.low_confidence,
                        "duration_ms": response.duration_ms,
                    }
                )
                st.session_state.last_debug_payload = response.debug_payload


def render_debug_logs(config: AppConfig) -> None:
    debug = st.session_state.last_debug_payload or {}
    if not debug:
        debug = {
            "embedding_preview": [0.0142, -0.0809, 0.0456, 0.1102, -0.0673, 0.0234, -0.0911, 0.0543],
            "retrieved_chunks": [
                {
                    "file_name": "ARCHIVE_POLICY_2024.PDF",
                    "chunk_index": "AF-88-01",
                    "source_page": 88,
                    "score": 0.892,
                    "text_preview": "All localized vector storage must be encrypted using AES-256 standard before being committed to the permanent archive.",
                },
                {
                    "file_name": "SECURITY_MANIFEST_V2.TXT",
                    "chunk_index": "SM-12-99",
                    "source_page": 12,
                    "score": 0.845,
                    "text_preview": "Encryption keys are managed via the hardware security module (HSM) and are never exposed to the LLM context window.",
                },
                {
                    "file_name": "USER_GUIDE_TECHNICAL.MD",
                    "chunk_index": "UG-04-45",
                    "source_page": 4,
                    "score": 0.791,
                    "text_preview": "Users should verify that the Local-Archive AI instance has read access to the directory containing sensitive documents.",
                },
                {
                    "file_name": "ARCHITECTURE_WHITEPAPER.DOCX",
                    "chunk_index": "AI-01-12",
                    "source_page": 1,
                    "score": 0.712,
                    "text_preview": "The RAG retrieval pipeline employs a semantic reranker to prioritize chunks that align with the user's intent.",
                },
            ],
            "prompt_text": "[SYSTEM]\nYou are a highly specialized AI assistant focused on technical document retrieval.",
        }
    diagnostics = vector_diagnostics(config.faiss_path, sample_size=300)
    overview_tab, vault_tab, vector_tab = st.tabs(["Trace Overview", "Vault", "Vector Lab"])

    with overview_tab:
        st.markdown(
            (
                "<div class='debug-title-row'>"
                "<div><h2 style='margin:0;'>DEBUG MODE</h2><div class='mono'>SESSION TRACE + RETRIEVAL INSPECTOR</div></div>"
                "<div class='debug-actions'>"
                "<span class='ghost-btn'>LOCAL ONLY</span>"
                "<span class='solid-btn'>TRACE ACTIVE</span>"
                "</div></div>"
            ),
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Retrieval Mode", config.retrieval_mode.upper())
        s2.metric("Rerank Top N", int(config.rerank_top_n))
        s3.metric("Vector Count", diagnostics.get("vector_count", 0))

        emb = debug.get("embedding_preview", [])
        emb_text = ", ".join(f"{float(x):.4f}" for x in emb[:16])
        st.markdown(
            (
                "<div class='debug-embed-box'>"
                "<div class='metric-head'>QUERY EMBEDDING PREVIEW</div>"
                "<div class='embed-inline mono'>"
                f"[ {emb_text} ]"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        retrieved = debug.get("retrieved_chunks", [])[:6]
        st.markdown(
            f"<div class='retrieval-head'>RETRIEVED CHUNKS (K={len(retrieved)})</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        for idx, item in enumerate(retrieved):
            col = col1 if idx % 2 == 0 else col2
            with col:
                st.markdown(
                    (
                        "<div class='debug-chunk-card'>"
                        f"<div class='debug-score'>SIM: {float(item.get('score', 0)):.3f}</div>"
                        f"<div class='chunk-title'>{str(item.get('file_name', 'unknown'))}</div>"
                        f"<div class='chunk-meta mono'>CHUNK_ID: {item.get('chunk_index', 'N/A')}</div>"
                        f"<div class='chunk-body'>\"...{str(item.get('text_preview', ''))}...\"</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='prompt-head'>FULL PROMPT SENT TO LLM</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='prompt-panel mono'>{str(debug.get('prompt_text', ''))[:2000]}</div>",
            unsafe_allow_html=True,
        )

    with vault_tab:
        render_vault_view(config)

    with vector_tab:
        render_vector_lab(config)


def _run_batch_query(queries: list[str], config: AppConfig) -> None:
    """Run batch queries with incremental display (Problem CC) and progress (Problem B)."""
    st.session_state.batch_results = []
    progress = st.progress(0.0, text="Queue initialized")
    results_container = st.container()

    for idx, query in enumerate(queries, start=1):
        row: dict[str, Any] = {
            "id": f"B{idx:03d}",
            "query": query,
            "status": "IN_PROGRESS",
            "progress": 0,
            "answer": "",
            "target": "LOCAL_ARCHIVE",
        }
        st.session_state.batch_results.append(row)
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
                confidence_threshold=config.confidence_threshold,
                temperature=config.temperature,
                max_context_tokens=config.max_context_tokens,
            )
            row["answer"] = payload.get("answer_markdown", "")
            row["status"] = "COMPLETED"
            row["progress"] = 100
        except Exception as exc:
            row["answer"] = str(exc)
            row["status"] = "FAILED"
            row["progress"] = 100

        # Incremental display (Problem CC)
        with results_container:
            st.caption(f"{row['id']}: {row['status']} - {query[:60]}")

        progress.progress(idx / len(queries), text=f"Processed {idx}/{len(queries)} queries")


def render_batch_queue(config: AppConfig) -> None:
    _section_panel(
        "BATCH QUERIES",
        "Run multiple archive prompts in sequence and export responses as CSV. "
        "You can also upload a CSV file with a 'query' column.",
    )

    # CSV upload for batch queries (Problem Q)
    csv_upload = st.file_uploader("Upload CSV (column: 'query')", type=["csv"], key="batch_csv_upload")

    queries_raw = st.text_area(
        "Input raw queries (one per line)",
        height=170,
        placeholder="Summarize documents in /archive/internal/reports\nExtract all expiration dates from service_contracts.pdf",
    )
    c1, c2 = st.columns(2)
    run = c1.button("RUN BATCH", use_container_width=True)
    clear = c2.button("CLEAR BUFFER", use_container_width=True)

    if clear:
        st.session_state.batch_results = []
        st.rerun()

    if run:
        queries: list[str] = []
        # Parse from CSV upload first
        if csv_upload is not None:
            try:
                import pandas as pd
                df = pd.read_csv(csv_upload)
                if "query" in df.columns:
                    queries = [str(q).strip() for q in df["query"].dropna().tolist() if str(q).strip()]
                else:
                    st.warning("CSV must contain a 'query' column.")
            except Exception as exc:
                st.error(f"Failed to parse CSV: {exc}")
        # Then add text area queries
        queries.extend([line.strip() for line in queries_raw.splitlines() if line.strip()])
        if not queries:
            st.warning("Add at least one query.")
        else:
            with st.status(f"Processing {len(queries)} queries...", expanded=True) as batch_status:
                _run_batch_query(queries, config)
                batch_status.update(label="Batch complete!", state="complete")

    rows = st.session_state.batch_results
    st.markdown("#### Active Process Monitor")
    if not rows:
        st.caption("No batch jobs executed yet.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Queued", len(rows))
    m2.metric("Completed", sum(1 for row in rows if row["status"] == "COMPLETED"))
    m3.metric("Failed", sum(1 for row in rows if row["status"] == "FAILED"))

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "id": row["id"],
                "query_directive": row["query"][:52],
                "target_data": row["target"],
                "status": row["status"],
                "progress": f"{row['progress']}%",
            }
        )
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "query_directive": st.column_config.TextColumn("QUERY_DIRECTIVE", width="large"),
            "target_data": st.column_config.TextColumn("TARGET_DATA", width="medium"),
            "status": st.column_config.TextColumn("STATUS", width="small"),
            "progress": st.column_config.TextColumn("PROGRESS", width="small"),
        },
    )

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["id", "query", "status", "answer"])
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "id": row["id"],
                "query": row["query"],
                "status": row["status"],
                "answer": row["answer"],
            }
        )
    st.download_button(
        "DOWNLOAD RESULTS AS CSV",
        data=csv_buf.getvalue(),
        file_name="batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_vault_view(config: AppConfig) -> None:
    _section_panel(
        "THE_DIGITAL_VAULT",
        "Browse indexed source chunks and inspect extracted text directly from your local vector store.",
    )
    metadata_file = Path(config.faiss_path) / "metadata.json"
    if not metadata_file.exists():
        st.info("No index metadata found yet. Index documents first from the sidebar.")
        return

    metadata = _load_metadata_cached(str(metadata_file), metadata_file.stat().st_mtime)

    files = sorted({item.get("file_name", "unknown") for item in metadata})
    if not files:
        st.info("Indexed metadata is empty.")
        return

    left, right = st.columns([2, 1])
    selected = left.selectbox("Select indexed document", options=files)
    search = right.text_input("Search within", value="")

    filtered = [m for m in metadata if m.get("file_name") == selected]
    if search.strip():
        needle = search.strip().lower()
        filtered = [m for m in filtered if needle in str(m.get("text", "")).lower()]

    st.caption(f"Chunks: {len(filtered)}")
    page_size = st.select_slider("Chunks per page", options=[6, 12, 18, 24], value=12)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page_no = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page_no) - 1) * page_size
    subset = filtered[start : start + page_size]

    for item in subset:
        _render_retrieval_card(
            title=f"Page {item.get('source_page', 'N/A')} · Chunk #{item.get('chunk_index', 'N/A')}",
            score=None,
            meta=str(item.get("file_name", "unknown")),
            body=str(item.get("text", ""))[:900],
        )
        file_path = str(item.get("file_path", "")).strip()
        if file_path:
            st.caption(f"[Open source file]({Path(file_path).resolve().as_uri()})")


def render_vector_lab(config: AppConfig) -> None:
    _section_panel(
        "VECTOR LAB",
        "Visualize embedding distribution, vector DB shape, and local retrieval traces.",
    )
    sample_size = st.slider("Vector sample size", min_value=100, max_value=1500, value=500, step=100)
    diagnostics = vector_diagnostics(config.faiss_path, sample_size=sample_size)
    if not diagnostics.get("ready"):
        st.info(diagnostics.get("message", "No vector diagnostics available."))
        return

    c1, c2 = st.columns(2)
    c1.metric("Vector Count", diagnostics.get("vector_count", 0))
    c2.metric("Embedding Dimension", diagnostics.get("dim", 0))

    points = diagnostics.get("points", [])
    if points:
        if go is not None:
            fig = go.Figure(
                data=[
                    go.Scattergl(
                        x=[p["x"] for p in points],
                        y=[p["y"] for p in points],
                        mode="markers",
                        marker={"size": 7, "opacity": 0.75},
                        text=[
                            f"{p['file_name']} | p={p.get('source_page')} | c={p.get('chunk_index')}"
                            for p in points
                        ],
                        hovertemplate="%{text}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                title="Embedding Projection (PCA-2D)",
                xaxis_title="PC-1",
                yaxis_title="PC-2",
                height=420,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Plotly not installed, using lightweight fallback chart.")
            st.scatter_chart(
                data=[{"x": p["x"], "y": p["y"]} for p in points],
                x="x",
                y="y",
                use_container_width=True,
            )

    debug = st.session_state.last_debug_payload or {}
    st.markdown("#### Local Retrieval Process")
    if not debug:
        st.caption("Run at least one query to see retrieval pipeline traces.")
        return

    emb = debug.get("embedding_preview", [])
    st.code(f"Embedding preview (first dims): {emb}", language="text")

    retrieved = debug.get("retrieved_chunks", [])
    if retrieved:
        if go is not None:
            bar = go.Figure(
                data=[
                    go.Bar(
                        x=[f"{r.get('file_name', 'unknown')}#{r.get('chunk_index', '?')}" for r in retrieved],
                        y=[float(r.get("score", 0)) for r in retrieved],
                    )
                ]
            )
            bar.update_layout(
                title="Top-K Similarity Scores",
                xaxis_title="Chunk",
                yaxis_title="Similarity",
                height=340,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
            )
            st.plotly_chart(bar, use_container_width=True)
        else:
            fallback = {
                f"{r.get('file_name', 'unknown')}#{r.get('chunk_index', '?')}": float(r.get("score", 0))
                for r in retrieved
            }
            st.bar_chart(fallback, use_container_width=True)

    with st.expander("Prompt used for local generation", expanded=False):
        st.code(debug.get("prompt_text", ""), language="text")


# ----- Keyboard shortcuts (Problem E) -----
_KEYBOARD_SHORTCUTS_JS = """
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl+K: focus search / query input
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        const textareas = document.querySelectorAll('textarea');
        if (textareas.length > 0) textareas[0].focus();
    }
    // Ctrl+Shift+C: clear chat (triggers the clear chat button)
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
            if (btn.textContent.trim().includes('CLEAR CHAT')) {
                btn.click();
                break;
            }
        }
    }
});
</script>
"""


def main() -> None:
    _init_state()
    config = st.session_state.config
    config = render_sidebar(config)
    st.session_state.config = config

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

    # Inject keyboard shortcuts (Problem E)
    st.components.v1.html(_KEYBOARD_SHORTCUTS_JS, height=0)

    head_left, head_right = st.columns([2, 1])
    head_left.title("LOCAL-ARCHIVE AI")
    head_right.markdown(
        "<div style='text-align:right; padding-top:0.8rem;'>"
        "<span class='system-pill'>SYSTEM_READY</span></div>",
        unsafe_allow_html=True,
    )

    _render_runtime_warning()

    idx = render_status_card(config)
    chips = [
        f"<span class='mini-stat'>DOCS: {idx.get('file_count', 0)}</span>",
        f"<span class='mini-stat'>CHUNKS: {idx.get('chunk_count', 0)}</span>",
        f"<span class='mini-stat'>TOP_K: {config.top_k}</span>",
        f"<span class='mini-stat'>MODEL: {config.model_name}</span>",
    ]
    st.markdown("".join(chips), unsafe_allow_html=True)

    nav_options = ["LANDING", "PIPELINE", "CHAT_ENGINE", "DEBUG_LOGS", "BATCH_QUEUE"]
    if st.session_state.nav_page not in nav_options:
        st.session_state.nav_page = "LANDING"

    # Apply pending navigation BEFORE widget instantiation.
    pending = st.session_state.get("nav_pending")
    if pending in nav_options:
        st.session_state.nav_page = pending
    st.session_state.nav_pending = None

    page = st.radio(
        "Navigation",
        options=nav_options,
        horizontal=True,
        label_visibility="collapsed",
        key="nav_page",
    )
    if page == "LANDING":
        render_landing_page(config)
    elif page == "PIPELINE":
        render_pipeline_view(config)
    elif page == "CHAT_ENGINE":
        render_query_form(config)
        render_chat(config)
    elif page == "DEBUG_LOGS":
        render_debug_logs(config)
    else:
        render_batch_queue(config)

    st.caption(f"Config file: {Path('config.yaml').resolve()}")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
