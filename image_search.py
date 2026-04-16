from __future__ import annotations

import tempfile
from pathlib import Path
from tkinter import Tk, filedialog

import streamlit as st

from local_archive_ai.config import load_config, python_runtime_status
from local_archive_ai.services import (
    collect_image_files,
    index_image_documents,
    search_image_chunks,
)
from local_archive_ai.styles import CSS

IMAGE_STORE_PATH = "data/image_faiss_index"


st.set_page_config(
    page_title="Local-Archive AI Image Search",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


def _pick_folder_native() -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Select image folder to index")
    root.destroy()
    return selected


def _init_state() -> None:
    if "image_folder" not in st.session_state:
        st.session_state.image_folder = ""
    if "image_results" not in st.session_state:
        st.session_state.image_results = []
    if "image_report" not in st.session_state:
        st.session_state.image_report = None


def _runtime_banner() -> None:
    runtime = python_runtime_status()
    if runtime["supported"]:
        return
    st.warning(
        f"{runtime['message']} Image search is supported best on Python {runtime['supported_version']}."
    )


def _render_result_card(result: dict[str, object], rank: int) -> None:
    file_path = Path(str(result.get("file_path", "")))
    title = f"{rank}. {result.get('file_name', 'unknown')}"
    score = float(result.get("score", 0.0))
    bbox = result.get("bbox")
    blocks = result.get("block_ids", [])

    left, right = st.columns([1.2, 1])
    with left:
        if file_path.exists():
            st.image(str(file_path), use_container_width=True)
        else:
            st.info("Image preview unavailable.")
    with right:
        st.markdown(
            (
                "<div class='retrieval-card'>"
                f"<span class='score'>SIM: {score:.3f}</span>"
                f"<div><strong>{title}</strong></div>"
                f"<div class='mono'>{file_path}</div>"
                f"<div style='margin-top:0.4rem;'>{str(result.get('text', ''))[:900]}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if bbox:
            st.caption(f"Bounding box: {bbox}")
        if blocks:
            st.caption(f"Blocks: {', '.join(str(block) for block in blocks)}")


def main() -> None:
    _init_state()
    config = load_config()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.title("LOCAL-ARCHIVE AI IMAGE SEARCH")
    st.caption("Offline OCR block extraction, local embeddings, and semantic image retrieval.")
    _runtime_banner()

    with st.sidebar:
        st.markdown("## Image Index")
        pick_col, clear_col = st.columns([2, 1])
        if pick_col.button("BROWSE_FOLDER", use_container_width=True):
            selected = _pick_folder_native()
            if selected:
                st.session_state.image_folder = selected
        if clear_col.button("CLEAR", use_container_width=True):
            st.session_state.image_folder = ""

        st.code(st.session_state.image_folder or "No folder selected", language="text")
        uploads = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
        )

        if st.session_state.image_folder:
            folder = Path(st.session_state.image_folder)
            if folder.exists():
                preview = collect_image_files(folder)
                st.caption(f"Indexed candidates: {len(preview)} image files")

        if st.button("INDEX_IMAGES", use_container_width=True):
            if not st.session_state.image_folder and not uploads:
                st.error("Select a folder or upload images first.")
            else:
                tmp_dir: tempfile.TemporaryDirectory[str] | None = None
                try:
                    folder_to_index = st.session_state.image_folder
                    if uploads:
                        tmp_dir = tempfile.TemporaryDirectory(prefix="local_archive_image_")
                        for upload in uploads:
                            if upload.name:
                                (Path(tmp_dir.name) / upload.name).write_bytes(upload.getbuffer())
                        folder_to_index = tmp_dir.name

                    report = index_image_documents(
                        folder_path=folder_to_index,
                        chunk_size=config.chunk_size,
                        store_path=IMAGE_STORE_PATH,
                    )
                    st.session_state.image_report = report
                    st.success(
                        f"Indexed {report.file_count} images into {report.chunk_count} searchable chunks."
                    )
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    if tmp_dir is not None:
                        tmp_dir.cleanup()

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.markdown(
            (
                "<div class='landing-hero'>"
                "<div class='metric-head'>IMAGE SEMANTIC SEARCH</div>"
                "<h1>Search screenshots, scans, and image-heavy notes with local OCR chunks.</h1>"
                "<p>Each image is split into nearby text blocks, embedded locally, and retrieved with the same offline-first search stack as the main app.</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with top_right:
        report = st.session_state.image_report
        st.metric("Image Index Path", IMAGE_STORE_PATH)
        st.metric("Files Indexed", getattr(report, "file_count", 0) if report else 0)
        st.metric("Chunks Indexed", getattr(report, "chunk_count", 0) if report else 0)

    query = st.text_input("Search indexed images", placeholder="Find references to vector databases or security policy")
    controls_left, controls_right = st.columns([1, 3])
    top_k = controls_left.selectbox("Top-k", [3, 5, 8], index=1)
    retrieval_mode = controls_right.selectbox(
        "Retrieval mode",
        ["vector", "hybrid", "hybrid+rerank"],
        index=["vector", "hybrid", "hybrid+rerank"].index(config.retrieval_mode)
        if config.retrieval_mode in {"vector", "hybrid", "hybrid+rerank"}
        else 0,
    )

    if st.button("SEARCH IMAGES", use_container_width=True):
        if not query.strip():
            st.warning("Enter a search query first.")
        else:
            try:
                st.session_state.image_results = search_image_chunks(
                    query=query.strip(),
                    top_k=int(top_k),
                    store_path=IMAGE_STORE_PATH,
                    retrieval_mode=retrieval_mode,
                    bm25_weight=config.bm25_weight,
                )
            except Exception as exc:
                st.session_state.image_results = []
                st.error(str(exc))

    if not st.session_state.image_results:
        st.info("Index images, then run a query to inspect matching OCR chunks.")
    else:
        st.markdown("### Results")
        for rank, result in enumerate(st.session_state.image_results, start=1):
            _render_result_card(result, rank)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
