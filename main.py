"""Local-Archive AI -- clean entry point.

Usage:
    streamlit run main.py
    python main.py --check-config      # validate config.yaml and exit
    python main.py --health             # print health-check JSON and exit
    python main.py --chat "question"    # CLI chat query
    python main.py --batch-csv in.csv   # batch process CSV queries
    python main.py --index /path/to/dir # index a directory
    python main.py --load-repo URL      # index a Git repository
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from local_archive_ai.logging_config import log


def _health_check() -> dict:
    """Return a JSON-serialisable health report (Problem DD)."""
    from local_archive_ai.config import load_config
    from local_archive_ai.services import check_ollama_status, get_index_status, runtime_mode

    cfg = load_config()
    idx = get_index_status(cfg.faiss_path)
    mode = runtime_mode()
    return {
        "status": "ok",
        "index_loaded": idx.get("exists", False),
        "index_size": idx.get("chunk_count", 0),
        "file_count": idx.get("file_count", 0),
        "model_loaded": check_ollama_status(cfg.ollama_endpoint, cfg.ollama_api_key),
        "runtime_mode": mode.get("mode", "unknown"),
        "faiss_path": cfg.faiss_path,
        "model_name": cfg.model_name,
        "modules": {
            "chat_engine": True,
            "batch_processor": True,
            "multi_format_loader": True,
        },
    }


def _run_chat(question: str) -> None:
    """Run a single chat query from CLI."""
    from local_archive_ai.config import load_config
    from local_archive_ai.chat_engine import ChatEngine

    cfg = load_config()
    engine = ChatEngine(
        store_path=cfg.faiss_path,
        model_name=cfg.model_name,
        ollama_endpoint=cfg.ollama_endpoint,
        ollama_api_key=cfg.ollama_api_key,
        top_k=cfg.top_k,
        retrieval_mode=cfg.retrieval_mode,
        bm25_weight=cfg.bm25_weight,
        rerank_top_n=cfg.rerank_top_n,
        confidence_threshold=cfg.confidence_threshold,
        temperature=cfg.temperature,
        max_context_tokens=cfg.max_context_tokens,
    )
    response = engine.query(question)
    print(f"\nAnswer ({response.duration_ms}ms):\n{response.answer}")
    if response.citations:
        print(f"\nSources ({len(response.citations)}):")
        for c in response.citations:
            print(f"  - {c['file_name']} (score: {c['score']:.3f})")


def _run_batch_csv(csv_path: str) -> None:
    """Run batch query processing from a CSV file."""
    from local_archive_ai.config import load_config
    from local_archive_ai.batch_processor import BatchProcessor

    cfg = load_config()
    processor = BatchProcessor(
        store_path=cfg.faiss_path,
        model_name=cfg.model_name,
        ollama_endpoint=cfg.ollama_endpoint,
        ollama_api_key=cfg.ollama_api_key,
        top_k=cfg.top_k,
        retrieval_mode=cfg.retrieval_mode,
        bm25_weight=cfg.bm25_weight,
        rerank_top_n=cfg.rerank_top_n,
        confidence_threshold=cfg.confidence_threshold,
        temperature=cfg.temperature,
        max_context_tokens=cfg.max_context_tokens,
    )
    csv_content = Path(csv_path).read_text(encoding="utf-8")
    output = processor.process_queries_csv(csv_content)
    out_path = Path(csv_path).stem + "_results.csv"
    Path(out_path).write_text(output, encoding="utf-8")
    print(f"Results written to {out_path}")


def _run_index(folder_path: str) -> None:
    """Index a directory from CLI using batch processor."""
    from local_archive_ai.config import load_config
    from local_archive_ai.batch_processor import BatchProcessor

    cfg = load_config()
    processor = BatchProcessor(
        store_path=cfg.faiss_path,
        model_name=cfg.model_name,
        ollama_endpoint=cfg.ollama_endpoint,
        ollama_api_key=cfg.ollama_api_key,
    )
    results, total_chunks = processor.process_documents(
        folder_path=folder_path,
        chunk_size=cfg.chunk_size,
        on_progress=lambda p: print(f"\r{p.bar_text()}", end="", flush=True),
    )
    print(f"\nIndexing complete: {total_chunks} chunks from {len(results)} files")


def _run_load_repo(url_or_path: str) -> None:
    """Load and index a Git repository from CLI."""
    from local_archive_ai.config import load_config
    from local_archive_ai.multi_format_loader import MultiFormatLoader
    from local_archive_ai.services import EmbeddingService
    from local_archive_ai.store import LocalVectorStore

    cfg = load_config()
    loader = MultiFormatLoader(chunk_size=cfg.chunk_size)
    results = loader.load_git_repo(url_or_path)

    all_chunks = []
    for r in results:
        if r.status == "success" and r.chunks:
            all_chunks.extend(r.chunks)

    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        embedder = EmbeddingService()
        vectors = embedder.embed(texts, batch_size=12)
        store = LocalVectorStore(Path(cfg.faiss_path))
        store.build(vectors, all_chunks)
        print(f"Repository indexed: {len(all_chunks)} chunks from {len(results)} files")
    else:
        print("No content extracted from repository.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-Archive AI")
    parser.add_argument("--check-config", action="store_true", help="Validate config and exit")
    parser.add_argument("--health", action="store_true", help="Print health-check JSON and exit")
    parser.add_argument("--chat", type=str, metavar="QUESTION", help="Run a chat query from CLI")
    parser.add_argument("--batch-csv", type=str, metavar="CSV_PATH", help="Batch process queries from CSV")
    parser.add_argument("--index", type=str, metavar="FOLDER", help="Index a directory")
    parser.add_argument("--load-repo", type=str, metavar="URL_OR_PATH", help="Index a Git repository")
    args, _unknown = parser.parse_known_args()

    if args.check_config:
        from local_archive_ai.config import load_config
        try:
            load_config()
            print("Config validation successful.")
            sys.exit(0)
        except Exception as e:
            print(f"Config validation failed: {e}")
            sys.exit(1)

    if args.health:
        report = _health_check()
        print(json.dumps(report, indent=2))
        sys.exit(0)

    if args.chat:
        _run_chat(args.chat)
        sys.exit(0)

    if args.batch_csv:
        _run_batch_csv(args.batch_csv)
        sys.exit(0)

    if args.index:
        _run_index(args.index)
        sys.exit(0)

    if args.load_repo:
        _run_load_repo(args.load_repo)
        sys.exit(0)

    # Default: launch the FastAPI web client
    log.info("Starting Local-Archive AI via FastAPI")
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
