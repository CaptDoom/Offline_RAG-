"""Local-Archive AI -- clean entry point.

Usage:
    streamlit run main.py
    python main.py --check-config   # validate config.yaml and exit
    python main.py --health          # print health-check JSON and exit
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-Archive AI")
    parser.add_argument("--check-config", action="store_true", help="Validate config and exit")
    parser.add_argument("--health", action="store_true", help="Print health-check JSON and exit")
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

    # Default: launch the Streamlit app
    log.info("Starting Local-Archive AI")
    from app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
