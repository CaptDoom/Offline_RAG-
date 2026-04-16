"""Real-time document folder watcher using watchdog for auto-indexing."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from local_archive_ai.logging_config import log

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
except Exception:  # pragma: no cover
    Observer = None  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    FileCreatedEvent = None  # type: ignore[assignment,misc]
    FileModifiedEvent = None  # type: ignore[assignment,misc]

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".json", ".csv",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java",
    ".c", ".cpp", ".go", ".rs", ".html", ".css",
    ".sql", ".yaml", ".yml", ".toml", ".ini",
}


class _DocumentHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Debounced handler that queues new/modified files for indexing."""

    def __init__(self, callback: Callable[[list[Path]], None], debounce_seconds: float = 5.0) -> None:
        super().__init__()
        self._callback = callback
        self._debounce = debounce_seconds
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _schedule_flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = [Path(p) for p in self._pending]
            self._pending.clear()
        if paths:
            log.info("Watcher auto-indexing %d new/modified files", len(paths))
            try:
                self._callback(paths)
            except Exception:
                log.exception("Watcher auto-index callback failed")

    def on_created(self, event: Any) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with self._lock:
                self._pending[str(path)] = time.time()
            self._schedule_flush()

    def on_modified(self, event: Any) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            with self._lock:
                self._pending[str(path)] = time.time()
            self._schedule_flush()


class FolderWatcher:
    """Watch a folder for new/modified documents and trigger auto-indexing."""

    def __init__(self, callback: Callable[[list[Path]], None], debounce_seconds: float = 5.0) -> None:
        self._callback = callback
        self._debounce = debounce_seconds
        self._observer: Any = None
        self._watch_path: str | None = None

    @property
    def available(self) -> bool:
        return Observer is not None

    @property
    def running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def start(self, folder: str) -> bool:
        if not self.available:
            log.warning("watchdog not installed – folder watcher unavailable")
            return False
        self.stop()
        handler = _DocumentHandler(self._callback, self._debounce)
        self._observer = Observer()
        self._observer.schedule(handler, folder, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._watch_path = folder
        log.info("Folder watcher started on %s", folder)
        return True

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None
            self._watch_path = None
