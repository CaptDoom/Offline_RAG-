"""Progress tracking service for indexing and query operations.

Provides thread-safe progress tracking with WebSocket and REST API support.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IndexingStatus(Enum):
    """Indexing operation status."""
    IDLE = "idle"
    SCANNING = "scanning"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TaskProgress:
    """Progress state for a single task."""
    task_id: str
    task_type: str  # 'indexing', 'query', 'batch_query', etc.
    status: TaskStatus = TaskStatus.PENDING
    indexing_status: Optional[IndexingStatus] = None
    total_steps: int = 0
    current_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    current_file: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_update: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        """Calculate percentage progress (0-100)."""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time based on current progress rate."""
        if self.current_step == 0 or self.total_steps == 0:
            return None
        elapsed = self.elapsed_seconds
        rate = self.current_step / elapsed if elapsed > 0 else 0
        if rate <= 0:
            return None
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps / rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "indexing_status": self.indexing_status.value if self.indexing_status else None,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "metadata": self.metadata,
            "errors": self.errors,
            "current_file": self.current_file,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class ProgressTracker:
    """Thread-safe progress tracker for async operations.

    Supports multiple concurrent tasks with real-time updates.
    Thread-safe using RLock for all shared state mutations.

    Example:
        tracker = ProgressTracker()
        task_id = tracker.start_task("indexing", 100)
        tracker.update_progress(task_id, 50, {"files_processed": 50})
        tracker.log_error(task_id, "Failed to process file.pdf")
        tracker.complete_task(task_id)
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskProgress] = {}
        self._callbacks: list[Callable[[str, TaskProgress], None]] = []
        self._active_tasks: set[str] = set()

    def start_task(
        self,
        task_type: str,
        total_steps: int,
        task_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Start a new task and return its ID.

        Args:
            task_type: Type of task (e.g., 'indexing', 'query')
            total_steps: Total number of steps for completion
            task_id: Optional custom task ID (auto-generated if not provided)
            metadata: Optional metadata dictionary

        Returns:
            Task ID string
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        with self._lock:
            task = TaskProgress(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.RUNNING,
                total_steps=total_steps,
                metadata=metadata or {},
                start_time=datetime.now(),
                last_update=datetime.now(),
            )
            self._tasks[task_id] = task
            self._active_tasks.add(task_id)

        self._notify_callbacks(task_id)
        return task_id

    def update_progress(
        self,
        task_id: str,
        current_step: int,
        metadata: Optional[dict[str, Any]] = None,
        indexing_status: Optional[IndexingStatus] = None,
    ) -> bool:
        """Update task progress.

        Args:
            task_id: Task identifier
            current_step: Current step number
            metadata: Optional metadata to merge into task
            indexing_status: Optional indexing status update

        Returns:
            True if update succeeded, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.current_step = current_step
            task.last_update = datetime.now()

            if metadata:
                task.metadata.update(metadata)

            if indexing_status:
                task.indexing_status = indexing_status

            # Auto-update current_file from metadata if present
            if metadata and "current_file" in metadata:
                task.current_file = metadata["current_file"]

        self._notify_callbacks(task_id)
        return True

    def log_error(self, task_id: str, error_message: str, file_name: str = "") -> bool:
        """Log an error for a task.

        Args:
            task_id: Task identifier
            error_message: Error description
            file_name: Optional file where error occurred

        Returns:
            True if logged successfully, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            error_entry = {
                "message": error_message,
                "file": file_name,
                "timestamp": datetime.now().isoformat(),
            }
            task.errors.append(error_entry)
            task.last_update = datetime.now()

        self._notify_callbacks(task_id)
        return True

    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task identifier
            success: Whether task completed successfully

        Returns:
            True if completed successfully, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.end_time = datetime.now()
            task.current_step = task.total_steps  # Mark as 100%
            task.last_update = datetime.now()
            self._active_tasks.discard(task_id)

        self._notify_callbacks(task_id)
        return True

    def fail_task(self, task_id: str, error_message: str = "") -> bool:
        """Mark a task as failed.

        Args:
            task_id: Task identifier
            error_message: Optional failure reason

        Returns:
            True if marked as failed, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.last_update = datetime.now()
            if error_message:
                task.errors.append({
                    "message": error_message,
                    "timestamp": datetime.now().isoformat(),
                })
            self._active_tasks.discard(task_id)

        self._notify_callbacks(task_id)
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled, False if task not found or not running
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            task.last_update = datetime.now()
            self._active_tasks.discard(task_id)

        self._notify_callbacks(task_id)
        return True

    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get task progress by ID.

        Args:
            task_id: Task identifier

        Returns:
            TaskProgress copy or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            # Return a copy to prevent external mutation
            import copy
            return copy.deepcopy(task)

    def get_active_tasks(self) -> list[TaskProgress]:
        """Get all currently active tasks.

        Returns:
            List of active TaskProgress objects
        """
        with self._lock:
            return [
                self._tasks[tid] for tid in self._active_tasks
                if tid in self._tasks
            ]

    def get_all_tasks(
        self,
        task_type: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> list[TaskProgress]:
        """Get tasks with optional filtering.

        Args:
            task_type: Filter by task type
            status: Filter by status
            limit: Maximum number of tasks to return

        Returns:
            List of matching TaskProgress objects
        """
        with self._lock:
            tasks = list(self._tasks.values())

        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by start time (most recent first)
        tasks.sort(key=lambda t: t.start_time or datetime.min, reverse=True)
        return tasks[:limit]

    def register_callback(self, callback: Callable[[str, TaskProgress], None]) -> None:
        """Register a callback for progress updates.

        Callbacks are invoked synchronously on progress changes.
        For WebSocket support, use with an async wrapper.

        Args:
            callback: Function(task_id, task_progress) to call on updates
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[str, TaskProgress], None]) -> bool:
        """Unregister a callback.

        Args:
            callback: Previously registered callback

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                return True
            return False

    def clear_old_tasks(self, max_age_seconds: float = 3600) -> int:
        """Remove completed tasks older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of tasks removed
        """
        now = datetime.now()
        to_remove = []

        with self._lock:
            for task_id, task in self._tasks.items():
                if task_id in self._active_tasks:
                    continue  # Don't remove active tasks
                if task.end_time and (now - task.end_time).total_seconds() > max_age_seconds:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]

        return len(to_remove)

    def _notify_callbacks(self, task_id: str) -> None:
        """Notify all registered callbacks of a task update."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        # Copy callbacks list to avoid holding lock during callbacks
        with self._lock:
            callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                callback(task_id, task)
            except Exception:
                # Callback errors should not affect progress tracking
                pass


# Global progress tracker instance (singleton)
_global_tracker: Optional[ProgressTracker] = None
_global_tracker_lock = threading.Lock()


def get_progress_tracker() -> ProgressTracker:
    """Get or create the global progress tracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        with _global_tracker_lock:
            if _global_tracker is None:
                _global_tracker = ProgressTracker()
    return _global_tracker
