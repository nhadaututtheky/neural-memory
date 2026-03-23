"""File watcher — auto-ingest files from watched directories.

Monitors directories for file changes and auto-ingests them into
Neural Memory using the existing DocTrainer pipeline.

Requires: `pip install neural-memory[watch]` (watchdog dependency).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.engine.doc_chunker import EXTENDED_EXTENSIONS
from neural_memory.utils.simhash import simhash

if TYPE_CHECKING:
    from neural_memory.engine.doc_trainer import DocTrainer
    from neural_memory.engine.watch_state import WatchStateTracker

logger = logging.getLogger(__name__)

# Security defaults
MAX_WATCHED_DIRS = 10
MAX_FILE_SIZE_MB = 10
DEFAULT_DEBOUNCE_SECONDS = 2.0
DEFAULT_IGNORE_PATTERNS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


@dataclass(frozen=True)
class WatchConfig:
    """Configuration for file watching."""

    watch_paths: tuple[str, ...] = ()
    extensions: frozenset[str] = EXTENDED_EXTENSIONS
    ignore_patterns: frozenset[str] = DEFAULT_IGNORE_PATTERNS
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    max_watched_dirs: int = MAX_WATCHED_DIRS
    memory_type: str = "reference"
    domain_tag: str = "file_watcher"
    consolidate: bool = True


@dataclass
class WatchEvent:
    """A debounced file system event."""

    path: Path
    event_type: str  # created, modified, deleted
    timestamp: float = 0.0


@dataclass
class WatchResult:
    """Result of processing a single file."""

    path: str
    success: bool
    neurons_created: int = 0
    skipped: bool = False
    error: str = ""


class FileWatcher:
    """Watches directories and auto-ingests files into Neural Memory.

    Usage:
        watcher = FileWatcher(trainer, state_tracker, config)
        await watcher.process_path(Path("docs/"))  # One-shot scan
        watcher.start()                              # Background watching
        watcher.stop()                               # Stop watching
    """

    def __init__(
        self,
        trainer: DocTrainer,
        state_tracker: WatchStateTracker,
        config: WatchConfig | None = None,
    ) -> None:
        self._trainer = trainer
        self._state = state_tracker
        self._config = config or WatchConfig()
        self._observer: Any = None
        self._pending: dict[str, WatchEvent] = {}
        self._debounce_task: asyncio.Task[None] | None = None
        self._running = False
        self._results: list[WatchResult] = []

    @property
    def is_running(self) -> bool:
        return self._running

    def validate_path(self, path: Path) -> str | None:
        """Validate a watch path for security.

        Returns None if valid, error message if invalid.
        """
        resolved = path.resolve()

        # Must be absolute
        if not resolved.is_absolute():
            return f"Path must be absolute: {path}"

        # Must exist and be a directory
        if not resolved.is_dir():
            return f"Path is not a directory: {path}"

        # No symlink traversal outside the resolved path
        if resolved.is_symlink():
            target = resolved.readlink().resolve()
            if not target.is_relative_to(resolved.parent):
                return f"Symlink traversal detected: {path} -> {target}"

        return None

    async def process_path(self, directory: Path) -> list[WatchResult]:
        """One-shot scan and ingest all eligible files in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of WatchResult for each processed file.
        """
        error = self.validate_path(directory)
        if error:
            return [WatchResult(path=str(directory), success=False, error=error)]

        results: list[WatchResult] = []
        resolved = directory.resolve()

        for file_path in sorted(resolved.rglob("*")):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in self._config.extensions:
                continue

            # Check ignore patterns
            if self._should_ignore(file_path):
                continue

            result = await self._process_file(file_path)
            results.append(result)

        return results

    async def _process_file(self, file_path: Path) -> WatchResult:
        """Process a single file through the ingestion pipeline."""
        resolved = file_path.resolve()
        str_path = str(resolved)

        # Security: check file size
        try:
            size_mb = resolved.stat().st_size / (1024 * 1024)
            if size_mb > self._config.max_file_size_mb:
                return WatchResult(
                    path=str_path,
                    success=False,
                    error=f"File too large: {size_mb:.1f}MB > {self._config.max_file_size_mb}MB",
                )
        except OSError as e:
            return WatchResult(path=str_path, success=False, error=f"Cannot stat file: {e}")

        # Check if file needs processing (mtime + simhash)
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            content_hash = simhash(content)

            needs_processing = await self._state.should_process_with_simhash(resolved, content_hash)
            if not needs_processing:
                return WatchResult(path=str_path, success=True, skipped=True)
        except Exception as e:
            logger.debug("Simhash check failed for %s: %s", str_path, e)
            content_hash = 0

        # Ingest via DocTrainer
        try:
            from neural_memory.engine.doc_trainer import TrainingConfig

            training_config = TrainingConfig(
                domain_tag=self._config.domain_tag,
                memory_type=self._config.memory_type,
                consolidate=self._config.consolidate,
                extensions=(file_path.suffix,),
            )

            result = await self._trainer.train_file(resolved, training_config)

            # Update watch state
            mtime = resolved.stat().st_mtime
            await self._state.mark_processed(
                resolved,
                mtime=mtime,
                content_hash=content_hash,
                neuron_count=result.neurons_created,
            )

            logger.info(
                "Ingested %s: %d neurons, %d chunks",
                file_path.name,
                result.neurons_created,
                result.chunks_encoded,
            )

            return WatchResult(
                path=str_path,
                success=True,
                neurons_created=result.neurons_created,
            )

        except Exception as e:
            logger.error("Failed to ingest %s: %s", str_path, e, exc_info=True)
            return WatchResult(path=str_path, success=False, error=str(e))

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on path patterns."""
        parts = file_path.parts
        return any(pattern in parts for pattern in self._config.ignore_patterns)

    def start(self) -> None:
        """Start watching directories in background.

        Requires watchdog: `pip install neural-memory[watch]`
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as e:
            raise ImportError(
                "watchdog is required for file watching. "
                "Install with: pip install neural-memory[watch]"
            ) from e

        if self._running:
            logger.warning("File watcher already running")
            return

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    watcher._queue_event(Path(event.src_path), "created")

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    watcher._queue_event(Path(event.src_path), "modified")

            def on_deleted(self, event: Any) -> None:
                if not event.is_directory:
                    watcher._queue_event(Path(event.src_path), "deleted")

        self._observer = Observer()
        handler = _Handler()

        watch_count = 0
        for watch_path in self._config.watch_paths:
            path = Path(watch_path).resolve()
            error = self.validate_path(path)
            if error:
                logger.warning("Skipping invalid watch path: %s", error)
                continue

            if watch_count >= self._config.max_watched_dirs:
                logger.warning(
                    "Max watched dirs (%d) reached, skipping: %s",
                    self._config.max_watched_dirs,
                    path,
                )
                break

            self._observer.schedule(handler, str(path), recursive=True)
            watch_count += 1
            logger.info("Watching: %s", path)

        if watch_count == 0:
            logger.warning("No valid watch paths configured")
            return

        self._observer.start()
        self._running = True
        logger.info("File watcher started (%d directories)", watch_count)

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        logger.info("File watcher stopped")

    def _queue_event(self, path: Path, event_type: str) -> None:
        """Queue a file event for debounced processing."""
        import time

        # Check extension
        if path.suffix.lower() not in self._config.extensions:
            return

        # Check ignore patterns
        if self._should_ignore(path):
            return

        self._pending[str(path)] = WatchEvent(
            path=path,
            event_type=event_type,
            timestamp=time.time(),
        )

    async def flush_pending(self) -> list[WatchResult]:
        """Process all pending debounced events.

        Call this periodically or after debounce timeout.
        """
        import time

        if not self._pending:
            return []

        now = time.time()
        ready: list[WatchEvent] = []
        still_pending: dict[str, WatchEvent] = {}

        for key, event in self._pending.items():
            if now - event.timestamp >= self._config.debounce_seconds:
                ready.append(event)
            else:
                still_pending[key] = event

        self._pending = still_pending

        results: list[WatchResult] = []
        for event in ready:
            if event.event_type == "deleted":
                await self._state.mark_deleted(event.path)
                results.append(WatchResult(path=str(event.path), success=True))
            else:
                result = await self._process_file(event.path)
                results.append(result)

        self._results.extend(results)
        return results

    def get_recent_results(self, limit: int = 20) -> list[WatchResult]:
        """Get recent processing results."""
        return self._results[-limit:]
