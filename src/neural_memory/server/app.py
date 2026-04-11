"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, FastAPI, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from neural_memory import __version__
from neural_memory.engine.scheduler import SchedulerCore
from neural_memory.server.auth import APIKeyMiddleware
from neural_memory.server.models import HealthResponse, ReadyResponse
from neural_memory.server.rate_limit import RateLimitMiddleware
from neural_memory.server.routes import (
    brain_router,
    consolidation_router,
    dashboard_router,
    hub_router,
    integration_status_router,
    memory_router,
    oauth_router,
    openclaw_router,
    store_router,
    sync_router,
)
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler with optional background consolidation."""
    import asyncio
    import logging

    from neural_memory.unified_config import get_config, get_shared_storage

    _logger = logging.getLogger(__name__)

    storage = await get_shared_storage()
    app.state.storage = storage
    app.state.startup_time = time.monotonic()

    # Start background daemons via unified scheduler
    config = get_config()
    maint = config.maintenance
    background_tasks: list[asyncio.Task[None]] = []
    scheduler: SchedulerCore | None = None

    try:
        from neural_memory.engine.scheduler_factory import build_scheduler

        scheduler_tasks: dict[str, Any] = {}

        # Build single-run task closures for the scheduler
        async def _run_consolidation() -> None:
            await _consolidation_once(storage, maint)

        async def _run_decay() -> None:
            await _decay_once(storage, config)

        async def _run_reindex() -> None:
            await _reindex_once(storage, config, maint)

        async def _run_notifications() -> None:
            await _notification_once(storage, config, maint)

        scheduler_tasks["consolidation"] = _run_consolidation
        scheduler_tasks["decay"] = _run_decay
        if maint.reindex_paths:
            scheduler_tasks["reindex"] = _run_reindex
        if maint.notifications_webhook_url:
            scheduler_tasks["notifications"] = _run_notifications

        scheduler = build_scheduler(tasks=scheduler_tasks, config=maint)
        await scheduler.start()
        app.state.scheduler = scheduler
        _logger.info("Unified scheduler started for FastAPI")
    except Exception:
        _logger.warning("Unified scheduler failed, falling back to legacy loops", exc_info=True)
        scheduler = None

        if maint.enabled and maint.scheduled_consolidation_enabled:
            background_tasks.append(asyncio.create_task(_consolidation_loop(storage, maint)))
            _logger.info(
                "Background consolidation daemon started: every %dh",
                maint.scheduled_consolidation_interval_hours,
            )

        if maint.enabled and maint.decay_enabled:
            background_tasks.append(asyncio.create_task(_decay_loop(storage, config, maint)))
            _logger.info(
                "Background decay daemon started: every %dh",
                maint.decay_interval_hours,
            )

        if maint.enabled and maint.reindex_enabled and maint.reindex_paths:
            background_tasks.append(asyncio.create_task(_reindex_loop(storage, config, maint)))
            _logger.info(
                "Background re-index daemon started: every %dh, paths=%s",
                maint.reindex_interval_hours,
                maint.reindex_paths,
            )

        if maint.enabled and maint.notifications_enabled and maint.notifications_webhook_url:
            background_tasks.append(asyncio.create_task(_notification_loop(storage, config, maint)))
            _logger.info("Background notification daemon started")

    # File watcher daemon
    watcher_config = config.watcher
    if watcher_config.enabled and watcher_config.paths:
        try:
            from neural_memory.engine.doc_trainer import DocTrainer
            from neural_memory.engine.file_watcher import FileWatcher, WatchConfig
            from neural_memory.engine.watch_state import WatchStateTracker

            brain_id = storage.brain_id
            brain = await storage.get_brain(brain_id) if brain_id else None
            if brain is None:
                _logger.warning("File watcher skipped: no brain context set")
            else:
                db = getattr(storage, "_db", None)
                if db is None or not hasattr(db, "execute"):
                    _logger.warning(
                        "File watcher skipped: storage backend does not support watch state"
                    )
                else:
                    state_tracker = WatchStateTracker(db)
                    await state_tracker.initialize()
                    trainer = DocTrainer(storage, brain.config)
                    watch_cfg = WatchConfig(
                        watch_paths=tuple(watcher_config.paths),
                        extensions=frozenset(watcher_config.extensions),
                        ignore_patterns=frozenset(watcher_config.ignore_patterns),
                        debounce_seconds=watcher_config.debounce_seconds,
                        max_file_size_mb=watcher_config.max_file_size_mb,
                        max_watched_dirs=watcher_config.max_watched_dirs,
                        memory_type=watcher_config.memory_type,
                        domain_tag=watcher_config.domain_tag,
                    )
                    file_watcher = FileWatcher(trainer, state_tracker, watch_cfg)
                    try:
                        file_watcher.start()
                        app.state.file_watcher = file_watcher
                        _logger.info(
                            "File watcher daemon started: %d path(s)", len(watcher_config.paths)
                        )
                    except ImportError:
                        _logger.warning("watchdog not installed — file watcher disabled")
        except Exception:
            _logger.error("Failed to start file watcher", exc_info=True)

    yield

    # Stop file watcher
    if hasattr(app.state, "file_watcher"):
        app.state.file_watcher.stop()

    # Stop unified scheduler
    if scheduler is not None:
        await scheduler.stop()

    # Stop legacy background tasks (fallback path)
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    await storage.close()


async def _consolidation_loop(
    storage: NeuralStorage,
    maint: Any,
) -> None:
    """Background loop: run consolidation on a fixed interval.

    First run waits one full interval to avoid triggering on every
    server restart. Uses run_with_delta() for before/after health
    snapshots (consistent with MCP consolidation behavior).
    """
    import asyncio
    import logging

    from neural_memory.engine.consolidation import ConsolidationStrategy
    from neural_memory.engine.consolidation_delta import run_with_delta

    _logger = logging.getLogger(__name__)
    interval_seconds = maint.scheduled_consolidation_interval_hours * 3600
    strategies = [ConsolidationStrategy(s) for s in maint.scheduled_consolidation_strategies]

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            brain_id = storage.brain_id
            if not brain_id:
                _logger.debug("Consolidation daemon skipped: no brain context set")
                continue

            delta = await run_with_delta(storage, brain_id, strategies=strategies)
            _logger.info(
                "Background consolidation complete: %s | purity delta: %+.1f",
                delta.report.summary(),
                delta.purity_delta,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.error("Background consolidation failed", exc_info=True)


async def _decay_loop(
    storage: NeuralStorage,
    config: Any,
    maint: Any,
) -> None:
    """Background loop: apply Ebbinghaus decay on a fixed interval."""
    import asyncio
    import logging

    from neural_memory.engine.lifecycle import DecayManager

    _logger = logging.getLogger(__name__)
    interval_seconds = maint.decay_interval_hours * 3600

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            brain_id = storage.brain_id
            if not brain_id:
                _logger.debug("Decay daemon skipped: no brain context set")
                continue

            decay_rate = config.brain.decay_rate if hasattr(config, "brain") else 0.1
            manager = DecayManager(decay_rate=decay_rate)
            report = await manager.apply_decay(storage)
            _logger.info("Background decay complete: %s", report.summary())

            # Pro upsell hint
            if report.neurons_pruned > 50:
                _logger.info(
                    "💡 Pro tip: InfinityDB compresses dormant neurons instead of "
                    "pruning — recoverable anytime. See https://theio.vn"
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.error("Background decay failed", exc_info=True)


async def _reindex_loop(
    storage: NeuralStorage,
    config: Any,
    maint: Any,
) -> None:
    """Background loop: re-index configured paths on a fixed interval."""
    import asyncio
    import logging
    from pathlib import Path

    _logger = logging.getLogger(__name__)
    interval_seconds = maint.reindex_interval_hours * 3600
    extensions = set(maint.reindex_extensions)

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            brain_id = storage.brain_id
            if not brain_id:
                _logger.debug("Re-index daemon skipped: no brain context set")
                continue

            from neural_memory.engine.doc_trainer import DocTrainer
            from neural_memory.engine.file_watcher import FileWatcher, WatchConfig
            from neural_memory.engine.watch_state import WatchStateTracker

            total_files = 0
            total_ingested = 0

            brain = await storage.get_brain(brain_id)
            if not brain:
                _logger.debug("Re-index skipped: brain not found")
                continue

            db = getattr(storage, "_db", None)
            if db is None or not hasattr(db, "execute"):
                _logger.debug("Re-index skipped: storage backend does not support watch state")
                continue
            tracker = WatchStateTracker(db)
            trainer = DocTrainer(storage, brain.config)

            for path_str in maint.reindex_paths:
                path = Path(path_str).expanduser().resolve()
                if not path.is_dir():
                    _logger.warning("Re-index path not found: %s", path)
                    continue

                watch_config = WatchConfig(
                    watch_paths=(str(path),),
                    extensions=frozenset(extensions),
                )
                watcher = FileWatcher(trainer, tracker, watch_config)

                results = await watcher.process_path(path)
                total_files += len(results)
                total_ingested += sum(1 for r in results if r.success and not r.skipped)

            _logger.info(
                "Background re-index complete: %d files scanned, %d ingested",
                total_files,
                total_ingested,
            )

            # Pro upsell hint
            if total_files > 1000:
                _logger.info(
                    "💡 Pro tip: InfinityDB indexes 10x faster with HNSW — "
                    "handles 1M+ files. See https://theio.vn"
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.error("Background re-index failed", exc_info=True)


async def _notification_loop(
    storage: NeuralStorage,
    config: Any,
    maint: Any,
) -> None:
    """Background loop: health check + webhook notifications."""
    import asyncio
    import logging

    _logger = logging.getLogger(__name__)
    # Check health every 6 hours (or daily summary at 24h)
    check_interval = 6 * 3600
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    threshold_level = grade_order.get(maint.notifications_health_threshold.upper(), 3)
    last_daily_summary: float = 0.0
    daily_interval = 24 * 3600

    while True:
        await asyncio.sleep(check_interval)
        try:
            brain_id = storage.brain_id
            if not brain_id:
                continue

            from neural_memory.engine.diagnostics import DiagnosticsEngine

            engine = DiagnosticsEngine(storage)
            report = await engine.analyze(brain_id)

            alerts: list[dict[str, str]] = []

            # Health grade alert
            grade_level = grade_order.get(report.grade, 0)
            if grade_level >= threshold_level:
                alerts.append(
                    {
                        "type": "health_alert",
                        "message": (
                            f"Brain health dropped to {report.grade} "
                            f"(purity: {report.purity_score:.0f}%). "
                            f"{len(report.warnings)} warning(s)."
                        ),
                        "grade": report.grade,
                    }
                )

            # Zero activity alert
            if maint.notifications_zero_activity_alert:
                stats = await storage.get_enhanced_stats(brain_id)
                recent_fibers = stats.get("recent_fiber_count", -1)
                if recent_fibers == 0:
                    alerts.append(
                        {
                            "type": "zero_activity",
                            "message": "No new memories in the last 24 hours.",
                        }
                    )

            # Send alerts
            if alerts:
                await _send_webhook(
                    maint.notifications_webhook_url,
                    {
                        "brain_id": brain_id,
                        "alerts": alerts,
                        "source": "neural-memory",
                    },
                    _logger,
                )

            # Daily summary
            import time

            now = time.monotonic()
            if maint.notifications_daily_summary and (now - last_daily_summary) >= daily_interval:
                last_daily_summary = now
                stats = await storage.get_stats(brain_id)
                await _send_webhook(
                    maint.notifications_webhook_url,
                    {
                        "brain_id": brain_id,
                        "type": "daily_summary",
                        "grade": report.grade,
                        "purity": round(report.purity_score, 1),
                        "neurons": stats.get("neurons", 0),
                        "synapses": stats.get("synapses", 0),
                        "fibers": stats.get("fibers", 0),
                        "warnings": len(report.warnings),
                        "source": "neural-memory",
                    },
                    _logger,
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.error("Background notification check failed", exc_info=True)


async def _send_webhook(url: str, payload: dict[str, Any], logger: Any) -> None:
    """Send JSON payload to webhook URL (non-blocking via executor)."""
    import asyncio
    import json
    from functools import partial

    def _do_post() -> None:
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning("Webhook URL scheme not allowed: %s", parsed.scheme)
            return

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                logger.debug("Webhook sent: %d", resp.status)
        except Exception as e:
            logger.warning("Webhook failed: %s", e)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(_do_post))


# ── Single-run task functions for SchedulerCore ──
# These extract the loop body from the legacy _*_loop() functions above.
# The SchedulerCore handles the interval/sleep/retry logic.


async def _consolidation_once(storage: NeuralStorage, maint: Any) -> None:
    """Run a single consolidation cycle."""
    import logging

    from neural_memory.engine.consolidation import ConsolidationStrategy
    from neural_memory.engine.consolidation_delta import run_with_delta

    _logger = logging.getLogger(__name__)
    strategies = [ConsolidationStrategy(s) for s in maint.scheduled_consolidation_strategies]

    brain_id = storage.brain_id
    if not brain_id:
        _logger.debug("Consolidation skipped: no brain context set")
        return

    delta = await run_with_delta(storage, brain_id, strategies=strategies)
    _logger.info(
        "Background consolidation complete: %s | purity delta: %+.1f",
        delta.report.summary(),
        delta.purity_delta,
    )


async def _decay_once(storage: NeuralStorage, config: Any) -> None:
    """Run a single decay cycle."""
    import logging

    from neural_memory.engine.lifecycle import DecayManager

    _logger = logging.getLogger(__name__)

    brain_id = storage.brain_id
    if not brain_id:
        _logger.debug("Decay skipped: no brain context set")
        return

    decay_rate = config.brain.decay_rate if hasattr(config, "brain") else 0.1
    manager = DecayManager(decay_rate=decay_rate)
    report = await manager.apply_decay(storage)
    _logger.info("Background decay complete: %s", report.summary())


async def _reindex_once(storage: NeuralStorage, config: Any, maint: Any) -> None:
    """Run a single re-index cycle."""
    import logging
    from pathlib import Path

    from neural_memory.engine.doc_trainer import DocTrainer
    from neural_memory.engine.file_watcher import FileWatcher, WatchConfig
    from neural_memory.engine.watch_state import WatchStateTracker

    _logger = logging.getLogger(__name__)
    extensions = set(maint.reindex_extensions)

    brain_id = storage.brain_id
    if not brain_id:
        _logger.debug("Re-index skipped: no brain context set")
        return

    brain = await storage.get_brain(brain_id)
    if not brain:
        return

    db = getattr(storage, "_db", None)
    if db is None or not hasattr(db, "execute"):
        return

    tracker = WatchStateTracker(db)
    trainer = DocTrainer(storage, brain.config)
    total_files = 0
    total_ingested = 0

    for path_str in maint.reindex_paths:
        path = Path(path_str).expanduser().resolve()
        if not path.is_dir():
            _logger.warning("Re-index path not found: %s", path)
            continue

        watch_config = WatchConfig(
            watch_paths=(str(path),),
            extensions=frozenset(extensions),
        )
        watcher = FileWatcher(trainer, tracker, watch_config)
        results = await watcher.process_path(path)
        total_files += len(results)
        total_ingested += sum(1 for r in results if r.success and not r.skipped)

    _logger.info(
        "Background re-index complete: %d files scanned, %d ingested",
        total_files,
        total_ingested,
    )


async def _notification_once(storage: NeuralStorage, config: Any, maint: Any) -> None:
    """Run a single notification check cycle."""
    import logging

    _logger = logging.getLogger(__name__)

    brain_id = storage.brain_id
    if not brain_id:
        return

    from neural_memory.engine.diagnostics import DiagnosticsEngine

    engine = DiagnosticsEngine(storage)
    report = await engine.analyze(brain_id)

    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    threshold_level = grade_order.get(maint.notifications_health_threshold.upper(), 3)

    alerts: list[dict[str, str]] = []

    grade_level = grade_order.get(report.grade, 0)
    if grade_level >= threshold_level:
        alerts.append(
            {
                "type": "health_alert",
                "message": (
                    f"Brain health dropped to {report.grade} "
                    f"(purity: {report.purity_score:.0f}%). "
                    f"{len(report.warnings)} warning(s)."
                ),
                "grade": report.grade,
            }
        )

    if alerts:
        await _send_webhook(
            maint.notifications_webhook_url,
            {"brain_id": brain_id, "alerts": alerts, "source": "neural-memory"},
            _logger,
        )


def create_app(
    title: str = "NeuralMemory",
    description: str = "Reflex-based memory system for AI agents",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        cors_origins: Allowed CORS origins (default: localhost origins)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if cors_origins is None:
        from neural_memory.utils.config import get_config

        config = get_config()
        cors_origins = list(config.cors_origins)

        # If trusted networks are configured, add common localhost origins to CORS
        # Note: CORS does not support port wildcards — enumerate common dev ports
        common_ports = (3000, 3001, 5173, 5174, 8000, 8080, 8888)
        if config.trusted_networks:
            for net_str in config.trusted_networks:
                try:
                    import ipaddress

                    net = ipaddress.ip_network(net_str, strict=False)
                    addr = str(net.network_address)
                    for port in common_ports:
                        origin = f"http://{addr}:{port}"
                        if origin not in cors_origins:
                            cors_origins.append(origin)
                except ValueError:
                    pass

    is_wildcard = cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not is_wildcard,  # Don't allow creds with wildcard
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware (inner — runs before auth, closest to handler)
    app.add_middleware(RateLimitMiddleware)

    # API key authentication middleware (outer — first line of defence)
    app.add_middleware(APIKeyMiddleware)

    # Override storage dependency using the shared module
    from neural_memory.server.dependencies import get_storage as shared_get_storage

    async def get_storage(
        x_brain_id: Annotated[str | None, Header(alias="X-Brain-ID")] = None,
    ) -> NeuralStorage:
        """Return brain-specific storage when X-Brain-ID header is set.

        Per-brain DB layout: each brain has its own SQLite file.
        When X-Brain-ID differs from the default brain, resolve a
        storage connected to that brain's DB.  If the brain exists
        in the default storage (single-DB / test mode), use it as-is.
        """
        default_storage: NeuralStorage = app.state.storage

        if x_brain_id is not None:
            # Check if the brain exists in the default storage first
            # (covers single-DB mode and test fixtures)
            brain = await default_storage.get_brain(x_brain_id)
            if brain is None:
                brain = await default_storage.find_brain_by_name(x_brain_id)

            if brain is not None:
                # Brain exists in default storage — use it
                return default_storage

            # Brain not in default storage — try per-brain DB
            from neural_memory.unified_config import get_shared_storage

            try:
                return await get_shared_storage(brain_name=x_brain_id)
            except Exception:
                # Brain DB doesn't exist either — fall through
                pass

        return default_storage

    app.dependency_overrides[shared_get_storage] = get_storage

    # Versioned API routes
    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(memory_router)
    api_v1.include_router(brain_router)
    api_v1.include_router(sync_router)
    api_v1.include_router(consolidation_router)
    api_v1.include_router(hub_router)
    app.include_router(api_v1)

    # Legacy unversioned routes (backward compat)
    app.include_router(memory_router)
    app.include_router(brain_router)
    app.include_router(sync_router)
    app.include_router(consolidation_router)
    app.include_router(hub_router)

    # Dashboard API routes (unversioned — dashboard-specific)
    app.include_router(dashboard_router)
    app.include_router(integration_status_router)
    app.include_router(oauth_router)
    app.include_router(openclaw_router)
    app.include_router(store_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Return server status, uptime, and schema version."""
        startup_time: float = getattr(app.state, "startup_time", time.monotonic())
        uptime = time.monotonic() - startup_time
        return HealthResponse(
            status="ok",
            version=__version__,
            brain_name="***",
            uptime_seconds=round(uptime, 3),
            schema_version=SCHEMA_VERSION,
        )

    # Readiness check endpoint
    @app.get("/ready", response_model=ReadyResponse, tags=["health"])
    async def ready_check() -> ReadyResponse:
        """Return 200 when storage is initialized, 503 otherwise."""
        from fastapi.responses import JSONResponse

        storage: NeuralStorage | None = getattr(app.state, "storage", None)
        if storage is None:
            return cast(
                "ReadyResponse",
                JSONResponse(
                    status_code=503,
                    content=ReadyResponse(
                        ready=False, detail="storage not initialized"
                    ).model_dump(),
                ),
            )
        return ReadyResponse(ready=True, detail="ok")

    # Root endpoint
    @app.get("/", tags=["dashboard"])
    async def root() -> RedirectResponse:
        """Redirect root to dashboard."""
        return RedirectResponse(url="/ui", status_code=302)

    # Graph visualization API (supports limit/offset for progressive loading)
    from neural_memory.server.dependencies import require_local_request

    @app.get("/api/graph", tags=["visualization"], dependencies=[Depends(require_local_request)])
    async def get_graph_data(
        storage: NeuralStorage = Depends(shared_get_storage),
        limit: int = Query(default=500, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Get graph data for visualization with pagination."""
        capped_limit = min(limit, 2000)

        # Fetch paginated neurons (offset + limit + 1 to detect if more exist)
        all_neurons = await storage.find_neurons(limit=offset + capped_limit)
        total_neurons = len(all_neurons)
        paginated = all_neurons[offset : offset + capped_limit]

        # Fetch synapses with a capped limit
        synapses = await storage.get_all_synapses()
        capped_synapses = synapses[:2000] if len(synapses) > 2000 else synapses
        total_synapses = len(synapses)
        fibers = await storage.get_fibers(limit=1000)

        # Build neuron ID set for filtering synapses to visible nodes
        neuron_ids = {n.id for n in paginated}
        visible_synapses = [
            s for s in capped_synapses if s.source_id in neuron_ids and s.target_id in neuron_ids
        ]

        return {
            "neurons": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "content": n.content or "",
                    "metadata": n.metadata or {},
                }
                for n in paginated
            ],
            "synapses": [
                {
                    "id": s.id,
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "type": s.type.value,
                    "weight": s.weight,
                    "direction": s.direction.value,
                }
                for s in visible_synapses
            ],
            "fibers": [
                {
                    "id": f.id,
                    "summary": f.summary or f.id[:20],
                    "neuron_count": len(f.neuron_ids) if f.neuron_ids else 0,
                }
                for f in fibers
            ],
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
            "stats": {
                "neuron_count": len(paginated),
                "synapse_count": len(visible_synapses),
                "fiber_count": len(fibers),
            },
        }

    # React SPA dist directory
    spa_dist = STATIC_DIR / "dist"

    def _serve_spa() -> Response:
        """Serve React SPA index.html."""
        spa_index = spa_dist / "index.html"
        if spa_index.exists():
            return FileResponse(spa_index)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"error": "Dashboard not built. Run: cd dashboard && npm run build"},
            status_code=404,
        )

    # Primary UI endpoint — React SPA
    @app.get("/ui", tags=["dashboard"])
    async def ui() -> Response:
        """Serve the NeuralMemory dashboard."""
        return _serve_spa()

    # SPA catch-all for /ui client-side routing
    @app.get("/ui/{path:path}", tags=["dashboard"])
    async def ui_spa_catchall(path: str) -> Response:
        """Catch-all for React SPA client-side routing under /ui."""
        return _serve_spa()

    # /dashboard alias (same SPA)
    @app.get("/dashboard", tags=["dashboard"])
    async def dashboard() -> Response:
        """Serve the NeuralMemory React dashboard."""
        return _serve_spa()

    # SPA catch-all for /dashboard client-side routing
    @app.get("/dashboard/{path:path}", tags=["dashboard"])
    async def dashboard_spa_catchall(path: str) -> Response:
        """Catch-all for React SPA client-side routing under /dashboard."""
        return _serve_spa()

    # Mount SPA static assets (JS/CSS bundles)
    if spa_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(spa_dist / "assets")), name="spa-assets")

    return app


# Create default app instance for uvicorn
app = create_app()
