"""Prometheus counters for the silent failures named in issue #214."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_memory.mcp.auto_capture import empty_capture_hint
from neural_memory.safety.capture_hygiene import (
    detect_encoding_damage,
    encoding_rejection,
)
from neural_memory.server.app import create_app
from neural_memory.server.auth import _is_public_path
from neural_memory.server.rate_limit import _EXEMPT_PREFIXES
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.utils.metrics import (
    is_sqlite_write_collision,
    metric_sample,
    metrics_enabled,
    render_metrics,
)

_COLLISION = "neural_memory_sqlite_write_collision_total"
_LATENCY_COUNT = "neural_memory_sqlite_write_latency_seconds_count"
_ENCODING = "neural_memory_cli_encoding_damage_rejected_total"
_EMPTY = "neural_memory_auto_capture_empty_total"
_SQLITE = {"dialect": "sqlite"}


def _sample(name: str, **labels: str) -> float:
    return metric_sample(name, labels)


def test_metrics_path_skips_auth_and_rate_limit() -> None:
    assert _is_public_path("/metrics")
    assert "/metrics" in _EXEMPT_PREFIXES


def test_collision_marker_matches_the_driver_error() -> None:
    assert is_sqlite_write_collision(
        sqlite3.OperationalError("cannot commit - SQL statements in progress")
    )
    assert not is_sqlite_write_collision(sqlite3.OperationalError("no such table: missing"))
    assert not is_sqlite_write_collision(RuntimeError("cannot commit"))


def test_encoding_rejection_counts_damage_and_skips_force() -> None:
    pytest.importorskip("prometheus_client")
    cases = (
        ("????????????????", "question_mark_flood"),
        (r"内容 \u4e2d\u6587 没有被解码", "unicode_escape"),
        ("内容\ufffd损坏", "replacement_char"),
    )
    for text, damage_type in cases:
        assert detect_encoding_damage(text) is not None
        before = _sample(_ENCODING, command="remember", damage_type=damage_type)
        reason = encoding_rejection(text, command="remember", force=False)
        assert reason is not None
        assert _sample(_ENCODING, command="remember", damage_type=damage_type) == before + 1

        forced = _sample(_ENCODING, command="remember", damage_type=damage_type)
        assert encoding_rejection(text, command="remember", force=True) is None
        assert _sample(_ENCODING, command="remember", damage_type=damage_type) == forced

    assert (
        encoding_rejection("We should refactor the auth module", command="remember", force=False)
        is None
    )


def test_unknown_command_label_stays_bounded() -> None:
    pytest.importorskip("prometheus_client")
    before = _sample(_ENCODING, command="other", damage_type="question_mark_flood")
    encoding_rejection("????????????????", command="drop table", force=False)
    assert _sample(_ENCODING, command="other", damage_type="question_mark_flood") == before + 1
    body, _content_type = render_metrics()
    assert b"drop table" not in body


def test_empty_capture_labels() -> None:
    pytest.importorskip("prometheus_client")
    cases = (
        ("hi", "too_short", "en", "below the"),
        ("待办：更新", "too_short", "zh", "below the"),
        ("あ" * 5, "too_short", "ja", "below the"),
        ("あ" * 30, "cjk_unsupported", "ja", "not yet supported"),
        ("가" * 30, "cjk_unsupported", "ko", "not yet supported"),
        ("随便写点什么内容，但是没有任何触发词存在", "no_trigger", "zh", "决定/选定"),
        (
            "Đây là một câu tiếng Việt đủ dài để vượt ngưỡng tối thiểu hôm nay.",
            "no_trigger",
            "vi",
            "",
        ),
        ("plain english sentence with nothing special", "no_trigger", "en", ""),
    )
    for text, reason, language, snippet in cases:
        before = _sample(_EMPTY, reason=reason, language=language)
        hint = empty_capture_hint(text)
        assert _sample(_EMPTY, reason=reason, language=language) == before + 1
        if snippet:
            assert snippet in hint
        else:
            assert hint == ""


@pytest.fixture
async def dialect(tmp_path: Path) -> AsyncIterator[SQLiteDialect]:
    write_dialect = SQLiteDialect(tmp_path / "writes.db")
    await write_dialect.initialize()
    yield write_dialect
    await write_dialect.close()


def _worker_alive(storage: SQLiteDialect) -> bool:
    return storage._write_task is not None and not storage._write_task.done()


async def test_successful_write_records_latency_without_a_collision(
    dialect: SQLiteDialect,
) -> None:
    pytest.importorskip("prometheus_client")
    latency = _sample(_LATENCY_COUNT, **_SQLITE)
    collisions = _sample(_COLLISION, **_SQLITE)
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    assert _sample(_LATENCY_COUNT, **_SQLITE) == latency + 1
    assert _sample(_COLLISION, **_SQLITE) == collisions


async def test_collision_error_increments_and_worker_survives(dialect: SQLiteDialect) -> None:
    pytest.importorskip("prometheus_client")
    before = _sample(_COLLISION, **_SQLITE)

    async def collide(_conn: object) -> None:
        raise sqlite3.OperationalError("cannot commit - SQL statements in progress")

    with pytest.raises(sqlite3.OperationalError):
        await dialect._dispatch(collide)

    assert _sample(_COLLISION, **_SQLITE) == before + 1
    assert _worker_alive(dialect)
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    await dialect.execute("INSERT INTO t VALUES (1)")
    rows = await dialect.fetch_all("SELECT x FROM t")
    assert rows == [{"x": 1}]


async def test_ordinary_write_error_is_not_a_collision(dialect: SQLiteDialect) -> None:
    pytest.importorskip("prometheus_client")
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    before = _sample(_COLLISION, **_SQLITE)
    with pytest.raises(sqlite3.Error):
        await dialect.execute("INSERT INTO missing VALUES (1)")
    assert _sample(_COLLISION, **_SQLITE) == before
    assert _worker_alive(dialect)


async def test_concurrent_writes_do_not_count_collisions(dialect: SQLiteDialect) -> None:
    pytest.importorskip("prometheus_client")
    await dialect.execute("CREATE TABLE t (x INTEGER PRIMARY KEY)")
    before = _sample(_COLLISION, **_SQLITE)
    latency = _sample(_LATENCY_COUNT, **_SQLITE)
    count = 20
    await asyncio.gather(*(dialect.execute("INSERT INTO t VALUES (?)", (i,)) for i in range(count)))
    assert _sample(_COLLISION, **_SQLITE) == before
    assert _sample(_LATENCY_COUNT, **_SQLITE) == latency + count
    rows = await dialect.fetch_all("SELECT x FROM t ORDER BY x")
    assert [int(row["x"]) for row in rows] == list(range(count))


def test_render_includes_the_series_names() -> None:
    pytest.importorskip("prometheus_client")
    assert metrics_enabled()
    body, content_type = render_metrics()
    text = body.decode()
    assert "text/plain" in content_type
    assert "neural_memory_sqlite_write_collision_total" in text
    assert "neural_memory_sqlite_write_latency_seconds" in text
    assert "neural_memory_cli_encoding_damage_rejected_total" in text
    assert "neural_memory_auto_capture_empty_total" in text


def test_render_comment_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import neural_memory.utils.metrics as metrics

    monkeypatch.setattr(metrics, "_INSTRUMENTS", None)
    body, content_type = metrics.render_metrics()
    assert b"metrics disabled" in body
    assert content_type.startswith("text/plain")
    metrics.record_sqlite_write(0.01, collision=True)


@pytest.fixture
def metrics_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    import neural_memory.unified_config as uc

    monkeypatch.setenv("NEURALMEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("NEURAL_MEMORY_API_KEY", "metrics-test-key")
    monkeypatch.setenv("NEURAL_MEMORY_RATE_LIMIT", "1")
    saved_cache = uc._storage_cache.copy()
    uc._storage_cache.clear()
    monkeypatch.setattr(uc, "_config", None)
    try:
        app = create_app()
        with TestClient(app) as client:
            yield client
    finally:
        uc._storage_cache.clear()
        uc._storage_cache.update(saved_cache)


def test_metrics_endpoint_is_public(metrics_client: TestClient) -> None:
    first = metrics_client.get("/metrics")
    second = metrics_client.get("/metrics")
    blocked = metrics_client.get("/api/v1/does-not-exist")

    assert first.status_code == 200
    assert second.status_code == 200
    assert "text/plain" in first.headers["content-type"]
    assert "neural_memory_sqlite_write_collision_total" in first.text
    assert blocked.status_code == 401
