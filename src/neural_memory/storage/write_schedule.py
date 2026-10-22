"""Schedule independent storage writes without breaking SQLite's single writer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import TypeVar

_T = TypeVar("_T")


def writes_run_concurrently(storage: object) -> bool:
    """Return whether independent writes on ``storage`` may overlap.

    Postgres checks out a pooled connection per call. SQLite has one writer
    connection, and the legacy store has no dialect queue, so those callers
    stay sequential. The SQLite dialect still serializes any write that does
    reach it.
    """
    dialect = getattr(storage, "_dialect", None)
    return getattr(dialect, "name", None) == "postgres"


async def run_writes(storage: object, coros: Sequence[Awaitable[_T]]) -> list[_T]:
    """Await every coroutine. Overlap them only on Postgres.

    Results stay in input order. The caller owns each coroutine: this does
    not retry, and the first error cancels the rest when they overlap.
    """
    if not coros:
        return []
    if len(coros) == 1 or not writes_run_concurrently(storage):
        results: list[_T] = []
        for coro in coros:
            results.append(await coro)
        return results
    return list(await asyncio.gather(*coros))
