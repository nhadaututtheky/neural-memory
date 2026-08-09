"""Smoke-check install profiles without full isolated venv (fast path).

Validates import contracts and optional capability gates. For true isolated
wheel installs, CI may invoke this inside a clean venv after
``pip install .`` / ``pip install '.[server]'``.

Usage:
    python scripts/check_install_profiles.py --profile base
    python scripts/check_install_profiles.py --profile base --profile server --profile sync
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def check_base() -> dict:
    report: dict = {"profile": "base", "ok": True, "checks": []}

    def ok(name: str, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": True, "detail": detail})

    def fail(name: str, detail: str) -> None:
        report["ok"] = False
        report["checks"].append({"name": name, "ok": False, "detail": detail})

    try:
        import neural_memory

        ok("import_neural_memory", neural_memory.__version__)
    except Exception as exc:
        fail("import_neural_memory", str(exc))
        return report

    try:
        from neural_memory.storage.memory_store import InMemoryStorage

        ok("import_inmemory", InMemoryStorage.__name__)
    except Exception as exc:
        fail("import_inmemory", str(exc))

    try:
        from neural_memory.utils.optional_dependencies import has_capability

        # Base must not *require* aiohttp/pydantic at import time
        ok(
            "optional_not_required_at_import",
            f"aiohttp={has_capability('aiohttp')} pydantic={has_capability('pydantic')}",
        )
    except Exception as exc:
        fail("optional_capability_check", str(exc))

    async def _encode_recall() -> None:
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.engine.retrieval import ReflexPipeline
        from neural_memory.storage.sql.sql_storage import SQLStorage
        from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLStorage(SQLiteDialect(f"{tmp}/p.db"))
            await store.initialize()
            brain = Brain.create(name="profile", config=BrainConfig())
            await store.save_brain(brain)
            store.set_brain(brain.id)
            enc = MemoryEncoder(store, brain.config)
            await enc.encode("Profile smoke: chose SQLite for zero-ops local memory.")
            pipe = ReflexPipeline(store, brain.config)
            result = await pipe.query("SQLite local memory")
            await store.close()
            assert result is not None

    try:
        asyncio.run(_encode_recall())
        ok("sqlite_encode_recall", "pass")
    except Exception as exc:
        fail("sqlite_encode_recall", str(exc))

    try:
        from neural_memory.mcp.tool_schemas import get_tool_schemas_for_tier

        std = get_tool_schemas_for_tier("standard")
        full = get_tool_schemas_for_tier("full")
        ok("mcp_tiers", f"standard={len(std)} full={len(full)}")
        if len(std) != 10:
            fail("standard_tier_count", f"expected 10 got {len(std)}")
    except Exception as exc:
        fail("mcp_tiers", str(exc))

    return report


def check_server() -> dict:
    report: dict = {"profile": "server", "ok": True, "checks": []}
    try:
        from neural_memory.utils.optional_dependencies import has_capability

        if not has_capability("fastapi") or not has_capability("pydantic"):
            report["ok"] = False
            report["checks"].append(
                {
                    "name": "server_extras",
                    "ok": False,
                    "detail": "fastapi/pydantic missing — install neural-memory[server]",
                }
            )
            return report
        from neural_memory.server.app import create_app

        app = create_app()
        report["checks"].append({"name": "create_app", "ok": True, "detail": type(app).__name__})
    except Exception as exc:
        report["ok"] = False
        report["checks"].append({"name": "server", "ok": False, "detail": str(exc)})
    return report


def check_sync() -> dict:
    report: dict = {"profile": "sync", "ok": True, "checks": []}
    try:
        from neural_memory.utils.optional_dependencies import has_capability, require_capability

        if not has_capability("aiohttp"):
            report["ok"] = False
            report["checks"].append(
                {
                    "name": "sync_extra",
                    "ok": False,
                    "detail": "aiohttp missing — install neural-memory[sync]",
                }
            )
            return report
        mod = require_capability("aiohttp", "sync", "sync profile")
        report["checks"].append({"name": "aiohttp", "ok": True, "detail": mod.__name__})
        from neural_memory.sync.client import SyncClient
        from neural_memory.storage.shared_store import SharedStorage

        report["checks"].append(
            {
                "name": "sync_classes",
                "ok": True,
                "detail": f"{SyncClient.__name__},{SharedStorage.__name__}",
            }
        )
    except Exception as exc:
        report["ok"] = False
        report["checks"].append({"name": "sync", "ok": False, "detail": str(exc)})
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        action="append",
        choices=["base", "server", "sync"],
        default=None,
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    profiles = args.profile or ["base"]
    runners = {"base": check_base, "server": check_server, "sync": check_sync}
    reports = [runners[name]() for name in profiles]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            status = "PASS" if r["ok"] else "FAIL"
            print(f"[{status}] profile={r['profile']}")
            for c in r.get("checks", []):
                mark = "ok" if c["ok"] else "FAIL"
                print(f"  - {mark}: {c['name']} {c.get('detail', '')}")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
