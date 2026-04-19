"""System health diagnostic — nmem doctor.

Checks Python version, dependencies, config validity, brain accessibility,
embedding provider, storage integrity, schema version, hooks, dedup,
and knowledge surface. Produces green/yellow/red status per check
with actionable fix suggestions. Supports --fix for auto-remediation.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from neural_memory.cli._helpers import run_async

# Check result constants
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

# Priority tiers (used to group output so users know which warnings actually matter)
TIER_CORE = "core"  # required for basic operation
TIER_RECOMMENDED = "recommended"  # recommended for full power (embeddings, MCP, hooks)
TIER_OPTIONAL = "optional"  # nice-to-have (surface snapshot, pro plugin, etc.)

# Check name → tier mapping. Unassigned defaults to RECOMMENDED.
_CHECK_TIERS: dict[str, str] = {
    "Python version": TIER_CORE,
    "Configuration": TIER_CORE,
    "Brain database": TIER_CORE,
    "Dependencies": TIER_CORE,
    "Schema version": TIER_CORE,
    "CLI tools": TIER_CORE,
    "Embedding provider": TIER_RECOMMENDED,
    "MCP configuration": TIER_RECOMMENDED,
    "MCP server": TIER_RECOMMENDED,
    "Hooks": TIER_RECOMMENDED,
    "Dedup": TIER_OPTIONAL,
    "Knowledge surface": TIER_OPTIONAL,
    "Config freshness": TIER_OPTIONAL,
    "Pro features": TIER_OPTIONAL,
}

QUICKSTART_URL = "https://nhadaututtheky.github.io/neural-memory/guides/quickstart/"


def run_doctor(*, json_output: bool = False, fix: bool = False) -> dict[str, Any]:
    """Run all diagnostic checks and return results.

    Args:
        json_output: Return machine-readable output.
        fix: Auto-fix what's possible (enable config flags, install hooks).
    """
    checks: list[dict[str, Any]] = []

    checks.append(_check_python_version())
    checks.append(_check_config())
    checks.append(_check_brain())
    checks.append(_check_dependencies())
    checks.append(_check_embedding_provider())
    checks.append(_check_schema_version())
    checks.append(_check_mcp_config())
    checks.append(_check_mcp_connection())
    checks.append(_check_hooks())
    checks.append(_check_dedup())
    checks.append(_check_surface())
    checks.append(_check_config_freshness())
    checks.append(_check_cli_tools())
    checks.append(_check_pro_plugin())

    # Auto-fix pass
    if fix:
        checks = _auto_fix(checks)

    # Annotate each check with its priority tier
    for c in checks:
        c["tier"] = _CHECK_TIERS.get(c.get("name", ""), TIER_RECOMMENDED)

    core_warn_fail = sum(
        1 for c in checks if c["tier"] == TIER_CORE and c["status"] in (WARN, FAIL)
    )

    result = {
        "checks": checks,
        "passed": sum(1 for c in checks if c["status"] == OK),
        "warnings": sum(1 for c in checks if c["status"] == WARN),
        "failed": sum(1 for c in checks if c["status"] == FAIL),
        "total": len(checks),
        "core_issues": core_warn_fail,
    }

    if not json_output:
        _render_results(result)

    return result


def _check_python_version() -> dict[str, Any]:
    """Check Python version is 3.11+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 11):
        return {"name": "Python version", "status": OK, "detail": version_str}

    return {
        "name": "Python version",
        "status": FAIL,
        "detail": f"{version_str} (requires 3.11+)",
        "fix": "Install Python 3.11 or newer",
    }


def _check_config() -> dict[str, Any]:
    """Check config.toml exists and is valid."""
    from neural_memory.unified_config import get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()
    config_path = data_dir / "config.toml"

    if not config_path.exists():
        return {
            "name": "Configuration",
            "status": FAIL,
            "detail": f"{config_path} not found",
            "fix": "Run: nmem init",
        }

    try:
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        return {
            "name": "Configuration",
            "status": OK,
            "detail": f"{config_path} (brain: {config.current_brain})",
        }
    except Exception:
        return {
            "name": "Configuration",
            "status": FAIL,
            "detail": "parse error — run: nmem init --force",
            "fix": "Run: nmem init --force",
        }


def _check_brain() -> dict[str, Any]:
    """Check default brain DB exists and is accessible."""
    from neural_memory.unified_config import get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()

    try:
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        brain_name = config.current_brain
    except Exception:
        brain_name = "default"

    brains_dir = data_dir / "brains"
    db_path = brains_dir / f"{brain_name}.db"

    if not db_path.exists():
        return {
            "name": "Brain database",
            "status": FAIL,
            "detail": f"{db_path} not found",
            "fix": f"Run: nmem brain create {brain_name}",
        }

    size_kb = db_path.stat().st_size / 1024
    return {
        "name": "Brain database",
        "status": OK,
        "detail": f"{brain_name} ({size_kb:.0f} KB)",
    }


def _check_dependencies() -> dict[str, Any]:
    """Check core dependencies are importable."""
    required = ["aiosqlite", "typer"]
    missing = []

    for dep in required:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        return {
            "name": "Dependencies",
            "status": FAIL,
            "detail": f"Missing: {', '.join(missing)}",
            "fix": "Run: pip install neural-memory",
        }

    return {"name": "Dependencies", "status": OK, "detail": "all core deps available"}


def _check_embedding_provider() -> dict[str, Any]:
    """Check embedding provider availability."""
    try:
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
    except Exception:
        return {
            "name": "Embedding provider",
            "status": SKIP,
            "detail": "config not loaded",
        }

    if not config.embedding.enabled:
        return {
            "name": "Embedding provider",
            "status": WARN,
            "detail": "disabled (semantic search unavailable)",
            "fix": "Run: nmem setup embeddings",
        }

    provider = config.embedding.provider

    # Check if provider package is importable
    provider_checks: dict[str, str] = {
        "sentence_transformer": "sentence_transformers",
        "openai": "openai",
        "openrouter": "openai",
        "gemini": "google.genai",
        "ollama": "ollama",
    }

    module_name = provider_checks.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
            return {
                "name": "Embedding provider",
                "status": OK,
                "detail": f"{provider} (model: {config.embedding.model})",
            }
        except ImportError:
            install_hint = {
                "sentence_transformer": "pip install neural-memory[embeddings]",
                "openai": "pip install neural-memory[embeddings-openai]",
                "openrouter": "pip install neural-memory[embeddings-openrouter]",
                "gemini": "pip install neural-memory[embeddings-gemini]",
                "ollama": "pip install neural-memory[embeddings]",
            }
            return {
                "name": "Embedding provider",
                "status": FAIL,
                "detail": f"{provider} configured but not installed",
                "fix": f"Run: {install_hint.get(provider, 'pip install neural-memory[embeddings]')}",
            }

    return {
        "name": "Embedding provider",
        "status": OK,
        "detail": f"{provider} (model: {config.embedding.model})",
    }


def _check_schema_version() -> dict[str, Any]:
    """Check database schema version."""
    try:
        from neural_memory.unified_config import get_config, get_neuralmemory_dir

        config = get_config(reload=True)
        brain_name = config.current_brain
        db_path = get_neuralmemory_dir() / "brains" / f"{brain_name}.db"

        if not db_path.exists() or db_path.stat().st_size == 0:
            return {
                "name": "Schema version",
                "status": SKIP,
                "detail": "empty database (schema created on first use)",
            }

        async def _get_version() -> int:
            import aiosqlite

            async with aiosqlite.connect(str(db_path)) as db:
                # NM stores schema version in schema_version table, not PRAGMA
                try:
                    cursor = await db.execute("SELECT version FROM schema_version LIMIT 1")
                    row = await cursor.fetchone()
                    return row[0] if row else 0
                except Exception:
                    # Table may not exist in very old databases
                    return 0

        version = run_async(_get_version())

        from neural_memory.storage.sqlite_schema import SCHEMA_VERSION as CURRENT_VERSION

        if version == CURRENT_VERSION:
            return {
                "name": "Schema version",
                "status": OK,
                "detail": f"v{version} (current)",
            }
        if version < CURRENT_VERSION:
            return {
                "name": "Schema version",
                "status": WARN,
                "detail": f"v{version} (latest: v{CURRENT_VERSION})",
                "fix": (
                    "Auto-migrates on next read/write — run any command "
                    "(e.g. 'nmem recall \"test\"' or 'nmem doctor --fix') to trigger now"
                ),
            }
        return {
            "name": "Schema version",
            "status": WARN,
            "detail": f"v{version} (newer than expected v{CURRENT_VERSION})",
        }
    except Exception:
        return {
            "name": "Schema version",
            "status": WARN,
            "detail": "could not check — ensure brain is accessible",
        }


def _check_mcp_config() -> dict[str, Any]:
    """Check MCP server is configured in Claude Code."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "~/.claude.json not found",
            "fix": "Run: nmem init",
        }

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        if "neural-memory" in servers:
            return {
                "name": "MCP configuration",
                "status": OK,
                "detail": "neural-memory registered in Claude Code",
            }
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "neural-memory not found in ~/.claude.json",
            "fix": "Run: nmem init",
        }
    except (json.JSONDecodeError, OSError):
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "could not parse ~/.claude.json",
            "fix": "Run: nmem init",
        }


def _check_mcp_connection() -> dict[str, Any]:
    """Test that the MCP server can actually start."""
    import subprocess

    nmem_mcp = shutil.which("nmem-mcp")
    if not nmem_mcp:
        # Fallback to module execution
        nmem_mcp = None

    try:
        cmd = [nmem_mcp] if nmem_mcp else [sys.executable, "-m", "neural_memory.mcp"]
        result = subprocess.run(
            cmd,
            input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"doctor","version":"1.0"}}}\n',
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # MCP server over stdio will output JSON-RPC response
        if result.stdout and "result" in result.stdout:
            return {
                "name": "MCP server",
                "status": OK,
                "detail": "server responds to initialize",
            }
        if result.returncode == 0 or result.stdout:
            return {
                "name": "MCP server",
                "status": OK,
                "detail": "server starts successfully",
            }
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": f"server exited with code {result.returncode}",
            "fix": "Check: nmem-mcp or python -m neural_memory.mcp",
        }
    except subprocess.TimeoutExpired:
        # Timeout is actually expected — MCP servers run indefinitely on stdio
        return {
            "name": "MCP server",
            "status": OK,
            "detail": "server starts (stdio mode)",
        }
    except FileNotFoundError:
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": "nmem-mcp not found on PATH",
            "fix": "Run: pip install neural-memory",
        }
    except Exception:
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": "could not test MCP server",
        }


def _check_cli_tools() -> dict[str, Any]:
    """Check CLI tools are on PATH."""
    tools = ["nmem", "nmem-mcp"]
    found = [t for t in tools if shutil.which(t)]
    missing = [t for t in tools if t not in found]

    if not missing:
        return {
            "name": "CLI tools",
            "status": OK,
            "detail": "nmem + nmem-mcp on PATH",
        }

    if "nmem" in missing:
        return {
            "name": "CLI tools",
            "status": FAIL,
            "detail": f"missing: {', '.join(missing)}",
            "fix": "Run: pip install neural-memory",
        }

    return {
        "name": "CLI tools",
        "status": WARN,
        "detail": f"missing: {', '.join(missing)} (nmem mcp fallback available)",
    }


def _check_pro_plugin() -> dict[str, Any]:
    """Check if Neural Memory Pro plugin is installed and active."""
    try:
        from neural_memory.plugins import get_plugins, has_pro

        if has_pro():
            plugins = get_plugins()
            names = [f"{p.name} v{p.version}" for p in plugins]
            return {
                "name": "Pro plugin",
                "status": OK,
                "detail": ", ".join(names),
            }

        # Check if Pro deps are available (built-in Pro)
        from neural_memory.pro import is_pro_deps_installed

        if is_pro_deps_installed():
            return {
                "name": "Pro features",
                "status": OK,
                "detail": "Pro dependencies installed (numpy, hnswlib, msgpack)",
            }

        # Check license
        from neural_memory.unified_config import get_config

        config = get_config()
        if config.is_pro():
            return {
                "name": "Pro features",
                "status": WARN,
                "detail": "License active but Pro deps not installed",
                "fix": "Run: pip install neural-memory",
            }

        return {
            "name": "Pro features",
            "status": SKIP,
            "detail": "Not installed (free tier)",
        }
    except Exception:
        return {
            "name": "Pro plugin",
            "status": SKIP,
            "detail": "Could not check",
        }


def _check_hooks() -> dict[str, Any]:
    """Check Claude Code hooks are installed."""
    claude_dir = Path.home() / ".claude"
    settings_path = claude_dir / "settings.json"

    if not settings_path.exists():
        return {
            "name": "Hooks",
            "status": WARN,
            "detail": "~/.claude/settings.json not found",
            "fix": "Run: nmem init",
            "fixable": True,
        }

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "name": "Hooks",
            "status": WARN,
            "detail": "could not parse settings.json",
        }

    hooks_section = data.get("hooks", {})
    expected = ["PreCompact", "Stop", "PostToolUse"]
    found: list[str] = []

    for event in expected:
        entries = hooks_section.get(event, [])
        for entry in entries:
            for hook in entry.get("hooks", []):
                cmd = hook.get("command", "")
                if "neural_memory" in cmd or "nmem" in cmd:
                    found.append(event)
                    break

    if len(found) == len(expected):
        return {
            "name": "Hooks",
            "status": OK,
            "detail": f"{len(found)}/{len(expected)} installed ({', '.join(found)})",
        }

    missing = [e for e in expected if e not in found]
    return {
        "name": "Hooks",
        "status": WARN,
        "detail": f"{len(found)}/{len(expected)} — missing: {', '.join(missing)}",
        "fix": "Run: nmem init",
        "fixable": True,
    }


def _check_dedup() -> dict[str, Any]:
    """Check dedup is enabled in config."""
    try:
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
    except Exception:
        return {"name": "Dedup", "status": SKIP, "detail": "config not loaded"}

    if config.dedup.enabled:
        return {"name": "Dedup", "status": OK, "detail": "enabled"}

    return {
        "name": "Dedup",
        "status": WARN,
        "detail": "disabled (duplicate memories not caught)",
        "fix": "Run: nmem init --full",
        "fixable": True,
    }


def _check_surface() -> dict[str, Any]:
    """Check knowledge surface (.nm file) exists."""
    try:
        from neural_memory.surface.resolver import get_surface_path
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        surface_path = get_surface_path(config.current_brain)

        if surface_path.exists():
            size_kb = surface_path.stat().st_size / 1024
            return {
                "name": "Knowledge surface",
                "status": OK,
                "detail": f"{surface_path.name} ({size_kb:.1f} KB)",
            }

        return {
            "name": "Knowledge surface",
            "status": WARN,
            "detail": "not generated yet",
            "fix": "Run: nmem surface generate (via MCP or after first session)",
        }
    except Exception:
        return {
            "name": "Knowledge surface",
            "status": SKIP,
            "detail": "surface module not available",
        }


def _check_config_freshness() -> dict[str, Any]:
    """Check if config.toml has all sections from current version."""
    try:
        import tomllib

        from neural_memory.unified_config import get_neuralmemory_dir

        config_path = get_neuralmemory_dir() / "config.toml"
        if not config_path.exists():
            return {
                "name": "Config freshness",
                "status": SKIP,
                "detail": "no config.toml",
            }

        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        expected_sections = [
            "brain",
            "embedding",
            "auto",
            "eternal",
            "maintenance",
            "conflict",
            "safety",
            "encryption",
            "write_gate",
            "dedup",
            "tool_memory",
        ]
        missing = [s for s in expected_sections if s not in raw]
        if missing:
            return {
                "name": "Config freshness",
                "status": WARN,
                "detail": f"missing sections: {', '.join(missing)}",
                "fix": "Run: nmem doctor --fix",
                "fixable": True,
            }

        return {
            "name": "Config freshness",
            "status": OK,
            "detail": "all sections present",
        }
    except Exception:
        return {
            "name": "Config freshness",
            "status": SKIP,
            "detail": "could not check config freshness",
        }


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def _auto_fix(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attempt to auto-fix fixable issues. Returns updated checks."""
    fixed_checks: list[dict[str, Any]] = []

    for check in checks:
        if check.get("fixable") and check["status"] in (WARN, FAIL):
            fixed = _try_fix(check)
            if fixed:
                fixed_checks.append(fixed)
                continue
        fixed_checks.append(check)

    return fixed_checks


def _try_fix(check: dict[str, Any]) -> dict[str, Any] | None:
    """Try to fix a single check. Returns updated check or None."""
    name = check["name"]

    handler = _FIX_HANDLERS.get(name)
    if handler is None:
        return None
    if name == "Embedding provider" and "disabled" not in check.get("detail", ""):
        return None
    result: dict[str, Any] | None = handler()
    if result:
        result["_fixed"] = True
    return result


_FIX_HANDLERS: dict[str, Any] = {
    "Hooks": lambda: _fix_hooks(),
    "Dedup": lambda: _fix_dedup(),
    "Embedding provider": lambda: _fix_embedding(),
    "Config freshness": lambda: _fix_config_freshness(),
}


def _fix_hooks() -> dict[str, Any]:
    """Auto-fix: install missing hooks."""
    try:
        from neural_memory.cli.setup import setup_hooks_claude

        status = setup_hooks_claude()
        if status in ("added", "exists"):
            return {
                "name": "Hooks",
                "status": OK,
                "detail": "auto-fixed: hooks installed",
            }
    except Exception:
        pass
    return {
        "name": "Hooks",
        "status": WARN,
        "detail": "auto-fix failed",
        "fix": "Run: nmem init",
    }


def _fix_dedup() -> dict[str, Any]:
    """Auto-fix: enable dedup in config."""
    try:
        from dataclasses import replace

        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        updated = replace(config, dedup=replace(config.dedup, enabled=True))
        updated.save()
        return {
            "name": "Dedup",
            "status": OK,
            "detail": "auto-fixed: enabled",
        }
    except Exception:
        pass
    return {
        "name": "Dedup",
        "status": WARN,
        "detail": "auto-fix failed",
    }


def _fix_embedding() -> dict[str, Any]:
    """Auto-fix: detect and enable embedding provider."""
    try:
        from neural_memory.cli.full_setup import detect_embedding_provider, enable_config_defaults

        provider = detect_embedding_provider()
        if provider:
            enable_config_defaults(embedding_provider=provider)
            return {
                "name": "Embedding provider",
                "status": OK,
                "detail": f"auto-fixed: {provider['key']} enabled",
            }
    except Exception:
        pass
    return {
        "name": "Embedding provider",
        "status": WARN,
        "detail": "no provider available to auto-enable",
        "fix": "Run: nmem setup embeddings",
    }


def _fix_config_freshness() -> dict[str, Any]:
    """Auto-fix: re-save config.toml to add missing sections with defaults."""
    try:
        from neural_memory.unified_config import get_config

        config = get_config(reload=True)
        config.save()
        return {
            "name": "Config freshness",
            "status": OK,
            "detail": "auto-fixed: config.toml updated with new sections",
        }
    except Exception:
        pass
    return {
        "name": "Config freshness",
        "status": WARN,
        "detail": "auto-fix failed",
    }


def _render_results(result: dict[str, Any]) -> None:
    """Render diagnostic results to terminal, grouped by priority tier."""
    typer.echo()
    typer.secho("  NeuralMemory Doctor", bold=True)
    typer.secho("  ───────────────────", dim=True)
    typer.echo()

    icons = {
        OK: typer.style("[OK]", fg=typer.colors.GREEN),
        WARN: typer.style("[!!]", fg=typer.colors.YELLOW),
        FAIL: typer.style("[XX]", fg=typer.colors.RED),
        SKIP: typer.style("[--]", fg=typer.colors.BRIGHT_BLACK),
    }

    tier_labels = {
        TIER_CORE: ("CORE", "required for basic operation"),
        TIER_RECOMMENDED: ("RECOMMENDED", "full-power setup"),
        TIER_OPTIONAL: ("OPTIONAL", "nice-to-have, not needed for basic use"),
    }

    for tier in (TIER_CORE, TIER_RECOMMENDED, TIER_OPTIONAL):
        tier_checks = [c for c in result["checks"] if c.get("tier") == tier]
        if not tier_checks:
            continue
        label, hint = tier_labels[tier]
        typer.secho(f"  [{label}] ", fg=typer.colors.CYAN, bold=True, nl=False)
        typer.secho(hint, dim=True)
        for check in tier_checks:
            icon = icons.get(check["status"], icons[SKIP])
            typer.echo(f"    {icon} {check['name']:<22}{check['detail']}")
            if "fix" in check:
                typer.secho(f"         Fix: {check['fix']}", dim=True)
        typer.echo()

    passed = result["passed"]
    total = result["total"]
    warns = result["warnings"]
    fails = result["failed"]
    core_issues = result.get("core_issues", 0)

    summary_parts = [f"{passed}/{total} passed"]
    if warns:
        summary_parts.append(f"{warns} warnings")
    if fails:
        summary_parts.append(f"{fails} failed")

    color = (
        typer.colors.RED
        if fails > 0
        else (typer.colors.YELLOW if warns > 0 else typer.colors.GREEN)
    )
    typer.secho(f"  {', '.join(summary_parts)}", fg=color, bold=True)

    if (warns > 0 or fails > 0) and core_issues == 0:
        typer.secho(
            "  All CORE checks green — warnings are in RECOMMENDED/OPTIONAL tiers.",
            fg=typer.colors.GREEN,
            dim=True,
        )

    # Suggest guide if there are issues
    if warns > 0 or fails > 0:
        typer.echo()
        typer.secho(f"  See full setup guide: {QUICKSTART_URL}", dim=True)
        if not any(c.get("_fixed") for c in result["checks"]):
            typer.secho("  Auto-fix available issues: nmem doctor --fix", dim=True)

    typer.echo()
