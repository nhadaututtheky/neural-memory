"""NeuralMemory MCP client — port of integrations/neuralmemory/src/mcp-client.ts.

JSON-RPC 2.0 over stdio, newline-delimited JSON Lines. Spawns
`python -m neural_memory.mcp` and manages its lifecycle.

Faithful port: same protocol constants, same env allowlist, same buffer
cap, same timeout defaults, same error messages.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from typing import Any

logger = logging.getLogger("hermes.plugins.neuralmemory")

# ── Constants (ported verbatim from mcp-client.ts) ────────────

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TIMEOUT = 30_000  # ms
CLIENT_NAME = "hermes-neuralmemory"
CLIENT_VERSION = "1.0.0"
MAX_BUFFER_BYTES = 10 * 1024 * 1024  # 10 MB safety cap
MAX_STDERR_LINES = 50

# Env vars forwarded to the MCP child process (least-privilege).
ALLOWED_ENV_KEYS: frozenset[str] = frozenset([
    "PATH",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "PYTHONHOME",
    "NEURALMEMORY_DIR",
    "NEURALMEMORY_BRAIN",
    "NEURAL_MEMORY_DIR",
    "NEURAL_MEMORY_JSON",
    "NEURAL_MEMORY_DEBUG",
])


def build_child_env(brain: str) -> dict[str, str]:
    """Build a minimal env for the child process (least-privilege). Port of buildChildEnv."""
    env: dict[str, str] = {}
    for key in ALLOWED_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    if brain != "default":
        env["NEURALMEMORY_BRAIN"] = brain
    return env


class _Pending:
    __slots__ = ("error", "event", "result")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: Exception | None = None


class NeuralMemoryMcpClient:
    """Python port of NeuralMemoryMcpClient (mcp-client.ts).

    Thread-safe: Hermes runs tools and hooks from multiple threads.
    """

    def __init__(self, python_path: str, brain: str, timeout: int = DEFAULT_TIMEOUT,
                 init_timeout: int = 90_000) -> None:
        self._python_path = python_path
        self._brain = brain
        self._timeout = timeout
        self._init_timeout = init_timeout
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._pending: dict[int, _Pending] = {}
        self._lock = threading.Lock()          # guards _proc/_pending/_request_id writes
        self._write_lock = threading.Lock()    # serializes stdin writes
        self._connected = False
        self._connecting: threading.Event | None = None
        self._connect_error: Exception | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Connection management ─────────────────────────

    def ensure_connected(self) -> None:
        """Ensure connected; concurrent callers share one connection attempt.

        Port of ensureConnected().
        """
        if self._connected:
            return
        connector = False
        with self._lock:
            if self._connected:
                return
            if self._connecting is not None:
                gate = self._connecting
            else:
                gate = threading.Event()
                self._connecting = gate
                self._connect_error = None
                connector = True
        if not connector:
            gate.wait(self._init_timeout / 1000 + 30)
            if self._connect_error:
                raise self._connect_error
            if not self._connected:
                raise RuntimeError("NeuralMemory MCP connect timed out")
            return
        try:
            self.connect()
            self._connected = True
        except Exception as exc:
            self._connect_error = exc
            raise
        finally:
            with self._lock:
                self._connecting = None
            gate.set()

    def connect(self) -> None:
        """Spawn the MCP process and complete the initialize handshake. Port of connect()."""
        env = build_child_env(self._brain)
        # Generation-scoped stderr capture (reviewer M3-5): a previous generation's
        # stderr thread must never append into the new generation's diagnostics.
        self._stderr_lines: list[str] = []
        stderr_lines = self._stderr_lines

        try:
            self._proc = subprocess.Popen(  # noqa: S603 — pythonPath comes from validated plugin config; args are static
                [self._python_path, "-m", "neural_memory.mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f'MCP process error: "{self._python_path}" not found. '
                "Check pythonPath in plugin config."
            ) from exc

        # Reader threads for stdout (JSON-RPC) and stderr (diagnostics)
        threading.Thread(target=self._read_stdout, daemon=True,
                         name="nmem-mcp-stdout").start()
        threading.Thread(target=self._read_stderr, args=(stderr_lines,), daemon=True,
                         name="nmem-mcp-stderr").start()
        threading.Thread(target=self._wait_exit, daemon=True,
                         name="nmem-mcp-exit").start()

        # MCP initialize handshake (uses longer timeout for cold starts)
        try:
            self.send("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            }, timeout_override=self._init_timeout)
        except Exception as err:
            # Reviewer M2: tear down the failed generation — a hung child and its
            # reader threads must not leak, and a later retry must not orphan it.
            self._teardown_failed_connect()
            stderr = "\n".join(stderr_lines)
            detail = (f"\nPython stderr:\n{stderr}" if stderr
                      else "\nNo stderr output — the Python process may have hung.")
            raise RuntimeError(
                f"MCP initialize failed: {err}{detail}\n"
                f"Verify: {self._python_path} -m neural_memory.mcp"
            ) from err

        # Send initialized notification (no response expected)
        self._notify("notifications/initialized", {})
        self._connected = True
        logger.info("MCP connected (brain: %s, protocol: %s)", self._brain, PROTOCOL_VERSION)

    def list_tools(self) -> list[dict]:
        """Fetch all available tools via tools/list. Port of listTools()."""
        result = self.send("tools/list", {})
        if not isinstance(result, dict):
            return []
        return result.get("tools") or []

    def call_tool(self, name: str, args: dict | None = None) -> str:
        """Call an MCP tool and return its first text content. Port of callTool()."""
        result = self.send("tools/call", {"name": name, "arguments": args or {}})
        if not isinstance(result, dict):
            return ""
        if result.get("isError"):
            content = result.get("content") or []
            text = content[0].get("text") if content and isinstance(content[0], dict) else None
            raise RuntimeError(text or "Unknown MCP error")
        content = result.get("content") or []
        if content and isinstance(content[0], dict):
            return content[0].get("text") or ""
        return ""

    def close(self) -> None:
        """Terminate the child process. Port of close()."""
        self._connected = False
        self._reject_all(RuntimeError("Client closing"))
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:  # noqa: BLE001, S110 — best-effort teardown
                pass
        logger.info("MCP client closed")

    def _teardown_failed_connect(self) -> None:
        """Kill a child whose handshake failed (reviewer M2).

        Popen succeeded but initialize failed/timed out — without this the
        hung child and its reader threads leak, and a later ensure_connected
        retry would overwrite self._proc and orphan the first process.
        """
        self._reject_all(RuntimeError("MCP initialize failed — connect aborted"))
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:  # noqa: BLE001, S110 — best-effort teardown
                pass

    # ── JSON-RPC protocol layer ──────────────────────────

    def send(self, method: str, params: Any, timeout_override: int | None = None) -> Any:
        """Send a request and wait for its response. Port of send()."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("MCP process not available")

        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            pending = _Pending()
            self._pending[req_id] = pending

        ms = timeout_override if timeout_override is not None else self._timeout
        self._write_message({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

        if not pending.event.wait(ms / 1000):
            with self._lock:
                self._pending.pop(req_id, None)
            raise RuntimeError(f"MCP timeout: {method} ({ms}ms)")

        if pending.error is not None:
            raise pending.error
        return pending.result

    def _notify(self, method: str, params: Any) -> None:
        self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def _write_message(self, message: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        frame = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                proc.stdin.write(frame)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    # ── Reader threads ───────────────────────────────

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stdout = proc.stdout
        # Port of mcp-client.ts drainBuffer(): cap the UNDRAINED buffer only.
        # Consumed bytes are removed as lines are processed, so a long-lived
        # connection never trips the cap on well-formed traffic.
        raw_buffer = b""
        for chunk in iter(lambda: stdout.read(65536), b""):
            raw_buffer += chunk
            if len(raw_buffer) > MAX_BUFFER_BYTES:
                logger.error("MCP buffer exceeded %d bytes — killing process", MAX_BUFFER_BYTES)
                try:
                    proc.kill()
                except Exception:
                    pass
                return
            raw_buffer = self._drain_buffer(raw_buffer)

    def _drain_buffer(self, raw_buffer: bytes) -> bytes:
        """Parse complete newline-delimited messages; return the undrained remainder."""
        while True:
            newline_index = raw_buffer.find(b"\n")
            if newline_index == -1:
                break
            line = raw_buffer[:newline_index]
            raw_buffer = raw_buffer[newline_index + 1:]
            self._handle_line(line)
        return raw_buffer

    def _read_stderr(self, stderr_lines: list[str]) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            msg = raw.decode("utf-8", errors="replace").strip()
            if msg:
                if len(stderr_lines) < MAX_STDERR_LINES:
                    stderr_lines.append(msg)
                logger.warning("[mcp stderr] %s", msg)

    def _wait_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        self._connected = False
        hint = " — check that neural-memory is installed: pip install neural-memory" if code == 1 else ""
        self._reject_all(RuntimeError(f"MCP process exited with code {code}{hint}"))
        logger.error("MCP process exited (code: %s)%s", code, hint)

    def _handle_line(self, line: bytes) -> None:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            message = json.loads(text)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse MCP message: %s", err)
            return
        self._handle_message(message)

    def _handle_message(self, message: dict) -> None:
        # Notifications (no id) — ignore silently
        if message.get("id") is None:
            return
        with self._lock:
            pending = self._pending.pop(message.get("id"), None)
        if pending is None:
            return
        error = message.get("error")
        if error:
            pending.error = RuntimeError(f"MCP error {error.get('code')}: {error.get('message')}")
        else:
            pending.result = message.get("result")
        pending.event.set()

    def _reject_all(self, error: Exception) -> None:
        with self._lock:
            pendings = list(self._pending.values())
            self._pending.clear()
        for pending in pendings:
            pending.error = error
            pending.event.set()
