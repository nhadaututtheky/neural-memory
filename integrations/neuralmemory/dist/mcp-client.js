/**
 * NeuralMemory MCP Client — JSON-RPC 2.0 over stdio.
 *
 * Spawns `python -m neural_memory.mcp` and communicates using the
 * MCP protocol (newline-delimited JSON Lines).
 *
 * Zero external dependencies — implements the protocol directly.
 */
import { spawn } from "node:child_process";
// ── Constants ──────────────────────────────────────────────
const PROTOCOL_VERSION = "2024-11-05";
const DEFAULT_TIMEOUT = 30_000;
const CLIENT_NAME = "openclaw-neuralmemory";
const CLIENT_VERSION = "1.7.0";
const MAX_BUFFER_BYTES = 10 * 1024 * 1024; // 10 MB safety cap
const MAX_STDERR_LINES = 50;
/** Env vars forwarded to the MCP child process (least-privilege). */
export const ALLOWED_ENV_KEYS = new Set([
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
]);
// ── Client ─────────────────────────────────────────────────
export class NeuralMemoryMcpClient {
    proc = null;
    requestId = 0;
    pending = new Map();
    rawBuffer = Buffer.alloc(0);
    pythonPath;
    brain;
    logger;
    timeout;
    initTimeout;
    _connected = false;
    _connecting = null;
    constructor(options) {
        this.pythonPath = options.pythonPath;
        this.brain = options.brain;
        this.logger = options.logger;
        this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
        this.initTimeout = options.initTimeout ?? 90_000;
    }
    get connected() {
        return this._connected;
    }
    /**
     * Ensure the MCP process is connected. Safe to call concurrently —
     * concurrent callers share the same in-flight connection attempt.
     */
    async ensureConnected() {
        if (this._connected)
            return;
        if (this._connecting) {
            await this._connecting;
            return;
        }
        this._connecting = this.connect().finally(() => {
            this._connecting = null;
        });
        await this._connecting;
    }
    async connect() {
        const env = buildChildEnv(this.brain);
        this.proc = spawn(this.pythonPath, ["-m", "neural_memory.mcp"], {
            stdio: ["pipe", "pipe", "pipe"],
            env,
        });
        this.proc.stdout.on("data", (chunk) => {
            this.rawBuffer = Buffer.concat([this.rawBuffer, chunk]);
            if (this.rawBuffer.length > MAX_BUFFER_BYTES) {
                this.logger.error(`MCP buffer exceeded ${MAX_BUFFER_BYTES} bytes — killing process`);
                this.proc?.kill("SIGKILL");
                return;
            }
            this.drainBuffer();
        });
        const stderrChunks = [];
        this.proc.stderr.on("data", (chunk) => {
            const msg = chunk.toString("utf-8").trim();
            if (msg) {
                if (stderrChunks.length < MAX_STDERR_LINES) {
                    stderrChunks.push(msg);
                }
                this.logger.warn(`[mcp stderr] ${msg}`);
            }
        });
        this.proc.on("exit", (code) => {
            this._connected = false;
            const hint = code === 1
                ? " — check that neural-memory is installed: pip install neural-memory"
                : "";
            this.rejectAll(new Error(`MCP process exited with code ${code}${hint}`));
            this.logger.error(`MCP process exited (code: ${code})${hint}`);
        });
        this.proc.on("error", (err) => {
            this._connected = false;
            const hint = err.message.includes("ENOENT")
                ? ` — "${this.pythonPath}" not found. Check pythonPath in plugin config.`
                : "";
            this.rejectAll(new Error(`MCP process error: ${err.message}${hint}`));
            this.logger.error(`MCP process error: ${err.message}${hint}`);
        });
        // MCP initialize handshake (uses longer timeout for cold starts)
        try {
            await this.send("initialize", {
                protocolVersion: PROTOCOL_VERSION,
                capabilities: {},
                clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
            }, this.initTimeout);
        }
        catch (err) {
            const stderr = stderrChunks.join("\n");
            const detail = stderr
                ? `\nPython stderr:\n${stderr}`
                : "\nNo stderr output — the Python process may have hung.";
            throw new Error(`MCP initialize failed: ${err.message}${detail}\n` +
                `Verify: ${this.pythonPath} -m neural_memory.mcp`);
        }
        // Send initialized notification (no response expected)
        this.notify("notifications/initialized", {});
        this._connected = true;
        this.logger.info(`MCP connected (brain: ${this.brain}, protocol: ${PROTOCOL_VERSION})`);
    }
    /**
     * Fetch all available tools from the MCP server via `tools/list`.
     * Returns the raw MCP tool definitions (name, description, inputSchema).
     */
    async listTools() {
        const result = (await this.send("tools/list", {}));
        return result.tools ?? [];
    }
    async callTool(name, args = {}) {
        const result = (await this.send("tools/call", {
            name,
            arguments: args,
        }));
        if (result.isError) {
            const text = result.content?.[0]?.text ?? "Unknown MCP error";
            throw new Error(text);
        }
        return result.content?.[0]?.text ?? "";
    }
    async close() {
        this._connected = false;
        this.rejectAll(new Error("Client closing"));
        const proc = this.proc;
        this.proc = null;
        this.rawBuffer = Buffer.alloc(0);
        if (proc) {
            proc.removeAllListeners();
            proc.stdout?.removeAllListeners();
            proc.stderr?.removeAllListeners();
            const exited = new Promise((resolve) => {
                proc.once("exit", () => resolve());
                setTimeout(() => {
                    proc.kill("SIGKILL");
                    resolve();
                }, 3_000);
            });
            proc.kill("SIGTERM");
            await exited;
        }
        this.logger.info("MCP client closed");
    }
    // ── JSON-RPC protocol layer ──────────────────────────────
    send(method, params, timeoutOverride) {
        return new Promise((resolve, reject) => {
            if (!this.proc?.stdin?.writable) {
                reject(new Error("MCP process not available"));
                return;
            }
            const id = ++this.requestId;
            const ms = timeoutOverride ?? this.timeout;
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`MCP timeout: ${method} (${ms}ms)`));
            }, ms);
            this.pending.set(id, { resolve, reject, timer });
            this.writeMessage({ jsonrpc: "2.0", id, method, params });
        });
    }
    notify(method, params) {
        if (!this.proc?.stdin?.writable)
            return;
        this.writeMessage({ jsonrpc: "2.0", method, params });
    }
    writeMessage(message) {
        if (!this.proc?.stdin?.writable)
            return;
        const json = JSON.stringify(message);
        const frame = `${json}\n`;
        this.proc.stdin.write(frame);
    }
    // ── Response parsing (newline-delimited JSON Lines) ────────
    drainBuffer() {
        while (true) {
            const newlineIndex = this.rawBuffer.indexOf("\n");
            if (newlineIndex === -1)
                break;
            const line = this.rawBuffer.subarray(0, newlineIndex).toString("utf-8");
            this.rawBuffer = this.rawBuffer.subarray(newlineIndex + 1);
            if (!line.trim())
                continue;
            try {
                const message = JSON.parse(line);
                this.handleMessage(message);
            }
            catch (err) {
                this.logger.error(`Failed to parse MCP message: ${err.message}`);
            }
        }
    }
    handleMessage(message) {
        // Notifications (no id) — ignore silently
        if (message.id == null)
            return;
        const pending = this.pending.get(message.id);
        if (!pending)
            return;
        this.pending.delete(message.id);
        clearTimeout(pending.timer);
        if (message.error) {
            pending.reject(new Error(`MCP error ${message.error.code}: ${message.error.message}`));
        }
        else {
            pending.resolve(message.result);
        }
    }
    rejectAll(error) {
        for (const [, pending] of this.pending) {
            clearTimeout(pending.timer);
            pending.reject(error);
        }
        this.pending.clear();
    }
}
// ── Helpers ─────────────────────────────────────────────────
/** Build a minimal env for the child process (least-privilege). */
export function buildChildEnv(brain) {
    const env = {};
    for (const key of ALLOWED_ENV_KEYS) {
        const value = process.env[key];
        if (value !== undefined) {
            env[key] = value;
        }
    }
    if (brain !== "default") {
        env.NEURALMEMORY_BRAIN = brain;
    }
    return env;
}
//# sourceMappingURL=mcp-client.js.map