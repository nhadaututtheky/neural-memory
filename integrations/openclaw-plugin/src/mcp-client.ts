/**
 * NeuralMemory MCP Client — JSON-RPC 2.0 over stdio.
 *
 * Spawns `python -m neural_memory.mcp` and communicates using the
 * MCP protocol (newline-delimited JSON Lines).
 *
 * Zero external dependencies — implements the protocol directly.
 */

import { spawn, type ChildProcess } from "node:child_process";
import type { PluginLogger } from "./types.js";

// ── Types ──────────────────────────────────────────────────

type PendingRequest = {
  readonly resolve: (value: unknown) => void;
  readonly reject: (error: Error) => void;
  readonly timer: ReturnType<typeof setTimeout>;
};

type JsonRpcMessage = {
  jsonrpc: "2.0";
  id?: number;
  method?: string;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
};

export type McpClientOptions = {
  readonly pythonPath: string;
  readonly brain: string;
  readonly logger: PluginLogger;
  readonly timeout?: number;
};

// ── Constants ──────────────────────────────────────────────

const PROTOCOL_VERSION = "2024-11-05";
const DEFAULT_TIMEOUT = 30_000;
const CLIENT_NAME = "openclaw-neuralmemory";
const CLIENT_VERSION = "1.4.1";
const MAX_BUFFER_BYTES = 10 * 1024 * 1024; // 10 MB safety cap
const MAX_STDERR_LINES = 50;

/** Env vars forwarded to the MCP child process (least-privilege). */
export const ALLOWED_ENV_KEYS: ReadonlySet<string> = new Set([
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
  private proc: ChildProcess | null = null;
  private requestId = 0;
  private readonly pending = new Map<number, PendingRequest>();
  private rawBuffer: Buffer = Buffer.alloc(0);
  private readonly pythonPath: string;
  private readonly brain: string;
  private readonly logger: PluginLogger;
  private readonly timeout: number;
  private _connected = false;

  constructor(options: McpClientOptions) {
    this.pythonPath = options.pythonPath;
    this.brain = options.brain;
    this.logger = options.logger;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
  }

  get connected(): boolean {
    return this._connected;
  }

  async connect(): Promise<void> {
    const env = buildChildEnv(this.brain);

    this.proc = spawn(this.pythonPath, ["-m", "neural_memory.mcp"], {
      stdio: ["pipe", "pipe", "pipe"],
      env,
    });

    this.proc.stdout!.on("data", (chunk: Buffer) => {
      this.rawBuffer = Buffer.concat([this.rawBuffer, chunk]);
      if (this.rawBuffer.length > MAX_BUFFER_BYTES) {
        this.logger.error(
          `MCP buffer exceeded ${MAX_BUFFER_BYTES} bytes — killing process`,
        );
        this.proc?.kill("SIGKILL");
        return;
      }
      this.drainBuffer();
    });

    const stderrChunks: string[] = [];

    this.proc.stderr!.on("data", (chunk: Buffer) => {
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
      const hint =
        code === 1
          ? " — check that neural-memory is installed: pip install neural-memory"
          : "";
      this.rejectAll(new Error(`MCP process exited with code ${code}${hint}`));
      this.logger.error(`MCP process exited (code: ${code})${hint}`);
    });

    this.proc.on("error", (err) => {
      this._connected = false;
      const hint =
        err.message.includes("ENOENT")
          ? ` — "${this.pythonPath}" not found. Check pythonPath in plugin config.`
          : "";
      this.rejectAll(new Error(`MCP process error: ${err.message}${hint}`));
      this.logger.error(`MCP process error: ${err.message}${hint}`);
    });

    // MCP initialize handshake
    try {
      await this.send("initialize", {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
      });
    } catch (err) {
      const stderr = stderrChunks.join("\n");
      const detail = stderr
        ? `\nPython stderr:\n${stderr}`
        : "\nNo stderr output — the Python process may have hung.";
      throw new Error(
        `MCP initialize failed: ${(err as Error).message}${detail}\n` +
          `Verify: ${this.pythonPath} -m neural_memory.mcp`,
      );
    }

    // Send initialized notification (no response expected)
    this.notify("notifications/initialized", {});

    this._connected = true;
    this.logger.info(
      `MCP connected (brain: ${this.brain}, protocol: ${PROTOCOL_VERSION})`,
    );
  }

  async callTool(
    name: string,
    args: Record<string, unknown> = {},
  ): Promise<string> {
    const result = (await this.send("tools/call", {
      name,
      arguments: args,
    })) as { content?: Array<{ type: string; text: string }>; isError?: boolean };

    if (result.isError) {
      const text = result.content?.[0]?.text ?? "Unknown MCP error";
      throw new Error(text);
    }

    return result.content?.[0]?.text ?? "";
  }

  async close(): Promise<void> {
    this._connected = false;
    this.rejectAll(new Error("Client closing"));

    const proc = this.proc;
    this.proc = null;
    this.rawBuffer = Buffer.alloc(0);

    if (proc) {
      proc.removeAllListeners();
      proc.stdout?.removeAllListeners();
      proc.stderr?.removeAllListeners();

      const exited = new Promise<void>((resolve) => {
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

  private send(method: string, params: unknown): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.proc?.stdin?.writable) {
        reject(new Error("MCP process not available"));
        return;
      }

      const id = ++this.requestId;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP timeout: ${method} (${this.timeout}ms)`));
      }, this.timeout);

      this.pending.set(id, { resolve, reject, timer });
      this.writeMessage({ jsonrpc: "2.0", id, method, params });
    });
  }

  private notify(method: string, params: unknown): void {
    if (!this.proc?.stdin?.writable) return;
    this.writeMessage({ jsonrpc: "2.0", method, params });
  }

  private writeMessage(message: object): void {
    if (!this.proc?.stdin?.writable) return;
    const json = JSON.stringify(message);
    const frame = `${json}\n`;
    this.proc.stdin.write(frame);
  }

  // ── Response parsing (newline-delimited JSON Lines) ────────

  private drainBuffer(): void {
    while (true) {
      const newlineIndex = this.rawBuffer.indexOf("\n");
      if (newlineIndex === -1) break;

      const line = this.rawBuffer.subarray(0, newlineIndex).toString("utf-8");
      this.rawBuffer = this.rawBuffer.subarray(newlineIndex + 1);

      if (!line.trim()) continue;

      try {
        const message = JSON.parse(line) as JsonRpcMessage;
        this.handleMessage(message);
      } catch (err) {
        this.logger.error(
          `Failed to parse MCP message: ${(err as Error).message}`,
        );
      }
    }
  }

  private handleMessage(message: JsonRpcMessage): void {
    // Notifications (no id) — ignore silently
    if (message.id == null) return;

    const pending = this.pending.get(message.id);
    if (!pending) return;

    this.pending.delete(message.id);
    clearTimeout(pending.timer);

    if (message.error) {
      pending.reject(
        new Error(
          `MCP error ${message.error.code}: ${message.error.message}`,
        ),
      );
    } else {
      pending.resolve(message.result);
    }
  }

  private rejectAll(error: Error): void {
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }
}

// ── Helpers ─────────────────────────────────────────────────

/** Build a minimal env for the child process (least-privilege). */
export function buildChildEnv(brain: string): Record<string, string> {
  const env: Record<string, string> = {};

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
