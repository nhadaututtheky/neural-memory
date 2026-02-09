/**
 * NeuralMemory MCP Client — JSON-RPC 2.0 over stdio.
 *
 * Spawns `python -m neural_memory.mcp` and communicates using the
 * MCP protocol (Content-Length framing, same as LSP).
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
const CLIENT_VERSION = "1.4.0";

// ── Client ─────────────────────────────────────────────────

export class NeuralMemoryMcpClient {
  private proc: ChildProcess | null = null;
  private requestId = 0;
  private readonly pending = new Map<number, PendingRequest>();
  private buffer = "";
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
    const env: Record<string, string> = {
      ...(process.env as Record<string, string>),
    };

    if (this.brain !== "default") {
      env.NEURALMEMORY_BRAIN = this.brain;
    }

    this.proc = spawn(this.pythonPath, ["-m", "neural_memory.mcp"], {
      stdio: ["pipe", "pipe", "pipe"],
      env,
    });

    this.proc.stdout!.on("data", (chunk: Buffer) => {
      this.buffer += chunk.toString("utf-8");
      this.drainBuffer();
    });

    this.proc.stderr!.on("data", (chunk: Buffer) => {
      const msg = chunk.toString("utf-8").trim();
      if (msg) this.logger.warn(`[mcp stderr] ${msg}`);
    });

    this.proc.on("exit", (code) => {
      this._connected = false;
      this.rejectAll(new Error(`MCP process exited with code ${code}`));
      this.logger.info(`MCP process exited (code: ${code})`);
    });

    this.proc.on("error", (err) => {
      this._connected = false;
      this.rejectAll(err);
      this.logger.error(`MCP process error: ${err.message}`);
    });

    // MCP initialize handshake
    await this.send("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
    });

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

    if (this.proc) {
      this.proc.kill("SIGTERM");
      this.proc = null;
    }

    this.buffer = "";
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
    this.writeMessage({ jsonrpc: "2.0", method, params });
  }

  private writeMessage(message: object): void {
    const json = JSON.stringify(message);
    const byteLength = Buffer.byteLength(json, "utf-8");
    const frame = `Content-Length: ${byteLength}\r\n\r\n${json}`;
    this.proc!.stdin!.write(frame);
  }

  // ── Response parsing ─────────────────────────────────────

  private drainBuffer(): void {
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) break;

      const header = this.buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length:\s*(\d+)/i);
      if (!match) {
        // Malformed header — skip past it
        this.buffer = this.buffer.slice(headerEnd + 4);
        continue;
      }

      const contentLength = parseInt(match[1], 10);
      const bodyStart = headerEnd + 4;

      if (this.buffer.length < bodyStart + contentLength) {
        break; // Incomplete body — wait for more data
      }

      const body = this.buffer.slice(bodyStart, bodyStart + contentLength);
      this.buffer = this.buffer.slice(bodyStart + contentLength);

      try {
        const message = JSON.parse(body) as JsonRpcMessage;
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
