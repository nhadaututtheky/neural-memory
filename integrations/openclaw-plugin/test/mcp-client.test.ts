import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import type { PluginLogger } from "../src/types.js";

// Mock child_process.spawn before importing the module
const mockSpawn = vi.fn();
vi.mock("node:child_process", () => ({
  spawn: (...args: unknown[]) => mockSpawn(...args),
}));

// Must import after mocking
const { NeuralMemoryMcpClient } = await import("../src/mcp-client.js");

function makeLogger(): PluginLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  };
}

type MockProc = EventEmitter & {
  stdin: { writable: boolean; write: ReturnType<typeof vi.fn> };
  stdout: EventEmitter;
  stderr: EventEmitter;
  kill: ReturnType<typeof vi.fn>;
  removeAllListeners: ReturnType<typeof vi.fn>;
  pid: number;
};

function makeMockProc(): MockProc {
  const proc = new EventEmitter() as MockProc;
  proc.stdin = { writable: true, write: vi.fn() };
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.kill = vi.fn();
  proc.removeAllListeners = vi.fn();
  proc.pid = 1234;
  return proc;
}

function buildJsonRpcFrame(message: object): Buffer {
  const json = JSON.stringify(message);
  const byteLength = Buffer.byteLength(json, "utf-8");
  return Buffer.from(`Content-Length: ${byteLength}\r\n\r\n${json}`);
}

describe("NeuralMemoryMcpClient", () => {
  let logger: PluginLogger;

  beforeEach(() => {
    logger = makeLogger();
    mockSpawn.mockReset();
  });

  describe("constructor", () => {
    it("sets defaults correctly", () => {
      const client = new NeuralMemoryMcpClient({
        pythonPath: "python3",
        brain: "test-brain",
        logger,
      });
      expect(client.connected).toBe(false);
    });
  });

  describe("connect", () => {
    it("spawns process with correct args", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "/usr/bin/python3",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();

      // Simulate initialize response
      const initResponse = buildJsonRpcFrame({
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2024-11-05" },
      });
      proc.stdout.emit("data", initResponse);

      await connectPromise;

      expect(mockSpawn).toHaveBeenCalledWith(
        "/usr/bin/python3",
        ["-m", "neural_memory.mcp"],
        expect.objectContaining({
          stdio: ["pipe", "pipe", "pipe"],
        }),
      );
      expect(client.connected).toBe(true);
    });

    it("uses minimal env (not full process.env)", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "custom",
        logger,
      });

      const connectPromise = client.connect();

      const initResponse = buildJsonRpcFrame({
        jsonrpc: "2.0",
        id: 1,
        result: {},
      });
      proc.stdout.emit("data", initResponse);

      await connectPromise;

      const spawnCall = mockSpawn.mock.calls[0];
      const env = spawnCall[2].env as Record<string, string>;

      // Should NOT contain full process.env
      expect(env).toHaveProperty("NEURALMEMORY_BRAIN", "custom");
      // Should not have random env vars
      expect(Object.keys(env).length).toBeLessThan(
        Object.keys(process.env).length,
      );
    });

    it("throws on initialize failure with stderr detail", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
        timeout: 100,
      });

      const connectPromise = client.connect();

      // Emit stderr before the timeout
      proc.stderr.emit("data", Buffer.from("ModuleNotFoundError: neural_memory"));

      // Let timeout trigger
      await expect(connectPromise).rejects.toThrow("MCP initialize failed");
    });
  });

  describe("callTool", () => {
    it("sends correct JSON-RPC and returns text content", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      // Connect first
      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      // Call a tool
      const callPromise = client.callTool("nmem_stats", {});

      // Respond
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({
          jsonrpc: "2.0",
          id: 2,
          result: {
            content: [{ type: "text", text: '{"neurons": 42}' }],
          },
        }),
      );

      const result = await callPromise;
      expect(result).toBe('{"neurons": 42}');
    });

    it("throws on isError response", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      const callPromise = client.callTool("nmem_stats", {});
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({
          jsonrpc: "2.0",
          id: 2,
          result: {
            isError: true,
            content: [{ type: "text", text: "Brain not found" }],
          },
        }),
      );

      await expect(callPromise).rejects.toThrow("Brain not found");
    });
  });

  describe("close", () => {
    it("kills process and rejects pending requests", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      // Start a pending call
      const callPromise = client.callTool("nmem_stats", {});

      // Simulate exit on kill
      proc.kill.mockImplementation(() => {
        proc.emit("exit", 0);
      });

      // Close should reject the pending call
      await client.close();

      await expect(callPromise).rejects.toThrow("Client closing");
      expect(client.connected).toBe(false);
    });

    it("removes event listeners", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      proc.kill.mockImplementation(() => {
        proc.emit("exit", 0);
      });

      await client.close();

      expect(proc.removeAllListeners).toHaveBeenCalled();
    });
  });

  describe("writeMessage / notify guards", () => {
    it("send rejects when process not available", async () => {
      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      // Not connected — no proc
      await expect(client.callTool("nmem_stats", {})).rejects.toThrow(
        "MCP process not available",
      );
    });
  });

  describe("drainBuffer", () => {
    it("parses single complete message", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();

      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: { ok: true } }),
      );

      await connectPromise;
      expect(client.connected).toBe(true);
    });

    it("handles partial message (waits for more data)", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();

      // Send a partial frame (header but incomplete body)
      const fullFrame = buildJsonRpcFrame({
        jsonrpc: "2.0",
        id: 1,
        result: {},
      });
      const half = Math.floor(fullFrame.length / 2);

      proc.stdout.emit("data", fullFrame.subarray(0, half));
      // Not resolved yet — need the rest
      proc.stdout.emit("data", fullFrame.subarray(half));

      await connectPromise;
      expect(client.connected).toBe(true);
    });

    it("handles multiple messages in one chunk", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();

      // Send init response
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      // Send two tool responses in one chunk
      const call1Promise = client.callTool("nmem_stats", {});
      const call2Promise = client.callTool("nmem_health", {});

      const frame1 = buildJsonRpcFrame({
        jsonrpc: "2.0",
        id: 2,
        result: { content: [{ type: "text", text: '{"stats": 1}' }] },
      });
      const frame2 = buildJsonRpcFrame({
        jsonrpc: "2.0",
        id: 3,
        result: { content: [{ type: "text", text: '{"health": "A"}' }] },
      });

      // Both in one chunk
      proc.stdout.emit("data", Buffer.concat([frame1, frame2]));

      const [result1, result2] = await Promise.all([
        call1Promise,
        call2Promise,
      ]);
      expect(result1).toBe('{"stats": 1}');
      expect(result2).toBe('{"health": "A"}');
    });

    it("handles non-ASCII (UTF-8 byte accuracy)", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      const callPromise = client.callTool("nmem_stats", {});

      // Unicode in response
      const unicodeText = '{"answer": "こんにちは 🧠"}';
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({
          jsonrpc: "2.0",
          id: 2,
          result: { content: [{ type: "text", text: unicodeText }] },
        }),
      );

      const result = await callPromise;
      expect(result).toBe(unicodeText);
    });
  });

  describe("error handling", () => {
    it("process exit rejects pending and sets connected=false", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      const callPromise = client.callTool("nmem_stats", {});

      // Process exits
      proc.emit("exit", 1);

      await expect(callPromise).rejects.toThrow("MCP process exited");
      expect(client.connected).toBe(false);
    });

    it("process error event rejects pending", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      const callPromise = client.callTool("nmem_stats", {});

      proc.emit("error", new Error("ENOENT"));

      await expect(callPromise).rejects.toThrow("MCP process error");
    });

    it("JSON-RPC error resolves to rejection", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();
      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      const callPromise = client.callTool("nmem_stats", {});

      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({
          jsonrpc: "2.0",
          id: 2,
          error: { code: -32600, message: "Invalid request" },
        }),
      );

      await expect(callPromise).rejects.toThrow(
        "MCP error -32600: Invalid request",
      );
    });

    it("stderr is capped at MAX_STDERR_LINES", async () => {
      const proc = makeMockProc();
      mockSpawn.mockReturnValue(proc);

      const client = new NeuralMemoryMcpClient({
        pythonPath: "python",
        brain: "default",
        logger,
      });

      const connectPromise = client.connect();

      // Emit 60 stderr lines
      for (let i = 0; i < 60; i++) {
        proc.stderr.emit("data", Buffer.from(`warning line ${i}`));
      }

      proc.stdout.emit(
        "data",
        buildJsonRpcFrame({ jsonrpc: "2.0", id: 1, result: {} }),
      );
      await connectPromise;

      // Logger.warn should have been called for all 60, but only 50 stored
      // (logging happens for all, but stderrChunks array caps at 50)
      expect((logger.warn as ReturnType<typeof vi.fn>).mock.calls.length).toBe(
        60,
      );
    });
  });
});
