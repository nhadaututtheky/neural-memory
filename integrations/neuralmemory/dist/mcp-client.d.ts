/**
 * NeuralMemory MCP Client — JSON-RPC 2.0 over stdio.
 *
 * Spawns `python -m neural_memory.mcp` and communicates using the
 * MCP protocol (newline-delimited JSON Lines).
 *
 * Zero external dependencies — implements the protocol directly.
 */
import type { PluginLogger } from "./types.js";
/** Raw tool definition from MCP `tools/list` response. */
export type McpToolDefinition = {
    name: string;
    description?: string;
    inputSchema?: Record<string, unknown>;
};
export type McpClientOptions = {
    readonly pythonPath: string;
    readonly brain: string;
    readonly logger: PluginLogger;
    readonly timeout?: number;
    readonly initTimeout?: number;
};
/** Env vars forwarded to the MCP child process (least-privilege). */
export declare const ALLOWED_ENV_KEYS: ReadonlySet<string>;
export declare class NeuralMemoryMcpClient {
    private proc;
    private requestId;
    private readonly pending;
    private rawBuffer;
    private readonly pythonPath;
    private readonly brain;
    private readonly logger;
    private readonly timeout;
    private readonly initTimeout;
    private _connected;
    private _connecting;
    constructor(options: McpClientOptions);
    get connected(): boolean;
    /**
     * Ensure the MCP process is connected. Safe to call concurrently —
     * concurrent callers share the same in-flight connection attempt.
     */
    ensureConnected(): Promise<void>;
    connect(): Promise<void>;
    /**
     * Fetch all available tools from the MCP server via `tools/list`.
     * Returns the raw MCP tool definitions (name, description, inputSchema).
     */
    listTools(): Promise<McpToolDefinition[]>;
    callTool(name: string, args?: Record<string, unknown>): Promise<string>;
    close(): Promise<void>;
    private send;
    private notify;
    private writeMessage;
    private drainBuffer;
    private handleMessage;
    private rejectAll;
}
/** Build a minimal env for the child process (least-privilege). */
export declare function buildChildEnv(brain: string): Record<string, string>;
