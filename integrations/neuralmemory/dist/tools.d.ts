/**
 * NeuralMemory dynamic tool proxy for OpenClaw.
 *
 * Fetches all available tools from the MCP server via `tools/list` and
 * converts them into OpenClaw tool definitions. This means the plugin
 * automatically exposes every tool the MCP server provides — no hardcoded
 * schemas to maintain.
 *
 * Provider compatibility:
 *   - Strips constraint keywords (`minimum`, `maximum`, `maxLength`,
 *     `maxItems`, `minLength`) that some providers reject
 *   - Adds `additionalProperties: false` on all object schemas for
 *     OpenAI strict mode
 *   - Ensures every object type has a `properties` field (required by
 *     Anthropic SDK validation)
 *   - Uses `number` instead of `integer` for Gemini compatibility
 */
import type { NeuralMemoryMcpClient } from "./mcp-client.js";
type JsonSchema = {
    readonly type: "object";
    readonly properties: Record<string, unknown>;
    readonly required?: readonly string[];
    readonly additionalProperties?: boolean;
};
export type ToolDefinition = {
    readonly name: string;
    readonly description: string;
    readonly parameters: JsonSchema;
    readonly execute: (id: string, args: Record<string, unknown>) => Promise<unknown>;
};
/**
 * Fetch all tools from the MCP server and convert them to OpenClaw format.
 * Must be called after MCP connection is established.
 */
export declare function createToolsFromMcp(mcp: NeuralMemoryMcpClient): Promise<ToolDefinition[]>;
/**
 * Fallback: create minimal hardcoded tools if MCP tools/list fails.
 * Ensures the plugin still works even if the MCP server is an older version.
 */
export declare function createFallbackTools(mcp: NeuralMemoryMcpClient): ToolDefinition[];
/**
 * Create backward-compatible shim tools that map legacy OpenClaw memory-core
 * tool names to NeuralMemory equivalents.
 *
 * This prevents "allowList contains unknown entries (memory_search, memory_get)"
 * warnings when NM occupies the `memory` plugin slot, which removes the built-in
 * memory-core tools but leaves the tools.profile allowList referencing them.
 */
export declare function createCompatibilityTools(mcp: NeuralMemoryMcpClient): ToolDefinition[];
export {};
