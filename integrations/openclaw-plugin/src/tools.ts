/**
 * NeuralMemory tool definitions for OpenClaw.
 *
 * Each tool proxies to the MCP server via JSON-RPC.
 *
 * Uses raw JSON Schema for parameters instead of Zod to avoid
 * schema conversion issues (Zod→JSON Schema can produce $ref wrappers
 * or omit `properties` which Anthropic API rejects).
 *
 * Registers 6 core tools:
 *   nmem_remember  — Store a memory
 *   nmem_recall    — Query/search memories
 *   nmem_context   — Get recent context
 *   nmem_todo      — Quick TODO shortcut
 *   nmem_stats     — Brain statistics
 *   nmem_health    — Brain health diagnostics
 */

import type { NeuralMemoryMcpClient } from "./mcp-client.js";

// ── Types ──────────────────────────────────────────────────

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

// ── Tool factory ───────────────────────────────────────────

export function createTools(mcp: NeuralMemoryMcpClient): ToolDefinition[] {
  const call = async (
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<unknown> => {
    if (!mcp.connected) {
      return {
        error: true,
        message: "NeuralMemory service not running. Start the service first.",
      };
    }

    try {
      const raw = await mcp.callTool(toolName, args);
      try {
        return JSON.parse(raw);
      } catch {
        return { text: raw };
      }
    } catch (err) {
      return {
        error: true,
        message: `Tool ${toolName} failed: ${(err as Error).message}`,
      };
    }
  };

  return [
    {
      name: "nmem_remember",
      description:
        "Store a memory in NeuralMemory. Use this to remember facts, decisions, " +
        "insights, todos, errors, and other information that should persist across sessions.",
      parameters: {
        type: "object",
        properties: {
          content: {
            type: "string",
            maxLength: 100_000,
            description: "The content to remember",
          },
          type: {
            type: "string",
            enum: [
              "fact",
              "decision",
              "preference",
              "todo",
              "insight",
              "context",
              "instruction",
              "error",
              "workflow",
              "reference",
            ],
            description: "Memory type (auto-detected if not specified)",
          },
          priority: {
            type: "integer",
            minimum: 0,
            maximum: 10,
            description: "Priority 0-10 (5=normal, 10=critical)",
          },
          tags: {
            type: "array",
            items: { type: "string", maxLength: 100 },
            maxItems: 50,
            description: "Tags for categorization",
          },
          expires_days: {
            type: "integer",
            minimum: 1,
            maximum: 3650,
            description: "Days until memory expires",
          },
        },
        required: ["content"],
      },
      execute: (_id, args) => call("nmem_remember", args),
    },

    {
      name: "nmem_recall",
      description:
        "Query memories from NeuralMemory. Use this to recall past information, " +
        "decisions, patterns, or context relevant to the current task.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            maxLength: 10_000,
            description: "The query to search memories",
          },
          depth: {
            type: "integer",
            minimum: 0,
            maximum: 3,
            description:
              "Search depth: 0=instant, 1=context, 2=habit, 3=deep",
          },
          max_tokens: {
            type: "integer",
            minimum: 1,
            maximum: 10000,
            description: "Maximum tokens in response (default: 500)",
          },
          min_confidence: {
            type: "number",
            minimum: 0,
            maximum: 1,
            description: "Minimum confidence threshold",
          },
        },
        required: ["query"],
      },
      execute: (_id, args) => call("nmem_recall", args),
    },

    {
      name: "nmem_context",
      description:
        "Get recent context from NeuralMemory. Use this at the start of " +
        "tasks to inject relevant recent memories.",
      parameters: {
        type: "object",
        properties: {
          limit: {
            type: "integer",
            minimum: 1,
            maximum: 200,
            description: "Number of recent memories (default: 10)",
          },
          fresh_only: {
            type: "boolean",
            description: "Only include memories less than 30 days old",
          },
        },
      },
      execute: (_id, args) => call("nmem_context", args),
    },

    {
      name: "nmem_todo",
      description:
        "Quick shortcut to add a TODO memory with 30-day expiry.",
      parameters: {
        type: "object",
        properties: {
          task: {
            type: "string",
            maxLength: 10_000,
            description: "The task to remember",
          },
          priority: {
            type: "integer",
            minimum: 0,
            maximum: 10,
            description: "Priority 0-10 (default: 5)",
          },
        },
        required: ["task"],
      },
      execute: (_id, args) => call("nmem_todo", args),
    },

    {
      name: "nmem_stats",
      description:
        "Get brain statistics including memory counts and freshness.",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: (_id, args) => call("nmem_stats", args),
    },

    {
      name: "nmem_health",
      description:
        "Get brain health diagnostics including grade, purity score, " +
        "and recommendations.",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: (_id, args) => call("nmem_health", args),
    },
  ];
}
