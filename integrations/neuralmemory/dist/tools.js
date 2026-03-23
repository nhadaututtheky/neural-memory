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
// ── Schema normalization ───────────────────────────────────
/** Keywords that some LLM providers reject in function schemas. */
const STRIP_KEYS = new Set([
    "minimum",
    "maximum",
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
    "exclusiveMinimum",
    "exclusiveMaximum",
]);
/**
 * Recursively normalize a JSON Schema node for provider compatibility:
 * - Strip constraint keywords
 * - Replace `integer` with `number` (Gemini compat)
 * - Add `additionalProperties: false` to objects (OpenAI strict mode)
 * - Ensure every object has `properties` (Anthropic SDK)
 */
function normalizeSchema(node) {
    if (node === null || node === undefined || typeof node !== "object") {
        return node;
    }
    if (Array.isArray(node)) {
        return node.map(normalizeSchema);
    }
    const obj = node;
    const result = {};
    for (const [key, value] of Object.entries(obj)) {
        if (STRIP_KEYS.has(key))
            continue;
        if (key === "type" && value === "integer") {
            result[key] = "number";
        }
        else if (key === "properties" && typeof value === "object" && value !== null) {
            // Recurse into each property definition
            const props = {};
            for (const [propName, propSchema] of Object.entries(value)) {
                props[propName] = normalizeSchema(propSchema);
            }
            result[key] = props;
        }
        else if (key === "items" && typeof value === "object" && value !== null) {
            result[key] = normalizeSchema(value);
        }
        else if ((key === "anyOf" || key === "oneOf" || key === "allOf") &&
            Array.isArray(value)) {
            result[key] = value.map(normalizeSchema);
        }
        else {
            result[key] = value;
        }
    }
    // Ensure objects have `properties` and `additionalProperties`
    if (result["type"] === "object") {
        if (!("properties" in result) || result["properties"] === undefined) {
            result["properties"] = {};
        }
        if (!("additionalProperties" in result)) {
            result["additionalProperties"] = false;
        }
    }
    return result;
}
/**
 * Convert an MCP inputSchema into a provider-safe OpenClaw JsonSchema.
 * Falls back to an empty-properties object if the schema is missing/invalid.
 */
function toSafeSchema(inputSchema) {
    if (!inputSchema || typeof inputSchema !== "object") {
        return { type: "object", properties: {}, additionalProperties: false };
    }
    const normalized = normalizeSchema(inputSchema);
    return {
        type: "object",
        properties: (normalized["properties"] ?? {}),
        ...(Array.isArray(normalized["required"]) && normalized["required"].length > 0
            ? { required: normalized["required"] }
            : {}),
        additionalProperties: false,
    };
}
// ── Tool factory ───────────────────────────────────────────
/**
 * Create a tool call helper that auto-reconnects to MCP.
 */
function makeCallFn(mcp) {
    return async (toolName, args) => {
        if (!mcp.connected) {
            try {
                await mcp.ensureConnected();
            }
            catch (err) {
                return {
                    error: true,
                    message: `NeuralMemory auto-connect failed: ${err.message}`,
                };
            }
        }
        try {
            const raw = await mcp.callTool(toolName, args);
            try {
                return JSON.parse(raw);
            }
            catch {
                return { text: raw };
            }
        }
        catch (err) {
            return {
                error: true,
                message: `Tool ${toolName} failed: ${err.message}`,
            };
        }
    };
}
/**
 * Convert a single MCP tool definition into an OpenClaw ToolDefinition.
 */
function mcpToolToOpenClaw(mcpTool, call) {
    return {
        name: mcpTool.name,
        description: mcpTool.description ?? `NeuralMemory tool: ${mcpTool.name}`,
        parameters: toSafeSchema(mcpTool.inputSchema),
        execute: (_id, args) => call(mcpTool.name, args),
    };
}
/**
 * Fetch all tools from the MCP server and convert them to OpenClaw format.
 * Must be called after MCP connection is established.
 */
export async function createToolsFromMcp(mcp) {
    const mcpTools = await mcp.listTools();
    const call = makeCallFn(mcp);
    return mcpTools.map((t) => mcpToolToOpenClaw(t, call));
}
/**
 * Fallback: create minimal hardcoded tools if MCP tools/list fails.
 * Ensures the plugin still works even if the MCP server is an older version.
 */
export function createFallbackTools(mcp) {
    const call = makeCallFn(mcp);
    return [
        {
            name: "nmem_remember",
            description: "Store a memory in NeuralMemory. Use this to remember facts, decisions, " +
                "insights, todos, errors, and other information that should persist across sessions.",
            parameters: {
                type: "object",
                properties: {
                    content: { type: "string", description: "The content to remember" },
                    type: {
                        type: "string",
                        enum: [
                            "fact", "decision", "preference", "todo", "insight",
                            "context", "instruction", "error", "workflow", "reference",
                        ],
                        description: "Memory type (auto-detected if not specified)",
                    },
                    priority: { type: "number", description: "Priority 0-10 (5=normal, 10=critical)" },
                    tags: {
                        type: "array",
                        items: { type: "string" },
                        description: "Tags for categorization",
                    },
                },
                required: ["content"],
                additionalProperties: false,
            },
            execute: (_id, args) => call("nmem_remember", args),
        },
        {
            name: "nmem_recall",
            description: "Query memories from NeuralMemory. Use this to recall past information, " +
                "decisions, patterns, or context relevant to the current task.",
            parameters: {
                type: "object",
                properties: {
                    query: { type: "string", description: "The query to search memories" },
                    depth: { type: "number", description: "Search depth: 0=instant, 1=context, 2=habit, 3=deep" },
                    max_tokens: { type: "number", description: "Maximum tokens in response (default: 500)" },
                },
                required: ["query"],
                additionalProperties: false,
            },
            execute: (_id, args) => call("nmem_recall", args),
        },
        {
            name: "nmem_context",
            description: "Get recent context from NeuralMemory.",
            parameters: {
                type: "object",
                properties: {
                    limit: { type: "number", description: "Number of recent memories (default: 10)" },
                },
                additionalProperties: false,
            },
            execute: (_id, args) => call("nmem_context", args),
        },
        {
            name: "nmem_stats",
            description: "Get brain statistics including memory counts and freshness.",
            parameters: { type: "object", properties: {}, additionalProperties: false },
            execute: (_id, args) => call("nmem_stats", args),
        },
        {
            name: "nmem_health",
            description: "Get brain health diagnostics including grade and recommendations.",
            parameters: { type: "object", properties: {}, additionalProperties: false },
            execute: (_id, args) => call("nmem_health", args),
        },
    ];
}
/**
 * Create backward-compatible shim tools that map legacy OpenClaw memory-core
 * tool names to NeuralMemory equivalents.
 *
 * This prevents "allowList contains unknown entries (memory_search, memory_get)"
 * warnings when NM occupies the `memory` plugin slot, which removes the built-in
 * memory-core tools but leaves the tools.profile allowList referencing them.
 */
export function createCompatibilityTools(mcp) {
    const call = makeCallFn(mcp);
    return [
        {
            name: "memory_search",
            description: "Search memories (legacy alias for nmem_recall). " +
                "Prefer nmem_recall for full NeuralMemory features.",
            parameters: {
                type: "object",
                properties: {
                    query: { type: "string", description: "The search query" },
                },
                required: ["query"],
                additionalProperties: false,
            },
            execute: (_id, args) => call("nmem_recall", { query: args.query, depth: 1 }),
        },
        {
            name: "memory_get",
            description: "Get a memory by ID (legacy alias for nmem_recall). " +
                "Prefer nmem_recall for full NeuralMemory features.",
            parameters: {
                type: "object",
                properties: {
                    id: { type: "string", description: "Memory identifier or query" },
                },
                required: ["id"],
                additionalProperties: false,
            },
            execute: (_id, args) => call("nmem_recall", { query: String(args.id), depth: 0 }),
        },
    ];
}
//# sourceMappingURL=tools.js.map