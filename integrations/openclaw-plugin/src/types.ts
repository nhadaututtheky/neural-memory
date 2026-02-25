/**
 * Minimal OpenClaw plugin types for standalone compilation.
 * At runtime, jiti resolves the full types from OpenClaw's codebase.
 */

export type PluginLogger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

export type PluginKind = "memory";

export type OpenClawPluginServiceContext = {
  config: unknown;
  workspaceDir?: string;
  stateDir: string;
  logger: PluginLogger;
};

export type OpenClawPluginService = {
  id: string;
  start: (ctx: OpenClawPluginServiceContext) => void | Promise<void>;
  stop?: (ctx: OpenClawPluginServiceContext) => void | Promise<void>;
};

export type BeforeAgentStartEvent = {
  prompt: string;
  messages?: unknown[];
};

export type BeforeAgentStartResult = {
  systemPrompt?: string;     // Appended to system prompt — last handler wins
  prependContext?: string;   // Prepended to conversation context — all handlers concatenated
  modelOverride?: string;    // Override model for this run — first defined wins
  providerOverride?: string; // Override provider for this run — first defined wins
};

export type AgentEndEvent = {
  messages: unknown[];
  success: boolean;
  error?: string;
  durationMs?: number;
};

export type AgentContext = {
  agentId?: string;
  sessionKey?: string;
  workspaceDir?: string;
};

export type OpenClawPluginApi = {
  id: string;
  name: string;
  config: unknown;
  pluginConfig?: Record<string, unknown>;
  runtime: unknown;
  logger: PluginLogger;
  registerTool: (
    tool: unknown,
    opts?: { name?: string; names?: string[] },
  ) => void;
  registerService: (service: OpenClawPluginService) => void;
  on: (
    hookName: string,
    handler: (...args: unknown[]) => unknown,
    opts?: { priority?: number },
  ) => void;
};

export type OpenClawPluginDefinition = {
  id?: string;
  name?: string;
  description?: string;
  version?: string;
  kind?: PluginKind;
  register?: (api: OpenClawPluginApi) => void | Promise<void>;
  activate?: (api: OpenClawPluginApi) => void | Promise<void>;
};
