import * as vscode from "vscode";

export interface NeuralMemoryConfig {
  readonly pythonPath: string;
  readonly autoStart: boolean;
  readonly serverUrl: string;
  readonly graphNodeLimit: number;
  readonly codeLensEnabled: boolean;
  readonly commentTriggers: readonly string[];
}

export function getConfig(): NeuralMemoryConfig {
  const cfg = vscode.workspace.getConfiguration("neuralmemory");

  return {
    pythonPath: cfg.get<string>("pythonPath", "python"),
    autoStart: cfg.get<boolean>("autoStart", true),
    serverUrl: cfg.get<string>("serverUrl", "http://127.0.0.1:8000"),
    graphNodeLimit: cfg.get<number>("graphNodeLimit", 1000),
    codeLensEnabled: cfg.get<boolean>("codeLensEnabled", true),
    commentTriggers: cfg.get<string[]>("commentTriggers", [
      "remember:",
      "note:",
      "decision:",
      "todo:",
    ]),
  };
}

/**
 * Extract host and port from a server URL string.
 */
export function parseServerUrl(url: string): { host: string; port: number } {
  const parsed = new URL(url);
  return {
    host: parsed.hostname,
    port: parsed.port ? parseInt(parsed.port, 10) : 8000,
  };
}
