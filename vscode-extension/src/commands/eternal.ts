import * as vscode from "vscode";
import { NeuralMemoryClient } from "../server/client";
import { ServerLifecycle } from "../server/lifecycle";
import type { RecapResponse } from "../server/types";
import { readCurrentBrain } from "./brain";

const RECAP_LEVELS: readonly vscode.QuickPickItem[] = [
  {
    label: "$(zap) Quick (Level 1)",
    description: "Project + current task (~500 tokens)",
  },
  {
    label: "$(list-unordered) Detailed (Level 2)",
    description: "+ decisions, errors, progress (~1300 tokens)",
  },
  {
    label: "$(book) Full (Level 3)",
    description: "+ conversation history, files (~3300 tokens)",
  },
];

/**
 * Register eternal context and recap commands.
 */
export function registerEternalCommands(
  context: vscode.ExtensionContext,
  server: ServerLifecycle,
): void {
  // ── Recap command ──
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.recap", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const levelPick = await vscode.window.showQuickPick(RECAP_LEVELS, {
        placeHolder: "Select recap detail level",
        title: "Session Recap",
      });

      if (!levelPick) {
        return;
      }

      const level = RECAP_LEVELS.indexOf(levelPick) + 1;
      const brainId = readCurrentBrain();
      const client = new NeuralMemoryClient(server.baseUrl);

      try {
        const result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: "Loading context...",
            cancellable: false,
          },
          () => client.recap(brainId, { level }),
        );

        await showRecapResult(result);
      } catch (err) {
        vscode.window.showErrorMessage(
          `Recap failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );

  // ── Recap by topic command ──
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.recapTopic", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const topic = await vscode.window.showInputBox({
        prompt: "Search context for a topic",
        placeHolder: "e.g. auth, database, deploy",
        ignoreFocusOut: true,
      });

      if (!topic) {
        return;
      }

      const brainId = readCurrentBrain();
      const client = new NeuralMemoryClient(server.baseUrl);

      try {
        const result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Searching context for "${topic}"...`,
            cancellable: false,
          },
          () => client.recap(brainId, { topic }),
        );

        await showRecapResult(result);
      } catch (err) {
        vscode.window.showErrorMessage(
          `Recap failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );

  // ── Eternal save command ──
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.eternalSave", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const brainId = readCurrentBrain();
      const client = new NeuralMemoryClient(server.baseUrl);

      try {
        const result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: "Saving eternal context...",
            cancellable: false,
          },
          () => client.eternal(brainId, { action: "save" }),
        );

        vscode.window.showInformationMessage(
          result.message ?? "Eternal context saved.",
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Save failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );

  // ── Eternal status command ──
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.eternalStatus", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const brainId = readCurrentBrain();
      const client = new NeuralMemoryClient(server.baseUrl);

      try {
        const result = await client.eternal(brainId, { action: "status" });

        const lines = [
          "Eternal Context Status",
          "═".repeat(40),
          "",
          `Enabled: ${result.enabled ? "Yes" : "No"}`,
          `Loaded: ${result.loaded ? "Yes" : "No"}`,
          "",
          "── Tier 1 (Critical) ──",
          `Project: ${result.brain?.project_name || "(not set)"}`,
          `Tech stack: ${result.brain?.tech_stack?.join(", ") || "(not set)"}`,
          `Decisions: ${result.brain?.decisions_count ?? 0}`,
          `Instructions: ${result.brain?.instructions_count ?? 0}`,
          "",
          "── Tier 2 (Session) ──",
          `Feature: ${result.session?.feature || "(none)"}`,
          `Task: ${result.session?.task || "(none)"}`,
          `Progress: ${Math.round((result.session?.progress ?? 0) * 100)}%`,
          `Errors: ${result.session?.errors_count ?? 0}`,
          `Pending tasks: ${result.session?.pending_tasks_count ?? 0}`,
          `Branch: ${result.session?.branch || "(none)"}`,
          "",
          "── Tier 3 (Context) ──",
          `Messages: ${result.context?.message_count ?? 0}`,
          `Summaries: ${result.context?.summaries_count ?? 0}`,
          `Recent files: ${result.context?.recent_files_count ?? 0}`,
          `Token estimate: ${result.context?.token_estimate ?? 0}`,
          "",
          `Context usage: ${Math.round((result.context_usage ?? 0) * 100)}%`,
        ];

        const doc = await vscode.workspace.openTextDocument({
          content: lines.join("\n"),
          language: "markdown",
        });

        await vscode.window.showTextDocument(doc, {
          preview: true,
          viewColumn: vscode.ViewColumn.Beside,
        });
      } catch (err) {
        vscode.window.showErrorMessage(
          `Status failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );
}

async function showRecapResult(result: RecapResponse): Promise<void> {
  const lines = [
    `Session Recap (Level ${result.level ?? "?"})`,
    "═".repeat(40),
    "",
    result.context || "(no context available)",
    "",
    "─".repeat(40),
    result.message ?? "",
  ];

  if (result.tokens_used) {
    lines.push(`Tokens used: ~${result.tokens_used}`);
  }

  const doc = await vscode.workspace.openTextDocument({
    content: lines.join("\n"),
    language: "markdown",
  });

  await vscode.window.showTextDocument(doc, {
    preview: true,
    viewColumn: vscode.ViewColumn.Beside,
  });
}
