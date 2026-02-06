import * as vscode from "vscode";
import { NeuralMemoryClient } from "../server/client";
import { ServerLifecycle } from "../server/lifecycle";
import { readCurrentBrain } from "./brain";

/**
 * Register codebase indexing commands.
 */
export function registerIndexCommands(
  context: vscode.ExtensionContext,
  server: ServerLifecycle,
): void {
  // Index codebase
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "neuralmemory.indexCodebase",
      async () => {
        if (!server.isRunning()) {
          vscode.window.showWarningMessage(
            "NeuralMemory server is not running.",
          );
          return;
        }

        // Determine workspace folder to index
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
          vscode.window.showWarningMessage(
            "No workspace folder open. Open a folder first.",
          );
          return;
        }

        let folderPath: string;
        if (folders.length === 1) {
          folderPath = folders[0].uri.fsPath;
        } else {
          const picked = await vscode.window.showWorkspaceFolderPick({
            placeHolder: "Select workspace folder to index",
          });
          if (!picked) {
            return;
          }
          folderPath = picked.uri.fsPath;
        }

        const brainId = readCurrentBrain();
        const client = new NeuralMemoryClient(server.baseUrl);

        try {
          const result = await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: "Indexing codebase...",
              cancellable: false,
            },
            () =>
              client.indexCodebase(brainId, {
                action: "scan",
                path: folderPath,
              }),
          );

          vscode.window.showInformationMessage(result.message);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Index failed: ${err instanceof Error ? err.message : err}`,
          );
        }
      },
    ),
  );
}
