import * as vscode from "vscode";
import { registerBrainCommands, readCurrentBrain } from "./commands/brain";
import { registerEncodeCommands } from "./commands/encode";
import { registerRecallCommands } from "./commands/recall";
import { MemoryCodeLensProvider } from "./editors/MemoryCodeLensProvider";
import { CommentTriggerWatcher } from "./editors/CommentTriggerWatcher";
import { ServerLifecycle } from "./server/lifecycle";
import { SyncClient } from "./server/websocket";
import { getConfig } from "./utils/config";
import { checkForUpdates } from "./utils/updateChecker";
import { GraphPanel } from "./views/graph/GraphPanel";
import { MemoryTreeProvider } from "./views/MemoryTreeProvider";
import { StatusBarManager } from "./views/StatusBarManager";

let server: ServerLifecycle | undefined;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  const outputChannel =
    vscode.window.createOutputChannel("NeuralMemory");
  context.subscriptions.push(outputChannel);
  outputChannel.appendLine("NeuralMemory extension activating...");

  // 1. Server lifecycle manager
  server = new ServerLifecycle(context);
  context.subscriptions.push(server);

  // 2. Status bar (show immediately, updates when server connects)
  const statusBar = new StatusBarManager(server);
  context.subscriptions.push(statusBar);
  statusBar.setStatus("starting");
  statusBar.show();

  // 3. Register CodeLens providers
  const codeLens = new MemoryCodeLensProvider(server);
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider({ scheme: "file" }, codeLens),
  );

  const triggerWatcher = new CommentTriggerWatcher(server);
  triggerWatcher.activate(context);

  // 4. Memory tree view
  const memoryTree = new MemoryTreeProvider(server);
  vscode.window.registerTreeDataProvider("neuralmemory.memories", memoryTree);
  memoryTree.registerCommands(context);

  // 5. Register commands
  registerLifecycleCommands(context, server, statusBar);
  registerBrainCommands(context, server, statusBar);
  registerEncodeCommands(context, server);
  registerRecallCommands(context, server);

  // 6. Connect or start server
  let syncClient: SyncClient | undefined;

  try {
    await server.connectOrStart();
    outputChannel.appendLine(
      `Connected to server at ${server.baseUrl}`,
    );
    statusBar.setStatus("connected");

    // Set initial brain from config
    const currentBrain = readCurrentBrain();
    await statusBar.setBrain(currentBrain);

    // 7. WebSocket for live updates
    syncClient = await connectWebSocket(
      server.baseUrl,
      currentBrain,
      outputChannel,
      { codeLens, triggerWatcher, memoryTree, statusBar },
    );
    if (syncClient) {
      context.subscriptions.push(syncClient);
    }
  } catch (err) {
    outputChannel.appendLine(
      `Server connection failed: ${err}. Extension features will be limited.`,
    );
    statusBar.setStatus("disconnected");
  }

  // 8. Register graph explorer command
  registerGraphCommands(context, server);

  // 9. Refresh everything when server reconnects
  const serverRef = server;
  serverRef.onReady(async () => {
    codeLens.refresh();
    triggerWatcher.refresh();
    memoryTree.refresh();

    // Reconnect WebSocket if server restarts
    if (syncClient) {
      syncClient.disconnect();
    }
    const currentBrain = readCurrentBrain();
    syncClient = await connectWebSocket(
      serverRef.baseUrl,
      currentBrain,
      outputChannel,
      { codeLens, triggerWatcher, memoryTree, statusBar },
    );
    if (syncClient) {
      context.subscriptions.push(syncClient);
    }
  });

  outputChannel.appendLine("NeuralMemory extension activated.");

  // 10. Non-blocking update check (fire and forget)
  checkForUpdates(context);
}

export function deactivate(): void {
  // ServerLifecycle.dispose() handles process cleanup via disposables
  server = undefined;
}

function registerLifecycleCommands(
  context: vscode.ExtensionContext,
  server: ServerLifecycle,
  statusBar: StatusBarManager,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.startServer", async () => {
      if (server.isRunning()) {
        vscode.window.showInformationMessage(
          `NeuralMemory server already running at ${server.baseUrl}`,
        );
        return;
      }
      try {
        statusBar.setStatus("starting");
        await server.start();
        statusBar.setStatus("connected");

        const currentBrain = readCurrentBrain();
        await statusBar.setBrain(currentBrain);

        vscode.window.showInformationMessage(
          `NeuralMemory server started at ${server.baseUrl}`,
        );
      } catch (err) {
        statusBar.setStatus("disconnected");
        vscode.window.showErrorMessage(
          `Failed to start server: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "neuralmemory.connectServer",
      async () => {
        const config = getConfig();
        const url = await vscode.window.showInputBox({
          prompt: "Enter NeuralMemory server URL",
          value: config.serverUrl,
          placeHolder: "http://127.0.0.1:8000",
          validateInput: (value) => {
            try {
              new URL(value);
              return null;
            } catch {
              return "Enter a valid URL (e.g., http://127.0.0.1:8000)";
            }
          },
        });

        if (!url) {
          return;
        }

        await vscode.workspace
          .getConfiguration("neuralmemory")
          .update("serverUrl", url, vscode.ConfigurationTarget.Global);

        try {
          statusBar.setStatus("starting");
          await server.connectOrStart();
          statusBar.setStatus("connected");

          const currentBrain = readCurrentBrain();
          await statusBar.setBrain(currentBrain);

          vscode.window.showInformationMessage(
            `Connected to NeuralMemory at ${server.baseUrl}`,
          );
        } catch (err) {
          statusBar.setStatus("disconnected");
          vscode.window.showErrorMessage(
            `Connection failed: ${err instanceof Error ? err.message : err}`,
          );
        }
      },
    ),
  );
}

/**
 * Register the Graph Explorer command (Phase 8).
 */
function registerGraphCommands(
  context: vscode.ExtensionContext,
  server: ServerLifecycle,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.openGraph", () => {
      GraphPanel.createOrShow(server, context.extensionUri);
    }),
  );
}

/**
 * Connect WebSocket for live sync events and wire to UI components.
 */
async function connectWebSocket(
  baseUrl: string,
  brainId: string,
  outputChannel: vscode.OutputChannel,
  targets: {
    readonly codeLens: MemoryCodeLensProvider;
    readonly triggerWatcher: CommentTriggerWatcher;
    readonly memoryTree: MemoryTreeProvider;
    readonly statusBar: StatusBarManager;
  },
): Promise<SyncClient | undefined> {
  const syncClient = new SyncClient(baseUrl);

  // Wire events to UI refresh
  syncClient.onEvent((event) => {
    const type = event.type as string;

    // Memory encoded/queried → refresh everything
    if (type === "memory_encoded" || type === "memory_queried") {
      targets.memoryTree.refresh();
      targets.codeLens.refresh();
      targets.triggerWatcher.refresh();
      targets.statusBar.refresh();
      GraphPanel.refreshIfOpen();
      return;
    }

    // Neuron/synapse/fiber changes → refresh tree + stats + graph
    if (
      type.startsWith("neuron_") ||
      type.startsWith("synapse_") ||
      type.startsWith("fiber_")
    ) {
      targets.memoryTree.refresh();
      targets.statusBar.refresh();
      GraphPanel.refreshIfOpen();
      return;
    }
  });

  try {
    await syncClient.connect();
    syncClient.subscribe(brainId);
    outputChannel.appendLine(`WebSocket connected, subscribed to brain: ${brainId}`);
    return syncClient;
  } catch (err) {
    outputChannel.appendLine(
      `WebSocket connection failed (non-critical): ${err}. Live updates disabled.`,
    );
    return undefined;
  }
}
