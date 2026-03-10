import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { NeuralMemoryClient } from "../server/client";
import { ServerLifecycle } from "../server/lifecycle";
import { StatusBarManager } from "../views/StatusBarManager";

const BRAINS_DIR = path.join(os.homedir(), ".neuralmemory", "brains");
const CONFIG_PATH = path.join(os.homedir(), ".neuralmemory", "config.toml");

/**
 * Read the current brain name from ~/.neuralmemory/config.toml.
 */
export function readCurrentBrain(): string {
  try {
    if (!fs.existsSync(CONFIG_PATH)) {
      return "default";
    }
    const content = fs.readFileSync(CONFIG_PATH, "utf-8");
    const match = content.match(/current_brain\s*=\s*"([^"]+)"/);
    return match?.[1] ?? "default";
  } catch {
    return "default";
  }
}

/**
 * List available brain names by scanning ~/.neuralmemory/brains/*.db.
 */
export function listLocalBrains(): readonly string[] {
  try {
    if (!fs.existsSync(BRAINS_DIR)) {
      return ["default"];
    }
    const files = fs.readdirSync(BRAINS_DIR);
    const brains = files
      .filter((f) => f.endsWith(".db"))
      .map((f) => f.replace(/\.db$/, ""));
    return brains.length > 0 ? brains : ["default"];
  } catch {
    return ["default"];
  }
}

/**
 * Write the current brain name to ~/.neuralmemory/config.toml.
 * Only updates the current_brain line, preserving the rest.
 */
function writeCurrentBrain(brainName: string): void {
  try {
    if (!fs.existsSync(CONFIG_PATH)) {
      // Create config directory + minimal config file
      const configDir = path.dirname(CONFIG_PATH);
      if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
      }
      fs.writeFileSync(
        CONFIG_PATH,
        `current_brain = "${brainName}"\n`,
        "utf-8",
      );
      return;
    }
    const content = fs.readFileSync(CONFIG_PATH, "utf-8");
    if (/current_brain\s*=/.test(content)) {
      const updated = content.replace(
        /current_brain\s*=\s*"[^"]*"/,
        `current_brain = "${brainName}"`,
      );
      fs.writeFileSync(CONFIG_PATH, updated, "utf-8");
    } else {
      // Key doesn't exist — append it
      fs.appendFileSync(CONFIG_PATH, `\ncurrent_brain = "${brainName}"\n`, "utf-8");
    }
  } catch {
    // Config write failed — non-critical, brain switch still works for this session
  }
}

interface BrainQuickPickItem extends vscode.QuickPickItem {
  readonly brainName: string;
  readonly action?: "create";
}

/**
 * Register brain-related commands.
 */
export function registerBrainCommands(
  context: vscode.ExtensionContext,
  server: ServerLifecycle,
  statusBar: StatusBarManager,
): void {
  // Switch Brain
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.switchBrain", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const currentBrain = readCurrentBrain();
      const brains = listLocalBrains();

      const items: BrainQuickPickItem[] = brains.map((name) => ({
        label: name === currentBrain ? `$(check) ${name}` : name,
        description: name === currentBrain ? "active" : undefined,
        brainName: name,
      }));

      // Add "Create new brain" option at the bottom
      items.push({
        label: "$(add) Create new brain...",
        brainName: "",
        action: "create",
        alwaysShow: true,
      });

      const selected = await vscode.window.showQuickPick(items, {
        placeHolder: `Current brain: ${currentBrain}`,
        title: "Switch Brain",
      });

      if (!selected) {
        return;
      }

      if (selected.action === "create") {
        await vscode.commands.executeCommand("neuralmemory.createBrain");
        return;
      }

      if (selected.brainName === currentBrain) {
        return; // Already active
      }

      writeCurrentBrain(selected.brainName);
      await statusBar.setBrain(selected.brainName);

      vscode.window.showInformationMessage(
        `Switched to brain: ${selected.brainName}`,
      );
    }),
  );

  // Create Brain
  context.subscriptions.push(
    vscode.commands.registerCommand("neuralmemory.createBrain", async () => {
      if (!server.isRunning()) {
        vscode.window.showWarningMessage(
          "NeuralMemory server is not running.",
        );
        return;
      }

      const existingBrains = listLocalBrains();

      const name = await vscode.window.showInputBox({
        prompt: "Enter a name for the new brain",
        placeHolder: "my-project",
        validateInput: (value) => {
          if (!value.trim()) {
            return "Brain name cannot be empty";
          }
          if (value.length > 100) {
            return "Brain name must be 100 characters or fewer";
          }
          if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
            return "Only letters, numbers, hyphens, and underscores allowed";
          }
          if (existingBrains.includes(value)) {
            return `Brain "${value}" already exists`;
          }
          return null;
        },
      });

      if (!name) {
        return;
      }

      try {
        const client = new NeuralMemoryClient(server.baseUrl);
        const brain = await client.createBrain({ name });

        writeCurrentBrain(name);
        await statusBar.setBrain(brain.id);

        vscode.window.showInformationMessage(
          `Created and switched to brain: ${name}`,
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Failed to create brain: ${err instanceof Error ? err.message : err}`,
        );
      }
    }),
  );
}
