import * as vscode from "vscode";
import { NeuralMemoryClient } from "../server/client";
import { ServerLifecycle } from "../server/lifecycle";
import { readCurrentBrain } from "../commands/brain";
import type { NeuronItem, NeuronType } from "../server/types";

/** Display order and metadata for neuron type groups. */
const TYPE_META: ReadonlyMap<
  NeuronType,
  { readonly label: string; readonly icon: string }
> = new Map([
  ["concept", { label: "Concepts", icon: "symbol-class" }],
  ["entity", { label: "Entities", icon: "symbol-variable" }],
  ["action", { label: "Actions", icon: "symbol-event" }],
  ["time", { label: "Time", icon: "calendar" }],
  ["state", { label: "State", icon: "symbol-enum" }],
  ["spatial", { label: "Files", icon: "file-code" }],
  ["sensory", { label: "Sensory", icon: "eye" }],
  ["intent", { label: "Intent", icon: "rocket" }],
]);

// ── Tree item types ──────────────────────────────────────────

class GroupItem extends vscode.TreeItem {
  constructor(
    public readonly neuronType: NeuronType,
    public readonly neurons: readonly NeuronItem[],
  ) {
    const meta = TYPE_META.get(neuronType) ?? {
      label: neuronType,
      icon: "symbol-misc",
    };
    super(`${meta.label} (${neurons.length})`, vscode.TreeItemCollapsibleState.Collapsed);
    this.iconPath = new vscode.ThemeIcon(meta.icon);
    this.contextValue = "neuralmemory.group";
    this.tooltip = `${neurons.length} ${meta.label.toLowerCase()} neurons`;
  }
}

class NeuronTreeItem extends vscode.TreeItem {
  constructor(public readonly neuron: NeuronItem) {
    const displayContent =
      neuron.content.length > 80
        ? `${neuron.content.slice(0, 77)}...`
        : neuron.content;

    super(displayContent, vscode.TreeItemCollapsibleState.None);

    const meta = TYPE_META.get(neuron.type) ?? {
      label: neuron.type,
      icon: "symbol-misc",
    };

    this.iconPath = new vscode.ThemeIcon(meta.icon);
    this.contextValue = "neuralmemory.neuron";
    this.description = formatTimestamp(neuron.created_at);
    this.tooltip = new vscode.MarkdownString(
      [
        `**${neuron.content}**`,
        "",
        `- **Type:** ${neuron.type}`,
        `- **ID:** \`${neuron.id}\``,
        `- **Created:** ${neuron.created_at}`,
      ].join("\n"),
    );

    // Click → recall related memories
    this.command = {
      command: "neuralmemory._recallFromNeuron",
      title: "Recall Related",
      arguments: [neuron.content],
    };
  }
}

// ── Provider ─────────────────────────────────────────────────

/**
 * TreeDataProvider that shows neurons grouped by type.
 *
 * ```
 * NEURALMEMORY
 * ├── Concepts (12)
 * │   ├── authentication
 * │   └── API design
 * ├── Entities (8)
 * │   ├── Alice
 * │   └── PostgreSQL
 * └── ...
 * ```
 */
export class MemoryTreeProvider
  implements vscode.TreeDataProvider<GroupItem | NeuronTreeItem>
{
  private readonly _onDidChange = new vscode.EventEmitter<
    GroupItem | NeuronTreeItem | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private _groups: readonly GroupItem[] = [];

  constructor(private readonly _server: ServerLifecycle) {}

  refresh(): void {
    this._groups = [];
    this._onDidChange.fire();
  }

  getTreeItem(
    element: GroupItem | NeuronTreeItem,
  ): vscode.TreeItem {
    return element;
  }

  async getChildren(
    element?: GroupItem | NeuronTreeItem,
  ): Promise<(GroupItem | NeuronTreeItem)[]> {
    if (!this._server.isRunning()) {
      return [];
    }

    // Root level → return groups
    if (!element) {
      return this._fetchGroups();
    }

    // Group level → return neurons in that group
    if (element instanceof GroupItem) {
      return element.neurons.map((n) => new NeuronTreeItem(n));
    }

    // Neuron level → no children
    return [];
  }

  /**
   * Register tree-view-specific commands.
   */
  registerCommands(context: vscode.ExtensionContext): void {
    // Recall from neuron (click handler)
    context.subscriptions.push(
      vscode.commands.registerCommand(
        "neuralmemory._recallFromNeuron",
        async (content: string) => {
          if (!this._server.isRunning()) {
            return;
          }

          const brainId = readCurrentBrain();
          const client = new NeuralMemoryClient(this._server.baseUrl);

          try {
            const result = await vscode.window.withProgress(
              {
                location: vscode.ProgressLocation.Notification,
                title: "Recalling related memories...",
                cancellable: false,
              },
              () => client.query(brainId, { query: content }),
            );

            if (!result.answer && result.fibers_matched.length === 0) {
              vscode.window.showInformationMessage(
                "No related memories found.",
              );
              return;
            }

            const answer = result.answer ?? result.context;
            const action = await vscode.window.showInformationMessage(
              `Found: ${truncate(answer, 100)}`,
              "Copy",
              "Full Details",
            );

            if (action === "Copy") {
              await vscode.env.clipboard.writeText(answer);
            } else if (action === "Full Details") {
              const doc = await vscode.workspace.openTextDocument({
                content: formatRecallDetails(content, result),
                language: "markdown",
              });
              await vscode.window.showTextDocument(doc, {
                preview: true,
                viewColumn: vscode.ViewColumn.Beside,
              });
            }
          } catch (err) {
            vscode.window.showErrorMessage(
              `Recall failed: ${err instanceof Error ? err.message : err}`,
            );
          }
        },
      ),
    );

    // Context menu: recall related
    context.subscriptions.push(
      vscode.commands.registerCommand(
        "neuralmemory.recallFromTree",
        async (item: NeuronTreeItem) => {
          if (item instanceof NeuronTreeItem) {
            await vscode.commands.executeCommand(
              "neuralmemory._recallFromNeuron",
              item.neuron.content,
            );
          }
        },
      ),
    );

    // Refresh
    context.subscriptions.push(
      vscode.commands.registerCommand("neuralmemory.refreshMemories", () => {
        this.refresh();
      }),
    );
  }

  // ── Private ──────────────────────────────────────────────

  private async _fetchGroups(): Promise<GroupItem[]> {
    if (this._groups.length > 0) {
      return [...this._groups];
    }

    const brainId = readCurrentBrain();
    const client = new NeuralMemoryClient(this._server.baseUrl);

    try {
      const result = await client.listNeurons(brainId, { limit: 500 });
      const grouped = groupByType(result.neurons);

      this._groups = grouped;
      return [...grouped];
    } catch {
      return [];
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────

function groupByType(
  neurons: readonly NeuronItem[],
): readonly GroupItem[] {
  const map = new Map<NeuronType, NeuronItem[]>();

  for (const n of neurons) {
    const existing = map.get(n.type);
    if (existing) {
      existing.push(n);
    } else {
      map.set(n.type, [n]);
    }
  }

  // Return in display order defined by TYPE_META
  const groups: GroupItem[] = [];
  for (const type of TYPE_META.keys()) {
    const items = map.get(type);
    if (items && items.length > 0) {
      groups.push(new GroupItem(type, items));
    }
  }

  // Any types not in TYPE_META
  for (const [type, items] of map) {
    if (!TYPE_META.has(type)) {
      groups.push(new GroupItem(type, items));
    }
  }

  return groups;
}

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60_000);

    if (diffMins < 1) {
      return "just now";
    }
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    }
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) {
      return `${diffHours}h ago`;
    }
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 30) {
      return `${diffDays}d ago`;
    }
    return date.toLocaleDateString();
  } catch {
    return iso;
  }
}

function truncate(text: string, maxLen: number): string {
  return text.length > maxLen
    ? `${text.slice(0, maxLen - 3)}...`
    : text;
}

function formatRecallDetails(
  query: string,
  result: {
    readonly answer: string | null;
    readonly confidence: number;
    readonly depth_used: number;
    readonly neurons_activated: number;
    readonly fibers_matched: readonly string[];
    readonly context: string;
    readonly latency_ms: number;
  },
): string {
  const depthLabels = ["INSTANT", "CONTEXT", "HABIT", "DEEP"];
  const depthLabel = depthLabels[result.depth_used] ?? `${result.depth_used}`;

  const lines = [
    `Query: "${query}"`,
    "─".repeat(60),
    `Confidence: ${Math.round(result.confidence * 100)}%`,
    `Depth: ${depthLabel}`,
    `Neurons activated: ${result.neurons_activated}`,
    `Fibers matched: ${result.fibers_matched.length}`,
    `Latency: ${result.latency_ms.toFixed(1)}ms`,
    "─".repeat(60),
    "",
    result.answer ?? "(no direct answer)",
    "",
    "─".repeat(60),
    "Context:",
    "",
    result.context,
  ];

  return lines.join("\n");
}
