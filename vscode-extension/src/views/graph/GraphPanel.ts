/**
 * Webview panel controller for the Graph Explorer.
 *
 * Renders a Cytoscape.js force-directed graph of neurons and synapses.
 * Supports client-side node capping with "Load more" expansion,
 * sub-graph navigation, and VS Code theme integration.
 */

import * as vscode from "vscode";
import { NeuralMemoryClient } from "../../server/client";
import type { ServerLifecycle } from "../../server/lifecycle";
import type { GraphData } from "../../server/types";
import { readCurrentBrain } from "../../commands/brain";
import { getConfig } from "../../utils/config";
import { getGraphHtml, getNonce } from "./graphTemplate";

const VIEW_TYPE = "neuralmemory.graphExplorer";

/**
 * Message types sent FROM the webview TO the extension.
 */
interface WebviewMessage {
  readonly type: "nodeSelected" | "loadMore" | "recenter" | "ready";
  readonly nodeId?: string;
}

/**
 * Message types sent FROM the extension TO the webview.
 */
interface ExtensionMessage {
  readonly type: "graphData" | "error" | "loading";
  readonly data?: GraphPayload;
  readonly error?: string;
}

interface GraphPayload {
  readonly neurons: GraphData["neurons"];
  readonly synapses: GraphData["synapses"];
  readonly stats: GraphData["stats"];
  readonly truncated: boolean;
  readonly totalNeurons: number;
  readonly visibleNeurons: number;
  readonly nodeLimit: number;
}

export class GraphPanel {
  private static _instance: GraphPanel | undefined;

  private readonly _panel: vscode.WebviewPanel;
  private readonly _server: ServerLifecycle;
  private readonly _disposables: vscode.Disposable[] = [];

  private _fullData: GraphData | undefined;
  private _visibleCount: number;

  private constructor(
    panel: vscode.WebviewPanel,
    server: ServerLifecycle,
    extensionUri: vscode.Uri,
  ) {
    this._panel = panel;
    this._server = server;
    this._visibleCount = getConfig().graphNodeLimit;

    this._panel.webview.html = this._getHtmlContent(extensionUri);

    // Handle messages from webview
    this._panel.webview.onDidReceiveMessage(
      (msg: WebviewMessage) => this._handleMessage(msg),
      undefined,
      this._disposables,
    );

    // Cleanup on close
    this._panel.onDidDispose(
      () => this._dispose(),
      undefined,
      this._disposables,
    );
  }

  /**
   * Create or reveal the Graph Explorer panel.
   */
  static createOrShow(
    server: ServerLifecycle,
    extensionUri: vscode.Uri,
  ): GraphPanel {
    // If panel already exists, reveal it
    if (GraphPanel._instance) {
      GraphPanel._instance._panel.reveal(vscode.ViewColumn.Beside);
      GraphPanel._instance.refresh();
      return GraphPanel._instance;
    }

    const panel = vscode.window.createWebviewPanel(
      VIEW_TYPE,
      "NeuralMemory Graph",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, "dist", "lib"),
          vscode.Uri.joinPath(extensionUri, "media"),
        ],
      },
    );

    panel.iconPath = vscode.Uri.joinPath(extensionUri, "media", "brain.svg");
    GraphPanel._instance = new GraphPanel(panel, server, extensionUri);
    return GraphPanel._instance;
  }

  /**
   * Refresh the graph data from the server.
   */
  async refresh(): Promise<void> {
    this._visibleCount = getConfig().graphNodeLimit;
    await this._fetchAndSend();
  }

  /**
   * Refresh the graph if the panel is currently open.
   * Used by WebSocket event handlers — no-op if panel is closed.
   */
  static refreshIfOpen(): void {
    if (GraphPanel._instance) {
      GraphPanel._instance.refresh();
    }
  }

  private async _handleMessage(msg: WebviewMessage): Promise<void> {
    switch (msg.type) {
      case "ready":
        await this._fetchAndSend();
        break;

      case "nodeSelected":
        if (msg.nodeId) {
          await this._showNodeDetails(msg.nodeId);
        }
        break;

      case "loadMore":
        this._visibleCount = Math.min(
          this._visibleCount + getConfig().graphNodeLimit,
          this._fullData?.neurons.length ?? this._visibleCount,
        );
        this._sendSlicedData();
        break;

      case "recenter":
        if (msg.nodeId) {
          await this._fetchSubgraph(msg.nodeId);
        }
        break;
    }
  }

  private async _fetchAndSend(): Promise<void> {
    if (!this._server.isRunning()) {
      this._postMessage({ type: "error", error: "Server not connected" });
      return;
    }

    this._postMessage({ type: "loading" });

    try {
      const client = new NeuralMemoryClient(this._server.baseUrl);
      this._fullData = await client.getGraph(readCurrentBrain());
      this._sendSlicedData();
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      this._postMessage({ type: "error", error: detail });
    }
  }

  private _sendSlicedData(): void {
    if (!this._fullData) {
      return;
    }

    const allNeurons = this._fullData.neurons;
    const slicedNeurons = allNeurons.slice(0, this._visibleCount);
    const visibleIds = new Set(slicedNeurons.map((n) => n.id));

    // Only include synapses where both endpoints are visible
    const filteredSynapses = this._fullData.synapses.filter(
      (s) => visibleIds.has(s.source_id) && visibleIds.has(s.target_id),
    );

    const payload: GraphPayload = {
      neurons: slicedNeurons,
      synapses: filteredSynapses,
      stats: this._fullData.stats,
      truncated: allNeurons.length > this._visibleCount,
      totalNeurons: allNeurons.length,
      visibleNeurons: slicedNeurons.length,
      nodeLimit: this._visibleCount,
    };

    this._postMessage({ type: "graphData", data: payload });
  }

  private async _fetchSubgraph(centerId: string): Promise<void> {
    if (!this._fullData) {
      return;
    }

    // Client-side sub-graph: find all nodes connected to centerId within depth 2
    const neighborIds = new Set<string>([centerId]);

    // Depth 1
    for (const s of this._fullData.synapses) {
      if (s.source_id === centerId) {
        neighborIds.add(s.target_id);
      }
      if (s.target_id === centerId) {
        neighborIds.add(s.source_id);
      }
    }

    // Depth 2
    const depth1 = new Set(neighborIds);
    for (const s of this._fullData.synapses) {
      if (depth1.has(s.source_id)) {
        neighborIds.add(s.target_id);
      }
      if (depth1.has(s.target_id)) {
        neighborIds.add(s.source_id);
      }
    }

    const subNeurons = this._fullData.neurons.filter((n) =>
      neighborIds.has(n.id),
    );
    const subSynapses = this._fullData.synapses.filter(
      (s) => neighborIds.has(s.source_id) && neighborIds.has(s.target_id),
    );

    const payload: GraphPayload = {
      neurons: subNeurons,
      synapses: subSynapses,
      stats: this._fullData.stats,
      truncated: false,
      totalNeurons: this._fullData.neurons.length,
      visibleNeurons: subNeurons.length,
      nodeLimit: this._visibleCount,
    };

    this._postMessage({ type: "graphData", data: payload });
  }

  private async _showNodeDetails(nodeId: string): Promise<void> {
    if (!this._fullData) {
      return;
    }

    const neuron = this._fullData.neurons.find((n) => n.id === nodeId);
    if (!neuron) {
      return;
    }

    const action = await vscode.window.showInformationMessage(
      `[${neuron.type}] ${neuron.content}`,
      "Recall Related",
      "View Neighborhood",
    );

    if (action === "Recall Related") {
      await vscode.commands.executeCommand("neuralmemory.recall");
    } else if (action === "View Neighborhood") {
      await this._fetchSubgraph(nodeId);
    }
  }

  private _postMessage(msg: ExtensionMessage): void {
    this._panel.webview.postMessage(msg);
  }

  private _dispose(): void {
    GraphPanel._instance = undefined;
    for (const d of this._disposables) {
      d.dispose();
    }
  }

  /**
   * Generate the webview HTML with Cytoscape.js, CSP, and theme support.
   */
  private _getHtmlContent(extensionUri: vscode.Uri): string {
    const webview = this._panel.webview;

    const cytoscapeUri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, "dist", "lib", "cytoscape.min.js"),
    );

    const nonce = getNonce();

    return getGraphHtml(webview, cytoscapeUri, nonce);
  }
}
