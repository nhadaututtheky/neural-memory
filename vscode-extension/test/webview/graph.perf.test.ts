/**
 * Graph webview performance benchmarks.
 *
 * Renders the Cytoscape.js graph in a headless browser via Playwright
 * to verify rendering performance targets:
 *
 *   - 100 nodes  → <500ms
 *   - 500 nodes  → <1s
 *   - 1000 nodes → <2s
 *   - 5000 nodes → first 1000 rendered <2s (truncation)
 *
 * Usage:
 *   npx playwright test test/webview/graph.perf.test.ts
 *
 * Requires: npm install --save-dev @playwright/test
 */

import { test, expect } from "@playwright/test";
import * as path from "path";
import * as fs from "fs";

const CYTOSCAPE_PATH = path.resolve(
  __dirname,
  "../../node_modules/cytoscape/dist/cytoscape.min.js",
);

/**
 * Generate synthetic graph data for benchmarks.
 */
function generateGraphData(
  neuronCount: number,
  edgeRatio = 1.5,
): { neurons: unknown[]; synapses: unknown[] } {
  const types = ["concept", "entity", "time", "action", "state"];

  const neurons = Array.from({ length: neuronCount }, (_, i) => ({
    id: `n${i}`,
    type: types[i % types.length],
    content: `Neuron ${i}: ${randomWord()} ${randomWord()} ${randomWord()}`,
    metadata: {},
  }));

  const synapseCount = Math.floor(neuronCount * edgeRatio);
  const synapses = Array.from({ length: synapseCount }, (_, i) => ({
    id: `s${i}`,
    source_id: `n${Math.floor(Math.random() * neuronCount)}`,
    target_id: `n${Math.floor(Math.random() * neuronCount)}`,
    type: "association",
    weight: 0.3 + Math.random() * 0.7,
    direction: i % 3 === 0 ? "bidirectional" : "forward",
  }));

  return { neurons, synapses };
}

function randomWord(): string {
  const words = [
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "sigma", "omega", "neural", "memory",
  ];
  return words[Math.floor(Math.random() * words.length)];
}

/**
 * Build a standalone HTML page for benchmarking Cytoscape.js rendering.
 */
function buildBenchmarkHtml(cytoscapeJs: string): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; }
    body { width: 100vw; height: 100vh; background: #1e1e1e; }
    #cy { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="cy"></div>
  <script>${cytoscapeJs}</script>
  <script>
    const TYPE_COLORS = {
      concept: "#e94560",
      entity: "#4ecdc4",
      time: "#ffe66d",
      action: "#95e1d3",
      state: "#f38181",
    };

    window.renderGraph = function(data) {
      const start = performance.now();

      const nodes = data.neurons.map(function(n) {
        return {
          data: {
            id: n.id,
            label: n.content.substring(0, 30),
            color: TYPE_COLORS[n.type] || "#888",
            size: 20,
          },
        };
      });

      var nodeIds = new Set(nodes.map(function(n) { return n.data.id; }));

      var edges = data.synapses
        .filter(function(s) {
          return nodeIds.has(s.source_id) && nodeIds.has(s.target_id);
        })
        .map(function(s) {
          return {
            data: {
              id: s.id,
              source: s.source_id,
              target: s.target_id,
              weight: Math.max(1, Math.min(s.weight * 3, 6)),
            },
          };
        });

      var elements = nodes.concat(edges);

      var cy = cytoscape({
        container: document.getElementById("cy"),
        elements: elements,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(color)",
              label: "data(label)",
              "font-size": "10px",
              color: "#ccc",
              width: "data(size)",
              height: "data(size)",
              "min-zoomed-font-size": 8,
            },
          },
          {
            selector: "edge",
            style: {
              width: "data(weight)",
              "line-color": "#555",
              "curve-style": "bezier",
              opacity: 0.6,
            },
          },
        ],
        layout: {
          name: "cose",
          animate: false,
          nodeOverlap: 20,
          idealEdgeLength: 80,
          edgeElasticity: 100,
          nestingFactor: 1.2,
          gravity: 0.25,
          numIter: 1000,
          randomize: true,
          componentSpacing: 100,
          nodeDimensionsIncludeLabels: true,
        },
        minZoom: 0.1,
        maxZoom: 5,
      });

      var elapsed = performance.now() - start;

      return {
        elapsed: elapsed,
        nodeCount: cy.nodes().length,
        edgeCount: cy.edges().length,
      };
    };

    window.__ready = true;
  </script>
</body>
</html>`;
}

test.describe("Graph Rendering Performance", () => {
  let cytoscapeJs: string;

  test.beforeAll(() => {
    cytoscapeJs = fs.readFileSync(CYTOSCAPE_PATH, "utf-8");
  });

  test("100 nodes should render in <500ms", async ({ page }) => {
    const html = buildBenchmarkHtml(cytoscapeJs);
    await page.setContent(html);
    await page.waitForFunction("window.__ready === true");

    const data = generateGraphData(100);
    const result = await page.evaluate(
      (d: unknown) => (window as any).renderGraph(d),
      data,
    );

    console.log(
      `  100 nodes: ${result.elapsed.toFixed(0)}ms ` +
        `(${result.nodeCount} nodes, ${result.edgeCount} edges)`,
    );
    expect(result.elapsed).toBeLessThan(500);
    expect(result.nodeCount).toBe(100);
  });

  test("500 nodes should render in <1s", async ({ page }) => {
    const html = buildBenchmarkHtml(cytoscapeJs);
    await page.setContent(html);
    await page.waitForFunction("window.__ready === true");

    const data = generateGraphData(500);
    const result = await page.evaluate(
      (d: unknown) => (window as any).renderGraph(d),
      data,
    );

    console.log(
      `  500 nodes: ${result.elapsed.toFixed(0)}ms ` +
        `(${result.nodeCount} nodes, ${result.edgeCount} edges)`,
    );
    expect(result.elapsed).toBeLessThan(1000);
    expect(result.nodeCount).toBe(500);
  });

  test("1000 nodes should render in <2s", async ({ page }) => {
    const html = buildBenchmarkHtml(cytoscapeJs);
    await page.setContent(html);
    await page.waitForFunction("window.__ready === true");

    const data = generateGraphData(1000);
    const result = await page.evaluate(
      (d: unknown) => (window as any).renderGraph(d),
      data,
    );

    console.log(
      `  1000 nodes: ${result.elapsed.toFixed(0)}ms ` +
        `(${result.nodeCount} nodes, ${result.edgeCount} edges)`,
    );
    expect(result.elapsed).toBeLessThan(2000);
    expect(result.nodeCount).toBe(1000);
  });

  test("5000 nodes truncated to 1000 should render in <2s", async ({
    page,
  }) => {
    const html = buildBenchmarkHtml(cytoscapeJs);
    await page.setContent(html);
    await page.waitForFunction("window.__ready === true");

    // Generate 5000 but only pass the first 1000
    const fullData = generateGraphData(5000);
    const truncatedData = {
      neurons: fullData.neurons.slice(0, 1000),
      synapses: fullData.synapses.filter(
        (s: any) => {
          const id = parseInt(s.source_id.slice(1));
          const tid = parseInt(s.target_id.slice(1));
          return id < 1000 && tid < 1000;
        },
      ),
    };

    const result = await page.evaluate(
      (d: unknown) => (window as any).renderGraph(d),
      truncatedData,
    );

    console.log(
      `  5000→1000 nodes: ${result.elapsed.toFixed(0)}ms ` +
        `(${result.nodeCount} nodes, ${result.edgeCount} edges)`,
    );
    expect(result.elapsed).toBeLessThan(2000);
    expect(result.nodeCount).toBe(1000);
  });
});
