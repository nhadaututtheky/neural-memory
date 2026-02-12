/**
 * NeuralMemory Dashboard — Cytoscape.js graph module
 * Neural graph visualization with fCOSE/COSE layout, search, filter, zoom,
 * layout switcher, dynamic node limit, compound clustering, progressive loading.
 */

const NM_GRAPH = {
  _cy: null,
  _allElements: [],
  _totalNodes: 0,
  _totalEdges: 0,
  _currentLimit: 500,
  _currentOffset: 0,
  _currentLayout: 'fcose',
  _clusteringEnabled: false,

  TYPE_COLORS: {
    concept: '#e94560',
    entity: '#4ecdc4',
    time: '#ffe66d',
    action: '#95e1d3',
    state: '#f38181',
    spatial: '#45b7d1',
    sensory: '#96ceb4',
    intent: '#dda0dd',
    default: '#aa96da',
  },

  LAYOUTS: {
    fcose: {
      name: typeof cytoscapeFcose !== 'undefined' ? 'fcose' : 'cose',
      animate: false,
      quality: 'default',
      nodeDimensionsIncludeLabels: true,
      idealEdgeLength: 80,
      nodeRepulsion: 4500,
      edgeElasticity: 0.45,
      nestingFactor: 0.1,
      gravity: 0.25,
      gravityRange: 3.8,
      numIter: 2500,
      randomize: true,
      componentSpacing: 100,
      tile: true,
    },
    cose: {
      name: 'cose',
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
    concentric: {
      name: 'concentric',
      animate: false,
      minNodeSpacing: 30,
      concentric: (node) => node.degree(),
      levelWidth: () => 2,
    },
    breadthfirst: {
      name: 'breadthfirst',
      animate: false,
      directed: false,
      spacingFactor: 1.25,
      avoidOverlap: true,
    },
  },

  async init(containerId) {
    // Register fCOSE extension if available
    if (typeof cytoscapeFcose !== 'undefined' && typeof cytoscape !== 'undefined') {
      try {
        cytoscape.use(cytoscapeFcose);
      } catch {
        // Already registered
      }
    }

    const data = await this.fetchData(this._currentLimit, 0);
    if (!data) return null;

    this._totalNodes = data.total_neurons || 0;
    this._totalEdges = data.total_synapses || 0;
    this._currentOffset = (data.neurons || []).length;

    const elements = this.buildElements(data);
    this._allElements = elements;

    if (elements.length === 0) return null;

    this._cy = cytoscape({
      container: document.getElementById(containerId),
      elements: elements,
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'width': 'data(size)',
            'height': 'data(size)',
            'font-size': '10px',
            'font-family': 'Fira Code, monospace',
            'color': '#F8FAFC',
            'text-outline-color': '#0F172A',
            'text-outline-width': 2,
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 6,
            'text-max-width': '80px',
            'text-wrap': 'ellipsis',
            'border-width': 2,
            'border-color': 'data(borderColor)',
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#F8FAFC',
            'overlay-opacity': 0.1,
          }
        },
        {
          selector: ':parent',
          style: {
            'background-color': '#1E293B',
            'background-opacity': 0.5,
            'border-width': 1,
            'border-color': '#475569',
            'label': 'data(label)',
            'font-size': '12px',
            'text-valign': 'top',
            'text-halign': 'center',
            'padding': '15px',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(weight)',
            'line-color': '#475569',
            'target-arrow-color': '#475569',
            'target-arrow-shape': 'data(arrowShape)',
            'curve-style': 'bezier',
            'opacity': 0.6,
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#22C55E',
            'target-arrow-color': '#22C55E',
            'opacity': 1,
          }
        },
        {
          selector: '.dimmed',
          style: {
            'opacity': 0.12,
          }
        },
        {
          selector: '.highlighted',
          style: {
            'border-width': 3,
            'border-color': '#22C55E',
            'opacity': 1,
          }
        },
        {
          selector: '.filtered-out',
          style: {
            'display': 'none',
          }
        }
      ],
      layout: this.LAYOUTS[this._currentLayout] || this.LAYOUTS.cose,
    });

    return this._cy;
  },

  async fetchData(limit, offset) {
    try {
      const params = new URLSearchParams({ limit: String(limit || 500), offset: String(offset || 0) });
      const resp = await fetch(`/api/graph?${params}`);
      if (!resp.ok) return null;
      return await resp.json();
    } catch {
      return null;
    }
  },

  buildElements(data) {
    const nodes = (data.neurons || []).map(n => {
      const color = this.TYPE_COLORS[n.type] || this.TYPE_COLORS.default;
      return {
        data: {
          id: n.id,
          label: this.truncate(n.content, 30),
          color: color,
          borderColor: color,
          size: 20,
          type: n.type,
          content: n.content,
          metadata: n.metadata,
        }
      };
    });

    const nodeIds = new Set(nodes.map(n => n.data.id));
    const edges = (data.synapses || [])
      .filter(s => nodeIds.has(s.source_id) && nodeIds.has(s.target_id))
      .map(s => {
        const weight = Math.min(6, Math.max(1, (s.weight || 0.5) * 3));
        const arrowShape = s.direction === 'bidirectional' ? 'none' : 'triangle';
        return {
          data: {
            id: s.id,
            source: s.source_id,
            target: s.target_id,
            weight: weight,
            arrowShape: arrowShape,
            type: s.type,
          }
        };
      });

    return [...nodes, ...edges];
  },

  truncate(text, max) {
    if (!text) return '';
    return text.length > max ? text.slice(0, max) + '...' : text;
  },

  async reload() {
    if (this._cy) {
      this._cy.destroy();
      this._cy = null;
    }
    this._currentOffset = 0;
    return await this.init('cy-graph');
  },

  onNodeClick(callback) {
    if (!this._cy) return;
    this._cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      if (node.isParent()) return; // ignore cluster clicks
      callback({
        id: node.data('id'),
        type: node.data('type'),
        content: node.data('content'),
        metadata: node.data('metadata'),
      });
    });
  },

  // ── Layout Switcher ──────────────────────────────────

  setLayout(layoutName) {
    if (!this._cy) return;
    if (!this.LAYOUTS[layoutName]) return;

    this._currentLayout = layoutName;
    const layout = this._cy.layout(this.LAYOUTS[layoutName]);
    layout.run();
  },

  getLayoutName() {
    return this._currentLayout;
  },

  // ── Dynamic Node Limit ───────────────────────────────

  async setNodeLimit(limit) {
    this._currentLimit = Math.min(Math.max(limit, 50), 2000);
    await this.reload();
  },

  getNodeLimit() {
    return this._currentLimit;
  },

  // ── Progressive Loading ──────────────────────────────

  async loadMore(count) {
    if (!this._cy || this._loadingMore) return 0;
    this._loadingMore = true;
    const batchSize = count || 200;

    try {
    const data = await this.fetchData(batchSize, this._currentOffset);
    if (!data) { this._loadingMore = false; return 0; }

    const newElements = this.buildElements(data);
    if (newElements.length === 0) return 0;

    this._cy.add(newElements);
    this._allElements = [...this._allElements, ...newElements];
    this._currentOffset += (data.neurons || []).length;

    // Re-run layout
    const layout = this._cy.layout(this.LAYOUTS[this._currentLayout] || this.LAYOUTS.cose);
    layout.run();

    return newElements.filter(e => !e.data.source).length; // node count only
    } finally { this._loadingMore = false; }
  },

  hasMore() {
    return this._currentOffset < this._totalNodes;
  },

  // ── Compound Node Clustering ─────────────────────────

  toggleClustering() {
    if (!this._cy) return;
    this._clusteringEnabled = !this._clusteringEnabled;

    if (this._clusteringEnabled) {
      this._applyClustering();
    } else {
      this._removeClustering();
    }
  },

  isClusteringEnabled() {
    return this._clusteringEnabled;
  },

  _applyClustering() {
    if (!this._cy) return;

    // Group nodes by type as compound parent nodes
    const types = this.getTypes();
    for (const type of types) {
      const parentId = `cluster_${type}`;
      // Add parent if not exists
      if (this._cy.getElementById(parentId).length === 0) {
        this._cy.add({
          data: {
            id: parentId,
            label: type.charAt(0).toUpperCase() + type.slice(1),
          },
          classes: 'cluster-parent',
        });
      }

      // Move nodes into cluster
      this._cy.nodes().forEach(n => {
        if (n.data('type') === type && !n.isParent()) {
          n.move({ parent: parentId });
        }
      });
    }

    // Re-layout
    const layout = this._cy.layout(this.LAYOUTS[this._currentLayout] || this.LAYOUTS.cose);
    layout.run();
  },

  _removeClustering() {
    if (!this._cy) return;

    // Move all children out of parents
    this._cy.nodes().forEach(n => {
      if (n.isChild()) {
        n.move({ parent: null });
      }
    });

    // Remove cluster parent nodes
    this._cy.nodes('.cluster-parent').remove();

    // Re-layout
    const layout = this._cy.layout(this.LAYOUTS[this._currentLayout] || this.LAYOUTS.cose);
    layout.run();
  },

  // ── Toolbar: Zoom ───────────────────────────────────

  zoomIn() {
    if (!this._cy) return;
    this._cy.zoom({
      level: this._cy.zoom() * 1.3,
      renderedPosition: { x: this._cy.width() / 2, y: this._cy.height() / 2 },
    });
  },

  zoomOut() {
    if (!this._cy) return;
    this._cy.zoom({
      level: this._cy.zoom() / 1.3,
      renderedPosition: { x: this._cy.width() / 2, y: this._cy.height() / 2 },
    });
  },

  fit() {
    if (!this._cy) return;
    this._cy.fit(undefined, 30);
  },

  // ── Toolbar: Search ─────────────────────────────────

  search(query) {
    if (!this._cy) return 0;
    this._cy.elements().removeClass('highlighted dimmed');

    if (!query || !query.trim()) return 0;

    const q = query.toLowerCase().trim();
    const matching = this._cy.nodes().filter(n => {
      const content = (n.data('content') || '').toLowerCase();
      const type = (n.data('type') || '').toLowerCase();
      const label = (n.data('label') || '').toLowerCase();
      return content.includes(q) || type.includes(q) || label.includes(q);
    });

    if (matching.length > 0) {
      this._cy.elements().addClass('dimmed');
      matching.removeClass('dimmed').addClass('highlighted');
      matching.connectedEdges().removeClass('dimmed');
      this._cy.fit(matching, 50);
    }

    return matching.length;
  },

  clearSearch() {
    if (!this._cy) return;
    this._cy.elements().removeClass('highlighted dimmed');
  },

  // ── Toolbar: Filter by type ─────────────────────────

  filterByType(type) {
    if (!this._cy) return;
    this._cy.elements().removeClass('filtered-out');

    if (!type) return;

    this._cy.nodes().forEach(n => {
      if (n.data('type') !== type) {
        n.addClass('filtered-out');
        n.connectedEdges().addClass('filtered-out');
      }
    });
  },

  // ── Real-time: Add node from WebSocket event ────────

  addNode(neuron) {
    if (!this._cy) return;
    if (this._cy.getElementById(neuron.id).length > 0) return; // skip duplicate
    const color = this.TYPE_COLORS[neuron.type] || this.TYPE_COLORS.default;
    this._cy.add({
      data: {
        id: neuron.id,
        label: this.truncate(neuron.content, 30),
        color: color,
        borderColor: color,
        size: 20,
        type: neuron.type,
        content: neuron.content,
        metadata: neuron.metadata,
      }
    });
    this._totalNodes++;
  },

  addEdge(synapse) {
    if (!this._cy) return;
    if (this._cy.getElementById(synapse.id).length > 0) return; // skip duplicate
    const src = this._cy.getElementById(synapse.source_id);
    const tgt = this._cy.getElementById(synapse.target_id);
    if (src.length === 0 || tgt.length === 0) return;

    this._cy.add({
      data: {
        id: synapse.id,
        source: synapse.source_id,
        target: synapse.target_id,
        weight: Math.min(6, Math.max(1, (synapse.weight || 0.5) * 3)),
        arrowShape: synapse.direction === 'bidirectional' ? 'none' : 'triangle',
        type: synapse.type,
      }
    });
    this._totalEdges++;
  },

  // ── Utilities ───────────────────────────────────────

  getTypes() {
    if (!this._cy) return [];
    const types = new Set();
    this._cy.nodes().forEach(n => {
      const t = n.data('type');
      if (t) types.add(t);
    });
    return Array.from(types).sort();
  },

  isEmpty() {
    return !this._cy || this._cy.nodes().length === 0;
  },

  nodeCount() {
    return this._cy ? this._cy.nodes().filter(n => !n.isParent()).length : 0;
  },

  edgeCount() {
    return this._cy ? this._cy.edges().length : 0;
  },

  totalNodeCount() {
    return this._totalNodes;
  },

  totalEdgeCount() {
    return this._totalEdges;
  },
};
