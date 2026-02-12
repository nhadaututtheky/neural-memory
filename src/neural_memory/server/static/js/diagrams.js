/**
 * NeuralMemory Dashboard — Mermaid diagram module
 * Renders fiber structure, type distribution pie, and synapse flow diagrams.
 */

const NM_DIAGRAMS = {
  _fibers: [],
  _currentFiber: null,
  _initialized: false,

  /**
   * Initialize Mermaid with dark theme.
   */
  async init() {
    if (this._initialized) return;
    if (typeof mermaid === 'undefined') return;

    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0F172A',
        primaryColor: '#22C55E',
        primaryTextColor: '#F8FAFC',
        primaryBorderColor: '#475569',
        secondaryColor: '#1E293B',
        secondaryTextColor: '#94A3B8',
        tertiaryColor: '#334155',
        lineColor: '#475569',
        textColor: '#F8FAFC',
        fontSize: '14px',
        fontFamily: 'Fira Code, monospace',
      },
      flowchart: { curve: 'basis', padding: 20 },
      pie: { textPosition: 0.75 },
      sequence: { actorMargin: 50, messageFontSize: 12 },
    });

    this._initialized = true;
    await this.loadFibers();
  },

  /** Load fiber list for dropdown. */
  async loadFibers() {
    try {
      const resp = await fetch('/api/dashboard/fibers?limit=100');
      if (resp.ok) {
        const data = await resp.json();
        this._fibers = data.fibers || [];
      }
    } catch {
      this._fibers = [];
    }
    return this._fibers;
  },

  get fibers() {
    return this._fibers;
  },

  // ── Fiber Structure Diagram ──────────────────────────

  /**
   * Render a fiber structure flowchart.
   * @param {string} fiberId - Fiber ID to visualize
   * @param {string} containerId - DOM element id to render into
   */
  async renderFiberStructure(fiberId, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !fiberId) return;

    try {
      const resp = await fetch(`/api/dashboard/fiber/${encodeURIComponent(fiberId)}/diagram`);
      if (!resp.ok) {
        container.innerHTML = '<p class="text-nm-muted text-sm">Failed to load fiber data</p>';
        return;
      }

      const data = await resp.json();
      const definition = this._buildFiberFlowchart(data);

      container.innerHTML = '';
      const { svg } = await mermaid.render(`fiber-${Date.now()}`, definition);
      container.innerHTML = svg;
    } catch (err) {
      container.innerHTML = `<p class="text-nm-muted text-sm">Error rendering diagram</p>`;
    }
  },

  _buildFiberFlowchart(data) {
    const neurons = data.neurons || [];
    const synapses = data.synapses || [];

    let def = 'flowchart TD\n';

    // Node shape mapping by type
    const shapeMap = {
      concept: { open: '(', close: ')' },      // rounded
      entity: { open: '([', close: '])' },      // stadium
      time: { open: '{{', close: '}}' },        // hexagon
      action: { open: '>', close: ']' },        // asymmetric
      state: { open: '[/', close: '/]' },       // parallelogram
      spatial: { open: '[', close: ']' },       // rectangle
      sensory: { open: '((', close: '))' },     // circle
      intent: { open: '[[', close: ']]' },      // subroutine
    };

    for (const n of neurons) {
      const shape = shapeMap[n.type] || { open: '[', close: ']' };
      const label = this._sanitize(this._truncate(n.content, 40));
      const id = this._sanitizeId(n.id);
      def += `    ${id}${shape.open}"${label}"${shape.close}\n`;
    }

    for (const s of synapses) {
      const src = this._sanitizeId(s.source_id);
      const tgt = this._sanitizeId(s.target_id);
      const label = this._sanitize(s.type || '');
      if (s.direction === 'bidirectional') {
        def += `    ${src} <--> |${label}| ${tgt}\n`;
      } else {
        def += `    ${src} --> |${label}| ${tgt}\n`;
      }
    }

    // Style nodes by type
    const typeGroups = {};
    for (const n of neurons) {
      const t = n.type || 'default';
      if (!typeGroups[t]) typeGroups[t] = [];
      typeGroups[t].push(this._sanitizeId(n.id));
    }

    const colors = {
      concept: '#e94560', entity: '#4ecdc4', time: '#ffe66d', action: '#95e1d3',
      state: '#f38181', spatial: '#45b7d1', sensory: '#96ceb4', intent: '#dda0dd',
    };

    let classIdx = 0;
    for (const [type, ids] of Object.entries(typeGroups)) {
      const color = colors[type] || '#aa96da';
      const cls = `cls${classIdx++}`;
      def += `    classDef ${cls} fill:${color},stroke:${color},color:#0F172A\n`;
      def += `    class ${ids.join(',')} ${cls}\n`;
    }

    return def;
  },

  // ── Type Distribution Pie Chart ──────────────────────

  /**
   * Render neuron type distribution as a Mermaid pie chart.
   * @param {string} containerId - DOM element id
   * @param {object} stats - Object with type counts { concept: 10, entity: 5, ... }
   */
  async renderTypePie(containerId, stats) {
    const container = document.getElementById(containerId);
    if (!container || !stats) return;

    let def = 'pie title Memory Type Distribution\n';
    for (const [type, count] of Object.entries(stats)) {
      if (count > 0) {
        const safeType = this._sanitize(type);
        def += `    "${safeType}" : ${count}\n`;
      }
    }

    try {
      container.innerHTML = '';
      const { svg } = await mermaid.render(`pie-${Date.now()}`, def);
      container.innerHTML = svg;
    } catch {
      container.innerHTML = '<p class="text-nm-muted text-sm">Error rendering pie chart</p>';
    }
  },

  // ── Synapse Flow Sequence Diagram ────────────────────

  /**
   * Render synapse flow as a sequence diagram.
   * @param {string} fiberId - Fiber to trace
   * @param {string} containerId - DOM element id
   */
  async renderSynapseFlow(fiberId, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !fiberId) return;

    try {
      const resp = await fetch(`/api/dashboard/fiber/${encodeURIComponent(fiberId)}/diagram`);
      if (!resp.ok) {
        container.innerHTML = '<p class="text-nm-muted text-sm">Failed to load fiber data</p>';
        return;
      }

      const data = await resp.json();
      const definition = this._buildSequenceDiagram(data);

      container.innerHTML = '';
      const { svg } = await mermaid.render(`seq-${Date.now()}`, definition);
      container.innerHTML = svg;
    } catch {
      container.innerHTML = '<p class="text-nm-muted text-sm">Error rendering sequence diagram</p>';
    }
  },

  _buildSequenceDiagram(data) {
    const neurons = data.neurons || [];
    const synapses = data.synapses || [];

    let def = 'sequenceDiagram\n';

    // Create participants in order
    const neuronMap = {};
    for (const n of neurons) {
      const alias = this._sanitizeId(n.id);
      const label = this._sanitize(this._truncate(n.content, 20));
      neuronMap[n.id] = alias;
      def += `    participant ${alias} as ${label}\n`;
    }

    // Add synapse interactions
    for (const s of synapses) {
      const src = neuronMap[s.source_id];
      const tgt = neuronMap[s.target_id];
      if (!src || !tgt) continue;

      const label = this._sanitize(s.type || 'signal');
      if (s.direction === 'bidirectional') {
        def += `    ${src} ->> ${tgt}: ${label}\n`;
        def += `    ${tgt} ->> ${src}: ${label}\n`;
      } else {
        def += `    ${src} ->> ${tgt}: ${label}\n`;
      }
    }

    return def;
  },

  // ── Utilities ────────────────────────────────────────

  _truncate(text, max) {
    if (!text) return '...';
    return text.length > max ? text.slice(0, max) + '...' : text;
  },

  _sanitize(text) {
    return (text || '')
      .replace(/["\n\r]/g, ' ')
      .replace(/[<>&'`]/g, '')
      .replace(/-->/g, ' ')
      .replace(/---/g, ' ')
      .replace(/[|()[\]{}]/g, '')
      .trim();
  },

  _sanitizeId(id) {
    // Mermaid node IDs must be alphanumeric + underscores
    return 'n_' + (id || 'unknown').replace(/[^a-zA-Z0-9]/g, '_').slice(0, 40);
  },
};
