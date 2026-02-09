/**
 * NeuralMemory Dashboard — Core Alpine.js application
 * Orchestrates tabs, stats, health, integrations status, brain management, i18n, toasts.
 */

/** Global toast dispatch — usable from any module */
function nmToast(message, type = 'success') {
  document.dispatchEvent(new CustomEvent('nm-toast', { detail: { message, type } }));
}

function dashboardApp() {
  return {
    // State
    version: '',
    locale: 'en',
    activeTab: 'overview',
    activeBrain: null,
    brains: [],
    stats: { total_brains: 0, total_neurons: 0, total_synapses: 0, total_fibers: 0 },
    healthGrade: 'F',
    purityScore: 0,
    healthWarnings: [],
    healthRecommendations: [],
    integrations: [],
    selectedNode: null,
    graphSearch: '',
    graphFilter: '',
    graphEmpty: false,
    toasts: [],
    loading: { stats: true, health: true, graph: false, integrations: false },
    _radarChart: null,
    _graphLoaded: false,
    _healthData: null,

    // Tab definitions (5 tabs)
    tabs: [
      { id: 'overview', label: 'overview', icon: 'layout-dashboard' },
      { id: 'graph', label: 'neural_graph', icon: 'share-2' },
      { id: 'integrations', label: 'integrations', icon: 'puzzle' },
      { id: 'health', label: 'brain_health', icon: 'heart-pulse' },
      { id: 'settings', label: 'settings', icon: 'sliders-horizontal' },
    ],

    // Initialize
    async init() {
      await NM_I18N.init();
      this.locale = NM_I18N.locale;

      // Listen for toast events from other modules
      document.addEventListener('nm-toast', (e) => {
        this.toast(e.detail.message, e.detail.type);
      });

      // Fetch version
      try {
        const resp = await fetch('/');
        if (resp.ok) {
          const data = await resp.json();
          this.version = data.version || '';
        }
      } catch {}

      await Promise.all([this.loadStats(), this.loadHealth()]);

      // Watch tab changes
      this.$watch('activeTab', (tab) => this.onTabChange(tab));

      // Init Lucide icons
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
    },

    // ── Toast system ───────────────────────────────────

    toast(message, type = 'success') {
      const id = Date.now() + Math.random();
      this.toasts = [...this.toasts, { id, message, type }];
      setTimeout(() => {
        this.toasts = this.toasts.filter(t => t.id !== id);
      }, 4000);
    },

    toastIcon(type) {
      const map = { success: 'check-circle', error: 'alert-circle', info: 'info', warning: 'alert-triangle' };
      return map[type] || 'info';
    },

    toastColor(type) {
      const map = {
        success: 'border-nm-cta text-nm-cta',
        error: 'border-nm-danger text-nm-danger',
        warning: 'border-nm-warning text-nm-warning',
        info: 'border-nm-info text-nm-info',
      };
      return map[type] || 'border-nm-border text-nm-muted';
    },

    // ── Tab change handlers ────────────────────────────

    async onTabChange(tab) {
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });

      if (tab === 'graph' && !this._graphLoaded) {
        this._graphLoaded = true;
        this.loading = { ...this.loading, graph: true };
        await this.$nextTick();
        const cy = await NM_GRAPH.init('cy-graph');
        this.loading = { ...this.loading, graph: false };
        this.graphEmpty = NM_GRAPH.isEmpty();
        if (cy) {
          NM_GRAPH.onNodeClick((node) => { this.selectedNode = node; });
        }
      }

      if (tab === 'integrations' && this.integrations.length === 0) {
        await this.loadIntegrations();
      }

      if (tab === 'health') {
        this.$nextTick(() => this.renderRadar());
      }
    },

    // ── i18n ───────────────────────────────────────────

    t(key) {
      return NM_I18N.t(key);
    },

    toggleLocale() {
      this.setLocale(this.locale === 'en' ? 'vi' : 'en');
    },

    async setLocale(locale) {
      await NM_I18N.setLocale(locale);
      this.locale = locale;
    },

    // ── Data loading ───────────────────────────────────

    async loadStats() {
      this.loading = { ...this.loading, stats: true };
      try {
        const resp = await fetch('/api/dashboard/stats');
        if (resp.ok) {
          const data = await resp.json();
          this.stats = data;
          this.activeBrain = data.active_brain;
          this.brains = data.brains || [];
          this.healthGrade = data.health_grade || 'F';
          this.purityScore = data.purity_score || 0;
        }
      } catch {
        this.toast(NM_I18N.t('connection_failed'), 'error');
      }
      this.loading = { ...this.loading, stats: false };
    },

    async loadHealth() {
      this.loading = { ...this.loading, health: true };
      try {
        const resp = await fetch('/api/dashboard/health');
        if (resp.ok) {
          const data = await resp.json();
          this.healthGrade = data.grade || 'F';
          this.purityScore = data.purity_score || 0;
          this.healthWarnings = data.warnings || [];
          this.healthRecommendations = data.recommendations || [];
          this._healthData = data;
        }
      } catch {
        // Health data optional — don't show error
      }
      this.loading = { ...this.loading, health: false };
    },

    async loadIntegrations() {
      this.loading = { ...this.loading, integrations: true };

      const [oauthResult, openclawResult] = await Promise.allSettled([
        fetch('/api/oauth/providers').then(r => r.ok ? r.json() : []),
        fetch('/api/openclaw/config').then(r => r.ok ? r.json() : null),
      ]);

      const providers = oauthResult.status === 'fulfilled' && Array.isArray(oauthResult.value) ? oauthResult.value : [];
      const config = openclawResult.status === 'fulfilled' ? openclawResult.value : null;

      const authCount = providers.filter(p => p.authenticated).length;
      const keyCount = config ? (config.api_keys || []).filter(k => k.enabled).length : 0;
      const tgEnabled = config?.telegram?.enabled || false;
      const dcEnabled = config?.discord?.enabled || false;

      this.integrations = [
        {
          id: 'mcp', name: 'MCP Server', icon: 'server',
          active: true,
          detail: this.version ? `v${this.version} — 16 tools` : '16 tools',
          color: '#22C55E', link: null, linkLabel: null,
        },
        {
          id: 'nanobot', name: 'Nanobot', icon: 'bot',
          active: true,
          detail: this.t('int_nanobot_detail'),
          color: '#3B82F6', link: null, linkLabel: null,
        },
        {
          id: 'cliproxy', name: 'CLIProxyAPI', icon: 'key',
          active: authCount > 0,
          detail: providers.length > 0
            ? `${authCount}/${providers.length} ` + this.t('int_providers_auth')
            : this.t('int_not_reachable'),
          color: '#D97757',
          link: 'http://127.0.0.1:8317',
          linkLabel: this.t('int_open_dashboard'),
        },
        {
          id: 'openclaw', name: 'OpenClaw', icon: 'terminal',
          active: keyCount > 0,
          detail: keyCount > 0
            ? `${keyCount} ` + this.t('int_api_keys_configured')
            : this.t('int_not_configured'),
          color: '#F59E0B',
          link: null, linkLabel: null,
        },
        {
          id: 'telegram', name: 'Telegram', icon: 'send',
          active: tgEnabled,
          detail: tgEnabled ? this.t('connected') : this.t('int_not_configured'),
          color: '#0088CC',
          link: tgEnabled ? null : 'http://127.0.0.1:8317',
          linkLabel: tgEnabled ? null : this.t('int_configure'),
        },
        {
          id: 'discord', name: 'Discord', icon: 'hash',
          active: dcEnabled,
          detail: dcEnabled ? this.t('connected') : this.t('int_not_configured'),
          color: '#5865F2',
          link: dcEnabled ? null : 'http://127.0.0.1:8317',
          linkLabel: dcEnabled ? null : this.t('int_configure'),
        },
      ];

      this.loading = { ...this.loading, integrations: false };
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
    },

    // ── Brain management ───────────────────────────────

    async switchBrain(name) {
      try {
        const resp = await fetch('/api/dashboard/brains/switch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ brain_name: name }),
        });
        if (resp.ok) {
          this.activeBrain = name;
          await Promise.all([this.loadStats(), this.loadHealth()]);
          this.toast(`Switched to ${name}`, 'success');
        }
      } catch {
        this.toast(NM_I18N.t('error_occurred'), 'error');
      }
    },

    async exportBrain() {
      if (!this.activeBrain) return;
      try {
        const resp = await fetch(`/brain/${this.activeBrain}/export`);
        if (resp.ok) {
          const data = await resp.json();
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${this.activeBrain}-brain-export.json`;
          a.click();
          URL.revokeObjectURL(url);
          this.toast(NM_I18N.t('export_success'), 'success');
        }
      } catch (err) {
        this.toast(NM_I18N.t('error_occurred') + ': ' + err.message, 'error');
      }
    },

    async importBrain(event) {
      const file = event.target.files?.[0];
      if (!file || !this.activeBrain) return;

      try {
        const text = await file.text();
        const snapshot = JSON.parse(text);
        const resp = await fetch(`/brain/${this.activeBrain}/import`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(snapshot),
        });
        if (resp.ok) {
          await Promise.all([this.loadStats(), this.loadHealth()]);
          this.toast(NM_I18N.t('import_success'), 'success');
        }
      } catch (err) {
        this.toast(NM_I18N.t('error_occurred') + ': ' + err.message, 'error');
      }
      event.target.value = '';
    },

    // ── Quick Actions ──────────────────────────────────

    async runHealthCheck() {
      await this.loadHealth();
      this.activeTab = 'health';
      this.$nextTick(() => this.renderRadar());
      this.toast(NM_I18N.t('run_health_check') + ' done', 'success');
    },

    viewWarnings() {
      this.activeTab = 'health';
    },

    // ── Graph toolbar ──────────────────────────────────

    graphZoomIn() { NM_GRAPH.zoomIn(); },
    graphZoomOut() { NM_GRAPH.zoomOut(); },
    graphFit() { NM_GRAPH.fit(); },

    searchGraph() {
      if (!this.graphSearch) {
        NM_GRAPH.clearSearch();
        return;
      }
      NM_GRAPH.search(this.graphSearch);
    },

    filterGraph() {
      NM_GRAPH.filterByType(this.graphFilter || '');
    },

    async reloadGraph() {
      this.loading = { ...this.loading, graph: true };
      this.selectedNode = null;
      this.graphSearch = '';
      this.graphFilter = '';
      const cy = await NM_GRAPH.reload();
      this.loading = { ...this.loading, graph: false };
      this.graphEmpty = NM_GRAPH.isEmpty();
      if (cy) {
        NM_GRAPH.onNodeClick((node) => { this.selectedNode = node; });
      }
    },

    // ── Integrations refresh ───────────────────────────

    async refreshIntegrations() {
      this.integrations = [];
      await this.loadIntegrations();
      this.toast(this.t('int_status_refreshed'), 'success');
    },

    // ── Health radar chart ─────────────────────────────

    renderRadar() {
      const canvas = document.getElementById('health-radar');
      if (!canvas || !this._healthData) return;

      if (this._radarChart) {
        this._radarChart.destroy();
      }

      const d = this._healthData;
      this._radarChart = new Chart(canvas, {
        type: 'radar',
        data: {
          labels: [
            this.t('connectivity'),
            this.t('diversity'),
            this.t('freshness'),
            this.t('consolidation'),
            this.t('activation'),
            this.t('recall'),
            '1 - ' + this.t('orphan_rate'),
          ],
          datasets: [{
            label: this.t('brain_health'),
            data: [
              d.connectivity || 0,
              d.diversity || 0,
              d.freshness || 0,
              d.consolidation_ratio || 0,
              d.activation_efficiency || 0,
              d.recall_confidence || 0,
              1.0 - (d.orphan_rate || 0),
            ],
            backgroundColor: 'rgba(34, 197, 94, 0.15)',
            borderColor: '#22C55E',
            pointBackgroundColor: '#22C55E',
            pointBorderColor: '#F8FAFC',
            pointBorderWidth: 1,
            borderWidth: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            r: {
              beginAtZero: true,
              max: 1,
              ticks: {
                stepSize: 0.25,
                color: '#94A3B8',
                backdropColor: 'transparent',
                font: { family: 'Fira Code', size: 10 },
              },
              grid: { color: '#334155' },
              angleLines: { color: '#334155' },
              pointLabels: {
                color: '#F8FAFC',
                font: { family: 'Fira Sans', size: 11 },
              },
            },
          },
          plugins: {
            legend: { display: false },
          },
        },
      });
    },

    // ── Utility ────────────────────────────────────────

    gradeColor(grade) {
      const map = {
        A: 'text-nm-cta',
        B: 'text-nm-info',
        C: 'text-nm-warning',
        D: 'text-orange-500',
        F: 'text-nm-danger',
      };
      return map[grade] || 'text-nm-muted';
    },
  };
}
