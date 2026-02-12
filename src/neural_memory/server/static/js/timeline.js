/**
 * NeuralMemory Dashboard — Timeline visualization module
 * Chronological memory timeline with time range filter and playback controls.
 */

const NM_TIMELINE = {
  _slider: null,
  _entries: [],
  _filtered: [],
  _playbackTimer: null,
  _playbackSpeed: 1,
  _isPlaying: false,
  _typeFilter: '',

  TYPE_ICONS: {
    concept: 'lightbulb',
    entity: 'tag',
    time: 'clock',
    action: 'zap',
    state: 'toggle-left',
    spatial: 'map-pin',
    sensory: 'eye',
    intent: 'target',
    default: 'circle',
  },

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

  /**
   * Initialize the timeline with noUiSlider.
   * @param {string} sliderId - DOM id for the slider container
   */
  async init(sliderId) {
    await this.loadEntries();
    this._initSlider(sliderId);
    this._applyFilters();
  },

  /** Fetch timeline entries from API. */
  async loadEntries() {
    try {
      const resp = await fetch('/api/dashboard/timeline?limit=500');
      if (resp.ok) {
        const data = await resp.json();
        this._entries = (data.entries || []).map(e => ({
          ...e,
          _date: new Date(e.created_at),
        }));
      }
    } catch {
      this._entries = [];
    }
  },

  _initSlider(sliderId) {
    const el = document.getElementById(sliderId);
    if (!el || typeof noUiSlider === 'undefined') return;

    if (this._slider) {
      this._slider.destroy();
    }

    const now = Date.now();
    const thirtyDaysAgo = now - 30 * 24 * 60 * 60 * 1000;

    // Find actual data range
    let minTime = thirtyDaysAgo;
    let maxTime = now;
    if (this._entries.length > 0) {
      const timestamps = this._entries.map(e => e._date.getTime());
      minTime = Math.min(...timestamps, thirtyDaysAgo);
      maxTime = Math.max(...timestamps, now);
    }

    this._slider = noUiSlider.create(el, {
      start: [minTime, maxTime],
      connect: true,
      range: { min: minTime, max: maxTime },
      step: 3600000, // 1 hour
      tooltips: [
        { to: v => this._formatSliderDate(v) },
        { to: v => this._formatSliderDate(v) },
      ],
      behaviour: 'drag',
    });

    this._slider.on('update', () => {
      this._applyFilters();
    });
  },

  _formatSliderDate(value) {
    const d = new Date(Number(value));
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  },

  /** Apply time range and type filter. */
  _applyFilters() {
    let range = [0, Date.now()];
    if (this._slider) {
      const values = this._slider.get();
      range = [Number(values[0]), Number(values[1])];
    }

    this._filtered = this._entries.filter(e => {
      const t = e._date.getTime();
      if (t < range[0] || t > range[1]) return false;
      if (this._typeFilter && e.neuron_type !== this._typeFilter) return false;
      return true;
    });

    // Dispatch update event for Alpine.js to pick up
    document.dispatchEvent(new CustomEvent('nm-timeline-update', {
      detail: { entries: this.getGroupedEntries() },
    }));
  },

  /** Get entries grouped by date. */
  getGroupedEntries() {
    const groups = {};
    for (const entry of this._filtered) {
      const dateKey = entry._date.toLocaleDateString(undefined, {
        year: 'numeric', month: 'long', day: 'numeric',
      });
      if (!groups[dateKey]) {
        groups[dateKey] = { date: dateKey, sortKey: entry._date.getTime(), entries: [] };
      }
      groups[dateKey].entries.push(entry);
    }

    return Object.values(groups)
      .sort((a, b) => b.sortKey - a.sortKey)
      .map(g => ({
        ...g,
        entries: g.entries.sort((a, b) => b._date.getTime() - a._date.getTime()),
      }));
  },

  /** Set type filter. */
  setTypeFilter(type) {
    this._typeFilter = type;
    this._applyFilters();
  },

  /** Get icon for a neuron type. */
  getIcon(type) {
    return this.TYPE_ICONS[type] || this.TYPE_ICONS.default;
  },

  /** Get color for a neuron type. */
  getColor(type) {
    return this.TYPE_COLORS[type] || this.TYPE_COLORS.default;
  },

  // ── Playback controls ────────────────────────────────

  play() {
    if (this._isPlaying || !this._slider) return;
    this._isPlaying = true;

    const range = this._slider.options.range;
    const values = this._slider.get();
    let current = Number(values[0]);
    const max = range.max;
    const step = 3600000 * this._playbackSpeed; // hours per tick

    this._playbackTimer = setInterval(() => {
      current += step;
      if (current >= max) {
        this.pause();
        return;
      }
      this._slider.set([current, null]);
    }, 200);

    document.dispatchEvent(new CustomEvent('nm-timeline-playback', { detail: { playing: true } }));
  },

  pause() {
    this._isPlaying = false;
    if (this._playbackTimer) {
      clearInterval(this._playbackTimer);
      this._playbackTimer = null;
    }
    document.dispatchEvent(new CustomEvent('nm-timeline-playback', { detail: { playing: false } }));
  },

  stepForward() {
    if (!this._slider) return;
    const values = this._slider.get();
    const step = 24 * 3600000; // 1 day
    this._slider.set([Number(values[0]) + step, null]);
  },

  stepBackward() {
    if (!this._slider) return;
    const values = this._slider.get();
    const step = 24 * 3600000;
    this._slider.set([Number(values[0]) - step, null]);
  },

  setSpeed(speed) {
    this._playbackSpeed = speed;
    if (this._isPlaying) {
      this.pause();
      this.play();
    }
  },

  get isPlaying() {
    return this._isPlaying;
  },

  get entryCount() {
    return this._filtered.length;
  },

  get totalCount() {
    return this._entries.length;
  },

  /** Add a new entry from WebSocket event. */
  addEntry(entry) {
    const enriched = { ...entry, _date: new Date(entry.created_at) };
    this._entries = [enriched, ...this._entries];
    this._applyFilters();
  },

  destroy() {
    this.pause();
    if (this._slider) {
      this._slider.destroy();
      this._slider = null;
    }
  },
};
