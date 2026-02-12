/**
 * NeuralMemory Dashboard — WebSocket real-time client
 * Connects to /sync/ws, subscribes to brain events, dispatches updates.
 */

const NM_WS = {
  _ws: null,
  _clientId: null,
  _state: 'disconnected', // disconnected | connecting | connected | reconnecting
  _reconnectAttempts: 0,
  _maxReconnectAttempts: 10,
  _reconnectTimer: null,
  _pingTimer: null,
  _handlers: {},
  _subscribedBrains: new Set(),

  /** Connection states */
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',

  get state() {
    return this._state;
  },

  get isConnected() {
    return this._state === this.CONNECTED && this._ws?.readyState === WebSocket.OPEN;
  },

  /**
   * Connect to the WebSocket server.
   * @param {string} [brainId] - Optional brain to auto-subscribe to
   */
  connect(brainId) {
    if (this._state === this.CONNECTED || this._state === this.CONNECTING) return;

    this._state = this.CONNECTING;
    this._dispatch('state_change', { state: this._state });

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/sync/ws`;

    try {
      this._ws = new WebSocket(url);
    } catch {
      this._state = this.DISCONNECTED;
      this._dispatch('state_change', { state: this._state });
      this._scheduleReconnect(brainId);
      return;
    }

    this._ws.onopen = () => {
      this._send({ action: 'connect', client_id: `dashboard-${Date.now().toString(36)}` });
    };

    this._ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this._handleMessage(data, brainId);
      } catch {
        // Ignore malformed messages
      }
    };

    this._ws.onclose = () => {
      this._stopPing();
      const wasConnected = this._state === this.CONNECTED;
      this._state = this.DISCONNECTED;
      this._dispatch('state_change', { state: this._state });
      if (wasConnected) {
        this._scheduleReconnect(brainId);
      }
    };

    this._ws.onerror = () => {
      // onclose will fire after onerror
    };
  },

  /** Disconnect and stop reconnection. */
  disconnect() {
    this._reconnectAttempts = this._maxReconnectAttempts; // prevent reconnect
    this._stopPing();
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._state = this.DISCONNECTED;
    this._dispatch('state_change', { state: this._state });
  },

  /**
   * Subscribe to a brain's events.
   * @param {string} brainId
   */
  subscribe(brainId) {
    if (!brainId) return;
    this._subscribedBrains.add(brainId);
    if (this.isConnected) {
      this._send({ action: 'subscribe', brain_id: brainId });
    }
  },

  /**
   * Unsubscribe from a brain's events.
   * @param {string} brainId
   */
  unsubscribe(brainId) {
    this._subscribedBrains.delete(brainId);
    if (this.isConnected) {
      this._send({ action: 'unsubscribe', brain_id: brainId });
    }
  },

  /**
   * Register an event handler.
   * @param {string} eventType - e.g. 'memory_encoded', 'neuron_created', 'state_change'
   * @param {Function} handler - callback(data)
   */
  on(eventType, handler) {
    if (!this._handlers[eventType]) {
      this._handlers[eventType] = [];
    }
    this._handlers[eventType].push(handler);
  },

  /**
   * Unregister an event handler.
   * @param {string} eventType
   * @param {Function} [handler] - specific handler, or all if omitted
   */
  off(eventType, handler) {
    if (!this._handlers[eventType]) return;
    if (!handler) {
      delete this._handlers[eventType];
    } else {
      this._handlers[eventType] = this._handlers[eventType].filter(h => h !== handler);
    }
  },

  // ── Internal ─────────────────────────────────────────

  _send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  },

  _handleMessage(data, initialBrainId) {
    const type = data.type;

    if (type === 'connected') {
      this._clientId = data.data?.client_id;
      this._state = this.CONNECTED;
      this._reconnectAttempts = 0;
      this._dispatch('state_change', { state: this._state, clientId: this._clientId });

      // Re-subscribe to all brains
      for (const brainId of this._subscribedBrains) {
        this._send({ action: 'subscribe', brain_id: brainId });
      }
      // Auto-subscribe to initial brain
      if (initialBrainId && !this._subscribedBrains.has(initialBrainId)) {
        this.subscribe(initialBrainId);
      }

      this._startPing();
      return;
    }

    if (type === 'pong') return;

    if (type === 'subscribed') {
      this._dispatch('subscribed', data);
      return;
    }

    // Dispatch all other event types
    this._dispatch(type, data);

    // Also dispatch a generic 'event' for catch-all listeners
    this._dispatch('event', data);
  },

  _dispatch(eventType, data) {
    const handlers = this._handlers[eventType] || [];
    for (const handler of handlers) {
      try {
        handler(data);
      } catch (err) {
        console.warn(`[NM_WS] Handler error for '${eventType}':`, err);
      }
    }
  },

  _scheduleReconnect(brainId) {
    if (this._reconnectAttempts >= this._maxReconnectAttempts) return;

    this._state = this.RECONNECTING;
    this._reconnectAttempts++;
    this._dispatch('state_change', { state: this._state, attempt: this._reconnectAttempts });

    // Exponential backoff: 2s, 4s, 8s, 16s, 32s cap
    const delay = Math.min(2000 * Math.pow(2, this._reconnectAttempts - 1), 32000);

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._state = this.DISCONNECTED;
      this.connect(brainId);
    }, delay);
  },

  _startPing() {
    this._stopPing();
    this._pingTimer = setInterval(() => {
      this._send({ action: 'ping' });
    }, 30000);
  },

  _stopPing() {
    if (this._pingTimer) {
      clearInterval(this._pingTimer);
      this._pingTimer = null;
    }
  },
};
