# Server API

NeuralMemory provides a FastAPI-based REST API server.

## Quick Start

```bash
pip install neural-memory[server]
nmem serve --host 0.0.0.0 --port 8000
```

Or with uvicorn directly:

```bash
uvicorn neural_memory.server:app --reload --port 8000
```

## Endpoints

### Health Check

#### GET /health

Check server health.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.6.0"
}
```

### Root

#### GET /

API information.

**Response:**

```json
{
  "name": "NeuralMemory",
  "description": "Reflex-based memory system for AI agents",
  "version": "0.6.0",
  "docs": "/docs",
  "health": "/health",
  "ui": "/ui"
}
```

---

## Memory Operations

### Encode Memory

#### POST /memory/encode

Store a new memory.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-Brain-ID` | Yes | Brain identifier |

**Request Body:**

```json
{
  "content": "Met Alice to discuss API design",
  "tags": ["meeting", "api"],
  "metadata": {
    "location": "office"
  }
}
```

**Response:**

```json
{
  "fiber_id": "fiber-abc123",
  "neurons_created": 4,
  "synapses_created": 6
}
```

### Query Memories

#### POST /memory/query

Query memories using spreading activation.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-Brain-ID` | Yes | Brain identifier |

**Request Body:**

```json
{
  "query": "What did Alice say about the API?",
  "depth": 1,
  "max_tokens": 500
}
```

**Response:**

```json
{
  "context": "Alice suggested using REST for the API design...",
  "confidence": 0.85,
  "neurons_activated": 12,
  "depth_used": 1,
  "fibers_matched": ["fiber-abc123"],
  "co_activations": 3,
  "use_reflex": true
}
```

### Get Neurons

#### GET /memory/neurons

List neurons with optional filters.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Filter by neuron type |
| `limit` | integer | Max results (default: 50) |

**Response:**

```json
{
  "neurons": [
    {
      "id": "neuron-123",
      "type": "entity",
      "content": "Alice",
      "metadata": {}
    }
  ],
  "total": 1
}
```

### Get Fiber

#### GET /memory/fiber/{fiber_id}

Get a specific fiber.

**Response:**

```json
{
  "id": "fiber-abc123",
  "neuron_ids": ["n1", "n2", "n3"],
  "synapse_ids": ["s1", "s2"],
  "pathway": ["n1", "n2", "n3"],
  "conductivity": 0.95,
  "last_conducted": "2026-02-05T09:30:00Z",
  "summary": "Meeting with Alice about API",
  "created_at": "2026-02-05T10:00:00Z"
}
```

---

## Brain Operations

### Create Brain

#### POST /brain/create

Create a new brain.

**Request Body:**

```json
{
  "name": "my-brain",
  "is_public": false,
  "config": {
    "decay_rate": 0.1,
    "max_spread_hops": 4
  }
}
```

**Response:**

```json
{
  "id": "brain-xyz789",
  "name": "my-brain",
  "created_at": "2026-02-05T10:00:00Z"
}
```

### Get Brain

#### GET /brain/{brain_id}

Get brain information.

**Response:**

```json
{
  "id": "brain-xyz789",
  "name": "my-brain",
  "config": {
    "decay_rate": 0.1,
    "reinforcement_delta": 0.05,
    "activation_threshold": 0.2,
    "max_spread_hops": 4,
    "max_context_tokens": 1500
  },
  "stats": {
    "neuron_count": 150,
    "synapse_count": 280,
    "fiber_count": 45
  },
  "created_at": "2026-02-05T10:00:00Z"
}
```

### Get Brain Stats

#### GET /brain/{brain_id}/stats

Get detailed brain statistics.

**Response:**

```json
{
  "neuron_count": 150,
  "synapse_count": 280,
  "fiber_count": 45,
  "memory_types": {
    "fact": 30,
    "decision": 20,
    "todo": 15,
    "insight": 10
  },
  "freshness": {
    "fresh": 40,
    "recent": 30,
    "aging": 20,
    "stale": 10
  }
}
```

### Export Brain

#### GET /brain/{brain_id}/export

Export brain as JSON snapshot.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `exclude_sensitive` | boolean | Exclude sensitive content |

**Response:**

```json
{
  "brain_id": "brain-xyz789",
  "exported_at": "2026-02-05T10:00:00Z",
  "version": "0.6.0",
  "neurons": [...],
  "synapses": [...],
  "fibers": [...]
}
```

### Import Brain

#### POST /brain/{brain_id}/import

Import brain from snapshot.

**Request Body:** Brain snapshot JSON

**Response:**

```json
{
  "imported": true,
  "neurons_imported": 150,
  "synapses_imported": 280,
  "fibers_imported": 45
}
```

### Delete Brain

#### DELETE /brain/{brain_id}

Delete a brain.

**Response:**

```json
{
  "deleted": true
}
```

---

## Sync Operations

### WebSocket Sync

#### WS /sync/ws

Real-time synchronization via WebSocket.

**Connect:**

```javascript
const ws = new WebSocket('ws://localhost:8000/sync/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'connect',
    client_id: 'client-1'
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data);
};
```

**Messages:**

Subscribe to brain updates:
```json
{"action": "subscribe", "brain_id": "brain-123"}
```

Receive updates:
```json
{"type": "neuron_created", "brain_id": "brain-123", "data": {...}}
{"type": "memory_encoded", "brain_id": "brain-123", "data": {...}}
```

---

## Visualization

### Get Graph Data

#### GET /api/graph

Get graph data for visualization.

**Response:**

```json
{
  "neurons": [
    {
      "id": "n1",
      "type": "entity",
      "content": "Alice",
      "metadata": {}
    }
  ],
  "synapses": [
    {
      "id": "s1",
      "source_id": "n1",
      "target_id": "n2",
      "type": "discussed",
      "weight": 0.8
    }
  ],
  "fibers": [
    {
      "id": "f1",
      "summary": "Meeting notes",
      "neuron_count": 5
    }
  ],
  "stats": {
    "neuron_count": 150,
    "synapse_count": 280,
    "fiber_count": 45
  }
}
```

### Web Dashboard

#### GET /ui

Serve the React dashboard (SPA). Pages: Overview, Health, Graph, Timeline, Evolution, Diagrams, Settings.

Built with React 19 + TailwindCSS 4 + shadcn/ui + Recharts + Sigma.js. Warm cream light theme.

#### GET /ui-legacy

Serve the legacy vis.js graph visualization (backward compat).

#### GET /dashboard-legacy

Serve the legacy Alpine.js dashboard (backward compat).

---

## Dashboard API

### Brain Files

#### GET /api/dashboard/brain-files

Get brain file paths and disk usage.

**Response:**

```json
{
  "brains_dir": "/home/user/.neuralmemory/brains",
  "brains": [
    {
      "name": "default",
      "path": "/home/user/.neuralmemory/brains/default.db",
      "size_bytes": 1048576,
      "is_active": true
    }
  ],
  "total_size_bytes": 1048576
}
```

### Telegram Status

#### GET /api/dashboard/telegram/status

Get Telegram integration status and bot info.

**Response:**

```json
{
  "configured": true,
  "bot_name": "MyBrainBot",
  "bot_username": "mybrainbot",
  "chat_ids": ["123456789"],
  "backup_on_consolidation": false,
  "error": null
}
```

### Telegram Test

#### POST /api/dashboard/telegram/test

Send a test message to all configured Telegram chats.

**Response:**

```json
{
  "status": "ok",
  "sent_to": 1,
  "failed": 0
}
```

### Telegram Backup

#### POST /api/dashboard/telegram/backup

Send brain .db file as backup to all configured Telegram chats.

**Request Body (optional):**

```json
{
  "brain_name": "my-brain"
}
```

**Response:**

```json
{
  "status": "ok",
  "brain": "my-brain",
  "size_mb": 1.5,
  "sent_to": 1,
  "failed": 0
}
```

---

## Error Responses

All errors return standard format:

```json
{
  "detail": "Error message here"
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 404 | Not found |
| 422 | Validation error |
| 500 | Server error |

---

## Python Client Example

```python
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Create brain
        response = await client.post(
            "http://localhost:8000/brain/create",
            json={"name": "my-brain"}
        )
        brain = response.json()

        # Encode memory
        response = await client.post(
            "http://localhost:8000/memory/encode",
            headers={"X-Brain-ID": brain["id"]},
            json={"content": "Important decision made"}
        )

        # Query
        response = await client.post(
            "http://localhost:8000/memory/query",
            headers={"X-Brain-ID": brain["id"]},
            json={"query": "What decision?", "depth": 1}
        )
        result = response.json()
        print(result["context"])
```

---

## CORS Configuration

The server allows all origins by default. For production, configure:

```python
from neural_memory.server import create_app

app = create_app(
    cors_origins=["https://yourdomain.com"]
)
```

---

## OpenAPI Documentation

Interactive API documentation available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
