/**
 * Unit tests for NeuralMemoryClient (HTTP client).
 *
 * Tests run with plain Mocha — no VS Code runtime needed.
 * Uses a mock fetch to verify request construction and response handling.
 */

import * as assert from "assert";

// --- Mock fetch infrastructure ---

interface MockCall {
  readonly url: string;
  readonly init: RequestInit;
}

const calls: MockCall[] = [];
let mockResponse: { status: number; ok: boolean; body: unknown } = {
  status: 200,
  ok: true,
  body: {},
};

function resetMock(): void {
  calls.length = 0;
  mockResponse = { status: 200, ok: true, body: {} };
}

function setMockResponse(
  status: number,
  body: unknown,
  ok?: boolean,
): void {
  mockResponse = { status, ok: ok ?? (status >= 200 && status < 300), body };
}

// Install mock fetch before importing client
const originalFetch = globalThis.fetch;

globalThis.fetch = (async (
  input: string | URL | Request,
  init?: RequestInit,
) => {
  const url = typeof input === "string" ? input : input.toString();
  calls.push({ url, init: init ?? {} });
  return {
    ok: mockResponse.ok,
    status: mockResponse.status,
    statusText: mockResponse.ok ? "OK" : "Error",
    json: async () => mockResponse.body,
    headers: new Headers(),
  } as Response;
}) as typeof fetch;

// Now import client (it uses global fetch)
import { NeuralMemoryClient, ApiError } from "../../src/server/client";

// Restore original fetch after all tests
after(() => {
  globalThis.fetch = originalFetch;
});

describe("NeuralMemoryClient", () => {
  const BASE_URL = "http://127.0.0.1:8000";
  let client: NeuralMemoryClient;

  beforeEach(() => {
    resetMock();
    client = new NeuralMemoryClient(BASE_URL);
  });

  describe("health()", () => {
    it("should GET /health", async () => {
      setMockResponse(200, { status: "healthy", version: "0.6.0" });

      const result = await client.health();

      assert.strictEqual(calls.length, 1);
      assert.ok(calls[0].url.endsWith("/health"));
      assert.strictEqual(calls[0].init.method, "GET");
      assert.strictEqual(result.status, "healthy");
      assert.strictEqual(result.version, "0.6.0");
    });
  });

  describe("encode()", () => {
    it("should POST /memory/encode with brain header", async () => {
      const body = {
        fiber_id: "f1",
        neurons_created: 3,
        neurons_linked: 1,
        synapses_created: 2,
      };
      setMockResponse(200, body);

      const result = await client.encode("brain-1", {
        content: "test content",
        tags: ["fact"],
      });

      assert.strictEqual(calls.length, 1);
      assert.ok(calls[0].url.endsWith("/memory/encode"));
      assert.strictEqual(calls[0].init.method, "POST");

      const headers = calls[0].init.headers as Record<string, string>;
      assert.strictEqual(headers["X-Brain-ID"], "brain-1");
      assert.strictEqual(headers["Content-Type"], "application/json");

      const sentBody = JSON.parse(calls[0].init.body as string);
      assert.strictEqual(sentBody.content, "test content");
      assert.deepStrictEqual(sentBody.tags, ["fact"]);

      assert.strictEqual(result.fiber_id, "f1");
      assert.strictEqual(result.neurons_created, 3);
    });
  });

  describe("query()", () => {
    it("should POST /memory/query with brain header", async () => {
      const body = {
        answer: "test answer",
        confidence: 0.85,
        depth_used: 1,
        neurons_activated: 5,
        fibers_matched: ["f1"],
        context: "ctx",
        latency_ms: 42,
        subgraph: null,
        metadata: {},
      };
      setMockResponse(200, body);

      const result = await client.query("brain-1", {
        query: "what is auth?",
        depth: 1,
      });

      assert.strictEqual(calls.length, 1);
      assert.ok(calls[0].url.endsWith("/memory/query"));

      const headers = calls[0].init.headers as Record<string, string>;
      assert.strictEqual(headers["X-Brain-ID"], "brain-1");

      assert.strictEqual(result.answer, "test answer");
      assert.strictEqual(result.confidence, 0.85);
    });
  });

  describe("listNeurons()", () => {
    it("should GET /memory/neurons with query params", async () => {
      setMockResponse(200, { neurons: [], count: 0 });

      await client.listNeurons("brain-1", {
        type: "concept",
        contentContains: "auth",
        limit: 10,
      });

      assert.strictEqual(calls.length, 1);
      const url = calls[0].url;
      assert.ok(url.includes("/memory/neurons?"));
      assert.ok(url.includes("type=concept"));
      assert.ok(url.includes("content_contains=auth"));
      assert.ok(url.includes("limit=10"));
    });

    it("should GET /memory/neurons without params when none given", async () => {
      setMockResponse(200, { neurons: [], count: 0 });

      await client.listNeurons("brain-1");

      assert.strictEqual(calls.length, 1);
      assert.ok(calls[0].url.endsWith("/memory/neurons"));
    });
  });

  describe("getFiber()", () => {
    it("should GET /memory/fiber/:id with brain header", async () => {
      const fiber = {
        id: "f1",
        neuron_ids: ["n1"],
        synapse_ids: ["s1"],
        anchor_neuron_id: null,
        time_start: null,
        time_end: null,
        coherence: 0.9,
        salience: 0.8,
        frequency: 1,
        summary: "test",
        tags: [],
        created_at: "2025-01-01T00:00:00Z",
      };
      setMockResponse(200, fiber);

      const result = await client.getFiber("brain-1", "f1");

      assert.ok(calls[0].url.includes("/memory/fiber/f1"));
      assert.strictEqual(result.id, "f1");
      assert.strictEqual(result.coherence, 0.9);
    });
  });

  describe("getGraph()", () => {
    it("should GET /api/graph", async () => {
      const graph = {
        neurons: [],
        synapses: [],
        fibers: [],
        stats: { neuron_count: 0, synapse_count: 0, fiber_count: 0 },
      };
      setMockResponse(200, graph);

      const result = await client.getGraph();

      assert.ok(calls[0].url.endsWith("/api/graph"));
      assert.deepStrictEqual(result.stats, graph.stats);
    });
  });

  describe("brain operations", () => {
    it("getBrain should GET /brain/:id", async () => {
      setMockResponse(200, {
        id: "b1",
        name: "test",
        owner_id: null,
        is_public: false,
        neuron_count: 0,
        synapse_count: 0,
        fiber_count: 0,
        created_at: "",
        updated_at: "",
      });

      const result = await client.getBrain("b1");
      assert.ok(calls[0].url.includes("/brain/b1"));
      assert.strictEqual(result.name, "test");
    });

    it("createBrain should POST /brain/create", async () => {
      setMockResponse(200, {
        id: "b2",
        name: "new-brain",
        owner_id: null,
        is_public: false,
        neuron_count: 0,
        synapse_count: 0,
        fiber_count: 0,
        created_at: "",
        updated_at: "",
      });

      const result = await client.createBrain({ name: "new-brain" });
      assert.ok(calls[0].url.endsWith("/brain/create"));
      assert.strictEqual(result.name, "new-brain");
    });

    it("deleteBrain should DELETE /brain/:id", async () => {
      setMockResponse(200, { status: "deleted", brain_id: "b1" });

      const result = await client.deleteBrain("b1");
      assert.ok(calls[0].url.includes("/brain/b1"));
      assert.strictEqual(calls[0].init.method, "DELETE");
      assert.strictEqual(result.status, "deleted");
    });
  });

  describe("error handling", () => {
    it("should throw ApiError on non-2xx response", async () => {
      setMockResponse(404, { error: "Not found", detail: "Brain not found" });

      try {
        await client.getBrain("nonexistent");
        assert.fail("Should have thrown");
      } catch (err) {
        assert.ok(err instanceof ApiError);
        assert.strictEqual(err.status, 404);
        assert.ok(err.message.includes("404"));
        assert.strictEqual(err.detail, "Brain not found");
      }
    });

    it("should handle non-JSON error body", async () => {
      mockResponse = {
        status: 500,
        ok: false,
        body: null,
      };
      // Override json to throw
      const origFetch = globalThis.fetch;
      globalThis.fetch = (async (
        input: string | URL | Request,
        init?: RequestInit,
      ) => {
        const url = typeof input === "string" ? input : input.toString();
        calls.push({ url, init: init ?? {} });
        return {
          ok: false,
          status: 500,
          statusText: "Internal Server Error",
          json: async () => {
            throw new Error("not JSON");
          },
          headers: new Headers(),
        } as Response;
      }) as typeof fetch;

      try {
        await client.health();
        assert.fail("Should have thrown");
      } catch (err) {
        assert.ok(err instanceof ApiError);
        assert.strictEqual(err.status, 500);
      } finally {
        globalThis.fetch = origFetch;
      }
    });
  });
});
