/**
 * Typed HTTP client for the NeuralMemory REST API.
 *
 * All methods correspond 1:1 to server endpoints.
 * Uses native fetch (Node 18+, available in VS Code runtime).
 */

import type {
  BrainResponse,
  BrainSnapshot,
  CreateBrainRequest,
  DeleteBrainResponse,
  EncodeRequest,
  EncodeResponse,
  ErrorResponse,
  EternalRequest,
  EternalResponse,
  FiberResponse,
  GraphData,
  HealthResponse,
  ImportRequest,
  ImportResponse,
  IndexRequest,
  IndexResponse,
  NeuronListResponse,
  QueryRequest,
  QueryResponse,
  RecapRequest,
  RecapResponse,
  StatsResponse,
  SyncStatsResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * HTTP client for the NeuralMemory server.
 *
 * All methods require a brain ID header (X-Brain-ID) for
 * brain-scoped operations. Pass brainId to those methods.
 */
export class NeuralMemoryClient {
  constructor(private readonly _baseUrl: string) {}

  get baseUrl(): string {
    return this._baseUrl;
  }

  // ============ Health ============

  async health(): Promise<HealthResponse> {
    return this._get<HealthResponse>("/health");
  }

  // ============ Memory ============

  async encode(
    brainId: string,
    request: EncodeRequest,
  ): Promise<EncodeResponse> {
    return this._post<EncodeResponse>("/memory/encode", request, brainId);
  }

  async query(
    brainId: string,
    request: QueryRequest,
  ): Promise<QueryResponse> {
    return this._post<QueryResponse>("/memory/query", request, brainId);
  }

  async getFiber(
    brainId: string,
    fiberId: string,
  ): Promise<FiberResponse> {
    return this._get<FiberResponse>(
      `/memory/fiber/${encodeURIComponent(fiberId)}`,
      brainId,
    );
  }

  async listNeurons(
    brainId: string,
    options?: {
      readonly type?: string;
      readonly contentContains?: string;
      readonly limit?: number;
    },
  ): Promise<NeuronListResponse> {
    const params = new URLSearchParams();
    if (options?.type) {
      params.set("type", options.type);
    }
    if (options?.contentContains) {
      params.set("content_contains", options.contentContains);
    }
    if (options?.limit !== undefined) {
      params.set("limit", String(options.limit));
    }

    const query = params.toString();
    const path = `/memory/neurons${query ? `?${query}` : ""}`;
    return this._get<NeuronListResponse>(path, brainId);
  }

  async indexCodebase(
    brainId: string,
    request: IndexRequest,
  ): Promise<IndexResponse> {
    return this._post<IndexResponse>("/memory/index", request, brainId);
  }

  async importMemories(
    brainId: string,
    request: ImportRequest,
  ): Promise<ImportResponse> {
    return this._post<ImportResponse>("/memory/import", request, brainId);
  }

  async eternal(
    brainId: string,
    request: EternalRequest,
  ): Promise<EternalResponse> {
    return this._post<EternalResponse>("/memory/eternal", request, brainId);
  }

  async recap(
    brainId: string,
    request: RecapRequest,
  ): Promise<RecapResponse> {
    return this._post<RecapResponse>("/memory/recap", request, brainId);
  }

  // ============ Graph ============

  async getGraph(): Promise<GraphData> {
    return this._get<GraphData>("/api/graph");
  }

  // ============ Brain ============

  async getBrain(brainId: string): Promise<BrainResponse> {
    return this._get<BrainResponse>(
      `/brain/${encodeURIComponent(brainId)}`,
    );
  }

  async getBrainStats(brainId: string): Promise<StatsResponse> {
    return this._get<StatsResponse>(
      `/brain/${encodeURIComponent(brainId)}/stats`,
    );
  }

  async createBrain(request: CreateBrainRequest): Promise<BrainResponse> {
    return this._post<BrainResponse>("/brain/create", request);
  }

  async deleteBrain(brainId: string): Promise<DeleteBrainResponse> {
    return this._delete<DeleteBrainResponse>(
      `/brain/${encodeURIComponent(brainId)}`,
    );
  }

  async exportBrain(brainId: string): Promise<BrainSnapshot> {
    return this._get<BrainSnapshot>(
      `/brain/${encodeURIComponent(brainId)}/export`,
    );
  }

  async importBrain(
    brainId: string,
    snapshot: BrainSnapshot,
  ): Promise<BrainResponse> {
    return this._post<BrainResponse>(
      `/brain/${encodeURIComponent(brainId)}/import`,
      snapshot,
    );
  }

  // ============ Sync stats ============

  async getSyncStats(): Promise<SyncStatsResponse> {
    return this._get<SyncStatsResponse>("/sync/stats");
  }

  // ============ Internal HTTP methods ============

  private async _get<T>(path: string, brainId?: string): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (brainId) {
      headers["X-Brain-ID"] = brainId;
    }

    const resp = await fetch(`${this._baseUrl}${path}`, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(10_000),
    });

    return this._handleResponse<T>(resp);
  }

  private async _post<T>(
    path: string,
    body: unknown,
    brainId?: string,
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (brainId) {
      headers["X-Brain-ID"] = brainId;
    }

    const resp = await fetch(`${this._baseUrl}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });

    return this._handleResponse<T>(resp);
  }

  private async _delete<T>(path: string, brainId?: string): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (brainId) {
      headers["X-Brain-ID"] = brainId;
    }

    const resp = await fetch(`${this._baseUrl}${path}`, {
      method: "DELETE",
      headers,
      signal: AbortSignal.timeout(10_000),
    });

    return this._handleResponse<T>(resp);
  }

  private async _handleResponse<T>(resp: Response): Promise<T> {
    if (!resp.ok) {
      let detail: string | undefined;
      try {
        const errBody = (await resp.json()) as ErrorResponse;
        detail = errBody.detail ?? errBody.error;
      } catch {
        // Response body wasn't JSON
      }

      throw new ApiError(
        `HTTP ${resp.status}: ${detail ?? resp.statusText}`,
        resp.status,
        detail,
      );
    }

    return (await resp.json()) as T;
  }
}
