const BASE_URL = ""

interface FetchOptions extends RequestInit {
  brain?: string
}

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function request<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { brain, ...fetchOptions } = options

  const headers = new Headers(fetchOptions.headers)
  if (brain) {
    headers.set("X-Brain-ID", brain)
  }
  if (fetchOptions.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...fetchOptions,
    headers,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new ApiError(response.status, text)
  }

  return response.json() as Promise<T>
}

export const api = {
  get: <T>(path: string, options?: FetchOptions) =>
    request<T>(path, { ...options, method: "GET" }),

  post: <T>(path: string, body?: unknown, options?: FetchOptions) =>
    request<T>(path, {
      ...options,
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),

  put: <T>(path: string, body?: unknown, options?: FetchOptions) =>
    request<T>(path, {
      ...options,
      method: "PUT",
      body: body ? JSON.stringify(body) : undefined,
    }),

  delete: <T>(path: string, options?: FetchOptions) =>
    request<T>(path, { ...options, method: "DELETE" }),
}

export { ApiError }
