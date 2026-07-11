/** A hand-rolled fetch stub — no msw, zero extra dependencies. */

import { Taguru } from "../../src/client.js";

export interface StubRequest {
  method: string;
  url: string;
  path: string;
  headers: Record<string, string>;
  body?: string;
}

export type StubResult =
  | { status: number; body: string; headers?: Record<string, string> }
  | Error;

export type StubHandler = (request: StubRequest) => StubResult;

export function stubFetch(handler: StubHandler): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries((init?.headers ?? {}) as Record<string, string>)) {
      headers[key.toLowerCase()] = value;
    }
    let body: string | undefined;
    if (typeof init?.body === "string") {
      body = init.body;
    } else if (init?.body instanceof Uint8Array) {
      body = new TextDecoder().decode(init.body);
    }
    const result = handler({ method: init?.method ?? "GET", url, path: new URL(url).pathname, headers, body });
    if (result instanceof Error) {
      throw result;
    }
    return new Response(result.body, { status: result.status, headers: result.headers });
  };
}

export function okBody(result: unknown): { status: number; body: string } {
  return { status: 200, body: JSON.stringify({ result, status: "ok", time: 0.001 }) };
}

export function errBody(
  status: number,
  message: string,
  headers?: Record<string, string>,
  code?: string,
): { status: number; body: string; headers?: Record<string, string> } {
  const payload: Record<string, unknown> = { status: "error", error: message, time: 0.001 };
  if (code !== undefined) {
    payload["code"] = code;
  }
  return { status, body: JSON.stringify(payload), headers };
}

export function stubClient(handler: StubHandler, options: { retries?: number; api_key?: string } = {}): Taguru {
  return new Taguru({
    base_url: "http://test",
    api_key: options.api_key ?? "",
    retries: options.retries,
    fetch: stubFetch(handler),
  });
}
