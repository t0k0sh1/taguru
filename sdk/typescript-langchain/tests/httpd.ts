/**
 * Minimal HTTP server harness for `HtmlConnector`'s URL-fetch tests (issue
 * #349) — a `node:http` server serving canned, in-memory responses,
 * synthesized at test time rather than depending on any external service
 * or recorded fixture. The mechanical mirror of the Python
 * `tests/_httpd.py` helper (issue #415).
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

/**
 * One canned response. `location` set makes this a 302 redirect
 * (`status`/`contentType`/`body` are then ignored); otherwise `status` is
 * served with `body`, and `contentType` becomes the `Content-Type` header
 * (`null` sends no `Content-Type` header at all; the default is
 * `"text/html; charset=utf-8"`). `delay` (seconds) sleeps before
 * responding at all — enough to exercise a caller's per-phase `timeout`.
 * `chunkDelay` instead trickles `body` out in small pieces, sleeping
 * between each — every individual gap stays under a per-phase `timeout`,
 * so this is what exercises a caller's separate total-time budget
 * instead.
 */
export interface Route {
  status?: number;
  contentType?: string | null;
  body?: Buffer | Uint8Array;
  delay?: number;
  chunkDelay?: number;
  location?: string;
  headers?: Record<string, string>;
}

export interface RouteServer {
  readonly baseUrl: string;
  close(): Promise<void>;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  routes: Readonly<Record<string, Route>>,
): Promise<void> {
  res.on("error", () => {});
  req.on("error", () => {});

  // `req.url` is the full request target, query string included — route
  // registration is by path only, so a request carrying a query string
  // (the credential/token-stripping tests deliberately send one) must not
  // be matched against it too.
  const pathname = new URL(req.url ?? "/", "http://localhost").pathname;
  const route = routes[pathname];
  if (route === undefined) {
    res.writeHead(404);
    res.end();
    return;
  }

  if (route.delay) {
    await sleep(route.delay * 1000);
  }
  if (res.destroyed) {
    return;
  }

  if (route.location !== undefined) {
    res.writeHead(302, { Location: route.location });
    res.end();
    return;
  }

  const status = route.status ?? 200;
  const contentType = route.contentType === undefined ? "text/html; charset=utf-8" : route.contentType;
  const body = route.body ?? Buffer.alloc(0);
  const headers: Record<string, string> = { ...route.headers };
  if (contentType !== null) {
    headers["Content-Type"] = contentType;
  }

  if (route.chunkDelay) {
    res.writeHead(status, headers);
    const chunkSize = 16;
    for (let offset = 0; offset < body.length; offset += chunkSize) {
      if (res.destroyed) {
        return;
      }
      try {
        res.write(body.subarray(offset, offset + chunkSize));
      } catch {
        return;
      }
      await sleep(route.chunkDelay * 1000);
    }
    if (!res.destroyed) {
      res.end();
    }
    return;
  }

  headers["Content-Length"] = String(body.length);
  res.writeHead(status, headers);
  res.end(body);
}

function createRouteServer(routes: Readonly<Record<string, Route>>): Server {
  const server = createServer((req, res) => {
    void handleRequest(req, res, routes);
  });
  server.on("clientError", (_error, socket) => {
    if (!socket.destroyed) {
      socket.destroy();
    }
  });
  return server;
}

/**
 * Serves `routes` (path -> `Route`) on an ephemeral `127.0.0.1` port until
 * `close()` is called. Usage::
 *
 *     const server = await serve({ "/page": { body: Buffer.from("<html>...</html>") } });
 *     try {
 *       const document = await new HtmlConnector().read(`${server.baseUrl}/page`);
 *       ...
 *     } finally {
 *       await server.close();
 *     }
 */
export async function serve(routes: Readonly<Record<string, Route>>): Promise<RouteServer> {
  const server = createRouteServer(routes);
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("failed to determine the test server's address");
  }
  const baseUrl = `http://127.0.0.1:${address.port}`;
  return {
    baseUrl,
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
        // A client that abandons a response mid-stream (the timeout/
        // oversized-body tests deliberately do this) can leave a
        // keep-alive socket open; force-close everything so `close()`
        // doesn't hang waiting for it to go idle.
        server.closeAllConnections();
      }),
  };
}
