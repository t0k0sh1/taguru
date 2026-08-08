/**
 * HtmlConnector — the URL-fetch half of the `.html` connector (ADR 0007
 * §6.1/§7/§8, issue #349), exercised against a real (loopback) HTTP
 * server via `tests/httpd.ts`. Local-file behavior is
 * `html-connector.test.ts`'s own file. TypeScript parity: issue #415.
 *
 * Every fetch here targets `127.0.0.1` — a private/loopback address the
 * connector refuses by default (its own SSRF mitigation, documented in
 * `html.ts`'s module header) — so every `HtmlConnector` below except
 * `a private destination is refused by default` itself passes
 * `allowPrivateNetworks: true` to reach the test server at all.
 */

import { expect, test } from "vitest";

import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import { type Route, type RouteServer, serve } from "../httpd.js";

const PAGE = Buffer.from(
  `<html><head><title>Page</title></head>
<body><main><h1 id="h">Heading</h1><p>Body text.</p></main></body></html>`,
);

async function withServer(
  routes: Record<string, Route>,
  fn: (server: RouteServer) => Promise<void>,
): Promise<void> {
  const server = await serve(routes);
  try {
    await fn(server);
  } finally {
    await server.close();
  }
}

test("a 200 html response is read normally", async () => {
  await withServer({ "/page": { body: PAGE } }, async (server) => {
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page`,
    );

    expect(document.diagnostics).toEqual([]);
    expect(document.text).toBe("Heading\n\nBody text.");
    expect(document.source).toBe(`${server.baseUrl}/page`);
    expect(document.metadata.title).toBe("Page");
  });
});

test("a private destination is refused by default", async () => {
  await withServer({ "/page": { body: PAGE } }, async (server) => {
    const document = await new HtmlConnector().read(`${server.baseUrl}/page`);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });
});

test("404 is reported unreadable", async () => {
  await withServer({ "/page": { status: 404, body: Buffer.from("not found") } }, async (server) => {
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page`,
    );

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
    expect(document.diagnostics[0]!.message).toContain("404");
  });
});

test("server error is reported unreadable", async () => {
  await withServer({ "/page": { status: 500, body: Buffer.from("boom") } }, async (server) => {
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page`,
    );

    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });
});

test("non-html content type is reported unsupported_format", async () => {
  await withServer(
    { "/page": { contentType: "application/json", body: Buffer.from("{}") } },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/page`,
      );

      expect(document.text).toBe("");
      expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
    },
  );
});

test("missing content type header is still parsed", async () => {
  await withServer({ "/page": { contentType: null, body: PAGE } }, async (server) => {
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page`,
    );

    expect(document.diagnostics).toEqual([]);
    expect(document.text).toBe("Heading\n\nBody text.");
  });
});

test("xhtml content type is accepted", async () => {
  await withServer(
    { "/page": { contentType: "application/xhtml+xml", body: PAGE } },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/page`,
      );

      expect(document.diagnostics).toEqual([]);
    },
  );
});

test("connection refused is reported unreadable", async () => {
  // An ephemeral loopback port nothing is listening on.
  const document = await new HtmlConnector({ timeout: 2.0, allowPrivateNetworks: true }).read(
    "http://127.0.0.1:1/page",
  );

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
});

test("a slow response past the timeout is reported unreadable", async () => {
  await withServer({ "/page": { body: PAGE, delay: 0.5 } }, async (server) => {
    const document = await new HtmlConnector({ timeout: 0.05, allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page`,
    );

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });
});

test("a response trickling past the total time budget is reported unreadable", async () => {
  // Each inter-chunk gap (0.02s) lands well inside `timeout` (5s) — no
  // phase ever goes idle for long enough to trip it — but the whole
  // trickled transfer still overruns `maxTotalSeconds`, the failure mode a
  // per-phase timeout alone can never catch.
  const body = Buffer.concat([PAGE, Buffer.from("<!-- padding -->".repeat(20))]);
  await withServer({ "/page": { body, chunkDelay: 0.02 } }, async (server) => {
    const document = await new HtmlConnector({
      timeout: 5.0,
      maxTotalSeconds: 0.1,
      allowPrivateNetworks: true,
    }).read(`${server.baseUrl}/page`);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });
});

test("a redirected failure names the final url, not the start", async () => {
  // The module contract — "the source id is the final URL" — holds on
  // failure paths too: two start URLs redirecting to one failing endpoint
  // must fold to one source, and the diagnostic must name the URL that
  // actually answered.
  await withServer(
    {
      "/start": { location: "/final-404" },
      "/final-404": { status: 404, body: Buffer.from("gone") },
    },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/start`,
      );
      expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
      expect(document.source).toBe(`${server.baseUrl}/final-404`);
    },
  );
});

test("a redirected wrong content type names the final url", async () => {
  await withServer(
    {
      "/start": { location: "/final-json" },
      "/final-json": { contentType: "application/json", body: Buffer.from("{}") },
    },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/start`,
      );
      expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
      expect(document.source).toBe(`${server.baseUrl}/final-json`);
    },
  );
});

test("the source id is the final url after a redirect", async () => {
  await withServer(
    {
      "/start": { location: "/final" },
      "/final": { body: PAGE },
    },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/start`,
      );

      expect(document.source).toBe(`${server.baseUrl}/final`);
      expect(document.diagnostics).toEqual([]);
    },
  );
});

test("userinfo and signed query params are stripped from the source id", async () => {
  await withServer({ "/page": { body: PAGE } }, async (server) => {
    const host = server.baseUrl.replace(/^http:\/\//, "");
    const reference = `http://user:pass@${host}/page?token=SECRET&keep=1`;
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(reference);

    // Also confirms the request actually reached the /page route (i.e.
    // the query string didn't defeat the test server's own route lookup)
    // rather than the pre-fetch canonicalized id merely happening to
    // match on a failed request.
    expect(document.diagnostics).toEqual([]);
    expect(document.source).toBe(`${server.baseUrl}/page?keep=1`);
    expect(document.source).not.toContain("pass");
    expect(document.source).not.toContain("SECRET");
  });
});

test("the page's own fragment is stripped from the source id", async () => {
  await withServer({ "/page": { body: PAGE } }, async (server) => {
    const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
      `${server.baseUrl}/page#section-two`,
    );

    expect(document.source).toBe(`${server.baseUrl}/page`);
  });
});

test("a response over the byte cap is refused mid-stream", async () => {
  const body = Buffer.concat([PAGE, Buffer.from("<!-- padding -->".repeat(10_000))]);
  await withServer({ "/page": { body } }, async (server) => {
    const document = await new HtmlConnector({
      maxFileBytes: PAGE.length,
      allowPrivateNetworks: true,
    }).read(`${server.baseUrl}/page`);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
  });
});

test("canonical link is resolved relative to the final fetched url", async () => {
  const body = Buffer.from(
    `<html><head><link rel="canonical" href="/canon"></head>
    <body><main><p>Text.</p></main></body></html>`,
  );
  await withServer(
    {
      "/start": { location: "/final" },
      "/final": { body },
    },
    async (server) => {
      const document = await new HtmlConnector({ allowPrivateNetworks: true }).read(
        `${server.baseUrl}/start`,
      );

      expect(document.metadata.canonicalUrl).toBe(`${server.baseUrl}/canon`);
    },
  );
});

test("supports true but unsupported scheme reads never touch the network", async () => {
  const document = await new HtmlConnector().read("ftp://127.0.0.1:1/doc.html");

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
});

test("oversized url source id is refused before any fetch", async () => {
  const reference = "http://127.0.0.1:1/" + "x".repeat(1200);
  const document = await new HtmlConnector().read(reference);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["source_id_too_long"]);
});
