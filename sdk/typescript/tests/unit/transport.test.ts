import { describe, expect, it } from "vitest";

import { Taguru } from "../../src/client.js";
import { citationKey } from "../../src/models.js";
import { TaguruError } from "../../src/errors.js";
import { chunkAssociations, isPreConnectFailure } from "../../src/transport.js";
import { errBody, okBody, stubClient, type StubRequest } from "./stub.js";

const DIRECTORY_ROW = {
  name: "sake",
  description: "酒蔵の知識",
  pinned: false,
  loaded: true,
  dice_floor: null,
  semantic_floor: 0.35,
  stats: {
    associations: 6,
    concepts: 5,
    labels: 4,
    sources: 1,
    footprint_bytes: 4096,
    top_concepts: [{ label: "青嶺酒造", count: 4 }],
    label_sample: ["代表銘柄"],
  },
  usage: { reads: 1, empty_reads: 0, writes: 2, last_read_epoch: 100, last_write_epoch: 90 },
};

describe("envelope and raw-body handling", () => {
  it("unwraps the envelope", async () => {
    const client = stubClient(() => okBody(DIRECTORY_ROW));
    const entry = await client.contexts.get("sake");
    expect(entry.stats.top_concepts[0]!.label).toBe("青嶺酒造");
    expect(entry.dice_floor).toBeNull();
  });

  it("tolerates unknown fields (additive evolution)", async () => {
    const row = { ...DIRECTORY_ROW, brand_new_field: { nested: true } };
    const client = stubClient(() => okBody(row));
    const entry = await client.contexts.get("sake");
    expect(entry.name).toBe("sake");
  });

  it("describe null result is null, not an error", async () => {
    const client = stubClient(() => okBody(null));
    await expect(client.context("sake").describe("unknown")).resolves.toBeNull();
  });

  it("throws a protocol error on a non-envelope 2xx", async () => {
    const client = stubClient(() => ({ status: 200, body: JSON.stringify({ weird: true }) }));
    await expect(client.context("sake").recall("cue")).rejects.toThrow(/envelope/);
  });

  it("raw text routes bypass the envelope", async () => {
    const client = stubClient((req) => {
      if (req.path === "/health") return { status: 200, body: "ok" };
      if (req.path === "/metrics") return { status: 200, body: "taguru_requests_total 1\n" };
      if (req.path === "/protocol") return { status: 200, body: "# Taguru client protocol\n" };
      throw new Error(req.path);
    });
    await client.health();
    expect(await client.metrics()).toContain("taguru_requests_total");
    expect(await client.protocol()).toMatch(/^# Taguru client protocol/);
  });

  it("export returns raw NDJSON", async () => {
    const ndjson = '{"taguru_batch":1}\n{"passage":"text"}\n';
    const client = stubClient(() => ({ status: 200, body: ndjson }));
    await expect(client.context("sake").export()).resolves.toBe(ndjson);
  });

  it("normalizes import outcomes to batches, defaulting groups to empty", async () => {
    const outcome = { context: "sake", source: "a", created: true };
    const single = stubClient(() => okBody(outcome));
    const result = await single.importBatches("{}");
    expect(result.batches).toHaveLength(1);
    expect(result.groups).toEqual([]);

    const multi = stubClient(() => okBody({ batches: [outcome, outcome] }));
    expect((await multi.importBatches("{}")).batches).toHaveLength(2);
  });

  it("carries the server's group restore outcomes through import", async () => {
    const outcome = { context: "sake", source: "a", created: true };
    const client = stubClient(() =>
      okBody({
        batches: [outcome],
        groups: [{ name: "brewers", outcome: "created", contexts: 2, groups: 0 }],
      }),
    );
    const result = await client.importBatches("{}");
    expect(result.batches).toHaveLength(1);
    expect(result.groups).toEqual([{ name: "brewers", outcome: "created", contexts: 2, groups: 0 }]);
  });

  it("sends the bearer header only when a key is configured", async () => {
    const seen: Array<string | undefined> = [];
    const record = (req: StubRequest) => {
      seen.push(req.headers["authorization"]);
      return okBody({ total: 0, matches: [] });
    };
    await stubClient(record, { api_key: "secret" }).context("sake").recall("cue");
    await stubClient(record).context("sake").recall("cue");
    expect(seen).toEqual(["Bearer secret", undefined]);
  });

  it("percent-encodes context names in paths", async () => {
    const paths: string[] = [];
    const client = stubClient((req) => {
      paths.push(req.path);
      return okBody({ total: 0, matches: [] });
    });
    await client.context("日本 酒/テスト").recall("cue");
    expect(paths).toEqual([
      "/contexts/%E6%97%A5%E6%9C%AC%20%E9%85%92%2F%E3%83%86%E3%82%B9%E3%83%88/recall",
    ]);
  });

  it("query sends one-or-many and drops absent fields", async () => {
    const bodies: Array<string | undefined> = [];
    const client = stubClient((req) => {
      bodies.push(req.body);
      return okBody({ total: 0, matches: [] });
    });
    await client.context("sake").query({ label: ["住所", "職歴"], subject: "高瀬" });
    expect(bodies[0]).toBe('{"subject":"高瀬","label":["住所","職歴"]}');
  });

  it("#60: after cursor rides the request body verbatim", async () => {
    // The client mints no cursor of its own — it only relays the last
    // page's last row back to the server, whatever shape it has.
    const bodies: Array<string | undefined> = [];
    const client = stubClient((req) => {
      bodies.push(req.body);
      return okBody({ total: 0, matches: [] });
    });

    await client
      .context("sake")
      .recall("cue", { after: { weight: 0.5, subject: "a", label: "b", object: "c" } });
    expect(bodies[bodies.length - 1]).toBe(
      '{"cue":"cue","after":{"weight":0.5,"subject":"a","label":"b","object":"c"}}',
    );

    await client
      .context("sake")
      .explore("a", { after: { distance: 2, subject: "a", label: "b", object: "c" } });
    expect(bodies[bodies.length - 1]).toBe(
      '{"origins":["a"],"after":{"distance":2,"subject":"a","label":"b","object":"c"}}',
    );

    await client.recall("cue", {
      contexts: ["sake"],
      after: { weight: 0.5, context: "sake", subject: "a", label: "b", object: "c" },
    });
    expect(bodies[bodies.length - 1]).toBe(
      '{"contexts":["sake"],"cue":"cue","after":' +
        '{"weight":0.5,"context":"sake","subject":"a","label":"b","object":"c"}}',
    );
  });

  it("keeps message, body, and time on errors", async () => {
    const client = stubClient(() => errBody(404, "context 'x' does not exist"), { retries: 0 });
    const error = await client.contexts.get("x").catch((caught: unknown) => caught);
    expect((error as TaguruError).body).toEqual({
      status: "error",
      error: "context 'x' does not exist",
      time: 0.001,
    });
  });
});

describe("pagination iterators", () => {
  it("walks directory pages with the keyset cursor", async () => {
    const cursors: Array<string | null> = [];
    const rowFor = (name: string) => ({ ...DIRECTORY_ROW, name });
    const client = stubClient((req) => {
      const after = new URL(req.url).searchParams.get("after");
      cursors.push(after);
      if (after === null) return okBody({ total: 4, contexts: [rowFor("a"), rowFor("b")] });
      // A short page (fewer rows than the limit) is not the last one — the
      // walk keeps paging, or a server-clamped limit would drop later rows.
      if (after === "b") return okBody({ total: 4, contexts: [rowFor("c")] });
      if (after === "c") return okBody({ total: 4, contexts: [rowFor("d")] });
      if (after === "d") return okBody({ total: 4, contexts: [] });
      throw new Error(String(after));
    });
    const names: string[] = [];
    for await (const entry of client.contexts.iter({ limit: 2 })) {
      names.push(entry.name);
    }
    expect(names).toEqual(["a", "b", "c", "d"]);
    expect(cursors).toEqual([null, "b", "c", "d"]);
  });

  it("flattens both alias namespaces and advances the two-namespace cursor", async () => {
    const cursors: Array<string | null> = [];
    const client = stubClient((req) => {
      const after = new URL(req.url).searchParams.get("after");
      cursors.push(after);
      if (after === null) {
        return okBody({ total: 3, concepts: { Aomine: "青嶺酒造", 青嶺: "青嶺酒造" }, labels: {} });
      }
      if (after === "concept:青嶺") {
        return okBody({ total: 3, concepts: {}, labels: { brand: "代表銘柄" } });
      }
      // The short second page is not terminal; the walk probes once more and
      // stops only on the empty page.
      if (after === "label:brand") {
        return okBody({ total: 3, concepts: {}, labels: {} });
      }
      throw new Error(String(after));
    });
    const entries = [];
    for await (const entry of client.context("sake").iterAliases({ limit: 2 })) {
      entries.push(entry);
    }
    expect(entries).toEqual([
      { namespace: "concept", alias: "Aomine", canonical: "青嶺酒造" },
      { namespace: "concept", alias: "青嶺", canonical: "青嶺酒造" },
      { namespace: "label", alias: "brand", canonical: "代表銘柄" },
    ]);
    expect(cursors).toEqual([null, "concept:青嶺", "label:brand"]);
  });

  it("orders integer-like alias keys lexicographically, matching the server's BTreeMap cursor", async () => {
    const cursors: Array<string | null> = [];
    const client = stubClient((req) => {
      const after = new URL(req.url).searchParams.get("after");
      cursors.push(after);
      // The server's BTreeMap<String, String> keeps keys in byte order,
      // not numeric order: "10" < "2" < "9". A client that lets
      // `Object.entries` (which numerically reorders integer-like keys)
      // decide the cursor would advance past "10" instead of "2" here,
      // then skip straight to "9" and never revisit "2" — or, on a page
      // boundary drawn differently, re-fetch and re-yield an alias
      // already produced.
      if (after === null) {
        return okBody({ total: 3, concepts: { "10": "ten", "2": "two" }, labels: {} });
      }
      if (after === "concept:2") {
        return okBody({ total: 3, concepts: { "9": "nine" }, labels: {} });
      }
      if (after === "concept:9") {
        return okBody({ total: 3, concepts: {}, labels: {} });
      }
      throw new Error(String(after));
    });
    const entries = [];
    for await (const entry of client.context("sake").iterAliases({ limit: 2 })) {
      entries.push(entry);
    }
    expect(entries).toEqual([
      { namespace: "concept", alias: "10", canonical: "ten" },
      { namespace: "concept", alias: "2", canonical: "two" },
      { namespace: "concept", alias: "9", canonical: "nine" },
    ]);
    expect(cursors).toEqual([null, "concept:2", "concept:9"]);
  });
});

describe("batching", () => {
  const op = (i: number) => ({ subject: `s${i}`, label: "l", object: "o", weight: 1.0 });

  it("chunks by count", () => {
    const chunks = [...chunkAssociations([op(0), op(1), op(2), op(3), op(4)], 2, 1e9)];
    expect(chunks.map((c) => c.length)).toEqual([2, 2, 1]);
  });

  it("chunks by serialized byte budget", () => {
    const one = Buffer.byteLength(JSON.stringify(op(0)), "utf-8");
    const budget = 2 + one + 1 + one;
    const chunks = [...chunkAssociations([op(0), op(1), op(2), op(3)], 10_000, budget)];
    expect(chunks.map((c) => c.length)).toEqual([2, 2]);
    for (const chunk of chunks) {
      expect(Buffer.byteLength(JSON.stringify(chunk), "utf-8")).toBeLessThanOrEqual(budget);
    }
  });

  it("addAssociationsBatched sums applied counts across chunks", async () => {
    const batchSizes: number[] = [];
    const client = stubClient((req) => {
      const ops = JSON.parse(req.body ?? "[]") as unknown[];
      batchSizes.push(ops.length);
      return okBody(ops.length);
    });
    const result = await client
      .context("sake")
      .addAssociationsBatched([op(0), op(1), op(2), op(3), op(4)], { chunk_size: 2 });
    expect(result).toEqual({ applied: 5, chunks: 3 });
    expect(batchSizes).toEqual([2, 2, 1]);
  });
});

describe("retrieve loop", () => {
  const ASSOCIATION = {
    subject: "青嶺酒造",
    label: "杜氏",
    object: "高瀬",
    weight: 2.0,
    count: 2,
    attributions: [
      { source: "docs/aomine.md", weight: 2.0, count: 2, paragraph: 1, section: null },
      { source: "unstored.md", weight: 1.0, count: 1, paragraph: 0, section: null },
    ],
  };

  function routed(calls: string[]) {
    return (req: StubRequest) => {
      calls.push(req.path);
      const body = req.body ? (JSON.parse(req.body) as Record<string, unknown>) : {};
      if (req.path.endsWith("/resolve")) {
        return okBody(
          body["cue"] === "青嶺"
            ? [{ name: "青嶺酒造", score: 1.0, tier: "lexical", kind: "exact" }]
            : [],
        );
      }
      if (req.path.endsWith("/describe")) {
        return okBody({ concept: "青嶺酒造", as_subject: [], as_object: [] });
      }
      if (req.path.endsWith("/activate")) {
        return okBody({
          total: 1,
          matches: [{ strength: 0.9, path: ["青嶺酒造"], association: ASSOCIATION }],
        });
      }
      if (req.path.endsWith("/citations")) {
        if (body["source"] === "unstored.md") {
          return errBody(404, "no stored passage");
        }
        return okBody({ text: "杜氏は高瀬。", source: body["source"], section: "人物" });
      }
      if (req.path.endsWith("/sources/search")) {
        return okBody([
          {
            source: "docs/aomine.md",
            paragraph: 1,
            score: 3.2,
            text: "杜氏は高瀬。",
            lanes: { bm25: { rank: 0, score: 3.2 } },
          },
        ]);
      }
      throw new Error(req.path);
    };
  }

  it("runs the documented loop and skips unstored citations", async () => {
    const calls: string[] = [];
    const client = stubClient(routed(calls), { retries: 0 });
    const result = await client.context("sake").retrieve("青嶺");
    expect(result.resolved["青嶺"]![0]!.name).toBe("青嶺酒造");
    expect(result.outline["青嶺酒造"]).not.toBeNull();
    expect(result.associations).toHaveLength(1);
    expect(result.citations.get(citationKey("docs/aomine.md", 1))?.section).toBe("人物");
    expect(result.citations.has(citationKey("unstored.md", 0))).toBe(false);
    expect(result.passage_hits).toEqual([]);
    expect(calls).toEqual([
      "/contexts/sake/resolve",
      "/contexts/sake/describe",
      "/contexts/sake/activate",
      "/contexts/sake/citations",
      "/contexts/sake/citations",
    ]);
  });

  it("fires the text fallback only when the graph is empty", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    const ctx = client.context("sake");

    const answered = await ctx.retrieve("青嶺", { text_fallback_query: "杜氏は高瀬である" });
    expect(answered.passage_hits).toEqual([]);

    const empty = await ctx.retrieve("無関係", { text_fallback_query: "杜氏は高瀬である" });
    expect(empty.associations).toEqual([]);
    expect(empty.passage_hits).toHaveLength(1);
    expect(empty.passage_hits[0]!.lanes.bm25).toBeDefined();
    expect(empty.passage_hits[0]!.lanes.vector).toBeUndefined();

    const always = await ctx.retrieve("青嶺", {
      text_fallback_query: "杜氏は高瀬である",
      text_fallback_only_if_empty: false,
    });
    expect(always.passage_hits).toHaveLength(1);
  });
});

describe("isPreConnectFailure", () => {
  it("recognizes undici's connect-phase timeout", () => {
    const error = new TypeError("fetch failed", {
      cause: Object.assign(new Error("connect timeout"), { code: "UND_ERR_CONNECT_TIMEOUT" }),
    });
    expect(isPreConnectFailure(error)).toBe(true);
  });

  it("recognizes refused connections and unresolvable hosts", () => {
    const refused = new TypeError("fetch failed", {
      cause: Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" }),
    });
    expect(isPreConnectFailure(refused)).toBe(true);

    const notFound = new TypeError("fetch failed", {
      cause: Object.assign(new Error("getaddrinfo ENOTFOUND"), { code: "ENOTFOUND" }),
    });
    expect(isPreConnectFailure(notFound)).toBe(true);
  });

  it("does not treat AbortSignal.timeout()'s TimeoutError as pre-connect", () => {
    // No `code` at all, and it can fire after the request already reached
    // the server — must stay ambiguous, unlike UND_ERR_CONNECT_TIMEOUT.
    const timeout = new DOMException("This operation was aborted", "TimeoutError");
    expect(isPreConnectFailure(timeout)).toBe(false);
  });

  it("stays false for an unrelated or mid-flight failure", () => {
    const midFlight = new TypeError("fetch failed", {
      cause: Object.assign(new Error("socket hang up"), { code: "UND_ERR_SOCKET" }),
    });
    expect(isPreConnectFailure(midFlight)).toBe(false);
    expect(isPreConnectFailure(new Error("boom"))).toBe(false);
  });
});

describe("exportStream", () => {
  it("aborts once the client's timeout elapses", async () => {
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      timeout: 0.05,
      fetch: (_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("This operation was aborted", "TimeoutError"));
          });
        }),
    });
    const stream = client.context("sake").exportStream();
    await expect(stream.next()).rejects.toThrow(/aborted/i);
  });
});
