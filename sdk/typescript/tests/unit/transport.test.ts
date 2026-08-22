import { describe, expect, it } from "vitest";

import { Taguru } from "../../src/client.js";
import { citationKey } from "../../src/models.js";
import { TaguruError } from "../../src/errors.js";
import {
  chunkAssociations,
  DEFAULT_BASE_URL,
  describeError,
  dropUndefined,
  ENV_TOKEN,
  ENV_URL,
  errorFromBody,
  isPreConnectFailure,
  normalizeHeaders,
  normalizeImportOutcomes,
  sleep,
  sortedEntries,
  unwrapEnvelopeFull,
} from "../../src/transport.js";
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

  it("query sends subject_types and object_types", async () => {
    // ADR 0009 §12: the type filter rides the request body beside the
    // position pins, dropped like every other absent field when omitted.
    const bodies: Array<string | undefined> = [];
    const client = stubClient((req) => {
      bodies.push(req.body);
      return okBody({ total: 0, matches: [] });
    });
    await client
      .context("sake")
      .query({ subject: "青嶺酒造", subject_types: ["Brewery", "Organization"] });
    expect(bodies[0]).toBe(
      '{"subject":"青嶺酒造","subject_types":["Brewery","Organization"]}',
    );

    await client.context("sake").query({ label: "杜氏", object_types: "Person" });
    expect(bodies[1]).toBe('{"label":"杜氏","object_types":"Person"}');
  });

  it("describe and resolve decode types", async () => {
    // ADR 0009 §12's read-side types decode into the existing interfaces
    // without any special casing.
    const described = stubClient(() =>
      okBody({
        concept: "青嶺酒造",
        as_subject: [],
        as_object: [],
        types: ["Brewery", "Organization"],
      }),
    );
    const outline = await described.context("sake").describe("青嶺酒造");
    expect(outline?.types).toEqual(["Brewery", "Organization"]);

    const resolved = stubClient(() =>
      okBody([{ name: "青嶺酒造", score: 1.0, tier: "lexical", types: ["Brewery"] }]),
    );
    const candidates = await resolved.context("sake").resolve("青嶺");
    expect(candidates[0]?.types).toEqual(["Brewery"]);
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

describe("constants", () => {
  it("pins the default base URL and env var names verbatim", () => {
    expect(DEFAULT_BASE_URL).toBe("http://127.0.0.1:8248");
    expect(ENV_URL).toBe("TAGURU_URL");
    expect(ENV_TOKEN).toBe("TAGURU_API_TOKEN");
  });
});

describe("sortedEntries", () => {
  it("sorts by UTF-8 byte order, diverging from UTF-16 code-unit order", () => {
    // U+1F600 (an astral emoji, UTF-16 surrogate pair starting 0xD83D)
    // sorts BEFORE "" in UTF-16 code-unit order (0xD83D < 0xE000)
    // but AFTER it in UTF-8 byte order (the emoji's lead byte 0xF0 beats
    // ""'s 3-byte encoding's lead byte 0xEE). A plain `<` string
    // comparison, or a non-UTF-8 encoding, would sort these the other way.
    const entries = sortedEntries({ "": "bmp", "\u{1F600}": "astral" });
    expect(entries).toEqual([
      ["", "bmp"],
      ["\u{1F600}", "astral"],
    ]);
  });
});

describe("errorFromBody", () => {
  it("falls back to HTTP {status} for a whitespace-only non-JSON body (trims first)", () => {
    const error = errorFromBody(500, null, "   ");
    expect(error.message).toBe("HTTP 500");
    expect(error.body).toBe("   ");
  });

  it("falls back to HTTP {status} for an empty non-JSON body", () => {
    const error = errorFromBody(404, null, "");
    expect(error.message).toBe("HTTP 404");
  });

  it("keeps a real non-JSON message verbatim (trim only strips surrounding whitespace)", () => {
    const error = errorFromBody(413, null, "  length limit exceeded  ");
    expect(error.message).toBe("length limit exceeded");
  });

  it("defaults the message to HTTP {status} for a JSON object with no .error field", () => {
    const error = errorFromBody(500, null, "{}");
    expect(error.message).toBe("HTTP 500");
    expect(error.code).toBeNull();
    expect(error.time).toBeNull();
  });

  it("does not crash on a bare JSON null body (typeof null is 'object')", () => {
    expect(() => errorFromBody(500, null, "null")).not.toThrow();
    expect(errorFromBody(500, null, "null").message).toBe("HTTP 500");
  });

  it("ignores a non-string .error, non-string .code, and non-number .time", () => {
    const error = errorFromBody(
      500,
      null,
      JSON.stringify({ error: 123, code: 42, time: "soon" }),
    );
    expect(error.message).toBe("HTTP 500");
    expect(error.code).toBeNull();
    expect(error.time).toBeNull();
  });

  it("takes a well-typed .error/.code/.time verbatim", () => {
    const error = errorFromBody(404, null, JSON.stringify({ error: "gone", code: "x", time: 1.5 }));
    expect(error.message).toBe("gone");
    expect(error.code).toBe("x");
    expect(error.time).toBe(1.5);
  });
});

describe("unwrapEnvelopeFull", () => {
  it("wraps a JSON parse failure in a TaguruError naming the status and raw body", () => {
    let caught: unknown;
    try {
      unwrapEnvelopeFull(200, "not json");
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(TaguruError);
    const error = caught as TaguruError;
    expect(error.message).toBe("expected a JSON envelope, got a non-JSON body");
    expect(error.status).toBe(200);
    expect(error.body).toBe("not json");
    expect(error.cause).toBeInstanceOf(Error);
  });

  it("rejects a bare JSON null body as not the envelope shape (does not crash)", () => {
    expect(() => unwrapEnvelopeFull(200, "null")).toThrow(TaguruError);
    expect(() => unwrapEnvelopeFull(200, "null")).toThrow(/envelope shape/);
  });

  it("rejects a JSON array body as not the envelope shape", () => {
    expect(() => unwrapEnvelopeFull(200, "[1,2,3]")).toThrow(/envelope shape/);
  });

  it("rejects a bare JSON primitive body cleanly, not with a raw TypeError", () => {
    // `"result" in parsed` throws on a non-object right-hand side — the
    // `typeof parsed === "object"` guard must actually short-circuit
    // before reaching it for a number/string/boolean body, or this
    // throws an uncaught TypeError instead of the clean envelope error.
    for (const body of ["42", '"just a string"', "true"]) {
      expect(() => unwrapEnvelopeFull(200, body)).toThrow(TaguruError);
      expect(() => unwrapEnvelopeFull(200, body)).toThrow(/envelope shape/);
    }
  });

  it("rejects a body with 'result' but status !== 'ok', naming the real status/body", () => {
    let caught: unknown;
    try {
      unwrapEnvelopeFull(200, JSON.stringify({ result: 5, status: "pending" }));
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(TaguruError);
    const error = caught as TaguruError;
    expect(error.message).toBe("response is not the taguru envelope shape");
    expect(error.status).toBe(200);
    expect(error.body).toEqual({ result: 5, status: "pending" });
  });
});

describe("isPreConnectFailure on an AggregateError", () => {
  it("is true when any inner error carries a pre-connect code", () => {
    const error = new AggregateError([
      Object.assign(new Error("socket hang up"), { code: "UND_ERR_SOCKET" }),
      Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" }),
    ]);
    expect(isPreConnectFailure(error)).toBe(true);
  });

  it("is false when no inner error carries a pre-connect code", () => {
    const error = new AggregateError([
      Object.assign(new Error("socket hang up"), { code: "UND_ERR_SOCKET" }),
      new Error("no code at all"),
    ]);
    expect(isPreConnectFailure(error)).toBe(false);
  });

  it("does not crash on a non-object entry inside an AggregateError", () => {
    const error = new AggregateError([
      null,
      Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" }),
    ]);
    expect(isPreConnectFailure(error)).toBe(true);
  });
});

describe("isPreConnectFailure code-recognition edge cases", () => {
  it("recognizes EAI_AGAIN (transient DNS failure)", () => {
    const error = Object.assign(new Error("getaddrinfo EAI_AGAIN"), { code: "EAI_AGAIN" });
    expect(isPreConnectFailure(error)).toBe(true);
  });

  it("does not crash when a .cause chain ends in null (not just undefined)", () => {
    const error = Object.assign(new Error("outer"), { cause: null });
    expect(isPreConnectFailure(error)).toBe(false);
  });

  it("does not crash when a .cause chain includes a bare primitive", () => {
    const error = Object.assign(new Error("outer"), { cause: 42 });
    expect(isPreConnectFailure(error)).toBe(false);
  });
});

describe("header normalization", () => {
  it("normalizeHeaders lower-cases every key", () => {
    expect(normalizeHeaders({ Authorization: "x", "X-Custom": "y" })).toEqual({
      authorization: "x",
      "x-custom": "y",
    });
  });

  // stubFetch's own toLowerCase()-and-assign can't reproduce this: only a
  // real Headers object comma-joins same-name keys that differ in case, so
  // these two tests build the actual Request the SDK would send.
  it("a conventionally-cased custom Authorization header does not collide with api_key's", async () => {
    let seenAuth: string | null = null;
    const client = new Taguru({
      base_url: "http://test",
      api_key: "secret",
      headers: { Authorization: "Bearer USER_SUPPLIED" },
      fetch: async (url, init) => {
        const req = new Request(url, init);
        seenAuth = req.headers.get("authorization");
        return new Response(JSON.stringify({ status: "ok", result: [], time: 0 }), { status: 200 });
      },
    });
    await client.contexts.list();
    expect(seenAuth).toBe("Bearer secret");
  });

  it("a conventionally-cased custom Content-Type header does not collide with the JSON body's", async () => {
    let seenType: string | null = null;
    const client = new Taguru({
      base_url: "http://test",
      headers: { "Content-Type": "text/plain" },
      fetch: async (url, init) => {
        const req = new Request(url, init);
        seenType = req.headers.get("content-type");
        return new Response(
          JSON.stringify({ status: "ok", result: { name: "x", description: "" }, time: 0 }),
          { status: 200 },
        );
      },
    });
    await client.contexts.create("x", { description: "" });
    expect(seenType).toBe("application/json");
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

  it("yields nothing for an empty ops list (no trailing empty chunk)", () => {
    expect([...chunkAssociations([], 10, 1e9)]).toEqual([]);
  });

  it("never flushes an empty leading chunk when the very first op alone exceeds the budget", () => {
    // maxChunkBytes smaller than even one op's serialized size: the
    // `chunk.length > 0` guard must still gate the flush on an empty
    // chunk — the first op is pushed unconditionally, never preceded by
    // a spurious empty yield.
    const chunks = [...chunkAssociations([op(0)], 1e9, 1)];
    expect(chunks).toEqual([[op(0)]]);
  });

  it("accounts for the separating comma exactly, one byte off the boundary", () => {
    // Budget is exactly ONE byte short of fitting two ops (2 + one + comma
    // + one); every op serializes to the same size here, so three ops
    // must land one per chunk — a `+`/`-` swap on the comma byte, or an
    // inverted `chunk.length > 0` ternary guarding it, would fit 2 into
    // the first chunk instead.
    const one = Buffer.byteLength(JSON.stringify(op(0)), "utf-8");
    const chunks = [...chunkAssociations([op(0), op(1), op(2)], 10_000, 2 * one + 2)];
    expect(chunks.map((c) => c.length)).toEqual([1, 1, 1]);
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
    expect(result).toEqual({ applied: 5, chunks: 3, issues: [], schema_violations: 0 });
    expect(batchSizes).toEqual([2, 2, 1]);
  });

  it("addAssociations surfaces the warn-mode envelope carrier", async () => {
    const issue = {
      path: "associations[0].object",
      kind: "range",
      expected: "one of [Brewery]",
      actual: "Prefecture",
    };
    const client = stubClient(() => ({
      status: 200,
      headers: {},
      body: JSON.stringify({
        result: 1,
        status: "ok",
        time: 0.001,
        issues: [issue],
        schema_violations: 3,
      }),
    }));
    const result = await client.context("sake").addAssociations([op(0)]);
    expect(result).toEqual({ applied: 1, issues: [issue], schema_violations: 3 });
  });

  it("addAssociationsBatched aggregates the warn-mode carrier in chunk order", async () => {
    // Each response is derived from the CHUNK THE REQUEST CARRIED, not
    // from handler call order — so a reordered transmission, a dropped
    // chunk, or a double-count would all be visible.
    const issueFor = (subject: string) => ({
      path: `associations[0].${subject}`,
      kind: "range",
      expected: "one of [Brewery]",
      actual: "Prefecture",
    });
    const violationsFor: Record<string, number> = { s0: 1, s1: 2 };
    const client = stubClient((req) => {
      const ops = JSON.parse(req.body ?? "[]") as { subject: string }[];
      const subject = ops[0]?.subject ?? "unknown";
      return {
        status: 200,
        headers: {},
        body: JSON.stringify({
          result: ops.length,
          status: "ok",
          time: 0.001,
          issues: [issueFor(subject)],
          schema_violations: violationsFor[subject] ?? 0,
        }),
      };
    });
    const result = await client
      .context("sake")
      .addAssociationsBatched([op(0), op(1)], { chunk_size: 1 });
    expect(result).toEqual({
      applied: 2,
      chunks: 2,
      issues: [issueFor("s0"), issueFor("s1")],
      schema_violations: 1 + 2,
    });
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
        return okBody({
          plan: {
            contexts: [
              {
                context: "sake",
                lanes: {
                  bm25: { ran: true },
                  vector: { ran: false, reason: "no embedding provider is configured" },
                },
              },
            ],
          },
          hits: [
            {
              source: "docs/aomine.md",
              paragraph: 1,
              score: 3.2,
              text: "杜氏は高瀬。",
              lanes: { bm25: { rank: 0, score: 3.2 } },
            },
          ],
        });
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
    // The fallback search's plan rides beside its hits (#151).
    expect(empty.search_plan?.contexts[0]?.lanes.vector.ran).toBe(false);

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

  it("terminates on a cyclic cause chain", () => {
    // A `cause` that loops back on itself must not hang the walk: the
    // visited set is what bounds it (without it, this test never returns).
    const loop = new Error("loop") as Error & { cause?: unknown };
    loop.cause = loop;
    expect(isPreConnectFailure(loop)).toBe(false);

    const outer = new TypeError("fetch failed", { cause: loop });
    expect(isPreConnectFailure(outer)).toBe(false);
  });
});

describe("describeError", () => {
  it("walks a plain Error cause", () => {
    const error = new TypeError("fetch failed", { cause: new Error("connect timeout") });
    expect(describeError(error)).toBe("fetch failed: connect timeout");
  });

  it("surfaces every inner error of an AggregateError cause", () => {
    // Node's dual-stack connect throws one AggregateError whose own
    // `.message` is "" once every resolved address refuses — the detail
    // lives only in `.errors`, which a plain `.cause` walk never reads.
    const error = new TypeError("fetch failed", {
      cause: new AggregateError([
        Object.assign(new Error("connect ECONNREFUSED ::1:8248"), { code: "ECONNREFUSED" }),
        Object.assign(new Error("connect ECONNREFUSED 127.0.0.1:8248"), { code: "ECONNREFUSED" }),
      ]),
    });
    expect(describeError(error)).toBe(
      "fetch failed: connect ECONNREFUSED ::1:8248; connect ECONNREFUSED 127.0.0.1:8248",
    );
  });

  it("recurses into an AggregateError nested inside another AggregateError's .errors", () => {
    const error = new AggregateError([new AggregateError([new Error("a"), new Error("b")]), new Error("c")]);
    expect(describeError(error)).toBe("a; b; c");
  });

  it("falls back to String(error) for a non-Error, non-AggregateError value", () => {
    // A value that is neither an AggregateError nor an Error (e.g. a
    // plain string someone `throw`s) must go straight to `String(error)`
    // — treating it as an Error and reading `.message`/`.name` off it
    // would silently produce `undefined` instead.
    expect(describeError("boom")).toBe("boom");
  });

  it("ignores a cause with a falsy (empty) message, falling back to the outer error", () => {
    // `cause.message` must be truthy, not merely present, or an
    // AggregateError-shaped empty message would print a dangling ": ".
    const error = new TypeError("outer failure", { cause: new Error("") });
    expect(describeError(error)).toBe("outer failure");
  });
});

describe("dropUndefined", () => {
  it("omits both undefined and null, but keeps defined falsy values", () => {
    expect(dropUndefined({ a: undefined, b: null, c: 0, d: "", e: false, f: "x" })).toEqual({
      c: 0,
      d: "",
      e: false,
      f: "x",
    });
  });
});

describe("normalizeImportOutcomes", () => {
  const BATCH = { context: "sake", source: "a", created: true };
  const GROUP = { name: "brewers", outcome: "created", contexts: 2, groups: 0 };
  const SCHEMA = { context: "sake", outcome: "applied" };

  it("takes the structured shape, defaulting missing groups and schemas to []", () => {
    expect(normalizeImportOutcomes({ batches: [BATCH] })).toEqual({
      batches: [BATCH],
      groups: [],
      schemas: [],
      issues: [],
      schema_violations: 0,
    });
  });

  it("keeps groups and schemas verbatim when the server sends them", () => {
    const result = normalizeImportOutcomes({ batches: [BATCH], groups: [GROUP], schemas: [SCHEMA] });
    expect(result.groups).toEqual([GROUP]);
    expect(result.schemas).toEqual([SCHEMA]);
  });

  it("falls back to a single-batch outcome for a batches-less or non-object result (pre-batching servers)", () => {
    expect(normalizeImportOutcomes(BATCH)).toEqual({
      batches: [BATCH],
      groups: [],
      schemas: [],
      issues: [],
      schema_violations: 0,
    });
    expect(normalizeImportOutcomes(null).batches).toEqual([null]);
    expect(normalizeImportOutcomes("not-an-object").batches).toEqual(["not-an-object"]);
    // `batches` present but not an array still takes the bare-outcome path.
    expect(normalizeImportOutcomes({ batches: "nope" }).batches).toEqual([{ batches: "nope" }]);
  });

  it("does not crash when the result itself is undefined", () => {
    // `result !== null` alone does not prove `result` is safe to index —
    // it is also `unknown`-typed and could be `undefined`, which throws
    // on property access unless the `typeof result === "object"` guard
    // actually short-circuits first.
    expect(() => normalizeImportOutcomes(undefined)).not.toThrow();
    expect(normalizeImportOutcomes(undefined).batches).toEqual([undefined]);
  });

  it("passes through given issues and schema_violations instead of the defaults", () => {
    const issue = { path: "x", kind: "range", expected: "a", actual: "b" };
    const result = normalizeImportOutcomes({ batches: [BATCH] }, [issue], 3);
    expect(result.issues).toEqual([issue]);
    expect(result.schema_violations).toBe(3);
  });
});

describe("sleep", () => {
  it("does not resolve well before the requested delay has elapsed", async () => {
    const outcome = await Promise.race([
      sleep(0.05).then(() => "resolved"),
      new Promise((resolve) => setTimeout(() => resolve("still-pending"), 5)),
    ]);
    expect(outcome).toBe("still-pending");
  });

  it("eventually resolves — a broken executor would hang the caller forever", async () => {
    const outcome = await Promise.race([
      sleep(0.001).then(() => "resolved"),
      new Promise((resolve) => setTimeout(() => resolve("still-pending"), 300)),
    ]);
    expect(outcome).toBe("resolved");
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

  /**
   * A body whose `pull` waits `delay` ms per chunk before enqueuing it, but
   * rejects immediately if `signal` fires while it's waiting — standing in
   * for how a real fetch stream errors its reader once the request's
   * AbortSignal aborts mid-download.
   */
  function delayedBody(signal: AbortSignal, delaysMs: number[]): ReadableStream<Uint8Array> {
    let index = 0;
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        // A real fetch stream fails the very next read once its signal
        // has aborted, even if the abort landed while nothing was
        // pulling — a timer left armed across a yield fires on its own
        // schedule, not while this callback happens to be on the stack.
        if (signal.aborted) {
          throw signal.reason;
        }
        if (index >= delaysMs.length) {
          controller.close();
          return;
        }
        const delay = delaysMs[index]!;
        index += 1;
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, delay);
          signal.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(signal.reason);
          });
        });
        controller.enqueue(new TextEncoder().encode("x"));
      },
    });
  }

  it("keeps streaming past the overall timeout as long as each chunk arrives in time", async () => {
    // Three chunks 15ms apart — 45ms total, more than the 30ms timeout, but
    // every individual gap is under it: the timeout must re-arm per chunk,
    // not apply once to the whole download.
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      timeout: 0.03,
      fetch: (_url, init) =>
        Promise.resolve(new Response(delayedBody(init!.signal!, [15, 15, 15]), { status: 200 })),
    });
    const chunks: Uint8Array[] = [];
    for await (const chunk of client.context("sake").exportStream()) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(3);
  });

  it("still aborts on a stalled chunk, even after earlier chunks arrived fine", async () => {
    // First chunk arrives quickly; the second never arrives within the
    // timeout window — must abort despite the earlier progress.
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      timeout: 0.02,
      fetch: (_url, init) =>
        Promise.resolve(new Response(delayedBody(init!.signal!, [5, 200]), { status: 200 })),
    });
    const stream = client.context("sake").exportStream();
    const first = await stream.next();
    expect(first.done).toBe(false);
    await expect(stream.next()).rejects.toThrow(/aborted/i);
  });

  it("does not count the consumer's own processing time against the timeout", async () => {
    // Every chunk arrives instantly — only the consumer is slow, taking
    // longer between chunks than the timeout allows. A timer left armed
    // across the yield would abort on the consumer's own pace even
    // though the network side never stalled.
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      timeout: 0.02,
      fetch: (_url, init) =>
        Promise.resolve(new Response(delayedBody(init!.signal!, [0, 0, 0]), { status: 200 })),
    });
    const chunks: Uint8Array[] = [];
    for await (const chunk of client.context("sake").exportStream()) {
      chunks.push(chunk);
      await new Promise((resolve) => setTimeout(resolve, 40));
    }
    expect(chunks).toHaveLength(3);
  });
});
