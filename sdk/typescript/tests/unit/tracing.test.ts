/**
 * Tracing (ADR 0008, #224): span tree shape, skip reasons, attribute
 * types, `traceparent` propagation, and the privacy sentinel.
 *
 * Unlike the Python SDK's `test_tracing.py`, this file needs no CI
 * gymnastics to prove the OTel-absent path works — `@opentelemetry/api`
 * is always present here as a devDependency, so `tracing-absent.test.ts`
 * simulates absence directly via `vi.mock` instead. `sdk/spec/
 * tracing.yaml` is this file's contract with the Python SDK's own
 * `test_tracing.py` — read together, not just here.
 */

import {
  context as otelContext,
  propagation as otelPropagation,
  SpanStatusCode,
} from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { ServerError } from "../../src/errors.js";
import * as tracing from "../../src/tracing.js";
import { errBody, okBody, stubClient, type StubRequest } from "./stub.js";

// A real, Node-standard context manager, registered once for this whole
// file — without one, `startActiveSpan` can't find its parent across
// nested calls (each span would start a fresh trace). Never done by
// production code itself (see tracing.ts's module docstring): this is
// purely a test-harness concern, the same as installing an SDK would be
// for a real application.
beforeAll(() => {
  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  otelContext.setGlobalContextManager(contextManager);
  // `propagation.inject` — what `injectHeaders` calls — is a no-op
  // against the API package's own default global propagator, same as
  // the tracer provider defaulting to no-op until something registers
  // one. Real applications get this from whichever SDK they install;
  // here that's this file's job.
  otelPropagation.setGlobalPropagator(new W3CTraceContextPropagator());
});

let exporter: InMemorySpanExporter;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  tracing._setProviderOverride(provider);
});

afterEach(() => {
  tracing._setProviderOverride(undefined);
});

function byName(name: string): ReadableSpan[] {
  return exporter.getFinishedSpans().filter((span) => span.name === name);
}

function one(name: string): ReadableSpan {
  const spans = byName(name);
  expect(spans, `expected exactly one ${name}, found ${spans.length}`).toHaveLength(1);
  return spans[0]!;
}

function skipReasons(span: ReadableSpan): unknown[] {
  return span.events
    .filter((event) => event.name === tracing.SKIP_EVENT)
    .map((event) => event.attributes?.[tracing.REASON_FIELD]);
}

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

function routed(calls: Array<{ path: string; headers: Record<string, string> }>) {
  return (req: StubRequest) => {
    calls.push({ path: req.path, headers: req.headers });
    const body = req.body ? (JSON.parse(req.body) as Record<string, unknown>) : {};
    if (req.path.endsWith("/resolve")) {
      return okBody(
        body["cue"] === "青嶺"
          ? [{ name: "青嶺酒造", score: 1.0, tier: "lexical", kind: "exact" }]
          : [],
      );
    }
    if (req.path.endsWith("/describe")) {
      return okBody({ concept: "青嶺酒造", as_subject: [{ label: "杜氏", count: 1 }], as_object: [] });
    }
    if (req.path.endsWith("/query")) {
      return okBody({ total: 1, matches: [ASSOCIATION] });
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
              lanes: { bm25: { ran: true }, vector: { ran: false, reason: "no provider" } },
            },
          ],
        },
        hits: [],
      });
    }
    throw new Error(req.path);
  };
}

describe("retrieve() tracing", () => {
  it("produces a root span and every phase span on a full run", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client.context("sake").retrieve("青嶺", {
      labels: "杜氏",
      text_fallback_query: "杜氏は誰か",
      text_fallback_only_if_empty: false,
    });

    const root = one(tracing.ROOT_SPAN);
    expect(root.attributes["taguru.origin.count"]).toBe(1);
    expect(root.attributes["taguru.anchor.count"]).toBe(1);
    expect(root.attributes["taguru.association.count"]).toBe(1);
    expect(root.attributes["taguru.activation.count"]).toBe(1);
    expect(root.attributes["taguru.fallback.ran"]).toBe(true);
    expect(skipReasons(root)).toEqual([]);

    for (const phase of [
      "taguru.resolve",
      "taguru.describe",
      "taguru.query",
      "taguru.activate",
      "taguru.citations",
      "taguru.passage_fallback",
    ]) {
      const span = one(phase);
      expect(span.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
      expect(span.spanContext().traceId).toBe(root.spanContext().traceId);
    }
  });

  it("records skip reasons for disabled steps", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client.context("sake").retrieve("青嶺", { describe_first: false, fetch_citations: false });

    const root = one(tracing.ROOT_SPAN);
    // Length checked before the Set comparison below — a duplicate skip
    // event (production trace noise the Set comparison alone would miss)
    // must still fail this test.
    expect(skipReasons(root)).toHaveLength(4);
    expect(new Set(skipReasons(root))).toEqual(
      new Set(["describe_disabled", "labels_absent", "citations_disabled", "fallback_not_requested"]),
    );
    for (const absent of [
      "taguru.describe",
      "taguru.query",
      "taguru.citations",
      "taguru.passage_fallback",
    ]) {
      expect(byName(absent)).toEqual([]);
    }
    // anchors resolved and labels absent, but activate still runs unconditionally.
    expect(byName("taguru.activate")).toHaveLength(1);
  });

  it("skips the whole graph cluster when no anchor resolves", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client.context("sake").retrieve("無関係", { labels: "杜氏" });

    const root = one(tracing.ROOT_SPAN);
    expect(root.attributes["taguru.anchor.count"]).toBe(0);
    expect(skipReasons(root)).toContain("no_anchors");
    for (const absent of ["taguru.describe", "taguru.query", "taguru.activate"]) {
      expect(byName(absent)).toEqual([]);
    }
    // fetch_citations defaults true, so the (empty) phase still opens.
    expect(byName("taguru.citations")).toHaveLength(1);
  });

  it("aggregates citation misses into one event", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client.context("sake").retrieve("青嶺");

    const citationsSpan = one("taguru.citations");
    const missing = citationsSpan.events.filter(
      (event) => event.name === tracing.CITATION_MISSING_EVENT,
    );
    expect(missing).toHaveLength(1);
    expect(missing[0]!.attributes?.[tracing.CITATION_MISSING_FIELD]).toBe(1);
  });

  it("does not emit the citation-missing event when nothing was missing", async () => {
    const allStored = {
      ...ASSOCIATION,
      attributions: [
        { source: "docs/aomine.md", weight: 2.0, count: 2, paragraph: 1, section: null },
        { source: "docs/other.md", weight: 1.0, count: 1, paragraph: 0, section: null },
      ],
    };
    const handler = (req: StubRequest) => {
      const body = req.body ? (JSON.parse(req.body) as Record<string, unknown>) : {};
      if (req.path.endsWith("/resolve")) {
        return okBody([{ name: "青嶺酒造", score: 1.0, tier: "lexical", kind: "exact" }]);
      }
      if (req.path.endsWith("/describe")) {
        return okBody({ concept: "青嶺酒造", as_subject: [], as_object: [] });
      }
      if (req.path.endsWith("/activate")) {
        return okBody({
          total: 1,
          matches: [{ strength: 0.9, path: ["青嶺酒造"], association: allStored }],
        });
      }
      if (req.path.endsWith("/citations")) {
        // Every source here is stored — none 404s.
        return okBody({ text: "杜氏は高瀬。", source: body["source"], section: "人物" });
      }
      throw new Error(req.path);
    };
    const client = stubClient(handler, { retries: 0 });
    await client.context("sake").retrieve("青嶺");

    const citationsSpan = one("taguru.citations");
    const missing = citationsSpan.events.filter(
      (event) => event.name === tracing.CITATION_MISSING_EVENT,
    );
    expect(missing).toHaveLength(0);
  });

  it("marks the fallback suppressed when the graph already answered", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client.context("sake").retrieve("青嶺", { text_fallback_query: "杜氏は誰か" });

    const root = one(tracing.ROOT_SPAN);
    expect(root.attributes["taguru.fallback.ran"]).toBe(false);
    expect(skipReasons(root)).toContain("fallback_suppressed");
    expect(byName("taguru.passage_fallback")).toEqual([]);
  });

  it("injects traceparent from the active phase span into outbound requests", async () => {
    const calls: Array<{ path: string; headers: Record<string, string> }> = [];
    const client = stubClient(routed(calls), { retries: 0 });
    await client.context("sake").retrieve("青嶺", { describe_first: false, fetch_citations: false });

    const resolveSpan = one("taguru.resolve");
    const resolveCall = calls.find((call) => call.path.endsWith("/resolve"));
    expect(resolveCall).toBeDefined();
    const traceparent = resolveCall?.headers["traceparent"];
    expect(traceparent).toBeDefined();
    const [, traceId, spanId] = traceparent!.split("-");
    expect(traceId).toBe(resolveSpan.spanContext().traceId);
    expect(spanId).toBe(resolveSpan.spanContext().spanId);
  });

  it("marks the failing phase and root span ERROR on a genuine 5xx", async () => {
    // Unlike Python's `start_as_current_span` (ERROR-on-exception by
    // default), `startActiveSpan` does nothing on its own — this is
    // exactly what `tracing.ts`'s own `span()` now does by hand. A
    // real 5xx must propagate as a thrown `ServerError` AND leave both
    // the failing phase span and the root ERROR, not fail silently.
    const client = stubClient(
      (req: StubRequest) => {
        if (req.path.endsWith("/resolve")) {
          return errBody(500, "boom");
        }
        throw new Error(req.path);
      },
      { retries: 0 },
    );

    await expect(client.context("sake").retrieve("青嶺")).rejects.toBeInstanceOf(ServerError);

    const resolveSpan = one("taguru.resolve");
    expect(resolveSpan.status.code).toBe(SpanStatusCode.ERROR);
    const root = one(tracing.ROOT_SPAN);
    expect(root.status.code).toBe(SpanStatusCode.ERROR);
  });

  it("never leaks raw cue/label/query text into any span", async () => {
    const client = stubClient(routed([]), { retries: 0 });
    await client
      .context("sake")
      .retrieve("青嶺", { labels: "杜氏", text_fallback_query: "SENTINEL-QUERY-9f2c" });

    const serialized = JSON.stringify(
      exporter.getFinishedSpans().map((span) => ({
        name: span.name,
        attributes: span.attributes,
        events: span.events.map((event) => ({ name: event.name, attributes: event.attributes })),
      })),
    );
    for (const nonce of [
      "青嶺",
      "青嶺酒造",
      "杜氏",
      "SENTINEL-QUERY-9f2c",
      "docs/aomine.md",
      "unstored.md",
    ]) {
      expect(serialized.includes(nonce)).toBe(false);
    }
  });
});

describe("retrieve() tracing with no tracer provider configured", () => {
  it("still works — a harmless no-op, not a crash", async () => {
    // Undoes the outer beforeEach's local override, falling back to the
    // real global default: `@opentelemetry/api` installed, but no
    // application has ever called `setGlobalTracerProvider`. That is the
    // state every real caller starts in before opting into tracing.
    tracing._setProviderOverride(undefined);
    const client = stubClient(routed([]), { retries: 0 });
    const result = await client.context("sake").retrieve("青嶺");
    expect(result.associations.length).toBeGreaterThan(0);
  });
});
