/**
 * Optional OpenTelemetry tracing for `retrieve()` (ADR 0008; #224).
 *
 * `@opentelemetry/api` is an OPTIONAL peer dependency (see package.json's
 * `peerDependencies` + `peerDependenciesMeta`), never a regular one: the
 * package self-registers under a well-known global symbol
 * (`Symbol.for("opentelemetry.js.api.1")`), so a second bundled copy
 * under `node_modules/taguru/node_modules/@opentelemetry/api` would
 * become a second, disconnected registration — whichever copy happens to
 * win makes every OTHER instrumented library's context propagation a
 * no-op too, not just this SDK's. Every access below goes through a
 * lazily cached dynamic `import()` instead of a static one, so this
 * module — and therefore this whole package — loads fine with the peer
 * dependency entirely absent; a plain `npm install taguru` never even
 * attempts to resolve it. The build (`tsup ... --external
 * @opentelemetry/api`) keeps it out of the bundle for the same reason.
 *
 * Privacy (ADR 0008 §9 mirrored client-side; see sdk/spec/tracing.yaml,
 * shared with the Python SDK's `taguru._tracing`): the only ways to put
 * a value on a span are `count`/`flag`/`skip` — a bounded int, a bool,
 * or a value from the closed `Reason` union. There is no free-form
 * string setter anywhere in this module's surface: cue text, origins,
 * labels, and passage content cannot reach a span through it.
 */

import type { Span as OtelSpan, TracerProvider } from "@opentelemetry/api";

export const TRACER_NAME = "taguru";
export const ROOT_SPAN = "taguru.retrieve";
export const SKIP_EVENT = "taguru.skip";
export const REASON_FIELD = "taguru.reason";
export const CITATION_MISSING_EVENT = "taguru.citation_missing";
export const CITATION_MISSING_FIELD = "taguru.citation.missing";

/** `retrieve()`'s phase spans, in the order it may open them — named here
 * once so `client.ts` and this SDK's own spec-parity test share a single
 * source instead of three copies of the same six strings. */
export const SPAN_RESOLVE = "taguru.resolve";
export const SPAN_DESCRIBE = "taguru.describe";
export const SPAN_QUERY = "taguru.query";
export const SPAN_ACTIVATE = "taguru.activate";
export const SPAN_CITATIONS = "taguru.citations";
export const SPAN_PASSAGE_FALLBACK = "taguru.passage_fallback";

/** The root span's attribute keys `retrieve()` records — same reasoning. */
export const ATTR_ORIGIN_COUNT = "taguru.origin.count";
export const ATTR_ANCHOR_COUNT = "taguru.anchor.count";
export const ATTR_ASSOCIATION_COUNT = "taguru.association.count";
export const ATTR_ACTIVATION_COUNT = "taguru.activation.count";
export const ATTR_CITATION_RETURNED = "taguru.citation.returned";
export const ATTR_PASSAGE_HIT_COUNT = "taguru.passage.hit_count";
export const ATTR_FALLBACK_RAN = "taguru.fallback.ran";

/** `retrieve()`'s closed skip-reason vocabulary — kept in lockstep with
 * `sdk/spec/tracing.yaml` and the Python SDK's `taguru._tracing.Reason`. */
export type Reason =
  | "describe_disabled"
  | "no_anchors"
  | "labels_absent"
  | "citations_disabled"
  | "fallback_not_requested"
  | "fallback_suppressed";

/** The only way `retrieve()` may touch a span — see the module docstring. */
export interface SpanHandle {
  count(key: string, value: number): void;
  flag(key: string, value: boolean): void;
  /** Record a `taguru.skip` event: a planned step did not run. */
  skip(reason: Reason): void;
  /** One aggregate event for every citation lookup that 404'd during this
   * call — never one event per miss (ADR 0008's per-item aggregation
   * rule: a citation locator is caller-shaped, unbounded data). */
  citationMissing(count: number): void;
}

const NULL_SPAN: SpanHandle = {
  count() {},
  flag() {},
  skip() {},
  citationMissing() {},
};

function wrap(otelSpan: OtelSpan): SpanHandle {
  return {
    count(key, value) {
      otelSpan.setAttribute(key, value);
    },
    flag(key, value) {
      otelSpan.setAttribute(key, value);
    },
    skip(reason) {
      otelSpan.addEvent(SKIP_EVENT, { [REASON_FIELD]: reason });
    },
    citationMissing(count) {
      if (count > 0) {
        otelSpan.addEvent(CITATION_MISSING_EVENT, { [CITATION_MISSING_FIELD]: count });
      }
    },
  };
}

// Test seam: production code never calls this. `trace.setGlobalTracerProvider`
// is process-global and one-shot (a second call just warns and is
// ignored), which would make the test suite order-dependent if tests used
// it to install per-test exporters. Tests instead set this module
// variable directly to a fresh, local `TracerProvider`, leaving the real
// global provider (if any) untouched — mirrors
// `taguru._tracing._provider_override` on the Python side.
let providerOverride: TracerProvider | undefined;

/** @internal test seam — not exported from `index.ts`. */
export function _setProviderOverride(provider: TracerProvider | undefined): void {
  providerOverride = provider;
}

type OtelApi = typeof import("@opentelemetry/api");

// `undefined` = not yet attempted; `null` = attempted and unavailable.
let otelModule: OtelApi | null | undefined;

async function loadOtel(): Promise<OtelApi | null> {
  if (otelModule !== undefined) {
    return otelModule;
  }
  try {
    otelModule = await import("@opentelemetry/api");
  } catch {
    // No `@opentelemetry/api` installed at all — the expected state for
    // most consumers (it is an optional peer dependency).
    otelModule = null;
  }
  return otelModule;
}

/**
 * Run `body` inside a child span under the current context, or — with no
 * `@opentelemetry/api` installed, or no `TracerProvider` configured — just
 * run `body` directly against a `SpanHandle` that silently drops
 * everything. Call sites never need to branch on whether tracing is
 * active.
 */
export async function span<T>(name: string, body: (span: SpanHandle) => Promise<T>): Promise<T> {
  const otel = await loadOtel();
  if (otel === null) {
    return body(NULL_SPAN);
  }
  const provider = providerOverride ?? otel.trace.getTracerProvider();
  const tracer = provider.getTracer(TRACER_NAME);
  return tracer.startActiveSpan(name, async (otelSpan) => {
    try {
      return await body(wrap(otelSpan));
    } finally {
      otelSpan.end();
    }
  });
}

/**
 * Inject the current span's W3C `traceparent`/`tracestate` into `headers`,
 * mutating it in place.
 *
 * A no-op with no `@opentelemetry/api` installed, no active span, or no
 * configured propagator — every one of those is the propagator's own
 * established no-op behavior, not something this function special-cases.
 */
export async function injectHeaders(headers: Record<string, string>): Promise<void> {
  const otel = await loadOtel();
  if (otel === null) {
    return;
  }
  otel.propagation.inject(otel.context.active(), headers);
}
