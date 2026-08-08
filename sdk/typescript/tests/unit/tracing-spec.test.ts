/**
 * `src/tracing.ts`'s vocabulary against `sdk/spec/tracing.yaml` — the
 * shared source of truth both SDKs' tracing test suites check themselves
 * against (see that file's own comments, and the Python SDK's
 * `tests/unit/test_tracing_spec.py`). Deliberately independent of
 * `tracing.test.ts`: no `TracerProvider` needed here, just `tracing.ts`'s
 * own exported constants and `SpanHandle` shape.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parse } from "yaml";
import { describe, expect, it } from "vitest";

import * as tracing from "../../src/tracing.js";

/**
 * Walk upward to `sdk/spec/tracing.yaml` instead of hard-coding
 * `../../../`: under Stryker the suite runs from a sandbox copy nested
 * inside `.stryker-tmp/`, which shifts the test file's depth relative to
 * the (uncopied) spec directory.
 */
function specPath(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "spec", "tracing.yaml");
    if (existsSync(candidate)) {
      return candidate;
    }
    const parent = resolve(dir, "..");
    if (parent === dir) {
      throw new Error("sdk/spec/tracing.yaml not found in any ancestor directory");
    }
    dir = parent;
  }
}

const SPEC_PATH = specPath();

interface TracingSpec {
  tracer_name: string;
  root_span: string;
  skip_event: string;
  reason_field: string;
  citation_missing_event: string;
  citation_missing_field: string;
  skip_reasons: string[];
  phase_spans: string[];
  root_attributes: Record<string, string>;
  privacy: { allowed_value_kinds: string[]; no_per_item_signals: boolean };
}

function loadSpec(): TracingSpec {
  return parse(readFileSync(SPEC_PATH, "utf-8")) as TracingSpec;
}

// `Reason` is a type, erased at compile time — there is nothing to
// reflect at runtime, so every skip-reason string this SDK actually
// emits lives here once, and `client.ts`'s `retrieve()` is the thing
// that has to keep using exactly these six literals.
const REASON_VALUES = [
  "describe_disabled",
  "no_anchors",
  "labels_absent",
  "citations_disabled",
  "fallback_not_requested",
  "fallback_suppressed",
] as const satisfies readonly tracing.Reason[];

describe("tracing.ts vocabulary vs. the shared spec", () => {
  it("names and reason vocabulary match sdk/spec/tracing.yaml", () => {
    const spec = loadSpec();
    expect(tracing.TRACER_NAME).toBe(spec.tracer_name);
    expect(tracing.ROOT_SPAN).toBe(spec.root_span);
    expect(tracing.SKIP_EVENT).toBe(spec.skip_event);
    expect(tracing.REASON_FIELD).toBe(spec.reason_field);
    expect(tracing.CITATION_MISSING_EVENT).toBe(spec.citation_missing_event);
    expect(tracing.CITATION_MISSING_FIELD).toBe(spec.citation_missing_field);
    expect(new Set(REASON_VALUES)).toEqual(new Set(spec.skip_reasons));
  });

  it("phase span names match sdk/spec/tracing.yaml", () => {
    const spec = loadSpec();
    const phaseSpans = [
      tracing.SPAN_RESOLVE,
      tracing.SPAN_DESCRIBE,
      tracing.SPAN_QUERY,
      tracing.SPAN_ACTIVATE,
      tracing.SPAN_CITATIONS,
      tracing.SPAN_PASSAGE_FALLBACK,
    ];
    expect(new Set(phaseSpans)).toEqual(new Set(spec.phase_spans));
  });

  it("root attribute keys match sdk/spec/tracing.yaml", () => {
    const spec = loadSpec();
    const rootAttributeKeys = [
      tracing.ATTR_ORIGIN_COUNT,
      tracing.ATTR_ANCHOR_COUNT,
      tracing.ATTR_ASSOCIATION_COUNT,
      tracing.ATTR_ACTIVATION_COUNT,
      tracing.ATTR_CITATION_RETURNED,
      tracing.ATTR_PASSAGE_HIT_COUNT,
      tracing.ATTR_FALLBACK_RAN,
    ];
    expect(new Set(rootAttributeKeys)).toEqual(new Set(Object.keys(spec.root_attributes)));
  });

  it("SpanHandle's surface is exactly the spec's closed privacy vocabulary", () => {
    const spec = loadSpec();
    expect(spec.privacy.allowed_value_kinds).toEqual(["count", "flag", "reason"]);
    expect(spec.privacy.no_per_item_signals).toBe(true);
    // `count`/`flag`/`skip` (this SDK's name for the spec's "reason"
    // kind)/`citationMissing`, nothing else — there is no free-form
    // string setter to accidentally add.
    const handle: tracing.SpanHandle = {
      count() {},
      flag() {},
      skip() {},
      citationMissing() {},
    };
    expect(Object.keys(handle).sort()).toEqual(["citationMissing", "count", "flag", "skip"]);
  });
});
