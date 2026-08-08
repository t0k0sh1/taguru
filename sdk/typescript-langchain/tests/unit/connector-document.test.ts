/** ConnectorDocument and its component types (ADR 0007 §5, issue #347;
 * TypeScript parity: issue #415). */

import { describe, expect, test } from "vitest";

import {
  CONNECTOR_DOCUMENT_VERSION,
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  FingerprintInputs,
  LocatorEntry,
  SectionEntry,
  optionsDigest,
} from "../../src/ingest-connectors/document.js";

function metadata(
  overrides: Partial<ConstructorParameters<typeof ConnectorMetadata>[0]> = {},
): ConnectorMetadata {
  return new ConnectorMetadata({ originUri: "doc.md", displayName: "doc.md", ...overrides });
}

function fingerprint(
  overrides: Partial<ConstructorParameters<typeof FingerprintInputs>[0]> = {},
): FingerprintInputs {
  return new FingerprintInputs({
    rawContentSha256: "deadbeef",
    parser: "taguru-text-connector",
    parserVersion: "1.0.0",
    parseOptionsDigest: "cafef00d",
    ...overrides,
  });
}

function document(
  overrides: Partial<ConstructorParameters<typeof ConnectorDocument>[0]> = {},
): ConnectorDocument {
  return new ConnectorDocument({
    source: "doc.md",
    text: "paragraph one.",
    metadata: metadata(),
    fingerprintInputs: fingerprint(),
    ...overrides,
  });
}

test("toDict round-trips through fromDict", () => {
  const original = document({
    text: "paragraph one.\n\nparagraph two.",
    locators: [new LocatorEntry({ paragraph: 1, locator: { kind: "page", value: "12" } })],
    sections: [new SectionEntry({ paragraph: 0, section: "導入" })],
    metadata: metadata({ title: "タイトル", tags: ["quarterly"], contentType: "text/markdown" }),
  });
  const restored = ConnectorDocument.fromDict(original.toDict());
  expect(restored).toEqual(original);
});

test("fromDict rejects an unrecognized connector_document version", () => {
  const payload = document().toDict();
  payload["connector_document"] = 2;
  expect(ConnectorDocument.fromDict(payload)).toBeNull();
});

describe("fromDict returns null on structural mismatch", () => {
  const cases: Array<[string, (payload: Record<string, unknown>) => void]> = [
    ["missing_metadata", (payload) => delete payload["metadata"]],
    ["missing_fingerprint", (payload) => delete payload["fingerprint_inputs"]],
    [
      "bad_locator",
      (payload) =>
        (payload["locators"] as unknown[]).push({ paragraph: "not-an-int", locator: {} }),
    ],
    ["missing_source", (payload) => delete payload["source"]],
    ["bad_diagnostic", (payload) => (payload["diagnostics"] = [{ code: "not-a-real-code" }])],
  ];
  test.each(cases)("%s", (_name, mutate) => {
    const payload = document({
      locators: [new LocatorEntry({ paragraph: 0, locator: { kind: "page", value: "1" } })],
    }).toDict();
    mutate(payload);
    expect(ConnectorDocument.fromDict(payload)).toBeNull();
  });
});

test("fromDict rejects non-dict input", () => {
  expect(ConnectorDocument.fromDict(["not", "a", "dict"])).toBeNull();
  expect(ConnectorDocument.fromDict(null)).toBeNull();
});

test("empty text requires at least one diagnostic", () => {
  expect(() => document({ text: "" })).toThrow(/diagnostic/);
});

test("empty text with a diagnostic is allowed", () => {
  const doc = document({
    text: "",
    diagnostics: [new Diagnostic({ code: "unreadable", message: "boom", source: "doc.md" })],
  });
  expect(doc.text).toBe("");
  expect(doc.diagnostics[0]!.code).toBe("unreadable");
});

test("source must not be empty", () => {
  expect(() => document({ source: "" })).toThrow(/source/);
});

test("oversized source is only allowed when text is empty", () => {
  const longSource = "x".repeat(1025);
  expect(() => document({ source: longSource })).toThrow(/1024/);
  // With an empty text and a diagnostic, the same oversized source is
  // exactly what a connector must be able to report (ADR 0007 §6.1).
  const doc = document({
    source: longSource,
    text: "",
    diagnostics: [
      new Diagnostic({ code: "source_id_too_long", message: "too long", source: longSource }),
    ],
  });
  expect(doc.source).toBe(longSource);
});

describe("locator kind and value are size-checked", () => {
  const cases: Array<[string, string, string]> = [
    ["empty_kind", "", "12"],
    ["empty_value", "page", ""],
    ["kind_too_long", "p".repeat(65), "12"],
    ["value_too_long", "page", "v".repeat(513)],
  ];
  test.each(cases)("%s", (_name, kind, value) => {
    expect(() =>
      document({ locators: [new LocatorEntry({ paragraph: 0, locator: { kind, value } })] }),
    ).toThrow();
  });
});

test("locator paragraph must not be negative", () => {
  expect(() =>
    document({
      locators: [new LocatorEntry({ paragraph: -1, locator: { kind: "page", value: "1" } })],
    }),
  ).toThrow(/paragraph/);
});

describe("section label is size-checked", () => {
  const cases: Array<[string, string]> = [
    ["empty", ""],
    ["too_long", "s".repeat(513)],
  ];
  test.each(cases)("%s", (_name, section) => {
    expect(() => document({ sections: [new SectionEntry({ paragraph: 0, section })] })).toThrow();
  });
});

test("more than the max tags is rejected", () => {
  const tags = Array.from({ length: 33 }, (_item, index) => `tag${index}`);
  expect(() => document({ metadata: metadata({ tags }) })).toThrow(/tags/);
});

test("oversized tag is rejected", () => {
  expect(() => document({ metadata: metadata({ tags: ["t".repeat(129)] }) })).toThrow();
});

test("optionsDigest is independent of key order", async () => {
  expect(await optionsDigest({ a: 1, b: 2 })).toBe(await optionsDigest({ b: 2, a: 1 }));
});

test("optionsDigest differs on value change", async () => {
  expect(await optionsDigest({ extract_headings: true })).not.toBe(
    await optionsDigest({ extract_headings: false }),
  );
});

test("CONNECTOR_DOCUMENT_VERSION constant is one", () => {
  // Checked for equality (not a range) — a protocol-breaking change bumps
  // this, never loosens the check (ADR 0007 §5, mirrors BATCH_VERSION).
  expect(CONNECTOR_DOCUMENT_VERSION).toBe(1);
});
