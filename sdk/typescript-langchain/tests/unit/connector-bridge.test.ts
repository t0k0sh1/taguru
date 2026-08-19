/**
 * ingestConnectorDocument/ingestConnectorDocuments (ADR 0007, issue #347) —
 * the connector-to-TaguruIngester bridge, "end to end" wiring. Mirrors the
 * Python suite's test_connector_bridge.py case-for-case where this SDK's
 * `TaguruIngester.ingestText` (ingest.ts) has the feature being asserted;
 * see bridge.ts's own module doc for the two cases where it deliberately
 * does not (sections/locators are not yet forwarded onto the wire; there is
 * one async function per Python sync/async pair, not two).
 * TypeScript parity: issue #415.
 */

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, it } from "vitest";

import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  FingerprintInputs,
  LocatorEntry,
  SectionEntry,
} from "../../src/ingest-connectors/document.js";
import {
  ingestConnectorDocument,
  ingestConnectorDocuments,
} from "../../src/ingest-connectors/bridge.js";
import { TaguruIngester } from "../../src/ingest.js";
import { FakeServer } from "./stub.js";

const EMPTY_ANSWER = JSON.stringify({ associations: [], aliases: [], questions: [] });

function fingerprint(): FingerprintInputs {
  return new FingerprintInputs({
    rawContentSha256: "deadbeef",
    parser: "taguru-text-connector",
    parserVersion: "1.0.0",
    parseOptionsDigest: "cafef00d",
  });
}

function document(options: {
  source?: string;
  text?: string;
  sections?: readonly SectionEntry[];
  locators?: readonly LocatorEntry[];
  diagnostics?: readonly Diagnostic[];
} = {}): ConnectorDocument {
  const source = options.source ?? "doc.md";
  return new ConnectorDocument({
    source,
    text: options.text ?? "paragraph one.\n\nparagraph two.",
    sections: options.sections ?? [],
    locators: options.locators ?? [],
    metadata: new ConnectorMetadata({ originUri: source, displayName: source }),
    fingerprintInputs: fingerprint(),
    diagnostics: options.diagnostics ?? [],
  });
}

function ingester(server: FakeServer, responses: string[] = [EMPTY_ANSWER, EMPTY_ANSWER]): TaguruIngester {
  return new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses }),
    client: server.client(),
  });
}

describe("ingestConnectorDocument", () => {
  it("ingests a document's text through the ingester's normal batch/import path", async () => {
    const server = new FakeServer();
    const outcome = await ingestConnectorDocument(ingester(server), document());
    expect(outcome.ok).toBe(true);
    expect(outcome.source).toBe("doc.md");
    expect(server.imported).toHaveLength(1);
  });

  it("carries sections and locators onto the wire", async () => {
    const server = new FakeServer();
    const doc = document({
      sections: [new SectionEntry({ paragraph: 0, section: "導入" })],
      locators: [new LocatorEntry({ paragraph: 1, locator: { kind: "page", value: "1" } })],
    });
    const outcome = await ingestConnectorDocument(ingester(server), doc);

    expect(outcome.ok).toBe(true);
    expect(outcome.source).toBe(doc.source);
    const lines = server.imported[0]!.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toContainEqual({ paragraph: 0, section: "導入" });
    expect(lines).toContainEqual({ paragraph: 1, locator: { kind: "page", value: "1" } });
  });

  it("include_passage: false drops sections and locators", async () => {
    const server = new FakeServer();
    const passageless = new TaguruIngester({
      context: "sake",
      llm: new FakeListChatModel({ responses: [EMPTY_ANSWER] }),
      client: server.client(),
      include_passage: false,
    });
    const doc = document({
      sections: [new SectionEntry({ paragraph: 0, section: "導入" })],
      locators: [new LocatorEntry({ paragraph: 1, locator: { kind: "page", value: "1" } })],
    });
    const outcome = await ingestConnectorDocument(passageless, doc);
    expect(outcome.ok).toBe(true);
    const ndjson = server.imported[0]!;
    expect(ndjson).not.toContain("section");
    expect(ndjson).not.toContain("locator");
  });

  it("a diagnostics-only document never reaches the network", async () => {
    const server = new FakeServer();
    const doc = document({
      text: "",
      diagnostics: [new Diagnostic({ code: "unreadable", message: "boom", source: "doc.md" })],
    });
    const outcome = await ingestConnectorDocument(ingester(server), doc);

    expect(outcome.ok).toBe(false);
    expect(outcome.error).toContain("unreadable");
    expect(outcome.error).toContain("boom");
    expect(server.imported).toEqual([]);
    expect(server.calls).toEqual([]);
  });

  it("include_passage: false still ingests a diagnostics-carrying document's text", async () => {
    const server = new FakeServer();
    const withoutPassage = new TaguruIngester({
      context: "sake",
      llm: new FakeListChatModel({ responses: [EMPTY_ANSWER] }),
      client: server.client(),
      include_passage: false,
    });
    const doc = document({
      sections: [new SectionEntry({ paragraph: 0, section: "導入" })],
      locators: [new LocatorEntry({ paragraph: 1, locator: { kind: "page", value: "1" } })],
    });
    const outcome = await ingestConnectorDocument(withoutPassage, doc);
    expect(outcome.ok).toBe(true);
    const ndjson = server.imported[0]!;
    expect(ndjson).not.toContain("section");
    expect(ndjson).not.toContain("locator");
  });

  it("ingest outcome counters come from the server's import outcome", async () => {
    const server = new FakeServer();
    const outcome = await ingestConnectorDocument(ingester(server), document());
    // stub.ts's FakeServer always answers /import with this fixed batch —
    // unlike the Python conftest's FakeServer, it has no
    // import_result_override knob to customize per test.
    expect(outcome.associations).toBe(2);
    expect(outcome.aliases).toBe(1);
    expect(outcome.passage_stored).toBe(true);
    expect(outcome.questions_stored).toBe(1);
  });
});

describe("ingestConnectorDocuments", () => {
  it("continues past a diagnostics-only document", async () => {
    const server = new FakeServer();
    const documents = [
      document({ source: "a.md" }),
      document({
        source: "b.md",
        text: "",
        diagnostics: [new Diagnostic({ code: "unreadable", message: "boom", source: "b.md" })],
      }),
      document({ source: "c.md" }),
    ];
    const outcomes = await ingestConnectorDocuments(
      ingester(server, [EMPTY_ANSWER, EMPTY_ANSWER]),
      documents,
    );
    expect(outcomes.map((outcome) => outcome.source)).toEqual(["a.md", "b.md", "c.md"]);
    expect(outcomes.map((outcome) => outcome.ok)).toEqual([true, false, true]);
    expect(server.imported).toHaveLength(2); // a.md and c.md only
  });
});

describe("ingestConnectorDocuments raise_on_error branches (issue #737)", () => {
  it("records each failure and continues at the default raise_on_error=false", async () => {
    const server = new FakeServer();
    server.failImport = true;
    const documents = [
      document({ source: "a.md" }),
      document({ source: "b.md" }),
      document({ source: "c.md" }),
    ];
    const outcomes = await ingestConnectorDocuments(
      ingester(server, [EMPTY_ANSWER, EMPTY_ANSWER, EMPTY_ANSWER]),
      documents,
    );
    expect(outcomes.map((outcome) => outcome.source)).toEqual(["a.md", "b.md", "c.md"]);
    expect(outcomes.map((outcome) => outcome.ok)).toEqual([false, false, false]);
    for (const outcome of outcomes) {
      expect(outcome.error).toBeTruthy();
    }
    const importCalls = server.calls.filter(([path]) => path.startsWith("/import"));
    expect(importCalls).toHaveLength(3);
  });

  it("re-raises the first failure with raise_on_error=true and attempts nothing after it", async () => {
    const server = new FakeServer();
    server.failImport = true;
    const failFast = new TaguruIngester({
      context: "sake",
      llm: new FakeListChatModel({ responses: [EMPTY_ANSWER, EMPTY_ANSWER, EMPTY_ANSWER] }),
      client: server.client(),
      raise_on_error: true,
    });
    const documents = [
      document({ source: "a.md" }),
      document({ source: "b.md" }),
      document({ source: "c.md" }),
    ];
    await expect(ingestConnectorDocuments(failFast, documents)).rejects.toThrow();
    const importCalls = server.calls.filter(([path]) => path.startsWith("/import"));
    expect(importCalls).toHaveLength(1);
  });
});
