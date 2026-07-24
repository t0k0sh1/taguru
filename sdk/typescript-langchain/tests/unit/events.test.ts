/** TaguruIngester.on_event: per-attempt progress/diagnostic events — mirrors the Python suite's test_events.py. */

import { Document } from "@langchain/core/documents";
import { AIMessage } from "@langchain/core/messages";
import { describe, expect, it, vi } from "vitest";

import type {
  AttemptFailed,
  AttemptStarted,
  EmbeddingRefreshCompleted,
  EmbeddingRefreshWarning,
  IngestEvent,
  ProviderMetadata,
} from "../../src/events.js";
import { DOC_TEXT, makeWithMessages, MODEL_ANSWER } from "./ingester.test.js";
import { FakeServer } from "./stub.js";

const SUCCESSFUL_KINDS = [
  "document_started",
  "chunk_started",
  "attempt_started",
  "chunk_completed",
  "import_started",
  "import_completed",
  "embedding_refresh_started",
  "embedding_refresh_completed",
];

describe("TaguruIngester on_event", () => {
  it("emits no attempt_failed on a successful first attempt", async () => {
    const events: IngestEvent[] = [];
    const { ingester } = makeWithMessages(new FakeServer(), [new AIMessage(MODEL_ANSWER)], {
      on_event: (event: IngestEvent) => events.push(event),
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });

    expect(outcome.ok).toBe(true);
    expect(events.map((event) => event.kind)).toEqual(SUCCESSFUL_KINDS);

    const documentStarted = events[0]!;
    expect(documentStarted).toMatchObject({
      kind: "document_started",
      source: "docs/aomine.md",
      text_bytes: Buffer.byteLength(DOC_TEXT, "utf-8"),
    });

    const chunkCompleted = events.find((event) => event.kind === "chunk_completed")!;
    expect(chunkCompleted).toMatchObject({
      llm_calls: 1,
      // Raw, pre-merge/dedup proposal counts (MODEL_ANSWER has 3
      // associations, one a duplicate that only gets folded during merge()).
      associations_proposed: 3,
      aliases_proposed: 1,
      questions_proposed: 1,
    });

    const refreshCompleted = events.find(
      (event) => event.kind === "embedding_refresh_completed",
    )! as EmbeddingRefreshCompleted;
    expect(refreshCompleted.configured).toBe(false); // FakeServer defaults to 501
    expect(refreshCompleted.source).toBe("docs/aomine.md");
  });

  it("emits one attempt_failed with provider metadata on a corrective success", async () => {
    const events: IngestEvent[] = [];
    const malformed = new AIMessage({
      content: "not json",
      response_metadata: { done_reason: "length" },
      usage_metadata: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
    });
    const { ingester } = makeWithMessages(
      new FakeServer(),
      [malformed, new AIMessage(MODEL_ANSWER)],
      { on_event: (event: IngestEvent) => events.push(event) },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);

    const failed = events.filter((event) => event.kind === "attempt_failed") as AttemptFailed[];
    expect(failed).toHaveLength(1);
    expect(failed[0]!.attempt).toBe(1);
    expect(failed[0]!.max_attempts).toBe(2);
    expect(failed[0]!.parse_error).not.toBe("");
    expect(failed[0]!.length_limited).toBe(true);
    const metadata = failed[0]!.provider_metadata as ProviderMetadata;
    expect(metadata).not.toBeNull();
    expect(metadata.finish_reason).toBe("length");
    expect(metadata.input_tokens).toBe(10);
    expect(metadata.output_tokens).toBe(20);
    expect(metadata.total_tokens).toBe(30);

    const started = events.filter((event) => event.kind === "attempt_started") as AttemptStarted[];
    expect(started.map((event) => event.attempt)).toEqual([1, 2]);
  });

  it("emits two attempt_failed on a terminal parse failure, with no chunk_completed/import_started after", async () => {
    const events: IngestEvent[] = [];
    const { ingester } = makeWithMessages(
      new FakeServer(),
      [new AIMessage("nope"), new AIMessage("still nope")],
      { on_event: (event: IngestEvent) => events.push(event) },
    );
    await expect(ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow(
      /would not produce the JSON object/,
    );

    const failed = events.filter((event) => event.kind === "attempt_failed") as AttemptFailed[];
    expect(failed.map((event) => event.attempt)).toEqual([1, 2]);
    // No response_metadata on either bad answer, so neither reads as cut
    // off at the provider's output cap.
    expect(failed.some((event) => event.length_limited)).toBe(false);
    // The chunk never completed, so no import/refresh events followed.
    expect(events.some((event) => event.kind === "chunk_completed")).toBe(false);
    expect(events.some((event) => event.kind === "import_started")).toBe(false);
  });

  it("emits a warning when embedding refresh hits a provider error", async () => {
    const server = new FakeServer();
    server.embeddingsRefreshStatus = 502;
    const events: IngestEvent[] = [];
    const { ingester } = makeWithMessages(server, [new AIMessage(MODEL_ANSWER)], {
      on_event: (event: IngestEvent) => events.push(event),
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.embeddings_refresh_warning).not.toBeNull();

    const warned = events.filter(
      (event) => event.kind === "embedding_refresh_warning",
    ) as EmbeddingRefreshWarning[];
    expect(warned).toHaveLength(1);
    expect(warned[0]!.message).toBe(outcome.embeddings_refresh_warning);
    expect(warned[0]!.source).toBe("docs/aomine.md");
    expect(events.some((event) => event.kind === "embedding_refresh_completed")).toBe(false);
  });

  it("reports counts when embedding refresh succeeds", async () => {
    const server = new FakeServer();
    server.embeddingsRefreshStatus = 200;
    server.embeddingsRefreshResult = { embedded: 3, total: 5 };
    const events: IngestEvent[] = [];
    const { ingester } = makeWithMessages(server, [new AIMessage(MODEL_ANSWER)], {
      on_event: (event: IngestEvent) => events.push(event),
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);

    const completed = events.filter(
      (event) => event.kind === "embedding_refresh_completed",
    ) as EmbeddingRefreshCompleted[];
    expect(completed).toHaveLength(1);
    expect(completed[0]!.configured).toBe(true);
    expect(completed[0]!.embedded).toBe(3);
    expect(completed[0]!.total).toBe(5);
    expect(completed[0]!.source).toBe("docs/aomine.md");
  });

  it("carries the right source on each document's refresh events (ingestDocuments)", async () => {
    // The regression this field exists for: ingestDocuments() ingests
    // multiple documents through the same ingester, so without a source on
    // the refresh events a caller can't tell which document just finished.
    const events: IngestEvent[] = [];
    const { ingester } = makeWithMessages(
      new FakeServer(),
      [new AIMessage(MODEL_ANSWER), new AIMessage(MODEL_ANSWER)],
      { on_event: (event: IngestEvent) => events.push(event) },
    );
    const outcomes = await ingester.ingestDocuments([
      new Document({ pageContent: DOC_TEXT, metadata: { source: "docs/aomine.md" } }),
      new Document({ pageContent: DOC_TEXT, metadata: { source: "docs/other.md" } }),
    ]);
    expect(outcomes.map((outcome) => outcome.ok)).toEqual([true, true]);

    const completed = events.filter(
      (event) => event.kind === "embedding_refresh_completed",
    ) as EmbeddingRefreshCompleted[];
    expect(completed.map((event) => event.source)).toEqual(["docs/aomine.md", "docs/other.md"]);
  });

  it("warns but does not break ingest when on_event throws", async () => {
    const server = new FakeServer();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const badCallback = () => {
        throw new Error("callback bug");
      };
      const { ingester } = makeWithMessages(server, [new AIMessage(MODEL_ANSWER)], {
        on_event: badCallback,
      });
      const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
      expect(outcome.ok).toBe(true);
      expect(server.imported).toHaveLength(1);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("callback bug"));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("no callback means no overhead surprises (the default on_event: undefined path)", async () => {
    const { ingester } = makeWithMessages(new FakeServer(), [new AIMessage(MODEL_ANSWER)]);
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
  });

  it("shares the Rust/Python diagnostics key set (attempt_failed and provider_metadata)", async () => {
    // Port of the Python suite's
    // test_attempt_failed_shares_the_rust_diagnostics_key_set /
    // extract.rs's attempt_record_serializes_the_shared_key_set (issue
    // #200, ADR 0001 §10) — the parity anchor on the TypeScript side.
    // AttemptFailed covers only the failure case, so its own `kind`
    // discriminator is excluded here.
    const events: IngestEvent[] = [];
    const malformed = new AIMessage({
      content: "not json",
      response_metadata: { done_reason: "length" },
      usage_metadata: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
    });
    const { ingester } = makeWithMessages(
      new FakeServer(),
      [malformed, new AIMessage(MODEL_ANSWER)],
      { on_event: (event: IngestEvent) => events.push(event) },
    );
    await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });

    const failed = events.find((event) => event.kind === "attempt_failed") as AttemptFailed;
    const fields = new Set(Object.keys(failed).filter((key) => key !== "kind"));
    expect(fields).toEqual(
      new Set([
        "source",
        "chunk_index",
        "attempt",
        "max_attempts",
        "parse_error",
        "elapsed_seconds",
        "provider_metadata",
        "length_limited",
        "stage",
        "validation_issues",
      ]),
    );

    const metadataFields = new Set(Object.keys(failed.provider_metadata as ProviderMetadata));
    expect(metadataFields).toEqual(
      new Set(["finish_reason", "input_tokens", "output_tokens", "total_tokens"]),
    );
  });
});
