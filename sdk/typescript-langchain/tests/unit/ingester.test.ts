/** TaguruIngester with a deterministic fake chat model — mirrors the Python suite. */

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, it } from "vitest";

import { TaguruIngester } from "../../src/ingest.js";
import { FakeServer } from "./stub.js";

const MODEL_ANSWER = JSON.stringify({
  associations: [
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, paragraph: 1 },
    { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, paragraph: 0 },
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }, // duplicate
    { subject: "", label: "壊", object: "れ", weight: 1.0 }, // invalid
  ],
  aliases: [{ alias: "Aomine", canonical: "青嶺酒造", kind: "concept" }],
  questions: [{ paragraph: 1, question: "杜氏は誰?" }],
});

const DOC_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬である。";

const make = (server: FakeServer, responses: string[], fields: Record<string, unknown> = {}) =>
  new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses }),
    client: server.client(),
    questions: 2,
    ...fields,
  });

describe("TaguruIngester", () => {
  it("builds the batch and imports it", async () => {
    const server = new FakeServer();
    const outcome = await make(server, [MODEL_ANSWER]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });

    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(1);
    expect(outcome.chunks).toBe(1);
    expect(outcome.duplicates_dropped).toBe(1);
    expect(outcome.invalid_dropped).toBe(1);
    expect(outcome.associations).toBe(2);
    expect(outcome.aliases).toBe(1);
    expect(outcome.passage_stored).toBe(true);
    expect(outcome.embeddings_refresh_warning).toBeNull(); // 501 stays silent

    expect(server.imported).toHaveLength(1);
    const lines = server.imported[0]!.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines[0].taguru_batch).toBe(1);
    expect(lines[0].create).toBeUndefined();
    expect(lines[1]).toEqual({ passage: DOC_TEXT }); // verbatim, unchunked
    expect(lines[2]).toEqual({ paragraph: 1, question: "杜氏は誰?" });
    expect(lines[3].paragraph).toBe(1);
    expect(lines[5]).toEqual({ alias: "Aomine", canonical: "青嶺酒造", kind: "concept" });
  });

  it("dry run renders but never sends", async () => {
    const server = new FakeServer();
    const outcome = await make(server, [MODEL_ANSWER]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
      dry_run: true,
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.ndjson).toContain('"taguru_batch"');
    expect(server.imported).toEqual([]);
    expect(outcome.associations).toBe(0);
  });

  it("gives one corrective turn on a malformed answer", async () => {
    const server = new FakeServer();
    const outcome = await make(server, ["I cannot answer in JSON!", MODEL_ANSWER]).ingestText(
      DOC_TEXT,
      { source: "docs/aomine.md" },
    );
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(2);

    await expect(
      make(new FakeServer(), ["nope", "still nope"]).ingestText(DOC_TEXT, {
        source: "docs/aomine.md",
      }),
    ).rejects.toThrow(/would not produce the JSON object/);
  });

  it("stamps a create block when asked", async () => {
    const server = new FakeServer();
    await make(server, [MODEL_ANSWER], {
      create_context: true,
      context_description: "酒蔵の知識",
    }).ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    const header = JSON.parse(server.imported[0]!.split("\n", 1)[0]!);
    expect(header.create).toEqual({ description: "酒蔵の知識" });
  });

  it("requires a source id on documents", async () => {
    const server = new FakeServer();
    const outcomes = await make(server, [MODEL_ANSWER, MODEL_ANSWER]).ingestDocuments([
      { pageContent: DOC_TEXT, metadata: {} },
      { pageContent: DOC_TEXT, metadata: { source: "docs/aomine.md" } },
    ]);
    expect(outcomes[0]!.ok).toBe(false);
    expect(outcomes[0]!.error).toMatch(/source/);
    expect(outcomes[1]!.ok).toBe(true);

    await expect(
      make(new FakeServer(), [MODEL_ANSWER], { raise_on_error: true }).ingestDocuments([
        { pageContent: DOC_TEXT, metadata: {} },
      ]),
    ).rejects.toThrow(/source/);
  });

  it("rejects bad construction", () => {
    const llm = new FakeListChatModel({ responses: ["{}"] });
    const client = new FakeServer().client();
    expect(
      () => new TaguruIngester({ context: "c", llm, client, create_context: true }),
    ).toThrow(/context_description/);
    expect(() => new TaguruIngester({ context: "c", llm, client, questions: 99 })).toThrow(
      /questions/,
    );
  });
});
