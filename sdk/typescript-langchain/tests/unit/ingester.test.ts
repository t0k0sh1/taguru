/** TaguruIngester with a deterministic fake chat model — mirrors the Python suite. */

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessageChunk } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { FakeChatModel, FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, it } from "vitest";

import { MAX_PASSAGE_BYTES } from "../../src/extract.js";
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

/**
 * Exercises the structured_output: true path end to end: unlike
 * FakeListChatModel (which implements bindTools() but ignores it — its own
 * _generate() only ever formats plain text, never tool_calls, and its
 * withStructuredOutput() override ignores includeRaw entirely), this fake
 * echoes pre-programmed tool_calls back through a real BaseChatModel, so
 * TaguruIngester's withStructuredOutput({ includeRaw: true }) pipeline runs
 * for real. _generate() must return an AIMessageChunk, not an AIMessage:
 * withStructuredOutput()'s default extraction step checks
 * AIMessageChunk.isInstance() on the model's output and silently falls back
 * to `parsed: null` otherwise (confirmed by direct probing — a plain
 * AIMessage fails that check even though it is otherwise identical).
 */
class ToolCallingFakeChatModel extends BaseChatModel {
  toolCallArgs: Record<string, unknown>[];
  calls = 0;

  constructor(fields: { toolCallArgs: Record<string, unknown>[] }) {
    super({});
    this.toolCallArgs = fields.toolCallArgs;
  }

  _llmType(): string {
    return "tool-calling-fake";
  }

  bindTools(): this {
    return this;
  }

  async _generate(): Promise<ChatResult> {
    const args = this.toolCallArgs[this.calls]!;
    this.calls += 1;
    const message = new AIMessageChunk({
      content: "",
      tool_calls: [{ name: "ModelOutput", args, id: `call_${this.calls}` }],
    });
    return { generations: [{ text: "", message }] };
  }
}

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

  it("records a real LLM-provider error as a failed outcome, not a rethrow", async () => {
    class ThrowingChatModel extends FakeListChatModel {
      async _generate(): Promise<never> {
        const error = new Error("rate limited");
        error.name = "RateLimitError";
        throw error;
      }
    }

    const outcomes = await new TaguruIngester({
      context: "sake",
      llm: new ThrowingChatModel({ responses: [] }),
      client: new FakeServer().client(),
      questions: 2,
    }).ingestDocuments([{ pageContent: DOC_TEXT, metadata: { source: "docs/aomine.md" } }]);
    expect(outcomes[0]!.ok).toBe(false);
    expect(outcomes[0]!.error).toMatch(/rate limited/);

    await expect(
      new TaguruIngester({
        context: "sake",
        llm: new ThrowingChatModel({ responses: [] }),
        client: new FakeServer().client(),
        questions: 2,
        raise_on_error: true,
      }).ingestDocuments([{ pageContent: DOC_TEXT, metadata: { source: "docs/aomine.md" } }]),
    ).rejects.toThrow(/rate limited/);
  });

  it("skips the passage-size cap when include_passage is false", async () => {
    const bigText = `${DOC_TEXT}\n\n${"a".repeat(MAX_PASSAGE_BYTES)}`;

    const outcome = await make(new FakeServer(), [MODEL_ANSWER], {
      include_passage: false,
      chunk_bytes: MAX_PASSAGE_BYTES + 1024,
    }).ingestText(bigText, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);

    // With include_passage true (the default), the same oversized text is
    // still rejected — the cap is real, just conditional on actually
    // needing it.
    await expect(
      make(new FakeServer(), [MODEL_ANSWER], {
        chunk_bytes: MAX_PASSAGE_BYTES + 1024,
      }).ingestText(bigText, { source: "docs/aomine.md" }),
    ).rejects.toThrow(/passage cap/);
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

  it("defaults structured_output to false, so a model without bindTools still constructs", () => {
    const llm = new FakeChatModel({});
    const ingester = new TaguruIngester({ context: "c", llm, client: new FakeServer().client() });
    expect(ingester.structured_output).toBe(false);
  });

  it("structured_output: true raises at construction when the model cannot bind tools", () => {
    const llm = new FakeChatModel({});
    expect(
      () =>
        new TaguruIngester({
          context: "c",
          llm,
          client: new FakeServer().client(),
          structured_output: true,
        }),
    ).toThrow(/bindTools/);
  });

  it("structured_output: true drives ingestText through withStructuredOutput", async () => {
    const server = new FakeServer();
    const llm = new ToolCallingFakeChatModel({ toolCallArgs: [JSON.parse(MODEL_ANSWER)] });
    const outcome = await new TaguruIngester({
      context: "sake",
      llm,
      client: server.client(),
      questions: 2,
      structured_output: true,
    }).ingestText(DOC_TEXT, { source: "docs/aomine.md" });

    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(1);
    expect(outcome.duplicates_dropped).toBe(1);
    expect(outcome.invalid_dropped).toBe(1);
    expect(outcome.associations).toBe(2);
    expect(outcome.aliases).toBe(1);
  });
});
