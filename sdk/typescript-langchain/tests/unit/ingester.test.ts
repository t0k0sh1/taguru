/** TaguruIngester with a deterministic fake chat model — mirrors the Python suite. */

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
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

/**
 * Returns pre-scripted AIMessage responses — so a test can attach
 * response_metadata, which FakeListChatModel's plain-string responses
 * cannot carry — and records every message array it was invoked with.
 * The twin of the Python suite's RecordingFakeMessagesListChatModel.
 */
class RecordingFakeMessagesChatModel extends BaseChatModel {
  responses: AIMessage[];
  calls = 0;
  seenPrompts: BaseMessage[][] = [];

  constructor(fields: { responses: AIMessage[] }) {
    super({});
    this.responses = fields.responses;
  }

  _llmType(): string {
    return "recording-fake-messages";
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.seenPrompts.push(messages);
    const message = this.responses[this.calls]!;
    this.calls += 1;
    const text = typeof message.content === "string" ? message.content : "";
    return { generations: [{ text, message }] };
  }
}

const makeWithMessages = (
  server: FakeServer,
  responses: AIMessage[],
  fields: Record<string, unknown> = {},
): { ingester: TaguruIngester; llm: RecordingFakeMessagesChatModel } => {
  const llm = new RecordingFakeMessagesChatModel({ responses });
  const ingester = new TaguruIngester({
    context: "sake",
    llm,
    client: server.client(),
    questions: 2,
    ...fields,
  });
  return { ingester, llm };
};

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

  it("rejects bad fact_budget, max_attempts, and corrective_context_bytes", () => {
    const llm = new FakeListChatModel({ responses: ["{}"] });
    const client = new FakeServer().client();
    expect(() => new TaguruIngester({ context: "c", llm, client, fact_budget: 0 })).toThrow(
      /fact_budget/,
    );
    expect(() => new TaguruIngester({ context: "c", llm, client, max_attempts: 0 })).toThrow(
      /max_attempts must be between 1 and 10/,
    );
    expect(() => new TaguruIngester({ context: "c", llm, client, max_attempts: 11 })).toThrow(
      /max_attempts must be between 1 and 10/,
    );
    expect(
      () => new TaguruIngester({ context: "c", llm, client, corrective_context_bytes: -1 }),
    ).toThrow(/corrective_context_bytes/);
  });

  it("folds fact_budget into the system prompt", async () => {
    const { ingester, llm } = makeWithMessages(new FakeServer(), [new AIMessage(MODEL_ANSWER)], {
      fact_budget: 3,
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    const system = llm.seenPrompts[0]![0]!.content as string;
    expect(system).toContain("Keep this answer to at most 3 association(s) total");
  });

  it("max_attempts extends corrective retries past the default, rebuilding not accumulating", async () => {
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [
        new AIMessage("still not json"),
        new AIMessage("nope, still not"),
        new AIMessage(MODEL_ANSWER),
      ],
      { max_attempts: 3 },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(3);
    expect(llm.seenPrompts).toHaveLength(3);
    // Every corrective turn rebuilds from the base: the third request must
    // still be 4 messages (system, user, latest bad answer, corrective) —
    // not 6 — proving the first bad answer was dropped, not accumulated.
    expect(llm.seenPrompts[2]).toHaveLength(4);
  });

  it("max_attempts of 1 skips the corrective turn", async () => {
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [new AIMessage("not json at all")],
      { max_attempts: 1 },
    );
    await expect(ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow(
      /would not produce the JSON object/,
    );
    expect(llm.seenPrompts).toHaveLength(1);
  });

  it("corrective_context_bytes caps the replayed bad answer", async () => {
    const badAnswer = "not json at all, definitely not a JSON object";
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [new AIMessage(badAnswer), new AIMessage(MODEL_ANSWER)],
      { corrective_context_bytes: 10 },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    const replayed = llm.seenPrompts[1]!.at(-2)!.content as string;
    expect(replayed).toContain("[truncated to 10 bytes]");
    expect(replayed).not.toContain(badAnswer);
  });

  it("corrective_context_bytes of 0 omits the bad answer", async () => {
    const badAnswer = "not json at all, definitely not a JSON object";
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [new AIMessage(badAnswer), new AIMessage(MODEL_ANSWER)],
      { corrective_context_bytes: 0 },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    const replayed = llm.seenPrompts[1]!.at(-2)!.content as string;
    expect(replayed).toBe("[omitted: not the requested JSON object]");
  });

  it("a length-limited bad answer asks for SHORTER and names the fact budget", async () => {
    const malformed = new AIMessage({
      content: "not json, and huge",
      response_metadata: { done_reason: "length" },
    });
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [malformed, new AIMessage(MODEL_ANSWER)],
      { fact_budget: 4 },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    const corrective = llm.seenPrompts[1]!.at(-1)!.content as string;
    expect(corrective).toContain("SHORTER");
    expect(corrective).toContain("cut off at the output limit");
    expect(corrective).toContain("Keep it to at most 4 association(s) total.");
    expect(corrective).not.toContain("Answer again with only the JSON object.");
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
