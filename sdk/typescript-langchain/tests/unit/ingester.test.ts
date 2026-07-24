/** TaguruIngester with a deterministic fake chat model — mirrors the Python suite. */

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { FakeChatModel, FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, it } from "vitest";

import { MAX_PASSAGE_BYTES } from "../../src/extract.js";
import type { AttemptFailed, AttemptStarted, IngestEvent } from "../../src/events.js";
import { TaguruIngester } from "../../src/ingest.js";
import { FakeServer } from "./stub.js";

export const MODEL_ANSWER = JSON.stringify({
  associations: [
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, paragraph: 1 },
    { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, paragraph: 0 },
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }, // duplicate
  ],
  aliases: [{ alias: "Aomine", canonical: "青嶺酒造", kind: "concept" }],
  questions: [{ paragraph: 1, question: "杜氏は誰?" }],
});

// The business-rule-invalid item issue #181's strict default now rejects on
// sight (empty subject) rather than letting merge() silently drop it — split
// out of MODEL_ANSWER so most fixtures stay a clean one-call success; reused
// by the lossy-mode and corrective-turn tests below.
const INVALID_ASSOCIATION = { subject: "", label: "壊", object: "れ", weight: 1.0 };

export const DOC_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬である。";

// chunk_bytes: 40 splits DOC_TEXT into exactly its two paragraphs (one chunk
// each) — the minimum needed to exercise cross-chunk alias validation (issue
// #181 Stage 2), which only ever fires with more than one chunk.
const CROSS_CHUNK_BYTES = 40;

// Chunk 1 (paragraph 0) introduces the concept name "1907年" as an
// association object.
const CHUNK1_ANSWER = JSON.stringify({
  associations: [{ subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0 }],
  aliases: [],
});
// Chunk 2 (paragraph 1) passes Stage 1 alone (every alias field present, not
// a self-alias) — the shadowing is only visible once chunk 1's "1907年" is
// in the merged name set, so only Stage 2's crossOutputIssues catches it.
const CHUNK2_SHADOWING_ANSWER = JSON.stringify({
  associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }],
  aliases: [{ alias: "1907年", canonical: "青嶺酒造", kind: "concept" }],
});
const CHUNK2_CORRECTED_ANSWER = JSON.stringify({
  associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }],
  aliases: [{ alias: "Sake", canonical: "青嶺酒造", kind: "concept" }],
});

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
export class RecordingFakeMessagesChatModel extends BaseChatModel {
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

export const makeWithMessages = (
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
    expect(outcome.invalid_dropped).toBe(0);
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
    expect(outcome.invalid_dropped).toBe(0);
    expect(outcome.associations).toBe(2);
    expect(outcome.aliases).toBe(1);
  });
});

describe("TaguruIngester (issue #181: lossless JSON repair and path-specific corrective retry)", () => {
  it("repairs a trailing-comma answer without an extra LLM call or item change", async () => {
    const answer =
      '{"associations": [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", ' +
      '"weight": 1.0},], "aliases": [], "questions": []}';
    const outcome = await make(new FakeServer(), [answer]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(1);
    expect(outcome.correction_attempts).toBe(0);
    expect(outcome.lossless_repairs).toEqual(["trailing_comma"]);
    // fake_server's /import stub echoes fixed counts — check the actual
    // rendered batch instead of outcome.associations.
    const facts = outcome
      .ndjson!.trim()
      .split("\n")
      .map((line) => JSON.parse(line))
      .filter((line) => "subject" in line);
    expect(facts).toEqual([{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }]);
  });

  it("gives a wrong-typed field a path-specific corrective turn", async () => {
    const badWeightAnswer = JSON.stringify({
      associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: "strong" }],
      aliases: [],
    });
    const events: IngestEvent[] = [];
    const { ingester, llm } = makeWithMessages(
      new FakeServer(),
      [new AIMessage(badWeightAnswer), new AIMessage(MODEL_ANSWER)],
      { on_event: (event: IngestEvent) => events.push(event) },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(2);
    expect(outcome.correction_attempts).toBe(1);

    const corrective = llm.seenPrompts[1]!.at(-1)!.content as string;
    expect(corrective).toContain("That was valid JSON but not a valid extraction (1 issue(s)):");
    expect(corrective).toContain("associations[0].weight: expected finite non-zero number");
    expect(corrective).toContain("keep every item");

    const failed = events.find((event) => event.kind === "attempt_failed") as AttemptFailed;
    expect(failed.validation_issues).toEqual([
      'associations[0].weight: expected finite non-zero number, got string "strong"',
    ]);
  });

  it("fails the source without import when a second answer is still invalid", async () => {
    const badAnswer = JSON.stringify({ associations: [INVALID_ASSOCIATION], aliases: [] });
    const server = new FakeServer();
    await expect(make(server, [badAnswer, badAnswer]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    })).rejects.toThrow(/invalid item/);
    expect(server.imported).toEqual([]);
  });

  it("never imports a length-limited answer even when its content happens to parse", async () => {
    // ADR 0001: a `length`-terminated answer is length-limited even when
    // its own content happens to parse cleanly — a valid prefix of a
    // cut-off extraction must never be imported as if complete.
    const truncatedButParseable = new AIMessage({
      content: MODEL_ANSWER,
      response_metadata: { done_reason: "length" },
    });
    const { ingester, llm } = makeWithMessages(new FakeServer(), [
      truncatedButParseable,
      new AIMessage(MODEL_ANSWER),
    ]);
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(2);
    const corrective = llm.seenPrompts[1]!.at(-1)!.content as string;
    expect(corrective).toContain("SHORTER");

    const server = new FakeServer();
    await expect(
      makeWithMessages(server, [truncatedButParseable], { max_attempts: 1 }).ingester.ingestText(
        DOC_TEXT,
        { source: "docs/aomine.md" },
      ),
    ).rejects.toThrow(/would not produce the JSON object/);
    expect(server.imported).toEqual([]);
  });

  it("gives an empty answer one corrective then the named diagnosis", async () => {
    const server = new FakeServer();
    const { ingester, llm } = makeWithMessages(
      server,
      [new AIMessage(""), new AIMessage("")],
      { max_attempts: 5 },
    );
    await expect(ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow(
      /the answer was empty/,
    );
    // Bounded to exactly one corrective regardless of how high max_attempts is.
    expect(llm.seenPrompts).toHaveLength(2);

    const { ingester: recovers } = makeWithMessages(
      new FakeServer(),
      [new AIMessage(""), new AIMessage(MODEL_ANSWER)],
      { max_attempts: 5 },
    );
    const outcome = await recovers.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(2);
  });

  it("treats a refusal finish reason as terminal, without a corrective turn", async () => {
    const refused = new AIMessage({
      content: "",
      response_metadata: { done_reason: "content_filter" },
    });
    const { ingester, llm } = makeWithMessages(new FakeServer(), [
      refused,
      new AIMessage(MODEL_ANSWER),
    ]);
    await expect(ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow(
      /policy refusal is terminal/,
    );
    expect(llm.seenPrompts).toHaveLength(1);
  });

  it("gives cross-chunk alias issues one targeted corrective turn", async () => {
    const server = new FakeServer();
    const events: IngestEvent[] = [];
    const { ingester, llm } = makeWithMessages(
      server,
      [
        new AIMessage(CHUNK1_ANSWER),
        new AIMessage(CHUNK2_SHADOWING_ANSWER),
        new AIMessage(CHUNK2_CORRECTED_ANSWER),
      ],
      { chunk_bytes: CROSS_CHUNK_BYTES, on_event: (event: IngestEvent) => events.push(event) },
    );
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks).toBe(2);
    expect(outcome.llm_calls).toBe(3);
    expect(outcome.correction_attempts).toBe(1);

    const correctivePrompt = llm.seenPrompts.at(-1)!;
    expect(correctivePrompt[1]!.content as string).toContain("杜氏は高瀬"); // chunk 2's OWN user turn, replayed
    expect(correctivePrompt.at(-1)!.content as string).toContain(
      "names something the associations already contain",
    );

    const crossChunkStarted = events.filter(
      (event): event is AttemptStarted => event.kind === "attempt_started" && event.stage === "cross_chunk",
    );
    expect(crossChunkStarted).toHaveLength(1);
    expect(crossChunkStarted[0]!.chunk_index).toBe(1);
  });

  it("fails without import when a cross-chunk correction still leaves the issue (bounded re-check)", async () => {
    // The "corrective" answer repeats the exact same shadowing alias — it
    // still passes Stage 1 alone, so only the bounded re-check (not a
    // second round) catches that the correction never actually fixed it.
    const server = new FakeServer();
    await expect(
      makeWithMessages(
        server,
        [
          new AIMessage(CHUNK1_ANSWER),
          new AIMessage(CHUNK2_SHADOWING_ANSWER),
          new AIMessage(CHUNK2_SHADOWING_ANSWER),
        ],
        { chunk_bytes: CROSS_CHUNK_BYTES },
      ).ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" }),
    ).rejects.toThrow(/still has 1 cross-chunk alias issue\(s\) after correction/);
    expect(server.imported).toEqual([]);
  });

  it("fails without import when Stage 2's own corrective reply is structurally invalid", async () => {
    const chunk2InvalidCorrection = JSON.stringify({
      associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: "strong" }],
      aliases: [{ alias: "Sake", canonical: "青嶺酒造", kind: "concept" }],
    });
    const server = new FakeServer();
    await expect(
      makeWithMessages(
        server,
        [
          new AIMessage(CHUNK1_ANSWER),
          new AIMessage(CHUNK2_SHADOWING_ANSWER),
          new AIMessage(chunk2InvalidCorrection),
        ],
        { chunk_bytes: CROSS_CHUNK_BYTES },
      ).ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" }),
    ).rejects.toThrow(/cross-chunk alias correction still left/);
    expect(server.imported).toEqual([]);
  });

  it("lossy: true restores drop-and-proceed", async () => {
    const answer = JSON.stringify({ associations: [INVALID_ASSOCIATION], aliases: [] });

    const lossyOutcome = await make(new FakeServer(), [answer], { lossy: true }).ingestText(
      DOC_TEXT,
      { source: "docs/aomine.md" },
    );
    expect(lossyOutcome.ok).toBe(true);
    expect(lossyOutcome.llm_calls).toBe(1);
    expect(lossyOutcome.invalid_dropped).toBe(1);

    await expect(
      make(new FakeServer(), [answer, answer]).ingestText(DOC_TEXT, { source: "docs/aomine.md" }),
    ).rejects.toThrow(/invalid item/);
  });

  it("leaves the existing source untouched after a failed re-ingest", async () => {
    const server = new FakeServer();
    const outcome = await make(server, [MODEL_ANSWER]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(server.imported).toHaveLength(1);

    const badAnswer = JSON.stringify({ associations: [INVALID_ASSOCIATION], aliases: [] });
    await expect(
      make(server, [badAnswer, badAnswer]).ingestText(DOC_TEXT, { source: "docs/aomine.md" }),
    ).rejects.toThrow(/invalid item/);
    // The failed re-ingest never called /import — the prior batch stands.
    expect(server.imported).toHaveLength(1);
  });

  it("gives structured-output invalid args a validation corrective turn", async () => {
    const badArgs = {
      associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: "strong" }],
      aliases: [],
    };
    const goodArgs = JSON.parse(MODEL_ANSWER) as Record<string, unknown>;
    const llm = new ToolCallingFakeChatModel({ toolCallArgs: [badArgs, goodArgs] });
    const events: IngestEvent[] = [];
    const outcome = await new TaguruIngester({
      context: "sake",
      llm,
      client: new FakeServer().client(),
      questions: 2,
      structured_output: true,
      on_event: (event: IngestEvent) => events.push(event),
    }).ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(2);
    expect(outcome.correction_attempts).toBe(1);

    const failed = events.find((event) => event.kind === "attempt_failed") as AttemptFailed;
    expect(failed.validation_issues).toEqual([
      'associations[0].weight: expected finite non-zero number, got string "strong"',
    ]);
  });
});
