/**
 * TaguruIngester checkpoint/resume (issue #212, the TypeScript twin of
 * #211/#210): durable per-chunk checkpoints, cooperative stop, and resume.
 * Mirrors the Python suite's test_checkpoints.py (dropping its four async
 * twins — this SDK is async-only, so there is nothing to twin).
 */

import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CheckpointStore } from "../../src/checkpoints.js";
import { TaguruIngester, type TaguruIngesterFields } from "../../src/ingest.js";
import { FakeServer } from "./stub.js";
import {
  CHUNK1_ANSWER,
  CHUNK2_CORRECTED_ANSWER,
  CHUNK2_SHADOWING_ANSWER,
  CROSS_CHUNK_BYTES,
  DOC_TEXT,
  MODEL_ANSWER,
  RecordingFakeMessagesChatModel,
  ToolCallingFakeChatModel,
} from "./ingester.test.js";

// Paragraph 2 gets one extra ASCII character relative to DOC_TEXT — still
// well inside CROSS_CHUNK_BYTES's 40-byte cap for either paragraph, so it
// stays a clean 2-chunk document, but its content hash (and therefore its
// checkpoint fingerprint) differs.
const CHANGED_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬xである。";

/** In-memory CheckpointStore recording every call in order, seedable with a
 * prior run's bytes — the double every checkpoint/resume test below is
 * built on. */
export class RecordingCheckpointStore implements CheckpointStore {
  state: Map<string, Uint8Array>;
  log: [string, string][] = [];

  constructor(seed?: Map<string, Uint8Array>) {
    this.state = new Map(seed ?? []);
  }

  async load(source: string): Promise<Uint8Array | null> {
    this.log.push(["load", source]);
    return this.state.get(source) ?? null;
  }

  async save(source: string, data: Uint8Array): Promise<void> {
    this.log.push(["save", source]);
    this.state.set(source, data);
  }

  async delete(source: string): Promise<void> {
    this.log.push(["delete", source]);
    this.state.delete(source);
  }
}

class FailingLoadStore extends RecordingCheckpointStore {
  override async load(source: string): Promise<Uint8Array | null> {
    this.log.push(["load", source]);
    throw new Error("simulated load failure");
  }
}

class FailingSaveStore extends RecordingCheckpointStore {
  override async save(source: string, _data: Uint8Array): Promise<void> {
    this.log.push(["save", source]);
    throw new Error("simulated save failure");
  }
}

class FailingDeleteStore extends RecordingCheckpointStore {
  override async delete(source: string): Promise<void> {
    this.log.push(["delete", source]);
    throw new Error("simulated delete failure");
  }
}

/**
 * Answers normally up to (but not including) call number `killAtCall`, then
 * throws — simulating a process kill/crash mid-document (an arbitrary
 * uncaught exception escaping ingestText with no cleanup, unlike the
 * ordinary attempt-exhaustion failure path).
 */
class KillPartwayThroughChatModel extends RecordingFakeMessagesChatModel {
  private readonly killAtCall: number;

  constructor(fields: { responses: AIMessage[]; killAtCall: number }) {
    super({ responses: fields.responses });
    this.killAtCall = fields.killAtCall;
  }

  override async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    if (this.calls >= this.killAtCall) {
      throw new Error("simulated process kill");
    }
    return super._generate(messages);
  }
}

/** A fake chat model exposing a `model` attribute, for deriveModelIdentity. */
class NamedFakeChatModel extends RecordingFakeMessagesChatModel {
  readonly model = "demo-v1";
}

/**
 * Full-control ingester factory for checkpoint tests: unlike the plain
 * `make`/`makeWithMessages` helpers (which hardcode `context`/`questions`),
 * every field can be overridden — needed to exercise fingerprint mismatches
 * on fields those fixate. String responses are wrapped in `AIMessage`;
 * pass an `AIMessage` directly for cases needing response_metadata.
 */
function buildIngester(
  server: FakeServer,
  responses: (string | AIMessage)[],
  overrides: Partial<TaguruIngesterFields> = {},
): { ingester: TaguruIngester; llm: RecordingFakeMessagesChatModel } {
  const llm = new RecordingFakeMessagesChatModel({
    responses: responses.map((r) => (typeof r === "string" ? new AIMessage(r) : r)),
  });
  const ingester = new TaguruIngester({
    context: "sake",
    llm,
    client: server.client(),
    questions: 2,
    checkpoint_model_id: "fake-model",
    ...overrides,
  });
  return { ingester, llm };
}

export { buildIngester };

function unitCount(store: RecordingCheckpointStore, source: string): number {
  const data = store.state.get(source);
  if (data === undefined) {
    return 0;
  }
  return Object.keys(JSON.parse(new TextDecoder().decode(data)).units as Record<string, unknown>)
    .length;
}

/**
 * Runs a 2-chunk document where chunk 0 succeeds and chunk 1 never produces
 * valid JSON, leaving exactly one checkpointed unit behind. Returns the
 * source id.
 */
async function seedFailedRun(
  server: FakeServer,
  store: CheckpointStore,
  overrides: Partial<TaguruIngesterFields> & { source?: string } = {},
): Promise<string> {
  const { source = "docs/aomine.md", ...rest } = overrides;
  const { ingester } = buildIngester(server, [CHUNK1_ANSWER, "not json", "not json"], {
    checkpoint_store: store,
    chunk_bytes: CROSS_CHUNK_BYTES,
    ...rest,
  });
  await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow();
  return source;
}

afterEach(() => {
  vi.restoreAllMocks();
});

// -- reuse across a failed-then-resumed document ------------------------------

describe("checkpoint reuse", () => {
  it("reuses a completed chunk after a failed document without recalling the model", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = await seedFailedRun(server, store);
    expect(unitCount(store, source)).toBe(1);
    expect(server.imported).toEqual([]);

    const { ingester: ingester2, llm: llm2 } = buildIngester(server, [CHUNK2_CORRECTED_ANSWER], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    const outcome = await ingester2.ingestText(DOC_TEXT, { source });

    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(1);
    expect(llm2.seenPrompts.length).toBe(1);
    // The one real call was for chunk index 1 (the second, never-cached
    // chunk) — chunk 0 was never re-asked.
    const humanContents = llm2.seenPrompts[0]!.filter((m) => m.getType() === "human").map(
      (m) => m.content,
    );
    expect(humanContents.some((c) => String(c).includes("part 2 of 2"))).toBe(true);
    expect(server.imported.length).toBe(1);
    expect(unitCount(store, source)).toBe(0); // cleared once the batch landed
  });

  it("no checkpoint store means no reuse and unchanged behavior", async () => {
    const server = new FakeServer();
    const { ingester, llm } = buildIngester(server, [CHUNK1_ANSWER, "not json", "not json"], {
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: undefined,
      checkpoint_model_id: undefined,
    });
    expect(ingester.checkpoint_store).toBeUndefined();
    await expect(ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow();
    expect(llm.seenPrompts.length).toBe(3);

    const { ingester: ingester2, llm: llm2 } = buildIngester(
      server,
      [CHUNK1_ANSWER, "not json", "not json"],
      { chunk_bytes: CROSS_CHUNK_BYTES, checkpoint_store: undefined, checkpoint_model_id: undefined },
    );
    await expect(ingester2.ingestText(DOC_TEXT, { source: "docs/aomine.md" })).rejects.toThrow();
    // No reduction whatsoever — nothing was ever cached without a store.
    expect(llm2.seenPrompts.length).toBe(3);
  });

  it("a changed fact_budget invalidates checkpoints even though content is unchanged", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = await seedFailedRun(server, store);
    expect(unitCount(store, source)).toBe(1);

    const { ingester: ingester2, llm: llm2 } = buildIngester(
      server,
      [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
      { checkpoint_store: store, chunk_bytes: CROSS_CHUNK_BYTES, fact_budget: 3 },
    );
    const outcome = await ingester2.ingestText(DOC_TEXT, { source });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(0);
    expect(llm2.seenPrompts.length).toBe(2); // both chunks re-extracted
  });

  it("resumes a killed multi-chunk document without recalling completed chunks", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const killer = new KillPartwayThroughChatModel({
      responses: [new AIMessage(CHUNK1_ANSWER)],
      killAtCall: 1,
    });
    const ingester = new TaguruIngester({
      context: "sake",
      llm: killer,
      client: server.client(),
      questions: 2,
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
      checkpoint_model_id: "fake-model",
    });
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow(
      "simulated process kill",
    );
    expect(unitCount(store, source)).toBe(1);
    expect(server.imported).toEqual([]);

    const { ingester: ingester2, llm: llm2 } = buildIngester(server, [CHUNK2_CORRECTED_ANSWER], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    const outcome = await ingester2.ingestText(DOC_TEXT, { source });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(1);
    expect(llm2.seenPrompts.length).toBe(1);
    expect(server.imported.length).toBe(1);
  });
});

// -- cooperative stop ----------------------------------------------------------

describe("cooperative stop", () => {
  it("stops between chunks, saves checkpoints, and skips import", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    let calls = 0;
    const shouldStop = () => {
      calls += 1;
      return calls > 1; // let chunk 0 run, stop before chunk 1
    };

    const { ingester, llm } = buildIngester(server, [CHUNK1_ANSWER], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source, should_stop: shouldStop });

    expect(outcome.interrupted).toBe(true);
    expect(outcome.ok).toBe(false);
    expect(outcome.ndjson).toBeNull();
    expect(llm.seenPrompts.length).toBe(1);
    expect(server.imported).toEqual([]);
    expect(unitCount(store, source)).toBe(1);

    const { ingester: ingester2, llm: llm2 } = buildIngester(server, [CHUNK2_CORRECTED_ANSWER], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    const outcome2 = await ingester2.ingestText(DOC_TEXT, { source });
    expect(outcome2.ok).toBe(true);
    expect(outcome2.chunks_reused).toBe(1);
    expect(llm2.seenPrompts.length).toBe(1);
    expect(server.imported.length).toBe(1);
    expect(unitCount(store, source)).toBe(0);
  });

  it("an aborted AbortSignal stops between chunks", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const controller = new AbortController();

    const { ingester, llm } = buildIngester(server, [CHUNK1_ANSWER], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
      on_event: (event) => {
        if (event.kind === "chunk_completed") {
          controller.abort(); // stop after chunk 0 completes, before chunk 1
        }
      },
    });
    const outcome = await ingester.ingestText(DOC_TEXT, {
      source,
      should_stop: controller.signal,
    });

    expect(outcome.interrupted).toBe(true);
    expect(outcome.ok).toBe(false);
    expect(llm.seenPrompts.length).toBe(1);
    expect(server.imported).toEqual([]);
    expect(unitCount(store, source)).toBe(1);
  });

  it("stop before the first chunk makes no model calls", async () => {
    const server = new FakeServer();
    const events: { kind: string }[] = [];
    const { ingester, llm } = buildIngester(server, [MODEL_ANSWER], {
      checkpoint_store: undefined,
      checkpoint_model_id: undefined,
      on_event: (event) => events.push(event),
    });
    const outcome = await ingester.ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
      should_stop: () => true,
    });
    expect(outcome.interrupted).toBe(true);
    expect(outcome.ok).toBe(false);
    expect(llm.seenPrompts).toEqual([]);
    expect(events.map((e) => e.kind)).toEqual(["document_started"]);
  });

  it("ingestDocuments stops between documents and reports only attempted ones", async () => {
    const server = new FakeServer();
    const { ingester, llm } = buildIngester(server, [MODEL_ANSWER, MODEL_ANSWER], {
      checkpoint_store: undefined,
      checkpoint_model_id: undefined,
    });
    let calls = 0;
    const shouldStop = () => {
      calls += 1;
      return calls > 2; // doc1's own document+chunk checks, then stop before doc2
    };

    const outcomes = await ingester.ingestDocuments(
      [
        { pageContent: DOC_TEXT, metadata: { source: "docs/a.md" } },
        { pageContent: DOC_TEXT, metadata: { source: "docs/b.md" } },
      ],
      { should_stop: shouldStop },
    );
    expect(outcomes.length).toBe(1);
    expect(outcomes[0]!.ok).toBe(true);
    expect(outcomes[0]!.source).toBe("docs/a.md");
    expect(llm.seenPrompts.length).toBe(1);
  });

  it("ingestDocuments interruption bypasses raise_on_error", async () => {
    const server = new FakeServer();
    let calls = 0;
    const shouldStop = () => {
      calls += 1;
      return calls > 5;
    };
    const { ingester } = buildIngester(
      server,
      [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER, CHUNK1_ANSWER],
      { chunk_bytes: CROSS_CHUNK_BYTES, raise_on_error: true, checkpoint_store: undefined, checkpoint_model_id: undefined },
    );
    const outcomes = await ingester.ingestDocuments(
      [
        { pageContent: DOC_TEXT, metadata: { source: "docs/a.md" } },
        { pageContent: DOC_TEXT, metadata: { source: "docs/b.md" } },
      ],
      { should_stop: shouldStop },
    );
    // No exception, despite raise_on_error: true — an interruption is not a
    // failure, so it never enters the catch/raise_on_error path.
    expect(outcomes.length).toBe(2);
    expect(outcomes[0]!.ok).toBe(true);
    expect(outcomes[1]!.interrupted).toBe(true);
    expect(outcomes[1]!.ok).toBe(false);
  });
});

// -- dry-run interaction --------------------------------------------------------

it("dry run records checkpoints but never deletes them", async () => {
  const server = new FakeServer();
  const store = new RecordingCheckpointStore();
  const source = "docs/aomine.md";
  const { ingester } = buildIngester(server, [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER], {
    checkpoint_store: store,
    chunk_bytes: CROSS_CHUNK_BYTES,
  });
  const outcome = await ingester.ingestText(DOC_TEXT, { source, dry_run: true });
  expect(outcome.ok).toBe(true);
  expect(server.imported).toEqual([]);
  expect(unitCount(store, source)).toBe(2);

  // A second dry run reuses both chunks — real model work already happened
  // once, and dry-run recording is what makes that not wasted.
  const { ingester: ingester2, llm: llm2 } = buildIngester(server, [], {
    checkpoint_store: store,
    chunk_bytes: CROSS_CHUNK_BYTES,
  });
  const outcome2 = await ingester2.ingestText(DOC_TEXT, { source, dry_run: true });
  expect(outcome2.ok).toBe(true);
  expect(outcome2.chunks_reused).toBe(2);
  expect(llm2.seenPrompts).toEqual([]);
  expect(server.imported).toEqual([]);
  expect(unitCount(store, source)).toBe(2); // still not deleted

  // A real (non-dry) run also reuses both chunks, then finally clears the
  // checkpoint once the batch actually lands.
  const { ingester: ingester3, llm: llm3 } = buildIngester(server, [], {
    checkpoint_store: store,
    chunk_bytes: CROSS_CHUNK_BYTES,
  });
  const outcome3 = await ingester3.ingestText(DOC_TEXT, { source });
  expect(outcome3.ok).toBe(true);
  expect(outcome3.chunks_reused).toBe(2);
  expect(llm3.seenPrompts).toEqual([]);
  expect(server.imported.length).toBe(1);
  expect(unitCount(store, source)).toBe(0);
});

// -- fingerprint gate and on-disk degradation -----------------------------------

describe("fingerprint gate", () => {
  const FINGERPRINT_MISMATCH_CASES: [string, Partial<TaguruIngesterFields>, string | null][] = [
    ["content_changed", {}, CHANGED_TEXT],
    ["questions_changed", { questions: 1 }, null],
    ["include_passage_changed", { include_passage: false }, null],
    ["lossy_changed", { lossy: true }, null],
    ["context_changed", { context: "tea" }, null],
    [
      "description_changed",
      { create_context: true, context_description: "蔵元一式" },
      null,
    ],
    ["model_changed", { checkpoint_model_id: "a-different-model" }, null],
  ];

  it.each(FINGERPRINT_MISMATCH_CASES)(
    "a fingerprint mismatch on each field invalidates (%s)",
    async (_label, overrides, textOverride) => {
      const server = new FakeServer();
      const store = new RecordingCheckpointStore();
      const source = await seedFailedRun(server, store);
      expect(unitCount(store, source)).toBe(1);

      const { ingester: ingester2, llm: llm2 } = buildIngester(
        server,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
        { checkpoint_store: store, chunk_bytes: CROSS_CHUNK_BYTES, ...overrides },
      );
      const text = textOverride ?? DOC_TEXT;
      const outcome = await ingester2.ingestText(text, { source });

      expect(outcome.ok).toBe(true);
      expect(outcome.chunks_reused).toBe(0);
      expect(llm2.seenPrompts.length).toBe(2); // nothing reused — a settings mismatch invalidates all
    },
  );

  it("a changed structured_output mode invalidates checkpoints", async () => {
    // The one fingerprint field the table above can't cover generically:
    // turning on structured_output needs a tool-calling fake model, not
    // the plain chat fake every other case shares.
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = await seedFailedRun(server, store);
    expect(unitCount(store, source)).toBe(1);

    const llm = new ToolCallingFakeChatModel({
      toolCallArgs: [JSON.parse(CHUNK1_ANSWER), JSON.parse(CHUNK2_CORRECTED_ANSWER)],
    });
    const ingester = new TaguruIngester({
      context: "sake",
      llm,
      client: server.client(),
      questions: 2,
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
      checkpoint_model_id: "fake-model",
      structured_output: true,
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source });

    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(0); // the structured_output mismatch invalidated everything
    expect(outcome.llm_calls).toBe(2);
  });
});

// -- semantics interlocks -------------------------------------------------------

it("a reused chunk still participates in cross-chunk correction", async () => {
  // The money test: a checkpointed unit's STORED user/answer, not a
  // recomputed one, is what Stage 2 replays — proving ChunkRecord
  // reconstruction from a checkpoint is faithful enough for cross-chunk
  // correction to work exactly like it does on a freshly-extracted chunk.
  const server = new FakeServer();
  const store = new RecordingCheckpointStore();
  const source = "docs/aomine.md";
  // Both chunks pass Stage 1, but chunk 1's alias shadows chunk 0's object
  // — Stage 2 flags chunk 1, and even its ONE corrective turn still
  // shadows, so the bounded re-check fails the whole document. Both
  // Stage-1-valid units are still checkpointed: checkpointing happens in
  // the Stage 1 loop, never touched by Stage 2's outcome.
  const { ingester: ingester1 } = buildIngester(
    server,
    [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
    { checkpoint_store: store, chunk_bytes: CROSS_CHUNK_BYTES },
  );
  await expect(ingester1.ingestText(DOC_TEXT, { source })).rejects.toThrow(/cross-chunk/);
  expect(unitCount(store, source)).toBe(2);

  // Rerun: BOTH chunks are satisfied from checkpoint (zero Stage 1 calls),
  // and Stage 2 still correctly rebuilds chunk 1's conversation from its
  // stored user/answer to fix the (still-cached) shadowing.
  const { ingester: ingester2, llm: llm2 } = buildIngester(server, [CHUNK2_CORRECTED_ANSWER], {
    checkpoint_store: store,
    chunk_bytes: CROSS_CHUNK_BYTES,
  });
  const outcome = await ingester2.ingestText(DOC_TEXT, { source });

  expect(outcome.ok).toBe(true);
  expect(outcome.chunks_reused).toBe(2);
  expect(outcome.correction_attempts).toBe(1);
  expect(llm2.seenPrompts.length).toBe(1); // only the Stage 2 corrective call

  const stage2Messages = llm2.seenPrompts[0]!;
  const humanContents = stage2Messages.filter((m) => m.getType() === "human").map((m) => m.content);
  const aiContents = stage2Messages.filter((m) => m.getType() === "ai").map((m) => m.content);
  expect(humanContents.some((c) => String(c).includes("part 2 of 2"))).toBe(true);
  // The replayed prior-answer turn is the STORED (checkpointed) answer,
  // verbatim — proof the reused record's own conversation was rebuilt from
  // checkpoint data, not re-derived some other way.
  expect(aiContents.some((c) => String(c).includes(CHUNK2_SHADOWING_ANSWER))).toBe(true);

  expect(server.imported.length).toBe(1);
  expect(unitCount(store, source)).toBe(0);
});

describe("checkpoints survive terminal failures", () => {
  it("survives a refusal", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const refused = new AIMessage({
      content: "",
      response_metadata: { done_reason: "content_filter" },
    });
    const { ingester } = buildIngester(server, [CHUNK1_ANSWER, refused], {
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
    });
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow(
      /policy refusal is terminal/,
    );
    expect(unitCount(store, source)).toBe(1);
  });

  it("survives an empty terminal failure", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const { ingester } = buildIngester(server, [CHUNK1_ANSWER, "", ""], {
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
    });
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow();
    expect(unitCount(store, source)).toBe(1);
  });

  it("survives a Stage 2 residual issue", async () => {
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const { ingester } = buildIngester(
      server,
      [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
      { chunk_bytes: CROSS_CHUNK_BYTES, checkpoint_store: store },
    );
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow(/cross-chunk/);
    expect(unitCount(store, source)).toBe(2);
  });

  it("survives an import failure", async () => {
    const server = new FakeServer();
    server.failImport = true;
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const { ingester } = buildIngester(server, [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER], {
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
    });
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow();
    expect(unitCount(store, source)).toBe(2);
    expect(server.imported).toEqual([]);
    expect(store.log).not.toContainEqual(["delete", source]);
  });
});

it("checkpoint deletion happens before embeddings refresh and a warning does not resurrect it", async () => {
  const server = new FakeServer();
  server.embeddingsRefreshStatus = 502;
  const store = new RecordingCheckpointStore();
  const { ingester } = buildIngester(server, [MODEL_ANSWER], {
    checkpoint_store: store,
    on_event: (event) => store.log.push(["event", event.kind]),
  });
  const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
  expect(outcome.ok).toBe(true);
  expect(outcome.embeddings_refresh_warning).not.toBeNull();

  const deleteIndex = store.log.findIndex(
    ([action, source]) => action === "delete" && source === "docs/aomine.md",
  );
  const warningIndex = store.log.findIndex(
    ([action, kind]) => action === "event" && kind === "embedding_refresh_warning",
  );
  expect(deleteIndex).toBeGreaterThanOrEqual(0);
  expect(warningIndex).toBeGreaterThanOrEqual(0);
  expect(deleteIndex).toBeLessThan(warningIndex);
  expect(store.state.size).toBe(0);
});

it("a checkpoint save failure warns but the unit still counts this run", async () => {
  const server = new FakeServer();
  const store = new FailingSaveStore();
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const { ingester } = buildIngester(server, [MODEL_ANSWER], { checkpoint_store: store });
  const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
  expect(outcome.ok).toBe(true);
  expect(server.imported.length).toBe(1);
  expect(warn.mock.calls.some((call) => String(call[0]).includes("failed to save a checkpoint"))).toBe(
    true,
  );
});

it("a checkpoint delete failure is silently ignored", async () => {
  const server = new FakeServer();
  const store = new FailingDeleteStore();
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const { ingester } = buildIngester(server, [MODEL_ANSWER], { checkpoint_store: store });
  const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
  expect(outcome.ok).toBe(true);
  expect(server.imported.length).toBe(1);
  expect(warn).not.toHaveBeenCalled();
});

it("a checkpoint load failure warns and starts without any cached chunks", async () => {
  const server = new FakeServer();
  const store = new FailingLoadStore();
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const { ingester, llm } = buildIngester(server, [MODEL_ANSWER], { checkpoint_store: store });
  const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
  expect(outcome.ok).toBe(true);
  expect(llm.seenPrompts.length).toBe(1);
  expect(warn.mock.calls.some((call) => String(call[0]).includes("failed to load"))).toBe(true);
});

// -- model-identity fail-fast ----------------------------------------------------

describe("model identity fail-fast", () => {
  it("checkpoint_store without a derivable model identity requires checkpoint_model_id", () => {
    const server = new FakeServer();
    const llm = new RecordingFakeMessagesChatModel({ responses: [] });
    expect(
      () =>
        new TaguruIngester({
          context: "sake",
          llm,
          client: server.client(),
          checkpoint_store: new RecordingCheckpointStore(),
        }),
    ).toThrow(/checkpoint_model_id/);
  });

  it("checkpoint_store with an explicit model id does not require derivation", () => {
    const server = new FakeServer();
    const llm = new RecordingFakeMessagesChatModel({ responses: [] });
    expect(
      () =>
        new TaguruIngester({
          context: "sake",
          llm,
          client: server.client(),
          checkpoint_store: new RecordingCheckpointStore(),
          checkpoint_model_id: "fake-model",
        }),
    ).not.toThrow();
  });

  it("model identity is derived from a model attribute when present", async () => {
    // Behavioral proof (deriveModelIdentity itself is module-private): two
    // separate instances, neither given an explicit checkpoint_model_id,
    // must derive the SAME identity from `llm.model` for a resume to reuse
    // anything at all.
    const server = new FakeServer();
    const store = new RecordingCheckpointStore();
    const source = "docs/aomine.md";
    const llm1 = new NamedFakeChatModel({
      responses: [new AIMessage(CHUNK1_ANSWER), new AIMessage("not json"), new AIMessage("not json")],
    });
    const ingester1 = new TaguruIngester({
      context: "sake",
      llm: llm1,
      client: server.client(),
      questions: 2,
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
    }); // must not raise — model identity derived from llm.model
    await expect(ingester1.ingestText(DOC_TEXT, { source })).rejects.toThrow();
    expect(unitCount(store, source)).toBe(1);

    const llm2 = new NamedFakeChatModel({ responses: [new AIMessage(CHUNK2_CORRECTED_ANSWER)] });
    const ingester2 = new TaguruIngester({
      context: "sake",
      llm: llm2,
      client: server.client(),
      questions: 2,
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
    });
    const outcome = await ingester2.ingestText(DOC_TEXT, { source });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(1);
  });
});

// -- stale store across instances ------------------------------------------------

it("a stale filesystem store is reused across store instances", async () => {
  const { FilesystemCheckpointStore } = await import("../../src/checkpoints.js");
  const { mkdtempSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = mkdtempSync(join(tmpdir(), "taguru-ckpt-"));
  try {
    const server = new FakeServer();
    const source = "docs/aomine.md";
    const store1 = new FilesystemCheckpointStore(dir);
    const { ingester: ingester1 } = buildIngester(
      server,
      [CHUNK1_ANSWER, "not json", "not json"],
      { checkpoint_store: store1, chunk_bytes: CROSS_CHUNK_BYTES },
    );
    await expect(ingester1.ingestText(DOC_TEXT, { source })).rejects.toThrow();

    const store2 = new FilesystemCheckpointStore(dir); // a brand new instance, same directory
    const { ingester: ingester2, llm: llm2 } = buildIngester(server, [CHUNK2_CORRECTED_ANSWER], {
      checkpoint_store: store2,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    const outcome = await ingester2.ingestText(DOC_TEXT, { source });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(1);
    expect(llm2.seenPrompts.length).toBe(1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

it("a corrupt checkpoint file on disk silently degrades to empty", async () => {
  const { FilesystemCheckpointStore } = await import("../../src/checkpoints.js");
  const { mkdtempSync, rmSync, mkdirSync, writeFileSync, readFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join, dirname } = await import("node:path");
  const dir = mkdtempSync(join(tmpdir(), "taguru-ckpt-"));
  try {
    const server = new FakeServer();
    const store = new FilesystemCheckpointStore(dir);
    const source = "docs/aomine.md";
    const path = await store.pathFor(source);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, "{not-valid-json-at-all");

    const { ingester, llm } = buildIngester(server, [CHUNK1_ANSWER, "not json", "not json"], {
      checkpoint_store: store,
      chunk_bytes: CROSS_CHUNK_BYTES,
    });
    await expect(ingester.ingestText(DOC_TEXT, { source })).rejects.toThrow();

    // Both chunks were attempted in full — nothing was reused from the
    // unreadable file.
    expect(llm.seenPrompts.length).toBe(3);

    // The file now holds valid, parseable state with the one unit this run
    // actually completed.
    const parsed = JSON.parse(readFileSync(path, "utf-8"));
    expect(Object.keys(parsed.units).length).toBe(1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
