/**
 * Mechanical removal before corrective turns (ADR 0013 Stage 1, issue #1098)
 * — the same cases as the Python twin's tests. Self-contained on purpose:
 * importing ingester.test.ts or checkpoints.test.ts for their helpers would
 * register those suites a second time in this file.
 */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, it } from "vitest";

import {
  DocumentCheckpoints,
  type CheckpointFingerprint,
  type CheckpointStore,
} from "../../src/checkpoints.js";
import type { ChunkCompleted, IngestEvent } from "../../src/events.js";
import {
  evaluateAnswer,
  InvalidFault,
  issueInAnswerTerms,
  mechanicalInterpret,
  removalFromJson,
  removalText,
  type ItemRules,
} from "../../src/extract.js";
import { TaguruIngester } from "../../src/ingest.js";
import { FakeServer } from "./stub.js";

const RULES: ItemRules = { paragraphCount: 1, questionsRequested: false };

// tests/fixtures/model_output/removed/ fixtures whose rule this port does not
// implement yet — each still takes the corrective path here. Listed by name
// so a fixture added to the shared corpus later fails this test until
// someone decides which side of the line it falls on.
const REMOVED_FIXTURES_NOT_YET_PORTED = new Set([
  "alias_dangling_canonical.json", // Stage 2 prune
  "association_fabricated_object.json", // occurrence check (ADR 0013 §4)
  "association_fabricated_sparse_object.json", // occurrence check (ADR 0036)
  "association_foreign_ideograph_label.json", // foreign ideographs (ADR 0039)
]);

interface RemovedFixture {
  rules: { paragraph_count: number; questions_cap: number };
  answer: unknown;
  removed: string[];
  surviving_associations: number;
  surviving_aliases: number;
}

describe("mechanicalInterpret (ADR 0013 Stage 1)", () => {
  const fixturesRoot = join(
    dirname(fileURLToPath(import.meta.url)),
    "../../../../tests/fixtures/model_output/removed",
  );
  const names = readdirSync(fixturesRoot).filter((name) => name.endsWith(".json"));
  const ported = names.filter((name) => !REMOVED_FIXTURES_NOT_YET_PORTED.has(name));

  it("keeps the pending-fixture list in step with the shared corpus", () => {
    for (const name of REMOVED_FIXTURES_NOT_YET_PORTED) {
      expect(names).toContain(name);
    }
    expect(ported.length).toBeGreaterThan(0);
  });

  it.each(ported)("removes %s mechanically with zero corrective issues", (name) => {
    const fixture = JSON.parse(
      readFileSync(join(fixturesRoot, name), "utf-8"),
    ) as RemovedFixture;
    const rules: ItemRules = {
      paragraphCount: fixture.rules.paragraph_count,
      questionsRequested: fixture.rules.questions_cap > 0,
    };
    const { output, removed, issues } = mechanicalInterpret(fixture.answer, rules);
    expect(issues).toEqual([]);
    expect(removed.map(removalText)).toEqual(fixture.removed);
    expect(output.associations).toHaveLength(fixture.surviving_associations);
    expect(output.aliases).toHaveLength(fixture.surviving_aliases);
  });

  it("removes each Stage 1 class with the model's original item", () => {
    const answer = {
      associations: [
        "not an object",
        { subject: " ", label: "is" },
        { subject: "a", label: "is", object: "b" },
      ],
      aliases: [
        7,
        { alias: "x", canonical: null },
        { alias: "b", canonical: "b", kind: "concept" },
        { alias: "B", canonical: "b", kind: "concept" },
      ],
    };
    const { output, removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual([]);
    expect(removed.map(removalText)).toEqual([
      'associations[0]: expected an object, got string "not an object"',
      "associations[1]: subject empty, object missing",
      "aliases[0]: expected an object, got number 7",
      "aliases[1]: canonical missing, kind missing",
      "aliases[2]: alias equals its canonical",
    ]);
    expect(removed[1]).toEqual({
      path: "associations[1]",
      reason: "subject empty, object missing",
      item: { subject: " ", label: "is" },
    });
    expect(output.associations.map((a) => [a.subject, a.object])).toEqual([["a", "b"]]);
    expect(output.aliases.map((a) => [a.alias, a.canonical])).toEqual([["B", "b"]]);
  });

  it("removes a single-character label", () => {
    const answer = {
      associations: [
        { subject: "青嶺酒造", label: "の", object: "高瀬" },
        { subject: "青嶺酒造", label: "杜氏", object: "高瀬" },
        // One code point outside the BMP is two UTF-16 units but one char.
        { subject: "青嶺酒造", label: "𠮷", object: "高瀬" },
      ],
    };
    const { output, removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual([]);
    expect(removed.map(removalText)).toEqual([
      'associations[0]: label "の" is a single character — too generic to be a relation',
      'associations[2]: label "𠮷" is a single character — too generic to be a relation',
    ]);
    expect(output.associations.map((a) => a.label)).toEqual(["杜氏"]);
  });

  it("removes a redaction placeholder from associations and aliases", () => {
    const mask = "«redacted email 1a2b»";
    const answer = {
      associations: [
        { subject: mask, label: "連絡先", object: `担当 ${mask}` },
        { subject: "青嶺酒造", label: "杜氏", object: "高瀬" },
      ],
      aliases: [
        { alias: mask, canonical: mask, kind: "concept" },
        { alias: "Aomine", canonical: "青嶺酒造", kind: "concept" },
      ],
    };
    const { output, removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual([]);
    expect(removed.map(removalText)).toEqual([
      `associations[0]: subject "${mask}" is a redaction placeholder, ` +
        `object "担当 ${mask}" is a redaction placeholder — not an entity`,
      `aliases[0]: alias "${mask}" is a redaction placeholder — not a spelling`,
    ]);
    expect(output.associations).toHaveLength(1);
    expect(output.aliases).toHaveLength(1);
  });

  it("needs the exact placeholder form", () => {
    const answer = {
      associations: [
        { subject: "«redacted email 1A2B»", label: "連絡先", object: "x" },
        { subject: "«redacted email 1a2»", label: "連絡先", object: "x" },
        { subject: "<redacted email 1a2b>", label: "連絡先", object: "x" },
      ],
    };
    const { removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual([]);
    expect(removed).toEqual([]);
  });

  it("keeps a correctable item from the removal rules", () => {
    const answer = { associations: [{ subject: "a", label: "の", object: "b", weight: 0 }] };
    const { removed, issues } = mechanicalInterpret(answer, RULES);
    expect(removed).toEqual([]);
    expect(issues).toEqual(["associations[0].weight: expected finite non-zero number, got 0"]);
  });

  it("drops a self-alias's other issues with it", () => {
    const answer = { aliases: [{ alias: "b", canonical: "b", kind: "entity" }] };
    const { removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual([]);
    expect(removed.map(removalText)).toEqual(["aliases[0]: alias equals its canonical"]);
  });

  it("keeps present-but-wrong values for the corrective turn", () => {
    const answer = {
      associations: [
        { subject: "a", label: "l", object: "" },
        { subject: "a", label: "l", object: "b", weight: 0 },
      ],
      aliases: [],
    };
    const { removed, issues } = mechanicalInterpret(answer, RULES);
    expect(issues).toEqual(["associations[1].weight: expected finite non-zero number, got 0"]);
    expect(removed.map(removalText)).toEqual(["associations[0]: object empty"]);
    expect(() => evaluateAnswer(JSON.stringify(answer), RULES)).toThrow(InvalidFault);
  });

  it("evaluateAnswer accepts after mechanical removal and records it", () => {
    const content = JSON.stringify({
      associations: [
        { subject: "青嶺酒造", label: "杜氏", object: "高瀬" },
        { subject: "リリース署名鍵", label: "管理者", object: "" },
      ],
      aliases: [{ alias: "nextest", canonical: "nextest", kind: "concept" }],
    });
    const { output, removed } = evaluateAnswer(content, RULES);
    expect(removed.map(removalText)).toEqual([
      "associations[1]: object empty",
      "aliases[0]: alias equals its canonical",
    ]);
    expect(output.associations).toHaveLength(1);
    expect(output.aliases).toEqual([]);
    // Lossy mode validates nothing and so removes nothing.
    expect(evaluateAnswer(content, null).removed).toEqual([]);
  });

  it("names an output item by its place in the answer", () => {
    const output = {
      associations: [
        { subject: "a", label: "l", object: "b", weight: 1, paragraph: null },
        { subject: "c", label: "l", object: "d", weight: 1, paragraph: null },
      ],
      aliases: [
        { alias: "x", canonical: "a", kind: "concept" },
        { alias: "y", canonical: "c", kind: "concept" },
      ],
      questions: [],
    };
    const removed = [
      { path: "associations[1]", reason: "object empty", item: null },
      { path: "aliases[0]", reason: "alias equals its canonical", item: null },
      { path: "aliases[2]", reason: "alias equals its canonical", item: null },
    ];
    const terms = (issue: string) => issueInAnswerTerms(output, removed, issue);
    expect(terms("aliases[0].alias: names something")).toBe("aliases[1].alias: names something");
    expect(terms("aliases[1]: conflicts with an earlier alias")).toBe(
      "aliases[3]: conflicts with an earlier alias",
    );
    expect(terms("associations[0].subject: x")).toBe("associations[0].subject: x");
    expect(terms("associations[1].object: y")).toBe("associations[2].object: y");
    expect(terms("aliases: expected an array")).toBe("aliases: expected an array");
    expect(issueInAnswerTerms(output, [], "aliases[1].alias: z")).toBe("aliases[1].alias: z");
  });

  it("round-trips a removal through its JSON form", () => {
    const removal = { path: "aliases[0]", reason: "alias equals its canonical", item: { alias: "b" } };
    expect(removalFromJson(JSON.parse(JSON.stringify(removal)))).toEqual(removal);
    expect(removalFromJson({ path: "aliases[0]" })).toBeNull();
    expect(removalFromJson("aliases[0]: alias equals its canonical")).toBeNull();
  });
});

// -- the ingester ------------------------------------------------------------

const DOC_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬である。";
// Splits DOC_TEXT into exactly its two paragraphs, one chunk each.
const CROSS_CHUNK_BYTES = 40;

// A self-alias and an association with an empty object: both removable, so
// the answer is accepted on the first call — the shape a real model produced
// in the #1098 report, which used to fail the whole source after a
// corrective turn.
const REMOVABLE_ANSWER = JSON.stringify({
  associations: [
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 },
    { subject: "青嶺酒造", label: "創業年", object: "", weight: 1.0 },
  ],
  aliases: [{ alias: "青嶺酒造", canonical: "青嶺酒造", kind: "concept" }],
});
const REMOVABLE_ITEMS = ["associations[1]: object empty", "aliases[0]: alias equals its canonical"];

const CHUNK1_SELF_ALIAS = JSON.stringify({
  associations: [{ subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0 }],
  aliases: [{ alias: "1907年", canonical: "1907年", kind: "concept" }],
});
const CHUNK2_ANSWER = JSON.stringify({
  associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }],
  aliases: [],
});

const make = (responses: string[], fields: Record<string, unknown> = {}) =>
  new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses }),
    client: new FakeServer().client(),
    questions: 2,
    ...fields,
  });

/** Echoes scripted tool-call args — the structured_output path end to end
 * (a local copy of ingester.test.ts's fake; see this file's header). */
class ToolCallingFake extends BaseChatModel {
  calls = 0;
  constructor(private readonly toolCallArgs: Record<string, unknown>[]) {
    super({});
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

/** Scripted text answers that records every prompt it was sent. */
class RecordingFake extends BaseChatModel {
  calls = 0;
  seenPrompts: BaseMessage[][] = [];
  constructor(private readonly responses: string[]) {
    super({});
  }
  _llmType(): string {
    return "recording-fake";
  }
  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.seenPrompts.push(messages);
    const text = this.responses[this.calls]!;
    this.calls += 1;
    return { generations: [{ text, message: new AIMessage(text) }] };
  }
}

class MemoryCheckpointStore implements CheckpointStore {
  state = new Map<string, Uint8Array>();
  async load(source: string): Promise<Uint8Array | null> {
    return this.state.get(source) ?? null;
  }
  async save(source: string, data: Uint8Array): Promise<void> {
    this.state.set(source, data);
  }
  async delete(source: string): Promise<void> {
    this.state.delete(source);
  }
}

describe("TaguruIngester mechanical removal (ADR 0013, issue #1098)", () => {
  it("removes removable items without a corrective turn", async () => {
    const events: IngestEvent[] = [];
    const outcome = await make([REMOVABLE_ANSWER], {
      on_event: (event: IngestEvent) => events.push(event),
    }).ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(1);
    expect(outcome.correction_attempts).toBe(0);
    expect(outcome.removed_items).toEqual(REMOVABLE_ITEMS);
    const completed = events.filter(
      (event): event is ChunkCompleted => event.kind === "chunk_completed",
    );
    expect(completed.map((event) => event.removed_items)).toEqual([REMOVABLE_ITEMS]);
    expect(completed[0]!.associations_proposed).toBe(1);
    expect(completed[0]!.aliases_proposed).toBe(0);
  });

  it("discards the removals of an answer that went back for correction", async () => {
    // ADR 0013 §3.5: only the accepted answer's own removals count.
    const removableAndWrong = JSON.stringify({
      associations: [
        { subject: "青嶺酒造", label: "杜氏", object: "", weight: 1.0 },
        { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 0 },
      ],
      aliases: [],
    });
    const outcome = await make([removableAndWrong, REMOVABLE_ANSWER]).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.correction_attempts).toBe(1);
    expect(outcome.removed_items).toEqual(REMOVABLE_ITEMS);
  });

  it("names the chunk of each removal across chunks", async () => {
    const outcome = await make([CHUNK1_SELF_ALIAS, CHUNK2_ANSWER], {
      chunk_bytes: CROSS_CHUNK_BYTES,
    }).ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks).toBe(2);
    expect(outcome.removed_items).toEqual(["chunk 1/2 aliases[0]: alias equals its canonical"]);
  });

  it("removes removable items on the structured-output path too", async () => {
    const ingester = new TaguruIngester({
      context: "sake",
      llm: new ToolCallingFake([JSON.parse(REMOVABLE_ANSWER) as Record<string, unknown>]),
      client: new FakeServer().client(),
      questions: 2,
      structured_output: true,
    });
    const outcome = await ingester.ingestText(DOC_TEXT, { source: "docs/aomine.md" });
    expect(outcome.ok).toBe(true);
    expect(outcome.llm_calls).toBe(1);
    expect(outcome.correction_attempts).toBe(0);
    expect(outcome.removed_items).toEqual(REMOVABLE_ITEMS);
  });

  it("replaces a chunk's removals with a Stage 2 re-answer's", async () => {
    // Chunk 2's first answer carries a removable self-alias AND a shadowing
    // alias only Stage 2 sees; the re-answer removes nothing, so the
    // chunk's removals are none (ADR 0013 §3.5).
    const chunk2SelfAliasAndShadowing = JSON.stringify({
      associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }],
      aliases: [
        { alias: "高瀬", canonical: "高瀬", kind: "concept" },
        { alias: "1907年", canonical: "青嶺酒造", kind: "concept" },
      ],
    });
    const chunk1 = JSON.stringify({
      associations: [{ subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0 }],
      aliases: [],
    });
    const corrected = JSON.stringify({
      associations: [{ subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0 }],
      aliases: [{ alias: "Sake", canonical: "青嶺酒造", kind: "concept" }],
    });
    const llm = new RecordingFake([chunk1, chunk2SelfAliasAndShadowing, corrected]);
    const outcome = await make([], { llm, chunk_bytes: CROSS_CHUNK_BYTES }).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.correction_attempts).toBe(1);
    expect(outcome.removed_items).toEqual([]);
    // The shadowing alias is the OUTPUT's first alias but the ANSWER's
    // second; the corrective ask names it as the replayed answer has it.
    const ask = llm.seenPrompts.at(-1)!.at(-1)!.content as string;
    expect(ask).toContain("aliases[1].alias: names something the associations already contain");
    expect(ask).not.toContain("aliases[0].alias");
  });

  it("reports nothing removed in lossy mode", async () => {
    const outcome = await make([REMOVABLE_ANSWER], { lossy: true }).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.removed_items).toEqual([]);
  });

  it("labels a reused chunk's removals by its position in this run", async () => {
    // Run 1 (26 bytes) chunks the text [ab, c]; run 2 (16 bytes) chunks it
    // [a, b, c] and reuses c, written as chunk 2 of 2.
    const text = "aaaaaaaa\n\nbbbbbbbb\n\ncccccccccccc";
    const plain = JSON.stringify({ associations: [{ subject: "a", label: "is", object: "b" }] });
    const selfAlias = JSON.stringify({
      associations: [{ subject: "c", label: "is", object: "d" }],
      aliases: [{ alias: "c", canonical: "c", kind: "concept" }],
    });
    const store = new MemoryCheckpointStore();
    const fields = { checkpoint_store: store, checkpoint_model_id: "fake-model" };
    const first = await make([plain, selfAlias], { ...fields, chunk_bytes: 26 }).ingestText(text, {
      source: "docs/abc.md",
      dry_run: true,
    });
    expect(first.removed_items).toEqual(["chunk 2/2 aliases[0]: alias equals its canonical"]);
    const second = await make([plain, plain], { ...fields, chunk_bytes: 16 }).ingestText(text, {
      source: "docs/abc.md",
      dry_run: true,
    });
    expect(second.chunks_reused).toBe(1);
    expect(second.removed_items).toEqual(["chunk 3/3 aliases[0]: alias equals its canonical"]);
  });

  it("reports the removals of completed chunks when interrupted", async () => {
    let checks = 0;
    const outcome = await make([CHUNK1_SELF_ALIAS], { chunk_bytes: CROSS_CHUNK_BYTES }).ingestText(
      DOC_TEXT,
      { source: "docs/aomine.md", should_stop: () => (checks += 1) > 1 },
    );
    expect(outcome.interrupted).toBe(true);
    expect(outcome.removed_items).toEqual(["chunk 1/2 aliases[0]: alias equals its canonical"]);
  });

  it("keeps a reused chunk's removals", async () => {
    const store = new MemoryCheckpointStore();
    const source = "docs/aomine.md";
    const options = {
      chunk_bytes: CROSS_CHUNK_BYTES,
      checkpoint_store: store,
      checkpoint_model_id: "fake-model",
    };
    await expect(
      make([CHUNK1_SELF_ALIAS, "not json", "not json"], options).ingestText(DOC_TEXT, { source }),
    ).rejects.toThrow();
    expect(store.state.has(source)).toBe(true);

    const outcome = await make([CHUNK2_ANSWER], options).ingestText(DOC_TEXT, { source });
    expect(outcome.ok).toBe(true);
    expect(outcome.chunks_reused).toBe(1);
    expect(outcome.removed_items).toEqual(["chunk 1/2 aliases[0]: alias equals its canonical"]);
  });
});

describe("checkpoint pieces and removals", () => {
  const fingerprint: CheckpointFingerprint = {
    sha256: "deadbeef",
    model: "fake-model",
    prompt_version: 2,
    context: "sake",
    questions_n: 2,
    no_passage: false,
    description: "",
    fact_budget: 0,
    structured_output: "",
    lossy: false,
    schema_digest: "",
  };
  const fileWith = (piece: Record<string, unknown>): Uint8Array =>
    new TextEncoder().encode(JSON.stringify({ fingerprint, pieces: { h: piece } }));
  const basePiece = {
    chunk_index: 0,
    output: { associations: [], aliases: [], questions: [] },
    user: "u",
    answer: "{}",
  };

  it("loads a piece written before removals existed with none", () => {
    const loaded = DocumentCheckpoints.fromBytes(fileWith(basePiece), fingerprint);
    expect(loaded.lookup("h")?.removed).toEqual([]);
  });

  it("treats an explicit null removal list as a corrupt file", () => {
    const loaded = DocumentCheckpoints.fromBytes(
      fileWith({ ...basePiece, removed: null }),
      fingerprint,
    );
    expect(loaded.lookup("h")).toBeNull();
  });

  it("treats a malformed removal list as a corrupt file", () => {
    const loaded = DocumentCheckpoints.fromBytes(
      fileWith({ ...basePiece, removed: [{ path: 1 }] }),
      fingerprint,
    );
    expect(loaded.lookup("h")).toBeNull();
  });
});
