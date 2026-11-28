/**
 * Mechanical removal before corrective turns (ADR 0013 Stage 1, issue #1098)
 * — the same cases as the Python twin's tests. Self-contained on purpose:
 * importing ingester.test.ts or checkpoints.test.ts for their helpers would
 * register those suites a second time in this file.
 */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

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
        { subject: " ", label: "l" },
        { subject: "a", label: "l", object: "b" },
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
      item: { subject: " ", label: "l" },
    });
    expect(output.associations.map((a) => [a.subject, a.object])).toEqual([["a", "b"]]);
    expect(output.aliases.map((a) => [a.alias, a.canonical])).toEqual([["B", "b"]]);
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

  it("reports nothing removed in lossy mode", async () => {
    const outcome = await make([REMOVABLE_ANSWER], { lossy: true }).ingestText(DOC_TEXT, {
      source: "docs/aomine.md",
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.removed_items).toEqual([]);
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

  it("treats a malformed removal list as a corrupt file", () => {
    const loaded = DocumentCheckpoints.fromBytes(
      fileWith({ ...basePiece, removed: [{ path: 1 }] }),
      fingerprint,
    );
    expect(loaded.lookup("h")).toBeNull();
  });
});
