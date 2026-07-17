/** The full loop against the real server binary — mirrors the Python suite. */

import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { Taguru } from "../../src/client.js";
import {
  AuthenticationError,
  ConflictError,
  EmbeddingUnavailableError,
  NotFoundError,
  PayloadTooLargeError,
  PermissionDeniedError,
  RateLimitError,
  ValidationError,
} from "../../src/errors.js";
import { citationKey } from "../../src/models.js";
import {
  ADMIN_TOKEN,
  READER_TOKEN,
  serverBinary,
  spawnServer,
  type SpawnedServer,
} from "./spawn.js";

const AOMINE_DOC = `青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。`;

const fresh = () => `ts-${randomUUID().slice(0, 12)}`;

let client: Taguru;
let reader: Taguru;
let binary: string;
let server: SpawnedServer;

beforeAll(async () => {
  binary = serverBinary();
  server = await spawnServer(binary, {
    TAGURU_API_TOKENS: `admin:${ADMIN_TOKEN},reader:${READER_TOKEN}`,
    TAGURU_KEY_SCOPES: '{"reader": "read"}',
  });
  client = new Taguru({ base_url: server.baseUrl, api_key: ADMIN_TOKEN });
  reader = new Taguru({ base_url: server.baseUrl, api_key: READER_TOKEN });
  await client.waitUntilReady({ timeout: 30 });
});

afterAll(() => {
  server.stop();
});

async function seed(name: string): Promise<void> {
  await client.contexts.create(name, { description: "青嶺酒造という架空の酒蔵の知識" });
  const ctx = client.context(name);
  await ctx.addAssociations([
    { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, source: "docs/aomine.md", paragraph: 0 },
    { subject: "青嶺酒造", label: "代表銘柄", object: "青嶺", weight: 1.0, source: "docs/aomine.md", paragraph: 0 },
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, source: "docs/aomine.md", paragraph: 1 },
    { subject: "高瀬", label: "重視する", object: "寒仕込み", weight: 1.0, source: "docs/aomine.md", paragraph: 1 },
    { subject: "青嶺酒造", label: "行う", object: "大量生産", weight: -1.0, source: "docs/aomine.md", paragraph: 2 },
  ]);
  await ctx.storePassages({ "docs/aomine.md": AOMINE_DOC });
}

describe("context lifecycle", () => {
  it("creates, lists, updates, deletes", async () => {
    const name = fresh();
    expect(await client.contexts.exists(name)).toBe(false);
    expect(await client.contexts.create(name, { description: "d" })).toBe(true);
    await expect(client.contexts.create(name)).rejects.toMatchObject({ code: "already_exists" });

    const entry = await client.contexts.get(name);
    expect(entry.description).toBe("d");

    const meta = await client.contexts.update(name, { description: "d2", dice_floor: 0.25 });
    expect(meta.description).toBe("d2");
    expect(meta.dice_floor).toBe(0.25);

    const names: string[] = [];
    for await (const row of client.contexts.iter({ limit: 2 })) {
      names.push(row.name);
    }
    expect(names).toContain(name);

    const renamed = `${name}-renamed`;
    expect(await client.contexts.rename(name, renamed)).toBe(true);
    const renamedAway = await client.contexts.get(name).catch((caught: unknown) => caught);
    expect(renamedAway).toBeInstanceOf(NotFoundError);
    expect((renamedAway as NotFoundError).code).toBe("no_context");
    const movedEntry = await client.contexts.get(renamed);
    expect(movedEntry.description).toBe("d2");
    expect(movedEntry.dice_floor).toBe(0.25);

    expect(await client.contexts.delete(renamed)).toBe(true);
    const missing = await client.contexts.get(renamed).catch((caught: unknown) => caught);
    expect(missing).toBeInstanceOf(NotFoundError);
    expect((missing as NotFoundError).code).toBe("no_context");
  });
});

describe("graph writes and reads", () => {
  it("accumulates weight and validates", async () => {
    const name = fresh();
    await client.contexts.create(name);
    const ctx = client.context(name);
    const op = { subject: "s", label: "l", object: "o", weight: 1.0, source: "a" };
    expect(await ctx.addAssociations([op])).toBe(1);
    expect(await ctx.addAssociations([{ ...op, source: "b" }])).toBe(1);

    const page = await ctx.query({ subject: "s", label: "l" });
    expect(page.total).toBe(1);
    expect(page.matches[0]!.weight).toBe(1.0);
    expect(page.matches[0]!.count).toBe(2);
    expect(page.matches[0]!.attributions.map((a) => a.source).sort()).toEqual(["a", "b"]);

    await expect(ctx.addAssociations([{ ...op, weight: 1e300 }])).rejects.toBeInstanceOf(
      ValidationError,
    );
    await expect(ctx.addAssociations([{ ...op, subject: "" }])).rejects.toBeInstanceOf(
      ValidationError,
    );
    await client.contexts.delete(name);
  });

  it("serves the read surface", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    expect((await ctx.recall("青嶺酒造")).total).toBeGreaterThanOrEqual(4);

    const both = await ctx.query({ subject: "青嶺酒造", label: ["杜氏", "代表銘柄"] });
    expect(new Set(both.matches.map((m) => m.object))).toEqual(new Set(["高瀬", "青嶺"]));

    const negated = await ctx.query({ subject: "青嶺酒造", label: "行う" });
    expect(negated.matches[0]!.weight).toBe(-1.0);

    const outline = await ctx.describe("青嶺酒造");
    expect(outline).not.toBeNull();
    expect(outline!.as_subject.some((u) => u.label === "杜氏")).toBe(true);
    expect(await ctx.describe("存在しない概念")).toBeNull();

    const walked = await ctx.explore(["青嶺酒造"], { max_depth: 2 });
    const twoHop = walked.matches.find((m) => m.association.object === "寒仕込み");
    expect(twoHop?.distance).toBe(2);

    const activated = await ctx.activate(["青嶺酒造"], { limit: 10 });
    const strengths = activated.matches.map((m) => m.strength);
    expect(strengths).toEqual([...strengths].sort((a, b) => b - a));

    expect((await ctx.unreachableFrom(["青嶺酒造"])).total).toBe(0);

    const labels = await ctx.listLabels();
    expect(labels.labels).toContain("杜氏");
    const iterated: string[] = [];
    for await (const label of ctx.iterLabels()) {
      iterated.push(label);
    }
    expect(iterated).toEqual(labels.labels);
    await client.contexts.delete(name);
  });

  it("resumes recall pagination from a match object without a 400 (MatchCursor structural trap)", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    const first = await ctx.recall("青嶺酒造", { limit: 1 });
    expect(first.matches).toHaveLength(1);
    expect(first.total).toBeGreaterThan(1);

    // `first.matches[0]` is a full `Association` — it carries `count` and
    // `attributions` on top of the four fields `MatchCursor` wants. Passing
    // it straight back as `after` type-checks (Association structurally
    // satisfies MatchCursor), but the server's `MatchCursor` rejects
    // unrecognized fields; the SDK must narrow it before it hits the wire.
    const second = await ctx.recall("青嶺酒造", { limit: 1, after: first.matches[0] });
    expect(second.matches).toHaveLength(1);
    expect(second.matches[0]!.object).not.toEqual(first.matches[0]!.object);

    await client.contexts.delete(name);
  });

  it("resolves cues across kinds and floors", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    const exact = await ctx.resolve("青嶺酒造");
    expect(exact[0]).toMatchObject({ name: "青嶺酒造", kind: "exact", tier: "lexical", score: 1.0 });

    const fuzzy = await ctx.resolve("青嶺酒蔵");
    expect(fuzzy.some((c) => c.name === "青嶺酒造")).toBe(true);

    const wide = await ctx.resolve("青嶺", { dice_floor: 0.1 });
    const narrow = await ctx.resolve("青嶺", { dice_floor: 0.9 });
    expect(wide.length).toBeGreaterThanOrEqual(narrow.length);

    expect((await ctx.resolveLabel("杜氏"))[0]!.name).toBe("杜氏");
    await client.contexts.delete(name);
  });
});

describe("aliases", () => {
  it("registers, iterates, re-registers as a no-op, removes", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    expect(
      await ctx.addAliases({ concepts: { "Aomine Brewery": "青嶺酒造" }, labels: { brewer: "杜氏" } }),
    ).toBe(2);
    const page = await ctx.getAliases();
    expect(page.concepts).toEqual({ "Aomine Brewery": "青嶺酒造" });

    const namespaces: string[] = [];
    for await (const entry of ctx.iterAliases()) {
      namespaces.push(entry.namespace);
    }
    expect(namespaces).toEqual(["concept", "label"]);

    expect((await ctx.resolve("Aomine Brewery"))[0]).toMatchObject({
      name: "青嶺酒造",
      kind: "alias",
    });

    // Identical re-registration is a no-op — pins add_aliases' retry-safety.
    expect(await ctx.addAliases({ concepts: { "Aomine Brewery": "青嶺酒造" } })).toBe(1);

    await expect(ctx.addAliases({ concepts: { 高瀬: "青嶺酒造" } })).rejects.toBeInstanceOf(
      ConflictError,
    );

    expect(await ctx.removeAliases({ concepts: ["Aomine Brewery"], labels: ["brewer"] })).toBe(2);
    expect((await ctx.getAliases()).concepts).toEqual({});
    await client.contexts.delete(name);
  });
});

describe("sources and citations", () => {
  it("stores, lists, looks up, searches, cites, retracts", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    const stored = await ctx.storePassages(
      { "docs/extra.md": "第一段落。\n\n第二段落。" },
      {
        questions: { "docs/extra.md": [{ paragraph: 1, question: "二番目は?" }, { paragraph: 9, question: "範囲外" }] },
        sections: { "docs/extra.md": [{ paragraph: 0, section: "冒頭" }, { paragraph: 8, section: "範囲外" }] },
      },
    );
    expect(stored).toMatchObject({
      stored: 1,
      questions_stored: 1,
      questions_dropped: 1,
      sections_stored: 1,
      sections_dropped: 1,
    });

    const sources = await ctx.listSources();
    expect(new Set(sources.sources)).toEqual(new Set(["docs/aomine.md", "docs/extra.md"]));

    const lookup = await ctx.lookupPassages(["docs/aomine.md", "missing.md"]);
    expect(lookup.passages["docs/aomine.md"]).toBe(AOMINE_DOC);
    expect(lookup.missing).toEqual(["missing.md"]);

    const hits = await ctx.searchPassages("杜氏は高瀬である", { limit: 3 });
    expect(hits[0]!.source).toBe("docs/aomine.md");
    expect(hits[0]!.lanes.bm25).toBeDefined();
    expect(hits[0]!.lanes.vector).toBeUndefined();

    const cited = await ctx.citePassage("docs/aomine.md", 1);
    expect(cited.text).toContain("杜氏は高瀬");
    await expect(ctx.citePassage("missing.md", 0)).rejects.toBeInstanceOf(NotFoundError);
    await expect(ctx.citePassage("docs/aomine.md", 99)).rejects.toBeInstanceOf(NotFoundError);

    const retracted = await ctx.retractSource("docs/extra.md");
    expect(retracted.passage_removed).toBe(true);
    await client.contexts.delete(name);
  });

  it("answers 501 for embeddings refresh without a provider", async () => {
    const name = fresh();
    await client.contexts.create(name);
    const error = await client
      .context(name)
      .refreshEmbeddings()
      .catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(EmbeddingUnavailableError);
    expect((error as EmbeddingUnavailableError).reason).toBe("not_configured");
    expect((error as EmbeddingUnavailableError).code).toBe("embeddings_unconfigured");
    await client.contexts.delete(name);
  });
});

describe("transfer and maintenance", () => {
  it("round-trips export → import idempotently", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);
    await ctx.addAliases({ concepts: { Aomine: "青嶺酒造" } });
    const stream = await ctx.export();

    const restoredName = `${name}-restored`;
    const renamed = stream.replaceAll(`"${name}"`, `"${restoredName}"`);
    const outcomes = await client.importBatches(renamed);
    expect(outcomes.batches.every((o) => o.context === restoredName)).toBe(true);

    const restored = client.context(restoredName);
    const before = (await restored.query({ subject: "青嶺酒造", label: "杜氏" })).matches[0]!;
    await client.importBatches(renamed);
    const after = (await restored.query({ subject: "青嶺酒造", label: "杜氏" })).matches[0]!;
    expect(after.weight).toBe(before.weight);
    expect(after.count).toBe(before.count);

    // exportStream yields the same bytes export() returns.
    const chunks: Uint8Array[] = [];
    for await (const chunk of ctx.exportStream()) {
      chunks.push(chunk);
    }
    expect(Buffer.concat(chunks).toString("utf-8")).toBe(stream);

    await client.contexts.delete(name);
    await client.contexts.delete(restoredName);
  });

  it("compacts and flushes", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);
    // Freshly seeded → dirty → flush names it. (Compact persists as part of
    // its rebuild, so the order matters: flush first.)
    expect(await client.flush()).toContain(name);
    await ctx.retractSource("docs/aomine.md");
    const outcome = await ctx.compact();
    expect(outcome.bytes_after).toBeLessThanOrEqual(outcome.bytes_before);
    await client.contexts.delete(name);
  });

  it("audits vocabulary twins", async () => {
    const name = fresh();
    await client.contexts.create(name);
    const ctx = client.context(name);
    await ctx.addAssociations([
      { subject: "株式会社青嶺", label: "kind", object: "会社", weight: 1.0 },
      { subject: "青嶺株式会社", label: "kind", object: "会社", weight: 1.0 },
    ]);
    const audit = await ctx.auditVocabulary({ dice_floor: 0.4 });
    expect(
      audit.lexical_concepts.some(
        (pair) => new Set([pair.a, pair.b]).size === 2 && [pair.a, pair.b].includes("株式会社青嶺"),
      ),
    ).toBe(true);
    await client.contexts.delete(name);
  });

  it("audits drift: unsourced weight and dead-canonical aliases", async () => {
    const name = fresh();
    await client.contexts.create(name);
    const ctx = client.context(name);
    // No `source`: this weight lands unexplained by any named source.
    await ctx.addAssociations([
      { subject: "青嶺酒造", label: "kind", object: "会社", weight: 1.0 },
    ]);
    await ctx.addAssociations([
      { subject: "高瀬", label: "kind", object: "杜氏", weight: 1.0, source: "docs/a.md" },
    ]);
    await ctx.retractAssociation("高瀬", "kind", "杜氏");
    expect(await ctx.addAliases({ concepts: { タカセ: "高瀬" } })).toBe(1);

    const audit = await ctx.auditDrift();
    expect(audit.total).toBe(1);
    expect(audit.unsourced[0]!.association.subject).toBe("青嶺酒造");
    expect(audit.unsourced[0]!.unsourced_weight).toBe(1.0);
    expect(audit.dead_concept_aliases).toEqual({ タカセ: "高瀬" });
    expect(audit.twins).toBeNull();

    const floored = await ctx.auditDrift({ unsourced_floor: 10.0 });
    expect(floored.total).toBe(0);
    expect(floored.unsourced).toEqual([]);

    const withTwins = await ctx.auditDrift({ include_twins: true, dice_floor: 0.4 });
    expect(withTwins.twins).not.toBeNull();
    await client.contexts.delete(name);
  });

  it("resumes drift-audit pagination from an unsourced entry's association (MatchCursor structural trap)", async () => {
    const name = fresh();
    await client.contexts.create(name);
    const ctx = client.context(name);
    // Two unsourced edges so a limit:1 page has somewhere to resume from.
    await ctx.addAssociations([{ subject: "青嶺酒造", label: "kind", object: "会社", weight: 1.0 }]);
    await ctx.addAssociations([{ subject: "高瀬", label: "kind", object: "杜氏", weight: 1.0 }]);

    const first = await ctx.auditDrift({ limit: 1 });
    expect(first.unsourced).toHaveLength(1);
    expect(first.total).toBe(2);

    // `unsourced[0].association` is a full `Association` — it carries
    // `count`/`attributions` beyond what `MatchCursor` wants. Passing it
    // straight back as `after` must not 400.
    const second = await ctx.auditDrift({ limit: 1, after: first.unsourced[0]!.association });
    expect(second.unsourced).toHaveLength(1);
    expect(second.unsourced[0]!.association.subject).not.toEqual(
      first.unsourced[0]!.association.subject,
    );

    await client.contexts.delete(name);
  });
});

describe("auth and limits", () => {
  it("rejects wrong or missing tokens with 401", async () => {
    const baseUrl = server.baseUrl;
    const wrong = new Taguru({ base_url: baseUrl, api_key: "wrong-token", retries: 0 });
    await expect(wrong.contexts.list()).rejects.toMatchObject({ code: "unauthorized" });
    const missing = new Taguru({ base_url: baseUrl, api_key: "", retries: 0 });
    await expect(missing.contexts.list()).rejects.toBeInstanceOf(AuthenticationError);
    // Probes stay token-free.
    await missing.live();
    await missing.health();
  });

  it("scopes a read key out of writes", async () => {
    const name = fresh();
    await seed(name);
    const ctx = reader.context(name);
    expect((await ctx.recall("青嶺酒造")).total).toBeGreaterThan(0);
    await expect(
      ctx.addAssociations([{ subject: "s", label: "l", object: "o", weight: 1.0 }]),
    ).rejects.toBeInstanceOf(PermissionDeniedError);
    await expect(reader.contexts.delete(name)).rejects.toBeInstanceOf(PermissionDeniedError);
    await expect(reader.flush()).rejects.toBeInstanceOf(PermissionDeniedError);
    await client.contexts.delete(name);
  });

  it("rate limits with retry-after and caps bodies with 413", async () => {
    const limited = await spawnServer(binary, {
      TAGURU_API_TOKEN: "rl-token",
      TAGURU_RATE_LIMIT_PER_MIN: "3",
      TAGURU_MAX_BODY_BYTES: "1024",
    });
    try {
      const c = new Taguru({ base_url: limited.baseUrl, api_key: "rl-token", retries: 0 });
      await c.waitUntilReady({ timeout: 30 });
      await c.contexts.create("cap");

      const big = Array.from({ length: 200 }, (_, i) => ({
        subject: `s${i}`,
        label: "l",
        object: "o".repeat(40),
        weight: 1.0,
      }));
      const capped = await c
        .context("cap")
        .addAssociations(big)
        .catch((caught: unknown) => caught);
      expect(capped).toBeInstanceOf(PayloadTooLargeError);
      // The cap breach speaks the JSON error shape, code included.
      expect((capped as PayloadTooLargeError).code).toBe("payload_too_large");

      const rateLimited = await (async () => {
        for (let i = 0; i < 10; i += 1) {
          await c.contexts.list();
        }
        return null;
      })().catch((caught: unknown) => caught);
      expect(rateLimited).toBeInstanceOf(RateLimitError);
      expect((rateLimited as RateLimitError).retry_after).toBeGreaterThan(0);
      await c.health(); // probes stay exempt while throttled
    } finally {
      limited.stop();
    }
  });
});

describe("retrieve end to end", () => {
  it("answers the qa_recall-style question with citations", async () => {
    const name = fresh();
    await seed(name);
    const result = await client
      .context(name)
      .retrieve("青嶺酒造", { text_fallback_query: "杜氏は高瀬である" });

    expect(result.resolved["青嶺酒造"]![0]!.kind).toBe("exact");
    expect(result.outline["青嶺酒造"]).not.toBeNull();
    const toji = result.associations.filter((a) => a.label === "杜氏");
    expect(toji[0]?.object).toBe("高瀬");
    expect(result.citations.size).toBeGreaterThan(0);
    const key = citationKey("docs/aomine.md", 1);
    if (result.citations.has(key)) {
      expect(result.citations.get(key)!.text).toContain("高瀬");
    }
    expect(result.passage_hits).toEqual([]);
    await client.contexts.delete(name);
  });
});

describe("groups and cross-context search", () => {
  /** Two contexts holding one distinct fact (graph + passage) each. */
  async function seededPair(base: string): Promise<[string, string]> {
    const sake = `${base}-sake`;
    const tea = `${base}-tea`;
    await client.contexts.create(sake, { description: "酒蔵の知識" });
    await client.contexts.create(tea, { description: "茶園の知識" });
    await client.context(sake).addAssociations([
      { subject: "青嶺酒造", label: "代表銘柄", object: "青嶺", weight: 1.0, source: "sake.md", paragraph: 0 },
    ]);
    await client.context(tea).addAssociations([
      { subject: "青嶺茶園", label: "代表銘柄", object: "露霜", weight: 1.0, source: "tea.md", paragraph: 0 },
    ]);
    await client.context(sake).storePassages({ "sake.md": "青嶺酒造の代表銘柄は「青嶺」である。" });
    await client.context(tea).storePassages({ "tea.md": "青嶺茶園の代表銘柄は「露霜」である。" });
    return [sake, tea];
  }

  it("creates, nests, updates by deltas, deletes the bundling only", async () => {
    const base = fresh();
    const [sake, tea] = await seededPair(base);
    const group = `${base}-g`;
    const child = `${base}-child`;

    expect(await client.groups.exists(group)).toBe(false);
    expect(await client.groups.create(group, { description: "蔵元一式", contexts: [sake] })).toBe(true);
    await expect(client.groups.create(group)).rejects.toMatchObject({ code: "already_exists" });

    let entry = await client.groups.get(group);
    expect(entry.description).toBe("蔵元一式");
    expect(entry.contexts).toEqual([sake]);
    expect(entry.groups).toEqual([]);

    // Deltas: removals are idempotent no-ops, additions demand existence.
    entry = await client.groups.update(group, {
      add_contexts: [tea],
      remove_contexts: ["never-a-member"],
    });
    expect(entry.contexts).toEqual([sake, tea].sort());
    await expect(
      client.groups.update(group, { add_contexts: [`${base}-missing`] }),
    ).rejects.toMatchObject({ code: "no_context" });

    // Nesting: a child group rides the row's `groups` list.
    expect(await client.groups.create(child, { contexts: [tea] })).toBe(true);
    entry = await client.groups.update(group, { add_groups: [child] });
    expect(entry.groups).toEqual([child]);

    const names: string[] = [];
    for await (const row of client.groups.iter({ limit: 2 })) {
      names.push(row.name);
    }
    expect(names).toContain(group);
    expect(names).toContain(child);

    // Rename: the old name is gone, the new one keeps the membership.
    const renamedGroup = `${base}-g-renamed`;
    expect(await client.groups.rename(group, renamedGroup)).toBe(true);
    await expect(client.groups.get(group)).rejects.toMatchObject({ code: "no_group" });
    entry = await client.groups.get(renamedGroup);
    expect(entry.contexts).toEqual([sake, tea].sort());
    expect(entry.groups).toEqual([child]);

    // Deleting the bundling leaves members (and the child group) alone.
    expect(await client.groups.delete(renamedGroup)).toBe(true);
    await expect(client.groups.get(renamedGroup)).rejects.toMatchObject({ code: "no_group" });
    expect(await client.groups.exists(child)).toBe(true);
    expect(await client.contexts.exists(sake)).toBe(true);

    await expect(reader.groups.create(`${base}-denied`)).rejects.toBeInstanceOf(
      PermissionDeniedError,
    );

    await client.groups.delete(child);
    await client.contexts.delete(sake);
    await client.contexts.delete(tea);
  });

  it("tags every cross-context match and resolves groups", async () => {
    const base = fresh();
    const [sake, tea] = await seededPair(base);
    const group = `${base}-g`;
    await client.groups.create(group, { contexts: [sake, tea] });

    // recall: named contexts, every match tagged with its origin.
    const page = await client.recall("代表銘柄", { contexts: [sake, tea] });
    expect(page.total).toBe(2);
    expect(new Set(page.matches.map((m) => m.context))).toEqual(new Set([sake, tea]));
    expect(new Set(page.matches.map((m) => m.object))).toEqual(new Set(["青嶺", "露霜"]));

    // query: a group resolves to every context it reaches; overlaps with
    // directly named contexts dedupe silently.
    const queried = await client.query({ label: "代表銘柄", groups: [group], contexts: [sake] });
    expect(queried.total).toBe(2);
    expect(new Set(queried.matches.map((m) => m.context))).toEqual(new Set([sake, tea]));

    // searchPassages: rank-interleaved, hits tagged; score is per-context.
    const hits = await client.searchPassages("代表銘柄は青嶺", {
      contexts: [sake, tea],
      limit: 4,
    });
    expect(new Set(hits.map((h) => h.context))).toEqual(new Set([sake, tea]));
    expect(hits.every((h) => h.text.length > 0)).toBe(true);

    // An empty target list is refused, an unknown group answers no_group.
    await expect(client.recall("青嶺", { contexts: [] })).rejects.toMatchObject({
      code: "invalid_argument",
    });
    await expect(client.recall("青嶺", { groups: [`${base}-missing`] })).rejects.toMatchObject({
      code: "no_group",
    });

    await client.groups.delete(group);
    await client.contexts.delete(sake);
    await client.contexts.delete(tea);
  });

  it("round-trips a group through export → import", async () => {
    const base = fresh();
    const [sake, tea] = await seededPair(base);
    const group = `${base}-g`;
    await client.groups.create(group, { description: "蔵元一式", contexts: [sake, tea] });

    const line = await client.groups.export(group);
    const record = JSON.parse(line) as { taguru_group: number; name: string; contexts: string[] };
    expect(typeof record.taguru_group).toBe("number");
    expect(record.name).toBe(group);
    expect(record.contexts).toEqual([sake, tea].sort());

    // The record is the group's complete truth: import restores it whole.
    await client.groups.delete(group);
    expect(await client.groups.exists(group)).toBe(false);
    const result = await client.importBatches(line);
    expect(result.groups).toEqual([{ name: group, outcome: "created", contexts: 2, groups: 0 }]);
    expect((await client.groups.get(group)).contexts).toEqual([sake, tea].sort());

    await client.groups.delete(group);
    await client.contexts.delete(sake);
    await client.contexts.delete(tea);
  });
});

describe("retract association", () => {
  it("withdraws one edge outright and answers found-nothing honestly", async () => {
    const name = fresh();
    await seed(name);
    const ctx = client.context(name);

    const outcome = await ctx.retractAssociation("青嶺酒造", "代表銘柄", "青嶺");
    expect(outcome.retracted).toBe(true);
    expect(outcome.attributions_removed).toBe(1);

    const again = await ctx.retractAssociation("青嶺酒造", "代表銘柄", "青嶺");
    expect(again.retracted).toBe(false);
    expect(again.attributions_removed).toBe(0);

    const toji = await ctx.query({ subject: "青嶺酒造", label: "杜氏" });
    expect(toji.matches[0]!.object).toBe("高瀬");
    const dead = await ctx.query({ subject: "青嶺酒造", label: "代表銘柄" });
    expect(dead.matches[0]!.weight).toBe(0.0);
    expect(dead.matches[0]!.count).toBe(0);
    await client.contexts.delete(name);
  });
});
