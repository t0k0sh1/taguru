/** Retriever + Ingester against the real server binary — mirrors the Python suite. */

import { randomUUID } from "node:crypto";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";

import { TaguruIngester } from "../../src/ingest.js";
import { TaguruRetriever } from "../../src/retrievers.js";

const TOKEN = "lc-ts-test-token";

const AOMINE_DOC = `青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。`;

let server: SpawnedServer;
let client: Taguru;
const seededContext = `sake-${randomUUID().slice(0, 8)}`;

beforeAll(async () => {
  server = await spawnServer(serverBinary(), { TAGURU_API_TOKEN: TOKEN });
  client = new Taguru({ base_url: server.baseUrl, api_key: TOKEN });
  await client.waitUntilReady({ timeout: 30 });

  await client.contexts.create(seededContext, { description: "青嶺酒造という架空の酒蔵の知識" });
  const ctx = client.context(seededContext);
  await ctx.addAssociations([
    { subject: "青嶺酒造", label: "創業年", object: "1907年", weight: 1.0, source: "docs/aomine.md", paragraph: 0 },
    { subject: "青嶺酒造", label: "代表銘柄", object: "青嶺", weight: 1.0, source: "docs/aomine.md", paragraph: 0 },
    { subject: "青嶺酒造", label: "杜氏", object: "高瀬", weight: 1.0, source: "docs/aomine.md", paragraph: 1 },
    { subject: "高瀬", label: "重視する", object: "寒仕込み", weight: 1.0, source: "docs/aomine.md", paragraph: 1 },
  ]);
  await ctx.storePassages({ "docs/aomine.md": AOMINE_DOC });
});

afterAll(() => {
  server.stop();
});

describe("TaguruRetriever (real server)", () => {
  it("serves both lanes from the seeded context", async () => {
    const retriever = new TaguruRetriever({ context: seededContext, client, k: 8 });
    const documents = await retriever.invoke("青嶺酒造");

    expect(documents.length).toBeGreaterThan(0);
    const located = documents.filter((d) => d.metadata["paragraph"] === 1);
    expect(located.length).toBeGreaterThan(0);
    expect(located[0]!.pageContent).toContain("高瀬");
    const labels = new Set(
      documents.flatMap((d) =>
        ((d.metadata["associations"] as Array<{ label: string }> | undefined) ?? []).map(
          (a) => a.label,
        ),
      ),
    );
    expect(labels.has("杜氏")).toBe(true);
  });

  it("catches answer-shaped queries through the text lane", async () => {
    const retriever = new TaguruRetriever({
      context: seededContext,
      client,
      include_graph: false,
      k: 3,
    });
    const documents = await retriever.invoke("1907年に創業した");
    expect(documents.length).toBeGreaterThan(0);
    expect(documents[0]!.metadata["lane"]).toBe("text");
    expect(documents[0]!.pageContent).toContain("1907年");
  });
});

const INGEST_ANSWER = JSON.stringify({
  associations: [
    { subject: "月白堂", label: "kind", object: "和菓子屋", weight: 1.0, paragraph: 0 },
    { subject: "月白堂", label: "名物", object: "栗きんとん", weight: 1.0, paragraph: 1 },
  ],
  aliases: [{ alias: "Geppakudo", canonical: "月白堂", kind: "concept" }],
  questions: [{ paragraph: 1, question: "名物は何ですか?" }],
});

const SHOP_DOC = "月白堂は架空の和菓子屋である。\n\n名物は栗きんとんである。";

describe("TaguruIngester (real server)", () => {
  it("ingests end to end, idempotently", async () => {
    const llm = new FakeListChatModel({
      responses: [INGEST_ANSWER, INGEST_ANSWER, INGEST_ANSWER],
    });
    const ingester = new TaguruIngester({
      context: "wagashi",
      llm,
      client,
      create_context: true,
      context_description: "和菓子屋の知識",
      questions: 2,
    });

    const dry = await ingester.ingestText(SHOP_DOC, {
      source: "docs/geppakudo.md",
      dry_run: true,
    });
    expect(dry.ok).toBe(true);
    expect(await client.contexts.exists("wagashi")).toBe(false);

    const outcomes = await ingester.ingestDocuments([
      { pageContent: SHOP_DOC, metadata: { source: "docs/geppakudo.md" } },
    ]);
    expect(outcomes[0]!.ok).toBe(true);
    expect(outcomes[0]!.created).toBe(true);
    expect(outcomes[0]!.associations).toBe(2);
    expect(outcomes[0]!.aliases).toBe(1);
    expect(outcomes[0]!.questions_stored).toBe(1);
    expect(outcomes[0]!.embeddings_refresh_warning).toBeNull();

    const ctx = client.context("wagashi");
    const match = (await ctx.query({ subject: "月白堂", label: "名物" })).matches[0]!;
    expect(match.object).toBe("栗きんとん");
    expect(match.attributions[0]!.paragraph).toBe(1);
    expect((await ctx.resolve("Geppakudo"))[0]!.name).toBe("月白堂");

    // Re-ingesting the same document is a per-source replace.
    const before = (await ctx.query({ subject: "月白堂", label: "名物" })).matches[0]!;
    const again = await ingester.ingestText(SHOP_DOC, { source: "docs/geppakudo.md" });
    expect(again.ok).toBe(true);
    expect(again.retracted).toBeGreaterThan(0);
    const after = (await ctx.query({ subject: "月白堂", label: "名物" })).matches[0]!;
    expect(after.weight).toBe(before.weight);
    expect(after.count).toBe(before.count);

    // Immediately retrievable through the retriever.
    const retriever = new TaguruRetriever({ context: "wagashi", client, k: 4 });
    const documents = await retriever.invoke("月白堂");
    expect(documents.some((d) => d.pageContent.includes("栗きんとん"))).toBe(true);

    await client.contexts.delete("wagashi");
  });
});
